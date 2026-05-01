"""EfficientNet-B0 CNN for multi-class game image classification.

We keep the pretrained ImageNet backbone and only replace the final classifier
head. Retraining from random weights on ~200 images per class would likely
overfit badly — starting from ImageNet features avoids that.

``predict_proba`` mirrors the interface of ``game_nn.model.GameClassifier``
so both models can be plugged into the same ensemble combiner without any
extra adapter code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class GameCNN(nn.Module):
    """EfficientNet-B0 backbone with a task-specific classifier head.

    Args:
        num_classes: Number of game classes to predict.
        dropout: Dropout rate applied inside the classifier head.
        freeze_backbone: If True, only the classifier head is trainable.
            Useful for a quick warm-up pass before fine-tuning the full net.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features: int = backbone.classifier[1].in_features  # 1280 for EfficientNet-B0
        # Replace the original 1000-class head with one sized for our 5-game problem
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.net = backbone

        if freeze_backbone:
            # Freeze only the feature extractor layers, not the new head
            for param in self.net.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, num_classes)."""
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities — mirrors game_nn.GameClassifier API."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)

    def unfreeze_backbone(self) -> None:
        """Make all backbone parameters trainable (call after head warm-up)."""
        for param in self.net.features.parameters():
            param.requires_grad = True
