"""Fully-connected neural network for multi-class game classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class GameClassifier(nn.Module):
    """FC network with BatchNorm, ReLU, Dropout, and softmax output layer.

    Architecture is configurable via *hidden_dims*.  The final linear layer
    produces raw logits; ``torch.nn.CrossEntropyLoss`` applies log-softmax
    internally, while :meth:`predict_proba` exposes explicit softmax
    probabilities for inference / ensembling.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, num_classes)."""
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)
