"""Training pipeline for the CNN with validation-based early stopping.

Two-phase training strategy (matches EfficientNet best practices on small datasets):
  Phase 1 — Head warm-up: backbone frozen, only classifier trained (fast).
  Phase 2 — Full fine-tune: all layers trainable with a lower learning rate.

The public API mirrors game_nn.pipeline so both outputs slot into the same
ensemble combiner without any glue code.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import LoaderBundle, build_loaders
from .model import GameCNN


# ── Training helpers ─────────────────────────────────────────────────────────


def _run_epoch(
    model: GameCNN,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one pass.  If optimizer is None, run in eval mode (no gradients)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)

    return total_loss / n, correct / n


# ── Public API ───────────────────────────────────────────────────────────────


def train_single_model(
    bundle: LoaderBundle,
    *,
    dropout: float = 0.3,
    # Phase 1 — head warm-up (backbone frozen)
    warmup_epochs: int = 5,
    warmup_lr: float = 1e-3,
    # Phase 2 — full fine-tune
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GameCNN, dict]:
    """Train one CNN; returns (best_model, metrics_dict).

    Phase 1 freezes the EfficientNet backbone and trains only the head for
    *warmup_epochs*.  Phase 2 unfreezes everything and fine-tunes with a
    lower learning rate and early stopping on validation accuracy.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  device: {device}")

    model = GameCNN(bundle.n_classes, dropout=dropout, freeze_backbone=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: head warm-up ────────────────────────────────────────────────
    if warmup_epochs > 0:
        head_params = [p for p in model.parameters() if p.requires_grad]
        opt_warmup = torch.optim.Adam(head_params, lr=warmup_lr)
        for epoch in range(1, warmup_epochs + 1):
            tr_loss, _ = _run_epoch(model, bundle.train, criterion, opt_warmup, device)
            va_loss, va_acc = _run_epoch(model, bundle.val, criterion, None, device)
            if verbose:
                print(
                    f"  [warm-up {epoch}/{warmup_epochs}]  "
                    f"tr_loss={tr_loss:.4f}  va_acc={va_acc:.4f}"
                )

    # ── Phase 2: full fine-tune ──────────────────────────────────────────────
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    best_val_acc = 0.0
    best_state: dict | None = None
    wait = 0

    for epoch in range(1, epochs + 1):
        tr_loss, _ = _run_epoch(model, bundle.train, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, bundle.val, criterion, None, device)
        scheduler.step(va_loss)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  epoch {epoch:3d}  tr_loss={tr_loss:.4f}  "
                f"va_loss={va_loss:.4f}  va_acc={va_acc:.4f}"
            )

        if wait >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    return model, {"best_val_acc": best_val_acc}


def evaluate(
    model: GameCNN,
    bundle: LoaderBundle,
    *,
    split: str = "test",
    device: torch.device | None = None,
) -> dict:
    """Evaluate a single model on test or val split.

    Returns accuracy, per-class accuracy, and a numpy array of probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = bundle.test if split == "test" else bundle.val
    model.to(device)
    model.eval()

    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            probs = model.predict_proba(images.to(device))
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    probs_t = torch.cat(all_probs)
    labels_t = torch.cat(all_labels)
    preds = probs_t.argmax(1)
    acc = (preds == labels_t).float().mean().item()

    per_class: dict[str, float] = {}
    for i, name in enumerate(bundle.label_names):
        mask = labels_t == i
        if mask.sum() > 0:
            per_class[name] = (preds[mask] == labels_t[mask]).float().mean().item()

    return {"accuracy": acc, "per_class": per_class, "probs": probs_t.numpy(), "y_true": labels_t.numpy()}


def ensemble_evaluate(
    models: list[GameCNN],
    bundle: LoaderBundle,
    device: torch.device | None = None,
) -> dict:
    """Average softmax probabilities across all models and evaluate on test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_member_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    labels_collected = False

    for model in models:
        model.to(device)
        model.eval()
        member_probs: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in bundle.test:
                probs = model.predict_proba(images.to(device))
                member_probs.append(probs.cpu())
                if not labels_collected:
                    all_labels.append(labels)

        labels_collected = True
        all_member_probs.append(torch.cat(member_probs))

    labels_t = torch.cat(all_labels)
    avg_probs = torch.stack(all_member_probs).mean(dim=0)
    preds = avg_probs.argmax(1)
    acc = (preds == labels_t).float().mean().item()

    per_class: dict[str, float] = {}
    for i, name in enumerate(bundle.label_names):
        mask = labels_t == i
        if mask.sum() > 0:
            per_class[name] = (preds[mask] == labels_t[mask]).float().mean().item()

    return {"accuracy": acc, "per_class": per_class}


def run(
    db_path: str | Path = "data/gameplay_data.duckdb",
    base_dir: str | Path = "data",
    batch_size: int = 32,
    warmup_epochs: int = 5,
    epochs: int = 30,
) -> None:
    """Full pipeline: load data → train CNN → evaluate."""
    print("=" * 60)
    print("  Game-CNN  ·  EfficientNet-B0 Training Pipeline")
    print("=" * 60)

    bundle = build_loaders(
        db_path=db_path,
        base_dir=base_dir,
        batch_size=batch_size,
    )
    print(f"[data] classes: {bundle.label_names}")

    model, info = train_single_model(
        bundle,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
    )
    result = evaluate(model, bundle)
    print(f"\n  val_acc={info['best_val_acc']:.4f}  test_acc={result['accuracy']:.4f}")
    for name, acc in result["per_class"].items():
        print(f"    {name:30s} {acc:.4f}")

    # ── Save model and test outputs for ensemble ──────────────────────────
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    torch.save(
        {"state_dict": model.state_dict(), "n_classes": bundle.n_classes,
         "label_names": bundle.label_names},
        save_dir / "cnn.pt",
    )
    np.savez(
        save_dir / "cnn_test.npz",
        probs=result["probs"],
        y_true=result["y_true"],
        label_names=np.array(bundle.label_names),
    )
    np.savez(
        save_dir / "shared_split.npz",
        test_record_ids=np.array(bundle.test_record_ids),
        label_names=np.array(bundle.label_names),
    )
    print(f"\n  Saved → models/cnn.pt  |  models/cnn_test.npz  |  models/shared_split.npz")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
