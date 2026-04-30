"""Training pipeline with validation-based early stopping and optional ensembling."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import SplitData, build_dataset, build_dataset_from_gameplay
from .model import GameClassifier


def _to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)


def train_single_model(
    data: SplitData,
    *,
    hidden_dims: tuple[int, ...] = (128, 64, 32),
    dropout: float = 0.3,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 40,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GameClassifier, dict]:
    """Train one model; returns (best_model, metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = _to_tensor(data.X_train, torch.float32)
    y_tr = _to_tensor(data.y_train, torch.long)
    X_va = _to_tensor(data.X_val, torch.float32)
    y_va = _to_tensor(data.y_val, torch.long)

    model = GameClassifier(data.n_features, data.n_classes, hidden_dims, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5,
    )

    best_val_acc = 0.0
    best_state: dict | None = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        for start in range(0, len(X_tr), batch_size):
            idx = perm[start : start + batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_loss = criterion(val_logits, y_va).item()
            val_acc = (val_logits.argmax(1) == y_va).float().mean().item()
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if verbose and epoch % 50 == 0:
            print(
                f"  epoch {epoch:3d}  loss={epoch_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
        if wait >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    return model, {"best_val_acc": best_val_acc}


def evaluate(
    model: GameClassifier,
    data: SplitData,
    *,
    tag: str = "test",
) -> dict:
    """Evaluate a single model on the test (or val) split."""
    X = _to_tensor(data.X_test if tag == "test" else data.X_val, torch.float32)
    y = _to_tensor(data.y_test if tag == "test" else data.y_val, torch.long)

    model.eval()
    probs = model.predict_proba(X)
    preds = probs.argmax(1)
    acc = (preds == y).float().mean().item()

    per_class: dict[str, float] = {}
    for i, name in enumerate(data.label_names):
        mask = y == i
        if mask.sum() > 0:
            per_class[name] = (preds[mask] == y[mask]).float().mean().item()

    return {"accuracy": acc, "per_class": per_class, "probs": probs.numpy()}


def ensemble_evaluate(
    models: list[GameClassifier],
    data: SplitData,
) -> dict:
    """Average softmax probabilities across an ensemble and evaluate."""
    X = _to_tensor(data.X_test, torch.float32)
    y = _to_tensor(data.y_test, torch.long)

    all_probs = torch.stack([m.predict_proba(X) for m in models])
    avg_probs = all_probs.mean(dim=0)
    preds = avg_probs.argmax(1)
    acc = (preds == y).float().mean().item()

    per_class: dict[str, float] = {}
    for i, name in enumerate(data.label_names):
        mask = y == i
        if mask.sum() > 0:
            per_class[name] = (preds[mask] == y[mask]).float().mean().item()

    return {"accuracy": acc, "per_class": per_class}


def run_on_gameplay_data(
    gameplay_db_path: str | Path = "data/gameplay_data.duckdb",
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 300,
    batch_size: int = 64,
) -> None:
    """Train NN on gameplay numerical features and save outputs for ensemble.

    Loads models/shared_split.npz (written by game_cnn) so the test set aligns
    with the CNN and Transformer — required for the ensemble to combine them.

    Run order:
        1. python -m game_cnn          → trains CNN, saves shared_split.npz
        2. python -m game_transformer  → trains Transformer on same test records
        3. python -m game_nn --gameplay → trains NN on same test records
        4. python -m ensemble.run      → combines all three
    """
    print("=" * 60)
    print("  Game-NN (gameplay)  ·  Numerical Feature Classifier")
    print("=" * 60)

    shared_split_path = Path("models/shared_split.npz")
    test_record_ids = None
    if shared_split_path.exists():
        split_data = np.load(shared_split_path, allow_pickle=True)
        test_record_ids = list(split_data["test_record_ids"].astype(int))
        print(f"[nn-gameplay] using shared split ({len(test_record_ids)} test records from CNN)")
    else:
        print("[nn-gameplay] no shared split found — using independent split")
        print("              (run game_cnn first to align test sets for ensemble)")

    data = build_dataset_from_gameplay(gameplay_db_path, test_record_ids=test_record_ids)

    model, info = train_single_model(
        data,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    result = evaluate(model, data)
    print(f"\n  val_acc={info['best_val_acc']:.4f}  test_acc={result['accuracy']:.4f}")
    for name, acc in result["per_class"].items():
        print(f"    {name:30s} {acc:.4f}")

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    torch.save(
        {"state_dict": model.state_dict(), "n_features": data.n_features,
         "n_classes": data.n_classes, "label_names": data.label_names},
        save_dir / "nn.pt",
    )
    np.savez(
        save_dir / "nn_test.npz",
        probs=result["probs"],
        y_true=data.y_test,
        label_names=np.array(data.label_names),
    )
    print(f"\n  Saved → models/nn.pt  |  models/nn_test.npz")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Run 'python -m ensemble.run' to combine all models.")
    print("=" * 60)


def run(
    db_path: str | Path = "data/steam_data.duckdb",
    n_ensemble: int = 3,
    epochs: int = 300,
    lr: float = 1e-2,
    batch_size: int = 32,
) -> None:
    """Full pipeline: build dataset → train ensemble → evaluate."""
    print("=" * 60)
    print("  Game-NN  ·  FC Neural Network Training Pipeline")
    print("=" * 60)

    data = build_dataset(db_path)

    models: list[GameClassifier] = []
    for i in range(n_ensemble):
        print(f"\n── Ensemble member {i + 1}/{n_ensemble} ──")
        model, info = train_single_model(
            data, seed=42 + i, epochs=epochs, lr=lr, batch_size=batch_size,
        )
        single_result = evaluate(model, data)
        print(f"  val_acc={info['best_val_acc']:.4f}  test_acc={single_result['accuracy']:.4f}")
        models.append(model)

    print("\n── Ensemble evaluation (averaged softmax) ──")
    ens = ensemble_evaluate(models, data)
    print(f"  Ensemble test accuracy: {ens['accuracy']:.4f}")
    for name, acc in ens["per_class"].items():
        print(f"    {name:30s} {acc:.4f}")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
