"""Training pipeline for the text embedding classifier.

Reuses game_nn.GameClassifier (same FC architecture) since the model
receives fixed-dim embedding vectors, not raw text.  The training loop
mirrors game_nn.pipeline for consistency.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from game_nn.pipeline import evaluate, train_single_model

from .data import SplitData, build_dataset


def run(
    db_path: str | Path = "data/gameplay_data.duckdb",
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.3,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> None:
    """Full pipeline: embed text → train classifier → evaluate."""
    print("=" * 60)
    print("  Game-Transformer  ·  Text Embedding Classifier")
    print("=" * 60)

    shared_split_path = Path("models/shared_split.npz")
    test_record_ids = None
    if shared_split_path.exists():
        split_data = np.load(shared_split_path, allow_pickle=True)
        test_record_ids = list(split_data["test_record_ids"].astype(int))
        print(f"[transformer] using shared split ({len(test_record_ids)} test records from CNN)")
    else:
        print("[transformer] no shared split found — using independent split")
        print("             (run game_cnn first to align test sets for ensemble)")

    data, scaler = build_dataset(db_path, test_record_ids=test_record_ids)

    model, info = train_single_model(
        data,
        hidden_dims=hidden_dims,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    result = evaluate(model, data)
    print(f"\n  val_acc={info['best_val_acc']:.4f}  test_acc={result['accuracy']:.4f}")
    for name, acc in result["per_class"].items():
        print(f"    {name:30s} {acc:.4f}")

    # ── Save model and test outputs for ensemble ──────────────────────────
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    torch.save(
        {"state_dict": model.state_dict(), "n_features": data.n_features,
         "n_classes": data.n_classes, "label_names": data.label_names},
        save_dir / "transformer.pt",
    )
    np.savez(
        save_dir / "transformer_test.npz",
        probs=result["probs"],
        y_true=data.y_test,
        label_names=np.array(data.label_names),
    )
    with open(save_dir / "transformer_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Saved → models/transformer.pt  |  models/transformer_test.npz  |  models/transformer_scaler.pkl")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
