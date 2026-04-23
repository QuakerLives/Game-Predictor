"""Ensemble evaluation runner.

Loads the test-set probability arrays saved by game_cnn and game_transformer
(both evaluated on the same shared test records), combines them with
EnsembleCombiner, and reports final accuracy.

Usage:
    python -m ensemble.run

Run order:
    1. python -m game_cnn          → trains CNN, saves shared_split.npz
    2. python -m game_transformer  → trains Transformer on same test records
    3. python -m ensemble.run      → combines and evaluates
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .combiner import EnsembleCombiner

_MODELS_DIR = Path("models")


def _load(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = _MODELS_DIR / f"{name}_test.npz"
    if not path.exists():
        print(f"[ensemble] ERROR: {path} not found.")
        print(f"           Run 'python -m game_{name}' first.")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return (
        data["probs"].astype(np.float64),
        data["y_true"].astype(np.int64),
        list(data["label_names"]),
    )


def run() -> None:
    print("=" * 60)
    print("  Ensemble  ·  CNN + Transformer Combiner")
    print("=" * 60)

    cnn_probs,   cnn_y,   cnn_labels   = _load("cnn")
    trans_probs, trans_y, trans_labels = _load("transformer")

    if len(cnn_y) != len(trans_y):
        print(
            f"\n[ensemble] Test set size mismatch: CNN={len(cnn_y)}, "
            f"Transformer={len(trans_y)}\n"
            f"  Make sure to run game_cnn BEFORE game_transformer so the\n"
            f"  Transformer picks up the shared split from models/shared_split.npz"
        )
        sys.exit(1)

    if cnn_labels != trans_labels:
        print("[ensemble] ERROR: label orderings differ between models.")
        sys.exit(1)

    label_names = cnn_labels
    y_true = cnn_y

    print(f"\n  Shared test set: {len(y_true)} records")
    print(f"  Classes: {label_names}\n")

    # ── Per-model baseline ────────────────────────────────────────────────
    EnsembleCombiner.compare(
        probs_list=[cnn_probs, trans_probs],
        y_true=y_true,
        label_names=label_names,
    )

    # ── Best combined result (learned weights) ────────────────────────────
    combiner = EnsembleCombiner(strategy="learned")
    combiner.fit([cnn_probs, trans_probs], y_true)
    result = combiner.evaluate([cnn_probs, trans_probs], y_true, label_names)

    print("── Final ensemble (learned weights) ──")
    print(f"  Accuracy : {result['accuracy']:.4f}")
    print(f"  Log-loss : {result['log_loss']:.4f}")
    print()
    for name, acc in result["per_class"].items():
        print(f"    {name:30s} {acc:.4f}")

    print("\n" + "=" * 60)
    print("  Ensemble complete.")
    print("=" * 60)


if __name__ == "__main__":
    run()
