"""Ensemble evaluation runner.

Loads the test-set probability arrays saved by game_cnn, game_transformer,
and (optionally) game_nn --gameplay.  All models must be evaluated on the
same shared test records (pinned by models/shared_split.npz from game_cnn).

Usage:
    python -m ensemble.run

Run order:
    1. python -m game_cnn                → trains CNN, saves shared_split.npz
    2. python -m game_transformer        → trains Transformer on same test records
    3. python -m game_nn --gameplay      → trains NN on same test records (optional)
    4. python -m ensemble.run            → combines and evaluates
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


def _try_load(name: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load an optional model's outputs; return None if the file doesn't exist."""
    path = _MODELS_DIR / f"{name}_test.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return (
        data["probs"].astype(np.float64),
        data["y_true"].astype(np.int64),
        list(data["label_names"]),
    )


def run() -> None:
    print("=" * 60)
    print("  Ensemble  ·  CNN + Transformer [+ NN] Combiner")
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
        print("[ensemble] ERROR: label orderings differ between CNN and Transformer.")
        sys.exit(1)

    probs_list = [cnn_probs, trans_probs]
    model_names = ["CNN", "Transformer"]

    # Optionally include the gameplay NN
    nn_result = _try_load("nn")
    if nn_result is not None:
        nn_probs, nn_y, nn_labels = nn_result
        if len(nn_y) != len(cnn_y):
            print(
                f"[ensemble] WARNING: NN test set size ({len(nn_y)}) != CNN ({len(cnn_y)}) — "
                f"NN excluded. Re-run 'python -m game_nn --gameplay' after game_cnn."
            )
        elif nn_labels != cnn_labels:
            print("[ensemble] WARNING: NN label ordering differs from CNN — NN excluded.")
        else:
            probs_list.append(nn_probs)
            model_names.append("NN (gameplay features)")
            print(f"  NN (gameplay features) included — {len(nn_y)} test records")
    else:
        print("  [optional] NN not found — run 'python -m game_nn --gameplay' to add it.")


    label_names = cnn_labels
    y_true = cnn_y

    print(f"\n  Shared test set: {len(y_true)} records")
    print(f"  Models: {model_names}")
    print(f"  Classes: {label_names}\n")

    # Per-model baselines + all three combination strategies
    n = len(probs_list)
    print(f"{'Model/Strategy':<32s} {'Accuracy':>10s} {'Log-Loss':>10s}")
    print("-" * 56)

    from sklearn.metrics import accuracy_score, log_loss
    for i, (probs, name) in enumerate(zip(probs_list, model_names)):
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_true, preds)
        safe = np.clip(probs, 1e-12, 1.0)
        safe /= safe.sum(axis=1, keepdims=True)
        ll = log_loss(y_true, safe)
        print(f"  {name:<30s} {acc:>10.4f} {ll:>10.4f}")

    print()
    for strategy in ("average", "weighted", "learned"):
        c = EnsembleCombiner(strategy=strategy)
        if strategy == "weighted":
            c.weights = np.ones(n) / n
        if strategy == "learned":
            c.fit(probs_list, y_true)
        result = c.evaluate(probs_list, y_true, label_names)
        print(
            f"  {strategy.capitalize():<30s} "
            f"{result['accuracy']:>10.4f} {result['log_loss']:>10.4f}"
        )

    print()

    # Best combined result (learned weights)
    combiner = EnsembleCombiner(strategy="learned")
    combiner.fit(probs_list, y_true)
    result = combiner.evaluate(probs_list, y_true, label_names)

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
