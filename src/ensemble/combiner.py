"""Ensemble combiner: merges softmax probability matrices from all sub-models.

Four combination strategies:
  - "average"       — equal-weight average of all prob matrices (default)
  - "weighted"      — manually supplied per-model weights (must sum to 1)
  - "learned"       — fits a non-negative weight vector via scipy.optimize on a
                      held-out validation set; minimises cross-entropy
  - "proportional"  — weights each model by its individual validation accuracy
                      (weight_i = accuracy_i / sum(accuracies)); simple and
                      interpretable, prevents any single model from dominating

All sub-models expose ``predict_proba(input) -> (N, C) numpy array``
before being passed here — the combiner is agnostic to model type.

Usage
-----
    combiner = EnsembleCombiner(strategy="average")
    # probs_list: one (N, C) array per sub-model, all on the same N samples
    final_probs = combiner.combine([nn_probs, cnn_probs, transformer_probs])
    preds = final_probs.argmax(axis=1)

    # Fit optimal weights on the validation set
    combiner_opt = EnsembleCombiner(strategy="learned")
    combiner_opt.fit(val_probs_list, y_val)
    test_probs = combiner_opt.combine(test_probs_list)

    # Accuracy-proportional weights
    combiner_prop = EnsembleCombiner(strategy="proportional")
    combiner_prop.fit(val_probs_list, y_val)
    test_probs = combiner_prop.combine(test_probs_list)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize


def _project_floor(w: np.ndarray, floor: float) -> np.ndarray:
    """Project w onto the probability simplex with per-element lower bound `floor`.

    Works iteratively: clamp any weight below the floor up to floor, then
    redistribute the excess proportionally across the unclamped weights.
    Repeat until nothing changes. This avoids the optimizer from silently
    driving one model's weight to zero, which would make the "ensemble" just
    a single model under the hood.
    """
    w = w.astype(np.float64).copy()
    for _ in range(1000):
        below = w < floor
        if not below.any():
            break
        w[below] = floor
        excess = w.sum() - 1.0
        above  = ~below
        if above.any():
            # Take the excess back proportionally from the unclamped weights
            w[above] -= excess * (w[above] / w[above].sum())
        else:
            # All weights are at the floor — can't redistribute, just normalize
            w = np.ones(len(w)) / len(w)
            break
    return w / w.sum()


class EnsembleCombiner:
    """Combine probability outputs from multiple classifiers.

    Args:
        strategy: One of ``"average"``, ``"weighted"``, ``"learned"``, or ``"proportional"``.
        weights: Per-model weights for the ``"weighted"`` strategy.
            Must be a sequence of non-negative floats that sum to 1 and
            match the number of models passed to ``combine``.
    """

    def __init__(
        self,
        strategy: str = "average",
        weights: list[float] | None = None,
    ) -> None:
        if strategy not in ("average", "weighted", "learned", "proportional"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.weights  = np.array(weights, dtype=np.float64) if weights else None
        self._fitted_weights: np.ndarray | None = None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        probs_list: list[np.ndarray],
        y_true: np.ndarray,
        min_weight: float = 0.15,
    ) -> "EnsembleCombiner":
        """Fit weights by minimising cross-entropy on ``y_true``.

        Each model is guaranteed at least ``min_weight`` so the ensemble
        remains genuinely multi-modal rather than collapsing to a single model.

        Only meaningful when ``strategy`` is "learned" or "proportional".
        Calling fit with any other strategy is a no-op.

        Args:
            probs_list: List of (N, C) probability arrays from each model.
            y_true: Ground-truth integer labels of length N.
            min_weight: Floor for each model's weight (default 0.15).
                        Must satisfy ``n_models * min_weight <= 1``.
        """
        if self.strategy not in ("learned", "proportional"):
            return self

        if self.strategy == "proportional":
            # Weight each model by its accuracy — simple and easy to explain
            accs = np.array([
                accuracy_score(y_true, p.argmax(axis=1)) for p in probs_list
            ])
            self._fitted_weights = accs / accs.sum()
            print(
                "[ensemble] proportional weights: "
                + "  ".join(f"model{i+1}={w:.4f}" for i, w in enumerate(self._fitted_weights))
            )
            return self

        # ── Learned strategy ──────────────────────────────────────────────
        n_models = len(probs_list)
        stacked  = np.stack(probs_list, axis=0)  # (M, N, C)

        # Optimize in unconstrained space (any real vector), then project back
        # onto the simplex. This sidesteps numerical issues that arise when
        # scipy tries to enforce constraints while also approaching a boundary.
        def _neg_log_likelihood(w: np.ndarray) -> float:
            # Softmax maps the unconstrained vector to weights that are positive and sum to 1
            w_norm    = np.exp(w) / np.exp(w).sum()
            # Clip and renormalize to enforce the per-model minimum
            w_floored = np.clip(w_norm, min_weight, 1.0)
            w_floored /= w_floored.sum()
            avg = (stacked * w_floored[:, None, None]).sum(0)  # (N, C)
            avg = np.clip(avg, 1e-12, 1.0)
            avg /= avg.sum(axis=1, keepdims=True)
            return log_loss(y_true, avg)

        result = minimize(
            _neg_log_likelihood,
            x0=np.zeros(n_models),
            method="L-BFGS-B",
        )
        # Final projection ensures the stored weights satisfy the floor constraint exactly
        raw_norm = np.exp(result.x) / np.exp(result.x).sum()
        self._fitted_weights = _project_floor(raw_norm, min_weight)
        print(
            "[ensemble] learned weights: "
            + "  ".join(f"model{i+1}={w:.4f}" for i, w in enumerate(self._fitted_weights))
        )
        return self

    # ── Combination ───────────────────────────────────────────────────────────

    def combine(self, probs_list: list[np.ndarray]) -> np.ndarray:
        """Return the combined (N, C) probability matrix.

        Args:
            probs_list: One (N, C) float array per sub-model.
        """
        stacked = np.stack(probs_list, axis=0)  # (M, N, C)

        if self.strategy == "average":
            # Equal weights — simplest possible combination
            return stacked.mean(axis=0)

        if self.strategy == "weighted":
            if self.weights is None:
                raise ValueError("Provide weights when using strategy='weighted'")
            # Normalize in case the user didn't supply weights that sum to exactly 1
            w = self.weights / self.weights.sum()
            return (stacked * w[:, None, None]).sum(axis=0)

        # strategy == "learned" or "proportional" — both use _fitted_weights
        if self._fitted_weights is None:
            raise RuntimeError(f"Call fit() before combine() with strategy={self.strategy!r}")
        w = self._fitted_weights
        return (stacked * w[:, None, None]).sum(axis=0)

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def evaluate(
        self,
        probs_list: list[np.ndarray],
        y_true: np.ndarray,
        label_names: list[str] | None = None,
    ) -> dict:
        """Combine probs and return accuracy, log-loss, and per-class accuracy."""
        combined = self.combine(probs_list)
        preds    = combined.argmax(axis=1)

        acc  = accuracy_score(y_true, preds)
        # Clip to avoid log(0) in log_loss, then renormalize to keep it a valid distribution
        safe = np.clip(combined, 1e-12, 1.0)
        safe /= safe.sum(axis=1, keepdims=True)
        ll   = log_loss(y_true, safe)

        per_class: dict[str, float] = {}
        for cls in np.unique(y_true):
            mask = y_true == cls
            name = label_names[cls] if label_names else str(cls)
            per_class[name] = accuracy_score(y_true[mask], preds[mask])

        return {"accuracy": acc, "log_loss": ll, "per_class": per_class}

    @staticmethod
    def compare(
        probs_list: list[np.ndarray],
        y_true: np.ndarray,
        label_names: list[str] | None = None,
    ) -> None:
        """Print a comparison table: individual models vs. all four strategies."""
        n_models = len(probs_list)
        print(f"\n{'Model/Strategy':<28s} {'Accuracy':>10s} {'Log-Loss':>10s}")
        print("-" * 52)

        # Individual model baselines first
        for i, probs in enumerate(probs_list):
            preds = probs.argmax(axis=1)
            acc   = accuracy_score(y_true, preds)
            safe  = np.clip(probs, 1e-12, 1.0)
            safe /= safe.sum(axis=1, keepdims=True)
            ll    = log_loss(y_true, safe)
            print(f"  Model {i + 1:<21d} {acc:>10.4f} {ll:>10.4f}")

        print()
        # Then all four combination strategies
        for strategy in ("average", "weighted", "learned", "proportional"):
            c = EnsembleCombiner(strategy=strategy)
            if strategy == "weighted":
                c.weights = np.ones(n_models) / n_models
            if strategy in ("learned", "proportional"):
                c.fit(probs_list, y_true)
            result = c.evaluate(probs_list, y_true, label_names)
            print(
                f"  {strategy.capitalize():<26s} "
                f"{result['accuracy']:>10.4f} {result['log_loss']:>10.4f}"
            )
        print()
