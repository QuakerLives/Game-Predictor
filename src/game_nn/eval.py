"""Comprehensive evaluation report for the Game-NN pipeline.

Retraces each preprocessing step to capture intermediate artifacts,
trains an ensemble, and prints detailed statistics to the console.

Usage:
    python -m game_nn.eval [--db steam_data.duckdb]
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    top_k_accuracy_score,
)

from .data import (
    apply_kmeans,
    apply_pca,
    extract_features,
    impute,
    resample_training,
    standardize,
    stratified_split,
    SplitData,
)
from .model import GameClassifier
from .pipeline import _to_tensor, train_single_model

# ── Box-drawing helpers ──────────────────────────────────────────────

W = 72  # report width (inner)


def _hdr(title: str) -> str:
    pad = W - len(title) - 2
    return f"┌─ {title} " + "─" * pad + "┐"


def _row(text: str = "") -> str:
    inner = f"  {text}"
    return f"│{inner:<{W}}│"


def _sep() -> str:
    return "├" + "─" * W + "┤"


def _bot() -> str:
    return "└" + "─" * W + "┘"


def _banner(title: str) -> str:
    top = "╔" + "═" * W + "╗"
    mid = f"║{title:^{W}}║"
    bot = "╚" + "═" * W + "╝"
    return f"{top}\n{mid}\n{bot}"


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:5.1f}%" if total else "  N/A"


def _bar(fraction: float, width: int = 30) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


# ── Core evaluation ──────────────────────────────────────────────────


def _class_dist_table(
    y: np.ndarray, labels: list[str], tag: str,
) -> list[str]:
    """Return formatted lines showing class distribution."""
    total = len(y)
    lines = [f"{tag} ({total} samples):"]
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        name = labels[cls]
        lines.append(
            f"  {name:<28s} {cnt:>5d}  ({_pct(cnt, total)})  "
            f"{_bar(cnt / total, 20)}"
        )
    return lines


def _model_summary(model: GameClassifier) -> list[str]:
    """Return lines describing model architecture and param counts."""
    lines: list[str] = []
    total_params = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable += n
    lines.append(f"Total parameters .......... {total_params:,}")
    lines.append(f"Trainable parameters ...... {trainable:,}")
    lines.append("")
    lines.append("Layer stack:")
    for i, mod in enumerate(model.net):
        s = str(mod).replace("\n", " ")
        if len(s) > W - 8:
            s = s[: W - 11] + "..."
        lines.append(f"  [{i:>2d}] {s}")
    return lines


def _confusion_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> list[str]:
    """Return formatted confusion matrix lines."""
    cm = confusion_matrix(y_true, y_pred)
    n = len(labels)

    abbrevs = [lb[:6] for lb in labels]
    col_w = max(7, max(len(a) for a in abbrevs) + 1)
    header = " " * 10 + "".join(f"{a:>{col_w}}" for a in abbrevs)
    lines = [header]
    lines.append(" " * 10 + "─" * (col_w * n))
    for i in range(n):
        row_vals = "".join(f"{cm[i, j]:>{col_w}d}" for j in range(n))
        lines.append(f"  {abbrevs[i]:<8s}{row_vals}")
    return lines


def run_eval(db_path: str | Path = "data/steam_data.duckdb") -> None:
    t0 = time.perf_counter()

    # ── 0. Banner ────────────────────────────────────────────────────
    print()
    print(_banner("Game-NN  ·  Full Evaluation Report"))
    print()

    # ── 1. Raw data extraction ───────────────────────────────────────
    X_raw, y_raw, label_names, feature_names = extract_features(db_path)
    n_samples, n_feat = X_raw.shape
    n_classes = len(label_names)
    nan_total = int(np.isnan(X_raw).sum())
    nan_per_feat = np.isnan(X_raw).sum(axis=0).astype(int)

    print(_hdr("1. Dataset Overview"))
    print(_row(f"Source .................. {db_path}"))
    print(_row(f"Total samples ........... {n_samples}"))
    print(_row(f"Raw features ............ {n_feat}  ({', '.join(feature_names)})"))
    print(_row(f"Classes ................. {n_classes}"))
    print(_row(f"Missing values .......... {nan_total}  "
               f"({_pct(nan_total, n_samples * n_feat)} of cells)"))
    for i, fn in enumerate(feature_names):
        print(_row(f"  {fn:<24s} {nan_per_feat[i]:>4d} NaN"))
    print(_row())
    for line in _class_dist_table(y_raw, label_names, "Full dataset"):
        print(_row(line))
    print(_bot())
    print()

    # ── 2. Train / Val / Test split ──────────────────────────────────
    X_tr, y_tr, X_va, y_va, X_te, y_te = stratified_split(X_raw, y_raw)

    print(_hdr("2. Train / Val / Test Split"))
    print(_row(f"Strategy ................ Stratified (70 / 15 / 15)"))
    print(_row())
    for tag, y_split in [("Train", y_tr), ("Val  ", y_va), ("Test ", y_te)]:
        print(_row(f"  {tag}:  {len(y_split):>5d} samples"))
    print(_row())
    for lines in [
        _class_dist_table(y_tr, label_names, "Train"),
        _class_dist_table(y_va, label_names, "Val"),
        _class_dist_table(y_te, label_names, "Test"),
    ]:
        for l in lines:
            print(_row(l))
        print(_row())
    print(_bot())
    print()

    # ── 3. Imputation ────────────────────────────────────────────────
    X_tr, X_va, X_te, imp = impute(X_tr, X_va, X_te)

    print(_hdr("3. Imputation"))
    print(_row(f"Strategy ................ Median (fit on train only)"))
    print(_row(f"Median values used:"))
    for i, fn in enumerate(feature_names):
        print(_row(f"  {fn:<24s} {imp.statistics_[i]:>18.4f}"))
    print(_row(f"Remaining NaNs after ..... {int(np.isnan(X_tr).sum() + np.isnan(X_va).sum() + np.isnan(X_te).sum())}"))
    print(_bot())
    print()

    # ── 4. Standardization ───────────────────────────────────────────
    X_tr, X_va, X_te, scaler = standardize(X_tr, X_va, X_te)

    print(_hdr("4. Standardization"))
    print(_row(f"Method .................. Z-score (fit on train only)"))
    print(_row())
    print(_row(f"  {'Feature':<24s} {'Mean':>12s} {'Std':>12s}"))
    print(_row(f"  {'─' * 24} {'─' * 12} {'─' * 12}"))
    for i, fn in enumerate(feature_names):
        print(_row(f"  {fn:<24s} {scaler.mean_[i]:>12.4f} {scaler.scale_[i]:>12.4f}"))
    print(_bot())
    print()

    # ── 5. PCA ───────────────────────────────────────────────────────
    n_comp = min(X_tr.shape[1], X_tr.shape[0])
    X_tr, X_va, X_te, pca = apply_pca(X_tr, X_va, X_te, n_components=n_comp)

    print(_hdr("5. PCA"))
    print(_row(f"Components retained ..... {pca.n_components_}"))
    print(_row(f"Total explained var ..... {pca.explained_variance_ratio_.sum():.6f}"))
    print(_row())
    print(_row(f"  {'PC':<6s} {'Variance':>10s} {'Ratio':>10s} {'Cumulative':>12s}  Bar"))
    print(_row(f"  {'─' * 6} {'─' * 10} {'─' * 10} {'─' * 12}  {'─' * 20}"))
    cum = 0.0
    for i, (var, ratio) in enumerate(
        zip(pca.explained_variance_, pca.explained_variance_ratio_)
    ):
        cum += ratio
        print(_row(
            f"  PC{i + 1:<3d} {var:>10.4f} {ratio:>10.6f} {cum:>12.6f}  "
            f"{_bar(ratio, 16)}"
        ))
    print(_bot())
    print()

    # ── 6. K-Means Clustering ────────────────────────────────────────
    X_tr, X_va, X_te, km = apply_kmeans(X_tr, X_va, X_te, n_clusters=5)

    print(_hdr("6. K-Means Clustering"))
    print(_row(f"K ....................... 5  (fit on train only)"))
    print(_row(f"Inertia ................. {km.inertia_:.4f}"))
    print(_row(f"Iterations .............. {km.n_iter_}"))
    print(_row())

    train_clusters = X_tr[:, -1].astype(int)
    cl_ids, cl_counts = np.unique(train_clusters, return_counts=True)
    print(_row(f"  {'Cluster':>8s} {'Samples':>10s} {'Share':>8s}  Bar"))
    print(_row(f"  {'─' * 8} {'─' * 10} {'─' * 8}  {'─' * 20}"))
    for cid, cnt in zip(cl_ids, cl_counts):
        frac = cnt / len(train_clusters)
        print(_row(f"  {cid:>8d} {cnt:>10d} {frac:>7.1%}  {_bar(frac, 18)}"))
    print(_bot())
    print()

    # ── 7. Resampling ────────────────────────────────────────────────
    pre_n = len(y_tr)
    X_tr, y_tr = resample_training(X_tr, y_tr)

    print(_hdr("7. Training Set Resampling"))
    print(_row(f"Method .................. Oversample minority (train only)"))
    print(_row(f"Before .................. {pre_n}"))
    print(_row(f"After ................... {len(y_tr)}"))
    print(_row())
    for line in _class_dist_table(y_tr, label_names, "Resampled train"):
        print(_row(line))
    print(_bot())
    print()

    # ── Build SplitData for training ─────────────────────────────────
    data = SplitData(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        label_names=label_names,
        feature_names=feature_names,
    )

    # ── 8. Model architecture ────────────────────────────────────────
    sample_model = GameClassifier(data.n_features, data.n_classes)
    print(_hdr("8. Model Architecture"))
    for line in _model_summary(sample_model):
        print(_row(line))
    print(_bot())
    print()

    # ── 9. Training (ensemble) ───────────────────────────────────────
    N_ENS = 3
    print(_hdr(f"9. Training ({N_ENS}-Model Ensemble)"))
    print(_row(f"Optimizer ............... Adam  (lr=0.01, wd=1e-4)"))
    print(_row(f"Scheduler ............... ReduceLROnPlateau (p=10, f=0.5)"))
    print(_row(f"Early stopping .......... patience=40"))
    print(_row(f"Batch size .............. 32"))
    print(_row())

    models: list[GameClassifier] = []
    member_stats: list[dict] = []
    for i in range(N_ENS):
        t_start = time.perf_counter()
        model, info = train_single_model(data, seed=42 + i, verbose=False)
        elapsed = time.perf_counter() - t_start
        models.append(model)
        member_stats.append({**info, "time": elapsed})
        print(_row(
            f"  Member {i + 1}  |  best_val_acc={info['best_val_acc']:.4f}  |  "
            f"time={elapsed:.2f}s"
        ))

    print(_bot())
    print()

    # ── 10. Single-model test results ────────────────────────────────
    X_te_t = _to_tensor(data.X_test, torch.float32)
    y_te_t = _to_tensor(data.y_test, torch.long)
    y_true = data.y_test

    print(_hdr("10. Individual Model Test Results"))
    print(_row(f"  {'Model':<10s} {'Accuracy':>10s} {'Log-Loss':>10s}  Top-2 Acc"))
    print(_row(f"  {'─' * 10} {'─' * 10} {'─' * 10}  {'─' * 10}"))
    for i, m in enumerate(models):
        probs = m.predict_proba(X_te_t).numpy()
        preds = probs.argmax(1)
        acc = (preds == y_true).mean()
        ll = log_loss(y_true, probs, labels=list(range(data.n_classes)))
        top2 = top_k_accuracy_score(y_true, probs, k=min(2, data.n_classes),
                                     labels=list(range(data.n_classes)))
        print(_row(f"  Model {i + 1:<3d} {acc:>10.4f} {ll:>10.4f}  {top2:>10.4f}"))
    print(_bot())
    print()

    # ── 11. Ensemble evaluation ──────────────────────────────────────
    all_probs = np.stack([m.predict_proba(X_te_t).numpy() for m in models])
    ens_probs = all_probs.mean(axis=0)
    ens_preds = ens_probs.argmax(1)
    ens_acc = (ens_preds == y_true).mean()
    ens_ll = log_loss(y_true, ens_probs, labels=list(range(data.n_classes)))
    ens_top2 = top_k_accuracy_score(y_true, ens_probs, k=min(2, data.n_classes),
                                     labels=list(range(data.n_classes)))

    print(_hdr("11. Ensemble Test Results (Averaged Softmax)"))
    print(_row(f"Accuracy ................ {ens_acc:.4f}"))
    print(_row(f"Log-loss ................ {ens_ll:.4f}"))
    print(_row(f"Top-2 accuracy .......... {ens_top2:.4f}"))
    print(_bot())
    print()

    # ── 12. Classification report ────────────────────────────────────
    print(_hdr("12. Classification Report (Ensemble, Test Set)"))
    print(_row())
    report = classification_report(
        y_true, ens_preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    for line in report.splitlines():
        print(_row(line))
    print(_bot())
    print()

    # ── 13. Confusion matrix ─────────────────────────────────────────
    print(_hdr("13. Confusion Matrix (rows=true, cols=predicted)"))
    print(_row())
    for line in _confusion_block(y_true, ens_preds, label_names):
        print(_row(line))
    print(_row())
    print(_row("Diagonal = correct predictions"))
    print(_bot())
    print()

    # ── 14. Per-class softmax confidence ─────────────────────────────
    print(_hdr("14. Prediction Confidence (Ensemble Softmax)"))
    print(_row(f"  {'Class':<28s} {'Mean':>7s} {'Std':>7s} {'Min':>7s} {'Max':>7s}"))
    print(_row(f"  {'─' * 28} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}"))
    for i, name in enumerate(label_names):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        conf = ens_probs[mask, i]
        print(_row(
            f"  {name:<28s} {conf.mean():>7.4f} {conf.std():>7.4f} "
            f"{conf.min():>7.4f} {conf.max():>7.4f}"
        ))
    print(_bot())
    print()

    # ── 15. Summary ──────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t0
    print(_hdr("15. Summary"))
    print(_row(f"Pipeline stages ......... 7 (split → impute → std → PCA → KM → resample → train)"))
    print(_row(f"Data leakage ............ None (.fit on train only)"))
    print(_row(f"Hyperparameter tuning ... Validation set (NOT test)"))
    print(_row(f"Ensemble members ........ {N_ENS}"))
    print(_row(f"Final test accuracy ..... {ens_acc:.4f}"))
    print(_row(f"Final test log-loss ..... {ens_ll:.4f}"))
    print(_row(f"Total wall time ......... {elapsed_total:.2f}s"))
    print(_bot())
    print()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Game-NN evaluation report")
    p.add_argument("--db", default="data/steam_data.duckdb", help="Path to DuckDB file")
    args = p.parse_args()
    run_eval(args.db)
