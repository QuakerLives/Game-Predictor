"""Text extraction and embedding pipeline for game classification.

Fields used:
  - gameplay_narration (0.0% null) — only field used; always present

Fields excluded (all leakage risks — null density is no longer the concern):
  - channel_description (~5% null) — frequently names the game directly (leakage)
  - identifying_quotes  (~5% null) — literally game titles and review text (severe leakage)
  - player_experience_narration (~6% null) — describes channel context, names the game
  - gameplay_level / total_playtime (~12% null each) — 87%+ records hold a -1 or 150
    sentinel/default value; near-zero variance, no discriminating signal

Before embedding, game-identifying terms are scrubbed from the narration so the
model must learn gameplay patterns, not just recognize game names in the text.

Each record is embedded as a single string using
``all-MiniLM-L6-v2`` (384-dim, fast, fits well on CPU).  Embeddings are
computed once over the full dataset and cached in memory; no fitting is
required since we use a pretrained encoder.

Preprocessing order:
  1. Pull narration + label from DuckDB
  2. Scrub game-identifying terms from narration text
  3. Encode with SentenceTransformer (batched)
  4. Stratified train / val / test split
  5. Z-score standardize embeddings (fit on train only)
  6. Oversample minority classes in training set
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, Apache-2.0

# Terms that directly name a game — ordered longest-first so multi-word phrases
# are matched before their substrings (e.g. "Apex Legends" before "Apex").
_SCRUB_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Longest phrases first so multi-word names match before their substrings
        r"apex legends",
        r"no man'?s sky",
        r"the elder scrolls\s+v",
        r"elder scrolls",
        r"skyrim",
        r"dragonborn",
        r"stardew valley",
        r"stellaris",
    ]
]


# ── DB query ─────────────────────────────────────────────────────────────────

_TEXT_QUERY = """
SELECT
    id,
    video_game_name,
    gameplay_narration
FROM gameplay_records
ORDER BY id
"""


# ── Text assembly ─────────────────────────────────────────────────────────────


def _sanitize(text: str) -> str:
    """Remove game-identifying terms so the model learns gameplay patterns."""
    for pat in _SCRUB_PATTERNS:
        text = pat.sub("the game", text)
    return text.strip()


# ── SplitData container ───────────────────────────────────────────────────────


@dataclass
class SplitData:
    """Preprocessed train/val/test splits ready for the FC classifier."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: list[str]
    n_features: int = field(init=False)
    n_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features = self.X_train.shape[1]
        self.n_classes = len(self.label_names)


# ── Preprocessing helpers ─────────────────────────────────────────────────────


def _standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        scaler,
    )


def _resample(
    X: np.ndarray, y: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match majority (training set only)."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())

    parts_X, parts_y = [], []
    for cls in classes:
        mask = y == cls
        Xc, yc = X[mask], y[mask]
        deficit = target - len(Xc)
        if deficit > 0:
            idx = rng.choice(len(Xc), size=deficit, replace=True)
            Xc = np.vstack([Xc, Xc[idx]])
            yc = np.concatenate([yc, yc[idx]])
        parts_X.append(Xc)
        parts_y.append(yc)

    X_out = np.vstack(parts_X)
    y_out = np.concatenate(parts_y)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ── Public API ────────────────────────────────────────────────────────────────


def build_dataset(
    db_path: str | Path = "data/gameplay_data.duckdb",
    embed_model: str = _EMBED_MODEL,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    batch_size: int = 64,
    test_record_ids: list[int] | None = None,
) -> tuple[SplitData, StandardScaler]:
    """Full pipeline: DB → sanitize → embed → split → standardize → resample.

    Args:
        test_record_ids: When provided (passed from the CNN's shared split),
            exactly those records become the test set and the remaining records
            are split into train/val.  This aligns the Transformer test set with
            the CNN so the ensemble can combine them directly.
    """

    # 1. Pull narration + id from DB
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(_TEXT_QUERY).fetchall()
    con.close()

    label_names = sorted({r[1] for r in rows})
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    record_ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [_sanitize(r[2] or "") for r in rows]
    y = np.array([label_to_idx[r[1]] for r in rows], dtype=np.int64)

    print(f"[transformer] {len(texts)} texts, {len(label_names)} classes")

    # 2. Encode with SentenceTransformer
    print(f"[transformer] encoding with {embed_model} ...")
    import transformers as _hf; _hf.logging.set_verbosity_error()
    encoder = SentenceTransformer(embed_model)
    X = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    X = X.astype(np.float64)
    print(f"[transformer] embedding dim: {X.shape[1]}")

    # 3. Split — shared or independent
    if test_record_ids is not None:
        # Build test set in the SAME ROW ORDER as the CNN's shared split so that
        # ensemble combination is row-aligned (same record at each index).
        id_to_pos = {int(rid): i for i, rid in enumerate(record_ids.tolist())}
        te_indices = [id_to_pos[rid] for rid in test_record_ids if rid in id_to_pos]
        te_set = set(te_indices)
        tr_val_indices = [i for i in range(len(record_ids)) if i not in te_set]

        X_te, y_te = X[te_indices], y[te_indices]
        X_rest, y_rest = X[tr_val_indices], y[tr_val_indices]

        # Recalculate val fraction relative to the remaining (non-test) records.
        # e.g. val_ratio=0.15, test_ratio=0.15 → adjusted = 0.15 / 0.85 ≈ 0.176
        adjusted_val = val_ratio / (1.0 - test_ratio)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_rest, y_rest, test_size=adjusted_val, stratify=y_rest, random_state=seed,
        )
        print(
            f"[transformer] shared split  "
            f"train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}"
        )
    else:
        holdout = val_ratio + test_ratio
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=holdout, stratify=y, random_state=seed,
        )
        rel_test = test_ratio / holdout
        X_va, X_te, y_va, y_te = train_test_split(
            X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed,
        )
        print(f"[transformer] split  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    # 4. Standardize (fit on train only)
    X_tr, X_va, X_te, scaler = _standardize(X_tr, X_va, X_te)

    # 5. Oversample training set
    pre = len(y_tr)
    X_tr, y_tr = _resample(X_tr, y_tr, seed=seed)
    print(f"[transformer] resample  {pre} → {len(y_tr)} training samples")

    return SplitData(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        label_names=label_names,
    ), scaler
