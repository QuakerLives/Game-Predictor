"""Data extraction from DuckDB and preprocessing pipeline.

Preprocessing order (all fitted on training split ONLY):
  1. Train / Val / Test stratified split
  2. Median imputation
  3. Z-score standardization
  4. PCA
  5. K-Means cluster feature
  6. Oversample minority classes in training set
"""

from __future__ import annotations

import duckdb
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    """Container for preprocessed train/val/test splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: list[str]
    feature_names: list[str]
    n_features: int = field(init=False)
    n_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features = self.X_train.shape[1]
        self.n_classes = len(self.label_names)


_FEATURE_QUERY = """
WITH ranked_ach AS (
    SELECT appid, name AS ach_name, percent,
           ROW_NUMBER() OVER (PARTITION BY appid ORDER BY name) AS rn
    FROM achievements
),
ranked_news AS (
    SELECT appid, CAST(gid AS DOUBLE) AS gid_numeric,
           ROW_NUMBER() OVER (PARTITION BY appid ORDER BY gid) AS rn
    FROM news
),
ach_counts AS (
    SELECT appid, COUNT(name) AS ach_count
    FROM achievement_schema
    GROUP BY appid
),
player_avg AS (
    SELECT appid, AVG(player_count) AS player_count
    FROM player_counts
    GROUP BY appid
)
SELECT
    g.name        AS game_name,
    ac.ach_count,
    ra.percent,
    rn.gid_numeric,
    pa.player_count
FROM ranked_ach ra
JOIN games g       ON ra.appid = g.appid
JOIN ach_counts ac ON ra.appid = ac.appid
LEFT JOIN ranked_news rn ON ra.appid = rn.appid AND ra.rn = rn.rn
JOIN player_avg pa ON ra.appid = pa.appid
ORDER BY g.name, ra.ach_name
"""


def extract_features(
    db_path: str | Path = "steam_data.duckdb",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build feature matrix X and integer label vector y from DuckDB.

    Returns (X, y, label_names, feature_names).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(_FEATURE_QUERY).fetchall()
    con.close()

    feature_names = ["ach_count", "percent", "gid", "player_count"]
    label_set: list[str] = sorted({r[0] for r in rows})
    label_to_idx = {name: i for i, name in enumerate(label_set)}

    X = np.array(
        [[r[1], r[2], r[3] if r[3] is not None else np.nan, r[4]] for r in rows],
        dtype=np.float64,
    )
    y = np.array([label_to_idx[r[0]] for r in rows], dtype=np.int64)
    return X, y, label_set, feature_names


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified three-way split."""
    holdout = val_ratio + test_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=seed,
    )
    relative_test = test_ratio / holdout
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test, stratify=y_tmp, random_state=seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def impute(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer]:
    """Median imputation fitted on training set only."""
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_val = imp.transform(X_val)
    X_test = imp.transform(X_test)
    return X_train, X_val, X_test, imp


def standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Z-score standardization fitted on training set only."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """PCA fitted on training set only."""
    if n_components is None:
        n_components = min(X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    return X_train, X_val, X_test, pca


def apply_kmeans(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_clusters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, KMeans]:
    """K-Means fitted on training set; cluster ID appended as an extra feature."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    c_train = km.fit_predict(X_train).reshape(-1, 1).astype(np.float64)
    c_val = km.predict(X_val).reshape(-1, 1).astype(np.float64)
    c_test = km.predict(X_test).reshape(-1, 1).astype(np.float64)
    return (
        np.hstack([X_train, c_train]),
        np.hstack([X_val, c_val]),
        np.hstack([X_test, c_test]),
        km,
    )


def resample_training(
    X: np.ndarray, y: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match the majority class (training set only)."""
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


def build_dataset(
    db_path: str | Path = "steam_data.duckdb",
    pca_components: int | None = None,
    n_clusters: int = 5,
    seed: int = 42,
) -> SplitData:
    """End-to-end pipeline: extract → split → impute → scale → PCA → KMeans → resample."""
    X, y, label_names, feature_names = extract_features(db_path)
    print(f"[data] {X.shape[0]} samples, {X.shape[1]} features, {len(label_names)} classes")
    nan_count = int(np.isnan(X).sum())
    print(f"[data] {nan_count} missing values before imputation")

    # 1. Split FIRST — no data leakage
    X_tr, y_tr, X_va, y_va, X_te, y_te = stratified_split(X, y, seed=seed)
    print(f"[data] split  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    # 2. Imputation (train-fit only)
    X_tr, X_va, X_te, _ = impute(X_tr, X_va, X_te)

    # 3. Standardization (train-fit only)
    X_tr, X_va, X_te, _ = standardize(X_tr, X_va, X_te)

    # 4. PCA (train-fit only)
    n_comp = pca_components or min(X_tr.shape[1], X_tr.shape[0])
    X_tr, X_va, X_te, pca = apply_pca(X_tr, X_va, X_te, n_components=n_comp)
    print(
        f"[data] PCA {pca.n_components_} components, "
        f"explained variance {pca.explained_variance_ratio_.sum():.4f}"
    )

    # 5. K-Means cluster feature (train-fit only)
    X_tr, X_va, X_te, _ = apply_kmeans(X_tr, X_va, X_te, n_clusters=n_clusters)
    print(f"[data] K-Means {n_clusters} clusters → feature dim now {X_tr.shape[1]}")

    # 6. Oversample training set ONLY
    pre_resample = len(y_tr)
    X_tr, y_tr = resample_training(X_tr, y_tr, seed=seed)
    print(f"[data] resample  {pre_resample} → {len(y_tr)} training samples")

    return SplitData(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        label_names=label_names,
        feature_names=feature_names,
    )
