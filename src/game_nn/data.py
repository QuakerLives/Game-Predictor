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
from sklearn.feature_extraction.text import TfidfVectorizer
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


_EXP_LEVEL_MAP: dict[str, float] = {
    "Poor": 0.0, "Fair": 1.0, "Good": 2.0, "Excellent": 3.0, "Superior": 4.0,
}

_GAMEPLAY_QUERY = """
SELECT
    id,
    video_game_name,
    experience_level,
    source_type,
    CASE WHEN player_name          IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_player_name,
    CASE WHEN gameplay_timestamp   IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_timestamp,
    CASE WHEN channel_description  IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_channel_desc,
    LENGTH(gameplay_narration) AS narration_len,
    COALESCE(gameplay_narration, '')              AS narration_text
FROM gameplay_records
ORDER BY id
"""

# gameplay_level (97% NULL) and total_playtime (95% NULL) were removed — after median
# imputation they become constants with zero variance and add no signal.

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
    db_path: str | Path = "data/steam_data.duckdb",
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
    db_path: str | Path = "data/steam_data.duckdb",
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


def build_dataset_from_gameplay(
    db_path: str | Path = "data/gameplay_data.duckdb",
    test_record_ids: list[int] | None = None,
    tfidf_max_features: int = 100,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitData:
    """Build a feature matrix from gameplay_records for the ensemble NN.

    Features (6 numerical + up to tfidf_max_features TF-IDF terms):
      experience_level (ordinal 0-4), is_youtube (binary),
      has_player_name, has_timestamp, has_channel_desc (all binary),
      narration_len (continuous),
      + TF-IDF top-N terms from gameplay_narration text

    TF-IDF is fitted on the training split only (no leakage). Game-specific
    vocabulary in the narrations ("fleet", "empire" → Stellaris; "harvest",
    "farm" → Stardew Valley) gives the model genuine signal to learn from.

    gameplay_level and total_playtime were dropped (97%/95% NULL → zero-variance
    constants after median imputation, no signal).

    When *test_record_ids* is provided (from models/shared_split.npz), those
    records form the test set so the NN aligns with the CNN and Transformer.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(_GAMEPLAY_QUERY).fetchall()
    con.close()

    label_names = sorted({r[1] for r in rows})
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    record_ids = np.array([r[0] for r in rows], dtype=np.int64)
    y = np.array([label_to_idx[r[1]] for r in rows], dtype=np.int64)

    # _GAMEPLAY_QUERY columns: id, video_game_name, experience_level,
    #   source_type, has_player_name, has_timestamp, has_channel_desc,
    #   narration_len, narration_text
    X_num_rows: list[list[float]] = []
    texts: list[str] = []
    for r in rows:
        exp_enc = _EXP_LEVEL_MAP.get(r[2], np.nan)
        is_yt = 1.0 if r[3] == "youtube" else 0.0
        X_num_rows.append([exp_enc, is_yt, float(r[4]), float(r[5]), float(r[6]), float(r[7] or 0)])
        texts.append(r[8])

    X_num = np.array(X_num_rows, dtype=np.float64)
    texts_arr = np.array(texts, dtype=object)
    num_feature_names = [
        "experience_level", "is_youtube",
        "has_player_name", "has_timestamp", "has_channel_desc", "narration_len",
    ]

    print(f"[nn-gameplay] {X_num.shape[0]} records, {len(label_names)} classes")
    nan_counts = {num_feature_names[i]: int(np.isnan(X_num[:, i]).sum()) for i in range(X_num.shape[1])}
    print(f"[nn-gameplay] NaN counts: {nan_counts}")

    # Split — shared or independent; carry texts through the same split
    if test_record_ids is not None:
        id_to_pos = {int(rid): i for i, rid in enumerate(record_ids.tolist())}
        te_indices = [id_to_pos[rid] for rid in test_record_ids if rid in id_to_pos]
        te_set = set(te_indices)
        tr_val_indices = [i for i in range(len(record_ids)) if i not in te_set]

        X_te, y_te = X_num[te_indices], y[te_indices]
        texts_te = texts_arr[te_indices]
        X_rest, y_rest = X_num[tr_val_indices], y[tr_val_indices]
        texts_rest = texts_arr[tr_val_indices]

        adjusted_val = val_ratio / (1.0 - test_ratio)
        X_tr, X_va, y_tr, y_va, texts_tr, texts_va = train_test_split(
            X_rest, y_rest, texts_rest, test_size=adjusted_val, stratify=y_rest, random_state=seed,
        )
        print(f"[nn-gameplay] shared split  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")
    else:
        holdout = val_ratio + test_ratio
        X_tr, X_tmp, y_tr, y_tmp, texts_tr, texts_tmp = train_test_split(
            X_num, y, texts_arr, test_size=holdout, stratify=y, random_state=seed,
        )
        rel_test = test_ratio / holdout
        X_va, X_te, y_va, y_te, texts_va, texts_te = train_test_split(
            X_tmp, y_tmp, texts_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed,
        )
        print(f"[nn-gameplay] split  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    # Impute NaNs on numerical features (train-fit only)
    X_tr, X_va, X_te, _ = impute(X_tr, X_va, X_te)

    # TF-IDF on narration text (train-fit only — no leakage)
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english", min_df=2)
    Xtf_tr = tfidf.fit_transform(texts_tr.tolist()).toarray()
    Xtf_va = tfidf.transform(texts_va.tolist()).toarray()
    Xtf_te = tfidf.transform(texts_te.tolist()).toarray()
    tfidf_names = tfidf.get_feature_names_out().tolist()
    print(f"[nn-gameplay] TF-IDF vocabulary size: {len(tfidf_names)}")

    X_tr = np.hstack([X_tr, Xtf_tr])
    X_va = np.hstack([X_va, Xtf_va])
    X_te = np.hstack([X_te, Xtf_te])
    feature_names = num_feature_names + tfidf_names

    # Standardize the full feature matrix (train-fit only)
    X_tr, X_va, X_te, _ = standardize(X_tr, X_va, X_te)

    # Oversample minority classes in training set only
    pre = len(y_tr)
    X_tr, y_tr = resample_training(X_tr, y_tr, seed=seed)
    print(f"[nn-gameplay] resample  {pre} → {len(y_tr)} training samples")

    return SplitData(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        label_names=label_names,
        feature_names=feature_names,
    )
