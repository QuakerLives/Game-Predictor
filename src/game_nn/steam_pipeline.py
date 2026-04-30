"""
Steam API ensemble member.

Aggregates per-game Steam API statistics (achievement counts, completion rates,
player counts, news volume) and joins them onto every gameplay record by game
name.  Trains a small FC classifier on these 7 features, aligned to the CNN's
shared test split, and saves steam_test.npz for the ensemble combiner.

Why this works: each of the 5 games has a unique combination of Steam stats
(e.g. Stellaris has ~280 achievements, Apex Legends has ~45; player-count
profiles differ substantially), so the model learns a robust game fingerprint
from external metadata rather than gameplay content — a genuinely independent
signal for the ensemble.

Run order:
    1. game-cnn-train                → trains CNN, saves shared_split.npz
    2. game-transformer-train        → Transformer on same test records
    3. python -m game_nn --gameplay  → gameplay NN on same test records
    4. game-steam-train              → this script, same test records
    5. game-ensemble                 → combines all four
"""

from __future__ import annotations

import copy
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .data import SplitData, impute, standardize, resample_training
from .model import GameClassifier

# ── Game name → Steam appid mapping ──────────────────────────────────────────
# gameplay_records uses "Skyrim"; Steam has "Skyrim Special Edition" (appid 489830).

_GAME_TO_APPID: dict[str, int] = {
    "Apex Legends":   1172470,
    "No Man's Sky":   275850,
    "Skyrim":         489830,
    "Stardew Valley": 413150,
    "Stellaris":      281990,
}

# ── SQL ───────────────────────────────────────────────────────────────────────

_STEAM_AGG_QUERY = """
SELECT
    g.appid,
    COUNT(DISTINCT as_.name)          AS total_achievements,
    COALESCE(AVG(a.percent),   0.0)   AS avg_completion_pct,
    COALESCE(MIN(a.percent),   0.0)   AS min_completion_pct,
    COALESCE(MAX(a.percent),   0.0)   AS max_completion_pct,
    COALESCE(AVG(pc.player_count), 0.0) AS avg_player_count,
    COALESCE(MAX(pc.player_count), 0.0) AS peak_player_count,
    COUNT(DISTINCT n.gid)             AS news_count
FROM games g
LEFT JOIN achievement_schema as_ ON g.appid = as_.appid
LEFT JOIN achievements       a   ON g.appid = a.appid
LEFT JOIN player_counts      pc  ON g.appid = pc.appid
LEFT JOIN news               n   ON g.appid = n.appid
GROUP BY g.appid
"""

_GAMEPLAY_ID_QUERY = """
SELECT id, video_game_name FROM gameplay_records ORDER BY id
"""

FEATURE_NAMES = [
    "total_achievements",
    "avg_completion_pct",
    "min_completion_pct",
    "max_completion_pct",
    "avg_player_count",
    "peak_player_count",
    "news_count",
]

# ── Dataset builder ───────────────────────────────────────────────────────────


def build_steam_dataset(
    steam_db:    str | Path = "data/steam_data.duckdb",
    gameplay_db: str | Path = "data/gameplay_data.duckdb",
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    seed:        int   = 42,
    test_record_ids: list[int] | None = None,
) -> SplitData:
    """Join Steam API aggregates onto gameplay records and return aligned splits."""

    # 1. Aggregate Steam API stats per appid
    conn_s = duckdb.connect(str(steam_db), read_only=True)
    steam_rows = conn_s.execute(_STEAM_AGG_QUERY).fetchall()
    conn_s.close()

    appid_to_feats: dict[int, list[float]] = {
        int(row[0]): [float(v) for v in row[1:]]
        for row in steam_rows
    }
    print("[steam] Steam API features per game:")
    for game, appid in _GAME_TO_APPID.items():
        f = appid_to_feats.get(appid, [0.0] * len(FEATURE_NAMES))
        print(f"  {game:20s}  ach={f[0]:.0f}  avg_compl={f[1]:.1f}%"
              f"  avg_players={f[4]:.0f}  news={f[6]:.0f}")

    # 2. Load gameplay record IDs + labels
    conn_g = duckdb.connect(str(gameplay_db), read_only=True)
    gp_rows = conn_g.execute(_GAMEPLAY_ID_QUERY).fetchall()
    conn_g.close()

    label_names = sorted(_GAME_TO_APPID.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    record_ids = np.array([r[0] for r in gp_rows], dtype=np.int64)
    y = np.array([label_to_idx[r[1]] for r in gp_rows], dtype=np.int64)

    # 3. Build feature matrix — each record gets its game's Steam feature vector
    X_rows: list[list[float]] = []
    for _, game_name in gp_rows:
        appid = _GAME_TO_APPID.get(game_name)
        feats = appid_to_feats.get(appid, [0.0] * len(FEATURE_NAMES)) if appid else [0.0] * len(FEATURE_NAMES)
        X_rows.append(feats)
    X = np.array(X_rows, dtype=np.float64)

    print(f"[steam] {X.shape[0]} gameplay records  |  "
          f"{len(label_names)} classes  |  {X.shape[1]} features")

    # 4. Split — aligned to CNN shared split when available
    if test_record_ids is not None:
        id_to_pos = {int(rid): i for i, rid in enumerate(record_ids.tolist())}
        te_indices   = [id_to_pos[rid] for rid in test_record_ids if rid in id_to_pos]
        te_set        = set(te_indices)
        tr_val_indices = [i for i in range(len(record_ids)) if i not in te_set]

        X_te, y_te     = X[te_indices],    y[te_indices]
        X_rest, y_rest = X[tr_val_indices], y[tr_val_indices]

        adj_val = val_ratio / (1.0 - test_ratio)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_rest, y_rest, test_size=adj_val, stratify=y_rest, random_state=seed,
        )
        print(f"[steam] shared split  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")
    else:
        holdout = val_ratio + test_ratio
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=holdout, stratify=y, random_state=seed,
        )
        rel_test = test_ratio / holdout
        X_va, X_te, y_va, y_te = train_test_split(
            X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed,
        )
        print(f"[steam] split  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    # 5. Impute → standardize → oversample (all fit on train only)
    X_tr, X_va, X_te, _ = impute(X_tr, X_va, X_te)
    X_tr, X_va, X_te, _ = standardize(X_tr, X_va, X_te)
    pre = len(y_tr)
    X_tr, y_tr = resample_training(X_tr, y_tr, seed=seed)
    print(f"[steam] resample  {pre} → {len(y_tr)} training samples")

    return SplitData(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        label_names=label_names,
        feature_names=FEATURE_NAMES,
    )


# ── Training helpers (mirrors pipeline.py) ────────────────────────────────────


def _train(
    data: SplitData,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 300,
    batch_size: int = 64,
    patience: int = 40,
    seed: int = 42,
) -> tuple[GameClassifier, float]:
    torch.manual_seed(seed)
    X_tr = torch.tensor(data.X_train, dtype=torch.float32)
    y_tr = torch.tensor(data.y_train, dtype=torch.long)
    X_va = torch.tensor(data.X_val,   dtype=torch.float32)
    y_va = torch.tensor(data.y_val,   dtype=torch.long)

    model = GameClassifier(data.n_features, data.n_classes, hidden_dims, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5,
    )

    best_acc, best_state, wait = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), batch_size):
            idx = perm[start:start + batch_size]
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_loss   = criterion(val_logits, y_va).item()
            val_acc    = (val_logits.argmax(1) == y_va).float().mean().item()
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc, best_state, wait = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1

        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}  val_acc={val_acc:.4f}")
        if wait >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_acc


def _evaluate(model: GameClassifier, data: SplitData) -> dict:
    X = torch.tensor(data.X_test, dtype=torch.float32)
    y = torch.tensor(data.y_test, dtype=torch.long)
    model.eval()
    probs = model.predict_proba(X)
    preds = probs.argmax(1)
    acc   = (preds == y).float().mean().item()
    per_class = {
        data.label_names[i]: float((preds[y == i] == i).float().mean())
        for i in range(data.n_classes) if (y == i).sum() > 0
    }
    return {"accuracy": acc, "per_class": per_class, "probs": probs.numpy()}


# ── Public entry point ────────────────────────────────────────────────────────


def run_steam_model(
    steam_db:    str | Path = "data/steam_data.duckdb",
    gameplay_db: str | Path = "data/gameplay_data.duckdb",
) -> None:
    print("=" * 60)
    print("  Game-Steam  ·  Steam API Feature Classifier")
    print("=" * 60)

    shared_split_path = Path("models/shared_split.npz")
    test_record_ids = None
    if shared_split_path.exists():
        split_data = np.load(shared_split_path, allow_pickle=True)
        test_record_ids = list(split_data["test_record_ids"].astype(int))
        print(f"[steam] using shared split ({len(test_record_ids)} test records from CNN)")
    else:
        print("[steam] no shared split found — using independent split")
        print("        (run game_cnn first to align test sets for ensemble)")

    data = build_steam_dataset(
        steam_db=steam_db,
        gameplay_db=gameplay_db,
        test_record_ids=test_record_ids,
    )

    model, best_val_acc = _train(data)
    result = _evaluate(model, data)

    print(f"\n  val_acc={best_val_acc:.4f}  test_acc={result['accuracy']:.4f}")
    for name, acc in result["per_class"].items():
        print(f"    {name:30s} {acc:.4f}")

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "state_dict":  model.state_dict(),
            "n_features":  data.n_features,
            "n_classes":   data.n_classes,
            "label_names": data.label_names,
            "feature_names": data.feature_names,
        },
        save_dir / "steam.pt",
    )
    np.savez(
        save_dir / "steam_test.npz",
        probs=result["probs"],
        y_true=data.y_test,
        label_names=np.array(data.label_names),
    )
    print(f"\n  Saved → models/steam.pt  |  models/steam_test.npz")
    print("\n" + "=" * 60)
    print("  Run 'game-ensemble' to combine all models.")
    print("=" * 60)
