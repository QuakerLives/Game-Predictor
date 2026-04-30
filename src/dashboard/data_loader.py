"""
Data loading utilities for the Dash dashboard.

All aggregations run as SQL inside DuckDB and are returned as plain Python
dicts/lists — no pandas, polars, or BLAS-backed libraries are imported here,
which avoids the OpenMP/thread-pool deadlock on Windows with PyTorch CUDA.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

DB_PATH    = Path("data/gameplay_data.duckdb")
MODELS_DIR = Path("models")

# ── Color palettes ────────────────────────────────────────────────────────────

GAME_COLORS: dict[str, str] = {
    "Apex Legends":   "#F44336",
    "No Man's Sky":   "#00BCD4",
    "Skyrim":         "#2196F3",
    "Stardew Valley": "#4CAF50",
    "Stellaris":      "#9C27B0",
}

MODEL_COLORS: dict[str, str] = {
    "CNN":         "#2196F3",
    "NN":          "#4CAF50",
    "Transformer": "#FF9800",
    "Ensemble":    "#9C27B0",
}

EXP_LEVELS = ["Poor", "Fair", "Good", "Excellent", "Superior"]

EXP_COLORS: dict[str, str] = {
    "Poor":      "#F44336",
    "Fair":      "#FF9800",
    "Good":      "#FFC107",
    "Excellent": "#8BC34A",
    "Superior":  "#4CAF50",
}

# ── Image preprocessing constants (used by CNN inference) ─────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── EDA data loader ───────────────────────────────────────────────────────────

_NULL_FIELDS = [
    "player_name", "gameplay_timestamp", "experience_level", "gameplay_level",
    "total_playtime", "channel_description", "player_experience_narration",
]


@lru_cache(maxsize=1)
def load_eda_data() -> dict:
    """
    Pre-aggregate all EDA data with SQL and return plain Python dicts/lists.
    No pandas or polars — avoids native thread-pool conflicts on Windows.
    """
    import duckdb
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Summary stats
    total  = conn.execute("SELECT COUNT(*) FROM gameplay_records").fetchone()[0]
    n_gg   = conn.execute("SELECT COUNT(*) FROM gameplay_records WHERE source_type = 'google_images'").fetchone()[0]
    n_yt   = conn.execute("SELECT COUNT(*) FROM gameplay_records WHERE source_type = 'youtube'").fetchone()[0]
    exp_ok = conn.execute("SELECT COUNT(*) FROM gameplay_records WHERE experience_level IS NOT NULL").fetchone()[0]

    # Sorted list of distinct games
    games: list[str] = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT video_game_name FROM gameplay_records ORDER BY video_game_name"
        ).fetchall()
    ]

    # Records per game: {game: count}
    records_per_game: dict[str, int] = {
        r[0]: r[1] for r in conn.execute(
            "SELECT video_game_name, COUNT(*) FROM gameplay_records "
            "GROUP BY video_game_name ORDER BY video_game_name"
        ).fetchall()
    }

    # Source type breakdown: {source_type: {game: count}}
    source_type_breakdown: dict[str, dict[str, int]] = {}
    for game, src, cnt in conn.execute(
        "SELECT video_game_name, source_type, COUNT(*) FROM gameplay_records "
        "GROUP BY video_game_name, source_type"
    ).fetchall():
        source_type_breakdown.setdefault(src, {})[game] = cnt

    # Experience distribution: {level: {game: count}}
    experience_dist: dict[str, dict[str, int]] = {}
    for game, level, cnt in conn.execute(
        "SELECT video_game_name, experience_level, COUNT(*) FROM gameplay_records "
        "WHERE experience_level IS NOT NULL GROUP BY video_game_name, experience_level"
    ).fetchall():
        experience_dist.setdefault(level, {})[game] = cnt

    # NULL density — single aggregation query for all fields
    null_sel = ", ".join(
        f"SUM(CASE WHEN {f} IS NULL THEN 1 ELSE 0 END) AS n_{f}"
        for f in _NULL_FIELDS
    )
    null_pct: dict[str, dict[str, float]] = {f: {} for f in _NULL_FIELDS}
    for row in conn.execute(
        f"SELECT video_game_name, COUNT(*) AS total, {null_sel} "
        "FROM gameplay_records GROUP BY video_game_name"
    ).fetchall():
        game_name  = row[0]
        total_game = row[1] or 1
        for i, field in enumerate(_NULL_FIELDS):
            null_pct[field][game_name] = row[2 + i] / total_game * 100

    # Narration lengths per game: {game: [int, ...]}
    narration_lengths: dict[str, list[int]] = {g: [] for g in games}
    for game, length in conn.execute(
        "SELECT video_game_name, LENGTH(gameplay_narration) FROM gameplay_records "
        "WHERE gameplay_narration IS NOT NULL AND LENGTH(gameplay_narration) > 0 "
        "ORDER BY video_game_name"
    ).fetchall():
        narration_lengths.setdefault(game, []).append(length)

    # Sample narrations — 2 per game, deterministic (lowest id first)
    sample_rows = conn.execute("""
        SELECT video_game_name, source_type, experience_level,
               SUBSTR(gameplay_narration, 1, 220) ||
               CASE WHEN LENGTH(gameplay_narration) > 220 THEN '...' ELSE '' END
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY video_game_name ORDER BY id) AS rn
            FROM gameplay_records
            WHERE gameplay_narration IS NOT NULL
        ) t WHERE rn <= 2
        ORDER BY video_game_name, rn
    """).fetchall()

    conn.close()
    return {
        "total": total, "n_gg": n_gg, "n_yt": n_yt, "exp_ok": exp_ok,
        "games": games,
        "records_per_game": records_per_game,
        "source_type_breakdown": source_type_breakdown,
        "experience_dist": experience_dist,
        "null_fields": _NULL_FIELDS,
        "null_pct": null_pct,
        "narration_lengths": narration_lengths,
        "sample_rows": sample_rows,
    }


# ── Model results loader ───────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_model_results() -> dict:
    """
    Load saved test-set predictions for CNN, NN, and Transformer models.

    Computes an ensemble by averaging the three softmax probability arrays,
    then returns accuracy, confusion matrix, and per-class accuracy for all
    four models (or however many .npz files are present).
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    out: dict = {}
    label_names: list[str] | None = None

    for key in ("cnn", "nn", "transformer"):
        path = MODELS_DIR / f"{key}_test.npz"
        if not path.exists():
            continue
        data      = np.load(path, allow_pickle=True)
        probs     = data["probs"]
        y_true    = data["y_true"].astype(int)
        lnames    = list(data["label_names"])
        if label_names is None:
            label_names = lnames
        preds     = probs.argmax(1)
        cm        = confusion_matrix(y_true, preds, labels=list(range(len(lnames))))
        acc       = float((preds == y_true).mean())
        per_class = {
            lnames[i]: float((preds[y_true == i] == i).mean())
            for i in range(len(lnames)) if (y_true == i).sum() > 0
        }
        out[key] = {
            "probs": probs, "y_true": y_true, "label_names": lnames,
            "accuracy": acc, "per_class": per_class, "cm": cm,
        }

    # Ensemble: average softmax probabilities — only when all three share the same test set
    available = [out[k] for k in ("cnn", "nn", "transformer") if k in out]
    shapes_match = len(available) == 3 and all(
        m["probs"].shape == available[0]["probs"].shape for m in available
    )
    if shapes_match and label_names is not None:
        avg_probs = np.stack([m["probs"] for m in available]).mean(0)
        y_true    = available[0]["y_true"]
        preds     = avg_probs.argmax(1)
        cm        = confusion_matrix(y_true, preds, labels=list(range(len(label_names))))
        acc       = float((preds == y_true).mean())
        per_class = {
            label_names[i]: float((preds[y_true == i] == i).mean())
            for i in range(len(label_names)) if (y_true == i).sum() > 0
        }
        out["ensemble"] = {
            "probs": avg_probs, "y_true": y_true, "label_names": label_names,
            "accuracy": acc, "per_class": per_class, "cm": cm,
        }

    out["label_names"] = label_names or []
    return out


# ── CNN live inference ────────────────────────────────────────────────────────

_cnn_model: "object | None" = None
_cnn_labels: "list[str] | None" = None


def get_cnn_model() -> tuple:
    """Lazy-load the trained CNN weights. Returns (model, label_names) or (None, None)."""
    global _cnn_model, _cnn_labels
    if _cnn_model is not None:
        return _cnn_model, _cnn_labels
    path = MODELS_DIR / "cnn.pt"
    if not path.exists():
        return None, None
    import torch
    from game_cnn.model import GameCNN
    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    _cnn_model = GameCNN(ckpt["n_classes"])
    _cnn_model.load_state_dict(ckpt["state_dict"])
    _cnn_model.eval()
    _cnn_labels = list(ckpt["label_names"])
    return _cnn_model, _cnn_labels


def predict_from_b64(image_b64: str) -> dict[str, float]:
    """
    Run CNN inference on a base64-encoded image string (from dcc.Upload).
    Returns a dict mapping each game name to its predicted probability.
    """
    import base64
    import io
    import torch
    from PIL import Image
    from torchvision import transforms

    model, labels = get_cnn_model()
    if model is None:
        return {}

    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    _, data = image_b64.split(",", 1)
    img     = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    tensor  = cnn_transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_proba(tensor).squeeze(0).numpy()
    return {name: float(p) for name, p in zip(labels, probs)}
