"""Image preprocessing pipeline and PyTorch Dataset.

Preprocessing order (fitted/computed on full manifest before splitting):
  1. Read id + image_path + video_game_name from DuckDB
  2. Blur detection  — Laplacian variance < threshold → dropped
  3. Perceptual-hash deduplication — pHash Hamming distance < tolerance → dropped
  4. Stratified train / val / test split

Each manifest entry carries a record_id so the split can be shared with the
Transformer pipeline, ensuring both models are evaluated on the same records.

At runtime (Dataset.__getitem__):
  5. Resize to 224 × 224
  6. Normalize to ImageNet mean/std
  7. Data augmentation on train split only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import imagehash
import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_SIZE = (224, 224)
BLUR_THRESHOLD = 100.0
PHASH_TOLERANCE = 5

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# ── Types ────────────────────────────────────────────────────────────────────

# (absolute_path, label_index, record_id)
ManifestEntry = tuple[Path, int, int]


@dataclass
class LoaderBundle:
    """Container returned by build_loaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    label_names: list[str]
    test_record_ids: list[int]
    n_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_classes = len(self.label_names)


# ── Blur detection ───────────────────────────────────────────────────────────


def _blur_score(img: Image.Image) -> float:
    gray = np.array(img.convert("L"), dtype=np.float64)
    return float(laplace(gray).var())


# ── DroppedReport ─────────────────────────────────────────────────────────────


@dataclass
class DroppedReport:
    """Collects details about every image rejected during preprocessing."""

    blurry: list[dict]      # [{path, game, score}] sorted blurriest first
    duplicates: list[dict]  # [{path, game, matched_path}]

    def summary(self) -> str:
        return f"Blurry: {len(self.blurry)}  |  Duplicates: {len(self.duplicates)}"


# ── Preprocessing ────────────────────────────────────────────────────────────


def preprocess_manifest(
    db_path: str | Path = "data/gameplay_data.duckdb",
    base_dir: str | Path = ".",
    blur_threshold: float = BLUR_THRESHOLD,
    phash_tolerance: int = PHASH_TOLERANCE,
    collect_dropped: bool = False,
) -> tuple[list[ManifestEntry], list[str]] | tuple[list[ManifestEntry], list[str], DroppedReport]:
    """Read the DB, filter blurry/duplicate images, return a clean manifest.

    Each manifest entry is (absolute_path, label_index, record_id).
    The record_id enables sharing the split with the Transformer pipeline.
    """
    base = Path(base_dir).resolve()

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT id, image_path, video_game_name FROM gameplay_records ORDER BY id"
    ).fetchall()
    con.close()

    label_names: list[str] = sorted({r[2] for r in rows})
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    seen_hashes: list[tuple[imagehash.ImageHash, Path]] = []
    manifest: list[ManifestEntry] = []
    dropped_missing = 0
    blurry_list: list[dict] = []
    dup_list: list[dict] = []

    for record_id, rel_path, game_name in rows:
        abs_path = base / rel_path

        if not abs_path.exists():
            dropped_missing += 1
            continue

        try:
            img = Image.open(abs_path).convert("RGB")
        except Exception:
            dropped_missing += 1
            continue

        score = _blur_score(img)
        if score < blur_threshold:
            blurry_list.append({"path": abs_path, "game": game_name, "score": score})
            continue

        img_hash = imagehash.phash(img)
        matched = next(
            (p for h, p in seen_hashes if img_hash - h < phash_tolerance), None
        )
        if matched is not None:
            dup_list.append({"path": abs_path, "game": game_name, "matched_path": matched})
            continue
        seen_hashes.append((img_hash, abs_path))

        manifest.append((abs_path, label_to_idx[game_name], record_id))

    blurry_list.sort(key=lambda x: x["score"])

    print(
        f"[data] manifest: {len(manifest)} kept  |  "
        f"{dropped_missing} missing  |  {len(blurry_list)} blurry  |  "
        f"{len(dup_list)} duplicates"
    )
    if not manifest:
        raise FileNotFoundError(
            f"No images found — expected images relative to: {base}\n"
            f"Example path checked: {base / rows[0][1] if rows else '(no rows)'}"
        )

    if collect_dropped:
        return manifest, label_names, DroppedReport(blurry=blurry_list, duplicates=dup_list)
    return manifest, label_names


def split_manifest(
    manifest: list[ManifestEntry],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[ManifestEntry], list[ManifestEntry], list[ManifestEntry]]:
    """Stratified three-way split; preserves full ManifestEntry tuples."""
    indices = list(range(len(manifest)))
    labels = [e[1] for e in manifest]

    holdout = val_ratio + test_ratio
    tr_i, tmp_i, _, tmp_l = train_test_split(
        indices, labels, test_size=holdout, stratify=labels, random_state=seed,
    )
    rel_test = test_ratio / holdout
    va_i, te_i, _, _ = train_test_split(
        tmp_i, tmp_l, test_size=rel_test, stratify=tmp_l, random_state=seed,
    )

    train = [manifest[i] for i in tr_i]
    val   = [manifest[i] for i in va_i]
    test  = [manifest[i] for i in te_i]
    print(f"[data] split  train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


# ── PyTorch Dataset ──────────────────────────────────────────────────────────


def _make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(TARGET_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class GameImageDataset(Dataset):
    def __init__(self, manifest: list[ManifestEntry], train: bool = False) -> None:
        self.manifest = manifest
        self.transform = _make_transforms(train)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        path, label, _record_id = self.manifest[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── End-to-end loader builder ────────────────────────────────────────────────


def build_loaders(
    db_path: str | Path = "data/gameplay_data.duckdb",
    base_dir: str | Path = ".",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    blur_threshold: float = BLUR_THRESHOLD,
    phash_tolerance: int = PHASH_TOLERANCE,
) -> LoaderBundle:
    """Full pipeline: DB → preprocess → split → DataLoaders."""
    manifest, label_names = preprocess_manifest(
        db_path, base_dir, blur_threshold, phash_tolerance,
    )
    train_m, val_m, test_m = split_manifest(manifest, seed=seed)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    return LoaderBundle(
        train=DataLoader(GameImageDataset(train_m, train=True), shuffle=True, **loader_kwargs),
        val=DataLoader(GameImageDataset(val_m, train=False), shuffle=False, **loader_kwargs),
        test=DataLoader(GameImageDataset(test_m, train=False), shuffle=False, **loader_kwargs),
        label_names=label_names,
        test_record_ids=[e[2] for e in test_m],
    )
