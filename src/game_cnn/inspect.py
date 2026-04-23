"""Preprocessing inspection report.

Runs the full preprocessing pass with dropped-image collection enabled and
writes two CSV files to the working directory:

  blurry_images.csv   — every image below the blur threshold, sorted blurriest first
  duplicate_images.csv — every image rejected as a near-duplicate

Usage:
    python -m game_cnn.inspect
    python -m game_cnn.inspect --blur-threshold 80 --out-dir reports/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .data import BLUR_THRESHOLD, PHASH_TOLERANCE, preprocess_manifest


def run(
    db_path: str = "data/gameplay_data.duckdb",
    base_dir: str = "data",
    blur_threshold: float = BLUR_THRESHOLD,
    phash_tolerance: int = PHASH_TOLERANCE,
    out_dir: str = ".",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running preprocessing inspection (blur_threshold={blur_threshold}, "
          f"phash_tolerance={phash_tolerance}) ...")

    _, _, report = preprocess_manifest(
        db_path=db_path,
        base_dir=base_dir,
        blur_threshold=blur_threshold,
        phash_tolerance=phash_tolerance,
        collect_dropped=True,
    )

    # ── Blurry report ─────────────────────────────────────────────────────────
    blurry_path = out / "blurry_images.csv"
    with blurry_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "score", "path"])
        writer.writeheader()
        for row in report.blurry:
            writer.writerow({
                "game": row["game"],
                "score": f"{row['score']:.2f}",
                "path": str(row["path"]),
            })
    print(f"\nBlurry images ({len(report.blurry)}) → {blurry_path}")
    if report.blurry:
        print(f"  Blurriest: score={report.blurry[0]['score']:.2f}  {report.blurry[0]['path'].name}")
        print(f"  Least blurry dropped: score={report.blurry[-1]['score']:.2f}  "
              f"{report.blurry[-1]['path'].name}")
        print(f"  Current threshold: {blur_threshold}  "
              f"(lower = keep more, higher = stricter)")

    # ── Duplicate report ──────────────────────────────────────────────────────
    dup_path = out / "duplicate_images.csv"
    with dup_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "dropped_path", "kept_match"])
        writer.writeheader()
        for row in report.duplicates:
            writer.writerow({
                "game": row["game"],
                "dropped_path": str(row["path"]),
                "kept_match": str(row["matched_path"]),
            })
    print(f"\nDuplicate images ({len(report.duplicates)}) → {dup_path}")

    print(f"\n{report.summary()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect dropped images from preprocessing")
    p.add_argument("--db", default="data/gameplay_data.duckdb")
    p.add_argument("--base-dir", default="data")
    p.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    p.add_argument("--phash-tolerance", type=int, default=PHASH_TOLERANCE)
    p.add_argument("--out-dir", default=".", help="Directory for output CSVs")
    args = p.parse_args()

    run(
        db_path=args.db,
        base_dir=args.base_dir,
        blur_threshold=args.blur_threshold,
        phash_tolerance=args.phash_tolerance,
        out_dir=args.out_dir,
    )
