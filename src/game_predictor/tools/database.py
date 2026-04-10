"""
DuckDB database initialization, record insertion, and validation utilities.
See design doc §3 for schema and sentinel conventions.
"""

import logging
from pathlib import Path

import duckdb
from PIL import Image

from game_predictor.config import DB_PATH, GAME_SLUGS, IMAGE_DIR, SENTINEL_STR
from game_predictor.models import GameplayRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def init_db() -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB and ensure the schema, sequences, and image dirs exist."""
    conn = duckdb.connect(DB_PATH)

    conn.execute("""
        CREATE TYPE IF NOT EXISTS qual
        AS ENUM ('Poor', 'Fair', 'Good', 'Excellent', 'Superior');
    """)

    conn.execute("CREATE SEQUENCE IF NOT EXISTS gameplay_id_seq START 1;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gameplay_records (
            id                          INTEGER DEFAULT nextval('gameplay_id_seq') PRIMARY KEY,
            video_game_name             VARCHAR NOT NULL,
            image_path                  VARCHAR NOT NULL,
            player_name                 VARCHAR NOT NULL,
            gameplay_timestamp          TIMESTAMP NOT NULL,
            experience_level            qual NOT NULL,
            gameplay_level              INTEGER NOT NULL,
            total_playtime              INTEGER NOT NULL,
            gameplay_narration          VARCHAR NOT NULL,
            channel_description         VARCHAR NOT NULL,
            player_experience_narration VARCHAR NOT NULL,
            identifying_quotes          VARCHAR[] NOT NULL,
            source_url                  VARCHAR NOT NULL,
            source_type                 VARCHAR NOT NULL,
            scraped_at                  TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
    """)

    for slug in GAME_SLUGS:
        game_dir = IMAGE_DIR / slug
        game_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured image directory: %s", game_dir)

    logger.info("Database initialised at %s", DB_PATH)
    return conn


# ---------------------------------------------------------------------------
# Record insertion
# ---------------------------------------------------------------------------

async def write_to_database(
    record: GameplayRecord,
    conn: duckdb.DuckDBPyConnection,
) -> int:
    """Insert a GameplayRecord and return the assigned row id.

    Iron Rules — these two fields must contain real data, never sentinels:
      • image_path must be a non-empty string.
      • gameplay_narration must not be empty or the sentinel value.
    """
    assert record.image_path, "Iron Rule violated: image_path is empty"
    assert (
        record.gameplay_narration
        and record.gameplay_narration != SENTINEL_STR
    ), "Iron Rule violated: gameplay_narration is sentinel or empty"

    result = conn.execute(
        """
        INSERT INTO gameplay_records (
            video_game_name, image_path, player_name,
            gameplay_timestamp, experience_level, gameplay_level,
            total_playtime, gameplay_narration, channel_description,
            player_experience_narration, identifying_quotes,
            source_url, source_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id;
        """,
        [
            record.video_game_name,
            record.image_path,
            record.player_name,
            record.gameplay_timestamp,
            record.experience_level,
            record.gameplay_level,
            record.total_playtime,
            record.gameplay_narration,
            record.channel_description,
            record.player_experience_narration,
            record.identifying_quotes,
            record.source_url,
            record.source_type,
        ],
    )

    row_id: int = result.fetchone()[0]
    logger.info("Inserted record id=%d for %s", row_id, record.video_game_name)
    return row_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_SENTINEL_MAP: dict[str, object] = {
    "player_name": "N/A",
    "gameplay_timestamp": "1970-01-01 00:00:00",
    "experience_level": "Fair",
    "gameplay_level": -1,
    "total_playtime": -1,
    "gameplay_narration": "N/A",
    "channel_description": "N/A",
    "player_experience_narration": "N/A",
}

_NARRATION_BANNED = [
    "screenshot",
    "image",
    "as shown",
    "depicted",
    "visible",
]


def validate_database(conn: duckdb.DuckDBPyConnection) -> dict:
    """Run quality-assurance checks against gameplay_records and return a report."""
    report: dict = {}

    # --- total count ---
    total = conn.execute("SELECT count(*) FROM gameplay_records;").fetchone()[0]
    report["total_records"] = total
    report["target_met"] = total >= 1000

    # --- per-game distribution ---
    rows = conn.execute(
        "SELECT video_game_name, count(*) AS cnt "
        "FROM gameplay_records GROUP BY video_game_name ORDER BY cnt DESC;"
    ).fetchall()
    report["per_game"] = {name: cnt for name, cnt in rows}

    # --- sentinel density ---
    sentinel_density: dict[str, dict] = {}
    for col, sentinel in _SENTINEL_MAP.items():
        if isinstance(sentinel, str):
            count = conn.execute(
                f"SELECT count(*) FROM gameplay_records WHERE {col} = ?;",
                [sentinel],
            ).fetchone()[0]
        else:
            count = conn.execute(
                f"SELECT count(*) FROM gameplay_records WHERE {col} = ?;",
                [sentinel],
            ).fetchone()[0]
        pct = (count / total * 100) if total else 0.0
        sentinel_density[col] = {"count": count, "pct": round(pct, 2)}
    report["sentinel_density"] = sentinel_density

    # --- image file existence & minimum size ---
    image_paths = conn.execute(
        "SELECT id, image_path FROM gameplay_records;"
    ).fetchall()
    missing, too_small, valid_images = [], [], 0
    for row_id, img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            missing.append({"id": row_id, "path": img_path})
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
            if w < 200 or h < 200:
                too_small.append({"id": row_id, "path": img_path, "size": (w, h)})
            else:
                valid_images += 1
        except Exception as exc:
            missing.append({"id": row_id, "path": img_path, "error": str(exc)})

    report["images"] = {
        "valid": valid_images,
        "missing_or_corrupt": missing,
        "too_small": too_small,
    }

    # --- narration independence ---
    violations: list[dict] = []
    for term in _NARRATION_BANNED:
        rows = conn.execute(
            "SELECT id, gameplay_narration FROM gameplay_records "
            "WHERE gameplay_narration ILIKE ?;",
            [f"%{term}%"],
        ).fetchall()
        for row_id, narration in rows:
            violations.append({"id": row_id, "term": term, "narration": narration})
    report["narration_violations"] = violations

    logger.info(
        "Validation complete — %d records, %d image issues, %d narration violations",
        total,
        len(missing) + len(too_small),
        len(violations),
    )
    return report


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    connection = init_db()

    sample = GameplayRecord(
        video_game_name="Stellaris",
        image_path="images/stellaris/test.png",
        gameplay_narration="The player expanded into a new star cluster using a science ship.",
        source_url="https://example.com",
        source_type="google_images",
    )

    rid = asyncio.run(write_to_database(sample, connection))
    print(f"Inserted test record with id={rid}")

    results = validate_database(connection)
    for key, value in results.items():
        print(f"\n{key}: {value}")

    connection.close()
