#!/usr/bin/env python3
"""
TRIAL RUN — configurable records per game (default 5, 25 total).
Validates the full pipeline end-to-end at minimal scale.

Usage:
    python test_run.py [-n 1]  # 1 record per game (fastest sanity check)
    python test_run.py         # default: 5 records per game (25 total)
    python test_run.py -n 10   # 10 records per game (50 total)

Writes to:
    gameplay_data_TEST.duckdb
    images_test/{game_slug}/
    test_run.log
"""

import asyncio
import argparse
import logging
import sys
import re
import json
from datetime import datetime
from pathlib import Path

import duckdb
from PIL import Image

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------

TEST_DB_PATH = "gameplay_data_TEST.duckdb"
TEST_IMAGE_DIR = Path("images_test")
TEST_LOG_FILE = "test_run.log"
DEFAULT_TARGET_PER_GAME = 5

GAMES = [
    {"name": "Stellaris",       "slug": "stellaris",
     "queries_img": ["Stellaris gameplay screenshot"],
     "queries_yt":  ["Stellaris gameplay"]},
    {"name": "No Man's Sky",    "slug": "no_mans_sky",
     "queries_img": ["No Man's Sky screenshot"],
     "queries_yt":  ["No Man's Sky exploration"]},
    {"name": "Apex Legends",    "slug": "apex_legends",
     "queries_img": ["Apex Legends gameplay"],
     "queries_yt":  ["Apex Legends ranked"]},
    {"name": "Stardew Valley",  "slug": "stardew_valley",
     "queries_img": ["Stardew Valley farm screenshot"],
     "queries_yt":  ["Stardew Valley gameplay"]},
    {"name": "Skyrim",          "slug": "skyrim",
     "queries_img": ["Skyrim screenshot"],
     "queries_yt":  ["Skyrim gameplay"]},
]

SENTINEL_STR       = "N/A"
SENTINEL_INT       = -1
SENTINEL_TIMESTAMP = datetime(1970, 1, 1)
SENTINEL_QUAL      = "Fair"
SENTINEL_QUOTES    = ["N/A"]

IMAGE_REF_PATTERN = re.compile(
    r'(?i)\b(image|screenshot|picture|shown|depicted|visible'
    r'|as seen|can be seen|looks like|appears in)\b'
)

# ---------------------------------------------------------------------------
# 1. LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(TEST_LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("test_run")

# ---------------------------------------------------------------------------
# 2. DATABASE INIT (test-isolated)
# ---------------------------------------------------------------------------

def init_test_db() -> duckdb.DuckDBPyConnection:
    """Create test DB. Drops and recreates table for a clean slate."""
    p = Path(TEST_DB_PATH)
    if p.exists():
        p.unlink()
    conn = duckdb.connect(TEST_DB_PATH)
    conn.execute("CREATE TYPE IF NOT EXISTS qual AS ENUM ('Poor','Fair','Good','Excellent','Superior');")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS gameplay_id_seq START 1;")
    conn.execute("""
        CREATE TABLE gameplay_records (
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
    for game in GAMES:
        (TEST_IMAGE_DIR / game["slug"]).mkdir(parents=True, exist_ok=True)
    logger.info(f"Test DB initialized: {TEST_DB_PATH}")
    return conn

# ---------------------------------------------------------------------------
# 3. TOOL IMPORTS
# ---------------------------------------------------------------------------

def import_tools(llm_base_url: str):
    """
    Import and configure all production tools.
    Returns a dict of tool callables.
    """
    from game_predictor.tools.search import search_google_images, search_youtube
    from game_predictor.tools.extract import click_article_and_extract
    from game_predictor.tools.screenshot import screenshot_youtube_video, extract_youtube_channel_info
    from game_predictor.tools.download import download_image
    from game_predictor.tools.narrate import generate_narration
    from game_predictor.tools.assess import assess_experience_level, validate_gameplay_image

    return {
        "search_google_images": search_google_images,
        "search_youtube": search_youtube,
        "click_article_and_extract": click_article_and_extract,
        "screenshot_youtube_video": screenshot_youtube_video,
        "extract_youtube_channel_info": extract_youtube_channel_info,
        "download_image": download_image,
        "generate_narration": generate_narration,
        "assess_experience_level": assess_experience_level,
        "validate_gameplay_image": validate_gameplay_image,
    }

# ---------------------------------------------------------------------------
# 4. PER-GAME TRIAL LOGIC (sequential for clear error attribution)
# ---------------------------------------------------------------------------

async def trial_game(
    game: dict,
    conn: duckdb.DuckDBPyConnection,
    tools: dict,
    target: int = DEFAULT_TARGET_PER_GAME,
):
    """
    Collect *target* records for one game.
    Processes candidates ONE AT A TIME for clear error attribution.
    Google Images gets ~60% of the target, YouTube gets the rest.
    """
    slug = game["slug"]
    name = game["name"]
    collected = 0
    google_target = max(1, round(target * 0.6))
    yt_target = target - google_target
    failures = {"image_dl": 0, "narration": 0, "article": 0, "yt_screenshot": 0, "ad_rejected": 0}

    # --- Phase A: Google Images ---
    logger.info(f"[{name}] Phase A: Google Images (target {google_target})")
    img_results = await tools["search_google_images"](game["queries_img"][0], num_results=max(10, google_target * 3))
    logger.info(f"[{name}] Got {len(img_results)} image candidates")

    for i, result in enumerate(img_results):
        if collected >= google_target:
            break
        record_id = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
        logger.info(f"[{name}] Processing Google Images candidate {i+1} -> record {record_id}")

        # Step 1: Download image (IRON RULE)
        img_path = await tools["download_image"](
            result.image_url, slug, record_id,
            image_dir=TEST_IMAGE_DIR
        )
        if img_path is None:
            failures["image_dl"] += 1
            logger.warning(f"[{name}] Image download failed for candidate {i+1}, skipping")
            continue

        # Step 1b: Validate image is actual gameplay (reject ads)
        is_valid = await tools["validate_gameplay_image"](img_path, name)
        if not is_valid:
            failures["ad_rejected"] += 1
            logger.warning(f"[{name}] Image rejected by Gemma 4 (ad/unrelated) for candidate {i+1}")
            Path(img_path).unlink(missing_ok=True)
            continue

        # Step 2: Extract article
        try:
            article = await tools["click_article_and_extract"](result.source_page_url, name)
        except Exception as e:
            failures["article"] += 1
            logger.warning(f"[{name}] Article extraction failed: {e}")
            article = None

        body_text = article.body_text if article and article.body_text else f"A {name} gameplay session."

        # Step 3: Generate narration (IRON RULE)
        narration = await tools["generate_narration"](body_text, name)
        if narration is None or narration == SENTINEL_STR:
            failures["narration"] += 1
            logger.warning(f"[{name}] Narration generation failed for candidate {i+1}, skipping")
            Path(img_path).unlink(missing_ok=True)
            continue

        if IMAGE_REF_PATTERN.search(narration):
            logger.warning(f"[{name}] Narration contains image reference, regenerating...")
            narration = await tools["generate_narration"](body_text, name)
            if narration is None or IMAGE_REF_PATTERN.search(narration or ""):
                failures["narration"] += 1
                Path(img_path).unlink(missing_ok=True)
                continue

        # Step 4: Assess experience
        try:
            exp_level = await tools["assess_experience_level"](img_path, body_text, name)
        except Exception:
            exp_level = SENTINEL_QUAL

        # Step 5: Assemble and write
        conn.execute("""
            INSERT INTO gameplay_records (
                video_game_name, image_path, player_name, gameplay_timestamp,
                experience_level, gameplay_level, total_playtime,
                gameplay_narration, channel_description, player_experience_narration,
                identifying_quotes, source_url, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            name,
            str(img_path),
            article.author_name if article else SENTINEL_STR,
            article.publish_date if article else SENTINEL_TIMESTAMP,
            exp_level,
            article.gameplay_level if article else SENTINEL_INT,
            article.total_playtime if article else SENTINEL_INT,
            narration,
            article.site_description if article else SENTINEL_STR,
            article.player_experience_summary if article else SENTINEL_STR,
            article.identifying_quotes if article else SENTINEL_QUOTES,
            result.source_page_url,
            "google_images",
        ])
        collected += 1
        logger.info(f"[{name}] Record {record_id} inserted ({collected}/{target})")

    # --- Phase B: YouTube ---
    if yt_target <= 0:
        logger.info(f"[{name}] Skipping YouTube phase (target met by Google Images)")
    else:
        logger.info(f"[{name}] Phase B: YouTube (target {yt_target})")
    yt_results = await tools["search_youtube"](game["queries_yt"][0], num_results=max(8, yt_target * 3)) if yt_target > 0 else []
    logger.info(f"[{name}] Got {len(yt_results)} YouTube candidates")

    for j, video in enumerate(yt_results):
        if collected >= target:
            break
        record_id = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
        logger.info(f"[{name}] Processing YouTube candidate {j+1} -> record {record_id}")

        img_path = await tools["screenshot_youtube_video"](
            video["url"], slug, record_id,
            image_dir=TEST_IMAGE_DIR
        )
        if img_path is None:
            failures["yt_screenshot"] += 1
            logger.warning(f"[{name}] YouTube screenshot failed for candidate {j+1}, skipping")
            continue

        # Validate screenshot is actual gameplay (reject ads)
        is_valid = await tools["validate_gameplay_image"](img_path, name)
        if not is_valid:
            failures["ad_rejected"] += 1
            logger.warning(f"[{name}] YouTube screenshot rejected by Gemma 4 (ad/unrelated) for candidate {j+1}")
            Path(img_path).unlink(missing_ok=True)
            continue

        try:
            channel = await tools["extract_youtube_channel_info"](video.get("channel_url", ""))
        except Exception:
            channel = {}

        yt_context = video.get("description", "") or video.get("title", f"A {name} video.")
        narration = await tools["generate_narration"](yt_context, name)
        if narration is None or narration == SENTINEL_STR:
            failures["narration"] += 1
            Path(img_path).unlink(missing_ok=True)
            continue

        if IMAGE_REF_PATTERN.search(narration):
            narration = await tools["generate_narration"](yt_context, name)
            if narration is None or IMAGE_REF_PATTERN.search(narration or ""):
                failures["narration"] += 1
                Path(img_path).unlink(missing_ok=True)
                continue

        try:
            exp_level = await tools["assess_experience_level"](img_path, yt_context, name)
        except Exception:
            exp_level = SENTINEL_QUAL

        conn.execute("""
            INSERT INTO gameplay_records (
                video_game_name, image_path, player_name, gameplay_timestamp,
                experience_level, gameplay_level, total_playtime,
                gameplay_narration, channel_description, player_experience_narration,
                identifying_quotes, source_url, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            name,
            str(img_path),
            video.get("channel_name", SENTINEL_STR),
            video.get("upload_date", SENTINEL_TIMESTAMP),
            exp_level,
            SENTINEL_INT,
            SENTINEL_INT,
            narration,
            channel.get("description", SENTINEL_STR),
            channel.get("description", SENTINEL_STR),
            SENTINEL_QUOTES,
            video["url"],
            "youtube",
        ])
        collected += 1
        logger.info(f"[{name}] Record {record_id} inserted ({collected}/{target})")

    logger.info(f"[{name}] Finished: {collected}/{target} records | Failures: {failures}")
    return collected, failures

# ---------------------------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------------------------

def validate_test_db(conn, expected_total: int = 25) -> dict:
    report = {}
    total = conn.execute("SELECT COUNT(*) FROM gameplay_records").fetchone()[0]
    report["total_records"] = total
    report["target_met"] = total >= expected_total

    per_game = conn.execute("""
        SELECT video_game_name, COUNT(*) FROM gameplay_records
        GROUP BY video_game_name ORDER BY video_game_name
    """).fetchall()
    report["per_game"] = {r[0]: r[1] for r in per_game}

    broken_images = conn.execute("""
        SELECT COUNT(*) FROM gameplay_records WHERE image_path = 'N/A' OR image_path = ''
    """).fetchone()[0]
    broken_narrations = conn.execute("""
        SELECT COUNT(*) FROM gameplay_records WHERE gameplay_narration = 'N/A' OR gameplay_narration = ''
    """).fetchone()[0]
    report["iron_rule_image_violations"] = broken_images
    report["iron_rule_narration_violations"] = broken_narrations

    image_refs = conn.execute("""
        SELECT COUNT(*) FROM gameplay_records
        WHERE gameplay_narration ILIKE '%screenshot%'
           OR gameplay_narration ILIKE '%image%'
           OR gameplay_narration ILIKE '%as shown%'
           OR gameplay_narration ILIKE '%depicted%'
    """).fetchone()[0]
    report["narration_image_ref_violations"] = image_refs

    for col, cond in [
        ("player_name", "= 'N/A'"), ("gameplay_level", "= -1"), ("total_playtime", "= -1"),
    ]:
        pct = conn.execute(f"""
            SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE {col} {cond}) / COUNT(*), 1)
            FROM gameplay_records
        """).fetchone()[0]
        report[f"sentinel_pct_{col}"] = pct

    rows = conn.execute("SELECT id, image_path FROM gameplay_records").fetchall()
    missing = []
    corrupt = []
    for rid, path in rows:
        p = Path(path)
        if not p.exists():
            missing.append(rid)
        else:
            try:
                img = Image.open(p)
                img.verify()
                if img.size[0] < 100 or img.size[1] < 100:
                    corrupt.append(rid)
            except Exception:
                corrupt.append(rid)
    report["missing_image_files"] = missing
    report["corrupt_image_files"] = corrupt

    return report

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Trial run: configurable records per game")
    parser.add_argument("-n", "--count", type=int, default=DEFAULT_TARGET_PER_GAME,
                        help=f"Records to collect per game (default: {DEFAULT_TARGET_PER_GAME})")
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1",
                        help="Base URL for the Ollama OpenAI-compatible endpoint")
    args = parser.parse_args()

    target_per_game = max(1, args.count)
    total_target = target_per_game * len(GAMES)

    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"TRIAL RUN -- {target_per_game} record(s) x {len(GAMES)} games = {total_target} total")
    logger.info(f"LLM endpoint: {args.llm_base_url}")
    logger.info(f"Test DB: {TEST_DB_PATH}")
    logger.info(f"Test images: {TEST_IMAGE_DIR}/")
    logger.info("=" * 60)

    conn = init_test_db()
    tools = import_tools(args.llm_base_url)

    total_collected = 0
    total_failures = {"image_dl": 0, "narration": 0, "article": 0, "yt_screenshot": 0, "ad_rejected": 0}
    for game in GAMES:
        logger.info(f"\n{'='*40}")
        logger.info(f"  GAME: {game['name']}")
        logger.info(f"{'='*40}")
        try:
            collected, failures = await trial_game(game, conn, tools, target=target_per_game)
            total_collected += collected
            for k in total_failures:
                total_failures[k] += failures.get(k, 0)
        except Exception as e:
            logger.error(f"Game {game['name']} crashed: {e}", exc_info=True)

    logger.info(f"\n{'='*40}")
    logger.info("  VALIDATION")
    logger.info(f"{'='*40}")
    report = validate_test_db(conn, expected_total=total_target)

    elapsed = datetime.now() - start
    logger.info(f"\nTrial run completed in {elapsed}")
    logger.info(f"Records collected: {total_collected}/{total_target}")
    logger.info(f"Failures: {total_failures}")
    logger.info(f"\nValidation report:")
    for k, v in report.items():
        status = "PASS" if not (
            (k == "target_met" and not v) or
            (k.endswith("violations") and v > 0) or
            (k in ("missing_image_files", "corrupt_image_files") and len(v) > 0)
        ) else "FAIL"
        logger.info(f"  {status} {k}: {v}")

    iron_ok = (report["iron_rule_image_violations"] == 0 and
               report["iron_rule_narration_violations"] == 0)
    narr_ok = report["narration_image_ref_violations"] == 0
    files_ok = len(report["missing_image_files"]) == 0

    if iron_ok and narr_ok and files_ok:
        logger.info("\nTRIAL PASSED -- safe to run full production pipeline.")
        logger.info(f"  Run: game-predictor-run --llm-base-url {args.llm_base_url}")
    else:
        logger.error("\nTRIAL FAILED -- fix issues before production run.")
        if not iron_ok:
            logger.error("  Iron Rule violations detected (missing image or narration).")
        if not narr_ok:
            logger.error("  Narration independence violations detected.")
        if not files_ok:
            logger.error(f"  Missing image files: {report['missing_image_files']}")

    conn.close()

def cli_main():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
