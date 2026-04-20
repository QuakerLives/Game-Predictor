#!/usr/bin/env python3
"""
PRODUCTION RUN -- 2000 records per game, 10000 total.
12-hour execution window with quality enrichment pass.

Usage:
    python run_production.py [--llm-base-url http://localhost:11434/v1]
                             [--resume]
                             [--skip-enrichment]
"""

import asyncio
import argparse
import logging
import signal
import sys
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
from PIL import Image

from game_predictor.config import (
    GAMES, SENTINEL_STR, SENTINEL_INT, SENTINEL_TIMESTAMP, SENTINEL_QUAL,
    SENTINEL_QUOTES, IMAGE_REF_PATTERN, DB_PATH, IMAGE_DIR,
    LLM_CONCURRENCY, BROWSER_CONCURRENCY, LLM_MODEL, ENRICHMENT_LLM_MODEL,
    ENRICHMENT_BATCH_SIZE, ENRICHMENT_RECORD_TIMEOUT,
    ENRICHMENT_BROWSER_TIMEOUT, ENRICHMENT_LLM_TIMEOUT,
)
from game_predictor.tools.database import (
    init_db, validate_database, migrate_schema_for_enrichment,
    update_record_fields,
)
from game_predictor.tools.extract import click_article_and_extract
from game_predictor.tools.narrate import generate_narration
from game_predictor.tools.screenshot import (
    extract_youtube_channel_info,
    extract_channel_url_from_video,
    extract_youtube_video_metadata,
)
from game_predictor.tools.assess import assess_experience_level
from game_predictor.agent import process_game
from game_predictor import agent as agent_module

PROD_LOG_FILE = "production_run.log"
DEADLINE_HOURS = 12
HEARTBEAT_FILE = Path("heartbeat.json")
HEARTBEAT_INTERVAL = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(PROD_LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("production")

shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    logger.warning(f"Received signal {signum} -- completing current records then shutting down.")
    shutdown_requested = True
    agent_module.shutdown_requested = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


async def heartbeat_loop(conn):
    while not shutdown_requested:
        try:
            total = conn.execute("SELECT COUNT(*) FROM gameplay_records").fetchone()[0]
            per_game = dict(conn.execute(
                "SELECT video_game_name, COUNT(*) FROM gameplay_records GROUP BY video_game_name"
            ).fetchall())
        except Exception:
            total = -1
            per_game = {}

        heartbeat = {
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "total_records": total,
            "per_game": per_game,
            "shutdown_requested": shutdown_requested,
            "status": "running",
        }
        HEARTBEAT_FILE.write_text(json.dumps(heartbeat))
        await asyncio.sleep(HEARTBEAT_INTERVAL)

    HEARTBEAT_FILE.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid(),
        "total_records": -1,
        "per_game": {},
        "shutdown_requested": True,
        "status": "stopped",
    }))


def _is_youtube_url(url: str) -> bool:
    """Check if a URL points to YouTube regardless of source_type."""
    return "youtube.com/" in url or "youtu.be/" in url


_EXP_ENUM = ("Poor", "Fair", "Good", "Excellent", "Superior")


async def _assess_experience(
    record_id: int,
    image_path: str,
    game_name: str,
    context_text: str = "",
) -> str | None:
    """Run experience assessment with timeout. Returns valid enum value or None."""
    if not image_path or not Path(image_path).exists():
        return None
    try:
        exp = await asyncio.wait_for(
            assess_experience_level(image_path, context_text, game_name),
            timeout=ENRICHMENT_LLM_TIMEOUT,
        )
        if exp and exp in _EXP_ENUM:
            return exp
    except asyncio.TimeoutError:
        logger.warning(f"[enrich:{record_id}] Experience assessment timed out")
    except Exception as e:
        logger.warning(f"[enrich:{record_id}] Experience assessment failed: {e}")
    return None


async def _article_phase(
    record_id: int,
    source_url: str,
    game_name: str,
):
    """Phase 1 for articles: extract structured data. Returns ArticleData or None."""
    try:
        return await asyncio.wait_for(
            click_article_and_extract(source_url, game_name),
            timeout=ENRICHMENT_BROWSER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[enrich:{record_id}] Article extraction timed out")
    except Exception as e:
        logger.warning(f"[enrich:{record_id}] Article extraction failed: {e}")
    return None


async def _youtube_phase(
    record_id: int,
    source_url: str,
    current: dict,
) -> dict[str, object]:
    """Phase 1 for YouTube: extract video metadata + channel description."""
    updates: dict[str, object] = {}

    # Sub-phase 1a: video metadata (channel name + upload date)
    if current.get("player_name") is None or current.get("gameplay_timestamp") is None:
        try:
            meta = await asyncio.wait_for(
                extract_youtube_video_metadata(source_url),
                timeout=ENRICHMENT_BROWSER_TIMEOUT,
            )
            if current.get("player_name") is None and meta.get("channel_name"):
                updates["player_name"] = meta["channel_name"]
            if current.get("gameplay_timestamp") is None and meta.get("upload_date"):
                from dateutil import parser as dateparser
                try:
                    updates["gameplay_timestamp"] = dateparser.parse(meta["upload_date"])
                except Exception:
                    pass
        except asyncio.TimeoutError:
            logger.warning(f"[enrich:{record_id}] YouTube metadata extraction timed out")
        except Exception as e:
            logger.warning(f"[enrich:{record_id}] YouTube metadata extraction failed: {e}")

    # Sub-phase 1b: channel description
    if (current.get("channel_description") is None
            or current.get("player_experience_narration") is None):
        try:
            channel_url = await asyncio.wait_for(
                extract_channel_url_from_video(source_url),
                timeout=ENRICHMENT_BROWSER_TIMEOUT,
            )
            if channel_url:
                channel_info = await asyncio.wait_for(
                    extract_youtube_channel_info(channel_url),
                    timeout=ENRICHMENT_BROWSER_TIMEOUT,
                )
                desc = channel_info.get("description")
                if desc and desc != SENTINEL_STR:
                    if current.get("channel_description") is None:
                        updates["channel_description"] = desc
                    if current.get("player_experience_narration") is None:
                        updates["player_experience_narration"] = desc
        except asyncio.TimeoutError:
            logger.warning(f"[enrich:{record_id}] YouTube channel extraction timed out")
        except Exception as e:
            logger.warning(f"[enrich:{record_id}] YouTube channel extraction failed: {e}")

    return updates


async def _enrich_article_record(
    record_id: int,
    source_url: str,
    game_name: str,
    image_path: str,
    current: dict,
) -> dict[str, object]:
    """Enrich a non-YouTube record.

    Runs Phase 1 (article extraction) and Phase 2 (image assessment) concurrently
    so experience_level is always attempted even if article extraction times out.
    Skips Phase 1 entirely if all article-derivable fields are already populated.
    """
    updates: dict[str, object] = {}

    article_fields = (
        "player_name", "gameplay_timestamp", "identifying_quotes",
        "channel_description", "player_experience_narration",
        "gameplay_level", "total_playtime",
    )
    needs_article = any(current.get(f) is None for f in article_fields)
    needs_experience = current.get("experience_level") is None

    tasks = []
    if needs_article:
        tasks.append(("article", _article_phase(record_id, source_url, game_name)))
    if needs_experience:
        tasks.append(("experience", _assess_experience(record_id, image_path, game_name)))

    if not tasks:
        return updates  # nothing to do

    results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
    named = {name: res for (name, _), res in zip(tasks, results)}

    # Apply experience assessment result
    exp_result = named.get("experience")
    if isinstance(exp_result, str):
        updates["experience_level"] = exp_result

    # Apply article extraction result
    article = named.get("article")
    if article is not None and not isinstance(article, Exception):
        if current.get("player_name") is None and article.author_name != SENTINEL_STR:
            updates["player_name"] = article.author_name
        if current.get("gameplay_timestamp") is None and article.publish_date != SENTINEL_TIMESTAMP:
            updates["gameplay_timestamp"] = article.publish_date
        if current.get("identifying_quotes") is None and article.identifying_quotes != SENTINEL_QUOTES:
            updates["identifying_quotes"] = article.identifying_quotes
        if current.get("channel_description") is None and article.site_description != SENTINEL_STR:
            updates["channel_description"] = article.site_description
        if current.get("player_experience_narration") is None and article.player_experience_summary != SENTINEL_STR:
            updates["player_experience_narration"] = article.player_experience_summary
        if current.get("gameplay_level") is None and article.gameplay_level != SENTINEL_INT:
            updates["gameplay_level"] = article.gameplay_level
        if current.get("total_playtime") is None and article.total_playtime != SENTINEL_INT:
            updates["total_playtime"] = article.total_playtime

    return updates


async def _enrich_youtube_record(
    record_id: int,
    source_url: str,
    game_name: str,
    image_path: str,
    current: dict,
) -> dict[str, object]:
    """Enrich a YouTube record.

    Runs YouTube browser phase and image assessment concurrently so
    experience_level is assessed even if browser work times out.
    Skips browser phase entirely if all its target fields are populated.
    """
    updates: dict[str, object] = {}

    yt_fields = (
        "player_name", "gameplay_timestamp",
        "channel_description", "player_experience_narration",
    )
    needs_yt = any(current.get(f) is None for f in yt_fields)
    needs_experience = current.get("experience_level") is None

    tasks = []
    if needs_yt:
        tasks.append(("youtube", _youtube_phase(record_id, source_url, current)))
    if needs_experience:
        tasks.append(("experience", _assess_experience(record_id, image_path, game_name)))

    if not tasks:
        return updates

    results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
    named = {name: res for (name, _), res in zip(tasks, results)}

    exp_result = named.get("experience")
    if isinstance(exp_result, str):
        updates["experience_level"] = exp_result

    yt_result = named.get("youtube")
    if isinstance(yt_result, dict):
        updates.update(yt_result)

    return updates


async def enrichment_pass(conn, deadline: datetime):
    """Full enrichment pass: re-extract sentinel-contaminated fields from
    source URLs. After schema migration, sentinels become NULLs, and this
    function attempts to fill them with real data. On failure, fields stay NULL.

    Records are processed sequentially (one at a time) to avoid browser
    resource contention — each gets the full CPU/memory/network bandwidth.
    """
    logger.info("=" * 50)
    logger.info("  ENRICHMENT PASS")
    logger.info("=" * 50)

    cutoff = deadline - timedelta(minutes=15)

    # Suppress noisy Playwright TargetClosedError futures
    _original_handler = asyncio.get_event_loop().get_exception_handler()

    def _suppress_target_closed(loop, context):
        exc = context.get("exception")
        if exc and "TargetClosedError" in type(exc).__name__:
            return  # silently discard
        if _original_handler:
            _original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    asyncio.get_event_loop().set_exception_handler(_suppress_target_closed)

    # Run schema migration (idempotent)
    migration_counts = migrate_schema_for_enrichment(conn)
    if migration_counts:
        logger.info(f"Migration converted sentinels to NULL: {migration_counts}")

    # Query all records that have at least one NULL enrichable field
    contaminated = conn.execute("""
        SELECT id, source_url, source_type, video_game_name, image_path,
               player_name, gameplay_timestamp, experience_level,
               gameplay_level, total_playtime, channel_description,
               player_experience_narration, identifying_quotes
        FROM gameplay_records
        WHERE player_name IS NULL
           OR gameplay_timestamp IS NULL
           OR experience_level IS NULL
           OR channel_description IS NULL
           OR player_experience_narration IS NULL
           OR identifying_quotes IS NULL
           OR gameplay_level IS NULL
           OR total_playtime IS NULL
        ORDER BY id
    """).fetchall()

    total_to_enrich = len(contaminated)
    logger.info(f"Found {total_to_enrich} records with NULL fields to enrich.")

    if not contaminated:
        logger.info("No contamination found. Enrichment pass complete.")
        return

    col_names = [
        "id", "source_url", "source_type", "video_game_name", "image_path",
        "player_name", "gameplay_timestamp", "experience_level",
        "gameplay_level", "total_playtime", "channel_description",
        "player_experience_narration", "identifying_quotes",
    ]

    enriched_count = 0
    fields_updated_total = 0

    # Process records one at a time — gives each the full browser/LLM bandwidth
    for idx, row in enumerate(contaminated):
        if datetime.now() > cutoff or shutdown_requested:
            logger.warning("Enrichment cutoff reached — stopping.")
            break

        row_dict = dict(zip(col_names, row))
        record_id = row_dict["id"]
        source_url = row_dict["source_url"]
        game_name = row_dict["video_game_name"]
        image_path = row_dict["image_path"]
        current = {k: row_dict[k] for k in col_names[5:]}

        # Route by URL content, not source_type — many YouTube URLs have
        # source_type="google_images" because they were found via Google.
        is_youtube = _is_youtube_url(source_url)

        try:
            if is_youtube:
                updates = await asyncio.wait_for(
                    _enrich_youtube_record(
                        record_id, source_url, game_name, image_path, current,
                    ),
                    timeout=ENRICHMENT_RECORD_TIMEOUT,
                )
            else:
                updates = await asyncio.wait_for(
                    _enrich_article_record(
                        record_id, source_url, game_name, image_path, current,
                    ),
                    timeout=ENRICHMENT_RECORD_TIMEOUT,
                )
        except asyncio.TimeoutError:
            logger.warning(
                f"[enrich:{record_id}] Overall timeout ({ENRICHMENT_RECORD_TIMEOUT}s)"
            )
            updates = {}
        except Exception as e:
            logger.error(f"[enrich:{record_id}] Unexpected error: {e}")
            updates = {}

        if updates:
            n = update_record_fields(conn, record_id, updates)
            fields_updated_total += n
            enriched_count += 1
            logger.info(
                f"[enrich:{record_id}] Updated {n} fields: {list(updates.keys())}"
            )

        # Heartbeat every record (sequential so no batching)
        processed = idx + 1
        progress = int((processed / total_to_enrich) * 100)
        if processed % 10 == 0 or updates:
            heartbeat = {
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid(),
                "total_records": conn.execute(
                    "SELECT COUNT(*) FROM gameplay_records"
                ).fetchone()[0],
                "per_game": {},
                "shutdown_requested": shutdown_requested,
                "status": "enriching",
                "enrichment_progress": progress,
                "enrichment_detail": {
                    "processed": processed,
                    "total": total_to_enrich,
                    "fields_updated_total": fields_updated_total,
                },
            }
            HEARTBEAT_FILE.write_text(json.dumps(heartbeat))
            logger.info(
                f"Enrichment: {processed}/{total_to_enrich} ({progress}%) "
                f"| {enriched_count} enriched | {fields_updated_total} fields updated"
            )

    # Narration re-generation: independence violations AND generic fallback strings
    # (fallback was written when Ollama was unavailable during collection)
    if datetime.now() < cutoff and not shutdown_requested:
        violations = conn.execute("""
            SELECT id, video_game_name, image_path,
                   COALESCE(player_experience_narration, channel_description, '') AS context
            FROM gameplay_records
            WHERE gameplay_narration ILIKE '%screenshot%'
               OR gameplay_narration ILIKE '%image%'
               OR gameplay_narration ILIKE '%as shown%'
               OR gameplay_narration ILIKE '%depicted%'
               OR gameplay_narration ILIKE '%visible%'
               OR gameplay_narration ILIKE '%working through game mechanics and pursuing progression goals%'
        """).fetchall()
        if violations:
            logger.info(f"Enrichment: {len(violations)} narration independence violations")
            for rid, narr in violations:
                if datetime.now() >= cutoff or shutdown_requested:
                    break
                try:
                    game_name = conn.execute(
                        "SELECT video_game_name FROM gameplay_records WHERE id = ?", [rid]
                    ).fetchone()[0]
                    new_narr = await generate_narration(narr, game_name)
                    if new_narr and not IMAGE_REF_PATTERN.search(new_narr):
                        conn.execute(
                            "UPDATE gameplay_records SET gameplay_narration = ? WHERE id = ?",
                            [new_narr, rid],
                        )
                        logger.info(f"Enrichment: regenerated narration for record {rid}")
                except Exception:
                    pass

            NARR_BATCH = LLM_CONCURRENCY * 2
            for batch_start in range(0, len(violations), NARR_BATCH):
                if datetime.now() >= cutoff or shutdown_requested:
                    break
                batch = violations[batch_start:batch_start + NARR_BATCH]
                await asyncio.gather(*[_regen(r[0], r[1], r[2], r[3]) for r in batch])
                pct = min(100, int((batch_start + len(batch)) / len(violations) * 100))
                logger.info(f"Narration regen: {batch_start + len(batch)}/{len(violations)} ({pct}%)")

    # Image integrity check
    if datetime.now() < cutoff and not shutdown_requested:
        rows = conn.execute("SELECT id, image_path FROM gameplay_records").fetchall()
        bad = []
        for rid, path in rows:
            try:
                img = Image.open(path)
                img.verify()
                if img.size[0] < 200 or img.size[1] < 200:
                    bad.append(rid)
            except Exception:
                bad.append(rid)
        if bad:
            logger.warning(f"Enrichment: {len(bad)} corrupt/undersized images: {bad}")

    # Restore original exception handler
    asyncio.get_event_loop().set_exception_handler(_original_handler)

    logger.info(
        f"Enrichment pass complete. "
        f"{enriched_count}/{total_to_enrich} records enriched, "
        f"{fields_updated_total} total field updates."
    )


async def main():
    global shutdown_requested

    parser = argparse.ArgumentParser(description="Production run: 10000 records")
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing DB, skipping games already at target")
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip the quality enrichment pass")
    parser.add_argument("--enrich-only", action="store_true",
                        help="Run only the enrichment pass (skip collection)")
    parser.add_argument("--max-duration", default="2h",
                        help="Max duration for enrichment-only mode (e.g. 2h, 1.5h)")
    args = parser.parse_args()

    # ── Enrichment-only mode ──────────────────────────────────────────────
    if args.enrich_only:
        try:
            hours_str = args.max_duration.replace("h", "")
            hours = float(hours_str)
        except ValueError:
            hours = 2.0
        deadline = datetime.now() + timedelta(hours=hours)

        logger.info("=" * 70)
        logger.info("ENRICHMENT-ONLY MODE")
        logger.info(f"Deadline: {deadline.isoformat()} ({hours}h window)")
        logger.info("=" * 70)

        conn = init_db()
        hb_task = asyncio.create_task(heartbeat_loop(conn))
        try:
            await enrichment_pass(conn, deadline)

            report = validate_database(conn)
            logger.info(f"Post-enrichment validation: {report['total_records']} records")
            if "null_density" in report:
                logger.info("NULL density after enrichment:")
                for col, info in report["null_density"].items():
                    logger.info(f"  {col}: {info['count']} ({info['pct']}%)")
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            HEARTBEAT_FILE.write_text(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid(),
                "total_records": -1,
                "per_game": {},
                "shutdown_requested": True,
                "status": "enrichment_complete",
            }))
            conn.close()
        return

    start = datetime.now()
    deadline = start + timedelta(hours=DEADLINE_HOURS)

    logger.info("=" * 70)
    logger.info("PRODUCTION RUN -- 2000 records x 5 games = 10000 total")
    logger.info(f"Started:  {start.isoformat()}")
    logger.info(f"Deadline: {deadline.isoformat()} ({DEADLINE_HOURS}h window)")
    logger.info(f"LLM:      {args.llm_base_url}")
    logger.info(f"Resume:   {args.resume}")
    logger.info("=" * 70)

    conn = init_db()
    hb_task = asyncio.create_task(heartbeat_loop(conn))

    async def _collect_game(game: dict, stagger_delay: float = 0.0) -> None:
        if stagger_delay:
            await asyncio.sleep(stagger_delay)
        if shutdown_requested:
            return
        if datetime.now() >= deadline - timedelta(minutes=30):
            logger.warning(f"[{game['name']}] Approaching deadline before start — skipping.")
            return
        existing = conn.execute(
            "SELECT COUNT(*) FROM gameplay_records WHERE video_game_name = ?",
            [game["name"]],
        ).fetchone()[0]
        if args.resume and existing >= game["target"]:
            logger.info(f"[{game['name']}] Already at {existing}/{game['target']} -- skipping.")
            return
        logger.info(f"\n{'='*50}")
        logger.info(f"  GAME: {game['name']} (existing: {existing}, target: {game['target']})")
        logger.info(f"{'='*50}")
        try:
            await process_game(game, conn)
        except Exception as e:
            logger.error(f"[{game['name']}] FATAL: {e}", exc_info=True)

    try:
        logger.info("Running all 5 games in parallel (staggered 10s apart)...")
        await asyncio.gather(*[
            _collect_game(game, stagger_delay=i * 10)
            for i, game in enumerate(GAMES)
        ])

        if not args.skip_enrichment and not shutdown_requested:
            time_remaining = deadline - datetime.now()
            if time_remaining > timedelta(minutes=30):
                logger.info(f"\n{time_remaining} remaining -- running enrichment pass...")
                await enrichment_pass(conn, deadline)
            else:
                logger.info(f"Only {time_remaining} remaining -- skipping enrichment.")

        logger.info(f"\n{'='*50}")
        logger.info("  FINAL VALIDATION")
        logger.info(f"{'='*50}")
        report = validate_database(conn)
        elapsed = datetime.now() - start

        logger.info(f"\nPipeline completed in {elapsed}")
        logger.info(f"Total records: {report['total_records']}/10000")
        logger.info(f"\nValidation report:")
        for k, v in report.items():
            logger.info(f"  {k}: {v}")

        if report["target_met"]:
            logger.info("\nPRODUCTION TARGET MET (10000 records)")
        else:
            logger.error(f"\nTARGET NOT MET: {report['total_records']}/10000")
            shortfall = {g: game["target"] - report["per_game"].get(g, 0)
                         for game in GAMES for g in [game["name"]]
                         if report["per_game"].get(g, 0) < game["target"]}
            if shortfall:
                logger.error(f"  Shortfall by game: {shortfall}")
                logger.error(f"  Re-run with: game-predictor-run --resume")

        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass
        HEARTBEAT_FILE.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "total_records": report.get("total_records", -1),
            "per_game": report.get("per_game", {}),
            "shutdown_requested": True,
            "status": "completed" if report.get("target_met") else "incomplete",
        }))
    finally:
        conn.close()

def cli_main():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


def enrich_cli_main():
    """Sync entry point for enrichment-only mode with Ollama preflight check."""
    import httpx

    # Preflight: verify Ollama is reachable before doing any work.
    try:
        r = httpx.get("http://localhost:11434/v1/models", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Ollama is not reachable at http://localhost:11434 ({e})")
        print("Start Ollama with: ollama serve")
        print("Then warm up the model: ollama run gemma4:26b --keepalive 9h")
        sys.exit(1)

    # Force --enrich-only into argv if not already there.
    if "--enrich-only" not in sys.argv:
        sys.argv.insert(1, "--enrich-only")
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
