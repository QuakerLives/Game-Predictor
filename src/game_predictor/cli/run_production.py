#!/usr/bin/env python3
"""
PRODUCTION RUN -- 200 records per game, 1000 total.
9-hour execution window with quality enrichment pass.

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
    GAMES, SENTINEL_STR, IMAGE_REF_PATTERN, DB_PATH, IMAGE_DIR,
    LLM_CONCURRENCY, BROWSER_CONCURRENCY,
)
from game_predictor.tools.database import init_db, validate_database
from game_predictor.tools.extract import click_article_and_extract
from game_predictor.tools.narrate import generate_narration
from game_predictor.agent import process_game
from game_predictor import agent as agent_module

PROD_LOG_FILE = "production_run.log"
DEADLINE_HOURS = 9
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


async def heartbeat_loop():
    while not shutdown_requested:
        try:
            conn_hb = duckdb.connect(DB_PATH, read_only=True)
            total = conn_hb.execute("SELECT COUNT(*) FROM gameplay_records").fetchone()[0]
            per_game = dict(conn_hb.execute(
                "SELECT video_game_name, COUNT(*) FROM gameplay_records GROUP BY video_game_name"
            ).fetchall())
            conn_hb.close()
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


async def enrichment_pass(conn, deadline: datetime):
    from asyncio import Semaphore
    LLM_SEM = Semaphore(LLM_CONCURRENCY)
    BROWSER_SEM = Semaphore(BROWSER_CONCURRENCY)

    logger.info("Starting quality enrichment pass...")
    cutoff = deadline - timedelta(minutes=15)

    if datetime.now() < cutoff:
        sentinels = conn.execute("""
            SELECT id, source_url, source_type FROM gameplay_records
            WHERE player_name = 'N/A' LIMIT 50
        """).fetchall()
        logger.info(f"Enrichment: {len(sentinels)} records with sentinel player_name")
        for rid, url, stype in sentinels:
            if datetime.now() >= cutoff or shutdown_requested:
                break
            try:
                async with BROWSER_SEM:
                    article = await click_article_and_extract(url, "")
                if article.author_name != SENTINEL_STR:
                    conn.execute(
                        "UPDATE gameplay_records SET player_name = ? WHERE id = ?",
                        [article.author_name, rid]
                    )
                    logger.info(f"Enrichment: Updated player_name for record {rid}")
            except Exception:
                pass

    if datetime.now() < cutoff:
        violations = conn.execute("""
            SELECT id, gameplay_narration FROM gameplay_records
            WHERE gameplay_narration ILIKE '%screenshot%'
               OR gameplay_narration ILIKE '%image%'
               OR gameplay_narration ILIKE '%as shown%'
               OR gameplay_narration ILIKE '%depicted%'
               OR gameplay_narration ILIKE '%visible%'
        """).fetchall()
        logger.info(f"Enrichment: {len(violations)} narration independence violations")
        for rid, narr in violations:
            if datetime.now() >= cutoff or shutdown_requested:
                break
            try:
                game_name = conn.execute(
                    "SELECT video_game_name FROM gameplay_records WHERE id = ?", [rid]
                ).fetchone()[0]
                async with LLM_SEM:
                    new_narr = await generate_narration(narr, game_name)
                if new_narr and not IMAGE_REF_PATTERN.search(new_narr):
                    conn.execute(
                        "UPDATE gameplay_records SET gameplay_narration = ? WHERE id = ?",
                        [new_narr, rid]
                    )
                    logger.info(f"Enrichment: Fixed narration for record {rid}")
            except Exception:
                pass

    if datetime.now() < cutoff:
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
        logger.info(f"Enrichment: {len(bad)} corrupt/undersized images found")
        if bad:
            logger.warning(f"Enrichment: Image issues in records: {bad}")

    logger.info("Quality enrichment pass complete.")


async def main():
    global shutdown_requested

    parser = argparse.ArgumentParser(description="Production run: 1000 records")
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing DB, skipping games already at target")
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip the quality enrichment pass")
    args = parser.parse_args()

    start = datetime.now()
    deadline = start + timedelta(hours=DEADLINE_HOURS)

    logger.info("=" * 70)
    logger.info("PRODUCTION RUN -- 200 records x 5 games = 1000 total")
    logger.info(f"Started:  {start.isoformat()}")
    logger.info(f"Deadline: {deadline.isoformat()} ({DEADLINE_HOURS}h window)")
    logger.info(f"LLM:      {args.llm_base_url}")
    logger.info(f"Resume:   {args.resume}")
    logger.info("=" * 70)

    conn = init_db()
    hb_task = asyncio.create_task(heartbeat_loop())

    for game in GAMES:
        if shutdown_requested:
            logger.warning("Shutdown requested -- stopping before next game.")
            break
        if datetime.now() >= deadline - timedelta(minutes=30):
            logger.warning("Approaching deadline -- stopping game processing.")
            break

        existing = conn.execute(
            "SELECT COUNT(*) FROM gameplay_records WHERE video_game_name = ?",
            [game["name"]]
        ).fetchone()[0]
        if args.resume and existing >= game["target"]:
            logger.info(f"[{game['name']}] Already at {existing}/{game['target']} -- skipping.")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"  GAME: {game['name']} (existing: {existing}, target: {game['target']})")
        logger.info(f"{'='*50}")
        try:
            await process_game(game, conn)
        except Exception as e:
            logger.error(f"[{game['name']}] FATAL: {e}", exc_info=True)
            continue

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
    logger.info(f"Total records: {report['total_records']}/1000")
    logger.info(f"\nValidation report:")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    if report["target_met"]:
        logger.info("\nPRODUCTION TARGET MET (1000 records)")
    else:
        logger.error(f"\nTARGET NOT MET: {report['total_records']}/1000")
        shortfall = {g: game["target"] - report["per_game"].get(g, 0)
                     for game in GAMES for g in [game["name"]]
                     if report["per_game"].get(g, 0) < game["target"]}
        if shortfall:
            logger.error(f"  Shortfall by game: {shortfall}")
            logger.error(f"  Re-run with: python run_production.py --resume")

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
    conn.close()

def cli_main():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
