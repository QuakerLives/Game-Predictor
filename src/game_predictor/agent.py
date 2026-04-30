"""
LangGraph agent definition for the Gemma 4 web scraping pipeline.
Orchestrates the ReAct loop: search -> extract -> download -> narrate -> assess -> write.
"""

import asyncio
import logging
import os
import random
from asyncio import Semaphore
from datetime import datetime
from pathlib import Path

from game_predictor.config import (
    BROWSER_CONCURRENCY, DOWNLOAD_CONCURRENCY, LLM_CONCURRENCY,
    IMAGE_REF_PATTERN, SENTINEL_INT, SENTINEL_QUAL, SENTINEL_QUOTES,
    SENTINEL_STR, SENTINEL_TIMESTAMP, IMAGE_DIR,
)
from game_predictor.models import ArticleData, GameplayRecord, ImageResult
from game_predictor.tools.search import search_google_images, search_youtube
from game_predictor.tools.extract import click_article_and_extract
from game_predictor.tools.screenshot import screenshot_youtube_video, extract_youtube_channel_info
from game_predictor.tools.download import download_image
from game_predictor.tools.narrate import generate_narration
from game_predictor.tools.assess import assess_experience_level, validate_gameplay_image
from game_predictor.tools.database import write_to_database

logger = logging.getLogger(__name__)

BROWSER_SEM = Semaphore(BROWSER_CONCURRENCY)
LLM_SEM = Semaphore(LLM_CONCURRENCY)
DOWNLOAD_SEM = Semaphore(DOWNLOAD_CONCURRENCY)

shutdown_requested = False


async def process_google_candidate(
    result: ImageResult, game: dict, conn, record_id: int
) -> bool:
    slug = game["slug"]
    name = game["name"]

    async with DOWNLOAD_SEM:
        img_path = await download_image(result.image_url, slug, record_id)
    if img_path is None:
        return False

    logger.info(f"[{name}:{record_id}] Download done — waiting for LLM_SEM (validation)")
    async with LLM_SEM:
        logger.info(f"[{name}:{record_id}] LLM_SEM acquired — starting validation")
        try:
            is_valid = await validate_gameplay_image(img_path, name)
        except Exception:
            logger.warning(f"[{name}:{record_id}] Validation failed — skipping")
            Path(img_path).unlink(missing_ok=True)
            return False
    if not is_valid:
        logger.warning(f"[{name}:{record_id}] Image rejected by LLM (ad/unrelated)")
        Path(img_path).unlink(missing_ok=True)
        return False

    logger.info(f"[{name}:{record_id}] Validation PASSED — queueing narration+assessment")

    body = result.title.strip() if result.title.strip() else f"A {name} gameplay session."

    async def _narrate():
        async with LLM_SEM:
            try:
                n = await generate_narration(body, name, image_path=img_path)
            except Exception:
                logger.warning(f"[{name}:{record_id}] _narrate: generation failed")
                return None
        if n and IMAGE_REF_PATTERN.search(n):
            async with LLM_SEM:
                try:
                    n = await generate_narration(body, name, image_path=img_path)
                except Exception:
                    return None
        return n

    async def _assess():
        async with LLM_SEM:
            try:
                return await assess_experience_level(img_path, body, name)
            except Exception:
                logger.warning(f"[{name}:{record_id}] _assess: failed")
                return SENTINEL_QUAL

    narration, exp = await asyncio.gather(_narrate(), _assess())

    if not narration or narration == SENTINEL_STR or IMAGE_REF_PATTERN.search(narration or ""):
        narration = (
            f"The player engaged in a {name} session, working through game mechanics "
            f"and pursuing progression goals."
        )

    record = GameplayRecord(
        video_game_name=name,
        image_path=img_path,
        player_name=SENTINEL_STR,
        gameplay_timestamp=SENTINEL_TIMESTAMP,
        experience_level=exp,
        gameplay_level=SENTINEL_INT,
        total_playtime=SENTINEL_INT,
        gameplay_narration=narration,
        channel_description=SENTINEL_STR,
        player_experience_narration=SENTINEL_STR,
        identifying_quotes=list(SENTINEL_QUOTES),
        source_url=result.source_page_url,
        source_type="google_images",
    )
    await write_to_database(record, conn)
    logger.info(f"[{name}:{record_id}] Record written to database ✓")
    return True


async def process_youtube_candidate(
    video: dict, game: dict, conn, record_id: int
) -> bool:
    slug = game["slug"]
    name = game["name"]

    async with BROWSER_SEM:
        img_path = await screenshot_youtube_video(video["url"], slug, record_id)
    if img_path is None:
        return False

    logger.info(f"[{name}:{record_id}] YT screenshot done — waiting for LLM_SEM (validation)")
    async with LLM_SEM:
        logger.info(f"[{name}:{record_id}] LLM_SEM acquired — starting YT validation")
        try:
            is_valid = await validate_gameplay_image(img_path, name)
        except Exception:
            logger.warning(f"[{name}:{record_id}] YT validation failed — skipping")
            Path(img_path).unlink(missing_ok=True)
            return False
    if not is_valid:
        logger.warning(f"[{name}:{record_id}] YouTube screenshot rejected (ad/unrelated)")
        Path(img_path).unlink(missing_ok=True)
        return False

    logger.info(f"[{name}:{record_id}] YT validation PASSED — fetching channel info")
    async with BROWSER_SEM:
        try:
            channel = await extract_youtube_channel_info(video.get("channel_url", ""))
        except Exception:
            channel = {}

    yt_context = video.get("description", "") or video.get("title", f"A {name} video.")

    async def _narrate():
        logger.info(f"[{name}:{record_id}] _narrate(YT): waiting for LLM_SEM")
        async with LLM_SEM:
            logger.info(f"[{name}:{record_id}] _narrate(YT): LLM_SEM acquired — calling Ollama")
            try:
                n = await generate_narration(yt_context, name, image_path=img_path)
            except Exception:
                logger.warning(f"[{name}:{record_id}] _narrate(YT): generation failed")
                return None
            finally:
                logger.info(f"[{name}:{record_id}] _narrate(YT): releasing LLM_SEM")
        if n and IMAGE_REF_PATTERN.search(n):
            logger.warning(f"[{name}:{record_id}] _narrate(YT): independence check failed — retrying")
            async with LLM_SEM:
                try:
                    n = await generate_narration(yt_context, name, image_path=img_path)
                except Exception:
                    return None
        return n

    async def _assess():
        logger.info(f"[{name}:{record_id}] _assess(YT): waiting for LLM_SEM")
        async with LLM_SEM:
            logger.info(f"[{name}:{record_id}] _assess(YT): LLM_SEM acquired — calling Ollama")
            try:
                return await assess_experience_level(img_path, yt_context, name)
            except Exception:
                logger.warning(f"[{name}:{record_id}] _assess(YT): failed")
                return SENTINEL_QUAL
            finally:
                logger.info(f"[{name}:{record_id}] _assess(YT): releasing LLM_SEM")

    logger.info(f"[{name}:{record_id}] Starting asyncio.gather (YT narration + assessment)")
    narration, exp = await asyncio.gather(_narrate(), _assess())
    logger.info(f"[{name}:{record_id}] YT narration+assessment complete")

    if not narration or narration == SENTINEL_STR or IMAGE_REF_PATTERN.search(narration or ""):
        narration = (
            f"The player engaged in a {name} session, working through game mechanics "
            f"and pursuing progression goals."
        )
        logger.warning(f"[{name}:{record_id}] Using fallback narration (YT)")

    record = GameplayRecord(
        video_game_name=name,
        image_path=img_path,
        player_name=video.get("channel_name", SENTINEL_STR),
        gameplay_timestamp=video.get("upload_date", SENTINEL_TIMESTAMP),
        experience_level=exp,
        gameplay_level=SENTINEL_INT,
        total_playtime=SENTINEL_INT,
        gameplay_narration=narration,
        channel_description=channel.get("description", SENTINEL_STR),
        player_experience_narration=channel.get("description", SENTINEL_STR),
        identifying_quotes=list(SENTINEL_QUOTES),
        source_url=video["url"],
        source_type="youtube",
    )
    await write_to_database(record, conn)
    logger.info(f"[{name}:{record_id}] YT record written to database ✓")
    return True


async def _run_search(coro, name: str, query: str, source: str) -> list:
    """Run a search coroutine with a 90s timeout; return results or [] on failure."""
    try:
        return await asyncio.wait_for(coro, timeout=150.0)
    except asyncio.TimeoutError:
        logger.warning(f"[{name}] {source} timed out (150s): '{query}' — skipping")
    except Exception as e:
        logger.warning(f"[{name}] {source} failed for '{query}': {e}")
    return []


async def _process_batch(batch, processor_fn, name: str, target: int, current_count_fn):
    """Run a batch of candidate coroutines concurrently and log results."""
    tasks = [processor_fn(item) for item in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if r is True)
    logger.info(
        f"[{name}] Batch done: {ok}/{len(batch)} succeeded | Total: {current_count_fn()}/{target}"
    )
    return ok


async def process_game(game: dict, conn):
    """
    Collect target records for one game.
    Processes each query's results immediately so records flow even if later
    searches hang or fail. Google/Bing first, then YouTube, then supplementary.
    """
    global shutdown_requested
    name = game["name"]
    slug = game["slug"]
    target = game["target"]

    skip_file = Path(f".skip_{slug}")
    if skip_file.exists():
        logger.warning(f"[{name}] SKIPPED by sleeper ({skip_file.read_text().strip()})")
        return 0

    def current_count() -> int:
        return conn.execute(
            "SELECT COUNT(*) FROM gameplay_records WHERE video_game_name = ?", [name]
        ).fetchone()[0]

    seen_urls: set[str] = set(
        row[0] for row in conn.execute(
            "SELECT source_url FROM gameplay_records WHERE video_game_name = ? AND source_type = 'google_images'",
            [name],
        ).fetchall()
    )
    logger.info(f"[{name}] Pre-seeded {len(seen_urls)} already-collected image URLs")
    google_target = min(1400, target)

    # Phase 1: Google/Bing Images — page through each query until we have enough
    # Each page is offset by 35 results (Bing &first= pagination).
    # We stop paging a query when it returns 0 new candidates (results exhausted).
    logger.info(f"[{name}] Phase 1: Image search (target ~{google_target})")
    for query in game["queries_img"]:
        if current_count() >= google_target or shutdown_requested:
            break

        for page in range(8):  # up to 8 pages (≈280 candidates) per query
            if current_count() >= google_target or shutdown_requested:
                break

            raw = await _run_search(
                search_google_images(query, num_results=40, page_offset=page * 35),
                name, query, "Image search"
            )
            await asyncio.sleep(random.uniform(8, 15))

            new_candidates = [r for r in raw if r.source_page_url not in seen_urls]
            for r in new_candidates:
                seen_urls.add(r.source_page_url)

            logger.info(
                f"[{name}] '{query}' p{page + 1}: {len(new_candidates)} new candidates"
            )

            if not new_candidates:
                break  # Bing has no more unique results for this query at this depth

            for batch_start in range(0, len(new_candidates), 8):
                if current_count() >= google_target or shutdown_requested:
                    break
                batch = new_candidates[batch_start:batch_start + 8]

                def make_google_task(r):
                    rid = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
                    return process_google_candidate(r, game, conn, rid)

                await _process_batch(
                    batch, make_google_task, name, target, current_count
                )

    # Phase 2: YouTube — process each query's results immediately
    remaining = target - current_count()
    if remaining > 0 and not shutdown_requested:
        logger.info(f"[{name}] Phase 2: YouTube (need ~{remaining} more)")
        seen_yt: set[str] = set(
            row[0] for row in conn.execute(
                "SELECT source_url FROM gameplay_records WHERE video_game_name = ? AND source_type = 'youtube'",
                [name],
            ).fetchall()
        )
        logger.info(f"[{name}] Pre-seeded {len(seen_yt)} already-collected YouTube URLs")

        for query in game["queries_yt"]:
            if current_count() >= target or shutdown_requested:
                break

            raw_yt = await _run_search(
                search_youtube(query, num_results=15), name, query, "YouTube search"
            )
            await asyncio.sleep(random.uniform(5, 10))

            new_yt = [v for v in raw_yt if v["url"] not in seen_yt]
            for v in new_yt:
                seen_yt.add(v["url"])

            logger.info(f"[{name}] YouTube '{query}': {len(new_yt)} new candidates")

            for batch_start in range(0, len(new_yt), 5):
                if current_count() >= target or shutdown_requested:
                    break
                batch = new_yt[batch_start:batch_start + 5]

                def make_yt_task(v):
                    rid = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
                    return process_youtube_candidate(v, game, conn, rid)

                await _process_batch(
                    batch, make_yt_task, name, target, current_count
                )

    # Phase 3: Supplementary image searches if still short
    remaining = target - current_count()
    if remaining > 0 and not shutdown_requested:
        logger.warning(f"[{name}] Short by {remaining} — running supplementary searches")
        supplementary_queries = [
            f"{name} review gameplay",
            f"{name} guide walkthrough",
            f"{name} tips progression",
            f"{name} site:reddit.com screenshot",
        ]
        for sq in supplementary_queries:
            if current_count() >= target or shutdown_requested:
                break

            extra = await _run_search(
                search_google_images(sq, num_results=40), name, sq, "Supplementary"
            )
            await asyncio.sleep(5)

            new_extra = [r for r in extra if r.source_page_url not in seen_urls]
            for r in new_extra:
                seen_urls.add(r.source_page_url)

            for r in new_extra:
                if current_count() >= target or shutdown_requested:
                    break
                rid = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
                await process_google_candidate(r, game, conn, rid)

    final = current_count()
    logger.info(f"[{name}] === FINAL: {final}/{target} records ===")
    return final
