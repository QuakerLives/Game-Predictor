"""
LangGraph agent definition for the Gemma 4 web scraping pipeline.
Orchestrates the ReAct loop: search -> extract -> download -> narrate -> assess -> write.
"""

import asyncio
import logging
import os
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

    async with LLM_SEM:
        is_valid = await validate_gameplay_image(img_path, name)
    if not is_valid:
        logger.warning(f"[{name}:{record_id}] Image rejected by Gemma 4 (ad/unrelated)")
        Path(img_path).unlink(missing_ok=True)
        return False

    async with BROWSER_SEM:
        try:
            article = await click_article_and_extract(result.source_page_url, name)
        except Exception as e:
            logger.warning(f"[{name}:{record_id}] Article extraction failed: {e}")
            article = ArticleData()

    body = article.body_text if article.body_text else f"A {name} gameplay session."

    async def _narrate():
        async with LLM_SEM:
            n = await generate_narration(body, name)
        if n and IMAGE_REF_PATTERN.search(n):
            async with LLM_SEM:
                n = await generate_narration(body, name)
        return n

    async def _assess():
        async with LLM_SEM:
            try:
                return await assess_experience_level(img_path, body, name)
            except Exception:
                return SENTINEL_QUAL

    narration, exp = await asyncio.gather(_narrate(), _assess())

    if narration is None or narration == SENTINEL_STR or IMAGE_REF_PATTERN.search(narration or ""):
        Path(img_path).unlink(missing_ok=True)
        return False

    record = GameplayRecord(
        video_game_name=name,
        image_path=img_path,
        player_name=article.author_name,
        gameplay_timestamp=article.publish_date,
        experience_level=exp,
        gameplay_level=article.gameplay_level,
        total_playtime=article.total_playtime,
        gameplay_narration=narration,
        channel_description=article.site_description,
        player_experience_narration=article.player_experience_summary,
        identifying_quotes=article.identifying_quotes,
        source_url=result.source_page_url,
        source_type="google_images",
    )
    await write_to_database(record, conn)
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

    async with LLM_SEM:
        is_valid = await validate_gameplay_image(img_path, name)
    if not is_valid:
        logger.warning(f"[{name}:{record_id}] YouTube screenshot rejected by Gemma 4 (ad/unrelated)")
        Path(img_path).unlink(missing_ok=True)
        return False

    async with BROWSER_SEM:
        try:
            channel = await extract_youtube_channel_info(video.get("channel_url", ""))
        except Exception:
            channel = {}

    yt_context = video.get("description", "") or video.get("title", f"A {name} video.")

    async def _narrate():
        async with LLM_SEM:
            n = await generate_narration(yt_context, name)
        if n and IMAGE_REF_PATTERN.search(n):
            async with LLM_SEM:
                n = await generate_narration(yt_context, name)
        return n

    async def _assess():
        async with LLM_SEM:
            try:
                return await assess_experience_level(img_path, yt_context, name)
            except Exception:
                return SENTINEL_QUAL

    narration, exp = await asyncio.gather(_narrate(), _assess())

    if narration is None or narration == SENTINEL_STR or IMAGE_REF_PATTERN.search(narration or ""):
        Path(img_path).unlink(missing_ok=True)
        return False

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
        identifying_quotes=SENTINEL_QUOTES,
        source_url=video["url"],
        source_type="youtube",
    )
    await write_to_database(record, conn)
    return True


async def process_game(game: dict, conn):
    """
    Collect target records for one game using parallelized candidate processing.
    Google Images first (target 140), then YouTube (target 60), then supplementary.
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

    # Phase 1: Google Images
    logger.info(f"[{name}] Phase 1: Google Images (target ~140)")
    img_candidates = []
    for query in game["queries_img"]:
        if shutdown_requested:
            break
        results = await search_google_images(query, num_results=20)
        img_candidates.extend(results)
        await asyncio.sleep(3)

    seen_urls = set()
    unique_img = []
    for r in img_candidates:
        if r.source_page_url not in seen_urls:
            seen_urls.add(r.source_page_url)
            unique_img.append(r)
    logger.info(f"[{name}] {len(unique_img)} unique Google Images candidates")

    google_target = min(140, target)
    for batch_start in range(0, len(unique_img), 8):
        if current_count() >= google_target or shutdown_requested:
            break
        batch = unique_img[batch_start:batch_start + 8]
        tasks = []
        for r in batch:
            rid = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
            tasks.append(process_google_candidate(r, game, conn, rid))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for r in results if r is True)
        logger.info(f"[{name}] Batch done: {ok}/{len(batch)} succeeded | Total: {current_count()}/{target}")

    # Phase 2: YouTube
    remaining = target - current_count()
    if remaining > 0 and not shutdown_requested:
        logger.info(f"[{name}] Phase 2: YouTube (target ~{remaining})")
        yt_candidates = []
        for query in game["queries_yt"]:
            if shutdown_requested:
                break
            results = await search_youtube(query, num_results=15)
            yt_candidates.extend(results)
            await asyncio.sleep(3)

        seen_yt = set()
        unique_yt = []
        for v in yt_candidates:
            if v["url"] not in seen_yt:
                seen_yt.add(v["url"])
                unique_yt.append(v)
        logger.info(f"[{name}] {len(unique_yt)} unique YouTube candidates")

        for batch_start in range(0, len(unique_yt), 5):
            if current_count() >= target or shutdown_requested:
                break
            batch = unique_yt[batch_start:batch_start + 5]
            tasks = []
            for v in batch:
                rid = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
                tasks.append(process_youtube_candidate(v, game, conn, rid))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            ok = sum(1 for r in results if r is True)
            logger.info(f"[{name}] YT batch: {ok}/{len(batch)} | Total: {current_count()}/{target}")

    # Phase 3: Supplementary
    remaining = target - current_count()
    if remaining > 0 and not shutdown_requested:
        logger.warning(f"[{name}] Short by {remaining} records -- running supplementary searches")
        supplementary_queries = [
            f"{name} review gameplay",
            f"{name} guide walkthrough",
            f"{name} tips progression",
            f"{name} site:reddit.com screenshot",
        ]
        for sq in supplementary_queries:
            if current_count() >= target or shutdown_requested:
                break
            try:
                extra = await search_google_images(sq, num_results=20)
                for r in extra:
                    if current_count() >= target or shutdown_requested:
                        break
                    if r.source_page_url in seen_urls:
                        continue
                    seen_urls.add(r.source_page_url)
                    rid = conn.execute("SELECT nextval('gameplay_id_seq')").fetchone()[0]
                    await process_google_candidate(r, game, conn, rid)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"[{name}] Supplementary search failed: {e}")

    final = current_count()
    logger.info(f"[{name}] === FINAL: {final}/{target} records ===")
    return final
