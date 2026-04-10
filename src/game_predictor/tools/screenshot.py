"""
Playwright-based YouTube screenshot capture and channel info extraction.
See design doc §9 for screenshot pipeline and §10 for channel metadata.
"""

import asyncio
import logging
import random
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext

from game_predictor.config import IMAGE_DIR, SENTINEL_STR, USER_AGENTS, MIN_DELAY_SECONDS

logger = logging.getLogger(__name__)

_browser: Browser | None = None
_context: BrowserContext | None = None

SCREENSHOT_TIMEOUT_MS = 20_000
_MAX_SEEK_ATTEMPTS = 3


async def _get_browser() -> tuple[Browser, BrowserContext]:
    """Return a lazily-initialised headless Chromium browser and context."""
    global _browser, _context
    if _browser is None or not _browser.is_connected():
        pw = await async_playwright().start()
        _browser = await pw.chromium.launch(headless=True)
        _context = await _browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 720},
        )
    return _browser, _context


async def screenshot_youtube_video(
    video_url: str,
    game_slug: str,
    record_id: int,
    image_dir: Path | None = None,
) -> str | None:
    """Capture a gameplay frame from a YouTube video.

    Seeks to a random point between 15–85% of video duration to avoid
    intro sequences and end screens. Retries with a new random point
    if a seek fails. Falls back to a player-area screenshot as last resort.

    Returns the relative file path on success, None if the page cannot load.
    """
    base = image_dir or IMAGE_DIR
    dest_dir = base / game_slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{record_id}.png"

    _, context = await _get_browser()
    page = await context.new_page()
    page.set_default_timeout(SCREENSHOT_TIMEOUT_MS)

    try:
        await page.goto(video_url, wait_until="domcontentloaded")
    except Exception:
        logger.exception("Page failed to load: %s", video_url)
        await page.close()
        return None

    try:
        # Wait for the video element to appear
        await page.wait_for_selector("video", timeout=10_000)

        # Attempt to dismiss the consent dialog if present
        try:
            agree_btn = page.locator("button:has-text('Accept all')")
            if await agree_btn.count() > 0:
                await agree_btn.first.click(timeout=3_000)
                await page.wait_for_timeout(1_000)
        except Exception:
            pass

        # Click play and let video buffer
        try:
            player = page.locator("#movie_player")
            await player.click(timeout=3_000)
        except Exception:
            pass
        await page.wait_for_timeout(2_000)

        # Wait for ads to finish — skip if possible, otherwise wait them out
        for _ad_attempt in range(3):
            try:
                ad_overlay = page.locator(".ytp-ad-player-overlay, .ad-showing, .ytp-ad-module")
                if await ad_overlay.count() == 0:
                    break

                skip_btn = page.locator(
                    "button.ytp-skip-ad-button, .ytp-ad-skip-button-text, "
                    "button.ytp-ad-skip-button-modern, .ytp-skip-ad-button"
                )
                if await skip_btn.count() > 0:
                    await skip_btn.first.click(timeout=3_000)
                    logger.info("Clicked skip-ad button")
                    await page.wait_for_timeout(1_500)
                    continue

                logger.info("Ad playing but not skippable yet — waiting 5s (attempt %d)", _ad_attempt + 1)
                await page.wait_for_timeout(5_000)
            except Exception:
                break

        # Seek to a random point in the video (15–85% to skip intros/outros)
        for attempt in range(_MAX_SEEK_ATTEMPTS):
            fraction = random.uniform(0.15, 0.85)
            try:
                result = await page.evaluate(f"""
                    (() => {{
                        const v = document.querySelector('video');
                        if (!v || !v.duration || isNaN(v.duration)) return null;
                        v.currentTime = v.duration * {fraction};
                        v.pause();
                        return v.duration;
                    }})()
                """)
                if result is None:
                    continue

                await page.wait_for_timeout(1_500)

                video_el = page.locator("video")
                if await video_el.count() > 0:
                    await video_el.first.screenshot(path=str(dest_path))
                    rel = str(dest_path)
                    logger.info(
                        "Screenshot saved at %.1f%% (attempt %d): %s",
                        fraction * 100, attempt + 1, rel,
                    )
                    return rel
            except Exception:
                logger.debug(
                    "Seek to %.1f%% failed for %s (attempt %d), retrying",
                    fraction * 100, video_url, attempt + 1,
                )
                continue

        # Last resort: screenshot the player container
        try:
            player = page.locator("#movie_player")
            if await player.count() > 0:
                await player.first.screenshot(path=str(dest_path))
                rel = str(dest_path)
                logger.warning("Fell back to player-area screenshot: %s", rel)
                return rel
        except Exception:
            logger.exception("Player-area screenshot also failed for %s", video_url)

        return None

    except Exception:
        logger.exception("Unexpected error screenshotting %s", video_url)
        return None

    finally:
        await page.close()


async def extract_youtube_channel_info(channel_url: str) -> dict:
    """Navigate to a YouTube channel's about page and extract metadata.

    Returns a dict with keys: description, subscriber_count, join_date, links.
    Missing values degrade to config.SENTINEL_STR.
    """
    sentinel_result = {
        "description": SENTINEL_STR,
        "subscriber_count": SENTINEL_STR,
        "join_date": SENTINEL_STR,
        "links": [],
    }

    if not channel_url or not channel_url.strip():
        logger.warning("Empty channel URL — returning sentinel dict")
        return sentinel_result

    about_url = channel_url.rstrip("/") + "/about"

    _, context = await _get_browser()
    page = await context.new_page()
    page.set_default_timeout(SCREENSHOT_TIMEOUT_MS)

    try:
        await asyncio.sleep(MIN_DELAY_SECONDS)
        await page.goto(about_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2_000)

        description = await page.evaluate("""
            (() => {
                const el = document.querySelector(
                    '#description-container yt-formatted-string, '
                    + 'yt-formatted-string#bio, '
                    + '[class*="description"] yt-formatted-string'
                );
                return el ? el.innerText.trim() : null;
            })()
        """)

        subscriber_count = await page.evaluate("""
            (() => {
                const el = document.querySelector('#subscriber-count, [id*="subscriber"]');
                return el ? el.innerText.trim() : null;
            })()
        """)

        join_date = await page.evaluate("""
            (() => {
                const items = document.querySelectorAll(
                    '#right-column yt-formatted-string, '
                    + 'yt-formatted-string.style-scope'
                );
                for (const el of items) {
                    const text = el.innerText || '';
                    if (/joined/i.test(text)) return text.trim();
                }
                return null;
            })()
        """)

        links = await page.evaluate("""
            (() => {
                const anchors = document.querySelectorAll(
                    '#link-list-container a, #links-section a, a.yt-simple-endpoint[href*="redirect"]'
                );
                return [...anchors].map(a => a.href).filter(Boolean);
            })()
        """)

        return {
            "description": description or SENTINEL_STR,
            "subscriber_count": subscriber_count or SENTINEL_STR,
            "join_date": join_date or SENTINEL_STR,
            "links": links if links else [],
        }

    except Exception:
        logger.exception("Failed to extract channel info from %s", about_url)
        return sentinel_result

    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        path = await screenshot_youtube_video(
            video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            game_slug="stellaris",
            record_id=0,
            image_dir=Path("images_test"),
        )
        print(f"Screenshot result: {path}")

        info = await extract_youtube_channel_info(
            "https://www.youtube.com/@RickAstleyYT"
        )
        print(f"Channel info: {info}")

    asyncio.run(_main())
