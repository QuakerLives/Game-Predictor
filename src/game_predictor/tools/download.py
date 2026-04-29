"""
Async image downloader using asyncio.to_thread + sync httpx.

Avoids aiohttp, which creates IOCP operations on Windows (ProactorEventLoop)
that starve asyncio scheduled callbacks (wait_for timers) when multiple
downloads run concurrently. Running sync httpx in threads keeps the event
loop free to fire timers reliably.
"""

import asyncio
import io
import logging
from pathlib import Path

import httpx
from PIL import Image

from game_predictor.config import IMAGE_DIR, USER_AGENTS

logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 10 * 1024        # 10 KB
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
_DOWNLOAD_TIMEOUT = 15.0
_DOWNLOAD_HEADERS = {"User-Agent": USER_AGENTS[0]}


async def download_image(
    image_url: str,
    game_slug: str,
    record_id: int,
    image_dir: Path | None = None,
) -> str | None:
    """Download an image, validate it, convert to PNG, and save to disk.

    Runs entirely in a thread (sync httpx + PIL) so no IOCP operations are
    created — the asyncio event loop stays free to process timer callbacks.

    Returns posix-style relative path on success, None on failure.
    """
    base = image_dir or IMAGE_DIR
    dest_dir = base / game_slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{record_id}.png"

    def _sync() -> str | None:
        # Download
        try:
            with httpx.Client(headers=_DOWNLOAD_HEADERS, follow_redirects=True) as client:
                resp = client.get(image_url, timeout=_DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    logger.warning("Non-image content-type %r from %s", content_type, image_url)
                    return None
                data = resp.content
        except Exception:
            logger.debug("Download failed for %s", image_url)
            return None

        # Size checks
        if len(data) < MIN_IMAGE_BYTES:
            logger.warning("Image too small (%d bytes) from %s", len(data), image_url)
            return None
        if len(data) > MAX_IMAGE_BYTES:
            logger.warning("Image too large (%d bytes) from %s", len(data), image_url)
            return None

        # Validate and convert
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            img = Image.open(io.BytesIO(data))  # re-open after verify()
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            img.save(dest_path, format="PNG")
        except Exception:
            logger.debug("PIL processing failed for %s", image_url)
            return None

        rel = dest_path.as_posix()
        logger.info("Saved %s (%d bytes)", rel, len(data))
        return rel

    try:
        return await asyncio.to_thread(_sync)
    except Exception:
        logger.exception("to_thread failed in download_image")
        return None


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    TEST_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "4/47/PNG_transparency_demonstration_1.png/"
        "280px-PNG_transparency_demonstration_1.png"
    )

    async def _main() -> None:
        result = await download_image(
            image_url=TEST_URL,
            game_slug="stellaris",
            record_id=0,
            image_dir=Path("images_test"),
        )
        print(f"Download result: {result}")

    asyncio.run(_main())
