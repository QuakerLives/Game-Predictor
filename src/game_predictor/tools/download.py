"""
Async image downloader with format validation, PNG conversion, and retry logic.
See design doc §6 for download pipeline and size constraints.
"""

import io
import logging
from pathlib import Path

import aiohttp
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from game_predictor.config import IMAGE_DIR, USER_AGENTS

logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 10 * 1024       # 10 KB
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
_DOWNLOAD_HEADERS = {"User-Agent": USER_AGENTS[0]}


@retry(
    retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
    stop=stop_after_attempt(3),  # initial + 2 retries
    wait=wait_fixed(2),
    reraise=True,
)
async def _fetch_image_bytes(
    session: aiohttp.ClientSession,
    url: str,
) -> tuple[bytes, str]:
    """Download raw bytes and return (data, content_type). Raises on bad response."""
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"Non-image Content-Type: {content_type}")
        data = await resp.read()
        return data, content_type


async def download_image(
    image_url: str,
    game_slug: str,
    record_id: int,
    image_dir: Path | None = None,
) -> str | None:
    """Download an image, validate it, convert to PNG, and save to disk.

    Parameters
    ----------
    image_url : str
        Remote URL of the image to fetch.
    game_slug : str
        Game identifier used as the sub-directory name.
    record_id : int
        Unique record id used as the file stem.
    image_dir : Path | None
        Override base image directory (useful for test runs with images_test/).

    Returns
    -------
    str | None
        Relative path to the saved PNG on success, None on failure.
    """
    base = image_dir or IMAGE_DIR
    dest_dir = base / game_slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{record_id}.png"

    try:
        async with aiohttp.ClientSession(headers=_DOWNLOAD_HEADERS) as session:
            data, content_type = await _fetch_image_bytes(session, image_url)
    except Exception:
        logger.exception("Failed to download %s after retries", image_url)
        return None

    if len(data) < MIN_IMAGE_BYTES:
        logger.warning(
            "Image too small (%d bytes) from %s — skipping", len(data), image_url
        )
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(
            "Image too large (%d bytes) from %s — skipping", len(data), image_url
        )
        return None

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
        logger.exception("PIL processing failed for %s", image_url)
        return None

    rel_path = str(dest_path)
    logger.info("Saved %s (%d bytes)", rel_path, len(data))
    return rel_path


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

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
