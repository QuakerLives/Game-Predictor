"""
Multimodal experience-level and gameplay-level assessment via Ollama (streaming).

Uses stream=True so each token has its own read timeout — more reliable on
Windows than stream=False where a single large recv() may not time out.
"""

import asyncio
import base64
import io
import json
import logging
from pathlib import Path

import httpx
from PIL import Image

from game_predictor.config import LLM_MODEL, ENRICHMENT_LLM_MODEL, OLLAMA_NATIVE_URL, SENTINEL_INT, SENTINEL_QUAL
from game_predictor.prompts import (
    EXPERIENCE_SYSTEM,
    EXPERIENCE_USER,
    GAMEPLAY_LEVEL_SYSTEM,
    GAMEPLAY_LEVEL_USER,
)

logger = logging.getLogger(__name__)

_VALID_EXPERIENCE_LEVELS = frozenset({"Poor", "Fair", "Good", "Excellent", "Superior"})
_GENERATE_ENDPOINT = f"{OLLAMA_NATIVE_URL}/api/generate"
_MAX_DIM = 800      # resize before sending — cuts a 2.3MB PNG to ~100KB JPEG
_CHUNK_TIMEOUT = 30.0   # seconds to wait for the next token before giving up


def _load_image_b64(image_path: str) -> str:
    """Read, resize to 800px max dimension, and return base64-encoded JPEG."""
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > _MAX_DIM:
            ratio = _MAX_DIM / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _call_ollama_multimodal(
    prompt: str,
    img_b64: str,
    *,
    num_predict: int | None = None,
    model: str = LLM_MODEL,
) -> str | None:
    """Stream a prompt+image from Ollama token-by-token.

    Each token is a separate recv() with _CHUNK_TIMEOUT applied, so timeouts
    fire reliably regardless of how long the model takes to generate.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": True,
    }
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}

    def _sync() -> str | None:
        try:
            parts: list[str] = []
            with httpx.Client() as client:
                with client.stream(
                    "POST",
                    _GENERATE_ENDPOINT,
                    json=payload,
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=_CHUNK_TIMEOUT,
                        write=10.0,
                        pool=5.0,
                    ),
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line.strip():
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        parts.append(chunk.get("response", ""))
                        if chunk.get("done"):
                            break
            result = "".join(parts).strip()
            return result if result else None
        except httpx.ReadTimeout:
            logger.warning("Assessment: ReadTimeout (no token for %.0fs)", _CHUNK_TIMEOUT)
            return None
        except Exception:
            logger.exception("Assessment: Ollama call failed")
            return None

    try:
        return await asyncio.to_thread(_sync)
    except Exception:
        logger.exception("to_thread failed in _call_ollama_multimodal")
        return None


async def assess_experience_level(
    image_path: str,
    article_text: str,
    game_name: str,
    model: str = LLM_MODEL,
) -> str:
    """Assess player experience level from a screenshot and article context.

    Returns config.SENTINEL_QUAL ('Fair') if the model returns an invalid response.
    """
    try:
        img_b64 = await asyncio.to_thread(_load_image_b64, image_path)
    except FileNotFoundError:
        logger.error("Image not found: %s — returning sentinel", image_path)
        return SENTINEL_QUAL

    context_snippet = article_text[:500].replace("\n", " ").strip() if article_text else ""

    prompt = (
        f"{EXPERIENCE_SYSTEM}\n\n"
        f"{EXPERIENCE_USER.format(game_name=game_name, context=context_snippet)}"
    )

    raw = await _call_ollama_multimodal(prompt, img_b64, num_predict=30, model=model)

    if raw:
        for word in raw.split():
            cleaned = word.strip(".,;:!?\"'()")
            if cleaned in _VALID_EXPERIENCE_LEVELS:
                logger.info("Experience level for %s: %s", game_name, cleaned)
                return cleaned

    logger.warning(
        "Invalid experience response for %s (got: %r) — returning sentinel '%s'",
        game_name, raw, SENTINEL_QUAL,
    )
    return SENTINEL_QUAL


async def validate_gameplay_image(image_path: str, game_name: str) -> bool:
    """Check whether an image actually shows gameplay from the named game.

    Returns True if the image is valid gameplay, False if it should be discarded.
    """
    try:
        img_b64 = await asyncio.to_thread(_load_image_b64, image_path)
    except FileNotFoundError:
        logger.error("Image not found for validation: %s", image_path)
        return False

    prompt = (
        f"You are a strict image classifier for a video game dataset.\n\n"
        f"Look at this image and determine: does it show ACTUAL GAMEPLAY or "
        f"GAME-RELATED CONTENT from the video game \"{game_name}\"?\n\n"
        f"REJECT the image (answer NO) if it shows ANY of these:\n"
        f"- An advertisement or sponsored content (e.g. product ads, brand logos like "
        f"Nike, Under Armour, Planet Fitness, political ads, movie trailers)\n"
        f"- A YouTube pre-roll or mid-roll ad\n"
        f"- Content completely unrelated to the video game \"{game_name}\"\n"
        f"- A blank, loading, or error screen\n\n"
        f"ACCEPT the image (answer YES) if it shows ANY of these:\n"
        f"- In-game footage, menus, HUD, or UI from \"{game_name}\"\n"
        f"- A streamer/YouTuber with the game visible on screen\n"
        f"- Game artwork, loading screens, or title screens from \"{game_name}\"\n\n"
        f"Answer with ONLY 'YES' or 'NO' on the first line, "
        f"then a brief reason on the second line."
    )

    raw = await _call_ollama_multimodal(prompt, img_b64, num_predict=60)

    if raw:
        first_line = raw.strip().split("\n")[0].strip().upper()
        is_valid = first_line.startswith("YES")
        logger.info(
            "Image validation for %s [%s]: %s — %s",
            game_name, image_path,
            "ACCEPTED" if is_valid else "REJECTED",
            raw.strip().replace("\n", " | ")[:120],
        )
        return is_valid

    logger.warning("Image validation call failed for %s — accepting by default", image_path)
    return True


async def extract_gameplay_level(
    image_path: str,
    game_name: str,
) -> int:
    """Extract a numeric gameplay progression indicator from a screenshot.

    Returns config.SENTINEL_INT (-1) if extraction or parsing fails.
    """
    try:
        img_b64 = await asyncio.to_thread(_load_image_b64, image_path)
    except FileNotFoundError:
        logger.error("Image not found: %s — returning sentinel", image_path)
        return SENTINEL_INT

    prompt = (
        f"{GAMEPLAY_LEVEL_SYSTEM}\n\n"
        f"{GAMEPLAY_LEVEL_USER.format(game_name=game_name)}"
    )

    raw = await _call_ollama_multimodal(prompt, img_b64)

    if raw:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(cleaned)
            level = int(data.get("gameplay_level", SENTINEL_INT))
            logger.info("Gameplay level for %s: %d", game_name, level)
            return level
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse gameplay level JSON for %s: %s", game_name, exc)

    logger.warning("Gameplay level extraction failed for %s — returning sentinel", game_name)
    return SENTINEL_INT


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        test_image = sys.argv[1] if len(sys.argv) > 1 else "images/stellaris/1.png"
        game_name = sys.argv[2] if len(sys.argv) > 2 else "Stellaris"

        print(f"\n--- validate_gameplay_image({test_image}, {game_name}) ---")
        valid = await validate_gameplay_image(test_image, game_name)
        print(f"Valid gameplay image: {valid}")

        print(f"\n--- assess_experience_level ---")
        exp = await assess_experience_level(
            image_path=test_image,
            article_text="A veteran player with 1200 hours who runs federation builds.",
            game_name=game_name,
        )
        print(f"Experience level: {exp}")

        print(f"\n--- extract_gameplay_level ---")
        level = await extract_gameplay_level(
            image_path=test_image,
            game_name=game_name,
        )
        print(f"Gameplay level: {level}")

    asyncio.run(_main())
