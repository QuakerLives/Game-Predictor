"""
LLM-powered gameplay narration generation using Ollama /api/generate (streaming).

Passes the gameplay image to Ollama alongside the text prompt so the model
(ministral-3:14b / any multimodal model) actually responds — text-only requests
to multimodal models hang indefinitely in Ollama.
"""

import asyncio
import base64
import io
import json
import logging

import httpx
from PIL import Image

from game_predictor.config import (
    IMAGE_REF_PATTERN,
    LLM_MODEL,
    ENRICHMENT_LLM_MODEL,
    OLLAMA_NATIVE_URL,
)
from game_predictor.prompts import NARRATION_MINIMAL_USER, NARRATION_SYSTEM, NARRATION_USER

logger = logging.getLogger(__name__)

MAX_ARTICLE_CHARS = 12_000
_GENERATE_ENDPOINT = f"{OLLAMA_NATIVE_URL}/api/generate"
_NUM_PREDICT = 100      # ~75 words — concise but still LLM-generated
_CHUNK_TIMEOUT = 30.0   # seconds to wait for the next streaming token
_MAX_DIM = 800
_STRICT_ADDENDUM = (
    " Your previous response referenced visual content. "
    "Rewrite WITHOUT any mention of images, screenshots, visuals, "
    "or anything that can be 'seen'. The narration must stand alone as text."
)


def _truncate(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


def _passes_independence_check(text: str) -> bool:
    return IMAGE_REF_PATTERN.search(text) is None


def _load_image_b64(image_path: str) -> str | None:
    """Resize to 800px max and return base64-encoded JPEG, or None on error."""
    try:
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
    except Exception:
        logger.warning("Could not load image for narration: %s", image_path)
        return None


async def _call_ollama(prompt: str, img_b64: str | None = None, model: str = LLM_MODEL) -> str | None:
    """Stream a prompt from Ollama, optionally with an image.

    Including the image is required for multimodal-only models (e.g.
    ministral-3:14b) that hang indefinitely on text-only requests.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": _NUM_PREDICT},
    }
    if img_b64:
        payload["images"] = [img_b64]

    def _sync() -> str | None:
        logger.info("Narration: sending request (image=%s)", "yes" if img_b64 else "no")
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
            logger.info("Narration: received %d chars", len(result))
            return result if result else None
        except httpx.ReadTimeout:
            logger.warning(
                "Narration thread → ReadTimeout (no token for %.0fs)", _CHUNK_TIMEOUT
            )
            return None
        except Exception:
            logger.exception("Narration thread → Ollama call failed")
            return None

    try:
        return await asyncio.to_thread(_sync)
    except Exception:
        logger.exception("to_thread failed in _call_ollama")
        return None


async def generate_narration(
    article_text: str,
    game_name: str,
    image_path: str | None = None,
    model: str = LLM_MODEL,
) -> str | None:
    """Generate a semantically independent gameplay narration.

    Pass ``image_path`` so the multimodal model has an image to anchor on —
    without it, multimodal-only models in Ollama never produce tokens.

    Returns the narration string on success, None if all attempts fail.
    """
    truncated = _truncate(article_text)
    img_b64 = await asyncio.to_thread(_load_image_b64, image_path) if image_path else None

    if image_path and not img_b64:
        logger.warning("Image load failed for %s — narration will proceed without image", game_name)

    user_prompt = NARRATION_USER.format(
        game_name=game_name,
        article_text=truncated,
    )
    prompt = f"{NARRATION_SYSTEM}\n\n{user_prompt}"

    # --- Primary attempt ---
    narration = await _call_ollama(prompt, img_b64, model=model)

    if narration and _passes_independence_check(narration):
        logger.info("Narration generated for %s (%d chars)", game_name, len(narration))
        return narration

    if narration:
        logger.warning("Narration independence check failed for %s — retrying", game_name)
        narration = await _call_ollama(prompt + _STRICT_ADDENDUM, img_b64, model=model)
        if narration and _passes_independence_check(narration):
            logger.info("Narration passed on stricter retry for %s", game_name)
            return narration
        logger.warning("Narration still references visuals after retry for %s", game_name)

    # --- Fallback: minimal prompt ---
    context_snippet = truncated[:200].replace("\n", " ").strip()
    minimal_user = NARRATION_MINIMAL_USER.format(
        game_name=game_name,
        context=context_snippet,
    )
    minimal_prompt = f"{NARRATION_SYSTEM}\n\n{minimal_user}"
    narration = await _call_ollama(minimal_prompt, img_b64, model=model)

    if narration and _passes_independence_check(narration):
        logger.info("Minimal fallback narration succeeded for %s", game_name)
        return narration

    logger.error("All narration attempts failed for %s — returning None", game_name)
    return None


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    SAMPLE_ARTICLE = (
        "After 1200 hours in Stellaris, I've finally completed a Fanatic Xenophile "
        "federation run on Grand Admiral difficulty. The key was stacking diplomatic "
        "weight through research agreements and leveraging the Galactic Community to "
        "sanction my rivals. My fleet power peaked at 340k just before the endgame "
        "crisis fired."
    )

    async def _main() -> None:
        img = sys.argv[1] if len(sys.argv) > 1 else None
        result = await generate_narration(SAMPLE_ARTICLE, "Stellaris", image_path=img)
        print(f"Narration: {result}")

    asyncio.run(_main())
