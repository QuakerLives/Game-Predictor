"""
LLM-powered gameplay narration generation using Gemma 4 via langchain_openai.
See design doc §13 for narration rules and independence validation.
"""

import logging
from typing import ClassVar

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from game_predictor.config import (
    DEFAULT_LLM_BASE_URL,
    IMAGE_REF_PATTERN,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from game_predictor.prompts import NARRATION_MINIMAL_USER, NARRATION_SYSTEM, NARRATION_USER

logger = logging.getLogger(__name__)

MAX_ARTICLE_CHARS = 12_000
_STRICT_ADDENDUM = (
    "\n\nYour previous response referenced visual content. "
    "Rewrite WITHOUT any mention of images, screenshots, visuals, "
    "or anything that can be 'seen'. The narration must stand alone."
)


class _LLMHolder:
    """Lazy singleton for the ChatOpenAI client."""

    _instance: ClassVar[ChatOpenAI | None] = None

    @classmethod
    def get(cls) -> ChatOpenAI:
        if cls._instance is None:
            cls._instance = ChatOpenAI(
                base_url=DEFAULT_LLM_BASE_URL,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key="not-needed",
                timeout=180.0,
            )
        return cls._instance


def _truncate(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


def _passes_independence_check(text: str) -> bool:
    return IMAGE_REF_PATTERN.search(text) is None


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    reraise=True,
)
async def _call_llm(messages: list) -> str:
    """Invoke the LLM with tenacity retries on transient errors."""
    llm = _LLMHolder.get()
    response = await llm.ainvoke(messages)
    return response.content.strip()


async def generate_narration(
    article_text: str,
    game_name: str,
) -> str | None:
    """Generate a semantically independent gameplay narration from article text.

    Truncates article_text to ~3000 tokens (~12k chars), sends to Gemma 4,
    and validates that the result contains no image-referencing language.
    Falls back to a minimal prompt if the full attempt fails.

    Returns the narration string on success, None if all attempts fail.
    """
    truncated = _truncate(article_text)

    user_prompt = NARRATION_USER.format(
        game_name=game_name,
        article_text=truncated,
    )
    messages = [
        SystemMessage(content=NARRATION_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    # --- Primary attempt ---
    try:
        narration = await _call_llm(messages)

        if _passes_independence_check(narration):
            logger.info("Narration generated for %s (%d chars)", game_name, len(narration))
            return narration

        # One retry with stricter instructions
        logger.warning("Narration independence check failed — retrying with stricter prompt")
        messages.append(HumanMessage(content=_STRICT_ADDENDUM))
        narration = await _call_llm(messages)

        if _passes_independence_check(narration):
            logger.info("Narration passed on stricter retry for %s", game_name)
            return narration

        logger.warning("Narration still references visuals after stricter retry")
    except Exception:
        logger.exception("Primary narration attempt failed for %s", game_name)

    # --- Fallback: minimal prompt ---
    try:
        context_snippet = truncated[:200].replace("\n", " ").strip()
        minimal_prompt = NARRATION_MINIMAL_USER.format(
            game_name=game_name,
            context=context_snippet,
        )
        minimal_messages = [
            SystemMessage(content=NARRATION_SYSTEM),
            HumanMessage(content=minimal_prompt),
        ]

        narration = await _call_llm(minimal_messages)

        if _passes_independence_check(narration):
            logger.info("Minimal fallback narration succeeded for %s", game_name)
            return narration

        logger.warning("Minimal narration also failed independence check for %s", game_name)
    except Exception:
        logger.exception("Minimal fallback narration failed for %s", game_name)

    logger.error("All narration attempts failed for %s — returning None", game_name)
    return None


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    SAMPLE_ARTICLE = (
        "After 1200 hours in Stellaris, I've finally completed a Fanatic Xenophile "
        "federation run on Grand Admiral difficulty. The key was stacking diplomatic "
        "weight through research agreements and leveraging the Galactic Community to "
        "sanction my rivals. My fleet power peaked at 340k just before the endgame "
        "crisis fired."
    )

    async def _main() -> None:
        result = await generate_narration(SAMPLE_ARTICLE, "Stellaris")
        print(f"Narration: {result}")

    asyncio.run(_main())
