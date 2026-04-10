"""
Content extraction tools: article text + structured data via Gemma 4,
and author profile scraping.
"""

import asyncio
import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI

from game_predictor.config import (
    DEFAULT_LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SENTINEL_INT,
    SENTINEL_QUOTES,
    SENTINEL_STR,
    SENTINEL_TIMESTAMP,
)
from game_predictor.models import ArticleData
from game_predictor.prompts import ARTICLE_EXTRACTION_SYSTEM, ARTICLE_EXTRACTION_USER

logger = logging.getLogger(__name__)

# ── Module-level LLM singleton ────────────────────────────────────────────

_llm = ChatOpenAI(
    base_url=DEFAULT_LLM_BASE_URL,
    api_key="not-needed",
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

STRIP_SELECTORS = [
    "nav", "footer", "header", "aside",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    ".sidebar", "#sidebar", ".ad", ".ads", ".advertisement",
    "[class*='cookie']", "[class*='popup']", "[class*='modal']",
    "[class*='newsletter']", "[class*='social-share']",
    "script", "style", "iframe", "noscript",
]

MAX_ARTICLE_CHARS = 24_000  # ~6000 tokens


async def click_article_and_extract(
    source_url: str, game_name: str
) -> ArticleData:
    """Navigate to an article page, strip boilerplate, extract structured
    data via Gemma 4, and return an ArticleData instance.

    On any failure, returns ArticleData with sentinel defaults.
    """
    from game_predictor.tools.search import create_browser

    pw, browser, context = await create_browser()
    try:
        page = await context.new_page()
        logger.info("Extracting article: %s", source_url)
        await page.goto(
            source_url, wait_until="domcontentloaded", timeout=15000
        )

        # Strip non-content elements
        for sel in STRIP_SELECTORS:
            try:
                await page.evaluate(
                    f"document.querySelectorAll('{sel}').forEach(e => e.remove())"
                )
            except Exception:
                pass

        # Extract main text: prefer <article>, fall back to <body>
        raw_text = ""
        for container in ["article", "main", "[role='main']", "body"]:
            try:
                raw_text = await page.inner_text(container)
                if raw_text and len(raw_text.strip()) > 200:
                    break
            except Exception:
                continue

        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning("No meaningful text at %s", source_url)
            return _default_article()

        article_text = raw_text.strip()[:MAX_ARTICLE_CHARS]

        # Ask Gemma 4 to extract structured data
        return await _llm_extract(article_text, game_name)

    except Exception as exc:
        logger.error("Article extraction failed for %s: %s", source_url, exc)
        return _default_article()
    finally:
        await browser.close()
        await pw.stop()


async def _llm_extract(article_text: str, game_name: str) -> ArticleData:
    """Send article text to Gemma 4 and parse the JSON response."""
    user_prompt = ARTICLE_EXTRACTION_USER.format(
        game_name=game_name, article_text=article_text
    )

    try:
        response = await _llm.ainvoke([
            {"role": "system", "content": ARTICLE_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])
        raw = response.content.strip()

        # Strip markdown fences if the model wraps its output
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("LLM extraction failed: %s", exc)
        return _default_article()

    # Parse publish_date
    publish_date = SENTINEL_TIMESTAMP
    raw_date = data.get("publish_date", "")
    if raw_date and raw_date != "1970-01-01T00:00:00":
        try:
            publish_date = datetime.fromisoformat(raw_date)
        except (ValueError, TypeError):
            pass

    quotes = data.get("identifying_quotes", list(SENTINEL_QUOTES))
    if not isinstance(quotes, list) or not quotes:
        quotes = list(SENTINEL_QUOTES)

    return ArticleData(
        author_name=data.get("author_name", SENTINEL_STR),
        publish_date=publish_date,
        body_text=article_text,
        gameplay_level=int(data.get("gameplay_level", SENTINEL_INT)),
        total_playtime=int(data.get("total_playtime", SENTINEL_INT)),
        identifying_quotes=quotes,
        site_description=data.get("site_description", SENTINEL_STR),
        player_experience_summary=data.get(
            "player_experience_summary", SENTINEL_STR
        ),
    )


def _default_article() -> ArticleData:
    """Return an ArticleData with all sentinel defaults."""
    return ArticleData()


# ── Author Profile Extraction ─────────────────────────────────────────────

AUTHOR_LINK_SELECTORS = [
    'a[rel="author"]',
    ".author-name a",
    "[class*='author'] a",
    ".byline a",
    "[rel='author']",
]


async def extract_author_profile(
    page_url: str, author_name: str
) -> dict:
    """Best-effort extraction of the author's bio from their profile page.

    Looks for a byline link on the article page, navigates to the profile,
    and extracts bio text. Returns sentinel dict on failure.
    """
    from game_predictor.tools.search import create_browser

    sentinel = {"bio": SENTINEL_STR, "profile_url": SENTINEL_STR}

    if not author_name or author_name == SENTINEL_STR:
        return sentinel

    pw, browser, context = await create_browser()
    try:
        page = await context.new_page()
        await page.goto(
            page_url, wait_until="domcontentloaded", timeout=15000
        )

        # Find author link
        author_href = None
        for sel in AUTHOR_LINK_SELECTORS:
            try:
                link = await page.query_selector(sel)
                if link:
                    href = await link.get_attribute("href") or ""
                    link_text = (await link.inner_text()).strip().lower()
                    if author_name.lower() in link_text or href:
                        author_href = href
                        break
            except Exception:
                continue

        if not author_href:
            logger.debug("No author link found for '%s' at %s", author_name, page_url)
            return sentinel

        if author_href.startswith("/"):
            from urllib.parse import urljoin
            author_href = urljoin(page_url, author_href)

        logger.info("Navigating to author profile: %s", author_href)
        await page.goto(
            author_href, wait_until="domcontentloaded", timeout=15000
        )

        # Extract bio text from common containers
        bio = ""
        for sel in [
            ".author-bio", ".bio", "[class*='biography']",
            "[class*='about']", "article", "main", "body",
        ]:
            try:
                text = await page.inner_text(sel)
                if text and len(text.strip()) > 30:
                    bio = text.strip()[:2000]
                    break
            except Exception:
                continue

        if not bio:
            return sentinel

        return {"bio": bio, "profile_url": author_href}

    except Exception as exc:
        logger.debug("Author profile extraction failed: %s", exc)
        return sentinel
    finally:
        await browser.close()
        await pw.stop()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    async def _test():
        print("=== Article Extraction ===")
        article = await click_article_and_extract(
            "https://www.pcgamer.com/stellaris/",
            "Stellaris",
        )
        print(f"  Author : {article.author_name}")
        print(f"  Date   : {article.publish_date}")
        print(f"  Level  : {article.gameplay_level}")
        print(f"  Quotes : {article.identifying_quotes}")
        print(f"  Body   : {article.body_text[:120]}...")

        print("\n=== Author Profile ===")
        profile = await extract_author_profile(
            "https://www.pcgamer.com/stellaris/",
            article.author_name,
        )
        print(f"  Bio URL: {profile['profile_url']}")
        print(f"  Bio    : {profile['bio'][:120]}...")

    asyncio.run(_test())
