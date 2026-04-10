"""
Web search tools: Google Images, Bing Images, and YouTube.
Uses Playwright for headless browser automation with anti-detection measures.
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from urllib.parse import quote

from playwright.async_api import async_playwright

from game_predictor.config import SENTINEL_TIMESTAMP, USER_AGENTS
from game_predictor.models import ImageResult

logger = logging.getLogger(__name__)

DELAY_MULTIPLIER = float(os.environ.get("SCRAPER_DELAY_MULTIPLIER", "1"))
SEARCH_FALLBACK = os.environ.get("SCRAPER_SEARCH_FALLBACK", "").lower()


async def create_browser():
    """Launch a headless Chromium browser with anti-detection args.

    Returns (playwright, browser, context) tuple. Caller is responsible
    for closing all three in reverse order.
    """
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ],
    )
    width = 1920 + random.randint(-50, 50)
    height = 1080 + random.randint(-50, 50)
    context = await browser.new_context(
        user_agent=random.choice(USER_AGENTS),
        viewport={"width": width, "height": height},
        locale="en-US",
    )
    return pw, browser, context


async def dismiss_cookie_consent(page) -> None:
    """Best-effort click on common cookie-consent buttons."""
    selectors = [
        "button:has-text('Accept')",
        "button:has-text('Accept all')",
        "button:has-text('I agree')",
        "button:has-text('Consent')",
        "button:has-text('OK')",
        "[aria-label='Accept cookies']",
        "#L2AGLb",  # Google consent
    ]
    for sel in selectors:
        try:
            await page.click(sel, timeout=2000)
            logger.debug("Dismissed cookie consent via %s", sel)
            return
        except Exception:
            continue


async def _delay(seconds: float = 2.0) -> None:
    """Sleep with jitter, scaled by the global delay multiplier."""
    await asyncio.sleep(seconds * DELAY_MULTIPLIER * random.uniform(0.8, 1.2))


# ── CAPTCHA Solver (Gemma 4 multimodal) ──────────────────────────────────


async def _attempt_captcha_solve(page) -> bool:
    """Use Gemma 4 to analyze and attempt to solve a Google CAPTCHA.

    Takes a screenshot of the current page, sends it to Gemma 4 for analysis,
    and follows the model's instructions to interact with CAPTCHA elements.
    Returns True if the CAPTCHA appears solved, False otherwise.
    """
    import base64
    import httpx
    from game_predictor.config import LLM_MODEL, OLLAMA_NATIVE_URL

    logger.info("Attempting CAPTCHA solve via Gemma 4 multimodal")

    try:
        screenshot_bytes = await page.screenshot(full_page=False)
        img_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = (
            "You are looking at a Google CAPTCHA or consent page.\n\n"
            "Describe exactly what you see:\n"
            "1. Is this a reCAPTCHA image challenge (e.g. 'Select all images with traffic lights')?\n"
            "2. Is this a cookie/consent dialog with an 'Accept' or 'I agree' button?\n"
            "3. Is this a 'verify you are human' checkbox?\n"
            "4. What specific text or buttons are visible?\n\n"
            "Answer as JSON: {\"type\": \"recaptcha_grid\"|\"consent\"|\"checkbox\"|\"other\", "
            "\"instruction\": \"what the page asks\", \"action\": \"what to click or do\"}"
        )

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_NATIVE_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

        logger.info("CAPTCHA analysis: %s", raw[:200])

        import json as _json
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            analysis = _json.loads(cleaned)
        except _json.JSONDecodeError:
            analysis = {"type": "other", "action": raw}

        captcha_type = analysis.get("type", "other")

        if captcha_type == "consent":
            await dismiss_cookie_consent(page)
            await page.wait_for_timeout(2000)
            return True

        if captcha_type == "checkbox":
            try:
                checkbox = page.locator("iframe[title*='reCAPTCHA']")
                if await checkbox.count() > 0:
                    frame = page.frame_locator("iframe[title*='reCAPTCHA']")
                    await frame.locator(".recaptcha-checkbox-border").click(timeout=5000)
                    await page.wait_for_timeout(3000)

                    challenge_frame = page.locator("iframe[title*='recaptcha challenge']")
                    if await challenge_frame.count() == 0:
                        logger.info("CAPTCHA checkbox click succeeded — no challenge appeared")
                        return True
                    logger.info("CAPTCHA checkbox triggered image challenge")
            except Exception:
                logger.debug("Checkbox click failed")

        if captcha_type == "recaptcha_grid":
            logger.warning(
                "reCAPTCHA image grid detected — Gemma 4 cannot reliably solve grid challenges. "
                "Falling back to Bing."
            )
            return False

        logger.warning("Unknown CAPTCHA type: %s — cannot solve", captcha_type)
        return False

    except Exception:
        logger.exception("CAPTCHA solve attempt failed")
        return False


# ── Google Images ─────────────────────────────────────────────────────────


async def search_google_images(
    query: str, num_results: int = 20
) -> list[ImageResult]:
    """Search Google Images and return full-resolution image results.

    Falls back to Bing if SCRAPER_SEARCH_FALLBACK=bing or on CAPTCHA/failure.
    """
    if SEARCH_FALLBACK == "bing":
        logger.info("SCRAPER_SEARCH_FALLBACK=bing — routing to Bing Images")
        return await search_bing_images(query, num_results)

    pw, browser, context = await create_browser()
    results: list[ImageResult] = []
    try:
        page = await context.new_page()
        url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
        logger.info("Google Images search: %s", query)
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await dismiss_cookie_consent(page)

        # Wait for the image grid
        try:
            await page.wait_for_selector("div#search img, div#islrg img", timeout=10000)
        except Exception:
            logger.warning("Image grid did not load — possible CAPTCHA, attempting solve")
            solved = await _attempt_captcha_solve(page)
            if solved:
                try:
                    await page.wait_for_selector(
                        "div#search img, div#islrg img", timeout=10000
                    )
                    logger.info("CAPTCHA solved — Google Images grid loaded")
                except Exception:
                    logger.warning("CAPTCHA solve did not restore image grid — falling back to Bing")
                    await browser.close()
                    await pw.stop()
                    return await search_bing_images(query, num_results)
            else:
                logger.warning("CAPTCHA solve failed — falling back to Bing")
                await browser.close()
                await pw.stop()
                return await search_bing_images(query, num_results)

        # Scroll to load more thumbnails
        for _ in range(3):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await _delay(1.0)

        thumbnails = await page.query_selector_all(
            "div#search img[data-src], div#islrg img"
        )
        if not thumbnails:
            thumbnails = await page.query_selector_all("img.Q4LuWd, img.rg_i")

        logger.info("Found %d thumbnails for '%s'", len(thumbnails), query)

        for thumb in thumbnails[: num_results + 10]:
            if len(results) >= num_results:
                break
            try:
                await thumb.scroll_into_view_if_needed()
                await thumb.click()
                await _delay(2.0)

                # Side panel: extract the full-res image
                full_img_el = await page.query_selector(
                    "img.sFlh5c.pT0Scc.iPVvYb, "
                    "img[jsname='kn3ccd'], "
                    "a[jsname='sTFXNd'] img"
                )
                if not full_img_el:
                    full_img_el = await page.query_selector(
                        "div[jscontroller] img[src^='http']"
                    )

                image_url = ""
                if full_img_el:
                    image_url = await full_img_el.get_attribute("src") or ""

                if not image_url or image_url.startswith("data:"):
                    continue

                # Source page link
                source_link = await page.query_selector(
                    "a.sFlh5c, a[jsname='sTFXNd'], "
                    "div[jscontroller] a[href^='http']:not([href*='google'])"
                )
                source_url = ""
                if source_link:
                    source_url = await source_link.get_attribute("href") or ""

                # Title text
                title_el = await page.query_selector(
                    "div[jscontroller] a[href^='http'] span, "
                    "div[jscontroller] h3"
                )
                title = ""
                if title_el:
                    title = (await title_el.inner_text()).strip()

                thumb_url = await thumb.get_attribute("src") or await thumb.get_attribute("data-src") or ""

                results.append(
                    ImageResult(
                        image_url=image_url,
                        source_page_url=source_url,
                        thumbnail_url=thumb_url if not thumb_url.startswith("data:") else None,
                        title=title,
                    )
                )
                logger.debug("Collected image %d: %s", len(results), image_url[:80])
            except Exception as exc:
                logger.debug("Skipping thumbnail: %s", exc)
                continue

    except Exception as exc:
        logger.error("Google Images failed: %s — falling back to Bing", exc)
        results = await search_bing_images(query, num_results)
    finally:
        await browser.close()
        await pw.stop()

    logger.info("Google Images returned %d results for '%s'", len(results), query)
    return results


# ── Bing Images ───────────────────────────────────────────────────────────


async def search_bing_images(
    query: str, num_results: int = 20
) -> list[ImageResult]:
    """Fallback image search using Bing Images."""
    pw, browser, context = await create_browser()
    results: list[ImageResult] = []
    try:
        page = await context.new_page()
        url = f"https://www.bing.com/images/search?q={quote(query)}"
        logger.info("Bing Images search: %s", query)
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await dismiss_cookie_consent(page)

        try:
            await page.wait_for_selector("img.mimg, .iusc img", timeout=10000)
        except Exception:
            logger.warning("Bing image grid did not load")
            return results

        for _ in range(3):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await _delay(1.0)

        thumbnails = await page.query_selector_all(
            ".iusc, .imgpt a, li[data-idx] a.iusc"
        )
        if not thumbnails:
            thumbnails = await page.query_selector_all("a.iusc")

        logger.info("Bing: found %d thumbnail containers for '%s'", len(thumbnails), query)

        for thumb in thumbnails[: num_results + 10]:
            if len(results) >= num_results:
                break
            try:
                # Bing stores metadata in an 'm' attribute as JSON
                m_attr = await thumb.get_attribute("m") or ""
                img_el = await thumb.query_selector("img")

                await thumb.scroll_into_view_if_needed()
                await thumb.click()
                await _delay(2.0)

                # Try the expanded panel
                full_img_el = await page.query_selector(
                    "img.nofocus, #mainImageWindow img, .imgContainer img"
                )
                image_url = ""
                if full_img_el:
                    image_url = await full_img_el.get_attribute("src") or ""

                if not image_url or image_url.startswith("data:"):
                    # Parse from the 'm' JSON attribute
                    if '"murl":"' in m_attr:
                        start = m_attr.index('"murl":"') + 8
                        end = m_attr.index('"', start)
                        image_url = m_attr[start:end]

                if not image_url:
                    continue

                source_url = ""
                if '"purl":"' in m_attr:
                    start = m_attr.index('"purl":"') + 8
                    end = m_attr.index('"', start)
                    source_url = m_attr[start:end]

                title = ""
                if img_el:
                    title = await img_el.get_attribute("alt") or ""

                thumb_url = ""
                if img_el:
                    thumb_url = await img_el.get_attribute("src") or ""

                results.append(
                    ImageResult(
                        image_url=image_url,
                        source_page_url=source_url,
                        thumbnail_url=thumb_url if thumb_url and not thumb_url.startswith("data:") else None,
                        title=title,
                    )
                )
                logger.debug("Bing image %d: %s", len(results), image_url[:80])
            except Exception as exc:
                logger.debug("Skipping Bing thumbnail: %s", exc)
                continue

    except Exception as exc:
        logger.error("Bing Images search failed: %s", exc)
    finally:
        await browser.close()
        await pw.stop()

    logger.info("Bing Images returned %d results for '%s'", len(results), query)
    return results


# ── YouTube ───────────────────────────────────────────────────────────────


async def search_youtube(
    query: str, num_results: int = 15
) -> list[dict]:
    """Search YouTube and extract video metadata from the results page."""
    pw, browser, context = await create_browser()
    results: list[dict] = []
    try:
        page = await context.new_page()
        url = f"https://www.youtube.com/results?search_query={quote(query)}"
        logger.info("YouTube search: %s", query)
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await dismiss_cookie_consent(page)

        try:
            await page.wait_for_selector(
                "ytd-video-renderer, ytd-rich-item-renderer", timeout=10000
            )
        except Exception:
            logger.warning("YouTube results did not render")
            return results

        for _ in range(2):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await _delay(2.0)

        renderers = await page.query_selector_all("ytd-video-renderer")
        logger.info("YouTube: found %d video renderers for '%s'", len(renderers), query)

        for renderer in renderers[:num_results]:
            try:
                # Title & URL
                title_el = await renderer.query_selector(
                    "#video-title, a#video-title"
                )
                title = ""
                video_url = ""
                if title_el:
                    title = (await title_el.get_attribute("title") or "").strip()
                    href = await title_el.get_attribute("href") or ""
                    if href.startswith("/"):
                        video_url = f"https://www.youtube.com{href}"
                    elif href:
                        video_url = href

                if not video_url:
                    continue

                # Channel info
                channel_el = await renderer.query_selector(
                    "ytd-channel-name a, #channel-name a, "
                    ".ytd-channel-name a"
                )
                channel_name = ""
                channel_url = ""
                if channel_el:
                    channel_name = (await channel_el.inner_text()).strip()
                    ch_href = await channel_el.get_attribute("href") or ""
                    if ch_href.startswith("/"):
                        channel_url = f"https://www.youtube.com{ch_href}"
                    elif ch_href:
                        channel_url = ch_href

                # Metadata line (views, date)
                meta_els = await renderer.query_selector_all(
                    "#metadata-line span, .inline-metadata-item"
                )
                upload_date_text = ""
                view_count_text = ""
                for mel in meta_els:
                    text = (await mel.inner_text()).strip()
                    if "ago" in text.lower() or any(
                        m in text.lower()
                        for m in ["jan", "feb", "mar", "apr", "may", "jun",
                                  "jul", "aug", "sep", "oct", "nov", "dec"]
                    ):
                        upload_date_text = text
                    elif "view" in text.lower():
                        view_count_text = text

                upload_date = SENTINEL_TIMESTAMP
                if upload_date_text:
                    try:
                        upload_date = _parse_relative_date(upload_date_text)
                    except Exception:
                        pass

                # Description snippet
                desc_el = await renderer.query_selector(
                    "#description-text, .metadata-snippet-text, "
                    "yt-formatted-string.metadata-snippet-text"
                )
                description = ""
                if desc_el:
                    description = (await desc_el.inner_text()).strip()

                results.append({
                    "url": video_url,
                    "title": title,
                    "channel_name": channel_name,
                    "channel_url": channel_url,
                    "upload_date": upload_date,
                    "view_count": view_count_text,
                    "description": description,
                })
                logger.debug("YouTube result %d: %s", len(results), title[:60])
            except Exception as exc:
                logger.debug("Skipping YouTube renderer: %s", exc)
                continue

    except Exception as exc:
        logger.error("YouTube search failed: %s", exc)
    finally:
        await browser.close()
        await pw.stop()

    logger.info("YouTube returned %d results for '%s'", len(results), query)
    return results


def _parse_relative_date(text: str) -> datetime:
    """Best-effort parse of YouTube relative dates like '3 days ago'."""
    text = text.lower().strip()
    now = datetime.utcnow()

    for unit, kwargs_key in [
        ("second", "seconds"), ("minute", "minutes"), ("hour", "hours"),
        ("day", "days"), ("week", "weeks"), ("month", "days"), ("year", "days"),
    ]:
        if unit in text:
            num_str = "".join(c for c in text.split(unit)[0] if c.isdigit())
            if not num_str:
                break
            num = int(num_str)
            if unit == "month":
                num *= 30
            elif unit == "year":
                num *= 365
            return now - timedelta(**{kwargs_key: num})

    return SENTINEL_TIMESTAMP


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    async def _test():
        print("=== Google Images ===")
        imgs = await search_google_images("Stellaris gameplay screenshot", num_results=3)
        for r in imgs:
            print(f"  {r.title[:50]:50s}  {r.image_url[:80]}")

        print("\n=== YouTube ===")
        vids = await search_youtube("Stellaris gameplay", num_results=3)
        for v in vids:
            print(f"  {v['title'][:50]:50s}  {v['url']}")

    asyncio.run(_test())
