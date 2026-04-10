"""
Gemma 4 prompt templates for structured extraction, narration generation,
and experience-level assessment. See design doc §8.
"""

# ---------------------------------------------------------------------------
# §8.1 — Structured Article Extraction
# ---------------------------------------------------------------------------

ARTICLE_EXTRACTION_SYSTEM = (
    "You are a data extraction agent. Return ONLY valid JSON, "
    "no markdown fences, no preamble. "
    "Use the specified default values for any field you cannot extract."
)

ARTICLE_EXTRACTION_USER = """\
Extract structured data from this {game_name} article:

---
{article_text}
---

Return JSON:
{{
  "author_name": "string (default: 'N/A')",
  "publish_date": "ISO-8601 string (default: '1970-01-01T00:00:00')",
  "gameplay_level": "integer — player level, XP, or rank number (default: -1)",
  "total_playtime": "integer hours (default: -1)",
  "identifying_quotes": ["1-3 direct quotes from the article (default: ['N/A'])"],
  "player_experience_summary": "2-3 sentences summarizing the player's experience (default: 'N/A')",
  "site_description": "1-sentence description of this publication (default: 'N/A')"
}}"""

# ---------------------------------------------------------------------------
# §8.2 — Semantically Independent Narration Generation
# ---------------------------------------------------------------------------

NARRATION_SYSTEM = (
    "You write gameplay narrations that are completely independent of any visual content. "
    "Never reference images, screenshots, or anything visual. 2-5 sentences only."
)

NARRATION_USER = """\
Based on this article about {game_name}:

---
{article_text}
---

Write a 2-5 sentence narration about the gameplay context.
Describe ONLY: player strategy, goals, progression, emotional experience,
decision-making, game mechanics used, or game lore context.

FORBIDDEN phrases: "as shown", "in the image", "the screenshot", "depicted",
"visible", "can be seen", "as seen", "picture", "looks like", "appears".

The narration must make complete sense to someone who has never seen any image."""

NARRATION_MINIMAL_USER = """\
In 2-3 sentences, describe a typical {game_name} gameplay moment \
involving {context}.

FORBIDDEN phrases: "as shown", "in the image", "the screenshot", "depicted",
"visible", "can be seen", "as seen", "picture", "looks like", "appears".

The narration must make complete sense to someone who has never seen any image."""

# ---------------------------------------------------------------------------
# §8.3 — Experience Level Assessment (Multimodal)
# ---------------------------------------------------------------------------

EXPERIENCE_SYSTEM = (
    "You assess video game player skill from screenshots and context. "
    "Respond with exactly one word from: Poor, Fair, Good, Excellent, Superior"
)

EXPERIENCE_USER = """\
Game: {game_name}
Context: {context}

Assess this player's experience level based on visible indicators
(UI state, equipment, progression, rank, achievements).

Respond with exactly one word."""

# ---------------------------------------------------------------------------
# §8.4 — Gameplay Level Extraction (Multimodal)
# ---------------------------------------------------------------------------

GAMEPLAY_LEVEL_SYSTEM = (
    "You extract numeric gameplay progression data from screenshots. "
    "Return only a JSON object."
)

GAMEPLAY_LEVEL_USER = """\
Game: {game_name}

Find any visible numeric progression indicator: player level, XP amount,
rank number, season level, skill points, or similar.

Return: {{"gameplay_level": <integer or -1 if not found>, "source": "<what you read>"}}"""

# ---------------------------------------------------------------------------
# Agent System Prompt (see §2.2)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a web-scraping agent building an ML training dataset. Your job is to
collect {target} gameplay images AND full structured metadata for {game_name}.

CRITICAL RULES:
- Every record MUST have both an image AND a gameplay narration.
- The narration must NOT describe or reference the image. It describes gameplay
  context, strategy, progression, or lore independently.
- Prefer Google Images results that link to articles (not raw image hosts).
- For each article, extract the author profile and gameplay stats.
- For YouTube results, screenshot a gameplay frame and extract channel info.
- ALL text fields (narration, channel_description, player_experience_narration,
  identifying_quotes) must be populated. Use "N/A" only as a last resort.
- You have a budget of {remaining} images remaining for this game.
- If a source fails after 2 retries, skip it and move to the next candidate."""
