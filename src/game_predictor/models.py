"""
Pydantic data models and sentinel constants for the scraping pipeline.
See design doc §4.1 for field definitions and sentinel conventions.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from game_predictor.config import (
    SENTINEL_INT,
    SENTINEL_QUAL,
    SENTINEL_QUOTES,
    SENTINEL_STR,
    SENTINEL_TIMESTAMP,
)


class ImageResult(BaseModel):
    """A single Google Images search result."""

    image_url: str
    source_page_url: str
    thumbnail_url: Optional[str] = None
    title: str = ""


class ArticleData(BaseModel):
    """Structured data extracted from an article page via Gemma 4."""

    author_name: str = SENTINEL_STR
    publish_date: datetime = SENTINEL_TIMESTAMP
    body_text: str = ""
    gameplay_level: int = SENTINEL_INT
    total_playtime: int = SENTINEL_INT
    identifying_quotes: list[str] = Field(
        default_factory=lambda: list(SENTINEL_QUOTES)
    )
    site_description: str = SENTINEL_STR
    player_experience_summary: str = SENTINEL_STR


class YouTubeData(BaseModel):
    """Structured data extracted from a YouTube video page."""

    channel_name: str = SENTINEL_STR
    channel_description: str = SENTINEL_STR
    video_title: str = ""
    upload_date: datetime = SENTINEL_TIMESTAMP
    screenshot_path: Optional[str] = None
    gameplay_level: int = SENTINEL_INT
    video_description: str = ""


class GameplayRecord(BaseModel):
    """
    Final record to insert into DuckDB. Every field is non-null.
    The Iron Rule: image_path and gameplay_narration must contain
    real data — never sentinels. All other fields degrade to sentinels.
    """

    video_game_name: str
    image_path: str
    player_name: str = SENTINEL_STR
    gameplay_timestamp: datetime = SENTINEL_TIMESTAMP
    experience_level: str = SENTINEL_QUAL
    gameplay_level: int = SENTINEL_INT
    total_playtime: int = SENTINEL_INT
    gameplay_narration: str = SENTINEL_STR
    channel_description: str = SENTINEL_STR
    player_experience_narration: str = SENTINEL_STR
    identifying_quotes: list[str] = Field(
        default_factory=lambda: list(SENTINEL_QUOTES)
    )
    source_url: str = SENTINEL_STR
    source_type: str = "google_images"
