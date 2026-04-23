"""
Global configuration: game definitions, sentinel constants, concurrency limits,
and shared regex patterns used across the entire pipeline.
"""

import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Database & Image Paths
# ---------------------------------------------------------------------------

DB_PATH = "data/gameplay_data.duckdb"
IMAGE_DIR = Path("data/images")

GAME_SLUGS = ["stellaris", "no_mans_sky", "apex_legends", "stardew_valley", "skyrim"]

# ---------------------------------------------------------------------------
# Sentinel Constants (ML-readable nulls — see §3.2)
# ---------------------------------------------------------------------------

SENTINEL_STR = "N/A"
SENTINEL_INT = -1
SENTINEL_TIMESTAMP = datetime(1970, 1, 1)
SENTINEL_QUAL = "Fair"
SENTINEL_QUOTES: list[str] = ["N/A"]

# ---------------------------------------------------------------------------
# Narration Independence Regex (see §13.2 rule 12)
# ---------------------------------------------------------------------------

IMAGE_REF_PATTERN = re.compile(
    r"(?i)\b(image|screenshot|picture|shown|depicted|visible"
    r"|as seen|can be seen|looks like|appears in)\b"
)

# ---------------------------------------------------------------------------
# Game Definitions (see §1.3 + §5.1)
# ---------------------------------------------------------------------------

GAMES = [
    {
        "name": "Stellaris",
        "slug": "stellaris",
        "target": 200,
        "queries_img": [
            "Stellaris gameplay screenshot",
            "Stellaris empire map",
            "Stellaris fleet battle",
            "Stellaris galaxy view",
            "Stellaris megastructure",
            "Stellaris federation",
            "Stellaris war",
        ],
        "queries_yt": [
            "Stellaris gameplay",
            "Stellaris let's play",
            "Stellaris tutorial",
            "Stellaris multiplayer",
            "Stellaris timelapse",
        ],
    },
    {
        "name": "No Man's Sky",
        "slug": "no_mans_sky",
        "target": 200,
        "queries_img": [
            "No Man's Sky screenshot",
            "No Man's Sky planet discovery",
            "No Man's Sky base building",
            "No Man's Sky space combat",
            "No Man's Sky exploration",
            "No Man's Sky freighter",
            "No Man's Sky fauna",
        ],
        "queries_yt": [
            "No Man's Sky exploration",
            "No Man's Sky gameplay",
            "No Man's Sky base building guide",
            "No Man's Sky space combat",
            "No Man's Sky update",
        ],
    },
    {
        "name": "Apex Legends",
        "slug": "apex_legends",
        "target": 200,
        "queries_img": [
            "Apex Legends gameplay",
            "Apex Legends match results",
            "Apex Legends ranked",
            "Apex Legends squad wipe",
            "Apex Legends champion",
            "Apex Legends legend select",
            "Apex Legends arena",
        ],
        "queries_yt": [
            "Apex Legends ranked",
            "Apex Legends gameplay",
            "Apex Legends tips",
            "Apex Legends season",
            "Apex Legends highlights",
        ],
    },
    {
        "name": "Stardew Valley",
        "slug": "stardew_valley",
        "target": 200,
        "queries_img": [
            "Stardew Valley farm screenshot",
            "Stardew Valley gameplay",
            "Stardew Valley progress",
            "Stardew Valley mine",
            "Stardew Valley fishing",
            "Stardew Valley community center",
            "Stardew Valley seasons",
        ],
        "queries_yt": [
            "Stardew Valley gameplay",
            "Stardew Valley farm tour",
            "Stardew Valley guide",
            "Stardew Valley tips",
            "Stardew Valley let's play",
        ],
    },
    {
        "name": "Skyrim",
        "slug": "skyrim",
        "target": 200,
        "queries_img": [
            "Skyrim screenshot",
            "Skyrim gameplay",
            "Skyrim character build",
            "Skyrim dragon fight",
            "Skyrim landscape",
            "Skyrim modded",
            "Skyrim dungeon",
        ],
        "queries_yt": [
            "Skyrim gameplay",
            "Skyrim build guide",
            "Skyrim modded playthrough",
            "Skyrim let's play",
            "Skyrim exploration",
        ],
    },
]

# ---------------------------------------------------------------------------
# Concurrency Limits (see §5.3)
# ---------------------------------------------------------------------------

BROWSER_CONCURRENCY = 5
LLM_CONCURRENCY = 5
DOWNLOAD_CONCURRENCY = 10

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

MIN_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Enrichment Pass (see addendum)
# ---------------------------------------------------------------------------

ENRICHMENT_BATCH_SIZE = 5
ENRICHMENT_RECORD_TIMEOUT = 120        # seconds per record
ENRICHMENT_BROWSER_TIMEOUT = 60        # seconds for article/channel extraction
ENRICHMENT_LLM_TIMEOUT = 30            # seconds for experience assessment

# Fields that can be enriched (allowlist for UPDATE queries)
ENRICHABLE_FIELDS = frozenset({
    "player_name",
    "gameplay_timestamp",
    "experience_level",
    "gameplay_level",
    "total_playtime",
    "channel_description",
    "player_experience_narration",
    "identifying_quotes",
})

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
OLLAMA_NATIVE_URL = "http://localhost:11434"
LLM_MODEL = "gemma4:26b"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# User-Agent Rotation Pool (see §7.1)
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 OPR/110.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]
