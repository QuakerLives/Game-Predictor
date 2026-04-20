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
        "target": 2000,
        "queries_img": [
            "Stellaris gameplay screenshot",
            "Stellaris empire map HUD",
            "Stellaris fleet battle in game",
            "Stellaris galaxy view UI",
            "Stellaris megastructure built",
            "Stellaris federation gameplay",
            "Stellaris war in progress",
            "site:reddit.com Stellaris screenshot",
            "site:reddit.com Stellaris funny moment",
            "site:reddit.com Stellaris galaxy",
            "Stellaris playthrough screenshot",
            "Stellaris species empire screen",
            "Stellaris ascension perk screen",
            "Stellaris diplomatic screen HUD",
            "Stellaris anomaly research screenshot",
            "Stellaris planet view colonized",
            "Stellaris crisis fleet battle",
            "Stellaris fallen empire awakened",
            "Stellaris synthetic dawn gameplay",
            "Stellaris machine empire screenshot",
            "site:reddit.com r/Stellaris galaxy screenshot",
            "site:reddit.com r/Stellaris endgame",
            "Stellaris 2024 gameplay screenshot",
            "Stellaris 2025 update screenshot",
            "Stellaris first contact gameplay",
            "Stellaris overlord DLC screenshot",
            "Stellaris aquatics species screenshot",
            "Stellaris toxic god crisis",
        ],
        "queries_yt": [
            "Stellaris gameplay",
            "Stellaris let's play",
            "Stellaris tutorial",
            "Stellaris multiplayer",
            "Stellaris timelapse",
            "Stellaris campaign 2024",
            "Stellaris grand admiral difficulty",
            "Stellaris crisis endgame",
            "Stellaris federation playthrough",
            "Stellaris roleplay let's play",
            "Stellaris megastructure guide",
            "Stellaris tall empire playthrough",
            "Stellaris machine uprising gameplay",
            "Stellaris 2025 gameplay",
            "Stellaris overlord gameplay",
            "Stellaris aquatics playthrough",
            "Stellaris synthetic ascension guide",
            "Stellaris multiplayer 2024",
            "Stellaris void dweller playthrough",
            "Stellaris lithoid empire gameplay",
        ],
    },
    {
        "name": "No Man's Sky",
        "slug": "no_mans_sky",
        "target": 2000,
        "queries_img": [
            "No Man's Sky gameplay screenshot HUD",
            "No Man's Sky planet surface in game",
            "No Man's Sky base building screenshot",
            "No Man's Sky multiplayer screenshot",
            "No Man's Sky inventory screen",
            "No Man's Sky space station interior",
            "No Man's Sky exocraft driving",
            "site:reddit.com No Man's Sky screenshot",
            "site:reddit.com No Man's Sky base",
            "site:reddit.com No Man's Sky planet",
            "No Man's Sky settlement gameplay",
            "No Man's Sky expedition screenshot",
            "No Man's Sky freighter interior screenshot",
            "No Man's Sky starship cockpit HUD",
            "No Man's Sky atlas path gameplay",
            "No Man's Sky underwater biome screenshot",
            "No Man's Sky portal glyphs gameplay",
            "No Man's Sky sentinel combat screenshot",
            "No Man's Sky technology upgrade screen",
            "No Man's Sky crashed freighter exploration",
            "site:reddit.com r/NoMansSkyTheGame base",
            "site:reddit.com r/NoMansSkyTheGame planet",
            "No Man's Sky 2024 update screenshot",
            "No Man's Sky 2025 gameplay screenshot",
            "No Man's Sky worlds part 2 screenshot",
            "No Man's Sky light beam ship screenshot",
            "No Man's Sky space combat screenshot",
            "No Man's Sky anomaly space station",
        ],
        "queries_yt": [
            "No Man's Sky exploration gameplay",
            "No Man's Sky base building",
            "No Man's Sky guide 2024",
            "No Man's Sky expedition",
            "No Man's Sky update gameplay",
            "No Man's Sky let's play",
            "No Man's Sky multiplayer gameplay",
            "No Man's Sky freighter base",
            "No Man's Sky permadeath run",
            "No Man's Sky creative mode",
            "No Man's Sky living ship gameplay",
            "No Man's Sky derelict freighter",
            "No Man's Sky 2025 gameplay",
            "No Man's Sky worlds update gameplay",
            "No Man's Sky 100 hours challenge",
            "No Man's Sky settlement guide",
            "No Man's Sky interceptor gameplay",
            "No Man's Sky expedition 2024",
            "No Man's Sky starter guide",
            "No Man's Sky omega update gameplay",
        ],
    },
    {
        "name": "Apex Legends",
        "slug": "apex_legends",
        "target": 2000,
        "queries_img": [
            "Apex Legends in game screenshot",
            "Apex Legends match HUD",
            "Apex Legends ranked gameplay",
            "Apex Legends squad wipe kill feed",
            "Apex Legends champion screen",
            "Apex Legends legend select screen",
            "Apex Legends ring map",
            "site:reddit.com Apex Legends screenshot",
            "site:reddit.com Apex Legends gameplay",
            "site:reddit.com Apex Legends ranked",
            "Apex Legends damage stats screen",
            "Apex Legends loot gameplay",
            "Apex Legends end of match screen",
            "Apex Legends abilities gameplay",
            "Apex Legends third party fight",
            "Apex Legends sniper gameplay screenshot",
            "Apex Legends zone rotation HUD",
            "Apex Legends respawn beacon gameplay",
            "Apex Legends care package drop",
            "Apex Legends gold armor loot",
            "site:reddit.com r/apexlegends screenshot",
            "site:reddit.com r/apexlegends ranked moment",
            "Apex Legends season 24 screenshot",
            "Apex Legends 2025 gameplay screenshot",
            "Apex Legends mobile controller gameplay",
            "Apex Legends firing range screenshot",
            "Apex Legends finisher animation",
            "Apex Legends kill leader banner",
        ],
        "queries_yt": [
            "Apex Legends ranked gameplay",
            "Apex Legends gameplay",
            "Apex Legends tips 2024",
            "Apex Legends season highlights",
            "Apex Legends pro player",
            "Apex Legends let's play",
            "Apex Legends predator ranked",
            "Apex Legends 20 bomb gameplay",
            "Apex Legends controller player",
            "Apex Legends legend guide",
            "Apex Legends squad wipe highlights",
            "Apex Legends no fill gameplay",
            "Apex Legends 2025 ranked gameplay",
            "Apex Legends season 24 gameplay",
            "Apex Legends diamond ranked grind",
            "Apex Legends movement tech gameplay",
            "Apex Legends solo vs squad",
            "Apex Legends caustic gameplay",
            "Apex Legends bloodhound gameplay",
            "Apex Legends wraith gameplay ranked",
        ],
    },
    {
        "name": "Stardew Valley",
        "slug": "stardew_valley",
        "target": 2000,
        "queries_img": [
            "Stardew Valley farm screenshot",
            "Stardew Valley gameplay HUD",
            "Stardew Valley year 1 progress",
            "Stardew Valley mine level",
            "Stardew Valley fishing minigame",
            "Stardew Valley community center completion",
            "Stardew Valley perfection screen",
            "site:reddit.com Stardew Valley farm",
            "site:reddit.com Stardew Valley screenshot",
            "site:reddit.com Stardew Valley progress",
            "Stardew Valley inventory screen",
            "Stardew Valley spouse heart events",
            "Stardew Valley greenhouse farm layout",
            "Stardew Valley skull cavern level screenshot",
            "Stardew Valley saloon night gameplay",
            "Stardew Valley festival gameplay screenshot",
            "Stardew Valley horse stable screenshot",
            "Stardew Valley junimo hut farm",
            "Stardew Valley qi challenge completion",
            "Stardew Valley island farm screenshot",
            "site:reddit.com r/StardewValley farm layout",
            "site:reddit.com r/StardewValley year 3",
            "Stardew Valley 1.6 update screenshot",
            "Stardew Valley 2024 gameplay screenshot",
            "Stardew Valley modded screenshot",
            "Stardew Valley expanded screenshot",
            "Stardew Valley cooking recipe screen",
            "Stardew Valley friendship hearts screen",
        ],
        "queries_yt": [
            "Stardew Valley gameplay",
            "Stardew Valley farm tour",
            "Stardew Valley guide 2024",
            "Stardew Valley tips beginner",
            "Stardew Valley let's play",
            "Stardew Valley perfection run",
            "Stardew Valley expanded mod gameplay",
            "Stardew Valley 1.6 update",
            "Stardew Valley fishing guide",
            "Stardew Valley multiplayer",
            "Stardew Valley speedrun",
            "Stardew Valley year 1 challenge",
            "Stardew Valley 2025 gameplay",
            "Stardew Valley 1.6 new features",
            "Stardew Valley modded playthrough",
            "Stardew Valley skull cavern guide",
            "Stardew Valley co-op gameplay",
            "Stardew Valley min max guide",
            "Stardew Valley jas penny maru romance",
            "Stardew Valley prairie king gameplay",
        ],
    },
    {
        "name": "Skyrim",
        "slug": "skyrim",
        "target": 2000,
        "queries_img": [
            "Skyrim in game screenshot HUD",
            "Skyrim gameplay first person",
            "Skyrim character stats screen",
            "Skyrim dragon fight in game",
            "Skyrim inventory menu open",
            "Skyrim modded gameplay screenshot",
            "Skyrim dungeon combat",
            "site:reddit.com Skyrim screenshot",
            "site:reddit.com Skyrim gameplay",
            "site:reddit.com Skyrim character",
            "Skyrim skill tree level up",
            "Skyrim sneak archer gameplay",
            "Skyrim shout cooldown HUD",
            "Skyrim follower companion gameplay",
            "Skyrim alchemy crafting screen",
            "Skyrim smithing enchanting bench",
            "Skyrim civil war battle screenshot",
            "Skyrim vampire lord transformation",
            "Skyrim werewolf gameplay screenshot",
            "Skyrim darkbrotherhood gameplay",
            "site:reddit.com r/skyrim screenshot character",
            "site:reddit.com r/skyrim modded",
            "Skyrim anniversary edition 2024 screenshot",
            "Skyrim 2025 modded gameplay",
            "Skyrim VR gameplay screenshot",
            "Skyrim together multiplayer screenshot",
            "Skyrim requiem screenshot HUD",
            "Skyrim ordinator perks gameplay",
        ],
        "queries_yt": [
            "Skyrim gameplay walkthrough",
            "Skyrim build guide 2024",
            "Skyrim modded playthrough",
            "Skyrim let's play",
            "Skyrim exploration gameplay",
            "Skyrim challenge run",
            "Skyrim stealth archer gameplay",
            "Skyrim anniversary edition gameplay",
            "Skyrim mage build playthrough",
            "Skyrim survival mode gameplay",
            "Skyrim requiem playthrough",
            "Skyrim speedrun",
            "Skyrim 2025 modded gameplay",
            "Skyrim god run challenge",
            "Skyrim master difficulty gameplay",
            "Skyrim spellsword build guide",
            "Skyrim together multiplayer",
            "Skyrim pure mage playthrough",
            "Skyrim thieves guild full playthrough",
            "Skyrim dragonborn DLC gameplay",
        ],
    },
]

# ---------------------------------------------------------------------------
# Concurrency Limits (see §5.3)
# ---------------------------------------------------------------------------

BROWSER_CONCURRENCY = 4
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

ENRICHMENT_BATCH_SIZE = 1              # sequential: one record at a time
ENRICHMENT_RECORD_TIMEOUT = 200        # Phase 1 + Phase 2 run in parallel; bound by slower path
ENRICHMENT_BROWSER_TIMEOUT = 120       # 2 min for article/channel extraction
ENRICHMENT_LLM_TIMEOUT = 60            # 1 min for experience assessment

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
LLM_MODEL = "ministral-3:14b"          # fast model — used during image collection
ENRICHMENT_LLM_MODEL = "gemma4:26b"   # quality model — used during enrichment pass
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
