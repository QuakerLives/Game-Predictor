# Game Predictor

Gemma 4 26B agentic web scraper & DuckDB pipeline for building a video game
classification training dataset (CNN / NN / Transformer ensemble).

Targets **1 000 fully-populated records** (200 per game) across five titles:
Stellaris, No Man's Sky, Apex Legends, Stardew Valley, and Skyrim.

## Project Structure

```
Game-Predictor/
├── src/game_predictor/          # Installable Python package
│   ├── config.py                # Game definitions, sentinels, concurrency limits
│   ├── models.py                # Pydantic data models (ImageResult, ArticleData, …)
│   ├── prompts.py               # Gemma 4 prompt templates
│   ├── agent.py                 # Per-game orchestrator (Google → YouTube → supplementary)
│   ├── tools/                   # Scraping & processing tools
│   │   ├── search.py            # Google/Bing Images + YouTube search
│   │   ├── extract.py           # Article text extraction via Gemma 4
│   │   ├── screenshot.py        # YouTube screenshot capture + channel info
│   │   ├── download.py          # Async image downloader with validation
│   │   ├── narrate.py           # Semantically-independent narration generation
│   │   ├── assess.py            # Multimodal experience assessment (Ollama native)
│   │   └── database.py          # DuckDB schema, insertion (Iron Rule gates), validation
│   └── cli/                     # Command-line entry points
│       ├── test_run.py          # Trial run — 5 records/game, 25 total
│       ├── run_production.py    # Production — 200 records/game, 1 000 total
│       ├── pre_flight.py        # Pre-flight system checks
│       └── sleeper.py           # Gemma 4-powered watchdog for overnight autonomy
├── scripts/                     # Standalone utility scripts
│   ├── extract_from_steam.py    # Steam API → DuckDB helper
│   └── test.py                  # Polars schema prototype
├── tests/                       # Test suite (future)
├── docs/
│   └── design.md                # Full design document (§1–§18)
├── pyproject.toml               # Package metadata, deps, console scripts
└── .gitignore
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Install Playwright browser
uv run playwright install chromium

# 3. Ensure Ollama is serving Gemma 4
ollama serve &
ollama pull gemma4:26b

# 4. Pre-flight check
uv run game-predictor-preflight

# 5. Trial run (always run first — validates every tool)
uv run game-predictor-test --llm-base-url http://localhost:11434/v1

# 6. Production run (overnight, via sleeper watchdog)
caffeinate -s uv run game-predictor-sleeper --llm-base-url http://localhost:11434/v1
```

## Console Scripts

| Command | Description |
|---|---|
| `game-predictor-preflight` | Checks disk, LLM, Playwright, memory, internet |
| `game-predictor-test` | Trial run: 5 records × 5 games = 25 total |
| `game-predictor-run` | Production: 200 records × 5 games = 1 000 total |
| `game-predictor-sleeper` | Watchdog: spawns production, monitors, auto-restarts |

## Design

See [`docs/design.md`](docs/design.md) for the full 2 900-line specification covering
schema, tool definitions, prompt templates, time budget, error handling, and the
sleeper agent architecture.
