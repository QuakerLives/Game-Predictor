# Game Predictor

Agentic data collection pipeline and multi-modal ensemble classifier for identifying
video games from gameplay screenshots and text metadata.

Classifies across five titles: **Apex Legends, No Man's Sky, Skyrim, Stardew Valley, Stellaris**

---

## Architecture

```
Gameplay Screenshot ──► CNN (EfficientNet-B0) ──────────────────┐
                                                                 ├──► EnsembleCombiner ──► Prediction
Gameplay Text       ──► Transformer (all-MiniLM-L6-v2 + FC) ───┘
```

The ensemble combines per-class softmax probabilities from each model using a
learned weighting strategy (L-BFGS-B on validation cross-entropy).

---

## Project Structure

```
Game-Predictor/
├── src/
│   ├── game_predictor/          # Agentic scraper (data collection)
│   │   ├── agent.py             # Per-game LangGraph orchestrator
│   │   ├── tools/               # search, screenshot, narrate, assess, database
│   │   ├── prompts.py           # Gemma 4 prompt templates
│   │   └── cli/                 # test_run, run_production, pre_flight, sleeper
│   │
│   ├── game_cnn/                # Image classifier
│   │   ├── data.py              # Blur detection, pHash dedup, stratified split, DataLoaders
│   │   ├── model.py             # EfficientNet-B0 with replaced classifier head
│   │   ├── pipeline.py          # 2-phase training (warm-up + fine-tune), early stopping
│   │   └── inspect.py           # Preprocessing report (blurry/duplicate CSVs)
│   │
│   ├── game_transformer/        # Text embedding classifier
│   │   ├── data.py              # DB → sanitize → SentenceTransformer embed → split
│   │   └── pipeline.py          # Trains FC classifier on fixed embeddings
│   │
│   ├── game_nn/                 # Steam metadata classifier
│   │   ├── data.py              # Impute, standardize, PCA, KMeans cluster feature
│   │   ├── model.py             # FC neural network (GameClassifier)
│   │   └── pipeline.py          # Training loop with early stopping
│   │
│   └── ensemble/
│       ├── combiner.py          # EnsembleCombiner (average / weighted / learned)
│       └── run.py               # Loads saved model outputs, reports combined accuracy
│
├── data/                        # DuckDB database + images (gitignored)
├── models/                      # Saved model weights + test outputs (gitignored)
├── scripts/                     # Standalone utility scripts
├── docs/
│   └── design.md                # Full design document
├── pyproject.toml
└── .gitignore
```

---

## Setup

```bash
# Install base + all ML extras
uv sync --extra cnn --extra transformer

# Install Playwright browser (for data collection only)
uv run playwright install chromium

# Ensure Ollama is serving Gemma 4 (for data collection only)
ollama serve &
ollama pull gemma4:26b
```

---

## Training the Models

Run in order — each step produces files consumed by the next.

```bash
# 1. Train the image CNN (saves models/cnn.pt + models/shared_split.npz)
uv run game-cnn-train

# 2. Train the text transformer (uses shared_split.npz to align test sets)
uv run game-transformer-train

# 3. Evaluate the ensemble
uv run game-ensemble
```

Model weights and test outputs are saved to `models/` (gitignored — retrain locally).

---

## Data Collection

```bash
# Pre-flight check (LLM, Playwright, disk, memory)
uv run game-predictor-preflight

# Trial run: 5 records × 5 games (always run first)
uv run game-predictor-test --llm-base-url http://localhost:11434/v1

# Production run (overnight via watchdog)
caffeinate -s uv run game-predictor-sleeper --llm-base-url http://localhost:11434/v1
```

---

## Console Scripts

| Command | Description |
|---|---|
| `game-cnn-train` | Train EfficientNet-B0 image classifier |
| `game-transformer-train` | Train text embedding classifier |
| `game-ensemble` | Evaluate ensemble on shared test set |
| `game-nn-train` | Train Steam metadata classifier |
| `game-predictor-preflight` | System checks before data collection |
| `game-predictor-test` | Trial scrape: 5 records × 5 games |
| `game-predictor-run` | Production scrape: 200 records × 5 games |
| `game-predictor-sleeper` | Overnight watchdog for production run |

---

## Data

`data/gameplay_data.duckdb` — ~1,002 records (~200 per game)

Key columns: `id`, `video_game_name`, `image_path`, `gameplay_narration`,
`channel_description`, `identifying_quotes`, `source_url`

Images live at `data/images/<game_slug>/<id>.png` (gitignored).

See [`docs/design.md`](docs/design.md) for the full specification covering schema,
tool definitions, prompt templates, and the scraper architecture.
