#!/bin/bash
# Clears macOS UF_HIDDEN flag on .venv contents.
#
# Why: when .venv lives under ~/Documents/ (iCloud-synced), macOS
# applies the UF_HIDDEN flag to dot-prefixed dirs and their contents.
# Python's site.addpackage() silently skips .pth files with that flag,
# breaking editable installs (ModuleNotFoundError on the local package).
#
# Run this any time `uv run` or `python -m game_predictor.*` fails with
# ModuleNotFoundError.

set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "fix_venv: .venv does not exist — run 'uv sync' first."
    exit 1
fi

before=$(find .venv -flags hidden 2>/dev/null | wc -l | tr -d ' ')
chflags -R nohidden .venv 2>/dev/null || true
after=$(find .venv -flags hidden 2>/dev/null | wc -l | tr -d ' ')

echo "fix_venv: cleared UF_HIDDEN on $((before - after)) entries (was $before, now $after)"

# Verify the editable install is reachable
if .venv/bin/python3 -c "import game_predictor" 2>/dev/null; then
    echo "fix_venv: game_predictor module importable ✓"
else
    echo "fix_venv: WARNING — game_predictor still not importable. Try: uv sync"
    exit 2
fi
