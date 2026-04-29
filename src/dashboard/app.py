"""
Entry point for the Game Predictor Dash dashboard.

Run with:
    game-dashboard
or directly:
    python -m dashboard.app
"""
from __future__ import annotations

# Must be set before any numpy/pandas/torch import to prevent OpenMP deadlock
# on Windows when PyTorch and pandas are both present in the same environment.
import os
# Prevent OpenMP/BLAS deadlock on Windows with PyTorch CUDA + pandas.
# Multiple runtimes (Intel MKL, OpenBLAS, AMD BLIS) fight over threads on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import dash
import dash_bootstrap_components as dbc

from .layout import build_shell
from . import callbacks  # noqa: F401 — importing registers all @callback decorators

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Game Predictor",
)

# build_shell() is fast (no DB): just navbar + spinner placeholder.
# The real content is loaded by the serve_main_content callback in callbacks.py
# when the browser first connects, so the server starts in < 1 second.
app.layout = build_shell()

server = app.server  # expose Flask server for production WSGI deployment


def main() -> None:
    """Launch the dashboard at http://localhost:8050."""
    print("Starting Game Predictor Dashboard at http://127.0.0.1:8050 ...", flush=True)
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
