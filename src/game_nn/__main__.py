"""Entry point: ``python -m game_nn``

Flags:
  --gameplay          Train on gameplay numerical features (for CNN/Transformer/NN ensemble).
                      Requires models/shared_split.npz from a prior game_cnn run.
  --gameplay-db PATH  Path to gameplay DuckDB (default: data/gameplay_data.duckdb).

Without --gameplay, trains on Steam API tabular features (original behaviour).
"""

import argparse
from .pipeline import run, run_on_gameplay_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game-NN classifier")
    parser.add_argument(
        "--gameplay", action="store_true",
        help="Use gameplay records numerical features (for ensemble with CNN + Transformer)",
    )
    parser.add_argument(
        "--gameplay-db", default="data/gameplay_data.duckdb",
        metavar="PATH",
        help="Path to gameplay DuckDB (default: data/gameplay_data.duckdb)",
    )
    args = parser.parse_args()

    if args.gameplay:
        run_on_gameplay_data(gameplay_db_path=args.gameplay_db)
    else:
        run()
