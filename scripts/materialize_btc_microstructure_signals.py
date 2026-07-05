#!/usr/bin/env python3
"""Materialize BTC microstructure signal features into DuckDB."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_btc_microstructure_dashboard import (  # noqa: E402
    DEFAULT_ALGO_VERSION,
    DEFAULT_DUCKDB_PATH,
    build_snapshot,
    persist_snapshot_features,
)


LOGGER = logging.getLogger("materialize_btc_microstructure_signals")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write BTC microstructure signal features to DuckDB")
    parser.add_argument("--minutes", type=int, default=360, help="Recent minutes to recalculate")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DUCKDB_PATH)
    parser.add_argument("--algo-version", default=DEFAULT_ALGO_VERSION)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    snapshot = build_snapshot(args.minutes, algo_version=args.algo_version, live_only=False)
    written = persist_snapshot_features(
        snapshot,
        db_path=args.db_path,
        algo_version=args.algo_version,
    )
    LOGGER.info("wrote %s signal feature rows to %s", written, args.db_path)


if __name__ == "__main__":
    main()
