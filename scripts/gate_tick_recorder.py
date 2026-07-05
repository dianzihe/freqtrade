#!/usr/bin/env python3
"""Binance full order book tick recorder (default: BTCUSDT, 5000-level depth).

This is the recommended entry point for recording full order book data. Uses Binance
REST snapshot (5000 levels) + diff WebSocket stream to maintain a complete order book.
Output is hourly Parquet files with Hive partitioning under user_data/orderbook_data/.

Supports --exchange gate for Gate.io spot data as well. Use --symbol to change pairs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cex_tick_recorder import *  # noqa: F403
from scripts.cex_tick_recorder import main


if __name__ == "__main__":
    main()
