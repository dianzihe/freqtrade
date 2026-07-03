"""直接通过 ccxt 同步接口下载 TAIKO/USDT 和 NFP/USDT 的 OHLCV 数据.

绕过 freqtrade download-data 的 aiohttp 代理问题.
"""

import os
import sys

import ccxt
import pandas as pd
from datetime import datetime, timezone

# Gate.io 现货
EXCHANGE = "gate"
PAIRS = ["TAIKO/USDT", "NFP/USDT"]
TIMEFRAMES = ["1m", "5m", "15m", "1h"]
DAYS = 90
OUTPUT_DIR = "user_data/data/gate"
DATA_FORMAT = "feather"

PROXY = "http://127.0.0.1:7890"

os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY


def download_pair_tf(exchange: ccxt.gate, pair: str, tf: str, since_ms: int):
    """下载单个交易对的单个时间框架数据."""
    symbol = pair.replace("/", "_")
    fname = os.path.join(OUTPUT_DIR, f"{symbol}-{tf}.{DATA_FORMAT}")

    all_candles = []
    current_since = since_ms

    while True:
        try:
            candles = exchange.fetch_ohlcv(pair, tf, since=current_since, limit=1000)
        except Exception as e:
            print(f"  [ERROR] fetch_ohlcv {pair} {tf} at {current_since}: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)

        last_ts = candles[-1][0]
        if len(candles) < 1000 or last_ts <= current_since:
            break
        current_since = last_ts + 1

    if not all_candles:
        print(f"  [WARN] {pair} {tf}: no data")
        return

    df = pd.DataFrame(
        all_candles, columns=["date", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 对齐日期索引格式 (freqtrade 使用 Arrow 的 date 列)
    if DATA_FORMAT == "feather":
        df.to_feather(fname)
    elif DATA_FORMAT == "parquet":
        df.to_parquet(fname, index=False)

    first = df["date"].iloc[0]
    last = df["date"].iloc[-1]
    print(f"  [{pair}] {tf}: {len(df)} rows, {first} → {last}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Connecting to {EXCHANGE} via proxy {PROXY} ...")
    exchange_class = getattr(ccxt, EXCHANGE)
    exchange = exchange_class(
        {
            "proxies": {
                "http": PROXY,
                "https": PROXY,
            },
            "enableRateLimit": True,
        }
    )

    # 只加载现货市场
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"[ERROR] Load markets failed: {e}")
        sys.exit(1)

    print(f"Loaded {len(exchange.markets)} markets")

    # 检查交易对是否可用
    for pair in PAIRS:
        if pair not in exchange.markets:
            print(f"[ERROR] {pair} not found in markets!")
            sys.exit(1)
        market = exchange.markets[pair]
        print(f"  {pair}: type={market.get('type')}, active={market.get('active')}")

    base_since = exchange.parse8601(
        (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
         - pd.Timedelta(days=DAYS)).isoformat()
    )

    # Gate API 限制: "Maximum 10000 points ago"
    # 1m: 10000 min ≈ 7 days, 5m: 10000*5min ≈ 35 days
    TF_MAX_DAYS = {"1m": 6, "5m": 30, "15m": 90, "1h": 90}

    for pair in PAIRS:
        print(f"\nDownloading {pair} ...")
        for tf in TIMEFRAMES:
            tf_days = TF_MAX_DAYS.get(tf, 90)
            tf_since = max(
                base_since,
                exchange.parse8601(
                    (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                     - pd.Timedelta(days=tf_days)).isoformat()
                ),
            )
            download_pair_tf(exchange, pair, tf, tf_since)

    print("\nDone!")


if __name__ == "__main__":
    main()
