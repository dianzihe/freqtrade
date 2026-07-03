# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data" / "l2data" / "coinapi" / "binance" / "BTCUSDT"
MODULE_PATH = ROOT / "user_data" / "strategies" / "微结构_信号模块.py"
OUTPUT_DIR = ROOT / "user_data" / "backtest_results"
LEVELS = 25


def load_signal_module():
    spec = importlib.util.spec_from_file_location("microstructure_signal_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import signal module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hour(path: Path) -> pd.Timestamp:
    match = re.search(r"_(\d{10})\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse hour from filename: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)


def parse_exchange_seconds(values: pd.Series) -> np.ndarray:
    parts = values.astype(str).str.extract(r"^(\d{2}):(\d{2}):(\d{2})", expand=True).astype("int64")
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def top_levels(book: dict[float, float], reverse: bool) -> list[list[float]]:
    return [[float(p), float(s)] for p, s in sorted(book.items(), reverse=reverse)[:LEVELS]]


def build_record(
    minute_key: int,
    base_day: pd.Timestamp,
    open_price: float,
    high: float,
    low: float,
    close: float,
    bid_book: dict[float, float],
    ask_book: dict[float, float],
    bid_add: float,
    bid_del: float,
    ask_add: float,
    ask_del: float,
    updates: int,
) -> dict | None:
    if not bid_book or not ask_book:
        return None
    bids = top_levels(bid_book, reverse=True)
    asks = top_levels(ask_book, reverse=False)
    if not bids or not asks:
        return None

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    bid_depth = sum(level[1] for level in bids)
    ask_depth = sum(level[1] for level in asks)

    # Positive flow means bid-side liquidity improved or ask-side liquidity was pulled.
    buy_volume = bid_add + ask_del
    sell_volume = ask_add + bid_del
    volume = max(buy_volume + sell_volume, float(updates), 1.0)

    return {
        "date": base_day + pd.Timedelta(minutes=int(minute_key)),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "bids": bids,
        "asks": asks,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth_25": bid_depth,
        "ask_depth_25": ask_depth,
        "l2_updates": updates,
    }


def reconstruct_minutes(files: list[Path], chunksize: int) -> pd.DataFrame:
    bid_book: dict[float, float] = {}
    ask_book: dict[float, float] = {}
    current_minute: int | None = None
    current_base_day: pd.Timestamp | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    last_snapshot_second: int | None = None
    open_price = high = low = close = np.nan
    bid_add = bid_del = ask_add = ask_del = 0.0
    updates = 0
    records: list[dict] = []

    for path in files:
        base_hour = file_hour(path)
        base_day = base_hour.normalize()
        print(f"reading {path.name}")
        reader = pd.read_csv(
            path,
            sep=";",
            chunksize=chunksize,
            usecols=["time_exchange", "update_type", "is_buy", "entry_px", "entry_sx"],
        )
        for chunk in reader:
            seconds = parse_exchange_seconds(chunk["time_exchange"])
            minute_keys = seconds // 60
            update_types = chunk["update_type"].astype(str).str.upper().to_numpy()
            sides = chunk["is_buy"].to_numpy()
            prices = chunk["entry_px"].to_numpy(dtype="float64")
            sizes = chunk["entry_sx"].to_numpy(dtype="float64")

            for second_key, minute_key, update_type, side, price, size in zip(
                seconds, minute_keys, update_types, sides, prices, sizes
            ):
                if update_type == "SNAPSHOT" and second_key != last_snapshot_second:
                    bid_book.clear()
                    ask_book.clear()
                    best_bid = None
                    best_ask = None
                    last_snapshot_second = second_key

                side_is_bid = int(side) == 1

                book = bid_book if side_is_bid else ask_book
                old_size = book.get(price, 0.0)

                if update_type == "SNAPSHOT":
                    book[price] = size
                elif update_type in {"ADD", "UPDATE", "SET"}:
                    if size > 0:
                        book[price] = size
                    else:
                        book.pop(price, None)
                elif update_type in {"DELETE", "REMOVE", "SUBTRACT"}:
                    book.pop(price, None)
                    size = 0.0
                else:
                    if size > 0:
                        book[price] = size
                    else:
                        book.pop(price, None)

                if side_is_bid:
                    if size > 0 and (best_bid is None or price > best_bid):
                        best_bid = price
                    elif old_size > 0 and size <= 0 and price == best_bid:
                        best_bid = max(bid_book) if bid_book else None
                else:
                    if size > 0 and (best_ask is None or price < best_ask):
                        best_ask = price
                    elif old_size > 0 and size <= 0 and price == best_ask:
                        best_ask = min(ask_book) if ask_book else None

                delta = size - old_size
                if update_type != "SNAPSHOT":
                    if side_is_bid:
                        if delta >= 0:
                            bid_add += delta
                        else:
                            bid_del += -delta
                    else:
                        if delta >= 0:
                            ask_add += delta
                        else:
                            ask_del += -delta

                if best_bid is None or best_ask is None:
                    continue

                mid = (best_bid + best_ask) / 2.0

                if current_minute is None:
                    current_minute = int(minute_key)
                    current_base_day = base_day
                    open_price = high = low = close = mid
                elif int(minute_key) != current_minute:
                    record = build_record(
                        current_minute,
                        current_base_day if current_base_day is not None else base_day,
                        open_price,
                        high,
                        low,
                        close,
                        bid_book,
                        ask_book,
                        bid_add,
                        bid_del,
                        ask_add,
                        ask_del,
                        updates,
                    )
                    if record is not None:
                        records.append(record)
                    current_minute = int(minute_key)
                    current_base_day = base_day
                    open_price = high = low = close = mid
                    bid_add = bid_del = ask_add = ask_del = 0.0
                    updates = 0

                high = max(high, mid)
                low = min(low, mid)
                close = mid
                updates += 1

    if current_minute is not None:
        record = build_record(
            current_minute,
            current_base_day if current_base_day is not None else file_hour(files[-1]).normalize(),
            open_price,
            high,
            low,
            close,
            bid_book,
            ask_book,
            bid_add,
            bid_del,
            ask_add,
            ask_del,
            updates,
        )
        if record is not None:
            records.append(record)

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def add_forward_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_t"] = df["close"]
    for minutes in (1, 5):
        future = df["close"].shift(-minutes)
        highs = df["high"].shift(-1).rolling(minutes, min_periods=minutes).max().shift(-(minutes - 1))
        lows = df["low"].shift(-1).rolling(minutes, min_periods=minutes).min().shift(-(minutes - 1))
        df[f"price_t+{minutes}m"] = future
        df[f"ret_{minutes}m_pct"] = (future / df["close"] - 1.0) * 100.0
        df[f"max_high_{minutes}m"] = highs
        df[f"min_low_{minutes}m"] = lows
        df[f"max_gain_{minutes}m_pct"] = (highs / df["close"] - 1.0) * 100.0
        df[f"max_loss_{minutes}m_pct"] = (lows / df["close"] - 1.0) * 100.0
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("*.csv"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    module = load_signal_module()
    candles = reconstruct_minutes(files, args.chunksize)
    signals_df = module.add_lob_regime_signals(
        candles,
        lookback_period=24,
        volatility_window=20,
        threshold_percentile=88,
        confirmation_bars=2,
        min_signal_strength=0.40,
        signal_valid_bars=25,
    )
    signals_df = add_forward_columns(signals_df)

    result_cols = [
        "date",
        "price_t",
        "ch1_vol_entropy",
        "ch2_depth_erosion",
        "ch3_spread_drift",
        "ch4_order_flow",
        "composite_smooth",
        "adaptive_threshold",
        "price_t+1m",
        "ret_1m_pct",
        "max_high_1m",
        "min_low_1m",
        "price_t+5m",
        "ret_5m_pct",
        "max_high_5m",
        "min_low_5m",
        "max_gain_5m_pct",
        "max_loss_5m_pct",
    ]
    triggered = signals_df.loc[signals_df["signal_trigger"] == 1, result_cols].copy()
    triggered = triggered.rename(
        columns={
            "date": "signal_time",
            "ch1_vol_entropy": "ch1",
            "ch2_depth_erosion": "ch2",
            "ch3_spread_drift": "ch3",
            "ch4_order_flow": "ch4",
            "composite_smooth": "composite",
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candles_path = OUTPUT_DIR / "coinapi_btcusdt_l2_1m_features.csv"
    results_path = OUTPUT_DIR / "coinapi_btcusdt_microstructure_signal_test.csv"
    candles.drop(columns=["bids", "asks"]).to_csv(candles_path, index=False, encoding="utf-8-sig")
    triggered.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\nsummary")
    print(f"files={len(files)}")
    print(f"minutes={len(candles)}")
    print(f"range={candles['date'].min()} -> {candles['date'].max()}")
    print(f"signals={len(triggered)}")
    print(f"features_csv={candles_path}")
    print(f"signals_csv={results_path}")
    if not triggered.empty:
        print(triggered.to_string(index=False, max_rows=200))


if __name__ == "__main__":
    main()
