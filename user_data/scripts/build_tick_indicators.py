# -*- coding: utf-8 -*-
"""
Build Freqtrade-ready candles and microstructure indicators from Binance aggTrades.

Input:
    user_data/tickdata/BTCUSDT-aggTrades-*.csv

Default outputs:
    user_data/data/binance/BTC_USDT-1m.feather
    user_data/data/binance/BTC_USDT-5m.feather
    user_data/data/binance/BTC_USDT-15m.feather
    user_data/data/binance/BTC_USDT-1h.feather

The raw CSV files are large. This script parses them once into 1m aggregate
statistics, then rolls those aggregates up to the requested higher timeframes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


TICK_DIR = Path("user_data/tickdata")
OUT_DIR = Path("user_data/data/binance")
PAIR = "BTC_USDT"
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h"]
CHUNKSIZE = 3_000_000
LARGE_TRADE_USD = 5_000

RAW_COLUMNS = ["datetime", "trade_id", "price", "volume", "BS"]
OUTPUT_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "buy_volume_ratio",
    "price_std",
    "trade_intensity",
    "large_trade_ratio",
    "vwap",
    "vwap_deviation",
]


def timeframe_to_minutes(timeframe: str) -> int:
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc

    if value <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def parse_timeframes(timeframes: list[str]) -> list[str]:
    parsed = []
    seen = set()
    for timeframe in timeframes:
        timeframe_to_minutes(timeframe)
        if timeframe not in seen:
            parsed.append(timeframe)
            seen.add(timeframe)
    return parsed


def output_path(output_dir: Path, pair: str, timeframe: str) -> Path:
    return output_dir / f"{pair}-{timeframe}.feather"


def aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["date"] = pd.to_datetime(chunk["datetime"], unit="us", utc=True).dt.floor("1min")
    chunk["is_buy"] = chunk["BS"] == "B"
    chunk["buy_volume"] = np.where(chunk["is_buy"], chunk["volume"], 0.0)
    chunk["sell_volume"] = np.where(chunk["is_buy"], 0.0, chunk["volume"])
    chunk["usd_volume"] = chunk["price"] * chunk["volume"]
    chunk["large_volume"] = np.where(chunk["usd_volume"] >= LARGE_TRADE_USD, chunk["volume"], 0.0)
    chunk["price_sum"] = chunk["price"]
    chunk["price_sq_sum"] = chunk["price"] ** 2
    chunk["price_count"] = 1

    return (
        chunk.groupby("date", sort=False)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_id", "count"),
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
            usd_volume=("usd_volume", "sum"),
            large_volume=("large_volume", "sum"),
            price_sum=("price_sum", "sum"),
            price_sq_sum=("price_sq_sum", "sum"),
            price_count=("price_count", "sum"),
        )
        .reset_index()
    )


def merge_aggregates(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    merged = (
        combined.groupby("date", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_count", "sum"),
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
            usd_volume=("usd_volume", "sum"),
            large_volume=("large_volume", "sum"),
            price_sum=("price_sum", "sum"),
            price_sq_sum=("price_sq_sum", "sum"),
            price_count=("price_count", "sum"),
        )
        .reset_index()
    )
    return merged.sort_values("date").reset_index(drop=True)


def aggregate_file_to_1m(filepath: Path, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    print(f"  Reading {filepath.name} ({filepath.stat().st_size / 1e9:.1f} GB)")
    start = time.time()
    total_rows = 0
    parts = []

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            filepath,
            usecols=RAW_COLUMNS,
            dtype={"price": "float64", "volume": "float64", "BS": "category"},
            chunksize=chunksize,
        )
    ):
        total_rows += len(chunk)
        if chunk_index % 5 == 0:
            print(f"    chunk {chunk_index}: {total_rows / 1e6:.1f}M rows")
        parts.append(aggregate_chunk(chunk))

    result = merge_aggregates(parts)
    print(f"    produced {len(result):,} 1m rows from {total_rows / 1e6:.1f}M rows in {time.time() - start:.0f}s")
    return result


def rollup_aggregates(base_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    minutes = timeframe_to_minutes(timeframe)
    freq = f"{minutes}min"
    data = base_1m.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data["window"] = data["date"].dt.floor(freq)

    rolled = (
        data.groupby("window", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_count", "sum"),
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
            usd_volume=("usd_volume", "sum"),
            large_volume=("large_volume", "sum"),
            price_sum=("price_sum", "sum"),
            price_sq_sum=("price_sq_sum", "sum"),
            price_count=("price_count", "sum"),
        )
        .reset_index()
        .rename(columns={"window": "date"})
    )
    return finalize_indicators(rolled, minutes)


def finalize_indicators(data: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    total_volume = data["buy_volume"] + data["sell_volume"]
    mean_price = data["price_sum"] / data["price_count"].replace(0, np.nan)
    mean_sq = data["price_sq_sum"] / data["price_count"].replace(0, np.nan)

    result = data.copy()
    result["buy_volume_ratio"] = result["buy_volume"] / total_volume.replace(0, np.nan)
    result["price_std"] = np.sqrt(np.maximum(0, mean_sq - mean_price**2))
    result["trade_intensity"] = result["trade_count"] / (timeframe_minutes * 60)
    result["large_trade_ratio"] = result["large_volume"] / total_volume.replace(0, np.nan)
    result["vwap"] = result["usd_volume"] / total_volume.replace(0, np.nan)
    result["vwap_deviation"] = (result["close"] - result["vwap"]) / result["vwap"].replace(0, np.nan)
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.as_unit("ms")

    result = result.sort_values("date").reset_index(drop=True)
    return result[OUTPUT_COLUMNS].fillna(0.0)


def validate_output(data: pd.DataFrame, timeframe: str) -> dict[str, object]:
    minutes = timeframe_to_minutes(timeframe)
    gaps = 0
    if len(data) > 1:
        gaps = int((data["date"].sort_values().diff().dropna() != pd.Timedelta(minutes=minutes)).sum())
    return {
        "rows": len(data),
        "start": data["date"].min() if len(data) else None,
        "end": data["date"].max() if len(data) else None,
        "duplicate_dates": int(data["date"].duplicated().sum()) if len(data) else 0,
        "non_standard_gaps": gaps,
    }


def build_tick_indicators(
    *,
    files: list[Path],
    output_dir: Path = OUT_DIR,
    timeframes: list[str] | None = None,
    pair: str = PAIR,
    chunksize: int = CHUNKSIZE,
) -> dict[str, Path]:
    selected_timeframes = parse_timeframes(timeframes or DEFAULT_TIMEFRAMES)
    existing_files = [Path(file) for file in files if Path(file).exists()]
    if not existing_files:
        raise FileNotFoundError("No input CSV files found.")

    month_parts = [aggregate_file_to_1m(file, chunksize=chunksize) for file in existing_files]
    base_1m = merge_aggregates([part for part in month_parts if len(part) > 0])
    if len(base_1m) == 0:
        raise ValueError("Input files did not contain usable tick rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for timeframe in selected_timeframes:
        candles = rollup_aggregates(base_1m, timeframe)
        path = output_path(output_dir, pair, timeframe)
        candles.to_feather(path)
        stats = validate_output(candles, timeframe)
        print(
            f"  {timeframe}: {stats['rows']:,} rows, {stats['start']} -> {stats['end']}, "
            f"duplicates={stats['duplicate_dates']}, non_standard_gaps={stats['non_standard_gaps']}, "
            f"size={path.stat().st_size / 1e6:.1f} MB"
        )
        outputs[timeframe] = path

    return outputs


def collect_files(months: list[str] | None) -> tuple[list[Path], list[Path]]:
    if months:
        files = [TICK_DIR / f"BTCUSDT-aggTrades-{month}.csv" for month in months]
    else:
        files = sorted(TICK_DIR.glob("BTCUSDT-aggTrades-*.csv"))
    existing = [file for file in files if file.exists()]
    missing = [file for file in files if not file.exists()]
    return existing, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-timeframe OHLCV and microstructure indicators from Binance aggTrades."
    )
    parser.add_argument("--months", nargs="+", help="Limit input months, for example: 2026-04 2026-05")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help="Output timeframes. Default: 1m 5m 15m 1h",
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory for feather files.")
    parser.add_argument("--pair", default=PAIR, help="Freqtrade pair name used in output filenames.")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="CSV rows per read chunk.")
    parser.add_argument(
        "--output",
        help="Legacy single-output path. Allowed only when exactly one timeframe is requested.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = parse_timeframes(args.timeframes)
    if args.output and len(timeframes) != 1:
        print("--output can only be used with a single --timeframes value.", file=sys.stderr)
        sys.exit(2)

    existing, missing = collect_files(args.months)
    if missing:
        print(f"Warning: missing files: {[file.name for file in missing]}")
    if not existing:
        print("Error: no input CSV files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Input directory: {TICK_DIR}")
    print(f"Input files: {len(existing)}")
    print(f"Timeframes: {' '.join(timeframes)}")
    print(f"Output directory: {args.output_dir}")
    print()

    outputs = build_tick_indicators(
        files=existing,
        output_dir=Path(args.output_dir),
        timeframes=timeframes,
        pair=args.pair,
        chunksize=args.chunksize,
    )

    if args.output:
        only_timeframe = timeframes[0]
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(outputs[only_timeframe].read_bytes())
        print(f"Copied {only_timeframe} output to legacy path: {target}")


if __name__ == "__main__":
    main()
