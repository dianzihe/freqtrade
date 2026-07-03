#!/usr/bin/env python3
"""Normalize Gate spot L1/L2 order book parquet data for replay-style backtests.

Raw L2 files are Gate OBU messages: an initial full snapshot followed by
incremental updates. This script replays those updates in sequence and writes
time-bucketed book features that are cheap to load in later strategy tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback for minimal environments
    orjson = None


DEFAULT_DATA_ROOT = Path("user_data/orderbook_data/gate/spot")
DEFAULT_OUTPUT_ROOT = Path("user_data/orderbook_data/formatted_v2")
DEFAULT_TIMEFRAMES = ("1s", "5s", "10s", "30s", "1min", "5min", "15min", "1h")


def parse_levels(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        try:
            value = orjson.loads(value) if orjson is not None else json.loads(value)
        except (json.JSONDecodeError, orjson.JSONDecodeError if orjson is not None else ValueError):
            return []
    if not isinstance(value, list):
        return []

    levels: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            price = float(item[0])
            amount = float(item[1])
        except (TypeError, ValueError):
            continue
        levels.append([price, amount])
    return levels


class L2Replayer:
    def __init__(self, levels: int = 25) -> None:
        self.levels = levels
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int | None = None
        self.sequence_gap_count = 0
        self.snapshot_count = 0
        self.update_count = 0

    def apply(self, row: dict[str, Any] | pd.Series) -> dict[str, Any]:
        update_info = self.apply_update(row)
        return self._feature_row(
            exchange_time_ms=update_info["exchange_time_ms"],
            first_update_id=update_info["first_update_id"],
            update_id=update_info["update_id"],
            is_snapshot=update_info["is_snapshot"],
            sequence_gap=update_info["sequence_gap"],
        )

    def apply_update(self, row: dict[str, Any] | pd.Series) -> dict[str, Any]:
        first_update_id = int(row["first_update_id"])
        update_id = int(row["update_id"])
        bid_updates = parse_levels(row["bid_updates"])
        ask_updates = parse_levels(row["ask_updates"])
        snapshot = self._is_full_snapshot(first_update_id, update_id, bid_updates, ask_updates)
        sequence_gap = False

        if self.last_update_id is not None and first_update_id > self.last_update_id + 1:
            sequence_gap = True
            self.sequence_gap_count += 1

        if snapshot:
            self.bids.clear()
            self.asks.clear()
            self.snapshot_count += 1

        self._apply_side(self.bids, bid_updates)
        self._apply_side(self.asks, ask_updates)
        self.last_update_id = max(update_id, self.last_update_id or update_id)
        self.update_count += 1

        return {
            "exchange_time_ms": int(row["exchange_time_ms"]),
            "first_update_id": first_update_id,
            "update_id": update_id,
            "is_snapshot": snapshot,
            "sequence_gap": sequence_gap,
        }

    def _is_full_snapshot(
        self,
        first_update_id: int,
        update_id: int,
        bid_updates: list[list[float]],
        ask_updates: list[list[float]],
    ) -> bool:
        return (
            first_update_id == update_id
            and len(bid_updates) >= self.levels
            and len(ask_updates) >= self.levels
        )

    @staticmethod
    def _apply_side(book: dict[float, float], updates: list[list[float]]) -> None:
        for price, amount in updates:
            if amount <= 0:
                book.pop(price, None)
            else:
                book[price] = amount

    def _top_bids(self) -> list[list[float]]:
        return [[price, amount] for price, amount in sorted(self.bids.items(), reverse=True)[: self.levels]]

    def _top_asks(self) -> list[list[float]]:
        return [[price, amount] for price, amount in sorted(self.asks.items())[: self.levels]]

    def _feature_row(
        self,
        *,
        exchange_time_ms: int,
        first_update_id: int,
        update_id: int,
        is_snapshot: bool,
        sequence_gap: bool,
    ) -> dict[str, Any]:
        bids = self._top_bids()
        asks = self._top_asks()
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
        spread_bps = (spread / mid_price) * 10_000 if spread is not None and mid_price else None

        bid_depth_5 = sum(amount for _, amount in bids[:5])
        ask_depth_5 = sum(amount for _, amount in asks[:5])
        bid_depth_10 = sum(amount for _, amount in bids[:10])
        ask_depth_10 = sum(amount for _, amount in asks[:10])
        bid_depth_25 = sum(amount for _, amount in bids[:25])
        ask_depth_25 = sum(amount for _, amount in asks[:25])
        total_depth = bid_depth_25 + ask_depth_25
        imbalance_25 = (bid_depth_25 - ask_depth_25) / total_depth if total_depth else None

        return {
            "exchange_time_ms": exchange_time_ms,
            "first_update_id": first_update_id,
            "update_id": update_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": spread_bps,
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "bid_depth_10": bid_depth_10,
            "ask_depth_10": ask_depth_10,
            "bid_depth_25": bid_depth_25,
            "ask_depth_25": ask_depth_25,
            "imbalance_25": imbalance_25,
            "is_snapshot": is_snapshot,
            "sequence_gap": sequence_gap,
            "bids": bids,
            "asks": asks,
        }


def iter_readable_parquet_files(path: Path) -> tuple[list[Path], list[str]]:
    files = sorted(path.rglob("*.parquet"))
    readable: list[Path] = []
    bad: list[str] = []
    for file_path in files:
        try:
            pd.read_parquet(file_path, columns=["exchange_time_ms"])
        except Exception as exc:
            bad.append(f"{file_path}: {exc}")
            continue
        readable.append(file_path)
    return readable, bad


def load_l1_ticks(pair_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    files, bad = iter_readable_parquet_files(pair_dir / "l1")
    frames: list[pd.DataFrame] = []
    for file_path in files:
        frame = pd.read_parquet(file_path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), bad

    ticks = pd.concat(frames, ignore_index=True)
    ticks = ticks.sort_values(["exchange_time_ms", "update_id"]).drop_duplicates("exchange_time_ms")
    ticks["date"] = pd.to_datetime(ticks["exchange_time_ms"], unit="ms", utc=True)
    ticks["spread_bps"] = (ticks["spread"] / ticks["mid_price"]) * 10_000
    return ticks, bad


def build_l1_ohlcv(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    indexed = ticks.set_index("date").sort_index()
    price = indexed["mid_price"]
    ohlcv = price.resample(timeframe).ohlc().dropna()
    ohlcv["volume"] = (indexed["bid_amount"] + indexed["ask_amount"]).resample(timeframe).sum()
    ohlcv["avg_spread"] = indexed["spread"].resample(timeframe).mean()
    ohlcv["max_spread"] = indexed["spread"].resample(timeframe).max()
    ohlcv["avg_spread_bps"] = indexed["spread_bps"].resample(timeframe).mean()
    ohlcv["tick_count"] = price.resample(timeframe).count()
    return ohlcv.reset_index()


def rebuild_l2_features(pair_dir: Path, levels: int) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    files, bad = iter_readable_parquet_files(pair_dir / "l2")
    replayer = L2Replayer(levels=levels)
    rows: list[dict[str, Any]] = []

    for file_path in files:
        frame = pd.read_parquet(
            file_path,
            columns=[
                "exchange_time_ms",
                "first_update_id",
                "update_id",
                "bid_updates",
                "ask_updates",
            ],
        )
        frame = frame.sort_values(["exchange_time_ms", "first_update_id", "update_id"])
        for row in frame.to_dict("records"):
            feature = replayer.apply(row)
            if feature["best_bid"] is not None and feature["best_ask"] is not None:
                rows.append(feature)

    features = pd.DataFrame(rows)
    stats = {
        "readable_files": len(files),
        "bad_files": bad,
        "raw_updates": replayer.update_count,
        "snapshots": replayer.snapshot_count,
        "sequence_gaps": replayer.sequence_gap_count,
    }
    return features, stats, bad


def rebuild_l2_1s(pair_dir: Path, levels: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    files, bad = iter_readable_parquet_files(pair_dir / "l2")
    replayer = L2Replayer(levels=levels)
    rows: list[dict[str, Any]] = []
    current_second: int | None = None
    current_meta: dict[str, Any] | None = None
    update_count = 0
    snapshot_count = 0
    sequence_gap_count = 0

    def flush_current() -> None:
        if current_second is None or current_meta is None:
            return
        feature = replayer._feature_row(
            exchange_time_ms=current_meta["exchange_time_ms"],
            first_update_id=current_meta["first_update_id"],
            update_id=current_meta["update_id"],
            is_snapshot=snapshot_count > 0,
            sequence_gap=sequence_gap_count > 0,
        )
        if feature["best_bid"] is None or feature["best_ask"] is None:
            return
        feature["date"] = pd.to_datetime(current_second * 1000, unit="ms", utc=True)
        feature["update_count"] = update_count
        feature["snapshot_count"] = snapshot_count
        feature["sequence_gap_count"] = sequence_gap_count
        rows.append(feature)

    for file_path in files:
        frame = pd.read_parquet(
            file_path,
            columns=[
                "exchange_time_ms",
                "first_update_id",
                "update_id",
                "bid_updates",
                "ask_updates",
            ],
        )
        frame = frame.sort_values(["exchange_time_ms", "first_update_id", "update_id"])
        for row in frame.to_dict("records"):
            update_info = replayer.apply_update(row)
            second = update_info["exchange_time_ms"] // 1000
            if current_second is None:
                current_second = second
            elif second != current_second:
                flush_current()
                current_second = second
                update_count = 0
                snapshot_count = 0
                sequence_gap_count = 0

            update_count += 1
            snapshot_count += int(update_info["is_snapshot"])
            sequence_gap_count += int(update_info["sequence_gap"])
            current_meta = update_info

    flush_current()

    l2_1s = pd.DataFrame(rows)
    stats = {
        "readable_files": len(files),
        "bad_files": bad,
        "raw_updates": replayer.update_count,
        "snapshots": replayer.snapshot_count,
        "sequence_gaps": replayer.sequence_gap_count,
        "one_second_rows": int(len(l2_1s)),
    }
    return l2_1s, stats


def resample_l2_features(features: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if features.empty:
        return features

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["exchange_time_ms"], unit="ms", utc=True)
    indexed = frame.set_index("date").sort_index()
    desired_last_cols = [
        "exchange_time_ms",
        "first_update_id",
        "update_id",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "spread_bps",
        "bid_depth_5",
        "ask_depth_5",
        "bid_depth_10",
        "ask_depth_10",
        "bid_depth_25",
        "ask_depth_25",
        "imbalance_25",
        "bids",
        "asks",
    ]
    last_cols = [column for column in desired_last_cols if column in indexed.columns]
    result = indexed[last_cols].resample(timeframe).last().dropna(subset=["mid_price"])
    count_column = "update_id" if "update_id" in indexed else "exchange_time_ms"
    result["update_count"] = indexed[count_column].resample(timeframe).count()
    result["snapshot_count"] = indexed["is_snapshot"].resample(timeframe).sum() if "is_snapshot" in indexed else 0
    result["sequence_gap_count"] = indexed["sequence_gap"].resample(timeframe).sum()
    return result.reset_index()


def build_l2_ohlcv(l2_1s: pd.DataFrame) -> pd.DataFrame:
    if l2_1s.empty:
        return l2_1s
    indexed = l2_1s.set_index("date").sort_index()
    price = indexed["mid_price"]
    out = price.resample("1min").ohlc().dropna()
    out["volume"] = (indexed["bid_depth_25"] + indexed["ask_depth_25"]).resample("1min").mean()
    out["best_bid"] = indexed["best_bid"].resample("1min").last()
    out["best_ask"] = indexed["best_ask"].resample("1min").last()
    out["spread_bps_mean"] = indexed["spread_bps"].resample("1min").mean()
    out["spread_bps_max"] = indexed["spread_bps"].resample("1min").max()
    out["bid_depth_25_mean"] = indexed["bid_depth_25"].resample("1min").mean()
    out["bid_depth_25_min"] = indexed["bid_depth_25"].resample("1min").min()
    out["ask_depth_25_mean"] = indexed["ask_depth_25"].resample("1min").mean()
    out["ask_depth_25_min"] = indexed["ask_depth_25"].resample("1min").min()
    out["imbalance_25_mean"] = indexed["imbalance_25"].resample("1min").mean()
    out["sequence_gap_count"] = indexed["sequence_gap_count"].resample("1min").sum()
    out["update_count"] = indexed["update_count"].resample("1min").sum()
    return out.reset_index()


def json_encode_levels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("bids", "asks"):
        if column in out:
            out[column] = out[column].map(lambda value: json.dumps(value, separators=(",", ":")))
    return out


def process_pair(
    pair: str,
    *,
    data_root: Path,
    output_root: Path,
    levels: int,
    timeframes: tuple[str, ...],
) -> dict[str, Any]:
    pair_dir = data_root / pair
    out_dir = output_root / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_ticks, l1_bad = load_l1_ticks(pair_dir)
    if l1_ticks.empty:
        raise RuntimeError(f"No readable L1 data for {pair}")
    l1_ticks.to_parquet(out_dir / "l1_ticks.parquet", index=False)
    for timeframe in timeframes:
        build_l1_ohlcv(l1_ticks, timeframe).to_parquet(out_dir / f"l1_ohlcv_{timeframe}.parquet", index=False)

    l2_1s, l2_stats = rebuild_l2_1s(pair_dir, levels)
    json_encode_levels(l2_1s).to_parquet(out_dir / "l2_book_1s.parquet", index=False)
    build_l2_ohlcv(l2_1s).to_parquet(out_dir / "l2_ohlcv_1min.parquet", index=False)

    start = min(l1_ticks["date"].min(), l2_1s["date"].min()) if not l2_1s.empty else l1_ticks["date"].min()
    end = max(l1_ticks["date"].max(), l2_1s["date"].max()) if not l2_1s.empty else l1_ticks["date"].max()
    manifest = {
        "pair": pair,
        "levels": levels,
        "time_range": {"start": start.isoformat(), "end": end.isoformat()},
        "l1": {
            "rows": int(len(l1_ticks)),
            "bad_files": l1_bad,
        },
        "l2": {
            **l2_stats,
        },
        "outputs": [
            "l1_ticks.parquet",
            *[f"l1_ohlcv_{timeframe}.parquet" for timeframe in timeframes],
            "l2_book_1s.parquet",
            "l2_ohlcv_1min.parquet",
        ],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pair", action="append", default=None, help="Pair such as BTC_USDT. Can be repeated.")
    parser.add_argument("--levels", type=int, default=25)
    parser.add_argument("--timeframe", action="append", default=None, help="L1 OHLCV timeframe. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = args.pair or sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
    timeframes = tuple(args.timeframe or DEFAULT_TIMEFRAMES)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifests = [
        process_pair(
            pair,
            data_root=args.data_root,
            output_root=args.output_root,
            levels=args.levels,
            timeframes=timeframes,
        )
        for pair in pairs
    ]
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"pairs": manifests}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
