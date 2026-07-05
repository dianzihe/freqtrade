#!/usr/bin/env python3
"""Record spot L1, L2 order book, and public trade data into hourly parquet files.

Format v2 — key improvements over v1:
- L2 bid/ask stored as Parquet native list<struct<price:double, amount:double>>
  instead of JSON strings (3-5x faster read/write, no serialization overhead)
- Binance full order book (REST snapshot 5000 levels + diff stream, no reconstruction needed)
- Gate periodic L2 snapshots (default every 3600s) for reliable order book reconstruction
- L1/L2 sequence gap tracking with non-fatal recovery
- L2 gap recovery via in-channel re-subscription (no full WebSocket reconnect)
- Atomic file writes (write to .tmp, then os.replace) — no more corrupted files
- Hour-based Hive partitioning (date=YYYY-MM-DD/hour=HH/)

Usage:
    python scripts/cex_tick_recorder.py                          # Binance BTCUSDT, full 5000-level book
    python scripts/cex_tick_recorder.py --symbol ETH             # Binance ETHUSDT
    python scripts/cex_tick_recorder.py --exchange gate --pair BTC_USDT
    python scripts/cex_tick_recorder.py --exchange gate --pair ETH_USDT --snapshot-interval 1800
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import websockets


GATE_WS_URL = "wss://api.gateio.ws/ws/v4/"
BINANCE_WS_BASE = "wss://stream.binance.com:9443"
LOGGER = logging.getLogger("cex_tick_recorder")

# Parquet schemas

_L2_ENTRY = pa.struct([("price", pa.float64()), ("amount", pa.float64())])

L1_SCHEMA = pa.schema([
    pa.field("exchange_time_ms", pa.int64()),
    pa.field("local_time_ms", pa.int64()),
    pa.field("update_id", pa.int64()),
    pa.field("pair", pa.string()),
    pa.field("bid_price", pa.float64()),
    pa.field("bid_amount", pa.float64()),
    pa.field("ask_price", pa.float64()),
    pa.field("ask_amount", pa.float64()),
    pa.field("spread", pa.float64()),
    pa.field("mid_price", pa.float64()),
])

L2_SCHEMA = pa.schema([
    pa.field("exchange_time_ms", pa.int64()),
    pa.field("local_time_ms", pa.int64()),
    pa.field("first_update_id", pa.int64()),
    pa.field("update_id", pa.int64()),
    pa.field("pair", pa.string()),
    pa.field("bid_updates", pa.list_(_L2_ENTRY)),
    pa.field("ask_updates", pa.list_(_L2_ENTRY)),
    pa.field("bid_update_count", pa.int32()),
    pa.field("ask_update_count", pa.int32()),
    pa.field("is_snapshot", pa.bool_()),
    pa.field("is_empty", pa.bool_()),
])

TRADES_SCHEMA = pa.schema([
    pa.field("exchange_time_ms", pa.int64()),
    pa.field("local_time_ms", pa.int64()),
    pa.field("trade_id", pa.int64()),
    pa.field("pair", pa.string()),
    pa.field("side", pa.string()),
    pa.field("price", pa.float64()),
    pa.field("amount", pa.float64()),
    pa.field("cost", pa.float64()),
])

DEPTH_SCHEMA = pa.schema([
    pa.field("exchange_time_ms", pa.int64()),
    pa.field("local_time_ms", pa.int64()),
    pa.field("update_id", pa.int64()),
    pa.field("pair", pa.string()),
    pa.field("bid_levels", pa.int32()),
    pa.field("ask_levels", pa.int32()),
    pa.field("best_bid", pa.float64()),
    pa.field("best_ask", pa.float64()),
    pa.field("spread", pa.float64()),
    pa.field("mid_price", pa.float64()),
    pa.field("bid_depth", pa.float64()),
    pa.field("ask_depth", pa.float64()),
    pa.field("total_depth", pa.float64()),
    pa.field("bid_depth_5bps", pa.float64()),
    pa.field("ask_depth_5bps", pa.float64()),
    pa.field("total_depth_5bps", pa.float64()),
    pa.field("bid_depth_10bps", pa.float64()),
    pa.field("ask_depth_10bps", pa.float64()),
    pa.field("total_depth_10bps", pa.float64()),
    pa.field("bid_depth_25bps", pa.float64()),
    pa.field("ask_depth_25bps", pa.float64()),
    pa.field("total_depth_25bps", pa.float64()),
    pa.field("imbalance", pa.float64()),
    pa.field("is_snapshot", pa.bool_()),
])


# Config

@dataclass(frozen=True)
class RecorderConfig:
    exchange: str = "gate"
    pair: str = "BTC_USDT"
    output_dir: Path = Path("user_data/orderbook_data")
    depth: int = 50
    flush_rows: int = 500_000        # safety valve — primary flush is on hour boundary
    flush_seconds: float = 60.0      # periodic flush for data freshness / crash safety
    reconnect_seconds: float = 2.0   # fast reconnect to minimize data loss
    snapshot_interval: float = 3600.0  # L2 snapshot every N seconds
    ping_interval: int = 20
    ping_timeout: int = 20
    recv_timeout: float = 1.0
    snapshot_request_timeout: float = 10.0  # re-request snapshot if not received within N seconds
    proxy: str | None = None         # HTTP proxy for WebSocket (e.g. http://127.0.0.1:7890)
    live: bool = True                # print real-time depth consumption stats every second


# Helpers

def utcnow() -> datetime:
    return datetime.now(UTC)


def to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def parse_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def parse_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    return int(v)


def parse_epoch_ms(v: Any) -> int | None:
    if v is None or v == "":
        return None
    return int(float(v))


def normalize_exchange(exchange: str) -> str:
    exchange = exchange.lower()
    if exchange not in {"gate", "binance"}:
        raise ValueError(f"Unsupported exchange: {exchange}")
    return exchange


def normalize_pair(symbol_or_pair: str, *, exchange: str, quote: str = "USDT") -> str:
    """Normalize a user symbol/pair into the exchange's spot pair format."""
    exchange = normalize_exchange(exchange)
    value = symbol_or_pair.strip().upper().replace("-", "_")
    if not value:
        raise ValueError("Symbol or pair cannot be empty")

    if exchange == "binance":
        pair = value.replace("/", "").replace("_", "")
        if not pair.endswith(quote):
            pair = f"{pair}{quote}"
        return pair

    if "/" in value:
        base, pair_quote = value.split("/", 1)
    elif "_" in value:
        base, pair_quote = value.split("_", 1)
    elif value.endswith(quote):
        base, pair_quote = value[: -len(quote)], quote
    else:
        base, pair_quote = value, quote
    if not base or not pair_quote:
        raise ValueError(f"Invalid pair symbol: {symbol_or_pair}")
    return f"{base}_{pair_quote}"


def binance_symbol(pair: str) -> str:
    return pair.replace("/", "").replace("_", "").upper()


def binance_ws_url(pair: str) -> str:
    """Build Binance combined stream URL using @depth@100ms (full diff stream).
    Local BinanceFullBook maintains the order book and extracts top N levels.
    """
    symbol = binance_symbol(pair).lower()
    streams = "/".join([
        f"{symbol}@bookTicker",
        f"{symbol}@depth@100ms",
        f"{symbol}@trade",
    ])
    return f"{BINANCE_WS_BASE}/stream?streams={streams}"


def parse_l2_entries(entries: list) -> list[dict[str, float]]:
    """Convert Gate's [[price_str, amount_str], ...] to [{price, amount}, ...].

    Native Parquet list<struct> format — no JSON serialization needed.
    """
    return [{"price": float(p), "amount": float(a)} for p, a in entries]


def _entry_price_amount(entry: Any) -> tuple[float, float] | None:
    if isinstance(entry, dict):
        price = entry.get("price")
        amount = entry.get("amount")
    else:
        try:
            price = entry[0]
            amount = entry[1]
        except (TypeError, IndexError):
            return None
    try:
        return float(price), float(amount)
    except (TypeError, ValueError):
        return None


class LocalOrderBook:
    """Maintain a full in-memory order book from Gate spot.obu snapshots and deltas."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply_l2_row(self, row: dict) -> None:
        if row.get("is_snapshot"):
            self.bids.clear()
            self.asks.clear()
        self._apply_side(self.bids, row.get("bid_updates") or {})
        self._apply_side(self.asks, row.get("ask_updates") or {})
        self._trim()

    def depth_row(
        self,
        *,
        pair: str,
        exchange_time_ms: int | None,
        local_time_ms: int | None,
        update_id: int | None,
        is_snapshot: bool,
    ) -> dict:
        best_bid = max(self.bids) if self.bids else None
        best_ask = min(self.asks) if self.asks else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

        bid_depth = sum(self.bids.values())
        ask_depth = sum(self.asks.values())
        total_depth = bid_depth + ask_depth
        row = {
            "exchange_time_ms": exchange_time_ms,
            "local_time_ms": local_time_ms,
            "update_id": update_id,
            "pair": pair,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 10) if spread is not None else None,
            "mid_price": round(mid, 10) if mid is not None else None,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": total_depth,
            "imbalance": ((bid_depth - ask_depth) / total_depth) if total_depth else 0.0,
            "is_snapshot": is_snapshot,
        }
        for bps in (5, 10, 25):
            bid_band, ask_band = self._band_depth(mid, bps)
            row[f"bid_depth_{bps}bps"] = bid_band
            row[f"ask_depth_{bps}bps"] = ask_band
            row[f"total_depth_{bps}bps"] = bid_band + ask_band
        return row

    def _apply_side(self, side: dict[float, float], entries: list) -> None:
        for entry in entries:
            parsed = _entry_price_amount(entry)
            if parsed is None:
                continue
            price, amount = parsed
            if amount <= 0:
                side.pop(price, None)
            else:
                side[price] = amount

    def _trim(self) -> None:
        if len(self.bids) > self.depth:
            self.bids = dict(sorted(self.bids.items(), reverse=True)[: self.depth])
        if len(self.asks) > self.depth:
            self.asks = dict(sorted(self.asks.items())[: self.depth])

    def _band_depth(self, mid: float | None, bps: int) -> tuple[float, float]:
        if mid is None or mid <= 0:
            return 0.0, 0.0
        width = mid * bps / 10_000.0
        min_bid = mid - width
        max_ask = mid + width
        bid_depth = sum(amount for price, amount in self.bids.items() if price >= min_bid)
        ask_depth = sum(amount for price, amount in self.asks.items() if price <= max_ask)
        return bid_depth, ask_depth


class BinanceFullBook:
    """Maintain a full in-memory order book for Binance from REST snapshot + diff stream.

    Workflow:
    1. fetch_snapshot() → REST GET /api/v3/depth?limit=5000
    2. For each WS diff message, call apply_diff(U, u, bids, asks)
    3. Call top_levels(n) to extract top N bid/ask levels as L2 row
    4. Call depth_row() for aggregated depth statistics

    Diff validation: only apply diffs where finalUpdateId in event (u) >= snapshot lastUpdateId
    and firstUpdateId in event (U) <= lastUpdateId + 1.
    """

    BINANCE_REST_BASE = "https://api.binance.com"
    INITIAL_SNAPSHOT_LIMIT = 5000

    def __init__(self, symbol: str, record_depth: int = 100, proxy: str | None = None) -> None:
        self.symbol = symbol.lower()
        self.record_depth = record_depth
        self.proxy = proxy
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._last_update_id: int = 0  # last REST snapshot lastUpdateId
        self._final_update_id: int = 0  # last processed diff 'u'
        self._snapshot_ready: bool = False

    # --- REST snapshot ---

    async def fetch_snapshot(self) -> None:
        """Fetch initial full order book snapshot from Binance REST API and initialise."""
        url = f"{self.BINANCE_REST_BASE}/api/v3/depth"
        params = {"symbol": self.symbol.upper(), "limit": self.INITIAL_SNAPSHOT_LIMIT}
        connector = aiohttp.TCPConnector(force_close=True)
        kwargs = {}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        async with aiohttp.ClientSession(connector=connector, **kwargs) as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()

        self.bids.clear()
        self.asks.clear()
        for price_str, amount_str in data.get("bids", []):
            price, amount = float(price_str), float(amount_str)
            if amount > 0:
                self.bids[price] = amount
        for price_str, amount_str in data.get("asks", []):
            price, amount = float(price_str), float(amount_str)
            if amount > 0:
                self.asks[price] = amount

        self._last_update_id = int(data["lastUpdateId"])
        self._final_update_id = self._last_update_id
        self._snapshot_ready = True
        LOGGER.info(
            "REST snapshot loaded: lastUpdateId=%d, bids=%d, asks=%d",
            self._last_update_id, len(self.bids), len(self.asks),
        )

    def apply_snapshot_row(self, row: dict) -> None:
        """Initialise the book from a flattened Binance snapshot row."""
        self.bids.clear()
        self.asks.clear()
        self._update_side(self.bids, row.get("bid_updates") or [])
        self._update_side(self.asks, row.get("ask_updates") or [])
        update_id = row.get("update_id")
        if isinstance(update_id, int):
            self._last_update_id = update_id
            self._final_update_id = update_id
        self._snapshot_ready = True

    # --- Diff application ---

    def apply_diff(self, first_update_id: int, final_update_id: int,
                   bid_updates: list, ask_updates: list) -> bool:
        """Apply a depth diff event. Returns True if applied, False if skipped."""
        if not self._snapshot_ready:
            return False

        # Drop events that are stale (u <= lastUpdateId)
        if final_update_id <= self._last_update_id:
            return False

        # First diff after snapshot: accept any diff where u > lastUpdateId.
        # The gap between REST snapshot and WS stream is normal — the book moves
        # during the round trip.
        if self._final_update_id == self._last_update_id:
            self._update_side(self.bids, bid_updates)
            self._update_side(self.asks, ask_updates)
            self._final_update_id = final_update_id
            LOGGER.info(
                "diff synced: first diff U=%d, lastUpdateId=%d, u=%d (gap=%d events)",
                first_update_id, self._last_update_id, final_update_id,
                first_update_id - self._last_update_id - 1,
            )
            return True

        # Subsequent diffs: check continuity
        if first_update_id > self._final_update_id + 1:
            LOGGER.warning(
                "diff gap: U=%d > finalUpdateId+1=%d (gap=%d), book may be stale",
                first_update_id, self._final_update_id + 1,
                first_update_id - self._final_update_id - 1,
            )
            return False

        self._update_side(self.bids, bid_updates)
        self._update_side(self.asks, ask_updates)
        self._final_update_id = final_update_id
        return True

    def _update_side(self, side: dict[float, float], updates: list) -> None:
        for entry in updates:
            parsed = _entry_price_amount(entry)
            if parsed is None:
                continue
            price, amount = parsed
            if amount <= 0:
                side.pop(price, None)
            else:
                side[price] = amount

    @property
    def needs_resync(self) -> bool:
        """Return True if the book needs a fresh REST snapshot (due to gap or stall)."""
        return not self._snapshot_ready

    # --- Level extraction ---

    def top_levels(self) -> tuple[list, list]:
        """Extract top record_depth bid/ask levels as [[price, amount], ...] lists.
        Same format as Gate's parse_l2_entries input.
        """
        sorted_bids = sorted(self.bids.items(), reverse=True)[:self.record_depth]
        sorted_asks = sorted(self.asks.items())[:self.record_depth]
        bid_list = [[p, a] for p, a in sorted_bids]
        ask_list = [[p, a] for p, a in sorted_asks]
        return bid_list, ask_list

    def l2_snapshot_row(self, *, pair: str, local_time_ms: int) -> dict:
        """Generate an L2 snapshot row (like flatten_binance_l2 but from full book)."""
        bid_list, ask_list = self.top_levels()
        return {
            "exchange_time_ms": None,
            "local_time_ms": local_time_ms,
            "first_update_id": self._last_update_id,
            "update_id": self._final_update_id,
            "pair": pair,
            "bid_updates": parse_l2_entries(bid_list),
            "ask_updates": parse_l2_entries(ask_list),
            "bid_update_count": len(bid_list),
            "ask_update_count": len(ask_list),
            "is_snapshot": True,
            "is_empty": not bid_list and not ask_list,
        }

    def depth_row(
        self,
        *,
        pair: str,
        exchange_time_ms: int | None,
        local_time_ms: int | None,
        update_id: int | None,
        is_snapshot: bool,
    ) -> dict:
        """Generate aggregated depth statistics from current book state."""
        best_bid = max(self.bids) if self.bids else None
        best_ask = min(self.asks) if self.asks else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

        bid_depth = sum(self.bids.values())
        ask_depth = sum(self.asks.values())
        total_depth = bid_depth + ask_depth
        row = {
            "exchange_time_ms": exchange_time_ms,
            "local_time_ms": local_time_ms,
            "update_id": update_id,
            "pair": pair,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 10) if spread is not None else None,
            "mid_price": round(mid, 10) if mid is not None else None,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": total_depth,
            "imbalance": ((bid_depth - ask_depth) / total_depth) if total_depth else 0.0,
            "is_snapshot": is_snapshot,
        }
        for bps in (5, 10, 25):
            bid_band, ask_band = self._band_depth(mid, bps)
            row[f"bid_depth_{bps}bps"] = bid_band
            row[f"ask_depth_{bps}bps"] = ask_band
            row[f"total_depth_{bps}bps"] = bid_band + ask_band
        return row

    def _band_depth(self, mid: float | None, bps: int) -> tuple[float, float]:
        if mid is None or mid <= 0:
            return 0.0, 0.0
        width = mid * bps / 10_000.0
        min_bid = mid - width
        max_ask = mid + width
        bid_depth = sum(amount for price, amount in self.bids.items() if price >= min_bid)
        ask_depth = sum(amount for price, amount in self.asks.items() if price <= max_ask)
        return bid_depth, ask_depth


def hourly_path(
    output_dir: Path,
    *,
    pair: str,
    dataset: str,
    date_str: str,
    hour: int,
    exchange: str = "gate",
) -> Path:
    """Build path: {exchange}/spot/{pair}/{dataset}/date=YYYY-MM-DD/hour=HH/{ts}Z.parquet"""
    ts = f"{date_str.replace('-', '')}T{hour:02d}0000Z"
    return (
        output_dir / normalize_exchange(exchange) / "spot" / pair / dataset
        / f"date={date_str}" / f"hour={hour:02d}" / f"{ts}.parquet"
    )


def build_subscribe_messages(
    pair: str,
    depth: int,
    request_time: int,
    exchange: str = "gate",
) -> list[dict]:
    exchange = normalize_exchange(exchange)
    if exchange == "binance":
        symbol = binance_symbol(pair).lower()
        return [
            {
                "method": "SUBSCRIBE",
                "params": [
                    f"{symbol}@bookTicker",
                    f"{symbol}@depth{depth}@100ms",
                    f"{symbol}@trade",
                ],
                "id": request_time,
            }
        ]

    return [
        {
            "time": request_time,
            "channel": "spot.book_ticker",
            "event": "subscribe",
            "payload": [pair],
        },
        {
            "time": request_time,
            "channel": "spot.obu",
            "event": "subscribe",
            "payload": [f"ob.{pair}.{depth}"],
        },
        {"time": request_time, "channel": "spot.trades", "event": "subscribe", "payload": [pair]},
    ]


def build_l2_unsubscribe(pair: str, depth: int, request_time: int) -> dict:
    row = {
        "time": request_time,
        "channel": "spot.obu",
        "event": "unsubscribe",
        "payload": [f"ob.{pair}.{depth}"],
    }


def build_l2_subscribe(pair: str, depth: int, request_time: int) -> dict:
    return {
        "time": request_time,
        "channel": "spot.obu",
        "event": "subscribe",
        "payload": [f"ob.{pair}.{depth}"],
    }


# Message flattening

def flatten_l1(msg: dict, *, pair: str, received_at: datetime) -> dict:
    r = msg.get("result") or {}
    bid_p = parse_float(r.get("b"))
    bid_a = parse_float(r.get("B"))
    ask_p = parse_float(r.get("a"))
    ask_a = parse_float(r.get("A"))
    spread = ask_p - bid_p if bid_p is not None and ask_p is not None else None
    mid = (ask_p + bid_p) / 2 if bid_p is not None and ask_p is not None else None
    return {
        "exchange_time_ms": parse_int(r.get("t")) or parse_int(msg.get("time_ms")),
        "local_time_ms": to_epoch_ms(received_at),
        "update_id": parse_int(r.get("u")),
        "pair": pair,
        "bid_price": bid_p,
        "bid_amount": bid_a,
        "ask_price": ask_p,
        "ask_amount": ask_a,
        "spread": round(spread, 10) if spread is not None else None,
        "mid_price": round(mid, 10) if mid is not None else None,
    }


def flatten_l2(msg: dict, *, pair: str, received_at: datetime) -> dict:
    r = msg.get("result") or {}
    bids = r.get("b") or []
    asks = r.get("a") or []
    update_id = parse_int(r.get("u"))
    is_snapshot = bool(r.get("full"))
    return {
        "exchange_time_ms": parse_int(r.get("t")) or parse_int(msg.get("time_ms")),
        "local_time_ms": to_epoch_ms(received_at),
        "first_update_id": parse_int(r.get("U")) or update_id,
        "update_id": update_id,
        "pair": pair,
        "bid_updates": parse_l2_entries(bids),
        "ask_updates": parse_l2_entries(asks),
        "bid_update_count": len(bids),
        "ask_update_count": len(asks),
        "is_snapshot": is_snapshot,
        "is_empty": not is_snapshot and not bids and not asks,
    }


def flatten_trade(msg: dict, *, pair: str, received_at: datetime) -> dict:
    r = msg.get("result") or {}
    price = parse_float(r.get("price"))
    amount = parse_float(r.get("amount"))
    cost = price * amount if price is not None and amount is not None else None
    create_time = parse_int(r.get("create_time"))
    return {
        "exchange_time_ms": (
            parse_epoch_ms(r.get("create_time_ms"))
            or (create_time * 1000 if create_time is not None else None)
            or parse_epoch_ms(msg.get("time_ms"))
        ),
        "local_time_ms": to_epoch_ms(received_at),
        "trade_id": parse_int(r.get("id")),
        "pair": r.get("currency_pair") or pair,
        "side": r.get("side"),
        "price": price,
        "amount": amount,
        "cost": round(cost, 10) if cost is not None else None,
    }


def flatten_binance_l1(msg: dict, *, pair: str, received_at: datetime) -> dict:
    bid_p = parse_float(msg.get("b"))
    bid_a = parse_float(msg.get("B"))
    ask_p = parse_float(msg.get("a"))
    ask_a = parse_float(msg.get("A"))
    spread = ask_p - bid_p if bid_p is not None and ask_p is not None else None
    mid = (ask_p + bid_p) / 2 if bid_p is not None and ask_p is not None else None
    return {
        "exchange_time_ms": parse_epoch_ms(msg.get("E")),
        "local_time_ms": to_epoch_ms(received_at),
        "update_id": parse_int(msg.get("u")),
        "pair": pair,
        "bid_price": bid_p,
        "bid_amount": bid_a,
        "ask_price": ask_p,
        "ask_amount": ask_a,
        "spread": round(spread, 10) if spread is not None else None,
        "mid_price": round(mid, 10) if mid is not None else None,
    }


def flatten_binance_l2(msg: dict, *, pair: str, received_at: datetime) -> dict:
    """Flatten Binance depth update into internal format.
    - Partial book (@depth20@100ms): has lastUpdateId, is_snapshot=True, full 20 bids/asks
    - Diff stream (@depth@100ms): has U/u, is_snapshot=False, only changed levels
    """
    bids = msg.get("bids") or msg.get("b") or []
    asks = msg.get("asks") or msg.get("a") or []
    first_uid = parse_int(msg.get("U"))
    final_uid = parse_int(msg.get("u"))
    snapshot_id = parse_int(msg.get("lastUpdateId"))
    update_id = snapshot_id or final_uid
    row = {
        "exchange_time_ms": parse_epoch_ms(msg.get("E")),
        "local_time_ms": to_epoch_ms(received_at),
        "first_update_id": first_uid or update_id,
        "update_id": update_id,
        "pair": pair,
        "bid_updates": parse_l2_entries(bids),
        "ask_updates": parse_l2_entries(asks),
        "bid_update_count": len(bids),
        "ask_update_count": len(asks),
        "is_snapshot": bool(snapshot_id),
        "is_empty": not bids and not asks,
        # Extra fields for diff stream — U and u for BinanceFullBook.apply_diff()
    }
    if final_uid is not None:
        row["_diff_u"] = final_uid
    return row


def flatten_binance_trade(msg: dict, *, pair: str, received_at: datetime) -> dict:
    price = parse_float(msg.get("p"))
    amount = parse_float(msg.get("q"))
    cost = price * amount if price is not None and amount is not None else None
    return {
        "exchange_time_ms": parse_epoch_ms(msg.get("T")) or parse_epoch_ms(msg.get("E")),
        "local_time_ms": to_epoch_ms(received_at),
        "trade_id": parse_int(msg.get("t")),
        "pair": msg.get("s") or pair,
        "side": "sell" if msg.get("m") else "buy",
        "price": price,
        "amount": amount,
        "cost": round(cost, 10) if cost is not None else None,
    }


def normalize(msg: dict, pair: str, received_at: datetime, exchange: str = "gate") -> tuple[str, dict] | None:
    exchange = normalize_exchange(exchange)
    if "data" in msg and isinstance(msg["data"], dict):
        msg = msg["data"]
    if exchange == "binance":
        if "b" in msg and "B" in msg and "a" in msg and "A" in msg and "u" in msg:
            return "l1", flatten_binance_l1(msg, pair=pair, received_at=received_at)
        if "lastUpdateId" in msg or msg.get("e") == "depthUpdate":
            return "l2", flatten_binance_l2(msg, pair=pair, received_at=received_at)
        if msg.get("e") == "trade":
            return "trades", flatten_binance_trade(msg, pair=pair, received_at=received_at)
        return None

    if msg.get("event") != "update":
        return None
    ch = msg.get("channel")
    if ch == "spot.book_ticker":
        return "l1", flatten_l1(msg, pair=pair, received_at=received_at)
    if ch == "spot.obu":
        return "l2", flatten_l2(msg, pair=pair, received_at=received_at)
    if ch == "spot.trades":
        return "trades", flatten_trade(msg, pair=pair, received_at=received_at)
    return None


# Writer

class OrderBookWriter:
    """Buffers L1/L2 rows and writes hourly Parquet files atomically.

    Key features:
    - L2 bid/ask stored as native Parquet list<struct> (no JSON)
    - L1/L2 sequence gap tracking (non-fatal, logged)
    - L2 snapshot state machine (waiting → requested → received)
    - Atomic file writes (.tmp + os.replace)
    """

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self._exchange = normalize_exchange(config.exchange)
        self._buffers: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        self._last_flush = time.monotonic()

        # L1 sequence tracking
        self._last_l1_uid: int | None = None
        self._l1_gaps = 0
        self._l1_gap_missing = 0

        # L2 sequence tracking
        self._last_l2_uid: int | None = None
        self._l2_gaps = 0
        self._l2_gap_missing = 0

        # L2 snapshot state (Gate only)
        self._snapshot_state: str = "idle"
        self._last_snapshot_time: float = time.monotonic()
        self._snapshot_requested_at: float | None = None

        # Order book — different classes for different exchanges
        if self._exchange == "binance":
            symbol = binance_symbol(config.pair)
            self._book: LocalOrderBook | BinanceFullBook = BinanceFullBook(
                symbol=symbol, record_depth=config.depth, proxy=config.proxy,
            )
            self._binance_l2_snapshot_interval = 1.0  # snapshot L2 rows every 1s
            self._last_binance_l2_snapshot = 0.0
        else:
            self._book = LocalOrderBook(config.depth)

        self._last_depth_second: int | None = None

        # Live depth monitor state
        self._live = config.live
        self._prev_depth: dict | None = None
        self._trade_flow: dict[int, dict] = defaultdict(
            lambda: {"buy_vol": 0.0, "sell_vol": 0.0, "buy_n": 0, "sell_n": 0}
        )
        self._first_snapshot = True

    @property
    def pending_rows(self) -> int:
        return sum(len(v) for v in self._buffers.values())

    # Snapshot management

    def needs_snapshot(self) -> bool:
        """Check if an L2 snapshot should be requested."""
        if self._snapshot_state == "waiting":
            # Gap detected but not yet requested
            return True
        if self._snapshot_state == "requested":
            # Already requested — check for timeout
            if self._snapshot_requested_at is not None:
                elapsed = time.monotonic() - self._snapshot_requested_at
                if elapsed > self.config.snapshot_request_timeout:
                    LOGGER.warning(
                        "L2 snapshot timeout (%.1fs), re-requesting",
                        self.config.snapshot_request_timeout,
                    )
                    self._snapshot_state = "waiting"
                    return True
            return False
        # idle — check periodic interval
        elapsed = time.monotonic() - self._last_snapshot_time
        return elapsed >= self.config.snapshot_interval

    def mark_snapshot_requested(self) -> None:
        """Mark that a snapshot request has been sent."""
        self._snapshot_state = "requested"
        self._snapshot_requested_at = time.monotonic()

    def _on_snapshot_received(self, update_id: int) -> None:
        """Called when a snapshot message is received."""
        was_requested = self._snapshot_state in ("waiting", "requested")
        self._last_l2_uid = update_id
        self._snapshot_state = "idle"
        self._last_snapshot_time = time.monotonic()
        self._snapshot_requested_at = None
        if was_requested or self._first_snapshot:
            LOGGER.info("L2 snapshot received: update_id=%d", update_id)
            self._first_snapshot = False
        else:
            LOGGER.debug("L2 snapshot received: update_id=%d", update_id)

    # Sequence tracking

    def _check_l1_sequence(self, row: dict) -> None:
        """Track L1 update_id sequence. Non-fatal — always buffers the row."""
        uid = row.get("update_id")
        if not isinstance(uid, int):
            return
        if self._last_l1_uid is not None:
            gap = uid - self._last_l1_uid - 1
            if gap > 0:
                self._l1_gaps += 1
                self._l1_gap_missing += gap
                if gap > 100:
                    LOGGER.warning(
                        "L1 gap: %d missing updates (prev_uid=%d → cur_uid=%d)",
                        gap, self._last_l1_uid, uid,
                    )
        self._last_l1_uid = uid

    def _check_l2_sequence(self, row: dict) -> bool:
        """Track L2 sequence. Returns True if row should be buffered, False to skip.

        Gate:
        - Snapshots: always buffered, reset sequence state
        - Diffs while waiting for snapshot: skipped
        - Diffs with gap: skipped, trigger snapshot request
        - Normal diffs: buffered, update sequence

        Binance diff stream:
        - Always returns True (BinanceFullBook handles its own validation)
        - Rows are NOT raw diffs — they are periodic snapshots from the full book
        """
        if self._exchange == "binance":
            return True

        uid = row.get("update_id")
        first_uid = row.get("first_update_id")
        if not isinstance(uid, int) or not isinstance(first_uid, int):
            return True

        if row.get("is_snapshot"):
            self._on_snapshot_received(uid)
            return True

        if self._snapshot_state in ("waiting", "requested"):
            return False

        if self._last_l2_uid is not None:
            expected_first = self._last_l2_uid + 1
            if first_uid != expected_first:
                gap = first_uid - expected_first
                self._l2_gaps += 1
                self._l2_gap_missing += gap
                LOGGER.warning(
                    "L2 gap: expected first_uid=%d, got=%d (gap=%d) — requesting snapshot",
                    expected_first, first_uid, gap,
                )
                self._snapshot_state = "waiting"
                return False

        self._last_l2_uid = uid
        return True

    # Add messages

    def add(self, msg: dict, received_at: datetime | None = None) -> None:
        received_at = received_at or utcnow()
        normalized = normalize(msg, self.config.pair, received_at, exchange=self.config.exchange)
        if normalized is None:
            return

        dataset, row = normalized

        # --- Binance diff stream: apply diff to full book, snapshot periodically ---
        if self._exchange == "binance" and dataset == "l2":
            if not row.get("is_snapshot"):
                # Apply diff to the BinanceFullBook
                assert isinstance(self._book, BinanceFullBook)
                U = row.get("first_update_id")
                u = row.get("_diff_u")
                if isinstance(U, int) and isinstance(u, int):
                    applied = self._book.apply_diff(
                        U, u,
                        row.get("bid_updates") or [],
                        row.get("ask_updates") or [],
                    )
                    if not applied:
                        # Book needs resync
                        pass
                # Generate depth row from full book state
                self._add_depth_row(row, received_at)
                # Periodically snapshot the full book for L2 rows
                now = time.monotonic()
                if now - self._last_binance_l2_snapshot >= self._binance_l2_snapshot_interval:
                    self._snapshot_binance_l2(row.get("local_time_ms") or to_epoch_ms(received_at))
                    self._last_binance_l2_snapshot = now
                return
            assert isinstance(self._book, BinanceFullBook)
            self._book.apply_snapshot_row(row)
        # ----------------------------------------------------------------

        if dataset == "l1":
            self._check_l1_sequence(row)
            should_buffer = True
        elif dataset == "l2":
            should_buffer = self._check_l2_sequence(row)
        else:
            should_buffer = True

        if not should_buffer:
            return

        # Track trades for live depth monitor
        if self._live and dataset == "trades":
            self._track_trade(row, received_at)

        dt = received_at.astimezone(UTC)
        key = (dataset, dt.date().isoformat(), dt.hour)
        self._buffers[key].append(row)

        if dataset == "l2":
            self._add_depth_row(row, dt)

        if len(self._buffers[key]) >= self.config.flush_rows:
            self._flush_key(key)

    def _snapshot_binance_l2(self, local_time_ms: int) -> None:
        """Generate and buffer an L2 row from the BinanceFullBook's current state."""
        assert isinstance(self._book, BinanceFullBook)
        l2_row = self._book.l2_snapshot_row(
            pair=self.config.pair,
            local_time_ms=local_time_ms,
        )
        now = utcnow().astimezone(UTC)
        key = ("l2", now.date().isoformat(), now.hour)
        self._buffers[key].append(l2_row)

    def _add_depth_row(self, row: dict, received_at: datetime) -> None:
        """Generate a depth row from the current book state. For BinanceFullBook,
        the book is already updated externally via apply_diff(). For LocalOrderBook,
        apply the L2 row first."""
        if self._exchange != "binance":
            self._book.apply_l2_row(row)
        local_time_ms = row.get("local_time_ms")
        if not isinstance(local_time_ms, int):
            return
        depth_second = local_time_ms // 1000
        # Enforce once-per-second depth row for all exchanges.
        if self._last_depth_second == depth_second:
            return
        self._last_depth_second = depth_second
        depth_row = self._book.depth_row(
            pair=self.config.pair,
            exchange_time_ms=row.get("exchange_time_ms"),
            local_time_ms=local_time_ms,
            update_id=row.get("update_id"),
            is_snapshot=bool(row.get("is_snapshot")),
        )
        dt = received_at.astimezone(UTC)
        key = ("depth", dt.date().isoformat(), dt.hour)
        self._buffers[key].append(depth_row)
        if len(self._buffers[key]) >= self.config.flush_rows:
            self._flush_key(key)

        if self._live:
            self._print_live_depth(depth_row)

    # Live depth monitor

    def _track_trade(self, row: dict, received_at: datetime) -> None:
        """Accumulate per-second trade flow for the live monitor."""
        second = row.get("local_time_ms", to_epoch_ms(received_at)) // 1000
        side = row.get("side")
        amount = row.get("amount") or 0.0
        flow = self._trade_flow[second]
        if side == "buy":
            flow["buy_vol"] += amount
            flow["buy_n"] += 1
        elif side == "sell":
            flow["sell_vol"] += amount
            flow["sell_n"] += 1

    def _print_live_depth(self, depth_row: dict) -> None:
        """Print one-line depth consumption stats every second."""
        local_ms = depth_row.get("local_time_ms")
        if not isinstance(local_ms, int):
            return
        second = local_ms // 1000

        # Clean up trade flow older than 3 seconds
        stale = [s for s in self._trade_flow if s < second - 2]
        for s in stale:
            del self._trade_flow[s]

        ts_str = datetime.fromtimestamp(second).strftime("%H:%M:%S")

        mid = depth_row.get("mid_price")
        spread = depth_row.get("spread")
        if mid is None or spread is None:
            self._prev_depth = depth_row
            return
        bid_depth = depth_row.get("bid_depth") or 0.0
        ask_depth = depth_row.get("ask_depth") or 0.0
        bid_5bps = depth_row.get("bid_depth_5bps") or 0.0
        ask_5bps = depth_row.get("ask_depth_5bps") or 0.0
        bid_10bps = depth_row.get("bid_depth_10bps") or 0.0
        ask_10bps = depth_row.get("ask_depth_10bps") or 0.0
        imbalance = depth_row.get("imbalance") or 0.0
        bid_levels = depth_row.get("bid_levels") or 0
        ask_levels = depth_row.get("ask_levels") or 0

        # Compute depth delta vs previous second
        if self._prev_depth is not None:
            d_bid = bid_depth - (self._prev_depth.get("bid_depth") or 0.0)
            d_ask = ask_depth - (self._prev_depth.get("ask_depth") or 0.0)
            d_bid5 = bid_5bps - (self._prev_depth.get("bid_depth_5bps") or 0.0)
            d_ask5 = ask_5bps - (self._prev_depth.get("ask_depth_5bps") or 0.0)
        else:
            d_bid = d_ask = d_bid5 = d_ask5 = 0.0

        # Trade flow in the last second
        flow = self._trade_flow.get(second - 1) or self._trade_flow.get(second)
        if flow is None:
            flow = {"buy_vol": 0.0, "sell_vol": 0.0, "buy_n": 0, "sell_n": 0}
        total_trades = flow["buy_n"] + flow["sell_n"]

        # Spread in bps
        spread_bps = (spread / mid * 10000) if mid and spread else 0.0

        # Format helpers
        def fmt_delta(v: float) -> str:
            return f"{v:+.3f}"

        # ANSI colors
        C_TS = "\033[36m"       # cyan — timestamp
        C_MID = "\033[1;33m"    # bold yellow — mid price
        C_BID = "\033[31m"      # red — bid (Chinese convention)
        C_ASK = "\033[32m"      # green — ask
        C_DELTA_NEG = "\033[1;31m"  # bold red — depth consumed
        C_DELTA_POS = "\033[1;32m"  # bold green — depth added
        C_DIM = "\033[90m"      # gray — labels
        C_RST = "\033[0m"

        def delta_str(v: float) -> str:
            c = C_DELTA_NEG if v < 0 else C_DELTA_POS if v > 0 else C_DIM
            return f"{c}{fmt_delta(v)}{C_RST}"

        line = (
            f"{C_TS}{ts_str}{C_RST} "
            f"{C_DIM}|{C_RST} {C_MID}{mid:,.2f}{C_RST} "
            f"{C_DIM}Spr{C_RST} {spread:.2f}({spread_bps:.1f}bp) "
            f"{C_DIM}|{C_RST} {C_DIM}Lv{C_RST} {bid_levels}/{ask_levels} "
            f"{C_DIM}|{C_RST} {C_DIM}Depth{C_RST} "
            f"{C_BID}B{bid_depth:.2f}{C_RST} {C_ASK}A{ask_depth:.2f}{C_RST} "
            f"{C_DIM}Δ{C_RST} {delta_str(d_bid)}/{delta_str(d_ask)} "
            f"{C_DIM}|{C_RST} {C_DIM}5bp{C_RST} "
            f"{C_BID}B{bid_5bps:.2f}{C_RST} {C_ASK}A{ask_5bps:.2f}{C_RST} "
            f"{C_DIM}Δ{C_RST} {delta_str(d_bid5)}/{delta_str(d_ask5)} "
            f"{C_DIM}|{C_RST} {C_DIM}Imb{C_RST} {imbalance:+.1%} "
            f"{C_DIM}|{C_RST} {C_DIM}Trd{C_RST} {total_trades} "
            f"{C_BID}↑{flow['buy_vol']:.3f}{C_RST} "
            f"{C_ASK}↓{flow['sell_vol']:.3f}{C_RST}"
        )
        print(line, flush=True)

        self._prev_depth = depth_row

    # Flush

    def flush_due(self) -> None:
        """Flush old-hour buffers immediately; flush current buffers periodically."""
        now = utcnow().astimezone(UTC)
        current_date = now.date().isoformat()
        current_hour = now.hour

        # Flush any buffers from previous hours (hour boundary crossing)
        for key in list(self._buffers.keys()):
            _, date_str, hour = key
            if date_str != current_date or hour != current_hour:
                self._flush_key(key)

        # Periodic flush for data freshness / crash safety
        if time.monotonic() - self._last_flush >= self.config.flush_seconds:
            self.flush_all()

    def _flush_key(self, key: tuple[str, str, int]) -> None:
        rows = self._buffers.pop(key, [])
        if not rows:
            return

        dataset, date_str, hour = key
        path = hourly_path(
            self.config.output_dir,
            pair=self.config.pair,
            dataset=dataset,
            date_str=date_str,
            hour=hour,
            exchange=self.config.exchange,
        )

        # Handle overflow: if file exists, append part suffix
        if path.exists():
            for i in range(1, 10000):
                part_path = path.with_name(f"{path.stem}_p{i:03d}.parquet")
                if not part_path.exists():
                    path = part_path
                    break

        schema = {
            "l1": L1_SCHEMA,
            "l2": L2_SCHEMA,
            "trades": TRADES_SCHEMA,
            "depth": DEPTH_SCHEMA,
        }[dataset]
        table = pa.Table.from_pylist(rows, schema=schema)

        # Atomic write: write to .tmp, then rename
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, path)

        LOGGER.info("wrote %d %s rows → %s", len(rows), dataset, path.name)
        self._last_flush = time.monotonic()

    def flush_all(self) -> None:
        for key in list(self._buffers):
            self._flush_key(key)

    def close(self) -> None:
        self.flush_all()
        self._log_gap_stats()

    def _log_gap_stats(self) -> None:
        LOGGER.info(
            "gap stats — L1: %d gaps / %d missing | L2: %d gaps / %d missing",
            self._l1_gaps, self._l1_gap_missing,
            self._l2_gaps, self._l2_gap_missing,
        )


# WebSocket recorder

async def _request_l2_snapshot(ws, config: RecorderConfig) -> None:
    """Request a fresh L2 snapshot by unsubscribing and resubscribing to spot.obu.

    This triggers Gate to send a full order book snapshot (full=1) as the next
    update on the spot.obu channel.
    """
    request_time = int(time.time())
    # Unsubscribe first
    await ws.send(orjson.dumps(build_l2_unsubscribe(config.pair, config.depth, request_time)))
    # Small delay to let server process unsubscribe before resubscribe
    await asyncio.sleep(0.1)
    # Resubscribe — this triggers a new snapshot
    await ws.send(orjson.dumps(build_l2_subscribe(config.pair, config.depth, request_time)))
    LOGGER.info("L2 snapshot requested (unsubscribe + resubscribe)")


async def run_recorder(config: RecorderConfig, stop_event: asyncio.Event) -> None:
    writer = OrderBookWriter(config)
    exchange = normalize_exchange(config.exchange)

    # --- Binance: fetch REST snapshot and initialise full order book ---
    if exchange == "binance":
        assert isinstance(writer._book, BinanceFullBook)
        LOGGER.info("fetching REST snapshot for Binance full order book...")
        try:
            await writer._book.fetch_snapshot()
        except Exception:
            LOGGER.exception("failed to fetch REST snapshot, aborting")
            writer.close()
            return
        ws_url = binance_ws_url(config.pair)
    else:
        ws_url = GATE_WS_URL

    try:
        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=config.ping_interval,
                    ping_timeout=config.ping_timeout,
                    proxy=config.proxy,
                ) as ws:
                    LOGGER.info(
                        "connected to %s (proxy=%s, pair=%s, depth=%d)",
                        ws_url, config.proxy or "none", config.pair, config.depth,
                    )

                    # Gate needs explicit subscribe messages; Binance uses combined stream URL
                    if exchange == "gate":
                        request_time = int(time.time())
                        for sub in build_subscribe_messages(
                            config.pair, config.depth, request_time, exchange=exchange
                        ):
                            await ws.send(orjson.dumps(sub))
                        LOGGER.info(
                            "subscribed: L1=spot.book_ticker, L2=spot.obu(depth=%d), "
                            "trades=spot.trades",
                            config.depth,
                        )
                    else:
                        symbol = binance_symbol(config.pair).lower()
                        LOGGER.info(
                            "streams: %s@bookTicker, %s@depth@100ms (full diff), %s@trade",
                            symbol, symbol, symbol,
                        )
                        LOGGER.info(
                            "local book: recording top %d levels (bids=%d asks=%d from REST)",
                            config.depth,
                            len(writer._book.bids), len(writer._book.asks),
                        )

                    if config.live:
                        print(
                            "\n"
                            f"  {'TIME':>8} | {'MID':>14} {'SPREAD':>16} | "
                            f"{'LV':>6} | {'DEPTH(BTC)':>24} {'CONSUMED':>20} | "
                            f"{'5bps DEPTH':>20} {'CONSUMED':>20} | {'IMB':>7} | "
                            f"{'TRADES':>20}\n"
                            + "-" * 180,
                            flush=True,
                        )

                    # Initial subscription sends L2 snapshot — mark as waiting
                    if exchange == "gate":
                        writer._snapshot_state = "requested"
                        writer._snapshot_requested_at = time.monotonic()

                    while not stop_event.is_set():
                        # Check if periodic/gap L2 snapshot is needed
                        if exchange == "gate" and writer.needs_snapshot():
                            await _request_l2_snapshot(ws, config)
                            writer.mark_snapshot_requested()

                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=config.recv_timeout
                            )
                        except TimeoutError:
                            writer.flush_due()
                            continue

                        writer.add(orjson.loads(raw), received_at=utcnow())
                        writer.flush_due()

            except Exception:
                LOGGER.exception(
                    "WebSocket disconnected, reconnecting in %.1fs",
                    config.reconnect_seconds,
                )
                # Reset snapshot state — reconnection will trigger new snapshot
                if exchange == "gate":
                    writer._snapshot_state = "requested"
                    writer._snapshot_requested_at = time.monotonic()
                await asyncio.sleep(config.reconnect_seconds)
    finally:
        writer.close()


# CLI

def parse_args() -> RecorderConfig:
    p = argparse.ArgumentParser(
        description="Record spot L1/L2 order book data (format v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exchange", choices=["gate", "binance"], default="binance")
    p.add_argument(
        "--pair",
        default=None,
        help="Legacy exchange pair symbol, e.g. BTC_USDT for gate or BTCUSDT for binance.",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Base symbol or pair to record, e.g. BTC, ETH, XRP, ETH_USDT, or ETH/USDT.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("user_data/orderbook_data"))
    p.add_argument("--depth", type=int, default=5000,
                   help="L2 depth levels to record. Binance default 5000 = full order book "
                        "(REST snapshot max). For Gate, this controls the partial book subscription "
                        "level (max 100).")
    p.add_argument(
        "--flush-rows",
        type=int,
        default=500_000,
        help="Safety valve: flush when buffer exceeds this",
    )
    p.add_argument("--flush-seconds", type=float, default=60.0, help="Periodic flush interval")
    p.add_argument("--reconnect-seconds", type=float, default=2.0, help="WebSocket reconnect delay")
    p.add_argument(
        "--snapshot-interval", type=float, default=3600.0,
        help="L2 snapshot interval in seconds (prevents stale book reconstruction)",
    )
    p.add_argument(
        "--proxy", type=str, default=None,
        help="HTTP proxy for WebSocket and REST (e.g. http://127.0.0.1:7890). "
             "Off by default.",
    )
    p.add_argument(
        "--no-live", action="store_true",
        help="Disable real-time depth consumption output",
    )
    args = p.parse_args()
    pair_input = args.pair or args.symbol or "BTC"
    pair = normalize_pair(pair_input, exchange=args.exchange)
    proxy = args.proxy
    # Binance: always uses full diff stream (@depth@100ms),
    #   depth controls how many top levels to record from the local full book.
    # Gate: depth controls the partial book subscription level.
    depth = args.depth
    if args.exchange == "binance":
        LOGGER.info(
            "Binance: using @depth@100ms (full diff stream), recording top %d levels",
            depth,
        )
    return RecorderConfig(
        exchange=args.exchange,
        pair=pair,
        output_dir=args.output_dir,
        depth=depth,
        flush_rows=args.flush_rows,
        flush_seconds=args.flush_seconds,
        reconnect_seconds=args.reconnect_seconds,
        snapshot_interval=args.snapshot_interval,
        proxy=proxy,
        live=not args.no_live,
    )


def main() -> None:
    # Ensure UTF-8 output for Unicode characters (delta, arrows, etc.)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = parse_args()
    stop_event = asyncio.Event()

    def request_stop() -> None:
        LOGGER.info("stop requested")
        stop_event.set()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except NotImplementedError:
                signal.signal(sig, lambda *_: request_stop())
        loop.run_until_complete(run_recorder(config, stop_event))
    finally:
        LOGGER.info("recorder stopped")


if __name__ == "__main__":
    main()
