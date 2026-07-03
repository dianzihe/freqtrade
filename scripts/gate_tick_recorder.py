#!/usr/bin/env python3
"""Record Gate.io spot L1, L2 order book, and public trade data into hourly parquet files.

Format v2 — key improvements over v1:
- L2 bid/ask stored as Parquet native list<struct<price:double, amount:double>>
  instead of JSON strings (3-5x faster read/write, no serialization overhead)
- Periodic L2 snapshots (default every 3600s) for reliable order book reconstruction
- L1/L2 sequence gap tracking with non-fatal recovery
- L2 gap recovery via in-channel re-subscription (no full WebSocket reconnect)
- Atomic file writes (write to .tmp, then os.replace) — no more corrupted files
- Hour-based Hive partitioning (date=YYYY-MM-DD/hour=HH/)

Usage:
    python scripts/gate_tick_recorder.py --pair BTC_USDT
    python scripts/gate_tick_recorder.py --pair ETH_USDT --snapshot-interval 1800
"""

import argparse
import asyncio
import logging
import os
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import websockets


GATE_WS_URL = "wss://api.gateio.ws/ws/v4/"
LOGGER = logging.getLogger("gate_book_recorder")

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


# Config

@dataclass(frozen=True)
class RecorderConfig:
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


def parse_l2_entries(entries: list) -> list[dict[str, float]]:
    """Convert Gate's [[price_str, amount_str], ...] to [{price, amount}, ...].

    Native Parquet list<struct> format — no JSON serialization needed.
    """
    return [{"price": float(p), "amount": float(a)} for p, a in entries]


def hourly_path(output_dir: Path, *, pair: str, dataset: str, date_str: str, hour: int) -> Path:
    """Build path: gate/spot/{pair}/{dataset}/date=YYYY-MM-DD/hour=HH/{ts}Z.parquet"""
    ts = f"{date_str.replace('-', '')}T{hour:02d}0000Z"
    return (
        output_dir / "gate" / "spot" / pair / dataset
        / f"date={date_str}" / f"hour={hour:02d}" / f"{ts}.parquet"
    )


def build_subscribe_messages(pair: str, depth: int, request_time: int) -> list[dict]:
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
    return {
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


def normalize(msg: dict, pair: str, received_at: datetime) -> tuple[str, dict] | None:
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

        # L2 snapshot state
        #   _snapshot_state: "idle" | "waiting" | "requested"
        self._snapshot_state: str = "idle"
        self._last_snapshot_time: float = time.monotonic()
        self._snapshot_requested_at: float | None = None

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
        self._last_l2_uid = update_id
        self._snapshot_state = "idle"
        self._last_snapshot_time = time.monotonic()
        self._snapshot_requested_at = None
        LOGGER.info("L2 snapshot received: update_id=%d", update_id)

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

        - Snapshots: always buffered, reset sequence state
        - Diffs while waiting for snapshot: skipped (meaningless without snapshot)
        - Diffs with gap: skipped, trigger snapshot request
        - Normal diffs: buffered, update sequence
        """
        uid = row.get("update_id")
        first_uid = row.get("first_update_id")
        if not isinstance(uid, int) or not isinstance(first_uid, int):
            return True

        if row.get("is_snapshot"):
            self._on_snapshot_received(uid)
            return True

        if self._snapshot_state in ("waiting", "requested"):
            # Skip diffs while waiting for snapshot — they're meaningless
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
        normalized = normalize(msg, self.config.pair, received_at)
        if normalized is None:
            return

        dataset, row = normalized
        if dataset == "l1":
            self._check_l1_sequence(row)
            should_buffer = True
        elif dataset == "l2":
            should_buffer = self._check_l2_sequence(row)
        else:
            should_buffer = True

        if not should_buffer:
            return

        dt = received_at.astimezone(UTC)
        key = (dataset, dt.date().isoformat(), dt.hour)
        self._buffers[key].append(row)

        if len(self._buffers[key]) >= self.config.flush_rows:
            self._flush_key(key)

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
        )

        # Handle overflow: if file exists, append part suffix
        if path.exists():
            for i in range(1, 10000):
                part_path = path.with_name(f"{path.stem}_p{i:03d}.parquet")
                if not part_path.exists():
                    path = part_path
                    break

        schema = {"l1": L1_SCHEMA, "l2": L2_SCHEMA, "trades": TRADES_SCHEMA}[dataset]
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

    try:
        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    GATE_WS_URL,
                    ping_interval=config.ping_interval,
                    ping_timeout=config.ping_timeout,
                ) as ws:
                    LOGGER.info("connected to %s", GATE_WS_URL)
                    request_time = int(time.time())

                    # Subscribe to L1, L2, and public trades
                    for sub in build_subscribe_messages(config.pair, config.depth, request_time):
                        await ws.send(orjson.dumps(sub))
                    LOGGER.info(
                        "subscribed: L1=spot.book_ticker, L2=spot.obu(depth=%d), "
                        "trades=spot.trades",
                        config.depth,
                    )

                    # Initial subscription sends L2 snapshot — mark as waiting
                    writer._snapshot_state = "requested"
                    writer._snapshot_requested_at = time.monotonic()

                    while not stop_event.is_set():
                        # Check if periodic/gap L2 snapshot is needed
                        if writer.needs_snapshot():
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
                writer._snapshot_state = "requested"
                writer._snapshot_requested_at = time.monotonic()
                await asyncio.sleep(config.reconnect_seconds)
    finally:
        writer.close()


# CLI

def parse_args() -> RecorderConfig:
    p = argparse.ArgumentParser(
        description="Record Gate.io L1/L2 order book data (format v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pair", default="BTC_USDT", help="Gate currency pair, e.g. BTC_USDT")
    p.add_argument("--output-dir", type=Path, default=Path("user_data/orderbook_data"))
    p.add_argument("--depth", type=int, default=50, help="L2 depth for spot.obu")
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
    args = p.parse_args()
    return RecorderConfig(
        pair=args.pair,
        output_dir=args.output_dir,
        depth=args.depth,
        flush_rows=args.flush_rows,
        flush_seconds=args.flush_seconds,
        reconnect_seconds=args.reconnect_seconds,
        snapshot_interval=args.snapshot_interval,
    )


def main() -> None:
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
        loop = asyncio.get_event_loop()
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
