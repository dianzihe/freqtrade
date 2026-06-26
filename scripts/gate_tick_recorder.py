#!/usr/bin/env python3
"""Record Gate.io spot L1 and L2 order book data into daily parquet files."""

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

L1_COLUMNS = [
    "exchange_time_ms",
    "local_time_ms",
    "update_id",
    "pair",
    "bid_price",
    "bid_amount",
    "ask_price",
    "ask_amount",
    "spread",
    "mid_price",
]
L2_COLUMNS = [
    "exchange_time_ms",
    "local_time_ms",
    "first_update_id",
    "update_id",
    "pair",
    "bid_updates",
    "ask_updates",
    "bid_update_count",
    "ask_update_count",
]


@dataclass(frozen=True)
class GateBookRecorderConfig:
    pair: str = "BTC_USDT"
    output_dir: Path = Path("user_data/orderbook_data")
    depth: int = 50
    flush_rows: int = 5000
    flush_seconds: float = 10.0
    reconnect_seconds: float = 5.0


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def json_dumps(value: Any) -> str:
    return orjson.dumps(value).decode("utf-8")


def daily_path(
    output_dir: Path,
    *,
    pair: str,
    dataset: str,
    received_at: datetime,
    run_id: str,
    part: int,
) -> Path:
    date = received_at.astimezone(UTC).date().isoformat()
    return output_dir / "gate" / "spot" / pair / dataset / f"date={date}" / f"{run_id}-{part:06d}.parquet"


def build_subscriptions(pair: str, *, depth: int, request_time: int = 1) -> list[dict[str, Any]]:
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
    ]


def flatten_l1_message(
    message: dict[str, Any],
    *,
    pair: str,
    received_at: datetime,
) -> dict[str, Any]:
    result = message.get("result") or {}
    bid_price = parse_float(result.get("b"))
    bid_amount = parse_float(result.get("B"))
    ask_price = parse_float(result.get("a"))
    ask_amount = parse_float(result.get("A"))
    spread = ask_price - bid_price if bid_price is not None and ask_price is not None else None
    mid_price = (ask_price + bid_price) / 2 if bid_price is not None and ask_price is not None else None

    return {
        "exchange_time_ms": parse_int(result.get("t")) or parse_int(message.get("time_ms")),
        "local_time_ms": to_epoch_ms(received_at),
        "update_id": parse_int(result.get("u")),
        "pair": pair,
        "bid_price": bid_price,
        "bid_amount": bid_amount,
        "ask_price": ask_price,
        "ask_amount": ask_amount,
        "spread": round(spread, 10) if spread is not None else None,
        "mid_price": round(mid_price, 10) if mid_price is not None else None,
    }


def flatten_l2_message(
    message: dict[str, Any],
    *,
    pair: str,
    received_at: datetime,
) -> dict[str, Any]:
    result = message.get("result") or {}
    bids = result.get("b") or []
    asks = result.get("a") or []
    update_id = parse_int(result.get("u"))

    return {
        "exchange_time_ms": parse_int(result.get("t")) or parse_int(message.get("time_ms")),
        "local_time_ms": to_epoch_ms(received_at),
        "first_update_id": parse_int(result.get("U")) or update_id,
        "update_id": update_id,
        "pair": pair,
        "bid_updates": json_dumps(bids),
        "ask_updates": json_dumps(asks),
        "bid_update_count": len(bids),
        "ask_update_count": len(asks),
    }


def normalize_message(
    message: dict[str, Any],
    pair: str,
    received_at: datetime,
) -> tuple[str, dict[str, Any]] | None:
    if message.get("event") != "update":
        return None

    channel = message.get("channel")
    if channel == "spot.book_ticker":
        return "l1", flatten_l1_message(message, pair=pair, received_at=received_at)
    if channel == "spot.obu":
        return "l2", flatten_l2_message(message, pair=pair, received_at=received_at)
    return None


class DailyParquetWriter:
    def __init__(self, config: GateBookRecorderConfig) -> None:
        self.config = config
        self.run_id = f"{utcnow().strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        self._buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._parts: dict[tuple[str, str], int] = defaultdict(int)
        self._last_flush = time.monotonic()

    @property
    def pending_rows(self) -> int:
        return sum(len(rows) for rows in self._buffers.values())

    def add(self, message: dict[str, Any], *, received_at: datetime | None = None) -> None:
        received_at = received_at or utcnow()
        normalized = normalize_message(message, self.config.pair, received_at)
        if normalized is None:
            return

        dataset, row = normalized
        date = received_at.astimezone(UTC).date().isoformat()
        key = (dataset, date)
        self._buffers[key].append(row)

        if len(self._buffers[key]) >= self.config.flush_rows:
            self.flush_key(key)

    def flush_due(self) -> None:
        if time.monotonic() - self._last_flush >= self.config.flush_seconds:
            self.flush_all()

    def flush_key(self, key: tuple[str, str]) -> None:
        rows = self._buffers.pop(key, [])
        if not rows:
            return

        dataset, date_value = key
        self._parts[key] += 1
        received_at = datetime.fromisoformat(f"{date_value}T00:00:00+00:00")
        path = daily_path(
            self.config.output_dir,
            pair=self.config.pair,
            dataset=dataset,
            received_at=received_at,
            run_id=self.run_id,
            part=self._parts[key],
        )
        columns = L1_COLUMNS if dataset == "l1" else L2_COLUMNS
        table = pa.Table.from_pylist([{column: row.get(column) for column in columns} for row in rows])

        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        LOGGER.info("wrote %s %s rows to %s", len(rows), dataset, path)
        self._last_flush = time.monotonic()

    def flush_all(self) -> None:
        for key in list(self._buffers):
            self.flush_key(key)

    def close(self) -> None:
        self.flush_all()


async def record_gate_books(config: GateBookRecorderConfig, stop_event: asyncio.Event) -> None:
    writer = DailyParquetWriter(config)

    try:
        while not stop_event.is_set():
            try:
                async with websockets.connect(GATE_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    request_time = int(time.time())
                    for subscription in build_subscriptions(
                        config.pair,
                        depth=config.depth,
                        request_time=request_time,
                    ):
                        await ws.send(json_dumps(subscription))
                        LOGGER.info("sent subscription: %s", subscription)

                    while not stop_event.is_set():
                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except TimeoutError:
                            writer.flush_due()
                            continue

                        writer.add(orjson.loads(raw_message), received_at=utcnow())
                        writer.flush_due()
            except Exception:
                LOGGER.exception(
                    "Gate websocket disconnected; reconnecting in %.1fs",
                    config.reconnect_seconds,
                )
                await asyncio.sleep(config.reconnect_seconds)
    finally:
        writer.close()


def parse_args() -> GateBookRecorderConfig:
    parser = argparse.ArgumentParser(description="Record Gate.io L1 and L2 order book data.")
    parser.add_argument("--pair", default="BTC_USDT", help="Gate currency pair, e.g. BTC_USDT.")
    parser.add_argument("--output-dir", type=Path, default=Path("user_data/orderbook_data"))
    parser.add_argument("--depth", type=int, default=50, help="L2 depth for spot.obu.")
    parser.add_argument("--flush-rows", type=int, default=5000)
    parser.add_argument("--flush-seconds", type=float, default=10.0)
    parser.add_argument("--reconnect-seconds", type=float, default=5.0)
    args = parser.parse_args()

    return GateBookRecorderConfig(
        pair=args.pair,
        output_dir=args.output_dir,
        depth=args.depth,
        flush_rows=args.flush_rows,
        flush_seconds=args.flush_seconds,
        reconnect_seconds=args.reconnect_seconds,
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
        loop.run_until_complete(record_gate_books(config, stop_event))
    finally:
        LOGGER.info("recorder stopped")


if __name__ == "__main__":
    main()
