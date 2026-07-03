from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.gate_tick_recorder import (
    OrderBookWriter,
    RecorderConfig,
    build_subscribe_messages,
    flatten_l1,
    flatten_l2,
    flatten_trade,
    hourly_path,
    normalize,
)


RECEIVED_AT = datetime(2026, 6, 26, 15, 30, tzinfo=UTC)


def test_build_subscribe_messages_includes_l1_l2_and_public_trades() -> None:
    subscriptions = build_subscribe_messages("BTC_USDT", depth=50, request_time=1)

    assert subscriptions == [
        {
            "time": 1,
            "channel": "spot.book_ticker",
            "event": "subscribe",
            "payload": ["BTC_USDT"],
        },
        {
            "time": 1,
            "channel": "spot.obu",
            "event": "subscribe",
            "payload": ["ob.BTC_USDT.50"],
        },
        {
            "time": 1,
            "channel": "spot.trades",
            "event": "subscribe",
            "payload": ["BTC_USDT"],
        },
    ]


def test_hourly_path_uses_dataset_utc_date_and_hour(tmp_path: Path) -> None:
    path = hourly_path(tmp_path, pair="BTC_USDT", dataset="l1", date_str="2026-06-26", hour=15)

    assert path == (
        tmp_path
        / "gate"
        / "spot"
        / "BTC_USDT"
        / "l1"
        / "date=2026-06-26"
        / "hour=15"
        / "20260626T150000Z.parquet"
    )


def test_flatten_l1_has_backtest_ready_columns() -> None:
    row = flatten_l1(
        {
            "time": 1782438622,
            "time_ms": 1782438622580,
            "channel": "spot.book_ticker",
            "event": "update",
            "result": {
                "t": 1782438622580,
                "u": 38149398912,
                "s": "BTC_USDT",
                "b": "59400.3",
                "B": "0.146214",
                "a": "59400.4",
                "A": "0.0402",
            },
        },
        pair="BTC_USDT",
        received_at=RECEIVED_AT,
    )

    assert row == {
        "exchange_time_ms": 1782438622580,
        "local_time_ms": 1782487800000,
        "update_id": 38149398912,
        "pair": "BTC_USDT",
        "bid_price": 59400.3,
        "bid_amount": 0.146214,
        "ask_price": 59400.4,
        "ask_amount": 0.0402,
        "spread": 0.1,
        "mid_price": 59400.35,
    }


def test_flatten_l2_has_native_delta_columns() -> None:
    row = flatten_l2(
        {
            "channel": "spot.obu",
            "result": {
                "t": 1782438627052,
                "s": "ob.BTC_USDT.50",
                "u": 38149401433,
                "U": 38149401423,
                "b": [["59386.4", "0.033713"], ["59386.9", "0"]],
                "a": [["59401.1", "0.12"]],
            },
            "time_ms": 1782438627053,
            "event": "update",
        },
        pair="BTC_USDT",
        received_at=RECEIVED_AT,
    )

    assert row == {
        "exchange_time_ms": 1782438627052,
        "local_time_ms": 1782487800000,
        "first_update_id": 38149401423,
        "update_id": 38149401433,
        "pair": "BTC_USDT",
        "bid_updates": [{"price": 59386.4, "amount": 0.033713}, {"price": 59386.9, "amount": 0.0}],
        "ask_updates": [{"price": 59401.1, "amount": 0.12}],
        "bid_update_count": 2,
        "ask_update_count": 1,
        "is_snapshot": False,
        "is_empty": False,
    }


def test_flatten_trade_has_public_trade_columns() -> None:
    row = flatten_trade(
        {
            "time": 1606292218,
            "time_ms": 1606292218214,
            "channel": "spot.trades",
            "event": "update",
            "result": {
                "id": 309143071,
                "create_time": 1606292218,
                "create_time_ms": "1606292218213.4578",
                "side": "sell",
                "currency_pair": "BTC_USDT",
                "amount": "16.4700000000",
                "price": "0.4705000000",
            },
        },
        pair="BTC_USDT",
        received_at=RECEIVED_AT,
    )

    assert row == {
        "exchange_time_ms": 1606292218213,
        "local_time_ms": 1782487800000,
        "trade_id": 309143071,
        "pair": "BTC_USDT",
        "side": "sell",
        "price": 0.4705,
        "amount": 16.47,
        "cost": 7.749135,
    }


def test_normalize_ignores_non_update_and_unknown_channels() -> None:
    subscribe_message = {"channel": "spot.book_ticker", "event": "subscribe"}
    unknown_message = {"channel": "spot.tickers", "event": "update"}

    assert normalize(subscribe_message, "BTC_USDT", RECEIVED_AT) is None
    assert normalize(unknown_message, "BTC_USDT", RECEIVED_AT) is None


def test_normalize_routes_public_trades() -> None:
    normalized = normalize(
        {
            "channel": "spot.trades",
            "event": "update",
            "result": {
                "id": 309143071,
                "create_time_ms": "1606292218213.4578",
                "side": "buy",
                "amount": "2",
                "price": "3",
            },
        },
        "BTC_USDT",
        RECEIVED_AT,
    )

    assert normalized == (
        "trades",
        {
            "exchange_time_ms": 1606292218213,
            "local_time_ms": 1782487800000,
            "trade_id": 309143071,
            "pair": "BTC_USDT",
            "side": "buy",
            "price": 3.0,
            "amount": 2.0,
            "cost": 6.0,
        },
    )


def test_orderbook_writer_writes_l1_l2_and_trades_to_hourly_files(tmp_path: Path) -> None:
    config = RecorderConfig(pair="BTC_USDT", output_dir=tmp_path, flush_rows=1)
    writer = OrderBookWriter(config)

    writer.add(
        {
            "time_ms": 1782438622580,
            "channel": "spot.book_ticker",
            "event": "update",
            "result": {
                "t": 1782438622580,
                "u": 1,
                "b": "59400.3",
                "B": "1",
                "a": "59400.4",
                "A": "2",
            },
        },
        received_at=RECEIVED_AT,
    )
    writer.add(
        {
            "time_ms": 1782438627053,
            "channel": "spot.obu",
            "event": "update",
            "result": {
                "t": 1782438627052,
                "u": 2,
                "full": True,
                "b": [["59386.4", "0.1"]],
                "a": [],
            },
        },
        received_at=RECEIVED_AT,
    )
    writer.add(
        {
            "time_ms": 1606292218214,
            "channel": "spot.trades",
            "event": "update",
            "result": {
                "id": 309143071,
                "create_time_ms": "1606292218213.4578",
                "side": "sell",
                "amount": "16.47",
                "price": "0.4705",
            },
        },
        received_at=RECEIVED_AT,
    )
    writer.close()

    pair_dir = tmp_path / "gate" / "spot" / "BTC_USDT"
    l1_files = list((pair_dir / "l1" / "date=2026-06-26" / "hour=15").glob("*.parquet"))
    l2_files = list((pair_dir / "l2" / "date=2026-06-26" / "hour=15").glob("*.parquet"))
    trade_files = list((pair_dir / "trades" / "date=2026-06-26" / "hour=15").glob("*.parquet"))

    assert len(l1_files) == 1
    assert len(l2_files) == 1
    assert len(trade_files) == 1

    trades = pd.read_parquet(trade_files[0])
    assert list(trades.columns) == [
        "exchange_time_ms",
        "local_time_ms",
        "trade_id",
        "pair",
        "side",
        "price",
        "amount",
        "cost",
    ]
    assert trades.loc[0, "trade_id"] == 309143071
    assert trades.loc[0, "side"] == "sell"
    assert trades.loc[0, "cost"] == 7.749135


def test_orderbook_writer_skips_l2_sequence_gap_until_snapshot(tmp_path: Path) -> None:
    config = RecorderConfig(pair="BTC_USDT", output_dir=tmp_path, flush_rows=100)
    writer = OrderBookWriter(config)

    writer.add(
        {
            "time_ms": 1782438627053,
            "channel": "spot.obu",
            "event": "update",
            "result": {
                "t": 1782438627052,
                "u": 100,
                "full": True,
                "b": [["59386.4", "0.1"]],
                "a": [["59386.5", "0.2"]],
            },
        },
        received_at=RECEIVED_AT,
    )
    writer.add(
        {
            "time_ms": 1782438628053,
            "channel": "spot.obu",
            "event": "update",
            "result": {
                "t": 1782438628052,
                "U": 102,
                "u": 103,
                "b": [["59386.4", "0.3"]],
                "a": [],
            },
        },
        received_at=RECEIVED_AT,
    )

    assert writer.pending_rows == 1
    assert writer.needs_snapshot() is True
