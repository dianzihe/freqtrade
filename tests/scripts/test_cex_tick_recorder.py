from datetime import UTC, datetime
import importlib
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.cex_tick_recorder import (
    LocalOrderBook,
    OrderBookWriter,
    RecorderConfig,
    build_subscribe_messages,
    flatten_l1,
    flatten_l2,
    flatten_trade,
    hourly_path,
    normalize,
    normalize_pair,
    parse_args,
)


ROOT = Path(__file__).resolve().parents[2]
RECEIVED_AT = datetime(2026, 6, 26, 15, 30, tzinfo=UTC)


def test_gate_tick_recorder_compatibility_entrypoint_exports_cex_recorder() -> None:
    legacy = importlib.import_module("scripts.gate_tick_recorder")

    assert legacy.RecorderConfig is RecorderConfig
    assert legacy.main is not None


def test_gate_tick_recorder_compatibility_entrypoint_runs_as_script() -> None:
    script = ROOT / "scripts" / "gate_tick_recorder.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--exchange {gate,binance}" in result.stdout


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


def test_build_subscribe_messages_supports_binance_btc_streams() -> None:
    subscriptions = build_subscribe_messages("BTCUSDT", depth=50, request_time=1, exchange="binance")

    assert subscriptions == [
        {
            "method": "SUBSCRIBE",
            "params": ["btcusdt@bookTicker", "btcusdt@depth50@100ms", "btcusdt@trade"],
            "id": 1,
        },
    ]


@pytest.mark.parametrize(
    ("exchange", "symbol", "expected_pair"),
    [
        ("gate", "BTC", "BTC_USDT"),
        ("gate", "ETH_USDT", "ETH_USDT"),
        ("gate", "xrp/usdt", "XRP_USDT"),
        ("binance", "BTC", "BTCUSDT"),
        ("binance", "eth_usdt", "ETHUSDT"),
        ("binance", "XRP/USDT", "XRPUSDT"),
    ],
)
def test_normalize_pair_maps_symbol_for_supported_exchanges(
    exchange: str,
    symbol: str,
    expected_pair: str,
) -> None:
    assert normalize_pair(symbol, exchange=exchange) == expected_pair


@pytest.mark.parametrize(
    ("argv", "expected_exchange", "expected_pair"),
    [
        (["cex_tick_recorder.py", "--exchange", "gate", "--symbol", "XRP"], "gate", "XRP_USDT"),
        (["cex_tick_recorder.py", "--exchange", "binance", "--symbol", "ETH"], "binance", "ETHUSDT"),
        (["cex_tick_recorder.py", "--exchange", "gate", "--pair", "BTC_USDT"], "gate", "BTC_USDT"),
    ],
)
def test_parse_args_accepts_user_symbol_or_legacy_pair(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_exchange: str,
    expected_pair: str,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    config = parse_args()

    assert config.exchange == expected_exchange
    assert config.pair == expected_pair


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


def test_hourly_path_can_write_binance_dataset(tmp_path: Path) -> None:
    path = hourly_path(
        tmp_path,
        pair="BTCUSDT",
        dataset="l1",
        date_str="2026-06-26",
        hour=15,
        exchange="binance",
    )

    assert path == (
        tmp_path
        / "binance"
        / "spot"
        / "BTCUSDT"
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


def test_local_orderbook_reconstructs_full_book_then_applies_deltas() -> None:
    book = LocalOrderBook(depth=3)

    snapshot = {
        "exchange_time_ms": 1782438627052,
        "local_time_ms": 1782487800000,
        "update_id": 100,
        "pair": "BTC_USDT",
        "bid_updates": [
            {"price": 100.0, "amount": 1.0},
            {"price": 99.5, "amount": 2.0},
        ],
        "ask_updates": [
            {"price": 101.0, "amount": 3.0},
            {"price": 101.5, "amount": 4.0},
        ],
        "is_snapshot": True,
    }
    delta = {
        "exchange_time_ms": 1782438628052,
        "local_time_ms": 1782487801000,
        "update_id": 101,
        "pair": "BTC_USDT",
        "bid_updates": [
            {"price": 100.0, "amount": 0.0},
            {"price": 100.5, "amount": 5.0},
        ],
        "ask_updates": [{"price": 101.0, "amount": 1.5}],
        "is_snapshot": False,
    }

    book.apply_l2_row(snapshot)
    book.apply_l2_row(delta)
    row = book.depth_row(pair="BTC_USDT", exchange_time_ms=1782438628052,
                         local_time_ms=1782487801000, update_id=101, is_snapshot=False)

    assert row["best_bid"] == 100.5
    assert row["best_ask"] == 101.0
    assert row["bid_depth"] == 7.0
    assert row["ask_depth"] == 5.5
    assert row["total_depth"] == 12.5
    assert row["bid_levels"] == 2
    assert row["ask_levels"] == 2
    assert row["imbalance"] == pytest.approx((7.0 - 5.5) / 12.5)


def test_local_orderbook_depth_bands_are_calculated_from_mid_price() -> None:
    book = LocalOrderBook(depth=5)
    book.apply_l2_row(
        {
            "exchange_time_ms": 1782438627052,
            "local_time_ms": 1782487800000,
            "update_id": 100,
            "pair": "BTC_USDT",
            "bid_updates": [
                {"price": 99.99, "amount": 1.0},
                {"price": 99.90, "amount": 2.0},
                {"price": 99.00, "amount": 10.0},
            ],
            "ask_updates": [
                {"price": 100.01, "amount": 3.0},
                {"price": 100.10, "amount": 4.0},
                {"price": 101.00, "amount": 10.0},
            ],
            "is_snapshot": True,
        }
    )

    row = book.depth_row(pair="BTC_USDT", exchange_time_ms=1782438627052,
                         local_time_ms=1782487800000, update_id=100, is_snapshot=True)

    assert row["mid_price"] == 100.0
    assert row["bid_depth_5bps"] == 1.0
    assert row["ask_depth_5bps"] == 3.0
    assert row["total_depth_5bps"] == 4.0
    assert row["bid_depth_10bps"] == 3.0
    assert row["ask_depth_10bps"] == 7.0
    assert row["total_depth_10bps"] == 10.0


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


def test_normalize_routes_binance_l1_l2_and_trades() -> None:
    l1 = normalize(
        {
            "u": 400900217,
            "s": "BTCUSDT",
            "b": "59400.3",
            "B": "0.146214",
            "a": "59400.4",
            "A": "0.0402",
        },
        "BTCUSDT",
        RECEIVED_AT,
        exchange="binance",
    )
    l2 = normalize(
        {
            "lastUpdateId": 160,
            "bids": [["59400.1", "1.5"]],
            "asks": [["59400.5", "2.5"]],
        },
        "BTCUSDT",
        RECEIVED_AT,
        exchange="binance",
    )
    trade = normalize(
        {
            "e": "trade",
            "E": 1782438622000,
            "s": "BTCUSDT",
            "t": 309143071,
            "p": "59400.3",
            "q": "0.25",
            "T": 1782438621999,
            "m": True,
        },
        "BTCUSDT",
        RECEIVED_AT,
        exchange="binance",
    )

    assert l1 == (
        "l1",
        {
            "exchange_time_ms": None,
            "local_time_ms": 1782487800000,
            "update_id": 400900217,
            "pair": "BTCUSDT",
            "bid_price": 59400.3,
            "bid_amount": 0.146214,
            "ask_price": 59400.4,
            "ask_amount": 0.0402,
            "spread": 0.1,
            "mid_price": 59400.35,
        },
    )
    assert l2 == (
        "l2",
        {
            "exchange_time_ms": None,
            "local_time_ms": 1782487800000,
            "first_update_id": 160,
            "update_id": 160,
            "pair": "BTCUSDT",
            "bid_updates": [{"price": 59400.1, "amount": 1.5}],
            "ask_updates": [{"price": 59400.5, "amount": 2.5}],
            "bid_update_count": 1,
            "ask_update_count": 1,
            "is_snapshot": True,
            "is_empty": False,
        },
    )
    assert trade == (
        "trades",
        {
            "exchange_time_ms": 1782438621999,
            "local_time_ms": 1782487800000,
            "trade_id": 309143071,
            "pair": "BTCUSDT",
            "side": "sell",
            "price": 59400.3,
            "amount": 0.25,
            "cost": 14850.075,
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


def test_orderbook_writer_writes_binance_rows_under_binance_path(tmp_path: Path) -> None:
    config = RecorderConfig(
        exchange="binance",
        pair="BTCUSDT",
        output_dir=tmp_path,
        flush_rows=1,
    )
    writer = OrderBookWriter(config)

    writer.add(
        {
            "u": 400900217,
            "s": "BTCUSDT",
            "b": "59400.3",
            "B": "1",
            "a": "59400.4",
            "A": "2",
        },
        received_at=RECEIVED_AT,
    )
    writer.add(
        {
            "lastUpdateId": 160,
            "bids": [["59400.1", "1.5"]],
            "asks": [["59400.5", "2.5"]],
        },
        received_at=RECEIVED_AT,
    )
    writer.add(
        {
            "e": "trade",
            "s": "BTCUSDT",
            "t": 309143071,
            "p": "59400.3",
            "q": "0.25",
            "T": 1782438621999,
            "m": False,
        },
        received_at=RECEIVED_AT,
    )
    writer.close()

    pair_dir = tmp_path / "binance" / "spot" / "BTCUSDT"
    l1_files = list((pair_dir / "l1" / "date=2026-06-26" / "hour=15").glob("*.parquet"))
    depth_files = list((pair_dir / "depth" / "date=2026-06-26" / "hour=15").glob("*.parquet"))
    trade_files = list((pair_dir / "trades" / "date=2026-06-26" / "hour=15").glob("*.parquet"))

    assert len(l1_files) == 1
    assert len(depth_files) == 1
    assert len(trade_files) == 1
    trades = pd.read_parquet(trade_files[0])
    assert trades.loc[0, "side"] == "buy"
    depth = pd.read_parquet(depth_files[0])
    assert depth.loc[0, "best_bid"] == 59400.1
    assert depth.loc[0, "best_ask"] == 59400.5


def test_orderbook_writer_writes_second_level_depth_from_full_l2_book(tmp_path: Path) -> None:
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
                "b": [["100", "1"], ["99.9", "2"]],
                "a": [["100.1", "3"], ["100.2", "4"]],
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
                "U": 101,
                "u": 101,
                "b": [["100", "1.5"]],
                "a": [["100.2", "0"]],
            },
        },
        received_at=RECEIVED_AT.replace(second=31),
    )
    writer.close()

    depth_files = list(
        (
            tmp_path / "gate" / "spot" / "BTC_USDT" / "depth"
            / "date=2026-06-26" / "hour=15"
        ).glob("*.parquet")
    )

    assert len(depth_files) == 1
    depth = pd.read_parquet(depth_files[0])
    assert list(depth["update_id"]) == [100, 101]
    assert depth.loc[1, "bid_depth"] == 3.5
    assert depth.loc[1, "ask_depth"] == 3.0
    assert depth.loc[1, "total_depth"] == 6.5
    assert depth.loc[1, "best_bid"] == 100.0
    assert depth.loc[1, "best_ask"] == 100.1


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

    assert writer.pending_rows == 2
    assert len(writer._buffers[("l2", "2026-06-26", 15)]) == 1
    assert len(writer._buffers[("depth", "2026-06-26", 15)]) == 1
    assert writer.needs_snapshot() is True
