from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.gate_tick_recorder import (
    DailyParquetWriter,
    GateBookRecorderConfig,
    L2SequenceGap,
    build_subscriptions,
    daily_path,
    flatten_l1_message,
    flatten_l2_message,
    normalize_message,
)


def test_build_subscriptions_only_l1_and_l2_depth50() -> None:
    subscriptions = build_subscriptions("BTC_USDT", depth=50, request_time=1)

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
    ]


def test_daily_path_uses_channel_and_utc_date(tmp_path: Path) -> None:
    path = daily_path(
        tmp_path,
        pair="BTC_USDT",
        dataset="l1",
        received_at=datetime(2026, 6, 26, 15, 30, tzinfo=UTC),
        run_id="run-test",
        part=3,
    )

    assert path == tmp_path / "gate" / "spot" / "BTC_USDT" / "l1" / "date=2026-06-26" / "run-test-000003.parquet"


def test_flatten_l1_message_has_backtest_ready_columns() -> None:
    row = flatten_l1_message(
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
        received_at=datetime(2026, 6, 26, 15, 30, tzinfo=UTC),
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


def test_flatten_l2_message_has_compact_delta_columns() -> None:
    row = flatten_l2_message(
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
        received_at=datetime(2026, 6, 26, 15, 30, tzinfo=UTC),
    )

    assert row == {
        "exchange_time_ms": 1782438627052,
        "local_time_ms": 1782487800000,
        "first_update_id": 38149401423,
        "update_id": 38149401433,
        "pair": "BTC_USDT",
        "bid_updates": '[["59386.4","0.033713"],["59386.9","0"]]',
        "ask_updates": '[["59401.1","0.12"]]',
        "bid_update_count": 2,
        "ask_update_count": 1,
        "is_snapshot": False,
        "is_empty": False,
    }


def test_flatten_l2_message_marks_snapshot_and_empty_delta() -> None:
    snapshot = flatten_l2_message(
        {
            "channel": "spot.obu",
            "result": {
                "t": 1782438627052,
                "s": "ob.BTC_USDT.50",
                "u": 38149401433,
                "full": True,
                "b": [["59386.4", "0.033713"]],
                "a": [["59401.1", "0.12"]],
            },
            "time_ms": 1782438627053,
            "event": "update",
        },
        pair="BTC_USDT",
        received_at=datetime(2026, 6, 26, 15, 30, tzinfo=UTC),
    )
    empty_delta = flatten_l2_message(
        {
            "channel": "spot.obu",
            "result": {
                "t": 1782438628052,
                "s": "ob.BTC_USDT.50",
                "u": 38149401434,
                "U": 38149401434,
            },
            "time_ms": 1782438628053,
            "event": "update",
        },
        pair="BTC_USDT",
        received_at=datetime(2026, 6, 26, 15, 30, tzinfo=UTC),
    )

    assert snapshot["is_snapshot"] is True
    assert snapshot["is_empty"] is False
    assert empty_delta["is_snapshot"] is False
    assert empty_delta["is_empty"] is True


def test_normalize_message_ignores_non_update_and_unknown_channels() -> None:
    received_at = datetime(2026, 6, 26, 15, 30, tzinfo=UTC)

    assert normalize_message({"channel": "spot.book_ticker", "event": "subscribe"}, "BTC_USDT", received_at) is None
    assert normalize_message({"channel": "spot.trades", "event": "update"}, "BTC_USDT", received_at) is None


def test_daily_writer_appends_l1_and_l2_to_daily_files(tmp_path: Path) -> None:
    config = GateBookRecorderConfig(pair="BTC_USDT", output_dir=tmp_path, flush_rows=1)
    writer = DailyParquetWriter(config)
    received_at = datetime(2026, 6, 26, 15, 30, tzinfo=UTC)

    writer.add(
        {
            "time_ms": 1782438622580,
            "channel": "spot.book_ticker",
            "event": "update",
            "result": {"t": 1782438622580, "u": 1, "b": "59400.3", "B": "1", "a": "59400.4", "A": "2"},
        },
        received_at=received_at,
    )
    writer.add(
        {
            "time_ms": 1782438627053,
            "channel": "spot.obu",
            "event": "update",
            "result": {"t": 1782438627052, "u": 2, "U": 2, "b": [["59386.4", "0.1"]], "a": []},
        },
        received_at=received_at,
    )
    writer.close()

    l1_files = list((tmp_path / "gate" / "spot" / "BTC_USDT" / "l1" / "date=2026-06-26").glob("*.parquet"))
    l2_files = list((tmp_path / "gate" / "spot" / "BTC_USDT" / "l2" / "date=2026-06-26").glob("*.parquet"))
    assert len(l1_files) == 1
    assert len(l2_files) == 1
    l1 = pd.read_parquet(l1_files[0])
    l2 = pd.read_parquet(l2_files[0])

    assert list(l1.columns) == [
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
    assert list(l2.columns) == [
        "exchange_time_ms",
        "local_time_ms",
        "first_update_id",
        "update_id",
        "pair",
        "bid_updates",
        "ask_updates",
        "bid_update_count",
        "ask_update_count",
        "is_snapshot",
        "is_empty",
    ]
    assert len(l1) == 1
    assert len(l2) == 1
    assert not bool(l2.loc[0, "is_snapshot"])
    assert not bool(l2.loc[0, "is_empty"])


def test_daily_writer_rejects_l2_sequence_gap(tmp_path: Path) -> None:
    config = GateBookRecorderConfig(pair="BTC_USDT", output_dir=tmp_path, flush_rows=100)
    writer = DailyParquetWriter(config)
    received_at = datetime(2026, 6, 26, 15, 30, tzinfo=UTC)

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
        received_at=received_at,
    )

    try:
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
            received_at=received_at,
        )
    except L2SequenceGap as exc:
        assert exc.expected_first_update_id == 101
        assert exc.actual_first_update_id == 102
    else:
        raise AssertionError("Expected L2SequenceGap")
