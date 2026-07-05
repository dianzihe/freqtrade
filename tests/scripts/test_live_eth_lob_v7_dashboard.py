from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "live_eth_lob_v7_dashboard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("live_eth_lob_v7_dashboard", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_ohlcv(minutes: int = 120) -> pd.DataFrame:
    times = pd.date_range("2026-07-05T10:00:00Z", periods=minutes, freq="1min")
    close = [1800.0 + i * 0.05 for i in range(minutes)]
    return pd.DataFrame(
        {
            "date": times,
            "open": close,
            "high": [v + 0.4 for v in close],
            "low": [v - 0.4 for v in close],
            "close": close,
            "volume": [20.0] * minutes,
            "trade_count": [30] * minutes,
            "buy_volume": [10.0] * minutes,
            "sell_volume": [10.0] * minutes,
        }
    )


def make_depth(minutes: int = 120) -> pd.DataFrame:
    times = pd.date_range("2026-07-05T10:00:00Z", periods=minutes, freq="1min")
    return pd.DataFrame(
        {
            "date": times,
            "bid_depth": [100.0] * minutes,
            "ask_depth": [110.0] * minutes,
            "total_depth": [210.0] * minutes,
            "bid_depth_5bps": [35.0] * minutes,
            "ask_depth_5bps": [40.0] * minutes,
            "total_depth_5bps": [75.0] * minutes,
            "bid_depth_10bps": [60.0] * minutes,
            "ask_depth_10bps": [65.0] * minutes,
            "total_depth_10bps": [125.0] * minutes,
            "bid_depth_25bps": [90.0] * minutes,
            "ask_depth_25bps": [100.0] * minutes,
            "total_depth_25bps": [190.0] * minutes,
            "best_bid": [1799.99 + i * 0.05 for i in range(minutes)],
            "best_ask": [1800.01 + i * 0.05 for i in range(minutes)],
            "spread": [0.02] * minutes,
            "mid_price": [1800.0 + i * 0.05 for i in range(minutes)],
            "imbalance": [-0.02] * minutes,
            "bid_levels": [100] * minutes,
            "ask_levels": [100] * minutes,
        }
    )


def test_l2_top10_snapshot_builds_cumulative_depth_ladder() -> None:
    dashboard = load_module()
    minute = pd.Timestamp("2026-07-05T10:15:00Z")
    l2 = pd.DataFrame(
        [
            {
                "time": minute,
                "is_snapshot": True,
                "bid_updates": [
                    {"price": 1800.0 - i, "amount": float(i + 1)}
                    for i in range(12)
                ],
                "ask_updates": [
                    {"price": 1801.0 + i, "amount": float((i + 1) * 2)}
                    for i in range(12)
                ],
            }
        ]
    )

    result = dashboard.build_l2_top10_1m(l2)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["top10_bid_depth"] == 55.0
    assert row["top10_ask_depth"] == 110.0
    assert row["top10_total_depth"] == 165.0
    assert row["top10_levels"]["bids"][0] == {
        "level": 1,
        "price": 1800.0,
        "amount": 1.0,
        "cumulative": 1.0,
    }
    assert row["top10_levels"]["asks"][9] == {
        "level": 10,
        "price": 1810.0,
        "amount": 20.0,
        "cumulative": 110.0,
    }


def test_l2_top10_snapshot_can_build_second_level_ladder() -> None:
    dashboard = load_module()
    l2 = pd.DataFrame(
        [
            {
                "time": pd.Timestamp("2026-07-05T10:15:01.250Z"),
                "is_snapshot": True,
                "bid_updates": [[1800.0 - i, float(i + 1)] for i in range(10)],
                "ask_updates": [[1801.0 + i, float(i + 1)] for i in range(10)],
            },
            {
                "time": pd.Timestamp("2026-07-05T10:15:02.250Z"),
                "is_snapshot": False,
                "bid_updates": [[1800.0, 20.0]],
                "ask_updates": [[1801.0, 30.0]],
            },
        ]
    )

    result = dashboard.build_l2_top10(l2, freq="1s")

    assert list(result["date"]) == [
        pd.Timestamp("2026-07-05T10:15:01Z"),
        pd.Timestamp("2026-07-05T10:15:02Z"),
    ]
    assert result.iloc[0]["top10_total_depth"] == 110.0
    assert result.iloc[1]["top10_bid_depth"] == 74.0
    assert result.iloc[1]["top10_ask_depth"] == 84.0


def test_compute_second_signals_exposes_ch1_to_ch5_and_trade_depth() -> None:
    dashboard = load_module()
    rows = 140
    times = pd.date_range("2026-07-05T10:15:00Z", periods=rows, freq="1s")
    ohlcv = pd.DataFrame(
        {
            "date": times,
            "open": [1800.0 + i * 0.01 for i in range(rows)],
            "high": [1800.2 + i * 0.01 for i in range(rows)],
            "low": [1799.8 + i * 0.01 for i in range(rows)],
            "close": [1800.0 + i * 0.01 for i in range(rows)],
            "volume": [2.0] * rows,
            "trade_count": [3] * rows,
            "buy_volume": [1.1] * rows,
            "sell_volume": [0.9] * rows,
        }
    )
    depth = pd.DataFrame(
        {
            "date": times,
            "bid_depth": [100.0] * rows,
            "ask_depth": [120.0] * rows,
            "total_depth": [220.0] * rows,
            "bid_depth_5bps": [30.0] * rows,
            "ask_depth_5bps": [40.0] * rows,
            "total_depth_5bps": [70.0] * rows,
            "bid_depth_10bps": [50.0] * rows,
            "ask_depth_10bps": [60.0] * rows,
            "total_depth_10bps": [110.0] * rows,
            "bid_depth_25bps": [90.0] * rows,
            "ask_depth_25bps": [100.0] * rows,
            "total_depth_25bps": [190.0] * rows,
            "best_bid": [1799.99 + i * 0.01 for i in range(rows)],
            "best_ask": [1800.01 + i * 0.01 for i in range(rows)],
            "spread": [0.02] * rows,
            "mid_price": [1800.0 + i * 0.01 for i in range(rows)],
        }
    )
    top10 = pd.DataFrame(
        {
            "date": times,
            "top10_bid_depth": [10.0] * rows,
            "top10_ask_depth": [12.0] * rows,
            "top10_total_depth": [22.0] * rows,
            "top10_levels": [
                {
                    "bids": [{"level": 1, "price": 1800.0, "amount": 10.0, "cumulative": 10.0}],
                    "asks": [{"level": 1, "price": 1801.0, "amount": 12.0, "cumulative": 12.0}],
                }
                for _ in range(rows)
            ],
        }
    )

    signals = dashboard.compute_lob_v7_signals(ohlcv, depth)
    signals = dashboard.attach_top10(signals, top10)
    row = dashboard.build_signal_rows(signals.tail(1), only_triggers=False)[0]

    assert row["timeframe"] == "1s"
    assert row["ch1"] is not None
    assert row["ch2"] is not None
    assert row["ch3"] is not None
    assert row["ch4"] is not None
    assert row["ch5"] is not None
    assert row["score_formula"] == "max(CH1..CH5) * drift_amp"
    assert row["orderbook_status"] == "ok"
    assert row["trade_count"] == 3
    assert row["second_buy_volume"] == 1.1
    assert row["second_sell_volume"] == 0.9
    assert row["top10_total_depth"] == 22.0


def test_compute_lob_v7_signals_uses_strategy_columns_and_l2_depth() -> None:
    dashboard = load_module()
    ohlcv = make_ohlcv()
    depth = make_depth()
    depth.loc[80:95, "total_depth"] = [180, 160, 140, 120, 100, 80, 70, 60, 55, 50, 48, 46, 44, 42, 40, 38]
    depth.loc[80:95, "bid_depth"] = depth.loc[80:95, "total_depth"] / 2
    depth.loc[80:95, "ask_depth"] = depth.loc[80:95, "total_depth"] / 2
    depth.loc[80:95, "best_bid"] = [1799.5 + i * 0.01 for i in range(16)]
    depth.loc[80:95, "best_ask"] = [1800.5 + i * 0.01 for i in range(16)]

    signals = dashboard.compute_lob_v7_signals(ohlcv, depth)

    assert "lob_score" in signals.columns
    assert "lob_depth_erosion" in signals.columns
    assert signals.loc[90, "lob_depth_erosion"] > 0.0
    assert signals.loc[90, "lob_spread_drift"] > 0.0
    assert signals.loc[90, "total_depth"] == 48.0


def test_build_signal_rows_exposes_time_signal_channels_and_depth() -> None:
    dashboard = load_module()
    signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-05T10:15:00Z"),
                "close": 1801.5,
                "lob_score": 0.71,
                "lob_threshold": 0.60,
                "lob_trigger": 1,
                "enter_short": 1,
                "lob_score_delta": 0.05,
                "lob_entropy": 0.11,
                "lob_depth_erosion": 0.22,
                "lob_spread_drift": 0.33,
                "lob_ofi_momentum": 0.44,
                "lob_prestress_proxy": 0.55,
                "bid_depth": 12.0,
                "ask_depth": 15.0,
                "total_depth": 27.0,
                "bid_depth_5bps": 2.0,
                "ask_depth_5bps": 3.0,
                "total_depth_5bps": 5.0,
                "bid_depth_10bps": 4.0,
                "ask_depth_10bps": 6.0,
                "total_depth_10bps": 10.0,
                "bid_depth_25bps": 8.0,
                "ask_depth_25bps": 9.0,
                "total_depth_25bps": 17.0,
                "spread": 0.1,
                "mid_price": 1801.4,
                "trade_count": 9,
                "buy_volume": 4.0,
                "sell_volume": 5.0,
                "top10_bid_depth": 12.0,
                "top10_ask_depth": 15.0,
                "top10_total_depth": 27.0,
                "top10_levels": {
                    "bids": [
                        {"level": 1, "price": 1800.0, "amount": 12.0, "cumulative": 12.0}
                    ],
                    "asks": [
                        {"level": 1, "price": 1801.0, "amount": 15.0, "cumulative": 15.0}
                    ],
                },
            }
        ]
    )

    rows = dashboard.build_signal_rows(signals, only_triggers=False)

    assert rows[0]["time"] == "2026-07-05T10:15:00+00:00"
    assert rows[0]["signal_value"] == 0.71
    assert rows[0]["ch1"] == 0.11
    assert rows[0]["ch2"] == 0.22
    assert rows[0]["ch3"] == 0.33
    assert rows[0]["ch4"] == 0.44
    assert rows[0]["l2_bid_depth"] == 12.0
    assert rows[0]["l2_ask_depth"] == 15.0
    assert rows[0]["l2_total_depth"] == 27.0
    assert rows[0]["l2_total_depth_25bps"] == 17.0
    assert rows[0]["top10_bid_depth"] == 12.0
    assert rows[0]["top10_ask_depth"] == 15.0
    assert rows[0]["top10_total_depth"] == 27.0
    assert rows[0]["top10_levels"]["bids"][0]["price"] == 1800.0


def test_html_refreshes_every_minute_and_names_required_fields() -> None:
    dashboard = load_module()

    html = dashboard.HTML_TEMPLATE

    assert "const REFRESH_MS = 1000;" in html
    assert "发出时间" in html
    assert "信号值" in html
    assert "CH1" in html
    assert "CH2" in html
    assert "CH3" in html
    assert "CH4" in html
    assert "CH5" in html
    assert "本秒成交量" in html
    assert "drift_amp" in html
    assert "L2买盘深度" in html
    assert "L2卖盘深度" in html
    assert "L2总体深度" in html
