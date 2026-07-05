from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import duckdb
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "live_btc_microstructure_dashboard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("live_btc_microstructure_dashboard", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_ohlcv(minutes: int = 80) -> pd.DataFrame:
    times = pd.date_range("2026-07-05T02:00:00Z", periods=minutes, freq="1min")
    close = [61000.0 + i * 0.2 for i in range(minutes)]
    return pd.DataFrame(
        {
            "date": times,
            "open": close,
            "high": [v + 0.5 for v in close],
            "low": [v - 0.5 for v in close],
            "close": close,
            "volume": [10.0] * minutes,
            "trade_count": [20] * minutes,
            "buy_volume": [5.0] * minutes,
            "sell_volume": [5.0] * minutes,
        }
    )


def make_depth(minutes: int = 80) -> pd.DataFrame:
    times = pd.date_range("2026-07-05T02:00:00Z", periods=minutes, freq="1min")
    return pd.DataFrame(
        {
            "date": times,
            "bid_depth": [10.0] * minutes,
            "ask_depth": [10.0] * minutes,
            "total_depth": [20.0] * minutes,
            "bid_depth_5bps": [4.0] * minutes,
            "ask_depth_5bps": [4.0] * minutes,
            "total_depth_5bps": [8.0] * minutes,
            "bid_depth_10bps": [7.0] * minutes,
            "ask_depth_10bps": [7.0] * minutes,
            "total_depth_10bps": [14.0] * minutes,
            "spread": [0.10] * minutes,
            "mid_price": [61000.0 + i * 0.2 for i in range(minutes)],
        }
    )


def test_build_signal_rows_attaches_matching_l2_depth() -> None:
    dashboard = load_module()

    minute = pd.Timestamp("2026-07-05T02:15:00Z")
    signals = pd.DataFrame(
        [
            {
                "date": minute,
                "close": 61000.5,
                "ch1": 0.11,
                "ch2": 0.22,
                "ch3": 0.33,
                "ch4": -0.44,
                "signal_value": 0.55,
                "threshold": 0.50,
                "trigger": 1,
            }
        ]
    )
    depth = pd.DataFrame(
        [
            {
                "date": minute,
                "bid_depth": 12.0,
                "ask_depth": 15.0,
                "total_depth": 27.0,
                "bid_depth_5bps": 2.0,
                "ask_depth_5bps": 3.0,
                "total_depth_5bps": 5.0,
                "bid_depth_10bps": 4.0,
                "ask_depth_10bps": 6.0,
                "total_depth_10bps": 10.0,
                "spread": 0.1,
                "mid_price": 61000.4,
            }
        ]
    )

    rows = dashboard.build_signal_rows(signals, depth, only_triggers=False)

    assert rows == [
        {
            "time_ms": 1783217700000,
            "time": "2026-07-05T02:15:00+00:00",
            "price": 61000.5,
            "ch1": 0.11,
            "ch2": 0.22,
            "ch3": 0.33,
            "ch4": -0.44,
            "signal_value": 0.55,
            "threshold": 0.5,
            "trigger": 1,
            "score_delta": None,
            "dominant_channel": None,
            "nonzero_channel_count": 0,
            "data_quality": None,
            "regime_model": None,
            "l2_bid_depth": 12.0,
            "l2_ask_depth": 15.0,
            "l2_total_depth": 27.0,
            "l2_total_depth_5bps": 5.0,
            "l2_total_depth_10bps": 10.0,
            "spread": 0.1,
            "mid_price": 61000.4,
        }
    ]


def test_paperlike_signal_requires_multichannel_rising_edge_and_gap() -> None:
    dashboard = load_module()
    ohlcv = make_ohlcv()
    depth = make_depth()
    depth.loc[45:55, "total_depth"] = [18, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4]
    half_depth = (depth.loc[45:55, "total_depth"] / 2).to_numpy()
    depth.loc[45:55, "bid_depth"] = half_depth
    depth.loc[45:55, "ask_depth"] = half_depth
    depth.loc[45:55, "spread"] = [0.12, 0.14, 0.17, 0.20, 0.24, 0.29, 0.34, 0.39, 0.45, 0.52, 0.60]
    ohlcv.loc[45:55, "buy_volume"] = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ohlcv.loc[45:55, "sell_volume"] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    signals = dashboard.compute_signals(ohlcv, depth, algo_version=dashboard.PAPERLIKE_ALGO_VERSION)

    assert signals["trigger"].sum() == 1
    triggered = signals[signals["trigger"] == 1].iloc[0]
    assert triggered["score_delta"] > 0
    assert triggered["nonzero_channel_count"] >= 2
    assert triggered["dominant_channel"] in {"ch1", "ch2", "ch3", "ch4"}
    assert signals.loc[signals.index > triggered.name, "trigger"].head(5).sum() == 0


def test_paperlike_signal_does_not_fire_on_single_ofi_channel() -> None:
    dashboard = load_module()
    ohlcv = make_ohlcv()
    depth = make_depth()
    ohlcv.loc[40:60, "buy_volume"] = 19.0
    ohlcv.loc[40:60, "sell_volume"] = 1.0

    signals = dashboard.compute_signals(ohlcv, depth, algo_version=dashboard.PAPERLIKE_ALGO_VERSION)

    assert signals["ch4"].max() > 0.5
    assert "stale_depth" in set(signals["data_quality"])
    assert signals["trigger"].sum() == 0


def test_paperlike_signal_uses_hmm_when_available(monkeypatch) -> None:
    dashboard = load_module()
    ohlcv = make_ohlcv(70)
    depth = make_depth(70)
    depth["total_depth"] = [20.0 - min(i, 40) * 0.1 for i in range(70)]
    depth["spread"] = [0.10 + min(i, 40) * 0.002 for i in range(70)]

    class FakeHMM:
        def __init__(self, **kwargs):
            self.means_ = None

        def fit(self, features):
            self.means_ = pd.DataFrame(features).iloc[[0, len(features) // 2, -1]].to_numpy()
            return self

        def predict_proba(self, features):
            rows = []
            for i in range(len(features)):
                if i < 20:
                    rows.append([0.90, 0.08, 0.02])
                elif i < 45:
                    rows.append([0.25, 0.55, 0.20])
                else:
                    rows.append([0.10, 0.20, 0.70])
            return pd.DataFrame(rows).to_numpy()

    monkeypatch.setattr(dashboard, "GaussianHMM", FakeHMM)

    signals = dashboard.compute_signals(ohlcv, depth, algo_version=dashboard.PAPERLIKE_ALGO_VERSION)

    assert "hmm" in set(signals["regime_model"])
    assert signals.loc[30:40, "ch1"].mean() > signals.loc[0:10, "ch1"].mean()


def test_build_snapshot_can_request_historical_file_scan(monkeypatch) -> None:
    dashboard = load_module()
    calls: list[bool] = []

    def fake_exchange_payload(exchange: str, minutes: int, *, algo_version: str, live_only: bool):
        calls.append(live_only)
        return {"pair": "BTC_USDT", "series": [], "algo_version": algo_version}

    monkeypatch.setattr(dashboard, "build_exchange_payload", fake_exchange_payload)

    snapshot = dashboard.build_snapshot(480, algo_version=dashboard.PAPERLIKE_ALGO_VERSION, live_only=False)

    assert snapshot["algo_version"] == dashboard.PAPERLIKE_ALGO_VERSION
    assert calls == [False, False]


def test_html_refreshes_every_minute_and_names_required_fields() -> None:
    dashboard = load_module()

    html = dashboard.HTML_TEMPLATE

    assert "const REFRESH_MS = 60000;" in html
    assert "发出时间" in html
    assert "信号值" in html
    assert "CH1" in html
    assert "CH2" in html
    assert "CH3" in html
    assert "CH4" in html
    assert "L2总深度" in html
    assert "5bps深度" in html
    assert "10bps深度" in html


def test_flatten_signal_features_includes_exchange_pair_and_algo_version() -> None:
    dashboard = load_module()
    snapshot = {
        "exchanges": {
            "gate": {
                "pair": "BTC_USDT",
                "series": [
                    {
                        "time_ms": 1783217700000,
                        "time": "2026-07-05T02:15:00+00:00",
                        "price": 61000.5,
                        "ch1": 0.11,
                        "ch2": 0.22,
                        "ch3": 0.33,
                        "ch4": -0.44,
                        "signal_value": 0.55,
                        "threshold": 0.5,
                        "trigger": 1,
                        "score_delta": 0.03,
                        "dominant_channel": "ch2",
                        "nonzero_channel_count": 3,
                        "data_quality": "ok",
                        "regime_model": "hmm",
                        "l2_bid_depth": 12.0,
                        "l2_ask_depth": 15.0,
                        "l2_total_depth": 27.0,
                        "l2_total_depth_5bps": 5.0,
                        "l2_total_depth_10bps": 10.0,
                        "spread": 0.1,
                        "mid_price": 61000.4,
                    }
                ],
            }
        }
    }

    rows = dashboard.flatten_signal_features(snapshot, algo_version="test_v1")

    assert rows[0]["exchange"] == "gate"
    assert rows[0]["pair"] == "BTC_USDT"
    assert rows[0]["algo_version"] == "test_v1"
    assert rows[0]["event_time"] == "2026-07-05T02:15:00+00:00"
    assert rows[0]["signal_value"] == 0.55
    assert rows[0]["l2_total_depth_10bps"] == 10.0
    assert rows[0]["score_delta"] == 0.03
    assert rows[0]["dominant_channel"] == "ch2"
    assert rows[0]["nonzero_channel_count"] == 3
    assert rows[0]["data_quality"] == "ok"
    assert rows[0]["regime_model"] == "hmm"


def test_persist_signal_features_to_duckdb_is_idempotent(tmp_path: Path) -> None:
    dashboard = load_module()
    db_path = tmp_path / "btc_microstructure.duckdb"
    rows = [
        {
            "event_time": "2026-07-05T02:15:00+00:00",
            "time_ms": 1783217700000,
            "exchange": "gate",
            "pair": "BTC_USDT",
            "price": 61000.5,
            "ch1": 0.11,
            "ch2": 0.22,
            "ch3": 0.33,
            "ch4": -0.44,
            "signal_value": 0.55,
            "threshold": 0.5,
            "trigger": 1,
            "score_delta": 0.03,
            "dominant_channel": "ch2",
            "nonzero_channel_count": 3,
            "data_quality": "ok",
            "regime_model": "hmm",
            "l2_bid_depth": 12.0,
            "l2_ask_depth": 15.0,
            "l2_total_depth": 27.0,
            "l2_total_depth_5bps": 5.0,
            "l2_total_depth_10bps": 10.0,
            "spread": 0.1,
            "mid_price": 61000.4,
            "algo_version": "test_v1",
            "created_at": "2026-07-05T02:16:00+00:00",
        }
    ]

    dashboard.persist_signal_feature_rows(rows, db_path=db_path)
    dashboard.persist_signal_feature_rows(rows, db_path=db_path)

    with duckdb.connect(str(db_path), read_only=True) as con:
        count = con.execute("select count(*) from btc_signal_features_1m").fetchone()[0]
        saved = con.execute(
            """
            select exchange, pair, signal_value, l2_total_depth, score_delta,
                   dominant_channel, nonzero_channel_count, data_quality, regime_model
            from btc_signal_features_1m
            """
        ).fetchone()

    assert count == 1
    assert saved == ("gate", "BTC_USDT", 0.55, 27.0, 0.03, "ch2", 3, "ok", "hmm")
