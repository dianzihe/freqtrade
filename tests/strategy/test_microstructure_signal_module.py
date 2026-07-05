from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "user_data" / "strategies" / "微结构_信号模块.py"


def load_module():
    spec = importlib.util.spec_from_file_location("microstructure_signal_module", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_frame(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": np.linspace(100.0, 101.0, rows),
            "volume": np.full(rows, 10.0),
            "buy_volume": np.full(rows, 6.0),
            "sell_volume": np.full(rows, 4.0),
            "bid_depth_10": np.full(rows, 500.0),
            "ask_depth_10": np.full(rows, 500.0),
            "best_bid": np.full(rows, 99.95),
            "best_ask": np.full(rows, 100.05),
        }
    )


def test_ch4_order_flow_keeps_signed_net_order_flow_zscore() -> None:
    module = load_module()
    df = _base_frame()
    df.loc[60:, "buy_volume"] = 3.0
    df.loc[60:, "sell_volume"] = 9.0

    result = module.add_lob_regime_signals(
        df,
        lookback_period=10,
        threshold_percentile=80,
        confirmation_bars=1,
        min_signal_strength=0.0,
    )

    assert result.loc[20, "ch4_order_flow"] >= 0
    assert result.loc[65, "ch4_order_flow"] < 0
    assert result["ch4_order_flow"].min() < -1.0


def test_ch2_ch3_use_l2_depth_and_best_quote_spread_not_proxy_columns() -> None:
    module = load_module()
    df = _base_frame()
    df["trade_intensity"] = 999.0
    df["price_std"] = 0.0
    df.loc[55:, ["bid_depth_10", "ask_depth_10"]] = 50.0
    df.loc[55:, "best_bid"] = 99.0
    df.loc[55:, "best_ask"] = 101.0

    result = module.add_lob_regime_signals(
        df,
        lookback_period=10,
        threshold_percentile=80,
        confirmation_bars=1,
        min_signal_strength=0.0,
    )

    assert result.loc[65, "ch2_depth_erosion"] > 0.8
    assert result.loc[65, "ch3_spread_drift"] > 0.8


def test_ch2_uses_second_level_full_book_depth_columns() -> None:
    module = load_module()
    df = _base_frame()
    df = df.drop(columns=["bid_depth_10", "ask_depth_10"])
    df["bid_depth"] = 500.0
    df["ask_depth"] = 500.0
    df["total_depth"] = 1000.0
    df.loc[55:, "bid_depth"] = 60.0
    df.loc[55:, "ask_depth"] = 40.0
    df.loc[55:, "total_depth"] = 100.0

    result = module.add_lob_regime_signals(
        df,
        lookback_period=10,
        threshold_percentile=80,
        confirmation_bars=1,
        min_signal_strength=0.0,
    )

    assert result.loc[65, "ch2_depth_erosion"] > 0.8


def test_ch2_ch3_extract_depth_and_spread_from_raw_l2_snapshots() -> None:
    module = load_module()
    df = _base_frame()
    df = df.drop(columns=["bid_depth_10", "ask_depth_10", "best_bid", "best_ask"])
    normal_bids = [[100 - idx * 0.01, 50.0] for idx in range(25)]
    normal_asks = [[100.1 + idx * 0.01, 50.0] for idx in range(25)]
    thin_bids = [[99 - idx * 0.01, 5.0] for idx in range(25)]
    wide_asks = [[101 + idx * 0.01, 5.0] for idx in range(25)]
    df["bids"] = [normal_bids for _ in range(len(df))]
    df["asks"] = [normal_asks for _ in range(len(df))]
    df.loc[55:, "bids"] = pd.Series([thin_bids for _ in range(len(df) - 55)], index=df.index[55:])
    df.loc[55:, "asks"] = pd.Series([wide_asks for _ in range(len(df) - 55)], index=df.index[55:])

    result = module.add_lob_regime_signals(
        df,
        lookback_period=10,
        threshold_percentile=80,
        confirmation_bars=1,
        min_signal_strength=0.0,
    )

    assert result.loc[65, "ch2_depth_erosion"] > 0.8
    assert result.loc[65, "ch3_spread_drift"] > 0.8


def test_raw_l2_depth_uses_twenty_five_levels_not_ten() -> None:
    module = load_module()
    df = _base_frame()
    df = df.drop(columns=["bid_depth_10", "ask_depth_10", "best_bid", "best_ask"])
    normal_bids = [[100 - idx * 0.01, 0.0 if idx < 10 else 50.0] for idx in range(25)]
    normal_asks = [[100.1 + idx * 0.01, 0.0 if idx < 10 else 50.0] for idx in range(25)]
    thin_bids = [[99 - idx * 0.01, 0.0 if idx < 10 else 5.0] for idx in range(25)]
    wide_asks = [[101 + idx * 0.01, 0.0 if idx < 10 else 5.0] for idx in range(25)]
    df["bids"] = [normal_bids for _ in range(len(df))]
    df["asks"] = [normal_asks for _ in range(len(df))]
    df.loc[55:, "bids"] = pd.Series([thin_bids for _ in range(len(df) - 55)], index=df.index[55:])
    df.loc[55:, "asks"] = pd.Series([wide_asks for _ in range(len(df) - 55)], index=df.index[55:])

    result = module.add_lob_regime_signals(
        df,
        lookback_period=10,
        threshold_percentile=80,
        confirmation_bars=1,
        min_signal_strength=0.0,
    )

    assert result.loc[65, "ch2_depth_erosion"] > 0.8


def test_composite_trigger_defaults_to_twenty_five_bar_validity_window() -> None:
    module = load_module()
    df = _base_frame()
    df["ch1_vol_entropy"] = 0.0
    df["ch2_depth_erosion"] = 0.0
    df["ch3_spread_drift"] = 0.0
    df["ch4_order_flow"] = 0.0
    df.loc[20:24, "ch2_depth_erosion"] = [0.80, 0.86, 0.92, 0.98, 1.0]
    df.loc[28:32, "ch2_depth_erosion"] = [0.80, 0.86, 0.92, 0.98, 1.0]
    df.loc[48:52, "ch2_depth_erosion"] = [0.80, 0.86, 0.92, 0.98, 1.0]

    result = module._composite_trigger(
        df,
        window=5,
        percentile=50,
        n_confirm=1,
        min_floor=0.1,
    )

    trigger_positions = np.flatnonzero(result["signal_trigger"].to_numpy())

    assert len(trigger_positions) >= 1
    assert all(
        later - earlier > 25
        for earlier, later in zip(trigger_positions, trigger_positions[1:])
    )
    first_trigger = trigger_positions[0]
    assert result.iloc[first_trigger + 1 : first_trigger + 26]["signal_trigger"].sum() == 0
