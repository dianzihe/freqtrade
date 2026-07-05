from types import SimpleNamespace

import numpy as np
import pandas as pd

from user_data.strategies.lob_latent_regime_v7_strategy import (
    LOBLatentRegimeV7Strategy,
)


def _strategy() -> LOBLatentRegimeV7Strategy:
    return LOBLatentRegimeV7Strategy(config={"candle_type_def": "futures"})


def _base_frame(rows: int = 140) -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 101.0, rows), dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=rows, freq="5min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0,
            "buy_volume": 560.0,
            "sell_volume": 440.0,
            "bid_depth_10": 500.0,
            "ask_depth_10": 500.0,
            "best_bid": close - 0.02,
            "best_ask": close + 0.02,
        }
    )


def test_indicators_add_v7_lob_columns() -> None:
    strategy = _strategy()

    result = strategy.populate_indicators(_base_frame(), {"pair": "BTC/USDT:USDT"})

    expected = {
        "lob_entropy",
        "lob_prestress_proxy",
        "lob_spread_drift",
        "lob_depth_erosion",
        "lob_ofi_momentum",
        "lob_score",
        "lob_score_delta",
        "lob_threshold",
        "lob_trigger",
    }
    assert expected.issubset(result.columns)
    assert result["lob_score"].dropna().between(0.0, 1.35).all()


def test_l2_snapshots_drive_depth_erosion_and_spread_drift() -> None:
    strategy = _strategy()
    dataframe = _base_frame()
    dataframe = dataframe.drop(
        columns=["bid_depth_10", "ask_depth_10", "best_bid", "best_ask"]
    )
    normal_bids = [[100.0 - idx * 0.01, 40.0] for idx in range(25)]
    normal_asks = [[100.1 + idx * 0.01, 40.0] for idx in range(25)]
    thin_bids = [[98.5 - idx * 0.01, 4.0] for idx in range(25)]
    wide_asks = [[101.5 + idx * 0.01, 4.0] for idx in range(25)]
    dataframe["bids"] = [normal_bids for _ in range(len(dataframe))]
    dataframe["asks"] = [normal_asks for _ in range(len(dataframe))]
    dataframe.loc[95:, "bids"] = pd.Series(
        [thin_bids for _ in range(len(dataframe) - 95)],
        index=dataframe.index[95:],
    )
    dataframe.loc[95:, "asks"] = pd.Series(
        [wide_asks for _ in range(len(dataframe) - 95)],
        index=dataframe.index[95:],
    )

    result = strategy.populate_indicators(dataframe, {"pair": "BTC/USDT:USDT"})

    assert result.loc[110, "lob_depth_erosion"] > 0.75
    assert result.loc[110, "lob_spread_drift"] > 0.75


def test_trigger_confirmation_and_cooldown_deduplicates_events() -> None:
    strategy = _strategy()
    dataframe = _base_frame()
    dataframe["lob_entropy"] = 0.0
    dataframe["lob_prestress_proxy"] = 0.0
    dataframe["lob_spread_drift"] = 0.0
    dataframe["lob_depth_erosion"] = 0.0
    dataframe["lob_ofi_momentum"] = 0.0
    dataframe.loc[50:53, "lob_depth_erosion"] = [0.70, 0.78, 0.86, 0.94]
    dataframe.loc[58:61, "lob_depth_erosion"] = [0.72, 0.80, 0.88, 0.96]
    dataframe.loc[88:91, "lob_depth_erosion"] = [0.74, 0.82, 0.90, 0.98]

    result = strategy._build_trigger(
        dataframe,
        window=10,
        signal_pct=65,
        confirm_bars=2,
        min_strength=0.20,
        cooldown_bars=25,
    )
    trigger_positions = np.flatnonzero(result["lob_trigger"].to_numpy())

    assert len(trigger_positions) >= 2
    assert all(
        later - earlier > 25
        for earlier, later in zip(trigger_positions, trigger_positions[1:])
    )


def test_bearish_stress_trigger_enters_short() -> None:
    strategy = _strategy()
    dataframe = _base_frame()
    dataframe["ema_slow"] = dataframe["close"] + 1.0
    dataframe["lob_trigger"] = 0
    dataframe["lob_score"] = 0.0
    dataframe["volume"] = 1000.0
    dataframe.loc[dataframe.index[-6:], "close"] = [101.0, 100.6, 100.2, 99.8, 99.4, 99.0]
    dataframe.loc[dataframe.index[-1], "lob_trigger"] = 1
    dataframe.loc[dataframe.index[-1], "lob_score"] = 0.9

    result = strategy.populate_entry_trend(
        dataframe,
        {"pair": "BTC/USDT:USDT"},
    )

    assert result.loc[dataframe.index[-1], "enter_short"] == 1
    assert result.loc[dataframe.index[-1], "enter_tag"] == "lob_v7_stress_short"
