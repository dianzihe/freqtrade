from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.trend_pyramid_anti_martingale_strategy import (
    TrendPyramidAntiMartingaleStrategy,
)


def _trend_frame(rows: int = 260) -> pd.DataFrame:
    base = pd.Series(range(rows), dtype="float64")
    close = 100 + base * 0.08
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="5min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1000 + base * 3,
        }
    )


def test_populate_indicators_adds_trend_pyramid_columns() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})

    dataframe = strategy.populate_indicators(_trend_frame(), {"pair": "BTC/USDT"})

    expected_columns = {
        "ema_fast",
        "ema_slow",
        "ema_trend",
        "atr_pct",
        "breakout_high",
        "trend_strength",
        "volume_ratio",
    }
    assert expected_columns.issubset(dataframe.columns)
    assert dataframe["atr_pct"].fillna(0).ge(0).all()


def test_entry_signal_requires_breakout_volume_and_trend() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})
    dataframe = _trend_frame()
    dataframe["ema_fast"] = [100.0] * 259 + [119.0]
    dataframe["ema_slow"] = [99.0] * 259 + [114.0]
    dataframe["ema_trend"] = [98.0] * 259 + [110.0]
    dataframe["trend_strength"] = [0.01] * 259 + [0.04]
    dataframe["breakout_high"] = [130.0] * 259 + [120.0]
    dataframe["volume_ratio"] = [1.0] * 259 + [1.6]
    dataframe["atr_pct"] = [0.006] * 260
    dataframe["momentum_3"] = [0.0] * 259 + [0.02]
    dataframe["rsi"] = [55.0] * 260

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[259, "enter_long"] == 1
    assert result.loc[259, "enter_tag"] == "trend_breakout"


def test_pyramid_adds_only_to_profitable_trades() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})
    trade = SimpleNamespace(nr_of_successful_entries=1, stake_amount=100.0)

    losing = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=98.0,
        current_profit=-0.02,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=98.0,
        current_exit_rate=98.0,
        current_entry_profit=-0.02,
        current_exit_profit=-0.02,
    )
    winning = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=106.0,
        current_profit=0.065,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=106.0,
        current_exit_rate=106.0,
        current_entry_profit=0.065,
        current_exit_profit=0.065,
    )

    assert losing is None
    assert winning == (45.0, "pyramid_add_1")


def test_circuit_breaker_blocks_entry_after_sharp_drop() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})
    dataframe = _trend_frame()
    dataframe["ema_fast"] = [100.0] * 259 + [119.0]
    dataframe["ema_slow"] = [99.0] * 259 + [114.0]
    dataframe["ema_trend"] = [98.0] * 259 + [110.0]
    dataframe["trend_strength"] = [0.01] * 259 + [0.04]
    dataframe["breakout_high"] = [130.0] * 259 + [120.0]
    dataframe["volume_ratio"] = [1.0] * 259 + [1.6]
    dataframe["atr_pct"] = [0.006] * 260
    dataframe["momentum_3"] = [0.0] * 259 + [0.02]
    dataframe["rsi"] = [55.0] * 260
    dataframe["market_circuit_breaker"] = [0] * 259 + [1]

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[259, "enter_long"] == 0
