from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.trend_filtered_grid_strategy import TrendFilteredGridStrategy


def _strategy() -> TrendFilteredGridStrategy:
    return TrendFilteredGridStrategy(config={"candle_type_def": "spot"})


def _base_frame(rows: int = 220) -> pd.DataFrame:
    close = pd.Series([100 + (idx * 0.03) for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_indicators_include_grid_and_risk_columns() -> None:
    strategy = _strategy()

    dataframe = strategy.populate_indicators(_base_frame(), {"pair": "BTC/USDT"})

    expected = {
        "ema_grid",
        "atr_pct",
        "grid_step_pct",
        "grid_pullback_level",
        "volume_ratio",
        "daily_drop_pct",
    }
    assert expected.issubset(dataframe.columns)
    assert dataframe["grid_step_pct"].fillna(0).ge(0).all()


def test_entry_requires_uptrend_and_grid_pullback() -> None:
    strategy = _strategy()
    dataframe = _base_frame()
    dataframe["trend_up_15m"] = 1
    dataframe["trend_weak_15m"] = 0
    dataframe["cooldown_risk"] = 0
    dataframe["daily_drop_pct"] = 0.0
    dataframe["ema_grid"] = 100.0
    dataframe["ema_fast"] = 100.2
    dataframe["ema_slow"] = 100.0
    dataframe["atr_pct"] = 0.006
    dataframe["grid_step_pct"] = 0.004
    dataframe["grid_pullback_level"] = 99.6
    dataframe["volume_ratio"] = 1.2
    dataframe["rsi"] = 52.0
    dataframe.loc[dataframe.index[-1], "low"] = 99.55
    dataframe.loc[dataframe.index[-1], "close"] = 99.90

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[dataframe.index[-1], "enter_long"] == 1
    assert result.loc[dataframe.index[-1], "enter_tag"] == "trend_grid_pullback"


def test_entry_pauses_when_higher_timeframe_trend_is_weak() -> None:
    strategy = _strategy()
    dataframe = _base_frame()
    dataframe["trend_up_15m"] = 0
    dataframe["trend_weak_15m"] = 1
    dataframe["cooldown_risk"] = 0
    dataframe["daily_drop_pct"] = 0.0
    dataframe["ema_grid"] = 100.0
    dataframe["ema_fast"] = 100.2
    dataframe["ema_slow"] = 100.0
    dataframe["atr_pct"] = 0.006
    dataframe["grid_step_pct"] = 0.004
    dataframe["grid_pullback_level"] = 99.6
    dataframe["volume_ratio"] = 1.2
    dataframe["rsi"] = 52.0
    dataframe.loc[dataframe.index[-1], "low"] = 99.55
    dataframe.loc[dataframe.index[-1], "close"] = 99.90

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result["enter_long"].sum() == 0


def test_position_management_adds_only_while_profitable() -> None:
    strategy = _strategy()
    trade = SimpleNamespace(nr_of_successful_entries=1, stake_amount=100.0)

    losing = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=99.0,
        current_profit=-0.01,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=99.0,
        current_exit_rate=99.0,
        current_entry_profit=-0.01,
        current_exit_profit=-0.01,
    )
    winning = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=101.0,
        current_profit=0.018,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=101.0,
        current_exit_rate=101.0,
        current_entry_profit=0.018,
        current_exit_profit=0.018,
    )

    assert losing is None
    assert winning == (35.0, "grid_profit_add_1")
