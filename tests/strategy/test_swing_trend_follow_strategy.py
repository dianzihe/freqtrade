from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.swing_trend_follow_strategy import SwingTrendFollowStrategy


def _base_frame(rows: int = 220) -> pd.DataFrame:
    idx = pd.Series(range(rows), dtype="float64")
    close = 100 + idx * 0.05
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.02,
            "high": close + 0.30,
            "low": close - 0.30,
            "close": close,
            "volume": 1000 + idx,
        }
    )


def _entry_ready_frame() -> pd.DataFrame:
    dataframe = _base_frame()
    last = dataframe.index[-1]
    dataframe["ema_fast"] = 110.0
    dataframe["ema_slow"] = 107.0
    dataframe["ema_trend"] = 103.0
    dataframe["trend_strength"] = 0.028
    dataframe["rsi"] = 61.0
    dataframe["atr_pct"] = 0.012
    dataframe["volume_ratio"] = 1.55
    dataframe["momentum_3"] = 0.006
    dataframe["higher_high"] = 1
    dataframe["higher_low"] = 1
    dataframe["pullback_holds_structure"] = 1
    dataframe["reclaim_after_pullback"] = 1
    dataframe["market_circuit_breaker"] = 0
    dataframe.loc[last, "close"] = 111.0
    dataframe.loc[last, "open"] = 110.4
    return dataframe


def test_entry_requires_higher_high_higher_low_and_pullback_hold() -> None:
    strategy = SwingTrendFollowStrategy(config={})
    dataframe = _entry_ready_frame()

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[dataframe.index[-1], "enter_long"] == 1
    assert result.loc[dataframe.index[-1], "enter_tag"] == "structure_pullback_reclaim"


def test_entry_rejects_broken_structure_or_weak_volume() -> None:
    strategy = SwingTrendFollowStrategy(config={})
    dataframe = _entry_ready_frame()
    last = dataframe.index[-1]

    dataframe.loc[last, "pullback_holds_structure"] = 0
    broken = strategy.populate_entry_trend(dataframe.copy(), {"pair": "BTC/USDT"})

    dataframe.loc[last, "pullback_holds_structure"] = 1
    dataframe.loc[last, "volume_ratio"] = 0.95
    weak_volume = strategy.populate_entry_trend(dataframe.copy(), {"pair": "BTC/USDT"})

    assert broken.loc[last, "enter_long"] == 0
    assert weak_volume.loc[last, "enter_long"] == 0


def test_strategy_documents_trade_management_profile() -> None:
    strategy = SwingTrendFollowStrategy(config={})

    assert strategy.position_adjustment_enable is True
    assert strategy.trailing_stop is True
    assert strategy.trailing_only_offset_is_reached is True
    assert strategy.stoploss <= -0.08
    assert "strong hourly or daily uptrends" in strategy.applicable_market


def test_custom_stake_keeps_reserve_for_pyramiding() -> None:
    strategy = SwingTrendFollowStrategy(config={})

    stake = strategy.custom_stake_amount(
        pair="BTC/USDT",
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=100.0,
        proposed_stake=100.0,
        min_stake=10.0,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag="structure_pullback_reclaim",
        side="long",
    )

    assert stake == 58.0


def test_pyramid_adds_only_when_trade_is_profitable() -> None:
    strategy = SwingTrendFollowStrategy(config={})
    trade = SimpleNamespace(nr_of_successful_entries=1, stake_amount=58.0)

    losing = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=97.0,
        current_profit=-0.03,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=97.0,
        current_exit_rate=97.0,
        current_entry_profit=-0.03,
        current_exit_profit=-0.03,
    )
    winning = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=105.0,
        current_profit=0.052,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=105.0,
        current_exit_rate=105.0,
        current_entry_profit=0.052,
        current_exit_profit=0.052,
    )

    assert losing is None
    assert winning == (23.2, "pyramid_add_1")
