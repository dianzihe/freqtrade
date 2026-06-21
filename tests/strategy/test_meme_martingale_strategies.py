from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.MemeMartingaleStrategies import (
    MemeAntiMartingaleTrendStrategy,
    MemeHedgeProxyMartingaleStrategy,
    MemeLimitedDcaMartingaleStrategy,
    MemeVolatilityGridMartingaleStrategy,
)


def sample_ohlcv() -> pd.DataFrame:
    rows = 260
    base = pd.Series(range(rows), dtype="float64")
    close = 100 + base * 0.02
    close.iloc[-5:] = [101, 98, 94, 96, 99]
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.97,
            "close": close,
            "volume": 1000 + base,
        }
    )


def test_all_four_strategies_emit_required_columns() -> None:
    for strategy_cls in (
        MemeLimitedDcaMartingaleStrategy,
        MemeVolatilityGridMartingaleStrategy,
        MemeAntiMartingaleTrendStrategy,
        MemeHedgeProxyMartingaleStrategy,
    ):
        strategy = strategy_cls(config={})
        dataframe = strategy.populate_indicators(sample_ohlcv(), {"pair": "H/USDT"})
        dataframe = strategy.populate_entry_trend(dataframe, {"pair": "H/USDT"})
        dataframe = strategy.populate_exit_trend(dataframe, {"pair": "H/USDT"})

        assert {"enter_long", "exit_long", "atr_pct", "volume_ratio"}.issubset(dataframe.columns)


def test_limited_dca_uses_low_multiplier_ladder() -> None:
    strategy = MemeLimitedDcaMartingaleStrategy(config={})
    trade = SimpleNamespace(nr_of_successful_entries=2, stake_amount=100.0)

    adjustment = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=91.0,
        current_profit=-0.14,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=91.0,
        current_exit_rate=91.0,
        current_entry_profit=-0.14,
        current_exit_profit=-0.14,
    )

    assert adjustment == (160.0, "limited_dca_2")


def test_anti_martingale_adds_only_to_winners() -> None:
    strategy = MemeAntiMartingaleTrendStrategy(config={})
    trade = SimpleNamespace(nr_of_successful_entries=1, stake_amount=100.0)

    losing = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=95.0,
        current_profit=-0.05,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=95.0,
        current_exit_rate=95.0,
        current_entry_profit=-0.05,
        current_exit_profit=-0.05,
    )
    winning = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=112.0,
        current_profit=0.12,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=112.0,
        current_exit_rate=112.0,
        current_entry_profit=0.12,
        current_exit_profit=0.12,
    )

    assert losing is None
    assert winning == (50.0, "anti_martingale_profit_add_1")
