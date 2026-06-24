from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from user_data.strategies.gate_xinghe_spot_strategy import GateXingheSpotStrategy


def _frame(rows: int = 260) -> pd.DataFrame:
    close = pd.Series([100 + idx * 0.02 for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1000.0,
        }
    )


def _strategy() -> GateXingheSpotStrategy:
    return GateXingheSpotStrategy(config={"candle_type_def": "spot"})


def test_spot_indicators_are_finite_and_include_grid_context() -> None:
    result = _strategy().populate_indicators(_frame(), {"pair": "BTC/USDT"})

    expected = {
        "ema_fast",
        "ema_slow",
        "ema_trend",
        "atr_pct",
        "grid_step_pct",
        "volume_ratio",
        "risk_fuse",
    }
    assert expected.issubset(result.columns)
    assert result[list(expected)].replace([float("inf"), float("-inf")], pd.NA).notna().all().all()


def test_spot_entry_is_long_only_and_blocked_by_risk_fuse() -> None:
    strategy = _strategy()
    frame = _frame()
    frame["ema_fast"] = 102.0
    frame["ema_slow"] = 101.0
    frame["ema_trend"] = 100.0
    frame["trend_slope"] = 0.002
    frame["atr_pct"] = 0.008
    frame["grid_step_pct"] = 0.01
    frame["pullback_level"] = 100.0
    frame["rsi"] = 48.0
    frame["volume_ratio"] = 1.1
    frame["risk_fuse"] = 0
    frame.loc[frame.index[-1], ["low", "close"]] = [99.8, 100.4]

    allowed = strategy.populate_entry_trend(frame.copy(), {"pair": "BTC/USDT"})
    frame.loc[frame.index[-1], "risk_fuse"] = 1
    blocked = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT"})

    assert allowed.iloc[-1]["enter_long"] == 1
    assert "enter_short" not in allowed or allowed.iloc[-1]["enter_short"] == 0
    assert blocked.iloc[-1]["enter_long"] == 0


def test_spot_dca_is_limited_widening_and_decreasing() -> None:
    strategy = _strategy()
    now = datetime(2026, 6, 1, tzinfo=UTC)

    first = strategy.adjust_trade_position(
        trade=SimpleNamespace(nr_of_successful_entries=1, stake_amount=100.0, is_short=False),
        current_time=now,
        current_rate=98.0,
        current_profit=-0.025,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=98.0,
        current_exit_rate=98.0,
        current_entry_profit=-0.025,
        current_exit_profit=-0.025,
    )
    too_shallow_second = strategy.adjust_trade_position(
        trade=SimpleNamespace(nr_of_successful_entries=2, stake_amount=100.0, is_short=False),
        current_time=now,
        current_rate=97.5,
        current_profit=-0.025,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=97.5,
        current_exit_rate=97.5,
        current_entry_profit=-0.025,
        current_exit_profit=-0.025,
    )
    second = strategy.adjust_trade_position(
        trade=SimpleNamespace(nr_of_successful_entries=2, stake_amount=100.0, is_short=False),
        current_time=now,
        current_rate=95.0,
        current_profit=-0.055,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=95.0,
        current_exit_rate=95.0,
        current_entry_profit=-0.055,
        current_exit_profit=-0.055,
    )
    exhausted = strategy.adjust_trade_position(
        trade=SimpleNamespace(nr_of_successful_entries=4, stake_amount=100.0, is_short=False),
        current_time=now,
        current_rate=85.0,
        current_profit=-0.15,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=85.0,
        current_exit_rate=85.0,
        current_entry_profit=-0.15,
        current_exit_profit=-0.15,
    )

    assert first == (60.0, "spot_grid_add_1")
    assert too_shallow_second is None
    assert second == (45.0, "spot_grid_add_2")
    assert exhausted is None
    assert strategy.max_entry_position_adjustment == 3
    assert strategy.stoploss < 0
    assert len(strategy.protections) >= 3


def test_spot_detects_moderate_pullbacks_in_a_rising_market() -> None:
    strategy = _strategy()
    rows = 600
    trend = np.linspace(100.0, 106.0, rows)
    pullbacks = np.where(np.arange(rows) % 45 == 0, -0.004, 0.0)
    close = pd.Series(trend * (1 + pullbacks))
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.0015,
            "low": close * 0.997,
            "close": close,
            "volume": 1000.0,
        }
    )

    frame = strategy.populate_indicators(frame, {"pair": "BTC/USDT"})
    result = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT"})

    assert result["enter_long"].sum() > 0


def test_spot_dca_uses_first_order_cost_not_cumulative_trade_stake() -> None:
    strategy = _strategy()
    trade = SimpleNamespace(
        nr_of_successful_entries=2,
        stake_amount=160.0,
        is_short=False,
        entry_side="buy",
        leverage=1.0,
        select_filled_orders=lambda side: [SimpleNamespace(safe_cost=100.0)],
    )

    adjustment = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=UTC),
        current_rate=95.0,
        current_profit=-0.055,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=95.0,
        current_exit_rate=95.0,
        current_entry_profit=-0.055,
        current_exit_profit=-0.055,
    )

    assert adjustment == (45.0, "spot_grid_add_2")
