from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.gate_xinghe_futures_strategy import GateXingheFuturesStrategy


def _strategy() -> GateXingheFuturesStrategy:
    return GateXingheFuturesStrategy(config={"candle_type_def": "futures"})


def _signal_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 99.5],
            "volume": [1000.0, 1000.0],
            "ema_fast": [102.0, 98.0],
            "ema_slow": [101.0, 99.0],
            "ema_trend": [100.0, 100.0],
            "trend_slope": [0.002, -0.002],
            "atr_pct": [0.008, 0.008],
            "grid_step_pct": [0.01, 0.01],
            "long_pullback_level": [100.0, 100.0],
            "short_rebound_level": [100.0, 100.0],
            "rsi": [48.0, 52.0],
            "volume_ratio": [1.1, 1.1],
            "risk_fuse": [0, 0],
            "funding_open": [0.0001, -0.0001],
        }
    )
    frame.loc[0, ["low", "close"]] = [99.8, 100.4]
    frame.loc[1, ["high", "close"]] = [100.2, 99.6]
    return frame


def test_futures_emits_symmetric_long_and_short_signals() -> None:
    result = _strategy().populate_entry_trend(_signal_frame(), {"pair": "BTC/USDT:USDT"})

    assert result.loc[0, "enter_long"] == 1
    assert result.loc[0, "enter_short"] == 0
    assert result.loc[1, "enter_short"] == 1
    assert result.loc[1, "enter_long"] == 0


def test_futures_blocks_direction_that_pays_extreme_funding() -> None:
    strategy = _strategy()
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-06-01 00:00:00+00:00", "2026-06-01 08:00:00+00:00"]
            ),
            "open": [0.0001, 0.0015],
        }
    )

    class FakeDataProvider:
        def get_pair_dataframe(self, pair, timeframe, candle_type):
            assert candle_type == "funding_rate"
            return funding

        def get_funding_rate_timeframe(self):
            return "1h"

    strategy.dp = FakeDataProvider()
    before_extreme = strategy.confirm_trade_entry(
        pair="BTC/USDT:USDT",
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=datetime(2026, 6, 1, 4, tzinfo=UTC),
        entry_tag="futures_long_pullback",
        side="long",
    )
    after_extreme = strategy.confirm_trade_entry(
        pair="BTC/USDT:USDT",
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=datetime(2026, 6, 1, 9, tzinfo=UTC),
        entry_tag="futures_long_pullback",
        side="long",
    )

    assert before_extreme is True
    assert after_extreme is False


def test_futures_leverage_and_dca_are_strictly_limited() -> None:
    strategy = _strategy()
    now = datetime(2026, 6, 1, tzinfo=UTC)
    leverage = strategy.leverage(
        pair="BTC/USDT:USDT",
        current_time=now,
        current_rate=100.0,
        proposed_leverage=5.0,
        max_leverage=20.0,
        entry_tag="futures_long_pullback",
        side="long",
    )
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
    second = strategy.adjust_trade_position(
        trade=SimpleNamespace(nr_of_successful_entries=2, stake_amount=100.0, is_short=True),
        current_time=now,
        current_rate=106.0,
        current_profit=-0.06,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=106.0,
        current_exit_rate=106.0,
        current_entry_profit=-0.06,
        current_exit_profit=-0.06,
    )
    exhausted = strategy.adjust_trade_position(
        trade=SimpleNamespace(nr_of_successful_entries=3, stake_amount=100.0, is_short=False),
        current_time=now,
        current_rate=90.0,
        current_profit=-0.10,
        min_stake=5.0,
        max_stake=1000.0,
        current_entry_rate=90.0,
        current_exit_rate=90.0,
        current_entry_profit=-0.10,
        current_exit_profit=-0.10,
    )

    assert leverage == 2.0
    assert first == (55.0, "futures_grid_add_1")
    assert second == (35.0, "futures_grid_add_2")
    assert exhausted is None
    assert strategy.max_entry_position_adjustment == 2
    assert strategy.stoploss < 0


def test_futures_dca_converts_first_order_notional_to_margin() -> None:
    strategy = _strategy()
    trade = SimpleNamespace(
        nr_of_successful_entries=2,
        stake_amount=155.0,
        is_short=False,
        entry_side="buy",
        leverage=2.0,
        select_filled_orders=lambda side: [SimpleNamespace(safe_cost=200.0)],
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

    assert adjustment == (35.0, "futures_grid_add_2")
