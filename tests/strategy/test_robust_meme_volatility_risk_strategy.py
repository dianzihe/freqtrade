from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.robust_meme_volatility_risk_strategy import (
    RobustMemeVolatilityRiskStrategy,
)


def _ohlcv(rows: int = 260) -> pd.DataFrame:
    close = pd.Series([100.0 + (idx * 0.01) for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": [1000.0] * rows,
        }
    )


def _entry_ready_frame() -> pd.DataFrame:
    df = _ohlcv()
    last = df.index[-1]
    df["bb_lower"] = 101.0
    df["bb_mid"] = 103.0
    df["bb_upper"] = 106.0
    df["ema_fast"] = 102.0
    df["ema_slow"] = 100.0
    df["rsi"] = 28.0
    df["atr_pct"] = 0.012
    df["volume_ratio"] = 1.4
    df["range_position"] = 0.40
    df["candle_range_pct"] = 0.01
    df["market_regime_ok"] = 1
    df["extreme_candle"] = 0
    df["lob_risk_trigger"] = False
    df["lob_score"] = 0.0
    df.loc[last, "close"] = 100.0
    df.loc[last, "low"] = 99.5
    return df


def test_entry_requires_reversion_setup_and_risk_ok() -> None:
    strategy = RobustMemeVolatilityRiskStrategy(config={})
    df = _entry_ready_frame()

    result = strategy.populate_entry_trend(df.copy(), {"pair": "H/USDT"})
    blocked = df.copy()
    blocked.loc[blocked.index[-1], "extreme_candle"] = 1
    blocked_result = strategy.populate_entry_trend(blocked, {"pair": "H/USDT"})

    assert result.loc[df.index[-1], "enter_long"] == 1
    assert result.loc[df.index[-1], "enter_tag"] == "robust_vol_reversion_long"
    assert blocked_result.loc[df.index[-1], "enter_long"] == 0


def test_confirm_trade_entry_blocks_wide_spread_and_bad_funding() -> None:
    class DummyDataProvider:
        runmode = SimpleNamespace(value="dry_run")

        def __init__(self, spread_book, funding):
            self.spread_book = spread_book
            self.funding = funding

        def orderbook(self, pair: str, maximum: int) -> dict:
            return self.spread_book

        def funding_rate(self, pair: str) -> float:
            return self.funding

    wide = {"bids": [[100.0, 1.0]], "asks": [[101.0, 1.0]]}
    tight = {"bids": [[100.0, 1.0]], "asks": [[100.05, 1.0]]}

    strategy = RobustMemeVolatilityRiskStrategy(config={})
    strategy.dp = DummyDataProvider(wide, 0.0)
    assert strategy.confirm_trade_entry("H/USDT", "limit", 100.0, 100.0, "gtc", datetime.now(timezone.utc)) is False

    strategy.dp = DummyDataProvider(tight, 0.01)
    assert strategy.confirm_trade_entry("H/USDT", "limit", 100.0, 100.0, "gtc", datetime.now(timezone.utc)) is False


def test_stake_is_capped_by_risk_and_notional_limits() -> None:
    strategy = RobustMemeVolatilityRiskStrategy(config={})

    stake = strategy.custom_stake_amount(
        pair="H/USDT",
        current_time=datetime.now(timezone.utc),
        current_rate=1.0,
        proposed_stake=2000.0,
        min_stake=10.0,
        max_stake=5000.0,
        leverage=1.0,
        entry_tag="robust_vol_reversion_long",
        side="long",
    )

    assert 10.0 <= stake <= strategy.single_trade_notional_cap.value


def test_dca_stops_when_interrupt_loss_is_hit() -> None:
    strategy = RobustMemeVolatilityRiskStrategy(config={})
    trade = SimpleNamespace(nr_of_successful_entries=2, stake_amount=100.0)

    result = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime.now(timezone.utc),
        current_rate=80.0,
        current_profit=-0.19,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=80.0,
        current_exit_rate=80.0,
        current_entry_profit=-0.19,
        current_exit_profit=-0.19,
    )

    assert result is None


def test_custom_exit_handles_time_stop_and_account_drawdown() -> None:
    strategy = RobustMemeVolatilityRiskStrategy(config={})
    stale_trade = SimpleNamespace(
        open_date_utc=datetime.now(timezone.utc) - timedelta(minutes=500),
        nr_of_successful_entries=1,
        stake_amount=100.0,
    )
    danger_trade = SimpleNamespace(
        open_date_utc=datetime.now(timezone.utc) - timedelta(minutes=30),
        nr_of_successful_entries=3,
        stake_amount=100.0,
    )

    time_exit = strategy.custom_exit(
        "H/USDT",
        stale_trade,
        datetime.now(timezone.utc),
        current_rate=99.0,
        current_profit=-0.02,
    )
    account_exit = strategy.custom_exit(
        "H/USDT",
        danger_trade,
        datetime.now(timezone.utc),
        current_rate=75.0,
        current_profit=-0.20,
    )
    strategy._portfolio_float_loss_ratio = lambda current_profit: 0.0
    dca_interrupt = strategy.custom_exit(
        "H/USDT",
        danger_trade,
        datetime.now(timezone.utc),
        current_rate=75.0,
        current_profit=-0.20,
    )

    assert time_exit == "time_stop_no_progress"
    assert account_exit == "account_float_loss_liquidation"
    assert dca_interrupt == "interrupt_after_failed_dca"


def test_leverage_is_clamped_to_strategy_cap() -> None:
    strategy = RobustMemeVolatilityRiskStrategy(config={})

    lev = strategy.leverage(
        pair="H/USDT",
        current_time=datetime.now(timezone.utc),
        current_rate=1.0,
        proposed_leverage=20.0,
        max_leverage=50.0,
        entry_tag="robust_vol_reversion_long",
        side="long",
    )

    assert lev == strategy.max_strategy_leverage.value
