from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.lottery_ticket_momentum_strategy import (
    LotteryTicketMomentumStrategy,
)


def _frame(rows: int = 320) -> pd.DataFrame:
    close = pd.Series([1.0 + idx * 0.001 for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="5min", tz="UTC"),
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [1000.0] * rows,
        }
    )


def test_lottery_entry_requires_volume_momentum_and_limited_pullback() -> None:
    strategy = LotteryTicketMomentumStrategy(config={})
    df = strategy.populate_indicators(_frame(), {"pair": "RIF/USDT"})
    last = df.index[-1]
    df.loc[last, "ret_24h"] = 0.35
    df.loc[last, "range_24h"] = 0.55
    df.loc[last, "off_high_24h"] = -0.12
    df.loc[last, "volume_ratio"] = 5.0
    df.loc[last, "quote_volume_24h"] = 30000.0
    df.loc[last, "close"] = df.loc[last, "ema20"] * 1.03
    df.loc[last, "wick_top_ratio"] = 0.15

    result = strategy.populate_entry_trend(df.copy(), {"pair": "RIF/USDT"})
    broken = df.copy()
    broken.loc[last, "off_high_24h"] = -0.70
    broken_result = strategy.populate_entry_trend(broken, {"pair": "RIF/USDT"})

    assert result.loc[last, "enter_long"] == 1
    assert result.loc[last, "enter_tag"] == "lottery_momentum_a"
    assert broken_result.loc[last, "enter_long"] == 0


def test_lottery_stake_is_one_dollar_ticket() -> None:
    strategy = LotteryTicketMomentumStrategy(config={})

    stake = strategy.custom_stake_amount(
        pair="RIF/USDT",
        current_time=datetime.now(timezone.utc),
        current_rate=1.0,
        proposed_stake=50.0,
        min_stake=None,
        max_stake=100.0,
        leverage=1.0,
        entry_tag="lottery_momentum_a",
        side="long",
    )

    assert stake == 1.0


def test_confirm_trade_entry_limits_daily_and_total_tickets() -> None:
    strategy = LotteryTicketMomentumStrategy(config={})
    now = datetime(2026, 7, 2, 10, 0, tzinfo=timezone.utc)

    accepted = [
        strategy.confirm_trade_entry("RIF/USDT", "limit", 1.0, 1.0, "GTC", now)
        for _ in range(6)
    ]

    assert accepted == [True, True, True, True, True, False]

    for day in range(3, 6):
        current_day = datetime(2026, 7, day, 10, 0, tzinfo=timezone.utc)
        for _ in range(5):
            assert strategy.confirm_trade_entry("HFT/USDT", "limit", 1.0, 1.0, "GTC", current_day)

    final_day = datetime(2026, 7, 6, 10, 0, tzinfo=timezone.utc)
    assert strategy.confirm_trade_entry("TA/USDT", "limit", 1.0, 1.0, "GTC", final_day) is False


def test_custom_exit_cleans_up_stale_and_underperforming_tickets() -> None:
    strategy = LotteryTicketMomentumStrategy(config={})
    now = datetime(2026, 7, 6, 10, 0, tzinfo=timezone.utc)

    max_hold_trade = SimpleNamespace(open_date_utc=now - timedelta(days=4, minutes=5))
    not_started_trade = SimpleNamespace(open_date_utc=now - timedelta(hours=25))
    weak_trade = SimpleNamespace(open_date_utc=now - timedelta(hours=49))
    healthy_trade = SimpleNamespace(open_date_utc=now - timedelta(hours=30))

    assert (
        strategy.custom_exit("RIF/USDT", max_hold_trade, now, current_rate=1.0, current_profit=0.50)
        == "max_4d_hold"
    )
    assert (
        strategy.custom_exit("RIF/USDT", not_started_trade, now, current_rate=1.0, current_profit=0.02)
        == "stale_not_started"
    )
    assert (
        strategy.custom_exit("RIF/USDT", weak_trade, now, current_rate=1.0, current_profit=0.08)
        == "stale_weak_profit"
    )
    assert strategy.custom_exit("RIF/USDT", healthy_trade, now, current_rate=1.0, current_profit=0.06) is None
