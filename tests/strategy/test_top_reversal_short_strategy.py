from datetime import datetime, timezone

import pandas as pd

from user_data.strategies.top_reversal_short_strategy import TopReversalShortStrategy


def _top_frame(rows: int = 260) -> pd.DataFrame:
    base = pd.Series(range(rows), dtype="float64")
    close = 10 + base * 0.02
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.006,
            "low": close * 0.996,
            "close": close,
            "volume": [1000.0] * rows,
        }
    )


def test_populate_indicators_adds_top_reversal_columns() -> None:
    strategy = TopReversalShortStrategy(config={})

    dataframe = strategy.populate_indicators(_top_frame(), {"pair": "DN/USDT"})

    expected_columns = {
        "runup_24h",
        "volume_ratio",
        "failed_breakout_high",
        "support_low",
        "atr_pct",
        "market_circuit_breaker",
    }
    assert expected_columns.issubset(dataframe.columns)
    assert dataframe["atr_pct"].fillna(0).ge(0).all()


def test_entry_signal_requires_failed_breakout_and_support_loss() -> None:
    strategy = TopReversalShortStrategy(config={})
    dataframe = _top_frame()
    dataframe["runup_24h"] = [0.2] * 259 + [2.6]
    dataframe["volume_ratio"] = [1.0] * 259 + [2.4]
    dataframe["atr_pct"] = [0.012] * 260
    dataframe["rsi"] = [60.0] * 259 + [79.0]
    dataframe["failed_breakout_high"] = [12.0] * 260
    dataframe["support_low"] = [11.0] * 260
    dataframe["breakdown_level"] = [11.0] * 260
    dataframe["close"] = [11.8] * 259 + [10.7]
    dataframe["open"] = [11.7] * 259 + [11.4]
    dataframe["high"] = [11.9] * 259 + [12.4]
    dataframe["low"] = [11.5] * 259 + [10.6]
    dataframe["market_circuit_breaker"] = [0] * 260

    result = strategy.populate_entry_trend(dataframe, {"pair": "DN/USDT"})

    assert result.loc[259, "enter_short"] == 1
    assert result.loc[259, "enter_tag"] == "failed_breakout_support_loss"


def test_circuit_breaker_blocks_short_entry() -> None:
    strategy = TopReversalShortStrategy(config={})
    dataframe = _top_frame()
    dataframe["runup_24h"] = [2.6] * 260
    dataframe["volume_ratio"] = [2.4] * 260
    dataframe["atr_pct"] = [0.012] * 260
    dataframe["rsi"] = [79.0] * 260
    dataframe["failed_breakout_high"] = [12.0] * 260
    dataframe["support_low"] = [11.0] * 260
    dataframe["breakdown_level"] = [11.0] * 260
    dataframe["close"] = [10.7] * 260
    dataframe["open"] = [11.4] * 260
    dataframe["market_circuit_breaker"] = [0] * 259 + [1]

    result = strategy.populate_entry_trend(dataframe, {"pair": "DN/USDT"})

    assert result.loc[259, "enter_short"] == 0


def test_exit_signal_covers_profit_snapback_and_squeeze_risk() -> None:
    strategy = TopReversalShortStrategy(config={})
    dataframe = _top_frame()
    dataframe["ema_fast"] = [11.0] * 258 + [10.0, 11.3]
    dataframe["rsi"] = [55.0] * 258 + [31.0, 70.0]
    dataframe["volume_ratio"] = [1.0] * 258 + [1.2, 2.3]
    dataframe["atr_pct"] = [0.01] * 260
    dataframe["momentum_3"] = [0.0] * 258 + [-0.03, 0.025]
    dataframe["close"] = [10.0] * 258 + [9.4, 11.6]
    dataframe["open"] = [10.0] * 258 + [9.8, 10.9]

    result = strategy.populate_exit_trend(dataframe, {"pair": "DN/USDT"})

    assert result.loc[258, "exit_short"] == 1
    assert result.loc[258, "exit_tag"] == "profit_snapback"
    assert result.loc[259, "exit_short"] == 1
    assert result.loc[259, "exit_tag"] == "squeeze_risk_exit"


def test_leverage_is_capped_for_short_squeeze_risk() -> None:
    strategy = TopReversalShortStrategy(config={})

    leverage = strategy.leverage(
        pair="DN/USDT",
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=10.0,
        proposed_leverage=5.0,
        max_leverage=10.0,
        entry_tag="failed_breakout_support_loss",
        side="short",
    )

    assert leverage == 1.5
