from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.gate_xinghe_futures_grid_strategy import (
    GateXingheFuturesGridStrategy,
)


def _strategy() -> GateXingheFuturesGridStrategy:
    return GateXingheFuturesGridStrategy(config={"candle_type_def": "futures"})


def _ohlcv_frame(rows: int = 260) -> pd.DataFrame:
    close = pd.Series([100.0 + idx * 0.04 for idx in range(rows)], dtype="float64")
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


def test_indicators_add_xinghe_signal_columns() -> None:
    strategy = _strategy()

    dataframe = strategy.populate_indicators(_ohlcv_frame(), {"pair": "BTC/USDT:USDT"})

    expected = {
        "ema_fast",
        "ema_slow",
        "atr_pct",
        "volatility_pct",
        "grid_step_pct",
        "trend_signal",
        "risk_fuse",
    }
    assert expected.issubset(dataframe.columns)
    assert dataframe["trend_signal"].dropna().isin([-1, 0, 1]).all()
    assert dataframe["grid_step_pct"].fillna(0).ge(0).all()


def test_long_entry_requires_bullish_mixed_signal() -> None:
    strategy = _strategy()
    dataframe = _ohlcv_frame()
    dataframe["trend_signal"] = 0
    dataframe["risk_fuse"] = 0
    dataframe["atr_pct"] = 0.006
    dataframe["volume"] = 1000.0
    dataframe.loc[dataframe.index[-1], "trend_signal"] = 1

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT:USDT"})

    assert result.loc[dataframe.index[-1], "enter_long"] == 1
    assert result.loc[dataframe.index[-1], "enter_tag"] == "xinghe_mixed_long"


def test_short_entry_requires_bearish_mixed_signal() -> None:
    strategy = _strategy()
    dataframe = _ohlcv_frame()
    dataframe["trend_signal"] = 0
    dataframe["risk_fuse"] = 0
    dataframe["atr_pct"] = 0.006
    dataframe["volume"] = 1000.0
    dataframe.loc[dataframe.index[-1], "trend_signal"] = -1

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT:USDT"})

    assert result.loc[dataframe.index[-1], "enter_short"] == 1
    assert result.loc[dataframe.index[-1], "enter_tag"] == "xinghe_mixed_short"


def test_atr_grid_threshold_is_bounded() -> None:
    strategy = _strategy()

    low = strategy._grid_threshold(0.0001, 0)
    middle = strategy._grid_threshold(0.006, 1)
    high = strategy._grid_threshold(0.20, 3)

    assert low == strategy.min_grid_step_pct
    assert middle > low
    assert high == strategy.max_grid_step_pct


def test_dca_uses_adverse_distance_and_returns_stake_ladder() -> None:
    strategy = _strategy()
    trade = SimpleNamespace(
        nr_of_successful_entries=1,
        stake_amount=100.0,
        open_rate=100.0,
        is_short=False,
        select_filled_orders=lambda side: [SimpleNamespace(safe_cost=100.0)],
    )

    result = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=98.0,
        current_profit=-0.02,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=100.0,
        current_exit_rate=98.0,
        current_entry_profit=-0.02,
        current_exit_profit=-0.02,
    )

    assert result == (130.0, "xinghe_dca_1")


def test_dca_rejects_deep_loss_risk_fuse() -> None:
    strategy = _strategy()
    trade = SimpleNamespace(
        nr_of_successful_entries=2,
        stake_amount=100.0,
        open_rate=100.0,
        is_short=True,
        select_filled_orders=lambda side: [SimpleNamespace(safe_cost=100.0)],
    )

    result = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=125.0,
        current_profit=-0.25,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=100.0,
        current_exit_rate=125.0,
        current_entry_profit=-0.25,
        current_exit_profit=-0.25,
    )

    assert result is None
