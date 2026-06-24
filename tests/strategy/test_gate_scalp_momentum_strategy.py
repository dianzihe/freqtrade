import pandas as pd

from user_data.strategies.gate_scalp_momentum_strategy import GateScalpMomentumStrategy


def _ohlcv_frame(rows: int = 80) -> pd.DataFrame:
    close = pd.Series([100.0 + (idx * 0.02) for idx in range(rows)])
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": [100.0] * rows,
        }
    )


def test_populate_indicators_adds_scalp_columns() -> None:
    strategy = GateScalpMomentumStrategy(config={})

    dataframe = strategy.populate_indicators(_ohlcv_frame(), {"pair": "BTC/USDT"})

    expected_columns = {
        "ema_fast",
        "ema_slow",
        "rsi",
        "atr_pct",
        "volume_mean",
        "pullback_pct",
        "momentum_3m",
    }
    assert expected_columns.issubset(dataframe.columns)
    assert dataframe["rsi"].between(0, 100).all()


def test_entry_signal_requires_pullback_rebound_with_volume() -> None:
    strategy = GateScalpMomentumStrategy(config={})
    dataframe = _ohlcv_frame()
    dataframe["ema_fast"] = [100.0] * 78 + [99.8, 100.4]
    dataframe["ema_slow"] = [99.0] * 80
    dataframe["ema_trend"] = [98.0] * 80
    dataframe["rsi"] = [52.0] * 76 + [36.0, 35.0, 39.0, 45.0]
    dataframe["volume"] = [100.0] * 79 + [150.0]
    dataframe["volume_mean"] = [100.0] * 80
    dataframe["atr_pct"] = [0.004] * 80
    dataframe["pullback_pct"] = [0.001] * 79 + [0.004]
    dataframe["momentum_3m"] = [0.0] * 79 + [0.003]

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[79, "enter_long"] == 1
    assert result.loc[79, "enter_tag"] == "scalp_pullback_rebound"


def test_exit_signal_marks_weak_trend() -> None:
    strategy = GateScalpMomentumStrategy(config={})
    dataframe = _ohlcv_frame()
    dataframe["ema_fast"] = [101.0] * 79 + [99.5]
    dataframe["ema_slow"] = [100.0] * 80
    dataframe["rsi"] = [55.0] * 79 + [44.0]
    dataframe["momentum_3m"] = [0.003] * 79 + [-0.002]
    dataframe["volume_mean"] = [100.0] * 80

    result = strategy.populate_exit_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[79, "exit_long"] == 1
    assert result.loc[79, "exit_tag"] == "scalp_weak_trend"
