import pandas as pd

from user_data.strategies.demon_breakout_short_strategy import DemonBreakoutShortStrategy


def _base_frame(rows: int = 620) -> pd.DataFrame:
    close = pd.Series([1.0 + idx * 0.0002 for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="5min", tz="UTC"),
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": [1000.0] * rows,
        }
    )


def test_demon_breakout_short_reuses_breakout_signal_as_short_entry() -> None:
    strategy = DemonBreakoutShortStrategy(config={})
    df = strategy.populate_indicators(_base_frame(), {"pair": "AIN/USDT"})
    last = df.index[-1]

    df.loc[last, "demon_score"] = 82.0
    df.loc[last, "quote_volume_24h"] = 100_000.0
    df.loc[last, "break_24h_high"] = 0.01
    df.loc[last, "ret_15m"] = 0.025
    df.loc[last, "ret_1h"] = 0.055
    df.loc[last, "volume_ratio_1h_24h"] = 4.0
    df.loc[last, "close"] = df.loc[last, "ema20"] * 1.03
    df.loc[last, "rsi"] = 65.0

    result = strategy.populate_entry_trend(df.copy(), {"pair": "AIN/USDT"})

    assert result.loc[last, "enter_short"] == 1
    assert result.loc[last, "enter_long"] == 0
    assert result.loc[last, "enter_tag"] == "inverse_breakout_short"


def test_demon_breakout_short_uses_four_percent_hard_stop() -> None:
    strategy = DemonBreakoutShortStrategy(config={})

    assert strategy.stoploss == -0.04
