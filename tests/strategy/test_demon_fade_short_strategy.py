import pandas as pd

from user_data.strategies.demon_fade_short_strategy import DemonFadeShortStrategy


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


def test_demon_fade_short_enters_failed_breakout_reclaim() -> None:
    strategy = DemonFadeShortStrategy(config={})
    df = strategy.populate_indicators(_base_frame(), {"pair": "AIN/USDT"})
    last = df.index[-1]

    df.loc[last, "fade_score"] = 80.0
    df.loc[last, "quote_volume_24h"] = 100_000.0
    df.loc[last, "prior_breakout_extension"] = 0.08
    df.loc[last, "break_24h_high"] = -0.015
    df.loc[last, "upper_wick_ratio"] = 0.55
    df.loc[last, "close_position"] = 0.25
    df.loc[last, "ret_15m"] = -0.025
    df.loc[last, "volume_ratio_1h_24h"] = 4.0

    result = strategy.populate_entry_trend(df.copy(), {"pair": "AIN/USDT"})

    assert result.loc[last, "enter_short"] == 1
    assert result.loc[last, "enter_tag"] == "demon_fade"


def test_demon_fade_short_blocks_when_breakout_still_holds() -> None:
    strategy = DemonFadeShortStrategy(config={})
    df = strategy.populate_indicators(_base_frame(), {"pair": "AIN/USDT"})
    last = df.index[-1]

    df.loc[last, "fade_score"] = 80.0
    df.loc[last, "quote_volume_24h"] = 100_000.0
    df.loc[last, "prior_breakout_extension"] = 0.08
    df.loc[last, "break_24h_high"] = 0.02
    df.loc[last, "upper_wick_ratio"] = 0.55
    df.loc[last, "close_position"] = 0.25
    df.loc[last, "ret_15m"] = -0.025
    df.loc[last, "volume_ratio_1h_24h"] = 4.0

    result = strategy.populate_entry_trend(df.copy(), {"pair": "AIN/USDT"})

    assert result.loc[last, "enter_short"] == 0
