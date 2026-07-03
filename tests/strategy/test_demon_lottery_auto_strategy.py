import pandas as pd

from user_data.strategies.demon_lottery_auto_strategy import DemonLotteryAutoStrategy


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


def test_demon_lottery_indicators_include_score_components() -> None:
    strategy = DemonLotteryAutoStrategy(config={})

    result = strategy.populate_indicators(_base_frame(), {"pair": "VELVET/USDT"})

    for column in [
        "demon_score",
        "quote_volume_24h",
        "bb_width_ratio_48h",
        "volume_ratio_1h_24h",
        "break_24h_high",
    ]:
        assert column in result.columns
    assert result["demon_score"].iloc[-1] >= 0


def test_demon_lottery_entry_requires_breakout_and_volume_confirmation() -> None:
    strategy = DemonLotteryAutoStrategy(config={})
    df = strategy.populate_indicators(_base_frame(), {"pair": "VELVET/USDT"})
    last = df.index[-1]

    df.loc[last, "demon_score"] = 82.0
    df.loc[last, "quote_volume_24h"] = 100_000.0
    df.loc[last, "break_24h_high"] = 0.01
    df.loc[last, "ret_15m"] = 0.025
    df.loc[last, "ret_1h"] = 0.055
    df.loc[last, "volume_ratio_1h_24h"] = 4.0

    result = strategy.populate_entry_trend(df.copy(), {"pair": "VELVET/USDT"})

    weak_volume = df.copy()
    weak_volume.loc[last, "volume_ratio_1h_24h"] = 1.0
    blocked = strategy.populate_entry_trend(weak_volume, {"pair": "VELVET/USDT"})

    assert result.loc[last, "enter_long"] == 1
    assert result.loc[last, "enter_tag"] == "demon_breakout"
    assert blocked.loc[last, "enter_long"] == 0


def test_demon_lottery_uses_tight_lottery_hard_stop() -> None:
    strategy = DemonLotteryAutoStrategy(config={})

    assert strategy.stoploss == -0.08


def test_demon_lottery_debounces_persistent_breakout_signals() -> None:
    strategy = DemonLotteryAutoStrategy(config={})
    df = strategy.populate_indicators(_base_frame(), {"pair": "VELVET/USDT"})
    trigger_rows = [df.index[-2], df.index[-1]]

    df.loc[trigger_rows, "demon_score"] = 82.0
    df.loc[trigger_rows, "quote_volume_24h"] = 100_000.0
    df.loc[trigger_rows, "break_24h_high"] = 0.01
    df.loc[trigger_rows, "ret_15m"] = 0.025
    df.loc[trigger_rows, "ret_1h"] = 0.055
    df.loc[trigger_rows, "volume_ratio_1h_24h"] = 4.0
    df.loc[trigger_rows, "close"] = df.loc[trigger_rows, "ema20"] * 1.03
    df.loc[trigger_rows, "rsi"] = 65.0

    result = strategy.populate_entry_trend(df.copy(), {"pair": "VELVET/USDT"})

    assert result.loc[trigger_rows[0], "enter_long"] == 1
    assert result.loc[trigger_rows[1], "enter_long"] == 0
