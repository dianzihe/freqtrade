import pandas as pd

from user_data.strategies.ChanlunFilteredStrategies import (
    ChanlunMaRsiSecondBuyStrategy,
    ChanlunMacdDivergenceStrategy,
    ChanlunRsiTimingStrategy,
    ChanlunVolumeConfirmationStrategy,
)


def _base_frame(rows: int = 8) -> pd.DataFrame:
    close = pd.Series([10.0 + i for i in range(rows)])
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [100.0] * rows,
            "chan_enter_long": [0] * (rows - 1) + [1],
            "chan_exit_long": [0] * (rows - 1) + [1],
        }
    )


def test_macd_strategy_requires_bullish_divergence_for_entry_and_bearish_for_exit() -> None:
    dataframe = _base_frame()
    dataframe["low"] = [11, 10, 9, 8, 7, 8, 6, 7]
    dataframe["high"] = [12, 13, 14, 15, 16, 15, 17, 16]
    dataframe["macdhist"] = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.5, -0.3, 0.1]
    strategy = ChanlunMacdDivergenceStrategy(config={})

    result = strategy.populate_entry_trend(dataframe.copy(), {"pair": "BTC/USDT"})
    assert result.loc[7, "enter_long"] == 1
    assert result.loc[7, "enter_tag"] == "chanlun_macd_bullish_divergence"

    exit_frame = _base_frame()
    exit_frame["high"] = [12, 13, 14, 15, 16, 15, 17, 16]
    exit_frame["macdhist"] = [0.2, 0.4, 0.6, 0.8, 1.0, 0.7, 0.6, 0.4]
    exit_result = strategy.populate_exit_trend(exit_frame, {"pair": "BTC/USDT"})
    assert exit_result.loc[7, "exit_long"] == 1
    assert exit_result.loc[7, "exit_tag"] == "chanlun_macd_bearish_divergence"


def test_rsi_strategy_requires_oversold_entry_and_overbought_exit() -> None:
    dataframe = _base_frame()
    dataframe["rsi"] = [50, 48, 44, 35, 28, 31, 33, 34]
    strategy = ChanlunRsiTimingStrategy(config={})

    result = strategy.populate_entry_trend(dataframe.copy(), {"pair": "BTC/USDT"})
    assert result.loc[7, "enter_long"] == 1
    assert result.loc[7, "enter_tag"] == "chanlun_rsi_oversold_rebound"

    dataframe["rsi"] = [50, 52, 60, 68, 72, 69, 67, 66]
    exit_result = strategy.populate_exit_trend(dataframe.copy(), {"pair": "BTC/USDT"})
    assert exit_result.loc[7, "exit_long"] == 1
    assert exit_result.loc[7, "exit_tag"] == "chanlun_rsi_overbought_rollover"


def test_volume_strategy_requires_dry_volume_entry_and_heavy_stall_exit() -> None:
    dataframe = _base_frame()
    dataframe["volume"] = [100, 105, 98, 102, 96, 20, 22, 24]
    dataframe["volume_mean"] = [100] * len(dataframe)
    dataframe["price_progress"] = [0.01] * len(dataframe)
    strategy = ChanlunVolumeConfirmationStrategy(config={})

    result = strategy.populate_entry_trend(dataframe.copy(), {"pair": "BTC/USDT"})
    assert result.loc[7, "enter_long"] == 1
    assert result.loc[7, "enter_tag"] == "chanlun_volume_dry_entry"

    dataframe["volume"] = [100, 105, 98, 102, 96, 210, 220, 230]
    dataframe["price_progress"] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    exit_result = strategy.populate_exit_trend(dataframe.copy(), {"pair": "BTC/USDT"})
    assert exit_result.loc[7, "exit_long"] == 1
    assert exit_result.loc[7, "exit_tag"] == "chanlun_volume_stall_exit"


def test_ma_rsi_strategy_requires_second_buy_with_ma_cross_and_rsi_rebound() -> None:
    dataframe = _base_frame()
    dataframe["chan_second_buy"] = [False] * 7 + [True]
    dataframe["ema_fast"] = [10, 10, 10, 10, 10, 9.5, 10.0, 10.7]
    dataframe["ema_slow"] = [10, 10, 10, 10, 10, 10.0, 10.2, 10.4]
    dataframe["rsi"] = [45, 42, 39, 34, 29, 27, 31, 36]
    strategy = ChanlunMaRsiSecondBuyStrategy(config={})

    result = strategy.populate_entry_trend(dataframe.copy(), {"pair": "BTC/USDT"})
    assert result.loc[7, "enter_long"] == 1
    assert result.loc[7, "enter_tag"] == "chanlun_second_buy_ma_rsi"
