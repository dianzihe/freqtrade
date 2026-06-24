import pandas as pd

from user_data.strategies.chanlun_center_breakout_strategy import ChanlunCenterBreakoutStrategy


def test_chanlun_strategy_maps_core_signals_to_freqtrade_columns() -> None:
    close = [9, 11, 7, 10, 6, 9, 7, 8, 12, 9, 8, 5.5, 5]
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=len(close), freq="1min", tz="UTC"),
            "open": close,
            "high": [10, 12, 9, 11, 8, 10, 9, 9, 13, 11, 10, 9, 8],
            "low": [8, 9, 6, 8, 5, 7, 6, 6, 9, 8, 7, 5, 4],
            "close": close,
            "volume": [100] * len(close),
        }
    )
    strategy = ChanlunCenterBreakoutStrategy(config={})
    strategy.min_stroke_gap = 1

    dataframe = strategy.populate_indicators(dataframe, {"pair": "BTC/USDT"})
    dataframe = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})
    dataframe = strategy.populate_exit_trend(dataframe, {"pair": "BTC/USDT"})

    assert dataframe.loc[8, "enter_long"] == 1
    assert dataframe.loc[8, "enter_tag"] == "chanlun_center_breakout"
    assert dataframe.loc[11, "exit_long"] == 1
    assert dataframe.loc[11, "exit_tag"] == "chanlun_center_breakdown"
