from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[2] / "user_data" / "strategies"))

from chanlun_core import add_chanlun_signals  # noqa: E402


def _frame(highs, lows, closes=None):
    if closes is None:
        closes = [(high + low) / 2 for high, low in zip(highs, lows, strict=True)]

    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * len(highs),
        }
    )


def test_marks_three_candle_fractals_without_lookahead_on_edges():
    dataframe = _frame(
        highs=[10, 13, 11, 12, 9],
        lows=[8, 9, 7, 8, 6],
    )

    result = add_chanlun_signals(dataframe, min_stroke_gap=1)

    assert result["chan_top"].tolist() == [False, True, False, True, False]
    assert result["chan_bottom"].tolist() == [False, False, True, False, False]


def test_builds_center_from_three_overlapping_strokes():
    dataframe = _frame(
        highs=[10, 12, 9, 11, 8, 10, 9, 9, 12],
        lows=[8, 9, 6, 8, 5, 7, 6, 6, 9],
        closes=[9, 11, 7, 10, 6, 9, 7, 8, 11],
    )

    result = add_chanlun_signals(dataframe, min_stroke_gap=1)

    assert result.loc[5, "chan_center_valid"]
    assert result.loc[5, "chan_center_low"] == 6
    assert result.loc[5, "chan_center_high"] == 10


def test_entry_breaks_above_prior_center_and_exit_breaks_below_prior_center():
    dataframe = _frame(
        highs=[10, 12, 9, 11, 8, 10, 9, 9, 13, 11, 10, 9, 8],
        lows=[8, 9, 6, 8, 5, 7, 6, 6, 9, 8, 7, 5, 4],
        closes=[9, 11, 7, 10, 6, 9, 7, 8, 12, 9, 8, 5.5, 5],
    )

    result = add_chanlun_signals(dataframe, min_stroke_gap=1)

    assert result.loc[8, "chan_enter_long"] == 1
    assert result.loc[11, "chan_exit_long"] == 1
