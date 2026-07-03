from __future__ import annotations

from pandas import DataFrame

from user_data.strategies.demon_lottery_auto_strategy import DemonLotteryAutoStrategy


class DemonBreakoutShortStrategy(DemonLotteryAutoStrategy):
    """
    Direct inverse of the demon breakout long scanner.

    This intentionally shorts the same right-confirmed breakout signal that the
    long strategy buys. It exists to test the hypothesis that most local demon
    breakout signals are false breakouts.
    """

    can_short = True
    trading_mode = "futures"
    stoploss = -0.04

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_entry_trend(dataframe, metadata)
        long_signal = df["enter_long"] == 1

        df["enter_short"] = 0
        df.loc[long_signal, "enter_short"] = 1
        df.loc[long_signal, "enter_tag"] = "inverse_breakout_short"
        df.loc[long_signal, "enter_long"] = 0
        return df
