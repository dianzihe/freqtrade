from pandas import DataFrame

from freqtrade.strategy import IStrategy


class ManualOnlyStrategy(IStrategy):
    """
    Strategy for controlled live smoke tests.

    It never creates automatic entries. Use REST/UI/CLI force-entry for the
    first real order, then let Freqtrade manage ROI, stoploss, and force-exit.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    startup_candle_count = 1
    process_only_new_candles = True

    minimal_roi = {"60": 0.01, "30": 0.02, "0": 0.04}
    stoploss = -0.03
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
