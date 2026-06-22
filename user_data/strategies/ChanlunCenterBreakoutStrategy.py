from pandas import DataFrame

from freqtrade.strategy import IStrategy

from user_data.strategies.chanlun_core import add_chanlun_signals


class ChanlunCenterBreakoutStrategy(IStrategy):
    """
    First-pass Chanlun center breakout strategy.

    Rule definition:
    - A fractal is confirmed by a three-candle local high or low.
    - Alternating fractals form strokes.
    - The overlapping price range of the latest three strokes forms a center.
    - A close above the previous center high enters long.
    - A close below the previous center low exits long.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "1m"
    startup_candle_count = 120
    process_only_new_candles = True

    minimal_roi = {"0": 0.0}
    stoploss = -0.05
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}
    min_stroke_gap = 3

    plot_config = {
        "main_plot": {
            "chan_center_high": {"color": "green"},
            "chan_center_low": {"color": "red"},
        },
        "subplots": {
            "Chanlun": {
                "chan_stroke_dir": {"color": "blue"},
            },
        },
    }

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return add_chanlun_signals(dataframe, min_stroke_gap=self.min_stroke_gap)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        dataframe.loc[
            dataframe["chan_enter_long"] == 1,
            ["enter_long", "enter_tag"],
        ] = (1, "chanlun_center_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        dataframe.loc[
            dataframe["chan_exit_long"] == 1,
            ["exit_long", "exit_tag"],
        ] = (1, "chanlun_center_breakdown")
        return dataframe
