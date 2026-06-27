# -*- coding: utf-8 -*-
"""
缠论中枢突破策略

基于缠论（K 线包含合并→分型→笔→中枢识别），在收盘价突破上一中枢上沿时做多，跌破中枢下沿时止损离场。
"""

from pandas import DataFrame

from freqtrade.strategy import IStrategy

try:
    from user_data.strategies.缠论_核心模块 import add_chanlun_signals
except ImportError:
    from 缠论_核心模块 import add_chanlun_signals


class ChanlunCenterBreakoutStrategy(IStrategy):
    """
    缠论中枢突破策略（首版逻辑，已修正）。

    规则：
    - 经过 K 线包含处理后，由三根合并K线确认分型。
    - 交替分型形成笔。
    - 最近三笔的重叠价格区间形成中枢。
    - 收盘突破上一中枢上沿做多，跌破中枢下沿离场。
    """

    INTERFACE_VERSION = 3

    can_short = False
    # 改动：从 1m 提升到 15m，降低噪音与假突破
    timeframe = "15m"
    # 改动：包含处理 + 中枢需要足够历史，提高 startup
    startup_candle_count = 200
    process_only_new_candles = True

    # 改动：原来 {"0": 0.0} 会在利润刚到 0 就强制平仓，
    # 造成“盈利砍在0、亏损放到-5%”的负期望，这里改成递减止盈表。
    minimal_roi = {"0": 0.04, "120": 0.02, "360": 0.01, "720": 0.0}
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
    # 改动：笔的最小长度从 3 提升到 5（更接近缠论标准）
    min_stroke_gap = 5

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
