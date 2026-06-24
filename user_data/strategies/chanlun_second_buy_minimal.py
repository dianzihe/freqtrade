#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二买策略最小化测试版本 —— 不做任何过滤，直接把 chan_second_buy 映射为 enter_long。
用于验证信号能否正确触发交易。
"""

from __future__ import annotations

import logging

import pandas as pd
from freqtrade.strategy import IStrategy
from pandas import DataFrame

try:
    from user_data.strategies.chanlun_core import add_chanlun_signals
except ImportError:
    from chanlun_core import add_chanlun_signals

logger = logging.getLogger(__name__)


class ChanlunSecondBuyMinimal(IStrategy):
    timeframe = "15m"
    startup_candle_count = 300
    stoploss = -0.05  # 收窄止损到 5%，原策略默认值
    minimal_roi = {"0": 0.04, "30": 0.02, "60": 0.01}
    # 只用 ROI 止盈 + trailing stop，去掉固定止损和策略退出
    use_exit_signal = False
    trailing_stop = True
    trailing_stop_positive = 0.01   # 盈利超 1% 后启动移动止损
    trailing_stop_positive_offset = 0.02  # 移动止损线距离当前价 2%
    trailing_only_offset_is_reached = True

    stoploss = -0.10  # 宽松固定止损兜底（很少触发）

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return add_chanlun_signals(dataframe.copy(), min_stroke_gap=5)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 最简：二买信号直接入场，不做任何过滤
        dataframe.loc[dataframe["chan_second_buy"] == 1, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 不填 exit_long，完全依赖 ROI + stoploss 出场
        return dataframe
