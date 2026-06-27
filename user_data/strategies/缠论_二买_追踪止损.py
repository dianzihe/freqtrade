#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论二买 + 追踪止损策略

在缠论第二类买点（中枢突破后回踩中枢上沿反弹）入场，ROI 表止盈，盈利超 1% 后启动移动止损保护利润。
"""
from __future__ import annotations

import logging

import pandas as pd
from freqtrade.strategy import IStrategy
from pandas import DataFrame

try:
    from user_data.strategies.缠论_核心模块 import add_chanlun_signals
except ImportError:
    from 缠论_核心模块 import add_chanlun_signals

logger = logging.getLogger(__name__)


class ChanlunSecondBuyTrailing(IStrategy):
    """
    二买回踩 + Trailing Stop：
    - 入场：chan_second_buy 信号
    - 止盈：ROI 表
    - 止损：Trailing Stop（盈利超 1% 后启动，止损线距当前价 2%）
    - 无策略退出信号
    """
    timeframe = "15m"
    startup_candle_count = 300

    # 固定止损（兜底）
    stoploss = -0.03   # 收紧到 3%，限制单笔最大亏损

    # ROI 止盈表
    minimal_roi = {"0": 0.04, "30": 0.02, "60": 0.01}

    # Trailing Stop 配置（保护盈利仓位）
    trailing_stop = True
    trailing_stop_positive = 0.01        # 盈利 >= 1% 后启动 trailing
    trailing_stop_positive_offset = 0.05   # trailing 止损线距当前价 5%（给更多空间）
    trailing_only_offset_is_reached = False

    process_only_new_candles = True
    use_exit_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return add_chanlun_signals(dataframe.copy(), min_stroke_gap=5)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["chan_second_buy"] == 1, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 不填 exit_long，完全依赖 ROI + Trailing Stop
        return dataframe
