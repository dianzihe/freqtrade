# -*- coding: utf-8 -*-
"""
Meme 有限 DCA 马丁策略

低倍数、硬失效条件的暴跌抄底策略。少量分散加仓层数，严格触发失效条件即终止加仓。
"""

from datetime import datetime

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from Meme_马丁_基类 import MemeMartingaleBaseStrategy


class MemeLimitedDcaMartingaleStrategy(MemeMartingaleBaseStrategy):
    """
    Limited crash-catching martingale: few layers, low multipliers, hard invalidation.
    """

    minimal_roi = {"90": 0.0, "25": 0.025, "0": 0.06}
    stoploss = -0.24
    dca_tag_prefix = "limited_dca"

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        
        crash_pullback = dataframe["drawdown_60"] < -0.09
        liquid_rebound = (dataframe["volume_ratio"] > 1.55) & (dataframe["rsi"] < 32)
        not_broken = dataframe["range_position"].between(0.12, 0.78)
        
        dataframe.loc[
            crash_pullback & liquid_rebound & not_broken,
            ["enter_long", "enter_tag"]
        ] = (1, "limited_crash_rebound")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        
        dataframe.loc[
            (dataframe["rsi"] > 63) | (dataframe["close"] > dataframe["ema_fast"] * 1.035),
            ["exit_long", "exit_tag"],
        ] = (1, "limited_rebound_exit")
        
        return dataframe
