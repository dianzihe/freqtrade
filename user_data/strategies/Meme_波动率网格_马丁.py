# -*- coding: utf-8 -*-
"""
Meme 波动率网格马丁策略

基于布林带下轨 + ATR 宽度判断超卖区间，交易均值回归而非假设每次暴跌都能恢复。
"""

from datetime import datetime

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from Meme_马丁_基类 import MemeMartingaleBaseStrategy


class MemeVolatilityGridMartingaleStrategy(MemeMartingaleBaseStrategy):
    """
    Volatility grid martingale: trades range reversion instead of assuming every dip recovers.
    """

    minimal_roi = {"180": 0.0, "45": 0.018, "0": 0.04}
    stoploss = -0.20
    dca_thresholds = [-0.05, -0.10, -0.18, -0.30]
    dca_multipliers = [1.0, 1.2, 1.4, 1.6]
    dca_tag_prefix = "vol_grid_dca"

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        
        wide_enough = dataframe["atr_pct"] > 0.006
        lower_band_touch = dataframe["close"] < dataframe["bb_lower"]
        range_not_dead = dataframe["range_position"].between(0.12, 0.72)
        
        dataframe.loc[
            wide_enough & lower_band_touch & range_not_dead,
            ["enter_long", "enter_tag"]
        ] = (1, "vol_grid_lower_band")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_mid"]) | (dataframe["rsi"] > 58),
            ["exit_long", "exit_tag"],
        ] = (1, "vol_grid_mean_exit")
        
        return dataframe
