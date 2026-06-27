# -*- coding: utf-8 -*-
"""
Meme 反马丁趋势策略

只在盈利时加仓，绝不摊平亏损单。追踪止损让利润奔跑，严格限制亏损端风险敞口。
"""

from datetime import datetime

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from Meme_马丁_基类 import MemeMartingaleBaseStrategy


class MemeAntiMartingaleTrendStrategy(MemeMartingaleBaseStrategy):
    """
    Anti-martingale trend strategy: add only to winners, never average down losers.
    """

    minimal_roi = {"240": 0.03, "80": 0.06, "0": 0.12}
    stoploss = -0.16
    max_entry_position_adjustment = 2
    position_adjustment_enable = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        
        breakout = dataframe["close"] > dataframe["rolling_high_240"].shift(1) * 1.01
        volume_confirmed = dataframe["volume_ratio"] > 1.4
        trend_confirmed = dataframe["trend_up"] & (dataframe["rsi"] > 56)
        
        dataframe.loc[
            breakout & volume_confirmed & trend_confirmed,
            ["enter_long", "enter_tag"]
        ] = (1, "anti_martingale_breakout")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        
        dataframe.loc[
            (dataframe["close"] < dataframe["ema_fast"] * 0.965) | (dataframe["rsi"] < 45),
            ["exit_long", "exit_tag"],
        ] = (1, "anti_martingale_trend_fail")
        
        return dataframe

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        entry_count = trade.nr_of_successful_entries
        profit_steps = [0.08, 0.16]
        
        if entry_count <= 0 or entry_count > len(profit_steps):
            return None
            
        if current_profit < profit_steps[entry_count - 1]:
            return None
            
        stake = min(trade.stake_amount * 0.5, max_stake)
        
        if min_stake and stake < min_stake:
            return None
            
        return stake, f"anti_martingale_profit_add_{entry_count}"
