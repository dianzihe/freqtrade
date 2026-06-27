# -*- coding: utf-8 -*-
"""
Meme 对冲代理马丁策略

更小仓位、更早出场的现货策略，模拟对冲马丁的运行方式，不做空。
"""

from datetime import datetime

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from Meme_马丁_基类 import MemeMartingaleBaseStrategy


class MemeHedgeProxyMartingaleStrategy(MemeMartingaleBaseStrategy):
    """
    Spot-data proxy for a hedged martingale: smaller inventory, earlier exits, no shorting.
    """

    minimal_roi = {"120": 0.0, "35": 0.015, "0": 0.035}
    stoploss = -0.14
    max_entry_position_adjustment = 3
    dca_thresholds = [-0.06, -0.12, -0.20]
    dca_multipliers = [0.8, 1.0, 1.2]
    dca_tag_prefix = "hedge_proxy_dca"

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = proposed_stake * 0.6
        return max(min(stake, max_stake), min_stake or 0)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        
        panic_but_liquid = (
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["volume_ratio"] > 1.8)
            & (dataframe["rsi"] < 34)
            & (dataframe["range_position"] > 0.10)
        )
        
        dataframe.loc[panic_but_liquid, ["enter_long", "enter_tag"]] = (
            1,
            "hedge_proxy_panic_entry",
        )
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_mid"]) | (dataframe["range_position"] < 0.06),
            ["exit_long", "exit_tag"],
        ] = (1, "hedge_proxy_reduce_risk")
        
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if current_profit < -0.11 and trade.nr_of_successful_entries >= 2:
            return "hedge_proxy_protective_exit"
        return None
