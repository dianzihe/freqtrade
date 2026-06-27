# -*- coding: utf-8 -*-
"""
Meme 马丁策略通用基类

为所有 Meme 马丁系列策略提供公共指标计算、DCA 加仓逻辑和仓位管理，不直接用于交易。
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class MemeMartingaleBaseStrategy(IStrategy):
    """
    Base class for all Meme Martingale strategies.
    Provides common indicators, DCA logic, and stake management.
    """
    
    INTERFACE_VERSION = 3

    timeframe = "1m"
    can_short = False
    startup_candle_count = 240
    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 4

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"120": 0.0, "30": 0.02, "0": 0.05}
    stoploss = -0.22

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    dca_thresholds = [-0.08, -0.13, -0.21, -0.34]
    dca_multipliers = [1.0, 1.3, 1.6, 2.0]
    dca_tag_prefix = "dca"

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _atr_pct(dataframe: DataFrame, period: int = 14) -> pd.Series:
        prev_close = dataframe["close"].shift(1)
        true_range = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - prev_close).abs(),
                (dataframe["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(period).mean() / dataframe["close"]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate common indicators used by all meme martingale strategies."""
        dataframe["ema_fast"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=120, adjust=False).mean()
        dataframe["rolling_high_60"] = dataframe["close"].rolling(60).max()
        dataframe["rolling_high_240"] = dataframe["close"].rolling(240).max()
        dataframe["rolling_low_240"] = dataframe["close"].rolling(240).min()
        dataframe["volume_mean_60"] = dataframe["volume"].rolling(60).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_mean_60"]
        dataframe["volume_ratio"] = (
            dataframe["volume_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
        )
        dataframe["atr_pct"] = self._atr_pct(dataframe).fillna(0)
        dataframe["rsi"] = self._rsi(dataframe["close"])

        bb_mid = dataframe["close"].rolling(80).mean()
        bb_std = dataframe["close"].rolling(80).std()
        dataframe["bb_mid"] = bb_mid
        dataframe["bb_lower"] = bb_mid - 2.2 * bb_std
        dataframe["bb_upper"] = bb_mid + 2.2 * bb_std

        dataframe["range_position"] = (
            (dataframe["close"] - dataframe["rolling_low_240"])
            / (dataframe["rolling_high_240"] - dataframe["rolling_low_240"])
        ).clip(0, 1)

        dataframe["drawdown_60"] = dataframe["close"] / dataframe["rolling_high_60"] - 1
        dataframe["trend_up"] = dataframe["ema_fast"] > dataframe["ema_slow"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Override in subclass."""
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Override in subclass."""
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe

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
        """Default stake amount logic. Override in subclass if needed."""
        return max(min(proposed_stake, max_stake), min_stake or 0)

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
        """
        Default DCA (Martingale) position adjustment logic.
        Override in subclass for custom behavior.
        """
        entry_count = trade.nr_of_successful_entries
        if entry_count <= 0 or entry_count > len(self.dca_thresholds):
            return None

        threshold = self.dca_thresholds[entry_count - 1]
        if current_profit > threshold:
            return None

        multiplier_index = min(entry_count, len(self.dca_multipliers) - 1)
        stake = min(trade.stake_amount * self.dca_multipliers[multiplier_index], max_stake)

        if min_stake and stake < min_stake:
            return None

        return stake, f"{self.dca_tag_prefix}_{entry_count}"
