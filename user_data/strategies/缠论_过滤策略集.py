# -*- coding: utf-8 -*-
"""
缠论过滤策略集

基类基于缠论中枢突破提供信号，子类分别通过 MACD 背驰、RSI 时机、成交量确认、MA+RSI 二买信号进行过滤优化。
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

try:
    from user_data.strategies.缠论_核心模块 import add_chanlun_signals
except ImportError:
    from 缠论_核心模块 import add_chanlun_signals


class ChanlunFilteredBaseStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    # 改动：1m -> 15m
    timeframe = "15m"
    # 改动：startup 从 30 提升，保证 EMA/RSI/中枢都已收敛
    startup_candle_count = 200
    process_only_new_candles = True

    # 改动：禁用“利润为 0 即平仓”的负期望 ROI，改成递减止盈表
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

    # 改动：笔最小长度 3 -> 5
    min_stroke_gap = 5

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
        return (left > right) & (left.shift(1) <= right.shift(1))

    @staticmethod
    def _crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
        return (left < right) & (left.shift(1) >= right.shift(1))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_chanlun_signals(dataframe, min_stroke_gap=self.min_stroke_gap)

        ema12 = dataframe["close"].ewm(span=12, adjust=False).mean()
        ema26 = dataframe["close"].ewm(span=26, adjust=False).mean()
        dataframe["macd"] = ema12 - ema26
        dataframe["macdsignal"] = dataframe["macd"].ewm(span=9, adjust=False).mean()
        dataframe["macdhist"] = dataframe["macd"] - dataframe["macdsignal"]

        dataframe["rsi"] = self._rsi(dataframe["close"])
        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=26, adjust=False).mean()
        dataframe["volume_mean"] = dataframe["volume"].rolling(20, min_periods=5).mean()
        dataframe["price_progress"] = dataframe["close"].pct_change(3).abs().fillna(0)

        # 二买：依赖修正后的中枢有效性（中枢突破后会失效，避免陈旧中枢误判）
        prior_center_low = dataframe["chan_center_low"].shift(1)
        dataframe["chan_second_buy"] = (
            dataframe["chan_center_valid"].shift(1, fill_value=False)
            & prior_center_low.notna()
            & (dataframe["low"].rolling(3).min() <= prior_center_low * 1.01)
            & (dataframe["close"] > prior_center_low)
            & (dataframe["close"] > dataframe["close"].shift(1))
            & (dataframe["volume"] > 0)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe


class ChanlunMacdDivergenceStrategy(ChanlunFilteredBaseStrategy):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        recent_low = dataframe["low"].rolling(3).min()
        recent_hist_low = dataframe["macdhist"].rolling(3).min()
        bullish_divergence = (recent_low < dataframe["low"].rolling(5).min().shift(3)) & (
            recent_hist_low > dataframe["macdhist"].rolling(5).min().shift(3)
        )
        dataframe.loc[
            (dataframe["chan_enter_long"] == 1) & bullish_divergence,
            ["enter_long", "enter_tag"],
        ] = (1, "chanlun_macd_bullish_divergence")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        recent_high = dataframe["high"].rolling(3).max()
        recent_hist_high = dataframe["macdhist"].rolling(3).max()
        bearish_divergence = (recent_high > dataframe["high"].rolling(5).max().shift(3)) & (
            recent_hist_high < dataframe["macdhist"].rolling(5).max().shift(3)
        )
        dataframe.loc[
            (dataframe["chan_exit_long"] == 1) & bearish_divergence,
            ["exit_long", "exit_tag"],
        ] = (1, "chanlun_macd_bearish_divergence")
        return dataframe


class ChanlunRsiTimingStrategy(ChanlunFilteredBaseStrategy):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        oversold_rebound = (dataframe["rsi"].rolling(5).min() < 32) & (
            dataframe["rsi"] > dataframe["rsi"].shift(1)
        )
        dataframe.loc[
            (dataframe["chan_enter_long"] == 1) & oversold_rebound,
            ["enter_long", "enter_tag"],
        ] = (1, "chanlun_rsi_oversold_rebound")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        overbought_rollover = (dataframe["rsi"].rolling(5).max() > 68) & (
            dataframe["rsi"] < dataframe["rsi"].shift(1)
        )
        dataframe.loc[
            (dataframe["chan_exit_long"] == 1) & overbought_rollover,
            ["exit_long", "exit_tag"],
        ] = (1, "chanlun_rsi_overbought_rollover")
        return dataframe


class ChanlunVolumeConfirmationStrategy(ChanlunFilteredBaseStrategy):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        dry_volume = dataframe["volume"].rolling(3).mean() < dataframe["volume_mean"] * 0.45
        dataframe.loc[
            (dataframe["chan_enter_long"] == 1) & dry_volume,
            ["enter_long", "enter_tag"],
        ] = (1, "chanlun_volume_dry_entry")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        heavy_volume = dataframe["volume"].rolling(3).mean() > dataframe["volume_mean"] * 1.8
        stalled_price = dataframe["price_progress"].rolling(3).mean() < 0.003
        dataframe.loc[
            (dataframe["chan_exit_long"] == 1) & heavy_volume & stalled_price,
            ["exit_long", "exit_tag"],
        ] = (1, "chanlun_volume_stall_exit")
        return dataframe


class ChanlunMaRsiSecondBuyStrategy(ChanlunFilteredBaseStrategy):
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        ma_golden_cross = self._crossed_above(dataframe["ema_fast"], dataframe["ema_slow"])
        rsi_rebound = (dataframe["rsi"].rolling(5).min() < 35) & (
            dataframe["rsi"] > dataframe["rsi"].shift(1)
        )
        dataframe.loc[
            dataframe["chan_second_buy"] & ma_golden_cross & rsi_rebound,
            ["enter_long", "enter_tag"],
        ] = (1, "chanlun_second_buy_ma_rsi")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        # 改动：原来用 ema_fast < ema_slow（持续成立）会一进场就被打出，
        # 改成均线死叉（穿越瞬间）才离场，减少震荡市频繁微亏。
        ma_death_cross = self._crossed_below(dataframe["ema_fast"], dataframe["ema_slow"])
        dataframe.loc[
            (dataframe["chan_exit_long"] == 1) | ma_death_cross,
            ["exit_long", "exit_tag"],
        ] = (1, "chanlun_second_buy_exit")
        return dataframe