#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缠论二买回踩策略

入场：中枢突破后，价格回踩中枢上沿并反弹（二买）
出场：
  - 动态止损：中枢下沿（center_low）
  - ROI 动态止盈：{0: 6%, 60: 3%, 120: 1.5%}
  - 策略退出：价格跌破中枢下沿

相比突破追多策略的优势：
  - 入场价更好（中枢上沿 vs 突破高点）
  - 止损更紧（中枢下沿，非固定百分比）
  - 震荡市胜率更高
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IStrategy
from freqtrade.strategy import merge_informative_pair

try:
    from user_data.strategies.chanlun_core import add_chanlun_signals
except ImportError:
    from chanlun_core import add_chanlun_signals

logger = logging.getLogger(__name__)


class ChanlunSecondBuyStrategy(IStrategy):
    """
    缠论二买回踩策略

    使用 chan_second_buy 信号入场，动态止损（中枢下沿）。
    """

    # ── 策略参数（可由 hyperopt 优化）────────────────────────────────────
    # ROI 止盈表（分钟 → 收益率）
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    # 止损（后备，实际用自定义动态止损）
    stoploss = -0.10

    #  trailing stop（可选）
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False

    # ── 基础设置 ────────────────────────────────────────────────────────
    timeframe = "15m"
    startup_candle_count = 200
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ── 过滤参数（可由 hyperopt 优化）────────────────────────────────
    chan_filter_enabled = CategoricalParameter([True, False], default=True, space="buy")
    chan_filter_bb_period = DecimalParameter(10, 30, default=20, space="buy")
    chan_filter_bb_std = DecimalParameter(1.0, 3.0, default=2.0, space="buy")
    chan_filter_adx_period = DecimalParameter(7, 21, default=14, space="buy")
    chan_filter_adx_threshold = DecimalParameter(15, 35, default=25, space="buy")
    chan_filter_volume_ratio = DecimalParameter(1.0, 3.0, default=1.5, space="buy")
    chan_filter_atr_period = DecimalParameter(7, 21, default=14, space="buy")
    chan_filter_min_atr_pct = DecimalParameter(0.001, 0.01, default=0.003, space="buy")

    # 二买回踩参数
    second_buy_lookback = DecimalParameter(10, 50, default=20, space="buy")

    # ── 自定义止损：基于中枢下沿 ─────────────────────────────────────
    use_custom_stoploss = True

    def custom_stoploss(
        self,
        pair: str,
        trade: type,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        动态止损：中枢下沿。
        返回止损百分比（负数）。
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        if pd.isna(last.get("chan_center_low")) or last.get("chan_center_valid") is not True:
            return self.stoploss  # 后备：固定止损

        center_low = last["chan_center_low"]
        if pd.isna(center_low) or center_low <= 0:
            return self.stoploss

        # 止损 = (center_low - entry_price) / entry_price
        # 但 custom_stoploss 需要返回相对当前价格的止损百分比
        # 这里返回固定百分比，实际止损在 populate_exit_trend 里用 center_low 硬执行
        return -0.10  # 后备，实际用 exit signal

    # ── 信号计算 ─────────────────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = add_chanlun_signals(dataframe.copy(), min_stroke_gap=5)

        # 附加过滤指标
        if self.chan_filter_enabled.value:
            df = self._add_filter_indicators(df)

        return df

    def _add_filter_indicators(self, df: DataFrame) -> DataFrame:
        """添加过滤指标（ATR、BB、ADX、成交量）。"""
        # ATR
        atr_period = int(self.chan_filter_atr_period.value)
        df["atr"] = self._compute_atr(df, atr_period)
        df["atr_pct"] = df["atr"] / df["close"]

        # Bollinger Bands
        bb_period = int(self.chan_filter_bb_period.value)
        bb_std = float(self.chan_filter_bb_std.value)
        df["bb_mid"] = df["close"].rolling(bb_period).mean()
        df["bb_std"] = df["close"].rolling(bb_period).std()
        df["bb_upper"] = df["bb_mid"] + bb_std * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - bb_std * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ADX
        adx_period = int(self.chan_filter_adx_period.value)
        df["adx"] = self._compute_adx(df, adx_period)

        # Volume SMA
        df["volume_sma"] = df["volume"].rolling(20).mean()

        return df

    def _chan_filter(self, dataframe: DataFrame) -> pd.Series:
        """复现 ChanlunCenterBreakoutStrategy 的过滤逻辑。"""
        if not self.chan_filter_enabled.value:
            return pd.Series(True, index=dataframe.index)

        min_atr_pct = float(self.chan_filter_min_atr_pct.value)
        adx_threshold = float(self.chan_filter_adx_threshold.value)
        vol_ratio = float(self.chan_filter_volume_ratio.value)

        return (
            (dataframe["atr_pct"] >= min_atr_pct)
            & (dataframe["bb_width"] > 0.02)
            & (dataframe["adx"] > adx_threshold)
            & (dataframe["volume"] > vol_ratio * dataframe["volume_sma"])
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 二买回踩信号 + 过滤
        filter_pass = self._chan_filter(dataframe)
        dataframe.loc[
            (dataframe["chan_second_buy"] == 1) & filter_pass.shift(1).fillna(False),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 退出信号：
        # 1. 价格跌破中枢下沿（中枢失效）
        # 2. ROI 止盈（由 freqtrade 框架处理）

        prior_center_low = dataframe["chan_center_low"].shift(1)
        prior_close = dataframe["close"].shift(1)

        exit_long = (
            dataframe["chan_center_valid"].shift(1, fill_value=False)
            & prior_center_low.notna()
            & (prior_close >= prior_center_low)
            & (dataframe["close"] < prior_center_low)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[exit_long, "exit_long"] = 1
        return dataframe

    # ── 辅助计算 ────────────────────────────────────────────────────
    @staticmethod
    def _compute_atr(df: DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _compute_adx(df: DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > (-minus_dm.abs())), 0.0)
        minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0.0)

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
        ).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).abs() * 100
        adx = dx.rolling(period).mean()
        return adx.fillna(0)
