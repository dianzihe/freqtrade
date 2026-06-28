# -*- coding: utf-8 -*-
"""
趋势跟踪反马丁策略

首次入场仓位较小，只在盈利时顺势加仓，绝不摊平亏损单。趋势辨识 + 金字塔加仓 + 追踪止损。
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class TrendPyramidAntiMartingaleStrategy(IStrategy):
    """
    Trend-following strategy that pyramids only into profitable positions.

    The first entry is intentionally smaller than the proposed stake so there is
    room to add on strength without averaging down losing trades.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    startup_candle_count = 300
    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"240": 0.02, "90": 0.04, "0": 0.10}
    stoploss = -0.08

    trailing_stop = True
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.055
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    breakout_window = 72
    volume_ratio_min = 1.50
    min_atr_pct = 0.0015
    max_atr_pct = 0.05
    circuit_atr_pct = 0.065
    circuit_drop_window = 6
    circuit_drop_pct = -0.055
    pyramid_profit_steps = [0.03, 0.075]
    pyramid_stake_multipliers = [0.45, 0.30]

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

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
        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
        return (atr / dataframe["close"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=60, adjust=False).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(span=120, adjust=False).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["atr_pct"] = self._atr_pct(dataframe, 14)

        dataframe["breakout_high"] = (
            dataframe["high"].rolling(self.breakout_window, min_periods=20).max().shift(1)
        )
        dataframe["volume_mean"] = dataframe["volume"].rolling(36, min_periods=12).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)

        dataframe["trend_strength"] = (
            (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["momentum_3"] = dataframe["close"].pct_change(3).fillna(0)
        dataframe["drawdown_6"] = dataframe["close"].pct_change(self.circuit_drop_window).fillna(0)
        dataframe["swing_low"] = dataframe["low"].rolling(12, min_periods=3).min()
        dataframe["market_circuit_breaker"] = (
            (dataframe["atr_pct"] > self.circuit_atr_pct)
            | (dataframe["drawdown_6"] < self.circuit_drop_pct)
            | ((dataframe["rsi"] < 32) & (dataframe["momentum_3"] < -0.01))
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        is_fast_window = self.timeframe == "1m"
        trend_strength_min = 0.025 if is_fast_window else 0.012
        volume_ratio_min = 1.80 if is_fast_window else self.volume_ratio_min
        momentum_min = 0.035 if is_fast_window else 0.004
        breakout_buffer = 1.0015 if is_fast_window else 1.0005
        min_atr_pct = 0.0120 if is_fast_window else self.min_atr_pct
        rsi_min = 58 if is_fast_window else 54
        rsi_max = 76 if is_fast_window else 78

        trend_ok = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow"] > dataframe["ema_trend"])
            & (dataframe["close"] > dataframe["ema_fast"])
            & (dataframe["trend_strength"] > trend_strength_min)
        )
        breakout = dataframe["close"] > dataframe["breakout_high"] * breakout_buffer
        liquidity_ok = dataframe["volume_ratio"] >= volume_ratio_min
        volatility_ok = dataframe["atr_pct"].between(min_atr_pct, self.max_atr_pct)
        momentum_ok = (
            (dataframe["close"] > dataframe["open"])
            & (dataframe["momentum_3"] > momentum_min)
            & dataframe["rsi"].between(rsi_min, rsi_max)
        )
        circuit_ok = dataframe.get("market_circuit_breaker", 0) == 0

        entry = trend_ok & breakout & liquidity_ok & volatility_ok & momentum_ok & circuit_ok
        dataframe.loc[entry, ["enter_long", "enter_tag"]] = (1, "trend_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        trend_failed = (dataframe["close"] < dataframe["ema_fast"]) | (dataframe["rsi"] < 45)
        exhaustion = (
            (dataframe["rsi"] > 82)
            & (dataframe["volume_ratio"] > 2.0)
            & (dataframe["close"] < dataframe["close"].shift(1))
        )

        dataframe.loc[trend_failed, ["exit_long", "exit_tag"]] = (1, "trend_failed")
        dataframe.loc[exhaustion, ["exit_long", "exit_tag"]] = (1, "trend_exhaustion")
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
        stake = proposed_stake * 0.60
        if min_stake is not None:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

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
        if entry_count <= 0 or entry_count > len(self.pyramid_profit_steps):
            return None

        threshold = self.pyramid_profit_steps[entry_count - 1]
        if current_profit < threshold:
            return None

        stake = trade.stake_amount * self.pyramid_stake_multipliers[entry_count - 1]
        stake = min(stake, max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        return stake, f"pyramid_add_{entry_count}"
