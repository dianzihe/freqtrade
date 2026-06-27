# -*- coding: utf-8 -*-
"""
趋势过滤网格策略

高时间框架上升趋势确认后，在日内回调时入场。15 分钟趋势转弱时暂停新入场，仅在盈利后网格加仓。
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative


class TrendFilteredGridStrategy(IStrategy):
    """
    Long-only trend filtered grid strategy for Gate spot data.

    It is designed for markets with a clear higher-timeframe uptrend and
    intraday pullbacks. It pauses new entries when the 15m trend weakens and
    avoids averaging down; grid additions are allowed only after the trade is
    already profitable.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "1m"
    startup_candle_count = 240
    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"180": 0.008, "60": 0.014, "0": 0.035}
    stoploss = -0.075

    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.018
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    grid_base_step = 0.0035
    grid_atr_multiplier = 0.80
    volume_ratio_min = 0.85
    max_atr_pct = 0.032
    min_atr_pct = 0.0008
    profit_add_steps = [0.014, 0.030]
    profit_add_multipliers = [0.35, 0.25]

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 20},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 240,
                "trade_limit": 3,
                "stop_duration_candles": 90,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 720,
                "trade_limit": 10,
                "stop_duration_candles": 180,
                "max_allowed_drawdown": 0.08,
            },
        ]

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

    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=60, adjust=False).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(span=120, adjust=False).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["trend_slope"] = dataframe["ema_slow"].pct_change(6).fillna(0)
        dataframe["trend_up"] = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow"] > dataframe["ema_trend"])
            & (dataframe["close"] > dataframe["ema_slow"])
            & (dataframe["trend_slope"] > 0)
            & dataframe["rsi"].between(48, 78)
        ).astype(int)
        dataframe["trend_weak"] = (
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            | (dataframe["close"] < dataframe["ema_slow"] * 0.994)
            | (dataframe["rsi"] < 42)
        ).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_grid"] = dataframe["close"].ewm(span=34, adjust=False).mean()
        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=55, adjust=False).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["atr_pct"] = self._atr_pct(dataframe, 14)
        dataframe["grid_step_pct"] = (
            self.grid_base_step + dataframe["atr_pct"] * self.grid_atr_multiplier
        ).clip(lower=self.grid_base_step, upper=0.018)
        dataframe["grid_pullback_level"] = dataframe["ema_grid"] * (1 - dataframe["grid_step_pct"])
        dataframe["volume_mean"] = dataframe["volume"].rolling(60, min_periods=10).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["daily_high"] = dataframe["high"].rolling(1440, min_periods=60).max()
        dataframe["daily_drop_pct"] = (
            dataframe["close"] / dataframe["daily_high"].replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["cooldown_risk"] = (
            (dataframe["daily_drop_pct"] < -0.09)
            | (dataframe["atr_pct"] > self.max_atr_pct)
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        trend_up = dataframe.get("trend_up_15m", 0) == 1
        trend_not_weak = dataframe.get("trend_weak_15m", 0) == 0
        risk_ok = dataframe["cooldown_risk"] == 0
        drawdown_ok = dataframe["daily_drop_pct"] > -0.055
        volatility_ok = dataframe["atr_pct"].between(self.min_atr_pct, self.max_atr_pct)
        pullback_hit = dataframe["low"] <= dataframe["grid_pullback_level"]
        reclaim = dataframe["close"] >= dataframe["grid_pullback_level"] * 1.001
        center_reclaim = dataframe["close"] >= (
            dataframe["ema_grid"] * (1 - dataframe["grid_step_pct"] * 0.35)
        )
        local_structure_ok = (
            (dataframe["ema_fast"] >= dataframe["ema_slow"] * 0.998)
            & dataframe["rsi"].between(42, 64)
            & (dataframe["volume_ratio"] >= self.volume_ratio_min)
        )

        entry = (
            trend_up
            & trend_not_weak
            & risk_ok
            & drawdown_ok
            & volatility_ok
            & pullback_hit
            & reclaim
            & center_reclaim
            & local_structure_ok
        )
        dataframe.loc[entry, ["enter_long", "enter_tag"]] = (1, "trend_grid_pullback")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        trend_weak = dataframe.get("trend_weak_15m", 0) == 1
        local_break = (
            (dataframe["close"] < dataframe["ema_grid"] * (1 - dataframe["grid_step_pct"] * 2.2))
            & (dataframe["rsi"] < 42)
        )
        fuse = dataframe["cooldown_risk"] == 1
        exhaustion = (
            (dataframe["rsi"] > 78)
            & (dataframe["volume_ratio"] > 1.8)
            & (dataframe["close"] < dataframe["close"].shift(1))
        )

        dataframe.loc[trend_weak | local_break, ["exit_long", "exit_tag"]] = (
            1,
            "trend_or_grid_break",
        )
        dataframe.loc[fuse, ["exit_long", "exit_tag"]] = (1, "risk_fuse")
        dataframe.loc[exhaustion, ["exit_long", "exit_tag"]] = (1, "exhaustion")
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
        stake = proposed_stake * 0.55
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
        if entry_count <= 0 or entry_count > len(self.profit_add_steps):
            return None

        threshold = self.profit_add_steps[entry_count - 1]
        if current_profit < threshold:
            return None

        stake = trade.stake_amount * self.profit_add_multipliers[entry_count - 1]
        stake = min(stake, max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        return stake, f"grid_profit_add_{entry_count}"
