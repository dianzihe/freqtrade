# -*- coding: utf-8 -*-
"""
顶部反转做空策略

纯做空策略。通过极端涨幅、成交量扩张、波动率扩张来代理资金费率过热信号，在顶部崩塌时做空。
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute


class TopReversalShortStrategy(IStrategy):
    """
    Short-only top reversal strategy for overheated moves.

    Local gate data only contains OHLCV, so funding-rate overheating is proxied by
    extreme runup, volume expansion, volatility expansion, and failed breakout.
    """

    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "1m"
    startup_candle_count = 120
    process_only_new_candles = True

    stoploss = -0.055
    minimal_roi = {"0": 0.045, "45": 0.025, "180": 0.0}

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.030
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = True

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 12,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 80,
                "trade_limit": 2,
                "stop_duration_candles": 80,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 240,
                "trade_limit": 4,
                "stop_duration_candles": 120,
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
    def _atr(dataframe: DataFrame, period: int = 14) -> pd.Series:
        prev_close = dataframe["close"].shift(1)
        true_range = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - prev_close).abs(),
                (dataframe["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1 / period, adjust=False).mean()

    def _periods(self) -> dict[str, int]:
        minutes = max(timeframe_to_minutes(self.timeframe), 1)
        return {
            "runup": max(24 * 60 // minutes, 48),
            "breakout": max(8 * 60 // minutes, 24),
            "support": max(90 // minutes, 12),
            "volume": max(90 // minutes, 12),
            "circuit": max(6 * 60 // minutes, 24),
        }

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        periods = self._periods()

        dataframe["ema_fast"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=55, adjust=False).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["atr"] = self._atr(dataframe, 14)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace(
            [np.inf, -np.inf], 0
        )

        volume_mean = dataframe["volume"].rolling(periods["volume"], min_periods=6).mean()
        dataframe["volume_mean"] = volume_mean
        dataframe["volume_ratio"] = (dataframe["volume"] / volume_mean).replace(
            [np.inf, -np.inf], 0
        )

        prior_low = dataframe["low"].rolling(periods["runup"], min_periods=24).min().shift(1)
        dataframe["runup_24h"] = (dataframe["high"] / prior_low - 1).replace(
            [np.inf, -np.inf], np.nan
        )

        dataframe["failed_breakout_high"] = (
            dataframe["high"].rolling(periods["breakout"], min_periods=12).max().shift(1)
        )
        dataframe["support_low"] = (
            dataframe["low"].rolling(periods["support"], min_periods=6).min().shift(1)
        )
        dataframe["breakdown_level"] = dataframe["support_low"] * 0.997

        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        body_high = dataframe[["open", "close"]].max(axis=1)
        dataframe["upper_wick_ratio"] = ((dataframe["high"] - body_high) / candle_range).fillna(0)
        dataframe["momentum_3"] = dataframe["close"].pct_change(3).fillna(0)
        dataframe["failed_breakout"] = (
            (dataframe["high"] > dataframe["failed_breakout_high"] * 1.003)
            & (dataframe["close"] < dataframe["failed_breakout_high"])
        ).astype(int)

        squeeze_move = dataframe["close"].pct_change(periods["circuit"])
        recent_atr = dataframe["atr_pct"].rolling(periods["support"], min_periods=6).mean()
        dataframe["market_circuit_breaker"] = (
            (squeeze_move > 0.45)
            | (dataframe["atr_pct"] > recent_atr * 2.8)
            | (dataframe["volume_ratio"] > 6.0)
        ).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        overheated_now = (
            (dataframe["runup_24h"] > 0.75)
            & (dataframe["volume_ratio"] > 1.7)
            & dataframe["atr_pct"].between(0.004, 0.09)
            & (dataframe["rsi"] > 68)
        )
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        body_high = dataframe[["open", "close"]].max(axis=1)
        upper_wick_ratio = dataframe.get(
            "upper_wick_ratio", ((dataframe["high"] - body_high) / candle_range).fillna(0)
        )
        failed_breakout_flag = dataframe.get("failed_breakout", pd.Series(0, index=dataframe.index))

        failed_breakout_now = (
            (failed_breakout_flag == 1)
            | (
                (dataframe["high"] > dataframe["failed_breakout_high"] * 1.003)
                & (dataframe["close"] < dataframe["failed_breakout_high"])
                & (upper_wick_ratio > 0.35)
            )
        )
        context_window = max(self._periods()["support"], 8)
        overheated_recent = overheated_now.rolling(context_window, min_periods=1).max().astype(bool)
        failed_breakout_recent = (
            failed_breakout_now.rolling(context_window, min_periods=1).max().astype(bool)
        )
        support_loss = (
            (dataframe["close"] < dataframe["breakdown_level"])
            & (dataframe["close"] < dataframe["open"])
        )
        not_meltdown = dataframe["market_circuit_breaker"] == 0

        dataframe.loc[
            overheated_recent & failed_breakout_recent & support_loss & not_meltdown,
            ["enter_short", "enter_tag"],
        ] = (1, "failed_breakout_support_loss")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        profit_snapback = (
            (dataframe["rsi"] < 34)
            & (dataframe["momentum_3"] < -0.018)
            & (dataframe["close"] < dataframe["ema_fast"] * 0.97)
        )
        squeeze_risk = (
            (dataframe["close"] > dataframe["ema_fast"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["volume_ratio"] > 2.0)
            & (dataframe["rsi"] > 64)
        )

        dataframe.loc[profit_snapback, ["exit_short", "exit_tag"]] = (1, "profit_snapback")
        dataframe.loc[squeeze_risk, ["exit_short", "exit_tag"]] = (1, "squeeze_risk_exit")

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(1.5, max_leverage)

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

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float | None:
        stop_price = trade.get_custom_data("stop_price")

        if stop_price is None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None

            entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            candle = dataframe.loc[dataframe["date"] == entry_date]
            if candle.empty:
                return None

            failed_high = float(candle["failed_breakout_high"].iloc[0])
            atr = float(candle["atr"].iloc[0])
            stop_price = max(failed_high + atr * 0.8, trade.open_rate * 1.018)
            stop_price = min(stop_price, trade.open_rate * 1.055)
            trade.set_custom_data("stop_price", stop_price)
            trade.set_custom_data("r_pct", float((stop_price - trade.open_rate) / trade.open_rate))

        if trade.get_custom_data("tp1_done"):
            stop_price = min(stop_price, trade.open_rate * 0.998)

        return stoploss_from_absolute(
            stop_price, current_rate, is_short=True, leverage=trade.leverage
        )

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
    ):
        r_pct = trade.get_custom_data("r_pct")
        if not r_pct or r_pct <= 0:
            return None

        if not trade.get_custom_data("tp1_done") and current_profit >= 1.0 * r_pct:
            trade.set_custom_data("tp1_done", True)
            return -(trade.stake_amount * 0.45), "take_profit_1r"

        if (
            trade.get_custom_data("tp1_done")
            and not trade.get_custom_data("tp2_done")
            and current_profit >= 1.8 * r_pct
        ):
            trade.set_custom_data("tp2_done", True)
            return -(trade.stake_amount * 0.35), "take_profit_18r"

        return None
