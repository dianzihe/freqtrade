# -*- coding: utf-8 -*-
"""
Gate Xinghe futures grid strategy.

Freqtrade adaptation of the Xinghe Quantitative MT5 idea:
mixed trend signal + ATR-spaced grid DCA + bounded futures risk controls.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class GateXingheFuturesGridStrategy(IStrategy):
    """
    Directional Gate futures grid strategy inspired by 星河量化.mq5.

    This is not a hedge-mode rescue clone. Freqtrade handles one trade object at
    a time, so adverse movement is handled with bounded DCA and hard risk exits.
    """

    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "1m"
    startup_candle_count = 240
    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 5

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 0.025}
    stoploss = -0.18

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    ema_fast_period = 20
    ema_slow_period = 60
    atr_period = 14
    volatility_period = 20

    min_atr_pct = 0.0006
    max_atr_pct = 0.055
    min_grid_step_pct = 0.006
    max_grid_step_pct = 0.060
    atr_grid_multiplier = 2.0
    grid_growth_per_entry = 0.25

    dca_multipliers = [1.3, 1.6, 2.0, 2.4, 3.0]
    deep_loss_fuse = -0.22
    entry_stake_reserve = 0.45

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

    def _grid_threshold(self, atr_pct: float, entry_count: int) -> float:
        raw = atr_pct * self.atr_grid_multiplier
        grown = raw * (1 + max(entry_count, 0) * self.grid_growth_per_entry)
        return max(self.min_grid_step_pct, min(grown, self.max_grid_step_pct))

    def _first_filled_entry_cost(self, trade: Trade) -> float | None:
        try:
            orders = trade.select_filled_orders("entry")
        except (AttributeError, TypeError):
            return None

        if not orders:
            return None

        safe_cost = getattr(orders[0], "safe_cost", None)
        if safe_cost is None or safe_cost <= 0:
            return None
        return float(safe_cost)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(
            span=self.ema_fast_period, adjust=False
        ).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(
            span=self.ema_slow_period, adjust=False
        ).mean()
        dataframe["atr_pct"] = self._atr_pct(dataframe, self.atr_period)
        dataframe["volatility_pct"] = (
            dataframe["close"].pct_change().rolling(self.volatility_period).std().fillna(0)
        )

        dataframe["ema_vote"] = np.select(
            [
                dataframe["ema_fast"] > dataframe["ema_slow"],
                dataframe["ema_fast"] < dataframe["ema_slow"],
            ],
            [1, -1],
            default=0,
        )
        dataframe["atr_vote"] = np.select(
            [
                (dataframe["atr_pct"] > dataframe["atr_pct"].shift(1))
                & (dataframe["close"] > dataframe["open"]),
                (dataframe["atr_pct"] > dataframe["atr_pct"].shift(1))
                & (dataframe["close"] < dataframe["open"]),
            ],
            [1, -1],
            default=0,
        )
        dataframe["volatility_vote"] = np.select(
            [
                (dataframe["volatility_pct"] > dataframe["volatility_pct"].rolling(60).mean())
                & (dataframe["close"] > dataframe["ema_fast"]),
                (dataframe["volatility_pct"] > dataframe["volatility_pct"].rolling(60).mean())
                & (dataframe["close"] < dataframe["ema_fast"]),
            ],
            [1, -1],
            default=0,
        )

        vote_sum = dataframe["ema_vote"] + dataframe["atr_vote"] + dataframe["volatility_vote"]
        dataframe["trend_signal"] = np.select([vote_sum >= 2, vote_sum <= -2], [1, -1], default=0)
        dataframe["grid_step_pct"] = (
            dataframe["atr_pct"] * self.atr_grid_multiplier
        ).clip(lower=self.min_grid_step_pct, upper=self.max_grid_step_pct)
        dataframe["risk_fuse"] = (
            (dataframe["atr_pct"] < self.min_atr_pct)
            | (dataframe["atr_pct"] > self.max_atr_pct)
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        tradable = (
            (dataframe["volume"] > 0)
            & (dataframe["risk_fuse"] == 0)
            & dataframe["atr_pct"].between(self.min_atr_pct, self.max_atr_pct)
        )

        long_entry = tradable & (dataframe["trend_signal"] == 1)
        short_entry = tradable & (dataframe["trend_signal"] == -1)

        dataframe.loc[long_entry, ["enter_long", "enter_tag"]] = (
            1,
            "xinghe_mixed_long",
        )
        dataframe.loc[short_entry, ["enter_short", "enter_tag"]] = (
            1,
            "xinghe_mixed_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        dataframe.loc[
            (dataframe["trend_signal"] == -1) | (dataframe["risk_fuse"] == 1),
            ["exit_long", "exit_tag"],
        ] = (1, "xinghe_long_risk_or_reversal")
        dataframe.loc[
            (dataframe["trend_signal"] == 1) | (dataframe["risk_fuse"] == 1),
            ["exit_short", "exit_tag"],
        ] = (1, "xinghe_short_risk_or_reversal")
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
        **kwargs: Any,
    ) -> float:
        stake = proposed_stake * self.entry_stake_reserve
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
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        entry_count = trade.nr_of_successful_entries
        if entry_count <= 0 or entry_count > len(self.dca_multipliers):
            return None

        if current_profit <= self.deep_loss_fuse or current_profit >= 0:
            return None

        open_rate = getattr(trade, "open_rate", current_entry_rate)
        if open_rate <= 0:
            return None

        is_short = bool(getattr(trade, "is_short", False))
        adverse_move = (
            (current_rate / open_rate - 1)
            if is_short
            else (open_rate / current_rate - 1)
        )
        threshold = self._grid_threshold(
            self.min_grid_step_pct / self.atr_grid_multiplier,
            entry_count - 1,
        )
        if adverse_move < threshold:
            return None

        base_cost = self._first_filled_entry_cost(trade) or getattr(trade, "stake_amount", 0.0)
        if base_cost <= 0:
            return None

        stake = base_cost * self.dca_multipliers[entry_count - 1]
        stake = min(stake, max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        return stake, f"xinghe_dca_{entry_count}"
