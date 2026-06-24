from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class GateXingheSpotStrategy(IStrategy):
    """Conservative long-only adaptation of Xinghe's adaptive basket grid."""

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "1m"
    startup_candle_count = 240
    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"360": 0.006, "120": 0.012, "0": 0.025}
    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    base_grid_step = 0.002
    max_grid_step = 0.035
    dca_profit_thresholds = (-0.020, -0.050, -0.090)
    dca_stake_multipliers = (0.60, 0.45, 0.30)

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 20},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 240,
                "trade_limit": 3,
                "stop_duration_candles": 120,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 720,
                "trade_limit": 12,
                "stop_duration_candles": 240,
                "max_allowed_drawdown": 0.10,
            },
        ]

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50).clip(0, 100)

    @staticmethod
    def _atr_pct(dataframe: DataFrame, period: int = 14) -> pd.Series:
        previous_close = dataframe["close"].shift(1)
        true_range = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - previous_close).abs(),
                (dataframe["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
        return (atr / dataframe["close"]).replace([np.inf, -np.inf], 0).fillna(0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=60, adjust=False).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(span=180, adjust=False).mean()
        dataframe["trend_slope"] = dataframe["ema_slow"].pct_change(6).fillna(0)
        dataframe["rsi"] = self._rsi(dataframe["close"])
        dataframe["atr_pct"] = self._atr_pct(dataframe)
        dataframe["grid_step_pct"] = (
            self.base_grid_step + dataframe["atr_pct"] * 0.75
        ).clip(self.base_grid_step, self.max_grid_step)
        dataframe["pullback_level"] = dataframe["ema_fast"] * (1 - dataframe["grid_step_pct"])

        volume_mean = dataframe["volume"].rolling(60, min_periods=1).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / volume_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["return_30"] = dataframe["close"].pct_change(30).fillna(0)
        atr_mean = dataframe["atr_pct"].rolling(60, min_periods=1).mean()
        dataframe["risk_fuse"] = (
            (dataframe["return_30"] < -0.075)
            | (dataframe["atr_pct"] > 0.045)
            | (dataframe["atr_pct"] > atr_mean * 3.0)
            | (dataframe["volume_ratio"] > 7.0)
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        trend_ok = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow"] > dataframe["ema_trend"])
            & (dataframe["trend_slope"] > 0)
        )
        pullback_reclaimed = (
            (dataframe["low"] <= dataframe["pullback_level"])
            & (dataframe["close"] >= dataframe["pullback_level"])
        )
        quality_ok = (
            dataframe["rsi"].between(38, 62)
            & dataframe["atr_pct"].between(0.0005, 0.04)
            & (dataframe["volume_ratio"] >= 0.70)
            & (dataframe["risk_fuse"] == 0)
        )
        entry = trend_ok & pullback_reclaimed & quality_ok & (dataframe["volume"] > 0)
        dataframe.loc[entry, ["enter_long", "enter_tag"]] = (1, "spot_trend_grid_pullback")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        trend_broken = (
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["close"] < dataframe["ema_trend"])
            & (dataframe["rsi"] < 42)
        )
        dataframe.loc[trend_broken, ["exit_long", "exit_tag"]] = (1, "spot_trend_broken")
        dataframe.loc[dataframe["risk_fuse"] == 1, ["exit_long", "exit_tag"]] = (
            1,
            "spot_risk_fuse",
        )
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
        stake = proposed_stake * 0.42
        if min_stake is not None:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    @staticmethod
    def _base_entry_stake(trade: Trade) -> float:
        try:
            orders = trade.select_filled_orders(trade.entry_side)
            if orders:
                order_cost = float(orders[0].safe_cost)
                if order_cost > 0:
                    return order_cost / max(float(getattr(trade, "leverage", 1.0)), 1.0)
        except (AttributeError, TypeError):
            pass
        return float(trade.stake_amount)

    def _latest_row(self, trade: Trade) -> pd.Series | None:
        pair = getattr(trade, "pair", None)
        dp = getattr(self, "dp", None)
        if not pair or dp is None:
            return None
        dataframe, _ = dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        return dataframe.iloc[-1]

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
        if entry_count <= 0 or entry_count > len(self.dca_profit_thresholds):
            return None

        row = self._latest_row(trade)
        threshold = self.dca_profit_thresholds[entry_count - 1]
        if row is not None:
            atr_threshold = -float(row["atr_pct"]) * (2.2 + entry_count * 1.1)
            threshold = min(threshold, atr_threshold)
            trend_valid = (
                row["ema_slow"] > row["ema_trend"]
                and row["trend_slope"] > -0.001
                and row["risk_fuse"] == 0
            )
            if not trend_valid:
                return None
        if current_profit > threshold:
            return None

        stake = round(
            self._base_entry_stake(trade) * self.dca_stake_multipliers[entry_count - 1], 8
        )
        stake = min(stake, max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        return stake, f"spot_grid_add_{entry_count}"

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if current_profit >= 0.014:
            return "spot_basket_profit"
        open_time = getattr(trade, "open_date_utc", None)
        if open_time and current_time - open_time >= timedelta(days=3):
            return "spot_timeout"
        return None
