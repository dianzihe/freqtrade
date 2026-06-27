from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class MemeMartingaleBaseStrategy(IStrategy):
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
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
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
        dataframe["ema_fast"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=120, adjust=False).mean()
        dataframe["rolling_high_60"] = dataframe["close"].rolling(60).max()
        dataframe["rolling_high_240"] = dataframe["close"].rolling(240).max()
        dataframe["rolling_low_240"] = dataframe["close"].rolling(240).min()
        dataframe["volume_mean_60"] = dataframe["volume"].rolling(60).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_mean_60"]
        dataframe["volume_ratio"] = dataframe["volume_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
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
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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


class MemeLimitedDcaMartingaleStrategy(MemeMartingaleBaseStrategy):
    """
    Limited crash-catching martingale: few layers, low multipliers, hard invalidation.
    """

    minimal_roi = {"90": 0.0, "25": 0.025, "0": 0.06}
    stoploss = -0.24
    dca_tag_prefix = "limited_dca"

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        crash_pullback = dataframe["drawdown_60"] < -0.09
        liquid_rebound = (dataframe["volume_ratio"] > 1.55) & (dataframe["rsi"] < 32)
        not_broken = dataframe["range_position"].between(0.12, 0.78)
        dataframe.loc[crash_pullback & liquid_rebound & not_broken, ["enter_long", "enter_tag"]] = (
            1,
            "limited_crash_rebound",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["rsi"] > 63) | (dataframe["close"] > dataframe["ema_fast"] * 1.035),
            ["exit_long", "exit_tag"],
        ] = (1, "limited_rebound_exit")
        return dataframe


class MemeVolatilityGridMartingaleStrategy(MemeMartingaleBaseStrategy):
    """
    Volatility grid martingale: trades range reversion instead of assuming every dip recovers.
    """

    minimal_roi = {"180": 0.0, "45": 0.018, "0": 0.04}
    stoploss = -0.20
    dca_thresholds = [-0.05, -0.10, -0.18, -0.30]
    dca_multipliers = [1.0, 1.2, 1.4, 1.6]
    dca_tag_prefix = "vol_grid_dca"

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        wide_enough = dataframe["atr_pct"] > 0.006
        lower_band_touch = dataframe["close"] < dataframe["bb_lower"]
        range_not_dead = dataframe["range_position"].between(0.12, 0.72)
        dataframe.loc[wide_enough & lower_band_touch & range_not_dead, ["enter_long", "enter_tag"]] = (
            1,
            "vol_grid_lower_band",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_mid"]) | (dataframe["rsi"] > 58),
            ["exit_long", "exit_tag"],
        ] = (1, "vol_grid_mean_exit")
        return dataframe


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
        dataframe.loc[breakout & volume_confirmed & trend_confirmed, ["enter_long", "enter_tag"]] = (
            1,
            "anti_martingale_breakout",
        )
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
