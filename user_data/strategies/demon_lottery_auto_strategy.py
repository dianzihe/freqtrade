from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_open


class DemonLotteryAutoStrategy(IStrategy):
    """
    Automated "demon coin" scanner/trader for Gate spot 5m data.

    The strategy is intentionally right-confirmed: it only enters after a
    24h-high breakout with short-term price and volume confirmation. The score
    combines the statistically useful OHLCV factors found in the local Gate
    sample with conservative liquidity filters.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 620

    max_open_trades = 5
    position_adjustment_enable = False
    use_custom_stoploss = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 10.0}
    stoploss = -0.08
    trailing_stop = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    ticket_stake = 1.0
    min_quote_volume_24h = 20_000.0
    live_score_threshold = 65.0
    signal_cooldown_candles = 12

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return (
            numerator / denominator.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        bars_1h = 12
        bars_3h = 36
        bars_6h = 72
        bars_24h = 288
        bars_48h = 576

        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()
        df["rsi"] = self._rsi(close)

        df["pre_6h_change"] = self._safe_ratio(close, close.shift(bars_6h)) - 1.0
        pre_high_6h = high.rolling(bars_6h, min_periods=bars_6h).max()
        pre_low_6h = low.rolling(bars_6h, min_periods=bars_6h).min()
        df["pre_6h_range"] = self._safe_ratio(pre_high_6h, pre_low_6h) - 1.0

        log_return = np.log(self._safe_ratio(close, close.shift(1)).replace(0, np.nan))
        df["pre_6h_volatility"] = (
            log_return.rolling(bars_6h, min_periods=bars_6h).std() * np.sqrt(288)
        ).fillna(0.0)

        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_width = 4.0 * close.rolling(20, min_periods=20).std() / bb_mid.replace(0, np.nan)
        df["bb_width_ratio_48h"] = self._safe_ratio(
            bb_width,
            bb_width.shift(1).rolling(bars_48h, min_periods=200).median(),
        )

        vol_1h = volume.rolling(bars_1h, min_periods=bars_1h).mean()
        vol_24h_base = volume.shift(bars_1h).rolling(bars_24h, min_periods=144).mean()
        df["volume_ratio_1h_24h"] = self._safe_ratio(vol_1h, vol_24h_base)

        sma12 = close.rolling(12, min_periods=12).mean()
        df["sma12_slope_6h"] = self._safe_ratio(sma12, sma12.shift(bars_6h)) - 1.0

        staircase = sum((close.shift(i) >= close.shift(i + 1)).astype(int) for i in range(12))
        df["staircase_1h"] = staircase / 12.0

        recent_range = (
            self._safe_ratio(
                high.rolling(bars_3h, min_periods=bars_3h).max(),
                low.rolling(bars_3h, min_periods=bars_3h).min(),
            )
            - 1.0
        )
        earlier_range = (
            self._safe_ratio(
                high.shift(bars_3h).rolling(bars_3h, min_periods=bars_3h).max(),
                low.shift(bars_3h).rolling(bars_3h, min_periods=bars_3h).min(),
            )
            - 1.0
        )
        df["range_expansion_ratio"] = self._safe_ratio(recent_range, earlier_range)

        df["quote_volume_24h"] = (volume * close).rolling(
            bars_24h, min_periods=144
        ).sum()
        previous_high_24h = high.shift(1).rolling(bars_24h, min_periods=144).max()
        df["break_24h_high"] = self._safe_ratio(close, previous_high_24h) - 1.0
        df["ret_15m"] = self._safe_ratio(close, close.shift(3)) - 1.0
        df["ret_1h"] = self._safe_ratio(close, close.shift(bars_1h)) - 1.0

        df["demon_score"] = 0.0
        df.loc[df["pre_6h_change"].between(-0.08, 0.12), "demon_score"] += 8
        df.loc[df["pre_6h_range"].between(0.04, 0.24), "demon_score"] += 8
        df.loc[df["bb_width_ratio_48h"].between(0.0, 1.2), "demon_score"] += 8
        df.loc[df["pre_6h_volatility"] >= 0.08, "demon_score"] += 8
        df.loc[df["volume_ratio_1h_24h"] >= 3.0, "demon_score"] += 14
        df.loc[df["sma12_slope_6h"] >= 0.008, "demon_score"] += 8
        df.loc[df["staircase_1h"] >= 0.65, "demon_score"] += 8
        df.loc[df["range_expansion_ratio"] >= 1.2, "demon_score"] += 8
        df.loc[df["break_24h_high"] >= 0.0, "demon_score"] += 14
        df.loc[df["ret_15m"] >= 0.015, "demon_score"] += 8
        df.loc[df["ret_1h"] >= 0.03, "demon_score"] += 8
        df.loc[df["quote_volume_24h"] >= 50_000.0, "demon_score"] += 5

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["enter_long"] = 0
        df["enter_tag"] = ""

        breakout_raw = (
            (df["demon_score"] >= self.live_score_threshold)
            & (df["quote_volume_24h"] >= self.min_quote_volume_24h)
            & (df["break_24h_high"] >= 0.0)
            & (df["ret_15m"] >= 0.015)
            & (df["ret_1h"] >= 0.03)
            & (df["volume_ratio_1h_24h"] >= 3.0)
            & (df["close"] > df["ema20"])
            & (df["rsi"].between(45, 88))
        )
        recent_breakout = (
            breakout_raw.shift(1)
            .rolling(self.signal_cooldown_candles, min_periods=1)
            .sum()
            .fillna(0)
        )
        breakout = breakout_raw & (recent_breakout == 0)
        df.loc[breakout, ["enter_long", "enter_tag"]] = (1, "demon_breakout")
        return df

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
        **kwargs: Any,
    ) -> float:
        return min(float(self.ticket_stake), float(max_stake))

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> float:
        if current_profit >= 1.0:
            return stoploss_from_open(0.50, current_profit)
        if current_profit >= 0.50:
            return stoploss_from_open(0.25, current_profit)
        if current_profit >= 0.20:
            return stoploss_from_open(0.10, current_profit)
        if current_profit >= 0.10:
            return stoploss_from_open(0.0, current_profit)
        return 1

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        opened = getattr(trade, "open_date_utc", current_time)
        age_h = max((current_time - opened).total_seconds() / 3600.0, 0.0)

        max_rate = getattr(trade, "max_rate", None)
        open_rate = getattr(trade, "open_rate", None)
        peak_profit = None
        if max_rate and open_rate:
            peak_profit = (max_rate - open_rate) / open_rate

        if age_h >= 6 and current_profit <= -0.08:
            return "failed_breakout"
        if age_h >= 12 and (peak_profit is None or peak_profit < 0.03):
            return "stale_no_followthrough"
        if age_h >= 36 and current_profit < 0.05:
            return "stale_weak"
        if age_h >= 72:
            return "max_hold"
        return None
