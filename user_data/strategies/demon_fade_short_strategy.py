from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class DemonFadeShortStrategy(IStrategy):
    """
    Short failed demon-coin breakouts after the first thrust is rejected.

    This is intentionally not a naive inverse of the long strategy. It waits
    for a prior 24h-high extension, then requires price to fall back below the
    prior high with a rejection wick and negative short-term momentum.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    trading_mode = "futures"
    process_only_new_candles = True
    startup_candle_count = 620

    max_open_trades = 5
    use_custom_stoploss = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 10.0}
    stoploss = -0.06
    trailing_stop = False

    ticket_stake = 1.0
    min_quote_volume_24h = 20_000.0
    fade_score_threshold = 55.0
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

        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()
        df["rsi"] = self._rsi(close)

        prior_high_24h = high.shift(1).rolling(bars_24h, min_periods=144).max()
        df["break_24h_high"] = self._safe_ratio(close, prior_high_24h) - 1.0
        df["prior_breakout_extension"] = (
            self._safe_ratio(high.rolling(bars_1h, min_periods=bars_1h).max(), prior_high_24h)
            - 1.0
        )

        candle_range = (high - low).replace(0, np.nan)
        upper_wick = high - df[["open", "close"]].max(axis=1)
        df["upper_wick_ratio"] = (upper_wick / candle_range).fillna(0.0).clip(0.0, 1.0)
        df["close_position"] = ((close - low) / candle_range).fillna(0.5).clip(0.0, 1.0)

        df["ret_15m"] = self._safe_ratio(close, close.shift(3)) - 1.0
        df["ret_1h"] = self._safe_ratio(close, close.shift(bars_1h)) - 1.0
        df["pre_6h_range"] = (
            self._safe_ratio(
                high.rolling(bars_6h, min_periods=bars_6h).max(),
                low.rolling(bars_6h, min_periods=bars_6h).min(),
            )
            - 1.0
        )
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

        vol_1h = volume.rolling(bars_1h, min_periods=bars_1h).mean()
        vol_24h_base = volume.shift(bars_1h).rolling(bars_24h, min_periods=144).mean()
        df["volume_ratio_1h_24h"] = self._safe_ratio(vol_1h, vol_24h_base)
        df["quote_volume_24h"] = (volume * close).rolling(
            bars_24h, min_periods=144
        ).sum()

        df["fade_score"] = 0.0
        df.loc[df["prior_breakout_extension"] >= 0.02, "fade_score"] += 16
        df.loc[df["break_24h_high"] <= 0.002, "fade_score"] += 16
        df.loc[df["upper_wick_ratio"] >= 0.25, "fade_score"] += 10
        df.loc[df["close_position"] <= 0.55, "fade_score"] += 8
        df.loc[df["ret_15m"] <= -0.005, "fade_score"] += 12
        df.loc[df["volume_ratio_1h_24h"] >= 1.5, "fade_score"] += 12
        df.loc[df["pre_6h_range"] >= 0.06, "fade_score"] += 8
        df.loc[df["range_expansion_ratio"] >= 1.1, "fade_score"] += 8
        df.loc[df["quote_volume_24h"] >= 50_000.0, "fade_score"] += 6

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""

        fade_raw = (
            (df["fade_score"] >= self.fade_score_threshold)
            & (df["quote_volume_24h"] >= self.min_quote_volume_24h)
            & (df["prior_breakout_extension"] >= 0.02)
            & (df["break_24h_high"] <= 0.002)
            & (df["upper_wick_ratio"] >= 0.25)
            & (df["close_position"] <= 0.55)
            & (df["ret_15m"] <= -0.005)
            & (df["volume_ratio_1h_24h"] >= 1.5)
        )
        recent_fade = (
            fade_raw.shift(1)
            .rolling(self.signal_cooldown_candles, min_periods=1)
            .sum()
            .fillna(0)
        )
        fade = fade_raw & (recent_fade == 0)
        df.loc[fade, ["enter_short", "enter_tag"]] = (1, "demon_fade")
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
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
