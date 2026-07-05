# -*- coding: utf-8 -*-
"""Standalone Freqtrade adaptation of the LOB latent-regime v7 detector."""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy


class LOBLatentRegimeV7Strategy(IStrategy):
    """Use v7-style LOB stress warnings as a conservative short filter."""

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    startup_candle_count = 220
    process_only_new_candles = True

    minimal_roi = {"0": 0.05, "30": 0.025, "90": 0.01, "180": 0.0}
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.045
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    lob_window = 24
    lob_signal_pct = 85
    lob_confirm_bars = 3
    lob_min_strength = 0.35
    lob_cooldown_bars = 25
    lob_edge_diff_steps = 3
    lob_gamma_amp = 0.35

    plot_config = {
        "subplots": {
            "LOB v7 channels": {
                "lob_entropy": {"color": "#34495E"},
                "lob_prestress_proxy": {"color": "#E67E22"},
                "lob_spread_drift": {"color": "#27AE60"},
                "lob_depth_erosion": {"color": "#8E44AD"},
                "lob_ofi_momentum": {"color": "#2980B9"},
            },
            "LOB v7 trigger": {
                "lob_score": {"color": "#1B4F8A"},
                "lob_threshold": {"color": "#E65100"},
                "lob_trigger": {"color": "#B5341B", "type": "bar"},
            },
        }
    }

    def informative_pairs(self) -> list:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        window = self.lob_window

        df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=48, adjust=False).mean()

        spread = self._spread_ratio(df)
        depth = self._total_depth(df)
        imbalance, ofi = self._order_flow_features(df)
        returns = df["close"].pct_change().fillna(0.0)

        df["lob_entropy"] = self._volatility_entropy(returns, window)
        df["lob_spread_drift"] = self._spread_drift(spread, window)
        df["lob_depth_erosion"] = self._depth_erosion(depth, window)
        df["lob_ofi_momentum"] = self._ofi_momentum(imbalance, ofi, window)
        df["lob_prestress_proxy"] = self._prestress_proxy(
            df["lob_entropy"],
            df["lob_spread_drift"],
            df["lob_depth_erosion"],
        )

        return self._build_trigger(
            df,
            window=window,
            signal_pct=self.lob_signal_pct,
            confirm_bars=self.lob_confirm_bars,
            min_strength=self.lob_min_strength,
            cooldown_bars=self.lob_cooldown_bars,
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        downside = dataframe["close"].pct_change(5) < -0.006
        below_slow = dataframe["close"] < dataframe["ema_slow"]
        volume_ok = dataframe["volume"] > 0
        stressed = (dataframe["lob_trigger"] == 1) & (dataframe["lob_score"] >= 0.45)

        mask = stressed & downside & below_slow & volume_ok
        dataframe.loc[mask, "enter_short"] = 1
        dataframe.loc[mask, "enter_tag"] = "lob_v7_stress_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        faded = dataframe["lob_score"] < (dataframe["lob_threshold"].fillna(0.0) * 0.55)
        recovered = dataframe["close"] > dataframe["ema_fast"]
        volume_ok = dataframe["volume"] > 0
        dataframe.loc[(faded | recovered) & volume_ok, "exit_short"] = 1

        return dataframe

    def _build_trigger(
        self,
        df: DataFrame,
        window: int,
        signal_pct: int,
        confirm_bars: int,
        min_strength: float,
        cooldown_bars: int,
    ) -> DataFrame:
        channels = [
            "lob_entropy",
            "lob_prestress_proxy",
            "lob_spread_drift",
            "lob_depth_erosion",
            "lob_ofi_momentum",
        ]
        channel_strength = df[channels].clip(lower=0.0)
        score_raw = channel_strength.max(axis=1)
        drift_rising = (df["lob_spread_drift"] > 0.30).astype(float)
        score_amp = score_raw * (1.0 + self.lob_gamma_amp * drift_rising)
        df["lob_score"] = score_amp.rolling(3, min_periods=1).mean().clip(0.0, 1.35)

        edge_lag = max(1, self.lob_edge_diff_steps)
        df["lob_score_delta"] = (df["lob_score"] - df["lob_score"].shift(edge_lag)).fillna(0.0)

        long_window = max(window * 4, 48)
        threshold = (
            df["lob_score"]
            .rolling(long_window, min_periods=max(window * 2, 10))
            .quantile(signal_pct / 100.0)
        )
        df["lob_threshold"] = threshold.clip(lower=min_strength)

        above = df["lob_score"] > df["lob_threshold"].fillna(np.inf)
        rising = df["lob_score_delta"] > 0
        confirmed = above.copy()
        for lag in range(1, max(1, confirm_bars)):
            confirmed = confirmed & above.shift(lag).fillna(False)

        trigger = (confirmed & rising & (df["lob_score"] > min_strength)).astype(int)
        df["lob_trigger"] = self._deduplicate_trigger(trigger, cooldown_bars)
        return df

    @staticmethod
    def _volatility_entropy(returns: Series, window: int) -> Series:
        short_w = max(4, window // 4)
        mid_w = max(8, window // 2)
        vol_s = returns.rolling(short_w, min_periods=2).std()
        vol_m = returns.rolling(mid_w, min_periods=2).std()
        vol_l = returns.rolling(window, min_periods=2).std()
        vol_sum = (vol_s + vol_m + vol_l).replace(0.0, np.nan)
        probs = [vol_s / vol_sum, vol_m / vol_sum, vol_l / vol_sum]
        eps = 1e-12
        entropy = sum(-(p * np.log(p.clip(lower=eps))) for p in probs)
        return (entropy / np.log(3)).fillna(0.0).clip(0.0, 1.0)

    def _prestress_proxy(self, entropy: Series, spread_drift: Series, depth_erosion: Series) -> Series:
        combined = 0.40 * entropy + 0.35 * spread_drift + 0.25 * depth_erosion
        return self._rolling_rank(combined, window=max(self.lob_window * 3, 48)).fillna(0.0)

    def _spread_drift(self, spread: Series, window: int) -> Series:
        fast = spread.rolling(max(4, window // 2), min_periods=2).mean()
        long_window = max(window * 2, 12)
        slow = spread.rolling(long_window, min_periods=window).mean()
        ma_cross = (fast - slow).clip(lower=0.0)
        spread_mom = spread.diff().rolling(max(4, window // 2), min_periods=2).mean().clip(lower=0.0)
        cum_drift = spread.diff().clip(lower=0.0).rolling(long_window, min_periods=2).sum()
        spread_std = spread.rolling(long_window, min_periods=window).std().replace(0.0, np.nan)
        spread_z = ((spread - slow) / spread_std).fillna(0.0)
        spread_shock = (spread_z / 3.0).clip(lower=0.0, upper=1.0)
        components = pd.concat(
            [
                self._norm01(ma_cross),
                self._norm01(spread_mom),
                self._norm01(cum_drift),
                spread_shock,
            ],
            axis=1,
        )
        return components.max(axis=1).fillna(0.0)

    def _depth_erosion(self, depth: Series, window: int) -> Series:
        short = depth.rolling(max(4, window // 2), min_periods=2).mean()
        long_window = max(window * 2, 12)
        long = depth.rolling(long_window, min_periods=window).mean()
        depth_below = (long - short).clip(lower=0.0)
        depth_velocity = (-depth.diff(5)).rolling(max(4, window // 2), min_periods=2).mean()
        depth_velocity = depth_velocity.clip(lower=0.0)
        depth_std = depth.rolling(long_window, min_periods=window).std().replace(0.0, np.nan)
        depth_z = ((depth - long) / depth_std).fillna(0.0)
        depth_shock = (-depth_z / 3.0).clip(lower=0.0, upper=1.0)
        components = pd.concat(
            [
                self._norm01(depth_below),
                self._norm01(depth_velocity),
                depth_shock,
            ],
            axis=1,
        )
        return components.max(axis=1).fillna(0.0)

    def _ofi_momentum(self, imbalance: Series, ofi: Series, window: int) -> Series:
        abs_imb = imbalance.abs().rolling(max(window, 8), min_periods=2).mean()
        abs_ofi = ofi.abs().rolling(max(window, 8), min_periods=2).mean()
        return (self._norm01(abs_imb) + self._norm01(abs_ofi)).div(2.0).fillna(0.0)

    def _order_flow_features(self, df: DataFrame) -> tuple[Series, Series]:
        buy = self._series_from_columns(df, ["buy_volume", "buy_qty", "taker_buy_volume"])
        sell = self._series_from_columns(df, ["sell_volume", "sell_qty", "taker_sell_volume"])
        if buy is None or sell is None:
            buy_ratio = df.get("buy_volume_ratio", Series(0.5, index=df.index)).fillna(0.5)
            volume = df.get("volume", Series(0.0, index=df.index)).fillna(0.0)
            buy = buy_ratio * volume
            sell = (1.0 - buy_ratio) * volume

        total = (buy + sell).replace(0.0, np.nan)
        imbalance = ((buy - sell) / total).fillna(0.0).clip(-1.0, 1.0)
        ofi = (buy - sell).diff().fillna(0.0)
        return imbalance, ofi

    def _total_depth(self, df: DataFrame) -> Series:
        raw_depth = self._depth_from_l2_columns(df)
        if raw_depth is not None:
            return raw_depth

        total = self._series_from_columns(
            df,
            ["total_depth", "top10_depth", "depth_10", "total_depth_10", "l2_total_depth_10"],
        )
        if total is not None:
            return total.fillna(0.0)

        bid = self._series_from_columns(df, ["bid_depth", "bid_depth_10", "bids_depth_10"])
        ask = self._series_from_columns(df, ["ask_depth", "ask_depth_10", "asks_depth_10"])
        if bid is not None and ask is not None:
            return bid.fillna(0.0) + ask.fillna(0.0)

        return Series(0.0, index=df.index)

    def _spread_ratio(self, df: DataFrame) -> Series:
        raw_spread = self._spread_from_l2_columns(df)
        if raw_spread is not None:
            return raw_spread

        bid = self._series_from_columns(df, ["best_bid", "bid_price", "bid"])
        ask = self._series_from_columns(df, ["best_ask", "ask_price", "ask"])
        if bid is not None and ask is not None:
            mid = ((bid + ask) / 2.0).replace(0.0, np.nan)
            return ((ask - bid) / mid).fillna(0.0).clip(lower=0.0)

        spread = self._series_from_columns(df, ["spread", "best_spread", "lob_spread"])
        mid = self._series_from_columns(df, ["mid_price", "mid"])
        if spread is not None and mid is not None:
            return (spread / mid.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0)

        close = df["close"].replace(0.0, np.nan)
        return ((df["high"] - df["low"]) / close).fillna(0.0).clip(lower=0.0)

    def _depth_from_l2_columns(self, df: DataFrame) -> Series | None:
        bids_col = self._first_existing_column(df, ["bids", "bid_updates", "l2_bids"])
        asks_col = self._first_existing_column(df, ["asks", "ask_updates", "l2_asks"])
        if bids_col is None or asks_col is None:
            return None

        return Series(
            [
                self._sum_top_levels(bids, 25) + self._sum_top_levels(asks, 25)
                for bids, asks in zip(df[bids_col], df[asks_col])
            ],
            index=df.index,
            dtype="float64",
        )

    def _spread_from_l2_columns(self, df: DataFrame) -> Series | None:
        bids_col = self._first_existing_column(df, ["bids", "bid_updates", "l2_bids"])
        asks_col = self._first_existing_column(df, ["asks", "ask_updates", "l2_asks"])
        if bids_col is None or asks_col is None:
            return None

        return Series(
            [self._best_spread_ratio(bids, asks) for bids, asks in zip(df[bids_col], df[asks_col])],
            index=df.index,
            dtype="float64",
        )

    @staticmethod
    def _deduplicate_trigger(trigger: Series, cooldown_bars: int) -> np.ndarray:
        values = trigger.fillna(0).astype(int).to_numpy().copy()
        last_trigger = -cooldown_bars - 1
        for pos, value in enumerate(values):
            if not value:
                continue
            if pos - last_trigger <= cooldown_bars:
                values[pos] = 0
                continue
            last_trigger = pos
        return values

    @staticmethod
    def _norm01(series: Series) -> Series:
        finite = series.replace([np.inf, -np.inf], np.nan)
        lo = finite.rolling(120, min_periods=5).min()
        hi = finite.rolling(120, min_periods=5).max()
        return ((finite - lo) / (hi - lo).replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)

    @staticmethod
    def _rolling_rank(series: Series, window: int) -> Series:
        return series.rolling(window, min_periods=max(10, window // 3)).apply(
            lambda values: float(np.mean(values[-1] >= values)),
            raw=True,
        )

    @staticmethod
    def _series_from_columns(df: DataFrame, names: Iterable[str]) -> Series | None:
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
        return None

    @staticmethod
    def _first_existing_column(df: DataFrame, names: Iterable[str]) -> str | None:
        for name in names:
            if name in df.columns:
                return name
        return None

    @classmethod
    def _sum_top_levels(cls, value, levels: int) -> float:
        total = 0.0
        for level in cls._parse_l2_levels(value)[:levels]:
            try:
                total += max(float(level[1]), 0.0)
            except (IndexError, TypeError, ValueError):
                continue
        return total

    @classmethod
    def _best_spread_ratio(cls, bids, asks) -> float:
        bid_levels = cls._parse_l2_levels(bids)
        ask_levels = cls._parse_l2_levels(asks)
        if not bid_levels or not ask_levels:
            return 0.0
        try:
            best_bid = float(bid_levels[0][0])
            best_ask = float(ask_levels[0][0])
        except (IndexError, TypeError, ValueError):
            return 0.0
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0.0:
            return 0.0
        return max((best_ask - best_bid) / mid, 0.0)

    @staticmethod
    def _parse_l2_levels(value) -> list:
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return []
            return parsed if isinstance(parsed, list) else []
        return value if isinstance(value, list) else []
