# -*- coding: utf-8 -*-
"""
市场状态策略集

基类基于比特币（BTC）价格走势判断市场风险偏好，子类分别实现：波动突破动量、成交量 ATR 均值回归、
多因子 BTC 同步、风险优先强趋势等策略，用于过滤山寨币交易信号。
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class MarketRegimeBaseStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "15m"
    startup_candle_count = 80
    process_only_new_candles = True

    minimal_roi = {"0": 0.03, "120": 0.015, "360": 0.0}
    stoploss = -0.03
    trailing_stop = False

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

    def informative_pairs(self) -> list[tuple[str, str]]:
        return [("BTC/USDT", self.timeframe)]

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=26, adjust=False).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(span=55, adjust=False).mean()
        dataframe["trend_strength"] = dataframe["ema_fast"] / dataframe["ema_slow"] - 1
        dataframe["rsi"] = self._rsi(dataframe["close"])

        dataframe["atr"] = self._atr(dataframe)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace(
            [np.inf, -np.inf], 0
        )
        dataframe["volume_mean"] = dataframe["volume"].rolling(20, min_periods=5).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"]
        ).replace([np.inf, -np.inf], 0)

        dataframe["breakout_high"] = dataframe["close"].rolling(24, min_periods=12).max().shift(1)
        dataframe["breakdown_low"] = dataframe["close"].rolling(12, min_periods=6).min().shift(1)

        bb_mid = dataframe["close"].rolling(20, min_periods=10).mean()
        bb_std = dataframe["close"].rolling(20, min_periods=10).std()
        dataframe["bb_mid"] = bb_mid
        dataframe["bb_lower"] = bb_mid - 2 * bb_std
        dataframe["bb_upper"] = bb_mid + 2 * bb_std

        dataframe["return"] = dataframe["close"].pct_change().fillna(0)
        dataframe["btc_corr"] = self._btc_correlation(dataframe, metadata)
        dataframe["factor_score"] = 0

        return dataframe

    def _btc_correlation(self, dataframe: DataFrame, metadata: dict) -> pd.Series:
        pair = metadata.get("pair", "")
        if pair == "BTC/USDT":
            return pd.Series(1.0, index=dataframe.index)

        if not self.dp:
            return pd.Series(0.0, index=dataframe.index)

        btc_dataframe = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        if btc_dataframe is None or btc_dataframe.empty or "close" not in btc_dataframe:
            return pd.Series(0.0, index=dataframe.index)

        btc_returns = btc_dataframe[["date", "close"]].copy()
        btc_returns["btc_return"] = btc_returns["close"].pct_change().fillna(0)
        merged = dataframe[["date", "return"]].merge(
            btc_returns[["date", "btc_return"]], on="date", how="left"
        )
        corr = merged["return"].rolling(24, min_periods=8).corr(merged["btc_return"])
        return corr.fillna(0).set_axis(dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe


class VolatilityBreakoutMomentumStrategy(MarketRegimeBaseStrategy):
    stoploss = -0.025
    minimal_roi = {"0": 0.035, "90": 0.018, "240": 0.0}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        breakout = dataframe["close"] > dataframe["breakout_high"] * 1.003
        volatility_ok = dataframe["atr_pct"].between(0.004, 0.035)
        trend_ok = dataframe["trend_strength"] > 0.006
        volume_ok = dataframe["volume_ratio"] > 1.2
        dataframe.loc[
            breakout & volatility_ok & trend_ok & volume_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "vol_breakout_momentum")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["close"] < dataframe["ema_fast"]) | (dataframe["rsi"] > 82),
            ["exit_long", "exit_tag"],
        ] = (1, "momentum_exit")
        return dataframe


class VolumeAtrMeanReversionStrategy(MarketRegimeBaseStrategy):
    stoploss = -0.035
    minimal_roi = {"0": 0.02, "120": 0.01, "360": 0.0}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        lower_band_flush = dataframe["close"] < dataframe["bb_lower"]
        volatility_ok = dataframe["atr_pct"].between(0.004, 0.04)
        volume_extreme = dataframe["volume_ratio"] > 1.8
        oversold = dataframe["rsi"] < 35
        dataframe.loc[
            lower_band_flush & volatility_ok & volume_extreme & oversold,
            ["enter_long", "enter_tag"],
        ] = (1, "volume_atr_mean_reversion")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["close"] >= dataframe["bb_mid"]) | (dataframe["rsi"] > 58),
            ["exit_long", "exit_tag"],
        ] = (1, "mean_reversion_mid_exit")
        return dataframe


class MultiFactorBtcSyncStrategy(MarketRegimeBaseStrategy):
    stoploss = -0.03
    minimal_roi = {"0": 0.03, "120": 0.015, "360": 0.0}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        dataframe["factor_score"] = (
            (dataframe["trend_strength"] > 0.01).astype(int)
            + dataframe["atr_pct"].between(0.004, 0.03).astype(int)
            + (dataframe["volume_ratio"] > 1.15).astype(int)
            + (dataframe["btc_corr"] > 0.25).astype(int)
            + (dataframe["close"] > dataframe["breakout_high"]).astype(int)
        )
        dataframe.loc[
            dataframe["factor_score"] >= 4,
            ["enter_long", "enter_tag"],
        ] = (1, "multi_factor_btc_sync")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["factor_score"] <= 2)
            | (dataframe["close"] < dataframe["ema_slow"])
            | (dataframe["btc_corr"] < -0.1),
            ["exit_long", "exit_tag"],
        ] = (1, "multi_factor_exit")
        return dataframe


class RiskFirstStrongTrendStrategy(MarketRegimeBaseStrategy):
    max_open_trades = 2
    stoploss = -0.015
    minimal_roi = {"0": 0.025, "60": 0.012, "180": 0.0}
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_entry_trend(dataframe, metadata)
        strong_trend = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow"] > dataframe["ema_trend"])
            & (dataframe["trend_strength"] > 0.015)
        )
        breakout = dataframe["close"] > dataframe["breakout_high"] * 1.002
        clean_volatility = dataframe["atr_pct"].between(0.003, 0.02)
        liquid = dataframe["volume_ratio"] > 1.1
        dataframe.loc[
            strong_trend & breakout & clean_volatility & liquid,
            ["enter_long", "enter_tag"],
        ] = (1, "risk_first_strong_trend")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        super().populate_exit_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["close"] < dataframe["ema_fast"])
            | (dataframe["trend_strength"] < 0.005)
            | (dataframe["close"] < dataframe["breakdown_low"]),
            ["exit_long", "exit_tag"],
        ] = (1, "risk_first_exit")
        return dataframe
