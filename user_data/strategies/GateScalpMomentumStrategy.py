import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class GateScalpMomentumStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "1m"
    startup_candle_count = 60
    process_only_new_candles = True

    minimal_roi = {
        "0": 0.008,
        "6": 0.004,
        "18": 0.0015,
        "36": 0,
    }
    stoploss = -0.012
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.007
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

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    @staticmethod
    def _atr(dataframe: DataFrame, period: int = 14) -> pd.Series:
        high_low = dataframe["high"] - dataframe["low"]
        high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
        low_close = (dataframe["low"] - dataframe["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.ewm(alpha=1 / period, adjust=False).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=8, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(span=55, adjust=False).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20, min_periods=5).mean()
        dataframe["atr"] = self._atr(dataframe, 14)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace([np.inf, -np.inf], 0)
        dataframe["recent_high"] = dataframe["close"].rolling(12, min_periods=3).max()
        dataframe["pullback_pct"] = (
            (dataframe["recent_high"] - dataframe["close"]) / dataframe["recent_high"]
        ).fillna(0)
        dataframe["momentum_3m"] = dataframe["close"].pct_change(3).fillna(0)
        dataframe["range_pct"] = ((dataframe["high"] - dataframe["low"]) / dataframe["close"]).fillna(0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        trend_ok = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["close"] > dataframe["ema_trend"])
        )
        pullback_rebound = (
            (dataframe["pullback_pct"].rolling(4, min_periods=2).max() > 0.0025)
            & (dataframe["rsi"].rolling(5, min_periods=2).min() < 42)
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["momentum_3m"] > 0.001)
        )
        liquidity_ok = (
            (dataframe["volume"] > dataframe["volume_mean"] * 1.05)
            & (dataframe["volume_mean"] > 0)
        )
        volatility_ok = dataframe["atr_pct"].between(0.0008, 0.035)

        dataframe.loc[
            trend_ok & pullback_rebound & liquidity_ok & volatility_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "scalp_pullback_rebound")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        momentum_fade = (
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            | ((dataframe["momentum_3m"] < -0.001) & (dataframe["rsi"] < 50))
        )
        exhaustion = (dataframe["rsi"] > 74) & (
            dataframe["momentum_3m"] < dataframe["momentum_3m"].shift(1)
        )

        dataframe.loc[momentum_fade, ["exit_long", "exit_tag"]] = (
            1,
            "scalp_momentum_fade",
        )
        dataframe.loc[
            exhaustion & (dataframe["exit_long"] == 0),
            ["exit_long", "exit_tag"],
        ] = (1, "scalp_exhaustion")
        return dataframe
