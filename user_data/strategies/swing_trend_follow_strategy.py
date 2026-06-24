from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class SwingTrendFollowStrategy(IStrategy):
    """
    Swing trend-following strategy for strong hourly/daily trend regimes.

    Logic summary:
    - Enter only when trend structure shows higher highs and higher lows.
    - Wait for a pullback that holds above the prior structure low.
    - Re-enter strength with volume confirmation instead of trying to bottom tick.
    - Let winners run with trailing stops and add only to profitable trades.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "15m"
    startup_candle_count = 240
    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"720": 0.03, "240": 0.06, "0": 0.18}
    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.022
    trailing_stop_positive_offset = 0.070
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    applicable_market = (
        "Best suited for strong hourly or daily uptrends with orderly pullbacks, "
        "rising market structure, and enough volume expansion to confirm continuation. "
        "Avoid flat, mean-reverting, crash, or illiquid regimes."
    )

    structure_window = 24
    pullback_window = 8
    volume_window = 36
    trend_window = 96
    min_trend_strength = 0.010
    min_volume_ratio = 1.20
    min_atr_pct = 0.002
    max_atr_pct = 0.060
    circuit_atr_pct = 0.080
    circuit_drop_window = 6
    circuit_drop_pct = -0.060
    pyramid_profit_steps = [0.045, 0.090]
    pyramid_stake_multipliers = [0.40, 0.25]

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

    def _timeframe_params(self) -> dict[str, float]:
        if self.timeframe == "1m":
            return {
                "trend_strength": 0.012,
                "volume_ratio": 1.35,
                "momentum": 0.0040,
                "min_atr_pct": 0.0015,
                "rsi_min": 54,
                "rsi_max": 74,
            }
        return {
            "trend_strength": self.min_trend_strength,
            "volume_ratio": self.min_volume_ratio,
            "momentum": 0.003,
            "min_atr_pct": self.min_atr_pct,
            "rsi_min": 52,
            "rsi_max": 78,
        }

    def _timeframe_windows(self) -> dict[str, int | float]:
        if self.timeframe == "1m":
            return {
                "structure": 72,
                "pullback": 20,
                "volume": 90,
                "trend": 240,
                "circuit_drop_window": 15,
                "circuit_drop_pct": -0.035,
            }
        return {
            "structure": self.structure_window,
            "pullback": self.pullback_window,
            "volume": self.volume_window,
            "trend": self.trend_window,
            "circuit_drop_window": self.circuit_drop_window,
            "circuit_drop_pct": self.circuit_drop_pct,
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        windows = self._timeframe_windows()
        dataframe["ema_fast"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        dataframe["ema_trend"] = dataframe["close"].ewm(
            span=int(windows["trend"]), adjust=False
        ).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["atr_pct"] = self._atr_pct(dataframe, 14)

        prior_high = dataframe["high"].shift(1)
        prior_low = dataframe["low"].shift(1)
        structure_window = int(windows["structure"])
        pullback_window = int(windows["pullback"])
        volume_window = int(windows["volume"])
        recent_high = prior_high.rolling(structure_window, min_periods=8).max()
        previous_high = prior_high.shift(structure_window).rolling(
            structure_window, min_periods=8
        ).max()
        recent_low = prior_low.rolling(structure_window, min_periods=8).min()
        previous_low = prior_low.shift(structure_window).rolling(
            structure_window, min_periods=8
        ).min()

        dataframe["higher_high"] = (recent_high > previous_high).fillna(False).astype(int)
        dataframe["higher_low"] = (recent_low > previous_low).fillna(False).astype(int)
        dataframe["structure_low"] = recent_low
        dataframe["pullback_low"] = prior_low.rolling(pullback_window, min_periods=3).min()
        dataframe["pullback_holds_structure"] = (
            dataframe["pullback_low"] > dataframe["structure_low"] * 0.995
        ).fillna(False).astype(int)

        dataframe["volume_mean"] = dataframe["volume"].rolling(
            volume_window, min_periods=12
        ).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)

        dataframe["trend_strength"] = (
            (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"]
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["momentum_3"] = dataframe["close"].pct_change(3).fillna(0)
        dataframe["drawdown_6"] = dataframe["close"].pct_change(
            int(windows["circuit_drop_window"])
        ).fillna(0)

        dataframe["near_fast_ema"] = (
            dataframe["low"] <= dataframe["ema_fast"] * 1.012
        ) & (dataframe["close"] >= dataframe["ema_slow"] * 0.995)
        dataframe["reclaim_after_pullback"] = (
            dataframe["near_fast_ema"]
            & (dataframe["close"] > dataframe["ema_fast"])
            & (dataframe["close"] > dataframe["open"])
        ).astype(int)

        dataframe["market_circuit_breaker"] = (
            (dataframe["atr_pct"] > self.circuit_atr_pct)
            | (dataframe["drawdown_6"] < float(windows["circuit_drop_pct"]))
            | ((dataframe["rsi"] < 34) & (dataframe["momentum_3"] < -0.012))
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        params = self._timeframe_params()

        trend_ok = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow"] > dataframe["ema_trend"])
            & (dataframe["close"] > dataframe["ema_fast"])
            & (dataframe["trend_strength"] >= params["trend_strength"])
        )
        structure_ok = (
            (dataframe["higher_high"] == 1)
            & (dataframe["higher_low"] == 1)
            & (dataframe["pullback_holds_structure"] == 1)
            & (dataframe["reclaim_after_pullback"] == 1)
        )
        volume_ok = dataframe["volume_ratio"] >= params["volume_ratio"]
        volatility_ok = dataframe["atr_pct"].between(params["min_atr_pct"], self.max_atr_pct)
        momentum_ok = (
            (dataframe["momentum_3"] >= params["momentum"])
            & dataframe["rsi"].between(params["rsi_min"], params["rsi_max"])
        )
        circuit_ok = dataframe.get("market_circuit_breaker", 0) == 0

        entry = trend_ok & structure_ok & volume_ok & volatility_ok & momentum_ok & circuit_ok
        dataframe.loc[entry, ["enter_long", "enter_tag"]] = (
            1,
            "structure_pullback_reclaim",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        structure_failed = dataframe["close"] < dataframe["structure_low"] * 0.992
        trend_failed = (dataframe["close"] < dataframe["ema_slow"]) & (dataframe["rsi"] < 46)
        circuit_exit = dataframe["market_circuit_breaker"] == 1

        dataframe.loc[structure_failed, ["exit_long", "exit_tag"]] = (
            1,
            "structure_failed",
        )
        dataframe.loc[trend_failed, ["exit_long", "exit_tag"]] = (1, "trend_failed")
        dataframe.loc[circuit_exit, ["exit_long", "exit_tag"]] = (1, "circuit_breaker")
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
        stake = proposed_stake * 0.58
        if min_stake is not None:
            stake = max(stake, min_stake)
        return round(min(stake, max_stake), 2)

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
        if entry_count <= 0 or entry_count > len(self.pyramid_profit_steps):
            return None

        threshold = self.pyramid_profit_steps[entry_count - 1]
        if current_profit < threshold:
            return None

        stake = trade.stake_amount * self.pyramid_stake_multipliers[entry_count - 1]
        stake = min(stake, max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        return round(stake, 2), f"pyramid_add_{entry_count}"
