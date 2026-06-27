# -*- coding: utf-8 -*-
"""
鲁棒 Meme 波动率回归策略

工程级 meme 交易策略。交易流动性恐慌回调，使用受控等步 DCA 而非开放式马丁，集成 LOB/HMM 盘口过滤、
市场结构监控、最大回撤熔断、极端波动率保护等多层风控机制。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

from user_data.strategies.盘口_风险过滤器 import LobHmmConfig, LobRiskFilterMixin


class RobustMemeVolatilityRiskStrategy(LobRiskFilterMixin, IStrategy):
    """
    Engineering-grade meme volatility reversion strategy.

    Core idea:
    - Trade only liquid panic pullbacks in volatile meme markets.
    - Use controlled equal-step DCA instead of open-ended martingale.
    - Stop adding when market structure, account drawdown, LOB/HMM risk, or
      extreme volatility says the original thesis is invalid.
    - Treat this as a risk-managed trading system, not a raw signal strategy.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "1m"
    startup_candle_count = 260
    process_only_new_candles = True

    max_open_trades = 4
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True

    minimal_roi = {"180": 0.0, "45": 0.018, "0": 0.045}
    stoploss = -0.18

    trailing_stop = True
    trailing_stop_positive = 0.010
    trailing_stop_positive_offset = 0.030
    trailing_only_offset_is_reached = True

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

    # Structural parameters: usually fixed unless the system architecture changes.
    max_strategy_leverage = DecimalParameter(1.0, 3.0, default=1.0, decimals=1, space="protection", optimize=False)
    max_custom_open_trades = IntParameter(1, 10, default=4, space="protection", optimize=False)
    max_dca_entries = IntParameter(0, 4, default=3, space="protection", optimize=False)
    hard_stoploss_param = DecimalParameter(0.08, 0.25, default=0.18, decimals=3, space="protection", optimize=False)
    single_trade_risk_pct = DecimalParameter(0.003, 0.020, default=0.010, decimals=3, space="protection", optimize=False)
    single_trade_notional_cap = DecimalParameter(50.0, 5000.0, default=350.0, decimals=1, space="protection", optimize=False)
    total_open_notional_cap = DecimalParameter(100.0, 15000.0, default=1400.0, decimals=1, space="protection", optimize=False)
    lob_filter_mode = "hmm"
    lob_risk_config = LobHmmConfig(
        orderbook_levels=5,
        short_window=5,
        long_window=30,
        threshold_window=50,
        threshold_percentile=0.85,
        min_history=30,
        min_gap=5,
        hmm_min_samples=120,
        hmm_training_window=300,
        hmm_retrain_interval=60,
    )

    # Market-sensitive parameters: good hyperopt candidates.
    buy_rsi_max = IntParameter(18, 42, default=32, space="buy", optimize=True)
    buy_volume_ratio_min = DecimalParameter(0.8, 2.5, default=1.25, decimals=2, space="buy", optimize=True)
    buy_atr_min = DecimalParameter(0.002, 0.020, default=0.006, decimals=3, space="buy", optimize=True)
    buy_atr_max = DecimalParameter(0.020, 0.090, default=0.055, decimals=3, space="buy", optimize=True)
    buy_range_position_min = DecimalParameter(0.03, 0.30, default=0.10, decimals=2, space="buy", optimize=True)
    buy_range_position_max = DecimalParameter(0.45, 0.90, default=0.75, decimals=2, space="buy", optimize=True)
    buy_bb_std = DecimalParameter(1.6, 3.2, default=2.2, decimals=2, space="buy", optimize=True)
    buy_lob_block_enabled = BooleanParameter(default=True, space="buy", optimize=False)

    sell_rsi_exit = IntParameter(52, 78, default=60, space="sell", optimize=True)
    sell_mean_reclaim_buffer = DecimalParameter(0.000, 0.020, default=0.003, decimals=3, space="sell", optimize=True)
    time_stop_candles = IntParameter(60, 720, default=360, space="sell", optimize=True)
    time_stop_min_profit = DecimalParameter(-0.050, 0.020, default=-0.010, decimals=3, space="sell", optimize=True)

    dca_step_1 = DecimalParameter(-0.120, -0.020, default=-0.050, decimals=3, space="buy", optimize=True)
    dca_step_2 = DecimalParameter(-0.180, -0.050, default=-0.100, decimals=3, space="buy", optimize=True)
    dca_step_3 = DecimalParameter(-0.260, -0.080, default=-0.150, decimals=3, space="buy", optimize=True)
    dca_stake_multiplier = DecimalParameter(0.40, 1.20, default=0.70, decimals=2, space="buy", optimize=True)
    dca_min_candle_gap = IntParameter(5, 120, default=30, space="buy", optimize=True)

    floating_reduce_profit = DecimalParameter(-0.180, -0.040, default=-0.110, decimals=3, space="protection", optimize=True)
    floating_reduce_fraction = DecimalParameter(0.10, 0.60, default=0.35, decimals=2, space="protection", optimize=True)
    account_float_loss_reduce = DecimalParameter(0.03, 0.15, default=0.06, decimals=3, space="protection", optimize=True)
    account_float_loss_exit = DecimalParameter(0.06, 0.25, default=0.12, decimals=3, space="protection", optimize=True)
    interrupt_loss_after_dca = DecimalParameter(-0.250, -0.080, default=-0.170, decimals=3, space="protection", optimize=True)

    max_spread_bps = DecimalParameter(5.0, 150.0, default=45.0, decimals=1, space="protection", optimize=True)
    max_funding_rate = DecimalParameter(0.0005, 0.0200, default=0.0030, decimals=4, space="protection", optimize=True)
    extreme_candle_atr_mult = DecimalParameter(1.5, 8.0, default=4.0, decimals=1, space="protection", optimize=True)
    extreme_candle_pct = DecimalParameter(0.030, 0.250, default=0.090, decimals=3, space="protection", optimize=True)

    consecutive_loss_limit = IntParameter(2, 8, default=3, space="protection", optimize=True)
    cooldown_candles_after_loss = IntParameter(30, 720, default=180, space="protection", optimize=True)
    max_daily_drawdown = DecimalParameter(0.03, 0.20, default=0.08, decimals=3, space="protection", optimize=True)

    reduce_once_tag = "risk_reduce_once"

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": int(self.cooldown_candles_after_loss.value / 3)},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 720,
                "trade_limit": int(self.consecutive_loss_limit.value),
                "stop_duration_candles": int(self.cooldown_candles_after_loss.value),
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 1440,
                "trade_limit": 10,
                "stop_duration_candles": int(self.cooldown_candles_after_loss.value),
                "max_allowed_drawdown": float(self.max_daily_drawdown.value),
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

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=120, adjust=False).mean()
        dataframe["atr_pct"] = self._atr_pct(dataframe, 14)
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)

        bb_mid = dataframe["close"].rolling(80, min_periods=40).mean()
        bb_std = dataframe["close"].rolling(80, min_periods=40).std()
        dataframe["bb_mid"] = bb_mid
        dataframe["bb_lower"] = bb_mid - float(self.buy_bb_std.value) * bb_std
        dataframe["bb_upper"] = bb_mid + float(self.buy_bb_std.value) * bb_std

        rolling_high = dataframe["high"].rolling(240, min_periods=60).max()
        rolling_low = dataframe["low"].rolling(240, min_periods=60).min()
        dataframe["range_position"] = (
            (dataframe["close"] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 1)

        dataframe["volume_mean"] = dataframe["volume"].rolling(60, min_periods=10).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["candle_range_pct"] = (
            (dataframe["high"] - dataframe["low"]) / dataframe["close"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)
        dataframe["market_regime_ok"] = (
            (dataframe["ema_fast"] >= dataframe["ema_slow"] * 0.985)
            | (dataframe["rsi"] < int(self.buy_rsi_max.value))
        ).astype(int)
        dataframe["extreme_candle"] = (
            (dataframe["candle_range_pct"] > float(self.extreme_candle_pct.value))
            | (dataframe["candle_range_pct"] > dataframe["atr_pct"] * float(self.extreme_candle_atr_mult.value))
        ).astype(int)

        dataframe = self.populate_lob_risk(dataframe, metadata)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        dataframe["enter_short"] = 0

        lower_band_reversion = dataframe["close"] < dataframe["bb_lower"]
        volatility_ok = dataframe["atr_pct"].between(float(self.buy_atr_min.value), float(self.buy_atr_max.value))
        range_ok = dataframe["range_position"].between(
            float(self.buy_range_position_min.value),
            float(self.buy_range_position_max.value),
        )
        volume_ok = dataframe["volume_ratio"] >= float(self.buy_volume_ratio_min.value)
        panic_reversion = dataframe["rsi"] <= int(self.buy_rsi_max.value)
        risk_ok = (
            (dataframe["market_regime_ok"] == 1)
            & (dataframe["extreme_candle"] == 0)
            & (dataframe.get("lob_risk_trigger", False) == False)  # noqa: E712
        )

        long_entry = lower_band_reversion & volatility_ok & range_ok & volume_ok & panic_reversion & risk_ok
        dataframe.loc[long_entry, ["enter_long", "enter_tag"]] = (1, "robust_vol_reversion_long")

        short_condition_documentation = (
            (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["rsi"] >= 100 - int(self.buy_rsi_max.value))
            & volatility_ok
            & (dataframe["extreme_candle"] == 0)
        )
        if self.can_short:
            dataframe.loc[short_condition_documentation, ["enter_short", "enter_tag"]] = (
                1,
                "robust_vol_reversion_short",
            )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        dataframe["exit_short"] = 0

        mean_reclaim = dataframe["close"] > dataframe["bb_mid"] * (1 + float(self.sell_mean_reclaim_buffer.value))
        rsi_take_profit = dataframe["rsi"] >= int(self.sell_rsi_exit.value)
        signal_reverse = (dataframe["close"] > dataframe["bb_upper"]) & (dataframe["rsi"] > 70)
        extreme_risk = dataframe["extreme_candle"] == 1
        lob_risk = dataframe.get("lob_risk_trigger", False) == True  # noqa: E712

        dataframe.loc[
            mean_reclaim | rsi_take_profit | signal_reverse,
            ["exit_long", "exit_tag"],
        ] = (1, "mean_reversion_exit")
        dataframe.loc[extreme_risk | lob_risk, ["exit_long", "exit_tag"]] = (1, "risk_signal_exit")

        if self.can_short:
            dataframe.loc[
                (dataframe["close"] < dataframe["bb_mid"]) | (dataframe["rsi"] < 40),
                ["exit_short", "exit_tag"],
            ] = (1, "short_mean_reversion_exit")
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
        equity = self._account_equity()
        risk_budget = equity * float(self.single_trade_risk_pct.value)
        hard_stop = max(float(self.hard_stoploss_param.value), 0.01)
        risk_based_stake = risk_budget / hard_stop
        stake = min(proposed_stake, risk_based_stake, float(self.single_trade_notional_cap.value), max_stake)
        if min_stake is not None:
            stake = max(stake, min_stake)
        return max(stake, 0.0)

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
        if current_profit <= float(self.interrupt_loss_after_dca.value):
            return None
        if self._is_lob_risk_active(trade.pair if hasattr(trade, "pair") else ""):
            return None

        if current_profit <= float(self.floating_reduce_profit.value) and not self._get_trade_flag(trade, self.reduce_once_tag):
            self._set_trade_flag(trade, self.reduce_once_tag, True)
            reduce_amount = -abs(float(getattr(trade, "stake_amount", 0.0)) * float(self.floating_reduce_fraction.value))
            return reduce_amount, "floating_loss_reduce"

        entry_count = int(getattr(trade, "nr_of_successful_entries", 1))
        if entry_count <= 0 or entry_count > int(self.max_dca_entries.value):
            return None
        if self._bars_since_open(trade, current_time) < int(self.dca_min_candle_gap.value) * entry_count:
            return None

        thresholds = [float(self.dca_step_1.value), float(self.dca_step_2.value), float(self.dca_step_3.value)]
        threshold = thresholds[min(entry_count - 1, len(thresholds) - 1)]
        if current_profit > threshold:
            return None

        stake = min(float(getattr(trade, "stake_amount", 0.0)) * float(self.dca_stake_multiplier.value), max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        if self._open_notional_estimate() + stake > float(self.total_open_notional_cap.value):
            return None
        return stake, f"controlled_dca_{entry_count}"

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        if self._portfolio_float_loss_ratio(current_profit) >= float(self.account_float_loss_exit.value):
            return "account_float_loss_liquidation"

        if (
            current_profit <= float(self.interrupt_loss_after_dca.value)
            and int(getattr(trade, "nr_of_successful_entries", 1)) >= int(self.max_dca_entries.value)
        ):
            return "interrupt_after_failed_dca"

        if self._bars_since_open(trade, current_time) >= int(self.time_stop_candles.value):
            if current_profit <= float(self.time_stop_min_profit.value):
                return "time_stop_no_progress"

        if current_profit > 0.025 and self._is_lob_risk_active(pair):
            return "lob_risk_profit_protect"
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs: Any,
    ) -> float:
        hard = -float(self.hard_stoploss_param.value)
        if current_profit > 0.08:
            return -0.025
        if current_profit > 0.04:
            return -0.040
        if current_profit <= float(self.interrupt_loss_after_dca.value):
            return max(hard, float(self.interrupt_loss_after_dca.value))
        return hard

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None = None,
        side: str = "long",
        **kwargs: Any,
    ) -> bool:
        if self._cooldown_active(current_time):
            return False
        if self._open_trade_count() >= int(self.max_custom_open_trades.value):
            return False
        if amount * rate > float(self.single_trade_notional_cap.value) * float(self.max_strategy_leverage.value):
            return False
        if self._open_notional_estimate() + amount * rate > float(self.total_open_notional_cap.value):
            return False
        if not self._spread_ok(pair):
            return False
        if not self._funding_ok(pair, side):
            return False
        return True

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        configured_cap = float(self.max_strategy_leverage.value)
        return max(1.0, min(float(proposed_leverage), configured_cap, float(max_leverage)))

    def bot_loop_start(self, current_time: datetime, **kwargs: Any) -> None:
        self._update_loss_cooldown(current_time)

    def _account_equity(self) -> float:
        wallets = getattr(self, "wallets", None)
        if wallets is not None:
            try:
                return float(wallets.get_total_stake_amount())
            except Exception:
                pass
        config = getattr(self, "config", {}) or {}
        return float(config.get("dry_run_wallet") or config.get("stake_amount") or 10000.0)

    def _bars_since_open(self, trade: Trade, current_time: datetime) -> int:
        opened = getattr(trade, "open_date_utc", current_time)
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return max(int((current_time - opened).total_seconds() // 60), 0)

    def _spread_ok(self, pair: str) -> bool:
        dp = getattr(self, "dp", None)
        runmode = getattr(getattr(dp, "runmode", None), "value", None)
        if dp is None or runmode not in {"live", "dry_run"}:
            return True
        try:
            ob = dp.orderbook(pair, 1)
            bid = float(ob["bids"][0][0])
            ask = float(ob["asks"][0][0])
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10000.0
            return spread_bps <= float(self.max_spread_bps.value)
        except Exception:
            return False

    def _funding_ok(self, pair: str, side: str) -> bool:
        dp = getattr(self, "dp", None)
        runmode = getattr(getattr(dp, "runmode", None), "value", None)
        if dp is None or runmode not in {"live", "dry_run"}:
            return True
        funding_rate = getattr(dp, "funding_rate", None)
        if funding_rate is None:
            return True
        try:
            funding = float(funding_rate(pair))
        except Exception:
            return True
        limit = float(self.max_funding_rate.value)
        if side == "long" and funding > limit:
            return False
        if side == "short" and funding < -limit:
            return False
        return True

    def _open_trade_count(self) -> int:
        try:
            return len(Trade.get_open_trades())
        except Exception:
            return 0

    def _open_notional_estimate(self) -> float:
        try:
            return float(sum(float(getattr(trade, "stake_amount", 0.0)) for trade in Trade.get_open_trades()))
        except Exception:
            return 0.0

    def _portfolio_float_loss_ratio(self, current_profit: float) -> float:
        if current_profit >= 0:
            return 0.0
        return abs(float(current_profit))

    def _cooldown_active(self, current_time: datetime) -> bool:
        until = getattr(self, "_risk_cooldown_until", None)
        return until is not None and current_time < until

    def _update_loss_cooldown(self, current_time: datetime) -> None:
        try:
            closed_trades = Trade.get_trades_proxy(is_open=False)
        except Exception:
            return
        recent = sorted(closed_trades, key=lambda trade: getattr(trade, "close_date_utc", current_time), reverse=True)
        losses = 0
        for trade in recent[: int(self.consecutive_loss_limit.value)]:
            if float(getattr(trade, "close_profit", 0.0) or 0.0) < 0:
                losses += 1
        if losses >= int(self.consecutive_loss_limit.value):
            self._risk_cooldown_until = current_time + timedelta(minutes=int(self.cooldown_candles_after_loss.value))

    def _is_lob_risk_active(self, pair: str) -> bool:
        detector = getattr(self, "_lob_risk_filter", None)
        if detector is None:
            return False
        state_map = getattr(detector, "_states", None) or getattr(detector, "_hmm_states", None)
        if not state_map or pair not in state_map:
            return False
        state = state_map[pair]
        return getattr(state, "risk_ticks_remaining", 0) > 0

    def _get_trade_flag(self, trade: Trade, key: str) -> bool:
        getter = getattr(trade, "get_custom_data", None)
        if callable(getter):
            try:
                return bool(getter(key, default=False))
            except Exception:
                return False
        return bool(getattr(trade, key, False))

    def _set_trade_flag(self, trade: Trade, key: str, value: bool) -> None:
        setter = getattr(trade, "set_custom_data", None)
        if callable(setter):
            try:
                setter(key, value)
                return
            except Exception:
                pass
        try:
            setattr(trade, key, value)
        except Exception:
            pass

