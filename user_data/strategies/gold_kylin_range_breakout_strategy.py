# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)


class GoldKylinRangeBreakoutStrategy(IStrategy):
    """
    金麒麟区间突破策略的工程化风控版。

    设计取向：
    - 原 MT5 EA 用上下 BuyStop/SellStop、对冲和马丁处理震荡行情。
    - CEX 永续/现货不适合照搬对锁和马丁，所以这里保留“区间两端触发”的核心，
      但把加仓改为递减式安全单，并用硬止损、减仓、熔断、盘口、资金费率等多层风控约束风险。
    - 所有指标都只使用当前或历史 K 线；区间高低点使用 shift(1)，避免 look-ahead bias。
    """

    INTERFACE_VERSION = 3

    # 上轮回测显示空头分支在样本内 3/3 亏损，且现货是主要验证场景。
    # 因此主策略默认改为 long-only；如果后续要研究合约做空，用底部的实验子类单独打开。
    can_short = False
    timeframe = "5m"
    startup_candle_count = 300
    process_only_new_candles = True

    # config.json 中的 max_open_trades 仍是最终上限；这里给出策略侧默认值，并在 confirm_trade_entry 中二次校验。
    max_open_trades = 4

    # 开启 position adjustment，主要用于浮亏减仓；默认不再逆势加仓，避免震荡策略在单边行情里越补越重。
    position_adjustment_enable = True
    max_entry_position_adjustment = 1

    # ROI 负责常规止盈；更细的“时间止损/中断退出/账户浮亏清仓”放在 custom_exit。
    minimal_roi = {
        "180": 0.0,
        "45": 0.006,
        "0": 0.014,
    }

    # 类级 stoploss 是最终本地硬闸。custom_stoploss 只会收紧，不会放宽到比它更差。
    stoploss = -0.08
    use_custom_stoploss = True

    # 原生 trailing stop 在盈利后保护利润；custom_stoploss 再按收益阶段动态收紧。
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        # 为了“直接可跑”默认不用交易所止损单；实盘强烈建议在支持的交易所开启。
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # -------------------------
    # 行情敏感参数：建议参与 hyperopt。
    # -------------------------
    buy_donchian_period = IntParameter(24, 96, default=48, space="buy", optimize=True)
    buy_min_range_width = DecimalParameter(0.003, 0.020, default=0.006, decimals=3, space="buy")
    buy_max_range_width = DecimalParameter(0.030, 0.120, default=0.075, decimals=3, space="buy")
    buy_max_adx_for_range = DecimalParameter(16.0, 34.0, default=26.0, decimals=1, space="buy")
    buy_min_atr_pct = DecimalParameter(0.0005, 0.0060, default=0.0010, decimals=4, space="buy")
    buy_max_atr_pct = DecimalParameter(0.0150, 0.0900, default=0.0450, decimals=4, space="buy")
    buy_min_volume_ratio = DecimalParameter(0.40, 1.50, default=0.75, decimals=2, space="buy")
    buy_entry_buffer_mult = DecimalParameter(0.15, 0.80, default=0.35, decimals=2, space="buy")
    buy_entry_buffer_min = DecimalParameter(0.0005, 0.0040, default=0.0015, decimals=4, space="buy")
    buy_entry_buffer_max = DecimalParameter(0.0040, 0.0200, default=0.0090, decimals=4, space="buy")
    buy_long_rsi_min = IntParameter(45, 58, default=50, space="buy")
    buy_long_rsi_max = IntParameter(62, 82, default=72, space="buy")
    buy_short_rsi_min = IntParameter(18, 38, default=28, space="buy")
    buy_short_rsi_max = IntParameter(42, 55, default=50, space="buy")
    buy_max_extension_pct = DecimalParameter(0.020, 0.100, default=0.055, decimals=3, space="buy")
    buy_breakout_confirmation = CategoricalParameter(
        ["either_confirm", "close_confirm", "retest_confirm"],
        default="either_confirm",
        space="buy",
        optimize=True,
    )
    buy_min_breakout_body = DecimalParameter(0.000, 0.010, default=0.002, decimals=3, space="buy")
    buy_retest_tolerance = DecimalParameter(0.000, 0.008, default=0.003, decimals=3, space="buy")
    buy_max_breakout_overextension = DecimalParameter(
        0.010, 0.060, default=0.032, decimals=3, space="buy"
    )
    buy_min_trend_bias = DecimalParameter(-0.004, 0.006, default=0.000, decimals=3, space="buy")

    sell_exit_adx = DecimalParameter(26.0, 48.0, default=34.0, decimals=1, space="sell")
    sell_exit_adx_slope = DecimalParameter(1.0, 8.0, default=3.0, decimals=1, space="sell")
    sell_time_stop_candles = IntParameter(24, 288, default=96, space="sell")
    sell_time_stop_min_profit = DecimalParameter(-0.010, 0.012, default=0.002, decimals=3, space="sell")
    sell_long_exhaust_rsi = IntParameter(68, 86, default=76, space="sell")
    sell_short_exhaust_rsi = IntParameter(14, 32, default=24, space="sell")

    # -------------------------
    # 结构性/风控参数：默认保守，通常先固定，再小范围调优。
    # -------------------------
    risk_single_trade_risk_pct = DecimalParameter(
        0.003, 0.020, default=0.010, decimals=3, space="protection", optimize=True
    )
    risk_hard_stoploss = DecimalParameter(
        -0.060, -0.020, default=-0.035, decimals=3, space="protection", optimize=True
    )
    risk_max_custom_open_trades = IntParameter(1, 8, default=4, space="protection", optimize=True)
    risk_single_trade_notional_cap = DecimalParameter(
        20.0, 100000.0, default=2000.0, decimals=0, space="protection", optimize=True
    )
    risk_total_open_notional_cap = DecimalParameter(
        50.0, 300000.0, default=6000.0, decimals=0, space="protection", optimize=True
    )
    risk_max_strategy_leverage = DecimalParameter(
        1.0, 5.0, default=2.0, decimals=1, space="protection", optimize=True
    )
    risk_account_float_loss_reduce = DecimalParameter(
        0.020, 0.120, default=0.050, decimals=3, space="protection", optimize=True
    )
    risk_account_float_loss_exit = DecimalParameter(
        0.050, 0.250, default=0.100, decimals=3, space="protection", optimize=True
    )
    risk_floating_reduce_profit = DecimalParameter(
        -0.060, -0.012, default=-0.030, decimals=3, space="protection", optimize=True
    )
    risk_floating_reduce_fraction = DecimalParameter(
        0.15, 0.55, default=0.30, decimals=2, space="protection", optimize=True
    )
    risk_max_dca_entries = IntParameter(0, 1, default=0, space="protection", optimize=True)
    risk_dca_trigger_step = DecimalParameter(
        -0.060, -0.012, default=-0.025, decimals=3, space="protection", optimize=True
    )
    risk_dca_sizing_mode = CategoricalParameter(
        ["decreasing", "equal"], default="decreasing", space="protection", optimize=True
    )
    risk_min_candles_between_dca = IntParameter(3, 48, default=12, space="protection", optimize=True)
    risk_interrupt_loss_after_dca = DecimalParameter(
        -0.080, -0.035, default=-0.055, decimals=3, space="protection", optimize=True
    )
    risk_max_spread_bps = DecimalParameter(
        3.0, 120.0, default=35.0, decimals=1, space="protection", optimize=True
    )
    risk_max_funding_rate = DecimalParameter(
        0.0003, 0.0100, default=0.0025, decimals=4, space="protection", optimize=True
    )
    risk_extreme_candle_atr_mult = DecimalParameter(
        1.8, 8.0, default=4.0, decimals=1, space="protection", optimize=True
    )
    risk_extreme_candle_pct = DecimalParameter(
        0.025, 0.200, default=0.080, decimals=3, space="protection", optimize=True
    )
    risk_consecutive_loss_limit = IntParameter(
        2, 8, default=3, space="protection", optimize=True
    )
    risk_cooldown_candles_after_loss = IntParameter(
        24, 720, default=144, space="protection", optimize=True
    )
    risk_max_daily_drawdown = DecimalParameter(
        0.025, 0.200, default=0.070, decimals=3, space="protection", optimize=True
    )

    reduce_once_key = "gold_kylin_reduced_once"
    initial_stake_key = "gold_kylin_initial_stake"

    def bot_start(self, **kwargs: Any) -> None:
        # 用内部冷静期配合原生 protections；这里不依赖外部脚本，实盘和回测都能工作。
        self._cooldown_until: datetime | None = None

    @property
    def protections(self) -> list[dict]:
        # 原生保护负责跨交易的熔断；自定义 confirm_trade_entry 再做实时二次校验。
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": max(4, int(self.risk_cooldown_candles_after_loss.value / 4)),
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 288,
                "trade_limit": int(self.risk_consecutive_loss_limit.value),
                "stop_duration_candles": int(self.risk_cooldown_candles_after_loss.value),
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 864,
                "trade_limit": 10,
                "stop_duration_candles": int(self.risk_cooldown_candles_after_loss.value),
                "max_allowed_drawdown": float(self.risk_max_daily_drawdown.value),
            },
        ]

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        # Wilder 平滑只依赖历史 close，适合和多数交易软件的 RSI 对齐。
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    @staticmethod
    def _atr(dataframe: DataFrame, period: int) -> pd.Series:
        # 真实波幅使用上一根 close，避免把未来 K 线信息混入当前决策。
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

    @staticmethod
    def _adx(dataframe: DataFrame, period: int) -> pd.Series:
        # ADX 用来判断“震荡策略是否还在自己的能力圈里”；ADX 走高时减少参与。
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=dataframe.index).ewm(
            alpha=1 / period, adjust=False
        ).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=dataframe.index).ewm(
            alpha=1 / period, adjust=False
        ).mean() / atr.replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)

    def informative_pairs(self) -> list[tuple[str, str]]:
        # 核心系统只依赖本周期；资金费率通过 DataProvider 在入场确认阶段读取。
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        donchian = int(self.buy_donchian_period.value)
        atr_period = 14
        rsi_period = 14
        adx_period = 14

        # 均线用于衡量价格是否过度偏离震荡中心，不作为趋势追涨的唯一依据。
        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_mid"] = dataframe["close"].ewm(span=34, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=89, adjust=False).mean()

        dataframe["rsi"] = self._rsi(dataframe["close"], rsi_period)
        dataframe["atr"] = self._atr(dataframe, atr_period)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        dataframe["adx"] = self._adx(dataframe, adx_period)
        dataframe["adx_slope"] = dataframe["adx"].diff(3).fillna(0)

        # 必须 shift(1)：当前 K 线只能突破此前已知的区间高低点。
        min_periods = max(12, int(donchian * 0.45))
        dataframe["range_high"] = dataframe["high"].rolling(donchian, min_periods=min_periods).max().shift(1)
        dataframe["range_low"] = dataframe["low"].rolling(donchian, min_periods=min_periods).min().shift(1)
        dataframe["range_mid"] = (dataframe["range_high"] + dataframe["range_low"]) / 2
        dataframe["range_width"] = (
            (dataframe["range_high"] - dataframe["range_low"])
            / dataframe["range_mid"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        rolling_mean = dataframe["close"].rolling(40, min_periods=20).mean()
        rolling_std = dataframe["close"].rolling(40, min_periods=20).std(ddof=0)
        dataframe["bb_mid"] = rolling_mean
        dataframe["bb_upper"] = rolling_mean + rolling_std * 2
        dataframe["bb_lower"] = rolling_mean - rolling_std * 2
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        volume_mean = dataframe["volume"].rolling(48, min_periods=12).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / volume_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        # ATR 缓冲把 MT5 点数距离转成跨币种更稳定的相对距离。
        buffer_min = float(self.buy_entry_buffer_min.value)
        buffer_max = max(buffer_min, float(self.buy_entry_buffer_max.value))
        dataframe["entry_buffer"] = (
            dataframe["atr_pct"] * float(self.buy_entry_buffer_mult.value)
        ).clip(lower=buffer_min, upper=buffer_max)
        dataframe["long_trigger"] = dataframe["range_high"] * (1 + dataframe["entry_buffer"])
        dataframe["short_trigger"] = dataframe["range_low"] * (1 - dataframe["entry_buffer"])

        dataframe["extension_pct"] = (
            dataframe["close"] / dataframe["ema_slow"].replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        candle_range_pct = ((dataframe["high"] - dataframe["low"]) / dataframe["close"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        dataframe["extreme_candle"] = (
            (candle_range_pct > float(self.risk_extreme_candle_pct.value))
            | (
                (dataframe["atr"] > 0)
                & ((dataframe["high"] - dataframe["low"]) > dataframe["atr"] * float(self.risk_extreme_candle_atr_mult.value))
            )
        ).astype(int)

        dataframe["regime_changed"] = (
            (dataframe["adx"] > float(self.sell_exit_adx.value))
            & (dataframe["adx_slope"] > float(self.sell_exit_adx_slope.value))
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # 只在“可交易的震荡宽度”内入场：太窄手续费不友好，太宽往往已经是趋势或异常波动。
        range_environment = (
            dataframe["range_width"].between(
                float(self.buy_min_range_width.value), float(self.buy_max_range_width.value)
            )
            & dataframe["bb_width"].between(
                float(self.buy_min_range_width.value) * 0.8,
                float(self.buy_max_range_width.value) * 1.2,
            )
            & dataframe["atr_pct"].between(
                float(self.buy_min_atr_pct.value), float(self.buy_max_atr_pct.value)
            )
            & (dataframe["adx"] < float(self.buy_max_adx_for_range.value))
            & (dataframe["regime_changed"] == 0)
            & (dataframe["extreme_candle"] == 0)
        )

        liquidity_ok = (dataframe["volume"] > 0) & (
            dataframe["volume_ratio"] >= float(self.buy_min_volume_ratio.value)
        )
        not_accelerating_trend = dataframe["adx_slope"] < 4.0
        extension_cap = float(self.buy_max_extension_pct.value)

        # 多头不再“触价即追”。上一轮亏损主要来自假突破后很快跌回区间，
        # 所以这里要求两类确认之一：连续收盘站上触发价，或突破后回踩区间上沿仍能收回。
        body_pct = ((dataframe["close"] - dataframe["open"]) / dataframe["close"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        breakout_overextension = (
            dataframe["close"] / dataframe["range_high"].replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], np.nan).fillna(999)
        trend_bias = (
            dataframe["ema_mid"] / dataframe["ema_slow"].replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], np.nan).fillna(-999)

        close_confirm = (
            (dataframe["close"] > dataframe["long_trigger"])
            & (dataframe["close"].shift(1) > dataframe["long_trigger"].shift(1))
        )
        retest_confirm = (
            (dataframe["close"].shift(1) > dataframe["long_trigger"].shift(1))
            & (dataframe["low"] <= dataframe["range_high"] * (1 + float(self.buy_retest_tolerance.value)))
            & (dataframe["close"] > dataframe["range_high"] * (1 + dataframe["entry_buffer"] * 0.10))
            & (dataframe["close"] > dataframe["open"])
        )
        confirmation_mode = self.buy_breakout_confirmation.value
        if confirmation_mode == "close_confirm":
            breakout_confirmed = close_confirm
        elif confirmation_mode == "retest_confirm":
            breakout_confirmed = retest_confirm
        else:
            breakout_confirmed = close_confirm | retest_confirm

        long_breakout = (
            breakout_confirmed
            & (body_pct >= float(self.buy_min_breakout_body.value))
            & (breakout_overextension <= float(self.buy_max_breakout_overextension.value))
            # 只做中期结构已经翻正的突破，过滤下跌尾端的第一根脉冲假突破。
            & (trend_bias >= float(self.buy_min_trend_bias.value))
            & (dataframe["ema_fast"] >= dataframe["ema_mid"] * 0.997)
            & dataframe["rsi"].between(int(self.buy_long_rsi_min.value), int(self.buy_long_rsi_max.value))
            & (dataframe["extension_pct"] < extension_cap)
        )

        # 空头：跌破上一段区间低点 - ATR 缓冲；仅在合约模式使用，现货可关闭 can_short。
        short_breakout = (
            (dataframe["close"] < dataframe["short_trigger"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["ema_fast"] <= dataframe["ema_mid"] * 1.003)
            & dataframe["rsi"].between(int(self.buy_short_rsi_min.value), int(self.buy_short_rsi_max.value))
            & (dataframe["extension_pct"] > -extension_cap)
        )

        dataframe.loc[
            range_environment & liquidity_ok & not_accelerating_trend & long_breakout,
            ["enter_long", "enter_tag"],
        ] = (1, "range_buy_stop_breakout")

        if self.can_short:
            dataframe.loc[
                range_environment & liquidity_ok & not_accelerating_trend & short_breakout,
                ["enter_short", "enter_tag"],
            ] = (1, "range_sell_stop_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # 过热/过冷反向 K 线是小盈利落袋或撤退信号，避免把震荡突破当趋势单长期持有。
        long_exhausted = (
            (dataframe["rsi"] > int(self.sell_long_exhaust_rsi.value))
            & (dataframe["close"] < dataframe["close"].shift(1))
            & (dataframe["extension_pct"] > 0.040)
        )
        short_exhausted = (
            (dataframe["rsi"] < int(self.sell_short_exhaust_rsi.value))
            & (dataframe["close"] > dataframe["close"].shift(1))
            & (dataframe["extension_pct"] < -0.040)
        )

        # 出场标签拆细，回测复盘时才能知道到底是跌回区间、均线失效、过热回落还是行情环境改变。
        dataframe.loc[dataframe["close"] < dataframe["range_mid"], ["exit_long", "exit_tag"]] = (
            1,
            "long_range_mid_failed",
        )
        dataframe.loc[
            ((dataframe["close"] < dataframe["ema_mid"]) & (dataframe["rsi"] < 48)),
            ["exit_long", "exit_tag"],
        ] = (1, "long_ema_failed")
        dataframe.loc[long_exhausted, ["exit_long", "exit_tag"]] = (1, "long_exhausted")
        dataframe.loc[dataframe["regime_changed"] == 1, ["exit_long", "exit_tag"]] = (
            1,
            "long_regime_changed",
        )

        if self.can_short:
            dataframe.loc[dataframe["close"] > dataframe["range_mid"], ["exit_short", "exit_tag"]] = (
                1,
                "short_range_mid_failed",
            )
            dataframe.loc[
                ((dataframe["close"] > dataframe["ema_mid"]) & (dataframe["rsi"] > 52)),
                ["exit_short", "exit_tag"],
            ] = (1, "short_ema_failed")
            dataframe.loc[short_exhausted, ["exit_short", "exit_tag"]] = (1, "short_exhausted")
            dataframe.loc[dataframe["regime_changed"] == 1, ["exit_short", "exit_tag"]] = (
                1,
                "short_regime_changed",
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
        **kwargs: Any,
    ) -> float:
        # 百分比风险仓位法：风险预算 = 账户权益 * 单笔风险百分比，仓位 = 风险预算 / 计划止损距离。
        equity = self._account_equity()
        hard_stop = max(abs(float(self.risk_hard_stoploss.value)), 0.01)
        risk_budget = equity * float(self.risk_single_trade_risk_pct.value)
        risk_based_stake = risk_budget / hard_stop

        # 预留给后续加仓的额度要跟默认 DCA 开关同步。
        # 默认 max_dca=0 时不再压低首单仓位；只有显式允许加仓，才按等额/递减方式预留现金。
        max_dca = int(self.risk_max_dca_entries.value)
        if max_dca <= 0:
            reserved_for_dca = 1.0
        elif self.risk_dca_sizing_mode.value == "equal":
            reserved_for_dca = 1.0 + max_dca
        else:
            reserved_for_dca = 1.0 + sum([0.50, 0.35, 0.25][:max_dca])
        stake = min(
            proposed_stake,
            risk_based_stake / reserved_for_dca,
            float(self.risk_single_trade_notional_cap.value),
            max_stake,
        )
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
        # 1) 浮亏到中间档位先减仓：这是“活下来优先”，防止还没到硬止损就把风险堆大。
        if (
            current_profit <= float(self.risk_floating_reduce_profit.value)
            and not self._get_trade_flag(trade, self.reduce_once_key)
        ):
            self._set_trade_flag(trade, self.reduce_once_key, True)
            reduce_stake = -abs(float(getattr(trade, "stake_amount", 0.0)) * float(self.risk_floating_reduce_fraction.value))
            return reduce_stake, "floating_loss_reduce"

        entry_count = int(getattr(trade, "nr_of_successful_entries", 1))
        dca_done = max(0, entry_count - 1)
        max_dca = int(self.risk_max_dca_entries.value)

        # 2) 中断线附近禁止继续逆势加仓，交给 custom_exit 强制退出。
        if current_profit <= float(self.risk_interrupt_loss_after_dca.value):
            return None
        if dca_done >= max_dca:
            return None
        if self._candles_since_last_fill(trade, current_time) < int(self.risk_min_candles_between_dca.value):
            return None

        # 3) 亏损每深入一档才允许一次递减式安全单；绝不使用马丁放大。
        trigger = float(self.risk_dca_trigger_step.value) * entry_count
        if current_profit > trigger:
            return None

        last = self._last_candle(getattr(trade, "pair", ""))
        if last is None:
            return None
        if bool(last.get("extreme_candle", 0)) or bool(last.get("regime_changed", 0)):
            return None
        if not self._dca_structure_ok(trade, last):
            return None

        initial_stake = self._initial_stake(trade, entry_count)
        multiplier = self._dca_multiplier(dca_done)
        stake = initial_stake * multiplier
        stake = min(stake, max_stake, self._remaining_notional_capacity())
        if min_stake is not None and stake < min_stake:
            return None
        if stake <= 0:
            return None
        return stake, f"controlled_decreasing_dca_{dca_done + 1}"

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        # 账户级浮亏清仓线优先级最高：任何单触发时都尽快退出，避免组合层面继续恶化。
        if self._portfolio_float_loss_ratio(trade, current_profit) >= float(self.risk_account_float_loss_exit.value):
            return "account_float_loss_liquidation"

        entry_count = int(getattr(trade, "nr_of_successful_entries", 1))
        max_entries = int(self.risk_max_dca_entries.value) + 1

        # 加仓耗尽仍继续亏损，说明原震荡假设失败，强制中断，不再等待均值回归。
        if current_profit <= float(self.risk_interrupt_loss_after_dca.value):
            return "interrupt_after_failed_dca"
        if entry_count >= max_entries and current_profit <= float(self.risk_dca_trigger_step.value) * entry_count:
            return "dca_exhausted_no_reversal"

        last = self._last_candle(pair)
        if last is not None:
            # 极端插针后如果持仓还在亏损，优先撤退；盈利单交给 trailing/ROI 处理。
            if bool(last.get("extreme_candle", 0)) and current_profit < 0:
                return "extreme_candle_emergency_exit"
            if self._opposite_signal(trade, last):
                return "opposite_signal_exit"

        # 时间止损：持仓超过 N 根 K 线仍没有达到最低收益，说明资金效率太低或突破失败。
        if self._bars_since_open(trade, current_time) >= int(self.sell_time_stop_candles.value):
            if current_profit <= float(self.sell_time_stop_min_profit.value):
                return "time_stop_no_progress"
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        # custom_stoploss 只收紧止损；类级 stoploss 仍是最后硬闸。
        hard = float(self.risk_hard_stoploss.value)
        if current_profit > 0.040:
            return -0.010
        if current_profit > 0.025:
            return -0.014
        if current_profit > 0.012:
            return -0.022
        return hard

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> bool:
        # 入场前置风控：任何一项失败都不下单，避免订单进入交易所后再被动处理。
        self._refresh_cooldown(current_time)
        if self._cooldown_active(current_time):
            return False
        if self._open_trade_count() >= int(self.risk_max_custom_open_trades.value):
            return False
        notional = amount * rate
        if notional > float(self.risk_single_trade_notional_cap.value) * float(self.risk_max_strategy_leverage.value):
            return False
        if self._open_notional_estimate() + notional > float(self.risk_total_open_notional_cap.value):
            return False
        if not self._spread_ok(pair):
            return False
        if not self._funding_ok(pair, side):
            return False
        last = self._last_candle(pair)
        if last is not None and bool(last.get("extreme_candle", 0)):
            return False
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs: Any,
    ) -> bool:
        # 风控退出不拦截；普通 ROI/信号退出也不做二次阻拦，避免因为本地判断错误导致无法平仓。
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
        # 这里硬限制策略可用杠杆，防止 config 或交易所默认值意外给到高杠杆。
        desired = min(float(self.risk_max_strategy_leverage.value), max_leverage)
        return max(1.0, desired)

    def _account_equity(self) -> float:
        wallets = getattr(self, "wallets", None)
        if wallets is not None:
            try:
                total = float(wallets.get_total_stake_amount())
                if total > 0:
                    return total
            except Exception:
                pass
        try:
            return float(self.config.get("dry_run_wallet", 1000.0))
        except Exception:
            return 1000.0

    def _last_candle(self, pair: str) -> pd.Series | None:
        dp = getattr(self, "dp", None)
        if dp is None or not pair:
            return None
        try:
            dataframe, _ = dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None
            return dataframe.iloc[-1]
        except Exception:
            return None

    def _bars_since_open(self, trade: Trade, current_time: datetime) -> int:
        minutes = max(timeframe_to_minutes(self.timeframe), 1)
        open_date = getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", current_time)
        if open_date.tzinfo is None:
            open_date = open_date.replace(tzinfo=UTC)
        now = current_time if current_time.tzinfo is not None else current_time.replace(tzinfo=UTC)
        return max(0, int((now - open_date).total_seconds() // (minutes * 60)))

    def _candles_since_last_fill(self, trade: Trade, current_time: datetime) -> int:
        minutes = max(timeframe_to_minutes(self.timeframe), 1)
        last_fill = getattr(trade, "date_last_filled_utc", None)
        if callable(last_fill):
            last_fill = last_fill()
        if last_fill is None:
            last_fill = getattr(trade, "open_date_utc", current_time)
        if last_fill.tzinfo is None:
            last_fill = last_fill.replace(tzinfo=UTC)
        now = current_time if current_time.tzinfo is not None else current_time.replace(tzinfo=UTC)
        return max(0, int((now - last_fill).total_seconds() // (minutes * 60)))

    def _initial_stake(self, trade: Trade, entry_count: int) -> float:
        stored = self._get_trade_value(trade, self.initial_stake_key)
        if stored:
            return float(stored)
        initial = float(getattr(trade, "stake_amount", 0.0)) / max(entry_count, 1)
        self._set_trade_value(trade, self.initial_stake_key, initial)
        return initial

    def _dca_multiplier(self, dca_done: int) -> float:
        if self.risk_dca_sizing_mode.value == "equal":
            return 0.35
        decreasing = [0.50, 0.35, 0.25]
        return decreasing[min(dca_done, len(decreasing) - 1)]

    def _dca_structure_ok(self, trade: Trade, last: pd.Series) -> bool:
        # 加仓不是“越跌越买”，而是亏损后仍未破坏结构才允许补一笔小仓。
        if bool(getattr(trade, "is_short", False)):
            return (
                float(last.get("close", 0.0)) < float(last.get("ema_mid", np.inf)) * 1.01
                and float(last.get("rsi", 50.0)) < 58.0
            )
        return (
            float(last.get("close", 0.0)) > float(last.get("ema_mid", 0.0)) * 0.99
            and float(last.get("rsi", 50.0)) > 42.0
        )

    def _opposite_signal(self, trade: Trade, last: pd.Series) -> bool:
        if bool(getattr(trade, "is_short", False)):
            return int(last.get("enter_long", 0)) == 1
        return int(last.get("enter_short", 0)) == 1

    def _portfolio_float_loss_ratio(self, current_trade: Trade, current_profit: float) -> float:
        equity = max(self._account_equity(), 1.0)
        loss_abs = 0.0
        for trade in self._open_trades():
            try:
                if getattr(trade, "id", None) == getattr(current_trade, "id", None):
                    profit = current_profit
                else:
                    last = self._last_candle(getattr(trade, "pair", ""))
                    if last is None:
                        continue
                    profit = float(trade.calc_profit_ratio(float(last["close"])))
                stake = float(getattr(trade, "stake_amount", 0.0))
                loss_abs += max(0.0, -profit * stake)
            except Exception:
                continue
        return loss_abs / equity

    def _open_trades(self) -> list[Trade]:
        try:
            return list(Trade.get_trades_proxy(is_open=True))
        except Exception:
            return []

    def _open_trade_count(self) -> int:
        return len(self._open_trades())

    def _open_notional_estimate(self) -> float:
        total = 0.0
        for trade in self._open_trades():
            stake = float(getattr(trade, "stake_amount", 0.0))
            leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
            total += stake * leverage
        return total

    def _remaining_notional_capacity(self) -> float:
        return max(0.0, float(self.risk_total_open_notional_cap.value) - self._open_notional_estimate())

    def _spread_ok(self, pair: str) -> bool:
        dp = getattr(self, "dp", None)
        runmode = getattr(getattr(dp, "runmode", None), "value", None)
        if dp is None or runmode not in {"live", "dry_run"}:
            return True
        try:
            orderbook = dp.orderbook(pair, 1)
            bid = float(orderbook["bids"][0][0])
            ask = float(orderbook["asks"][0][0])
            mid = (bid + ask) / 2
            if mid <= 0:
                return False
            spread_bps = (ask - bid) / mid * 10000
            return spread_bps <= float(self.risk_max_spread_bps.value)
        except Exception:
            # 实盘读不到盘口时禁止新开仓；这是数据异常容错，避免盲下单。
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
            funding_data = funding_rate(pair)
            if isinstance(funding_data, dict):
                funding = float(funding_data.get("fundingRate", 0.0) or 0.0)
            else:
                funding = float(funding_data)
        except Exception:
            # 资金费率读取失败时不断言风险，交给交易所 funding fee 结算和其它风控。
            return True
        limit = float(self.risk_max_funding_rate.value)
        if side == "long" and funding > limit:
            return False
        if side == "short" and funding < -limit:
            return False
        return True

    def _refresh_cooldown(self, current_time: datetime) -> None:
        if self._daily_drawdown_hit(current_time) or self._consecutive_losses_hit():
            minutes = max(timeframe_to_minutes(self.timeframe), 1)
            self._cooldown_until = current_time + timedelta(
                minutes=minutes * int(self.risk_cooldown_candles_after_loss.value)
            )

    def _cooldown_active(self, current_time: datetime) -> bool:
        return self._cooldown_until is not None and current_time < self._cooldown_until

    def _consecutive_losses_hit(self) -> bool:
        closed = self._closed_trades_sorted()
        if not closed:
            return False
        losses = 0
        for trade in closed:
            if float(getattr(trade, "close_profit", 0.0) or 0.0) < 0:
                losses += 1
            else:
                break
        return losses >= int(self.risk_consecutive_loss_limit.value)

    def _daily_drawdown_hit(self, current_time: datetime) -> bool:
        equity = max(self._account_equity(), 1.0)
        now = current_time if current_time.tzinfo is not None else current_time.replace(tzinfo=UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        pnl = 0.0
        for trade in self._closed_trades_sorted():
            close_date = getattr(trade, "close_date_utc", None) or getattr(trade, "close_date", None)
            if close_date is None:
                continue
            if close_date.tzinfo is None:
                close_date = close_date.replace(tzinfo=day_start.tzinfo)
            if close_date >= day_start:
                pnl += float(getattr(trade, "close_profit_abs", 0.0) or 0.0)
        return pnl / equity <= -float(self.risk_max_daily_drawdown.value)

    def _closed_trades_sorted(self) -> list[Trade]:
        try:
            trades = list(Trade.get_trades_proxy(is_open=False))
        except Exception:
            return []
        return sorted(
            trades,
            key=self._trade_close_sort_key,
            reverse=True,
        )

    @staticmethod
    def _trade_close_sort_key(trade: Trade) -> datetime:
        close_date = getattr(trade, "close_date_utc", None) or getattr(trade, "close_date", None)
        if close_date is None:
            return datetime.min.replace(tzinfo=UTC)
        if close_date.tzinfo is None:
            return close_date.replace(tzinfo=UTC)
        return close_date

    def _get_trade_flag(self, trade: Trade, key: str) -> bool:
        return bool(self._get_trade_value(trade, key, False))

    def _get_trade_value(self, trade: Trade, key: str, default: Any = None) -> Any:
        getter = getattr(trade, "get_custom_data", None)
        if getter is None:
            return getattr(trade, key, default)
        try:
            return getter(key, default)
        except Exception:
            return getattr(trade, key, default)

    def _set_trade_flag(self, trade: Trade, key: str, value: bool) -> None:
        self._set_trade_value(trade, key, value)

    def _set_trade_value(self, trade: Trade, key: str, value: Any) -> None:
        setter = getattr(trade, "set_custom_data", None)
        if setter is not None:
            try:
                setter(key, value)
                return
            except Exception:
                pass
        try:
            setattr(trade, key, value)
        except Exception:
            pass


class GoldKylinRangeBreakoutLongOnlyStrategy(GoldKylinRangeBreakoutStrategy):
    """
    现货/低杠杆验证专用子类。

    主策略已经默认 long-only，保留这个类名是为了兼容上一轮回测配置和已有命令。
    """

    can_short = False


class GoldKylinRangeBreakoutBiDirectionalExperimentalStrategy(GoldKylinRangeBreakoutStrategy):
    """
    合约双向实验子类。

    上轮样本中 short 分支明显拖累收益，因此只把它作为研究入口保留；
    未经过单独样本外验证前，不建议实盘开启。
    """

    can_short = True
