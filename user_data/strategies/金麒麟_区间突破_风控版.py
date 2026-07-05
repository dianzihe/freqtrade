# -*- coding: utf-8 -*-
"""
金麒麟区间突破 · BTC/ETH 重构版

保留原 MT5 EA“区间两端触发”的核心思路,但针对币圈主流对(BTC/USDT、ETH/USDT)重写:
- 纯多头(原做空分支样本内持续亏损,已移除)
- 去掉 DCA / 马丁,风险交给硬止损 + 移动止损 + 组合浮亏清仓
- 参数大幅精简,风控参数固定,只对少量行情敏感参数做 hyperopt
- 内置滑点模型、标记价风控、成本对齐后的 ROI

config 侧建议:
  pair_whitelist: ["BTC/USDT", "ETH/USDT"]  (合约则用 :USDT)
  fee 按目标交易所真实值填写
  现货: trading_mode=spot
  合约: trading_mode=futures, margin_mode=isolated (资金费才会被引擎真实结算)
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class GoldKylinBtcEthBreakoutStrategy(IStrategy):
    """BTC/ETH 区间突破风控版(纯多头)。"""

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "15m"                    # 从 5m 上移:降低手续费/滑点磨损,主流对突破更有意义
    startup_candle_count = 220
    process_only_new_candles = True

    max_open_trades = 2                  # 只交易两个标的
    position_adjustment_enable = False   # 不再加仓/DCA,逻辑更干净、更抗过拟合

    # ROI 已按“覆盖 taker 费 + 滑点后仍为正”的思路抬高
    minimal_roi = {
        "0": 0.030,
        "120": 0.018,
        "480": 0.008,
        "960": 0.0,
    }

    # 类级 stoploss 是最终本地硬闸;custom_stoploss 只会收紧
    stoploss = -0.05
    use_custom_stoploss = True

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.020
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",                 # 尽量吃 maker,减少出场滑点(配 unfilledtimeout 兜底)
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,    # 合约实盘强烈建议改 True,防断线
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ---------------- 行情敏感参数:参与 hyperopt(约 10 个) ----------------
    buy_donchian_period = IntParameter(24, 96, default=48, space="buy", optimize=True)
    buy_min_range_width = DecimalParameter(0.004, 0.020, default=0.008, decimals=3, space="buy")
    buy_max_range_width = DecimalParameter(0.030, 0.100, default=0.060, decimals=3, space="buy")
    buy_max_adx_for_range = DecimalParameter(18.0, 32.0, default=26.0, decimals=1, space="buy")
    buy_entry_buffer_mult = DecimalParameter(0.15, 0.80, default=0.35, decimals=2, space="buy")
    buy_rsi_min = IntParameter(45, 58, default=52, space="buy")
    buy_rsi_max = IntParameter(62, 80, default=72, space="buy")
    buy_max_extension_pct = DecimalParameter(0.020, 0.080, default=0.045, decimals=3, space="buy")
    buy_breakout_confirmation = CategoricalParameter(
        ["either_confirm", "close_confirm", "retest_confirm"],
        default="either_confirm", space="buy", optimize=True,
    )
    sell_time_stop_candles = IntParameter(24, 192, default=96, space="sell", optimize=True)
    sell_exit_adx = DecimalParameter(28.0, 46.0, default=36.0, decimals=1, space="sell", optimize=True)

    # ---------------- 风控参数:全部固定,不参与 hyperopt ----------------
    risk_single_trade_risk_pct = 0.010          # 单笔风险预算 = 权益 * 1%
    risk_hard_stoploss = 0.040                   # 计划止损距离(绝对值),用于仓位换算
    risk_single_trade_notional_cap = 0.40        # 单笔名义价值上限 = 权益 * 40%
    risk_total_open_notional_cap = 0.80          # 组合名义价值上限 = 权益 * 80%
    risk_max_strategy_leverage = 1.0             # 现货=1;合约子类覆盖
    risk_account_float_loss_exit = 0.10          # 组合浮亏达权益 10% → 全部清仓
    risk_max_spread_bps = 8.0                    # 主流对盘口很窄,8bp 已很宽松
    risk_max_funding_rate = 0.0010               # 仅合约生效
    risk_extreme_candle_atr_mult = 4.0
    risk_extreme_candle_pct = 0.060
    risk_consecutive_loss_limit = 3
    risk_cooldown_candles_after_loss = 96
    risk_max_daily_drawdown = 0.06

    # 滑点模型:仅回测/hyperopt 注入;实盘=0(由真实盘口决定)
    # 现实基准 0.1%;压力测试可临时改 0.005 (0.5%)
    backtest_slippage_pct = 0.0010

    def bot_start(self, **kwargs: Any) -> None:
        self._cooldown_until: datetime | None = None

    @property
    def protections(self) -> list[dict]:
        return [
            {"method": "CooldownPeriod",
             "stop_duration_candles": max(4, int(self.risk_cooldown_candles_after_loss / 4))},
            {"method": "StoplossGuard", "lookback_period_candles": 288,
             "trade_limit": self.risk_consecutive_loss_limit,
             "stop_duration_candles": self.risk_cooldown_candles_after_loss,
             "only_per_pair": False},
            {"method": "MaxDrawdown", "lookback_period_candles": 864, "trade_limit": 8,
             "stop_duration_candles": self.risk_cooldown_candles_after_loss,
             "max_allowed_drawdown": self.risk_max_daily_drawdown},
        ]

    # ---------------- 指标 ----------------
    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    @staticmethod
    def _atr(df: DataFrame, period: int) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _adx(df: DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low, (high - prev_close).abs(), (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        donchian = int(self.buy_donchian_period.value)

        dataframe["ema_fast"] = dataframe["close"].ewm(span=12, adjust=False).mean()
        dataframe["ema_mid"] = dataframe["close"].ewm(span=34, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=89, adjust=False).mean()

        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["atr"] = self._atr(dataframe, 14)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        dataframe["adx"] = self._adx(dataframe, 14)
        dataframe["adx_slope"] = dataframe["adx"].diff(3).fillna(0)

        # shift(1):当前 K 线只能突破此前已知的区间高低点(避免 look-ahead)
        min_p = max(12, int(donchian * 0.45))
        dataframe["range_high"] = dataframe["high"].rolling(donchian, min_periods=min_p).max().shift(1)
        dataframe["range_low"] = dataframe["low"].rolling(donchian, min_periods=min_p).min().shift(1)
        dataframe["range_mid"] = (dataframe["range_high"] + dataframe["range_low"]) / 2
        dataframe["range_width"] = (
            (dataframe["range_high"] - dataframe["range_low"]) / dataframe["range_mid"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        vol_mean = dataframe["volume"].rolling(48, min_periods=12).mean()
        dataframe["volume_ratio"] = (
            dataframe["volume"] / vol_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        buffer = (dataframe["atr_pct"] * float(self.buy_entry_buffer_mult.value)).clip(0.0008, 0.010)
        dataframe["entry_buffer"] = buffer
        dataframe["long_trigger"] = dataframe["range_high"] * (1 + buffer)

        dataframe["extension_pct"] = (
            dataframe["close"] / dataframe["ema_slow"].replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        rng_pct = ((dataframe["high"] - dataframe["low"]) / dataframe["close"]).replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        dataframe["extreme_candle"] = (
            (rng_pct > self.risk_extreme_candle_pct)
            | ((dataframe["atr"] > 0)
               & ((dataframe["high"] - dataframe["low"]) > dataframe["atr"] * self.risk_extreme_candle_atr_mult))
        ).astype(int)

        dataframe["regime_changed"] = (
            (dataframe["adx"] > float(self.sell_exit_adx.value)) & (dataframe["adx_slope"] > 3.0)
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        env = (
            dataframe["range_width"].between(
                float(self.buy_min_range_width.value), float(self.buy_max_range_width.value))
            & (dataframe["adx"] < float(self.buy_max_adx_for_range.value))
            & (dataframe["regime_changed"] == 0)
            & (dataframe["extreme_candle"] == 0)
            & (dataframe["volume_ratio"] >= 0.75)
            & (dataframe["volume"] > 0)
        )

        body_pct = ((dataframe["close"] - dataframe["open"]) / dataframe["close"]).replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        overext = (dataframe["close"] / dataframe["range_high"].replace(0, np.nan) - 1).replace(
            [np.inf, -np.inf], np.nan).fillna(999)
        trend_bias = (dataframe["ema_mid"] / dataframe["ema_slow"].replace(0, np.nan) - 1).replace(
            [np.inf, -np.inf], np.nan).fillna(-999)

        # 收盘确认:连续两根收盘站上触发价
        close_confirm = (
            (dataframe["close"] > dataframe["long_trigger"])
            & (dataframe["close"].shift(1) > dataframe["long_trigger"].shift(1))
        )
        # 回踩确认:突破后回踩区间上沿但仍能收回
        retest_confirm = (
            (dataframe["close"].shift(1) > dataframe["long_trigger"].shift(1))
            & (dataframe["low"] <= dataframe["range_high"] * 1.003)
            & (dataframe["close"] > dataframe["range_high"] * (1 + dataframe["entry_buffer"] * 0.10))
            & (dataframe["close"] > dataframe["open"])
        )
        mode = self.buy_breakout_confirmation.value
        confirmed = {"close_confirm": close_confirm,
                     "retest_confirm": retest_confirm}.get(mode, close_confirm | retest_confirm)

        long_ok = (
            confirmed
            & (body_pct >= 0.002)
            & (overext <= 0.032)                                  # 不追已经拉飞的突破
            & (trend_bias >= 0.0)                                 # 中期结构翻正
            & (dataframe["ema_fast"] >= dataframe["ema_mid"] * 0.997)
            & dataframe["rsi"].between(int(self.buy_rsi_min.value), int(self.buy_rsi_max.value))
            & (dataframe["extension_pct"] < float(self.buy_max_extension_pct.value))
        )

        dataframe.loc[env & long_ok, ["enter_long", "enter_tag"]] = (1, "range_breakout_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        long_exhausted = (
            (dataframe["rsi"] > 76) & (dataframe["close"] < dataframe["close"].shift(1))
            & (dataframe["extension_pct"] > 0.040)
        )
        dataframe.loc[dataframe["close"] < dataframe["range_mid"],
                      ["exit_long", "exit_tag"]] = (1, "range_mid_failed")
        dataframe.loc[(dataframe["close"] < dataframe["ema_mid"]) & (dataframe["rsi"] < 48),
                      ["exit_long", "exit_tag"]] = (1, "ema_failed")
        dataframe.loc[long_exhausted, ["exit_long", "exit_tag"]] = (1, "exhausted")
        dataframe.loc[dataframe["regime_changed"] == 1,
                      ["exit_long", "exit_tag"]] = (1, "regime_changed")
        return dataframe

    # ---------------- 仓位 ----------------
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        equity = self._account_equity()
        hard_stop = max(self.risk_hard_stoploss, 0.01)
        # 名义价值 = 风险预算 / 止损距离;stake(保证金) = 名义 / 杠杆
        notional = (equity * self.risk_single_trade_risk_pct) / hard_stop
        stake = notional / max(leverage, 1.0)
        stake = min(stake, equity * self.risk_single_trade_notional_cap / max(leverage, 1.0), max_stake)
        if min_stake is not None:
            stake = max(stake, min_stake)
        return max(stake, 0.0)

    # ---------------- 滑点模型(仅回测/hyperopt) ----------------
    def _slippage(self) -> float:
        rm = getattr(getattr(self, "dp", None), "runmode", None)
        if getattr(rm, "value", None) in {"backtest", "hyperopt"}:
            return self.backtest_slippage_pct
        return 0.0

    def custom_entry_price(self, pair, trade, current_time, proposed_rate,
                           entry_tag, side, **kwargs) -> float:
        # 开多按对自己不利的方向(买得更贵);当 K 线未覆盖到该价时引擎判不成交 → 天然模拟“跳空落空”
        return proposed_rate * (1 + self._slippage())

    def custom_exit_price(self, pair, trade, current_time, proposed_rate,
                          current_profit, exit_tag, **kwargs) -> float:
        return proposed_rate * (1 - self._slippage())   # 平多卖得更差

    # ---------------- 出场 ----------------
    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # 组合浮亏清仓线优先级最高
        if self._portfolio_float_loss_ratio(trade, current_profit) >= self.risk_account_float_loss_exit:
            return "account_float_loss_liquidation"
        last = self._last_candle(pair)
        if last is not None and bool(last.get("extreme_candle", 0)) and current_profit < 0:
            return "extreme_candle_emergency_exit"
        if self._bars_since_open(trade, current_time) >= int(self.sell_time_stop_candles.value):
            if current_profit <= 0.002:
                return "time_stop_no_progress"
        return None

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        if current_profit > 0.05:
            return -0.012
        if current_profit > 0.03:
            return -0.018
        if current_profit > 0.015:
            return -0.025
        return -self.risk_hard_stoploss

    # ---------------- 入场前置风控 ----------------
    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        self._refresh_cooldown(current_time)
        if self._cooldown_active(current_time):
            return False
        if self._open_trade_count() >= self.max_open_trades:
            return False
        equity = self._account_equity()
        notional = amount * rate
        if notional > equity * self.risk_single_trade_notional_cap:
            return False
        if self._open_notional_estimate() + notional > equity * self.risk_total_open_notional_cap:
            return False
        if not self._spread_ok(pair):
            return False
        if not self._funding_ok(pair, side):
            return False
        last = self._last_candle(pair)
        if last is not None and bool(last.get("extreme_candle", 0)):
            return False
        return True

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return max(1.0, min(self.risk_max_strategy_leverage, max_leverage))

    # ---------------- 辅助 ----------------
    def _account_equity(self) -> float:
        wallets = getattr(self, "wallets", None)
        if wallets is not None:
            try:
                total = float(wallets.get_total_stake_amount())
                if total > 0:
                    return total
            except Exception as e:
                logger.warning("读取钱包权益失败,回退 dry_run_wallet: %s", e)
        try:
            return float(self.config.get("dry_run_wallet", 1000.0))
        except Exception:
            return 1000.0

    def _mark_or_close(self, pair: str, fallback: float) -> float:
        # 合约模式优先用标记价做风控线;现货/取不到则回退 close
        dp = getattr(self, "dp", None)
        if dp is None or self.config.get("trading_mode") != "futures":
            return fallback
        try:
            t = dp.ticker(pair)
            mark = t.get("mark") or t.get("markPrice")
            return float(mark) if mark else fallback
        except Exception:
            return fallback

    def _last_candle(self, pair: str):
        dp = getattr(self, "dp", None)
        if dp is None or not pair:
            return None
        try:
            df, _ = dp.get_analyzed_dataframe(pair, self.timeframe)
            return None if df is None or df.empty else df.iloc[-1]
        except Exception as e:
            logger.warning("读取 %s 分析数据失败: %s", pair, e)
            return None

    def _bars_since_open(self, trade, current_time) -> int:
        minutes = max(timeframe_to_minutes(self.timeframe), 1)
        od = getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", current_time)
        if od.tzinfo is None:
            od = od.replace(tzinfo=UTC)
        now = current_time if current_time.tzinfo else current_time.replace(tzinfo=UTC)
        return max(0, int((now - od).total_seconds() // (minutes * 60)))

    def _portfolio_float_loss_ratio(self, current_trade, current_profit) -> float:
        equity = max(self._account_equity(), 1.0)
        loss = 0.0
        for trade in self._open_trades():
            try:
                if getattr(trade, "id", None) == getattr(current_trade, "id", None):
                    profit = current_profit
                else:
                    last = self._last_candle(getattr(trade, "pair", ""))
                    if last is None:
                        continue
                    price = self._mark_or_close(trade.pair, float(last["close"]))
                    profit = float(trade.calc_profit_ratio(price))
                loss += max(0.0, -profit * float(getattr(trade, "stake_amount", 0.0)))
            except Exception as e:
                logger.warning("组合浮亏计算跳过一笔: %s", e)
        return loss / equity

    def _open_trades(self):
        try:
            return list(Trade.get_trades_proxy(is_open=True))
        except Exception:
            return []

    def _open_trade_count(self) -> int:
        return len(self._open_trades())

    def _open_notional_estimate(self) -> float:
        return sum(float(getattr(t, "stake_amount", 0.0)) * float(getattr(t, "leverage", 1.0) or 1.0)
                   for t in self._open_trades())

    def _spread_ok(self, pair: str) -> bool:
        dp = getattr(self, "dp", None)
        rm = getattr(getattr(dp, "runmode", None), "value", None)
        if dp is None or rm not in {"live", "dry_run"}:
            return True
        try:
            ob = dp.orderbook(pair, 1)
            bid, ask = float(ob["bids"][0][0]), float(ob["asks"][0][0])
            mid = (bid + ask) / 2
            if mid <= 0:
                return False
            return (ask - bid) / mid * 10000 <= self.risk_max_spread_bps
        except Exception as e:
            logger.warning("读取 %s 盘口失败,保守禁止开仓: %s", pair, e)
            return False   # 失败偏保守

    def _funding_ok(self, pair: str, side: str) -> bool:
        dp = getattr(self, "dp", None)
        rm = getattr(getattr(dp, "runmode", None), "value", None)
        if dp is None or rm not in {"live", "dry_run"} or self.config.get("trading_mode") != "futures":
            return True
        fr = getattr(dp, "funding_rate", None)
        if fr is None:
            return True
        try:
            data = fr(pair)
            funding = float(data.get("fundingRate", 0.0) or 0.0) if isinstance(data, dict) else float(data)
        except Exception as e:
            logger.warning("读取 %s 资金费失败,保守禁止开多: %s", pair, e)
            return False   # 失败偏保守(与盘口一致)
        return not (side == "long" and funding > self.risk_max_funding_rate)

    def _refresh_cooldown(self, current_time) -> None:
        if self._daily_drawdown_hit(current_time) or self._consecutive_losses_hit():
            minutes = max(timeframe_to_minutes(self.timeframe), 1)
            self._cooldown_until = current_time + timedelta(
                minutes=minutes * self.risk_cooldown_candles_after_loss)

    def _cooldown_active(self, current_time) -> bool:
        return self._cooldown_until is not None and current_time < self._cooldown_until

    def _closed_sorted(self):
        try:
            trades = list(Trade.get_trades_proxy(is_open=False))
        except Exception:
            return []
        return sorted(trades, key=self._close_key, reverse=True)

    @staticmethod
    def _close_key(trade):
        cd = getattr(trade, "close_date_utc", None) or getattr(trade, "close_date", None)
        if cd is None:
            return datetime.min.replace(tzinfo=UTC)
        return cd if cd.tzinfo else cd.replace(tzinfo=UTC)

    def _consecutive_losses_hit(self) -> bool:
        losses = 0
        for t in self._closed_sorted():
            if float(getattr(t, "close_profit", 0.0) or 0.0) < 0:
                losses += 1
            else:
                break
        return losses >= self.risk_consecutive_loss_limit

    def _daily_drawdown_hit(self, current_time) -> bool:
        equity = max(self._account_equity(), 1.0)
        now = current_time if current_time.tzinfo else current_time.replace(tzinfo=UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        pnl = 0.0
        for t in self._closed_sorted():
            cd = getattr(t, "close_date_utc", None) or getattr(t, "close_date", None)
            if cd is None:
                continue
            if cd.tzinfo is None:
                cd = cd.replace(tzinfo=day_start.tzinfo)
            if cd >= day_start:
                pnl += float(getattr(t, "close_profit_abs", 0.0) or 0.0)
        return pnl / equity <= -self.risk_max_daily_drawdown


class GoldKylinBtcEthFuturesStrategy(GoldKylinBtcEthBreakoutStrategy):
    """
    合约低杠杆子类(需要单独样本外验证后再实盘)。
    config 必须: trading_mode=futures, margin_mode=isolated,资金费才会被引擎真实结算。
    """
    can_short = False
    risk_max_strategy_leverage = 2.0
    stoploss_on_exchange = True   # 合约实盘建议交易所侧止损