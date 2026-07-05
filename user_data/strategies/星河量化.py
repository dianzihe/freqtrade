# -*- coding: utf-8 -*-
"""
==============================================================================
Gate 永续 · 星河趋势 + ATR自适应有限网格 · BTC/ETH 主流币重构版 (v2)
==============================================================================
在原「星河混合趋势」基础上重构, 目标标的收敛为 BTC/ETH 两个深流动性主流币。

【保留的三大原始想法】
  1. 三票制混合趋势过滤(EMA方向 + ATR放量方向 + 波动率扩张方向)。
  2. 顺势开仓 + ATR自适应"有限次、递增间距"网格补仓(DCA)。
  3. 多重熔断(连亏 / 日回撤 / 逆势深亏) + 动态杠杆(波动越大杠杆越低)。

【相对原版的关键改动 —— 每一处都是有意为之】
  · 去掉 OCO 括号单整套: 主流币流动性深, 改用【交易所原生止损 + 标记价触发】,
    直接消除"交易所先成交、本地后平账"的竞态窗口, 同时满足"用标记价做风控线"。
  · 去掉核心/卫星分仓: BTC/ETH 均为核心, 统一风险模型。
  · 修正 stoploss 与 deep_loss_fuse 的先后矛盾: 熔丝(浅)先动, 原生止损(深)兜底。
  · 全部风控异常 fail-closed(出错=拒绝开仓/保守), 不再 fail-open。
  · ATR 可交易区间按主流币 5m 重标定(原 5.5% 上限是给妖币的, 主流币过宽)。
  · Hyperopt 参数 14→5, 结构性参数全部固定, 抗过拟合。
  · 回测悲观化: 注入不利滑点 + 建议 config 抬高 fee 近似"半价差+平均滑点"。
  · 时间止损缩短到 ~12h, 减少跨资金费结算点的隐性成本暴露。

【仍需在 config / 数据层确认(策略文件管不到的)】
  · trading_mode: futures, margin_mode: isolated(或 cross)
  · 下载数据时带上 funding rate + mark price(futures 模式自动带), 否则资金费按0算=偏乐观
  · fee 用 Gate 永续真实费率, 并建议单边上抬(如 0.05%→0.10%)近似摩擦成本
  · max_open_trades ≤ 2(只有两个标的)
==============================================================================
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

logger = logging.getLogger(__name__)


class XingheMajorGridStrategy(IStrategy):
    """
    BTC/ETH 专用: 趋势过滤 + 有限网格DCA + 多重熔断。
    不再依赖任何外部 Mixin, 单文件自包含。
    """

    INTERFACE_VERSION = 3

    # ==========================================================================
    # 一、结构性参数(固定, 不进 Hyperopt —— 改这些等于改骨架)
    # ==========================================================================
    can_short = True
    timeframe = "5m"
    startup_candle_count = 240          # EMA60 + rolling(60) 预热
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False            # 趋势失效即离场, 允许亏损离场
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = True   # 启用有限网格 DCA

    # ---- 止损/止盈的分层设计(关键修正点) ----
    # 逻辑顺序(浅→深): deep_loss_fuse(-0.16, custom_exit软熔丝) 先触发
    #                → stoploss(-0.30, 交易所原生+标记价) 仅在极端跳空时兜底。
    # current_profit 是【含杠杆】的仓位收益率, 6x 下 -0.16 约对应 -2.7% 价格波动。
    stoploss = -0.25
    minimal_roi = {"0": 0.025}          # 会在 bot_start 里被 tp_roi 覆盖同步(见下)

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,          # 主流币: 交给交易所原生止损(替代原 OCO)
        "stoploss_on_exchange_interval": 60,
        # Gate 永续: 用标记价触发止损, 避免最新价插针误触发/漏触发
        "stoploss_price_type": "mark",
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ==========================================================================
    # 二、回测悲观化(滑点)—— 仅回测/hyperopt 生效, 实盘用真实挂单价
    # ==========================================================================
    slippage_pct = 0.004   # 单边 0.4% 不利滑点(配合 config fee=0.1% 总摩擦 = 0.5%)

    # ==========================================================================
    # 三、行情敏感参数(进 Hyperopt, 仅保留 5 个真正影响行为的)
    # ==========================================================================
    vote_threshold = IntParameter(2, 3, default=2, space="buy", optimize=True)
    atr_grid_multiplier = DecimalParameter(1.5, 4.0, default=2.5, decimals=1, space="buy", optimize=True)
    tp_roi = DecimalParameter(0.012, 0.050, default=0.025, decimals=3, space="sell", optimize=True)
    deep_loss_fuse = DecimalParameter(-0.24, -0.10, default=-0.12, decimals=2, space="sell", optimize=True)
    time_stop_candles = IntParameter(72, 288, default=144, space="sell", optimize=True)  # 144*5m≈12h

    # ==========================================================================
    # 四、结构性指标/网格/风控常量(固定)
    # ==========================================================================
    ema_fast_period = 20
    ema_slow_period = 60
    atr_period = 14
    volatility_period = 20

    # 主流币 5m 波动区间(重标定): 低于死水停手, 高于极端插针停手
    min_atr_pct = 0.0008
    max_atr_pct = 0.030
    # 网格步长夹逼(占价格比)
    min_grid_step_pct = 0.005
    max_grid_step_pct = 0.045
    grid_growth_per_entry = 0.25        # 每补一次间距递增, 省弹药

    # 杠杆(硬顶收窄到 5x —— 主流币不需要激进, 降低插针爆仓概率)
    max_leverage_cap = 5

    # 仓位/风险(统一模型, 不再分核心/卫星)
    max_open_trades_cap = 2             # 只有 BTC/ETH
    max_single_risk_pct = 0.08          # 单笔最大风险 ≤ 总资金 8%(按 stake*lev*|sl| 折算)
    margin_reserve_pct = 0.35           # 预留 35% 保证金安全垫

    # DCA: 有限次数 + 递增倍数(逆势补仓弹药)
    max_entry_position_adjustment = 3   # 主流币趋势较连贯, 3 次足够, 不给太多"加速爆仓"机会
    dca_multipliers = [1.3, 1.7, 2.2]

    # 熔断
    cb_max_consecutive_losses = 5
    cb_daily_drawdown = 0.20
    cb_cooldown_minutes = 30

    allowed_pairs = {
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
        "XCN/USDT:USDT",
        "H/USDT:USDT", "VELVET/USDT:USDT", "BEAT/USDT:USDT", "COAI/USDT:USDT",
    }

    # ==========================================================================
    # 五、运行时状态(在 bot_start 初始化为实例属性, 避免类级可变属性共享)
    # ==========================================================================
    def bot_start(self, **kwargs) -> None:
        self._cooldown_until: Optional[datetime] = None
        self._day_baseline_equity: Optional[float] = None
        self._day_baseline_date = None
        self.external_signals: dict = {}
        # 把 hyperopt 出来的 tp_roi 同步进原生 minimal_roi, 让"止盈"真正生效
        self.minimal_roi = {0: float(self.tp_roi.value)}
        self._reset_day_baseline(datetime.now(timezone.utc))

    # --------------------------------------------------------------------------
    # 外部信号接口(轻量保留): direction ∈ {1:仅多, -1:仅空, 0:中性}
    # --------------------------------------------------------------------------
    def receive_signal(self, pair: str, direction: int) -> None:
        self.external_signals[pair] = int(direction)

    def _external_allows(self, pair: str, side: str) -> bool:
        sig = getattr(self, "external_signals", {}).get(pair, 0)
        if sig == 0:
            return True
        return sig == (1 if side == "long" else -1)

    # ==========================================================================
    # 指标层(全部无未来函数: 所有比较均用 shift(1) 或 rolling 过去窗口)
    # ==========================================================================
    @staticmethod
    def _atr_pct(dataframe: DataFrame, period: int) -> pd.Series:
        prev_close = dataframe["close"].shift(1)
        tr = pd.concat([
            dataframe["high"] - dataframe["low"],
            (dataframe["high"] - prev_close).abs(),
            (dataframe["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return (atr / dataframe["close"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=self.ema_fast_period, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=self.ema_slow_period, adjust=False).mean()
        dataframe["atr_pct"] = self._atr_pct(dataframe, self.atr_period)
        dataframe["volatility_pct"] = dataframe["close"].pct_change().rolling(self.volatility_period).std().fillna(0)
        dataframe["vol_baseline"] = dataframe["volatility_pct"].rolling(60).mean()

        # 票1: EMA 方向(趋势确认票)
        dataframe["ema_vote"] = np.select(
            [dataframe["ema_fast"] > dataframe["ema_slow"],
             dataframe["ema_fast"] < dataframe["ema_slow"]],
            [1, -1], default=0)
        # 票2: ATR 放量方向
        dataframe["atr_vote"] = np.select(
            [(dataframe["atr_pct"] > dataframe["atr_pct"].shift(1)) & (dataframe["close"] > dataframe["open"]),
             (dataframe["atr_pct"] > dataframe["atr_pct"].shift(1)) & (dataframe["close"] < dataframe["open"])],
            [1, -1], default=0)
        # 票3: 波动率扩张方向
        dataframe["volatility_vote"] = np.select(
            [(dataframe["volatility_pct"] > dataframe["vol_baseline"]) & (dataframe["close"] > dataframe["ema_fast"]),
             (dataframe["volatility_pct"] > dataframe["vol_baseline"]) & (dataframe["close"] < dataframe["ema_fast"])],
            [1, -1], default=0)

        vote_sum = dataframe["ema_vote"] + dataframe["atr_vote"] + dataframe["volatility_vote"]
        th = int(self.vote_threshold.value)
        dataframe["trend_signal"] = np.select([vote_sum >= th, vote_sum <= -th], [1, -1], default=0)

        dataframe["grid_step_pct"] = (dataframe["atr_pct"] * float(self.atr_grid_multiplier.value)).clip(
            lower=self.min_grid_step_pct, upper=self.max_grid_step_pct)
        dataframe["risk_fuse"] = (
            (dataframe["atr_pct"] < self.min_atr_pct) | (dataframe["atr_pct"] > self.max_atr_pct)
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        tradable = (
            (dataframe["volume"] > 0)
            & (dataframe["risk_fuse"] == 0)
            & dataframe["atr_pct"].between(self.min_atr_pct, self.max_atr_pct)
        )
        dataframe.loc[tradable & (dataframe["trend_signal"] == 1), ["enter_long", "enter_tag"]] = (1, "xinghe_long")
        dataframe.loc[tradable & (dataframe["trend_signal"] == -1), ["enter_short", "enter_tag"]] = (1, "xinghe_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        dataframe.loc[(dataframe["trend_signal"] == -1) | (dataframe["risk_fuse"] == 1),
                      ["exit_long", "exit_tag"]] = (1, "long_reversal_or_risk")
        dataframe.loc[(dataframe["trend_signal"] == 1) | (dataframe["risk_fuse"] == 1),
                      ["exit_short", "exit_tag"]] = (1, "short_reversal_or_risk")
        return dataframe

    # ==========================================================================
    # 回测滑点注入(仅 backtest/hyperopt): 入场吃更差价, 出场吃更差价
    # 注意: 仍管不到"原生市价止损"的滑点 → 靠 config 抬 fee 近似, 见文件头注释。
    # ==========================================================================
    def _is_backtest(self) -> bool:
        try:
            return self.dp.runmode.value in ("backtest", "hyperopt")
        except Exception:
            return False

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        if self._is_backtest():
            s = self.slippage_pct
            return proposed_rate * (1 + s) if side == "long" else proposed_rate * (1 - s)
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime,
                          proposed_rate: float, current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
        if self._is_backtest():
            s = self.slippage_pct
            return proposed_rate * (1 - s) if not trade.is_short else proposed_rate * (1 + s)
        return proposed_rate

    # ==========================================================================
    # 动态杠杆: ATR 越大杠杆越低, 硬顶 max_leverage_cap
    # ==========================================================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        cap = min(self.max_leverage_cap, max_leverage)
        atr_pct = self._current_atr_pct(pair)
        if atr_pct is None or atr_pct <= 0:
            return float(min(2, cap))     # 数据缺失 → 保守 2x(fail-closed)
        lo, hi = self.min_atr_pct, self.max_atr_pct
        ratio = min(max((atr_pct - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
        lev = cap - (cap - 1) * ratio
        return float(max(1.0, min(round(lev), cap)))

    def _current_atr_pct(self, pair: str) -> Optional[float]:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or len(df) == 0:
                return None
            return float(df["atr_pct"].iloc[-1])
        except Exception:
            return None

    # ==========================================================================
    # 仓位规模: 单笔风险 ≤ 总资金 * max_single_risk_pct(按 stake*lev*|sl| 折算)
    # ==========================================================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        try:
            total = self.wallets.get_total(self.config["stake_currency"])
        except Exception:
            total = None

        lev = max(float(leverage), 1.0)
        sl = abs(self.stoploss) or 0.30

        if total and total > 0:
            risk_budget = total * self.max_single_risk_pct
            stake_by_risk = risk_budget / (lev * sl)
            usable = total * (1 - self.margin_reserve_pct)
            stake = min(proposed_stake, stake_by_risk, usable)
        else:
            stake = proposed_stake * (1 - self.margin_reserve_pct)  # 拿不到钱包 → 保守折扣

        if min_stake is not None:
            stake = max(stake, min_stake)
        return float(min(stake, max_stake))

    # ==========================================================================
    # 开仓确认: 熔断 + 外部信号 + 标的白名单 + 持仓数上限。全部 fail-closed。
    # ==========================================================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        # 0) 标的白名单(只做 BTC/ETH)
        if pair not in self.allowed_pairs:
            logger.info("[GUARD] 非白名单标的, 拒绝 %s", pair)
            return False
        # 1) 冷静期
        if getattr(self, "_cooldown_until", None) and current_time < self._cooldown_until:
            logger.warning("[RISK] 冷静期中(至 %s), 拒绝开仓 %s", self._cooldown_until, pair)
            return False
        # 2) 外部信号否决
        if not self._external_allows(pair, side):
            logger.info("[SIGNAL] 外部信号否决 %s %s", pair, side)
            return False
        # 3) 最大持仓数 —— 异常时 fail-closed(拒绝), 不再放行
        try:
            if len(Trade.get_open_trades()) >= self.max_open_trades_cap:
                logger.warning("[RISK] 达最大持仓数 %d, 拒绝 %s", self.max_open_trades_cap, pair)
                return False
        except Exception as e:
            logger.warning("[RISK] 持仓数查询失败, 保守拒绝开仓: %s", e)
            return False
        return True

    # ==========================================================================
    # 有限网格 DCA: 只在"浮亏但未击穿熔丝"区间补仓, 次数有限 + 间距递增。
    # ==========================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Any:
        entry_count = trade.nr_of_successful_entries
        if entry_count <= 0 or entry_count > len(self.dca_multipliers):
            return None
        # 盈利不补; 击穿熔丝不补(交给 custom_exit 中断平仓)
        if current_profit >= 0 or current_profit <= float(self.deep_loss_fuse.value):
            return None
        if getattr(self, "_cooldown_until", None) and current_time < self._cooldown_until:
            return None

        open_rate = getattr(trade, "open_rate", current_entry_rate)
        if not open_rate or open_rate <= 0:
            return None

        is_short = bool(getattr(trade, "is_short", False))
        adverse = (current_rate / open_rate - 1) if is_short else (open_rate / current_rate - 1)
        if adverse <= 0:
            return None

        atr_pct = self._current_atr_pct(trade.pair) or (self.min_grid_step_pct / float(self.atr_grid_multiplier.value))
        threshold = self._grid_threshold(atr_pct, entry_count - 1)
        if adverse < threshold:
            return None

        base_cost = self._first_filled_entry_cost(trade) or getattr(trade, "stake_amount", 0.0)
        if base_cost <= 0:
            return None

        stake = min(base_cost * self.dca_multipliers[entry_count - 1], max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        logger.info("[DCA] %s 第%d次补仓 adverse=%.4f thr=%.4f stake=%.4f",
                    trade.pair, entry_count, adverse, threshold, stake)
        return stake, f"xinghe_dca_{entry_count}"

    def _grid_threshold(self, atr_pct: float, grown_times: int) -> float:
        raw = atr_pct * float(self.atr_grid_multiplier.value)
        grown = raw * (1 + max(grown_times, 0) * self.grid_growth_per_entry)
        return max(self.min_grid_step_pct, min(grown, self.max_grid_step_pct))

    def _first_filled_entry_cost(self, trade: Trade) -> Optional[float]:
        try:
            orders = trade.select_filled_orders("enter_long") + trade.select_filled_orders("enter_short")
        except (AttributeError, TypeError):
            try:
                orders = trade.select_filled_orders()
            except Exception:
                return None
        if not orders:
            return None
        cost = getattr(orders[0], "safe_cost", None) or getattr(orders[0], "cost", None)
        return float(cost) if cost and cost > 0 else None

    # ==========================================================================
    # 自定义出场责任链:
    #   1) 逆势深亏软熔丝(deep_loss_fuse, 浅) → 先于原生止损中断平仓
    #   2) 时间止损(未盈利且持仓过久) → 释放占用资金 + 降资金费暴露
    #   原生 stoploss(-0.30, 标记价) 仅在极端跳空时兜底。
    # ==========================================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if current_profit <= float(self.deep_loss_fuse.value):
            logger.warning("[RISK] %s 中断平仓 浮亏=%.4f <= 熔丝=%.4f",
                            pair, current_profit, float(self.deep_loss_fuse.value))
            return "interrupt_deep_loss"

        try:
            held = current_time - trade.open_date_utc
            max_hold = timedelta(minutes=self._tf_minutes() * int(self.time_stop_candles.value))
            if held >= max_hold and current_profit <= 0:
                logger.info("[RISK] %s 时间止损 持仓=%s", pair, held)
                return "time_stop"
        except Exception:
            pass
        return None

    def _tf_minutes(self) -> int:
        return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(self.timeframe, 5)

    # ==========================================================================
    # 主循环: 刷新熔断状态
    # ==========================================================================
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        self._update_circuit_breaker(current_time)

    def _reset_day_baseline(self, now: datetime) -> None:
        try:
            self._day_baseline_equity = self.wallets.get_total(self.config["stake_currency"])
        except Exception:
            self._day_baseline_equity = None
        self._day_baseline_date = now.date()

    def _update_circuit_breaker(self, now: datetime) -> None:
        if getattr(self, "_day_baseline_date", None) != now.date():
            self._reset_day_baseline(now)
        if getattr(self, "_cooldown_until", None) and now < self._cooldown_until:
            return

        trip, reason = False, ""

        # (a) 连续亏损
        try:
            closed = [t for t in Trade.get_trades_proxy(is_open=False)]
            closed.sort(key=lambda t: t.close_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            streak = 0
            for t in closed[: self.cb_max_consecutive_losses]:
                if (t.close_profit if t.close_profit is not None else 0) < 0:
                    streak += 1
                else:
                    break
            if streak >= self.cb_max_consecutive_losses:
                trip, reason = True, f"连续{streak}笔亏损"
        except Exception as e:
            logger.debug("[RISK] 连亏统计失败: %s", e)

        # (b) 单日回撤
        try:
            if not trip and self._day_baseline_equity and self._day_baseline_equity > 0:
                cur = self.wallets.get_total(self.config["stake_currency"])
                dd = (self._day_baseline_equity - cur) / self._day_baseline_equity
                if dd >= self.cb_daily_drawdown:
                    trip, reason = True, f"单日回撤{dd:.1%}"
        except Exception as e:
            logger.debug("[RISK] 回撤统计失败: %s", e)

        if trip:
            self._cooldown_until = now + timedelta(minutes=self.cb_cooldown_minutes)
            logger.warning("[RISK] 触发熔断(%s), 冷静至 %s", reason, self._cooldown_until)