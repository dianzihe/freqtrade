# -*- coding: utf-8 -*-
"""
Meme 对冲代理马丁策略 (Hedge-Proxy Martingale) —— 集成动态β + 交易所端OCO失效保险
=====================================================================
（策略意图见上一版文件头；本版新增两块）
  [新增2] 动态 β: 对冲规模按"名义敞口"中性, hedge_notional = ratio × β × meme_notional,
          β = Cov(r_meme,r_proxy)/Var(r_proxy) 滚动估计, clip[0.3,3.0], 不足回退1.0。
  [新增6] 交易所端 OCO 失效保险: 见 gate_oco.GateOCOFailsafeMixin 文件头警告。

⚠️【回测限制】对冲腿/β加权依赖跨腿实时持仓 → 回测不忠实, 用 dry-run/forward test 验证。
⚠️【OCO限制】gate 永续无原子OCO, 仅作 dead-man's switch; 实盘前小仓验证三动作生效。

参数名                     | 含义                         | 默认值      | 建议调优范围            | 影响
buy_atr_pct_min            | 入场最小波动率(ATR占比)      | 0.006       | 0.004–0.020             | 太低→做小波动无回归空间；太高→信号稀少
buy_rng_low / buy_rng_high | 入场区间位置带               | 0.12 / 0.72 | 0.05–0.30 / 0.55–0.85   | 控制"接刀深度"，过窄信号少，过宽接最深的刀
sell_rsi_min               | RSI 转强止盈线               | 58          | 50–70                   | 偏低→早走少赚，偏高→错过回落
hedge_ratio                | 对冲覆盖主腿名义比例         | 0.5         | 0.2–1.0                 | 越高越中性但越贵(资金费率+错失反弹)
hedge_trigger_level        | 第几层马丁后启动对冲         | 2           | 1–3                     | 越小越早对冲(保守)，越大越激进
proxy_ema_fast/slow        | 代理趋势EMA                  | 20/60       | 偏结构性                | 决定"大盘确认下跌"的灵敏度与滞后
stoploss                   | 单腿硬止损(结构性)           | -0.20       | 一般固定                | 最后防线，配合马丁防单笔无限放大
max_leverage_cap           | 杠杆硬顶(结构性)             | 6.0         | 固定                    | 风控红线，不可越过
margin_reserve_ratio       | 预留保证金(结构性)           | 0.30        | 0.2–0.4                 | 安全垫，越大越抗极端但资金效率越低
max_holding_candles        | 时间止损K线数(结构性)        | 192         | 96–384                  | 防温水煮青蛙式套牢
beta_window                | β 滚动估计窗口（K线数）      | 96         | 48–192     | 短→β灵敏但抖动大；长→稳但滞后
BETA_MIN/MAX               | β 截断边界（结构性）         | 0.3 / 3.0  | 固定       | 防噪声算出荒谬值导致对冲过度/不足
oco_tp_pct                 | 交易所端止盈距离             | 0.06       | 设宽于 ROI | 让 freqtrade 先止盈，交易所只兜底
oco_sl_buffer              | 交易所止损放宽系数           | 1.15       | 1.05–1.3   | 越大越靠后兜底，freqtrade 止损先动手
oco_enabled                | OCO 总开关                   | True       | —          | 实盘验证前可设 False
"""

from datetime import datetime
from typing import Optional

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
)

from Meme_马丁_基类 import MemeMartingaleBaseStrategy
from gate_oco import GateOCOFailsafeMixin


# MRO: Mixin 在前, 保证 order_filled/confirm_trade_exit 先走 OCO 钩子再 super() 到基类
class MemeHedgeProxyMartingaleStrategy(GateOCOFailsafeMixin, MemeMartingaleBaseStrategy):
    """对冲代理马丁: 主腿做多妖币马丁, 系统性风险用代理永续空单(β中性)对冲 + 交易所OCO兜底。"""

    # ===================== 结构性参数 =====================
    timeframe = "15m"
    can_short = True
    position_adjustment_enable = True
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 200          # β 需要更长预热(滚动窗口 + EMA)

    minimal_roi = {"180": 0.0, "45": 0.018, "0": 0.04}
    stoploss = -0.20

    proxy_pair = "BTC/USDT:USDT"
    max_leverage_cap = 6.0
    margin_reserve_ratio = 0.30
    max_holding_candles = 192

    # β clip 边界(结构性: 防噪声算出荒谬值)
    BETA_MIN, BETA_MAX, BETA_FALLBACK = 0.3, 3.0, 1.0

    # OCO 失效保险开关(继承自 Mixin, 这里显式覆盖以便集中管理)
    oco_enabled = True
    oco_tp_pct = 0.06
    oco_sl_buffer = 1.15

    # ===================== 行情敏感参数(Hyperopt) =====================
    buy_atr_pct_min = DecimalParameter(0.004, 0.020, default=0.006, decimals=4,
                                       space="buy", optimize=True)
    buy_rng_low = DecimalParameter(0.05, 0.30, default=0.12, decimals=2,
                                   space="buy", optimize=True)
    buy_rng_high = DecimalParameter(0.55, 0.85, default=0.72, decimals=2,
                                    space="buy", optimize=True)
    sell_rsi_min = IntParameter(50, 70, default=58, space="sell", optimize=True)

    proxy_ema_fast = IntParameter(10, 30, default=20, space="buy", optimize=False)
    proxy_ema_slow = IntParameter(40, 120, default=60, space="buy", optimize=False)

    hedge_ratio = DecimalParameter(0.2, 1.0, default=0.5, decimals=2,
                                   space="buy", optimize=True)
    hedge_trigger_level = IntParameter(1, 3, default=2, space="buy", optimize=True)
    beta_window = IntParameter(48, 192, default=96, space="buy", optimize=False)  # β滚动窗口(结构性)

    HEDGE_TAG = "hedge_proxy_short"

    # 跨腿共享状态
    _hedge_state: dict = {
        "meme_stake": 0.0,
        "beta_weighted_notional": 0.0,   # Σ(β_i × 名义_i): 对冲名义的目标基数
        "max_dca_level": 0,
        "hedge_needed": False,
        "has_hedge": False,
    }

    # ------------------------------------------------------------------
    @property
    def protections(self):
        cooldown = max(1, int(30 / 15))
        day_candles = int(24 * 60 / 15)
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": cooldown},
            {"method": "StoplossGuard", "lookback_period_candles": day_candles,
             "trade_limit": 5, "stop_duration_candles": cooldown, "only_per_pair": False},
            {"method": "MaxDrawdown", "lookback_period_candles": day_candles,
             "trade_limit": 4, "stop_duration_candles": day_candles,
             "max_allowed_drawdown": 0.30},
        ]
    def bot_start(self, **kwargs) -> None:
        """bot 启动时清理上次遗留的 OCO 孤儿单并重建映射(见 gate_oco 文件头)。"""
        try:
            super().bot_start(**kwargs)
        except (AttributeError, TypeError):
            pass
        self.reconcile_orphans()
    # ------------------------------------------------------------------
    def _is_hedge_trade(self, trade: Trade) -> bool:
        return bool(trade.enter_tag) and trade.enter_tag.startswith("hedge_proxy")

    @staticmethod
    def _dca_level(trade: Trade) -> int:
        entries = [o for o in trade.orders
                   if getattr(o, "ft_is_entry", o.ft_order_side == trade.entry_side)
                   and o.status == "closed"]
        return max(0, len(entries) - 1)

    def informative_pairs(self):
        return [(self.proxy_pair, self.timeframe)]

    # ------------------------------------------------------------------
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        try:
            super().bot_loop_start(current_time=current_time, **kwargs)
        except (AttributeError, TypeError):
            pass

        meme_stake, max_level, has_hedge, beta_w_notional = 0.0, 0, False, 0.0
        try:
            open_trades = Trade.get_open_trades()
        except Exception:
            open_trades = []

        for t in open_trades:
            if self._is_hedge_trade(t):
                has_hedge = True
                continue
            meme_stake += (t.stake_amount or 0.0)
            max_level = max(max_level, self._dca_level(t))

            # —— β 加权名义: 读该主腿最新 β 与现价, 累加 β_i × 名义_i ——
            beta_i, last_close = self.BETA_FALLBACK, t.open_rate
            try:
                df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                if df is not None and len(df):
                    if "beta" in df.columns and not np.isnan(df["beta"].iloc[-1]):
                        beta_i = float(df["beta"].iloc[-1])
                    last_close = float(df["close"].iloc[-1])
            except Exception:
                pass
            notional_i = (t.amount or 0.0) * last_close      # 名义 = 张数 × 现价
            beta_w_notional += beta_i * notional_i

        self._hedge_state = {
            "meme_stake": meme_stake,
            "beta_weighted_notional": beta_w_notional,
            "max_dca_level": max_level,
            "hedge_needed": max_level >= self.hedge_trigger_level.value and meme_stake > 0,
            "has_hedge": has_hedge,
        }

        # 交易所端 OCO 轮询(dry-run/未启用时内部直接返回)
        try:
            self.sync_brackets()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)

        proxy = self.dp.get_pair_dataframe(self.proxy_pair, self.timeframe)
        if proxy is not None and len(proxy) > self.proxy_ema_slow.value:
            proxy = proxy.copy()
            proxy["ema_fast"] = ta.EMA(proxy, timeperiod=self.proxy_ema_fast.value)
            proxy["ema_slow"] = ta.EMA(proxy, timeperiod=self.proxy_ema_slow.value)
            proxy["downtrend"] = (
                (proxy["ema_fast"] < proxy["ema_slow"]) & (proxy["close"] < proxy["ema_slow"])
            ).astype(int)
            dataframe = merge_informative_pair(
                dataframe, proxy[["date", "downtrend", "close"]],
                self.timeframe, self.timeframe, ffill=True,
            )
        else:
            dataframe[f"downtrend_{self.timeframe}"] = 0
            dataframe[f"close_{self.timeframe}"] = np.nan

        # —— 动态 β: 滚动 Cov(r_meme,r_proxy)/Var(r_proxy) ——
        # 仅在非代理 pair 上算(代理对自己 β=1 无意义)
        proxy_close_col = f"close_{self.timeframe}"
        if metadata["pair"] != self.proxy_pair and proxy_close_col in dataframe:
            w = self.beta_window.value
            r_meme = np.log(dataframe["close"] / dataframe["close"].shift(1))
            r_proxy = np.log(dataframe[proxy_close_col] / dataframe[proxy_close_col].shift(1))
            cov = r_meme.rolling(w).cov(r_proxy)
            var = r_proxy.rolling(w).var()
            beta = cov / var.replace(0, np.nan)
            dataframe["beta"] = beta.clip(self.BETA_MIN, self.BETA_MAX).fillna(self.BETA_FALLBACK)
        else:
            dataframe["beta"] = self.BETA_FALLBACK
        return dataframe

    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        proxy_down_col = f"downtrend_{self.timeframe}"

        if metadata["pair"] == self.proxy_pair:
            if self._hedge_state.get("hedge_needed") and not self._hedge_state.get("has_hedge"):
                dataframe.loc[
                    dataframe[proxy_down_col] == 1,
                    ["enter_short", "enter_tag"],
                ] = (1, self.HEDGE_TAG)
            return dataframe

        wide_enough = dataframe["atr_pct"] > self.buy_atr_pct_min.value
        lower_band_touch = dataframe["close"] < dataframe["bb_lower"]
        range_alive = dataframe["range_position"].between(
            self.buy_rng_low.value, self.buy_rng_high.value
        )
        proxy_safe = dataframe.get(proxy_down_col, 0) == 0

        dataframe.loc[
            wide_enough & lower_band_touch & range_alive & proxy_safe,
            ["enter_long", "enter_tag"],
        ] = (1, "hedge_proxy_lower_band")
        return dataframe

    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)

        if metadata["pair"] == self.proxy_pair:
            if not self._hedge_state.get("hedge_needed"):
                dataframe.loc[:, ["exit_short", "exit_tag"]] = (1, "hedge_release")
            dataframe.loc[
                dataframe.get(f"downtrend_{self.timeframe}", 0) == 0,
                ["exit_short", "exit_tag"],
            ] = (1, "hedge_proxy_up")
            return dataframe

        dataframe.loc[
            (dataframe["close"] > dataframe["bb_mid"])
            | (dataframe["rsi"] > self.sell_rsi_min.value),
            ["exit_long", "exit_tag"],
        ] = (1, "hedge_proxy_mean_exit")
        return dataframe

    # ------------------------------------------------------------------
    def adjust_trade_position(self, trade: Trade, current_time, current_rate,
                              current_profit, min_stake, max_stake,
                              current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        if self._is_hedge_trade(trade):
            return None
        if self._dca_level(trade) >= len(self.dca_thresholds) and current_profit <= self.dca_thresholds[-1]:
            return None
        return super().adjust_trade_position(
            trade, current_time, current_rate, current_profit, min_stake, max_stake,
            current_entry_rate, current_exit_rate, current_entry_profit,
            current_exit_profit, **kwargs,
        )

    # ------------------------------------------------------------------
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        # OCO 对账: 交易所端括号已触发(仓位实际已平) → 让 freqtrade 平掉本地记录
        if trade.get_custom_data("oco_triggered"):
            leg = trade.get_custom_data("oco_triggered")
            trade.set_custom_data("oco_triggered", None)
            return f"oco_exchange_{leg}"

        if self._is_hedge_trade(trade):
            return None

        if (self._dca_level(trade) >= len(self.dca_thresholds)
                and current_profit <= self.dca_thresholds[-1]):
            return "martingale_exhausted_cut"

        held = (current_time - trade.open_date_utc).total_seconds() / 60
        if held >= self.max_holding_candles * 15 and current_profit < 0:
            return "time_stop"
        return None

    # ------------------------------------------------------------------
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str, **kwargs) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            atr_pct = float(df["atr_pct"].iloc[-1]) if len(df) else 0.01
        except Exception:
            atr_pct = 0.01
        dyn = max(1.0, min(self.max_leverage_cap, 0.05 / max(atr_pct, 0.001)))
        return float(min(dyn, self.max_leverage_cap, max_leverage))

    # ------------------------------------------------------------------
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        if entry_tag == self.HEDGE_TAG:
            # β中性: 对冲名义 = ratio × Σ(β_i × 主腿名义_i); 再换算成保证金 = 名义 / 杠杆
            target_notional = self._hedge_state.get("beta_weighted_notional", 0.0) * self.hedge_ratio.value
            stake = target_notional / max(leverage, 1.0)
            if min_stake:
                stake = max(stake, min_stake)
            return float(min(stake, max_stake))
        return proposed_stake

    # ------------------------------------------------------------------
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:
        if entry_tag == self.HEDGE_TAG and self._hedge_state.get("has_hedge"):
            return False
        try:
            total = self.wallets.get_total_stake_amount()
            free = self.wallets.get_available_stake_amount()
            if total > 0 and free < total * self.margin_reserve_ratio:
                return False
        except Exception:
            pass
        return True