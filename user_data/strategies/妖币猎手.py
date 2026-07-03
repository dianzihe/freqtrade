"""
MojiCoinHunter (妖币猎手) v4 — Gate 现货妖币自动猎取策略
=========================================================
v4 相比 v3 的关键修复与重构:
  [BUGFIX] medium/weak 分支死代码 — 之前被 score_ok/rules_ok 抬到与 strong 同门槛
  [BUGFIX] 参数重复定义覆盖 (sma_slope)
  [BUGFIX] RSI 在纯上涨时算成 0 反而误杀强势币
  [BUGFIX] startup_candle_count 200 < 288(24h量) 导致开头流动性误判
  [重构]   入场从"追动量"改为"蓄势压缩 → 量能异动 → 突破触发"
  [重构]   用自参照百分位(current vs 自身24h历史)替代固定绝对阈值
  [重构]   冷却改为 per-pair, 配额与 max_open_trades 对齐
  [新增]   短周期 ROI 阶梯, 让小脉冲能落袋

⚠️ 仅供学习研究, 不构成投资建议. Meme币波动极大, 实盘风险极高.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    stoploss_from_open,
)

logger = logging.getLogger(__name__)


class MojiCoinHunter(IStrategy):
    """
    Gate 现货妖币猎手 v4

    逻辑链: 前段安静蓄势(低波动) + 量能相对自身历史异动(高百分位)
            + 价格突破前段盘整高点 → 买入.
    分散 20 仓, 短ROI落袋 + 棘轮保利润 + 时间废票清理.
    """

    INTERFACE_VERSION = 3

    # ── 基础设置 ──
    can_short = False
    timeframe = "5m"
    startup_candle_count = 300  # 覆盖 288(24h量) + shift(72) 预热
    process_only_new_candles = True

    # ── 仓位管理 ──
    max_open_trades = 20
    position_adjustment_enable = False
    use_custom_stoploss = True

    stake_per_ticket = 1.0
    max_total_tickets = 200

    # ── 退出参数 ──
    # v4: 加短周期 ROI 阶梯, 让小脉冲能落袋 (原来只有 0:0.5 太钝)
    minimal_roi = {
        "0": 0.50,    # 立刻 +50% 直接走
        "30": 0.20,   # 30min 后 +20% 走
        "120": 0.10,  # 2h 后 +10% 走
        "360": 0.05,  # 6h 后 +5% 走 (避免长期占用仓位)
    }
    stoploss = -0.08
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

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

    # ═══════════════════════════════════════════════════════════════════════
    #  可调参数 (v4)
    # ═══════════════════════════════════════════════════════════════════════

    # 总闸门 (最低档, 分级在其上叠加标签)
    buy_score_threshold = IntParameter(40, 70, default=50, space="buy")
    buy_min_rules = IntParameter(1, 3, default=2, space="buy")

    # R1: 量比绝对阈值 (moji 2.93, ctrl 1.93)
    buy_vol_ratio = DecimalParameter(1.5, 4.0, default=2.0, decimals=1, space="buy")
    buy_vol_ratio_weight = IntParameter(20, 40, default=30, space="buy")

    # R1b: 量比自参照百分位 (当前量比 vs 该币自身24h历史) — 抗噪核心
    buy_vol_rank = DecimalParameter(0.70, 0.98, default=0.85, decimals=2, space="buy")
    buy_vol_rank_weight = IntParameter(15, 35, default=25, space="buy")

    # R2: SMA12 斜率 (moji 2.75%)  ※修复了原代码的重复定义
    buy_sma_slope_pct = DecimalParameter(0.5, 4.0, default=1.5, decimals=1, space="buy")
    buy_sma_slope_weight = IntParameter(10, 30, default=20, space="buy")

    # R3: 累计涨幅 (moji 3.88%)
    buy_cum_change_pct = DecimalParameter(1.0, 6.0, default=3.0, decimals=1, space="buy")
    buy_cum_change_weight = IntParameter(10, 25, default=15, space="buy")

    # R4: 蓄势压缩 — 前段(bars -72~-36)振幅要"安静", 越小越好
    #     compression = range_earlier < 阈值 → 命中
    buy_quiet_range_pct = DecimalParameter(2.0, 10.0, default=5.0, decimals=1, space="buy")
    buy_quiet_weight = IntParameter(10, 25, default=20, space="buy")

    # R5: 突破 — close 突破前段盘整高点 (必要条件之一, 也给分)
    buy_breakout_weight = IntParameter(15, 35, default=25, space="buy")

    # 流动性 & 价格
    buy_min_quote_volume = IntParameter(10000, 200000, default=50000, space="buy")
    min_price = 0.0001
    max_price = 500.0

    # RSI 安全下限 (避免急跌中接刀)
    buy_rsi_min = IntParameter(30, 50, default=38, space="buy")

    # ── 冷却与配额 (v4: per-pair, 与 20 仓对齐) ──
    per_pair_cooldown_min = 360   # 同一币 6h 内不重复买
    global_cooldown_min = 15      # 全局最小间隔仅 15min (原 120 卡死了仓位)
    max_daily_tickets = 40        # 放宽, 让 20 仓能开满并有周转

    # ═══════════════════════════════════════════════════════════════════════
    #  指标计算
    # ═══════════════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # ── 均线 ──
        df["sma12"] = df["close"].rolling(12, min_periods=6).mean()

        # ── R1: 量比 (最近1h / 之前5h) ──
        vol_recent_1h = df["volume"].rolling(12, min_periods=3).mean()
        vol_earlier_5h = df["volume"].shift(12).rolling(60, min_periods=24).mean()
        df["vol_ratio"] = (
            vol_recent_1h / vol_earlier_5h.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ── R1b: 量比自参照百分位 (当前值在过去24h窗口内的分位) ──
        #     这是把"绝对阈值"换成"相对自身异动"的关键, 抗横截面噪声
        df["vol_ratio_rank"] = (
            df["vol_ratio"].rolling(288, min_periods=72).rank(pct=True)
        ).fillna(0.0)

        # ── R2: SMA12 斜率 (6h) ──
        sma_6h_ago = df["sma12"].shift(72)
        df["sma_slope_pct"] = (
            (df["sma12"] - sma_6h_ago) / sma_6h_ago.replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ── R3: 累计涨幅 (6h) ──
        close_6h_ago = df["close"].shift(72)
        df["cum_change_pct"] = (
            (df["close"] - close_6h_ago) / close_6h_ago.replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ── R4: 蓄势压缩 — 前段(bars -72~-36)振幅要小 ──
        high_earlier = df["high"].shift(36).rolling(36, min_periods=18).max()
        low_earlier = df["low"].shift(36).rolling(36, min_periods=18).min()
        df["range_earlier_pct"] = (
            (high_earlier - low_earlier) / low_earlier.replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(999.0)  # 缺失当"不安静"

        # 参考: 后半段振幅 (用于观察扩张, 不做硬性)
        high_recent = df["high"].rolling(36, min_periods=18).max()
        low_recent = df["low"].rolling(36, min_periods=18).min()
        df["range_recent_pct"] = (
            (high_recent - low_recent) / low_recent.replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ── R5: 突破 — close 突破"前段盘整高点" ──
        #     prior_high = 过去 [-48, -1] 根 (4h) 的最高价, 不含当前bar (shift(1))
        prior_high = df["high"].rolling(48, min_periods=24).max().shift(1)
        df["prior_high"] = prior_high
        df["breakout"] = (df["close"] > prior_high).astype(int)

        # ── 流动性: 24h 报价量 ──
        df["quote_volume_24h"] = (
            df["volume"] * df["close"]
        ).rolling(288, min_periods=72).sum()

        # ── RSI (修复: 纯上涨时给 100 而非 0) ──
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(loss != 0, 100.0)                 # 无下跌 → 100
        rsi = rsi.where(~((loss == 0) & (gain == 0)), 50.0)  # 完全走平 → 50
        df["rsi"] = rsi.fillna(50.0)

        # ── 综合评分 & 规则计数 ──
        df["moji_score"] = self._compute_score(df)
        df["moji_rules_triggered"] = self._count_rules(df)

        return df

    def _rule_hits(self, df: DataFrame) -> dict[str, pd.Series]:
        """所有规则的命中布尔序列, score 和 count 共用, 保持一致"""
        return {
            "vol_ratio": df["vol_ratio"] >= self.buy_vol_ratio.value,
            "vol_rank": df["vol_ratio_rank"] >= self.buy_vol_rank.value,
            "sma_slope": df["sma_slope_pct"] >= self.buy_sma_slope_pct.value,
            "cum_change": df["cum_change_pct"] >= self.buy_cum_change_pct.value,
            "quiet": df["range_earlier_pct"] <= self.buy_quiet_range_pct.value,
            "breakout": df["breakout"] == 1,
        }

    def _compute_score(self, df: DataFrame) -> pd.Series:
        h = self._rule_hits(df)
        score = pd.Series(0.0, index=df.index)
        score += h["vol_ratio"].astype(float) * self.buy_vol_ratio_weight.value
        score += h["vol_rank"].astype(float) * self.buy_vol_rank_weight.value
        score += h["sma_slope"].astype(float) * self.buy_sma_slope_weight.value
        score += h["cum_change"].astype(float) * self.buy_cum_change_weight.value
        score += h["quiet"].astype(float) * self.buy_quiet_weight.value
        score += h["breakout"].astype(float) * self.buy_breakout_weight.value
        return score

    def _count_rules(self, df: DataFrame) -> pd.Series:
        h = self._rule_hits(df)
        count = pd.Series(0, index=df.index, dtype=int)
        for s in h.values():
            count += s.astype(int)
        return count

    # ═══════════════════════════════════════════════════════════════════════
    #  入场信号
    # ═══════════════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df["enter_long"] = 0
        df["enter_tag"] = ""

        # ── 必要条件 (硬闸门) ──
        # 1) 突破前段盘整高点 (妖币"启动"的定义性动作)
        # 2) 量能相对自身历史异动 (量比高分位 OR 绝对量比达标)
        # 3) 流动性 / 价格 / 不接刀
        vol_surge = (
            (df["vol_ratio_rank"] >= self.buy_vol_rank.value) |
            (df["vol_ratio"] >= self.buy_vol_ratio.value)
        )
        liquidity_ok = df["quote_volume_24h"] >= float(self.buy_min_quote_volume.value)
        price_ok = (df["close"] >= self.min_price) & (df["close"] <= self.max_price)
        rsi_ok = df["rsi"] >= self.buy_rsi_min.value

        gate = (
            (df["breakout"] == 1) &
            vol_surge &
            liquidity_ok &
            price_ok &
            rsi_ok &
            (df["moji_score"] >= self.buy_score_threshold.value) &
            (df["moji_rules_triggered"] >= self.buy_min_rules.value)
        )

        # ── 信号分级 (修复: 不再叠加 score_ok/rules_ok, medium/weak 现在真的能触发) ──
        strong = gate & (df["moji_score"] >= 90) & (df["moji_rules_triggered"] >= 4)
        df.loc[strong, ["enter_long", "enter_tag"]] = (1, "moji_strong")

        medium = gate & (df["enter_long"] == 0) & (df["moji_score"] >= 70)
        df.loc[medium, ["enter_long", "enter_tag"]] = (1, "moji_medium")

        weak = gate & (df["enter_long"] == 0)
        df.loc[weak, ["enter_long", "enter_tag"]] = (1, "moji_weak")

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe

    # ═══════════════════════════════════════════════════════════════════════
    #  棘轮止损
    # ═══════════════════════════════════════════════════════════════════════

    def custom_stoploss(
        self, pair: str, trade: Any, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs: Any,
    ) -> float:
        if current_profit >= 1.00:
            return stoploss_from_open(0.50, current_profit)
        if current_profit >= 0.50:
            return stoploss_from_open(0.25, current_profit)
        if current_profit >= 0.20:
            return stoploss_from_open(0.10, current_profit)
        if current_profit >= 0.10:
            return stoploss_from_open(0.0, current_profit)
        return 1  # 保持默认 -8%

    # ═══════════════════════════════════════════════════════════════════════
    #  自定义退出 (时间止损 + 废票清理)
    # ═══════════════════════════════════════════════════════════════════════

    def custom_exit(
        self, pair: str, trade: Any, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs: Any,
    ) -> str | bool | None:
        opened = getattr(trade, "open_date_utc", current_time)
        age_h = max((current_time - opened).total_seconds() / 3600.0, 0.0)

        peak_profit = None
        max_rate = getattr(trade, "max_rate", None)
        if max_rate and trade.open_rate:
            peak_profit = (max_rate - trade.open_rate) / trade.open_rate

        if age_h >= 6.0 and current_profit < -0.05:
            return "dud_early"
        if age_h >= 12.0 and current_profit < 0.0:
            return "stale_half_day"
        if age_h >= 24.0 and current_profit < 0.01:
            return "stale_dud"
        if age_h >= 48.0 and current_profit < 0.05:
            return "medium_2d"
        if age_h >= 72.0:
            return "max_hold_3d"

        # 从高点回撤保护
        if peak_profit is not None and peak_profit >= 0.15 and current_profit >= 0.08:
            if (peak_profit - current_profit) >= 0.12:
                return "pullback_exit"

        return None

    # ═══════════════════════════════════════════════════════════════════════
    #  资金管理
    # ═══════════════════════════════════════════════════════════════════════

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float | None, max_stake: float,
        leverage: float, entry_tag: str | None, side: str, **kwargs: Any,
    ) -> float:
        stake = self.stake_per_ticket
        if min_stake and stake < min_stake:
            return float(min_stake)
        return min(float(stake), float(max_stake))

    # ═══════════════════════════════════════════════════════════════════════
    #  入场确认 (per-pair 冷却 + 配额)
    # ═══════════════════════════════════════════════════════════════════════

    def bot_start(self, **kwargs: Any) -> None:
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._daily_ticket_counts: dict[str, int] = {}
        self._total_tickets_bought = 0
        self._last_entry_time: Optional[datetime] = None
        self._last_entry_time_per_pair: dict[str, datetime] = {}

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime,
        entry_tag: str | None = None, side: str = "long", **kwargs: Any,
    ) -> bool:
        if not hasattr(self, "_daily_ticket_counts"):
            self._reset_counters()

        if rate < self.min_price or rate > self.max_price:
            return False

        if self._total_tickets_bought >= self.max_total_tickets:
            return False

        day_key = current_time.date().isoformat()
        day_count = self._daily_ticket_counts.get(day_key, 0)
        if day_count >= self.max_daily_tickets:
            return False

        # per-pair 冷却: 同一币短期不重复买
        last_pair = self._last_entry_time_per_pair.get(pair)
        if last_pair is not None:
            gap = (current_time - last_pair).total_seconds() / 60.0
            if gap < self.per_pair_cooldown_min:
                return False

        # 全局冷却 (很小, 只防同一tick爆买)
        if self._last_entry_time is not None:
            gap = (current_time - self._last_entry_time).total_seconds() / 60.0
            if gap < self.global_cooldown_min:
                return False

        self._daily_ticket_counts[day_key] = day_count + 1
        self._total_tickets_bought += 1
        self._last_entry_time = current_time
        self._last_entry_time_per_pair[pair] = current_time
        logger.info(
            f"Confirm {pair} ({entry_tag}): day={day_count + 1}, "
            f"total={self._total_tickets_bought}, rate={rate}"
        )
        return True