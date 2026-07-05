# -*- coding: utf-8 -*-
"""
风控扩展 Mixin：标记价格取价、强平价硬校验、回测滑点建模。
"""
import time
import logging
from datetime import datetime
from typing import Optional

from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class RiskExtMixin:
    SLIPPAGE_PCT = 0.005
    MAINTENANCE_MARGIN_RATE = 0.005
    LIQ_SAFETY_RATIO = 0.75
    _MARK_CACHE_TTL = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mark_cache: dict = {}

    def _is_backtesting(self) -> bool:
        try:
            return self.dp.runmode.value in ("backtest", "hyperopt")
        except Exception:
            return False

    def _mark_price(self, pair: str, fallback: float) -> float:
        if self._is_backtesting():
            return fallback
        ex_getter = getattr(self, "_oco_exchange", None)
        ex = ex_getter() if callable(ex_getter) else None
        if ex is None:
            return fallback
        now = time.time()
        cached = self._mark_cache.get(pair)
        if cached and now - cached[0] < self._MARK_CACHE_TTL:
            return cached[1]
        try:
            t = ex.fetch_ticker(pair)
            mp = (t.get("info") or {}).get("mark_price") or t.get("markPrice")
            if mp:
                mp = float(mp)
                self._mark_cache[pair] = (now, mp)
                return mp
        except Exception as e:
            logger.debug("[RISK] 取 mark price 失败 %s: %s", pair, e)
        return fallback

    def _liq_price_move(self, trade: Trade) -> float:
        """返回价格层面到强平的距离(负数)。实盘优先用真实 liquidationPrice。"""
        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
        if not self._is_backtesting():
            ex_getter = getattr(self, "_oco_exchange", None)
            ex = ex_getter() if callable(ex_getter) else None
            if ex is not None:
                try:
                    for p in ex.fetch_positions([trade.pair]):
                        lp = p.get("liquidationPrice")
                        if lp and trade.open_rate:
                            lp = float(lp)
                            return (lp - trade.open_rate) / trade.open_rate
                except Exception as e:
                    logger.debug("[RISK] 取强平价失败 %s: %s", trade.pair, e)
        return -(1.0 / lev - self.MAINTENANCE_MARGIN_RATE)

    def _safe_interrupt_level(self, trade: Trade, wanted_level: float) -> float:
        """把设计中断线与强平安全线取更靠近 0 者，保证中断永远先于强平。"""
        liq_move = abs(self._liq_price_move(trade))
        safe_level = -(liq_move * self.LIQ_SAFETY_RATIO)
        eff = max(wanted_level, safe_level)
        if eff != wanted_level:
            logger.warning(
                "[%s] 中断线被强平安全线收紧: 设计=%.3f 强平≈%.3f 生效=%.3f",
                trade.pair, wanted_level, -liq_move, eff)
        return eff

    # ---- 滑点：仅回测/超参注入 ----
    def custom_entry_price(self, pair: str, trade: Optional[Trade],
                           current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        if not self._is_backtesting():
            return proposed_rate
        slip = self.SLIPPAGE_PCT
        return proposed_rate * (1 + slip) if side == "long" else proposed_rate * (1 - slip)

    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime,
                          proposed_rate: float, current_profit: float,
                          exit_tag: Optional[str], **kwargs) -> float:
        if not self._is_backtesting():
            return proposed_rate
        slip = self.SLIPPAGE_PCT
        return proposed_rate * (1 + slip) if trade.is_short else proposed_rate * (1 - slip)