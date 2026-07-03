# -*- coding: utf-8 -*-
"""
gate.io 永续 OCO 保护括号 (失效保险 / Dead-man's Switch) + 孤儿单重启清理
=====================================================================
⚠️ 重要事实(同上一版, 不再赘述):
  1. gate 永续无原生原子OCO; 这里用"两条独立reduceOnly触发单 + 轮询模拟一撤一"实现。
  2. freqtrade 默认不知道我们挂在交易所端的触发单 → 本模块仅作【失效保险】。
  3. dry_run 不发真单。
  4. ccxt 各版本对 gate 触发单 params 命名可能不同, 首次实盘前小仓验证。

【本版新增 — 孤儿单清理】
  - 每条触发单的 clientOrderId 编码 trade.id: 格式 't-oco_{trade_id}_{leg}'
    (gate 要求 text 以 't-' 开头, ≤28字符, [0-9a-zA-Z_-])。
  - bot_start 时调用 reconcile_orphans():
      · 交易所端有括号、但对应 trade 已不在 open_trades → 撤掉(真正的孤儿);
      · 对应 trade 仍 open → 把 order_id 重建回 trade.custom_data(恢复映射);
  - 解决"挂括号成功但 commit 前进程崩溃"导致的交易所有单/本地无记录问题。

【仍存局限】
  - reconcile 按 pair_whitelist 逐个 fetch_open_orders(gate 要求带 symbol);
    若某孤儿的 pair 已从 whitelist 移除, 则扫不到 → 改 whitelist 后建议手动核对一次。
"""

import logging
from datetime import datetime
from typing import Optional

from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    ccxt = None


class GateOCOFailsafeMixin:
    """需与 IStrategy 子类多重继承: class X(GateOCOFailsafeMixin, BaseStrategy)。"""

    # ---- 结构性开关 ----
    oco_enabled = True
    oco_tp_pct = 0.06
    oco_sl_buffer = 1.15

    OCO_COID_PREFIX = "t-oco"   # gate text 必须 't-' 开头; 全前缀 't-oco' 用于识别本策略的单
    _oco_ex = None

    # ------------------------------------------------------------------
    def _oco_exchange(self):
        if not self.oco_enabled or ccxt is None:
            return None
        if self.config.get("dry_run", True):
            return None
        if self._oco_ex is None:
            cfg = self.config.get("exchange", {})
            self._oco_ex = ccxt.gateio({
                "apiKey": cfg.get("key", ""),
                "secret": cfg.get("secret", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
        return self._oco_ex

    @staticmethod
    def _close_side(trade: Trade) -> str:
        return "buy" if trade.is_short else "sell"

    # ---- clientOrderId 编解码 ----
    @classmethod
    def _oco_coid(cls, trade_id: int, leg: str) -> str:
        """生成 't-oco_{id}_{leg}'。leg ∈ {'tp','sl'}。"""
        coid = f"{cls.OCO_COID_PREFIX}_{trade_id}_{leg}"
        return coid[:28]  # gate 长度上限保护(trade_id 极大时截断, 实际不会触及)

    @classmethod
    def _parse_coid(cls, coid: str):
        """'t-oco_123_tp' → (123, 'tp'); 非本策略单返回 None。"""
        if not coid or not coid.startswith(cls.OCO_COID_PREFIX):
            return None
        try:
            body = coid[len(cls.OCO_COID_PREFIX):].strip("_")  # '123_tp'
            tid_str, leg = body.split("_", 1)
            return int(tid_str), leg
        except (ValueError, AttributeError):
            return None

    # ------------------------------------------------------------------
    def place_bracket(self, trade: Trade) -> None:
        """挂 TP+SL 两条 reduceOnly 触发单, 带 clientOrderId 以便重启反查。"""
        ex = self._oco_exchange()
        if ex is None:
            logger.info("[OCO] dry-run/未启用, 跳过挂括号: %s", trade.pair)
            return

        self.cancel_bracket(trade)  # 均价变了, 先撤旧括号

        avg, amount, sym = trade.open_rate, trade.amount, trade.pair
        sl_dist = abs(self.stoploss) * self.oco_sl_buffer
        if trade.is_short:
            tp_price, sl_price = avg * (1 - self.oco_tp_pct), avg * (1 + sl_dist)
        else:
            tp_price, sl_price = avg * (1 + self.oco_tp_pct), avg * (1 - sl_dist)
        side = self._close_side(trade)

        ids = {}
        try:
            # clientOrderId → ccxt 映射到 gate 'text' 字段; gate 强制 't-' 前缀, 已在 _oco_coid 保证。
            tp_o = ex.create_order(sym, "market", side, amount, None, {
                "reduceOnly": True, "takeProfitPrice": tp_price,
                "clientOrderId": self._oco_coid(trade.id, "tp"),
            })
            ids["tp"] = tp_o["id"]
            sl_o = ex.create_order(sym, "market", side, amount, None, {
                "reduceOnly": True, "stopLossPrice": sl_price,
                "clientOrderId": self._oco_coid(trade.id, "sl"),
            })
            ids["sl"] = sl_o["id"]
            trade.set_custom_data("oco_orders", ids)
            logger.info("[OCO] 已挂括号 %s tp=%.8g sl=%.8g amount=%.8g",
                        sym, tp_price, sl_price, amount)
        except Exception as e:
            for oid in ids.values():
                try:
                    ex.cancel_order(oid, sym, params={"stop": True})
                except Exception as cancel_error:
                    logger.debug("[OCO] 回滚括号单 %s 失败: %s", oid, cancel_error)
            trade.set_custom_data("oco_orders", {})
            logger.warning("[OCO] 挂括号失败 %s: %s", sym, e)

    # ------------------------------------------------------------------
    def cancel_bracket(self, trade: Trade) -> None:
        ex = self._oco_exchange()
        if ex is None:
            return
        ids = trade.get_custom_data("oco_orders") or {}
        for leg, oid in ids.items():
            try:
                ex.cancel_order(oid, trade.pair, params={"stop": True})
            except Exception as e:
                logger.debug("[OCO] 撤单 %s(%s) 失败(可能已成交/已撤): %s", oid, leg, e)
        trade.set_custom_data("oco_orders", {})

    # ------------------------------------------------------------------
    def sync_brackets(self) -> None:
        """轮询模拟 OCO: 某腿成交 → 撤另一腿 + 写 flag 给 custom_exit 对账。"""
        ex = self._oco_exchange()
        if ex is None:
            return
        for trade in Trade.get_open_trades():
            ids = trade.get_custom_data("oco_orders") or {}
            if not ids:
                continue
            triggered_leg = None
            for leg, oid in ids.items():
                try:
                    o = ex.fetch_order(oid, trade.pair, params={"stop": True})
                except Exception:
                    continue
                if o.get("status") in ("closed", "finished", "filled"):
                    triggered_leg = leg
                    break
            if triggered_leg:
                other = "sl" if triggered_leg == "tp" else "tp"
                oid_other = ids.get(other)
                if oid_other:
                    try:
                        ex.cancel_order(oid_other, trade.pair, params={"stop": True})
                    except Exception:
                        pass
                trade.set_custom_data("oco_orders", {})
                trade.set_custom_data("oco_triggered", triggered_leg)
                logger.warning("[OCO] %s 交易所端 %s 腿已触发; 已撤另一腿, 待 freqtrade 平账",
                               trade.pair, triggered_leg)

    # ------------------------------------------------------------------
    def reconcile_orphans(self) -> None:
        """
        bot_start 时调用一次: 凭 clientOrderId 反查交易所端触发单归属。
          - 归属于已关闭 trade → 撤掉(孤儿);
          - 归属于仍 open 的 trade → 重建 custom_data 映射。
        """
        ex = self._oco_exchange()
        if ex is None:
            logger.info("[OCO] dry-run/未启用, 跳过孤儿单清理")
            return

        try:
            open_trade_ids = {t.id for t in Trade.get_open_trades()}
        except Exception:
            open_trade_ids = set()

        # gate 拉触发单必须带 symbol → 遍历 whitelist + proxy
        pairs = set(self.config.get("exchange", {}).get("pair_whitelist", []))
        if hasattr(self, "proxy_pair"):
            pairs.add(self.proxy_pair)

        rebuilt: dict = {}      # trade_id -> {leg: order_id}
        orphan_cnt = 0
        for p in pairs:
            try:
                open_orders = ex.fetch_open_orders(p, params={"stop": True})
            except Exception as e:
                logger.debug("[OCO] reconcile 拉取 %s 触发单失败: %s", p, e)
                continue
            for o in open_orders:
                coid = o.get("clientOrderId") or (o.get("info") or {}).get("text", "")
                parsed = self._parse_coid(coid)
                if not parsed:
                    continue   # 不是本策略挂的单, 不碰
                tid, leg = parsed
                if tid not in open_trade_ids:
                    # 孤儿: 对应 trade 已关闭, 但交易所还挂着 → 撤
                    try:
                        ex.cancel_order(o["id"], p, params={"stop": True})
                        orphan_cnt += 1
                        logger.warning("[OCO] 清理孤儿单 trade=%s leg=%s id=%s pair=%s",
                                       tid, leg, o["id"], p)
                    except Exception as e:
                        logger.warning("[OCO] 撤孤儿单失败 id=%s: %s", o["id"], e)
                else:
                    rebuilt.setdefault(tid, {})[leg] = o["id"]

        # 把重建的映射写回各 open trade(覆盖可能已丢失的本地记录)
        rebuilt_cnt = 0
        for t in Trade.get_open_trades():
            if t.id in rebuilt:
                t.set_custom_data("oco_orders", rebuilt[t.id])
                rebuilt_cnt += 1
        logger.info("[OCO] reconcile 完成: 清理孤儿 %d 条, 重建映射 %d 个 trade",
                    orphan_cnt, rebuilt_cnt)

    # ------------------------------------------------------------------
    # freqtrade 回调钩子
    # ------------------------------------------------------------------
    def order_filled(self, pair, trade, order, current_time, **kwargs):
        try:
            is_entry = (order.ft_order_side == trade.entry_side) and order.status == "closed"
            if is_entry:
                self.place_bracket(trade)
        except Exception as e:
            logger.warning("[OCO] order_filled 挂括号异常: %s", e)
        return super().order_filled(pair, trade, order, current_time, **kwargs)

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time, **kwargs) -> bool:
        self.cancel_bracket(trade)
        try:
            return super().confirm_trade_exit(
                pair, trade, order_type, amount, rate,
                time_in_force, exit_reason, current_time, **kwargs)
        except (AttributeError, TypeError):
            return True

    def bot_start(self, **kwargs) -> None:
        try:
            super().bot_start(**kwargs)
        except (AttributeError, TypeError):
            pass
        self.reconcile_orphans()

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        try:
            super().bot_loop_start(current_time=current_time, **kwargs)
        except (AttributeError, TypeError):
            pass
        self.sync_brackets()

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        triggered_leg = trade.get_custom_data("oco_triggered")
        if triggered_leg:
            trade.set_custom_data("oco_triggered", None)
            return f"oco_exchange_{triggered_leg}"
        try:
            return super().custom_exit(
                pair, trade, current_time, current_rate, current_profit, **kwargs
            )
        except (AttributeError, TypeError):
            return None
