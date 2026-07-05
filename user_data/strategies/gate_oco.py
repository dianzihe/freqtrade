# -*- coding: utf-8 -*-
"""
gate.io 永续 OCO 保护括号 (失效保险 / Dead-man's Switch) + 孤儿单重启清理
      + 插针防御体系 L0~L5 (Wick Defense Stack)
=====================================================================
⚠️ 重要事实(同前版, 不再赘述):
  1. gate 永续无原生原子OCO; 这里用"两条独立触发单 + 轮询模拟一撤一"实现。
  2. freqtrade 默认不知道我们挂在交易所端的触发单 → 本模块仅作【失效保险】。
  3. dry_run 不发真单。
  4. ccxt 各版本对 gate 触发单 params 命名可能不同, 首次实盘前小仓验证。

============================== 本次安全修复 CHANGELOG ==============================
  [FIX-2] place_bracket 的 SL 距离不再锚 self.stoploss(马丁下它常=-0.99会算出负价
          导致交易所拒单、SL 静默消失)。改为优先取马丁"中断线"，并加 50% 硬上限，
          确保永远算得出一个有效、合理的交易所端止损。见 _oco_sl_distance()。
  [FIX-3] _handle_wg_fill 挂的紧止盈单：
          (a) reduceOnly 由 False→True，避免 hedge 模式下"反向开新仓"而非平掉意外仓位；
          (b) 补上带前缀的 clientOrderId，使其可被 reconcile_orphans 识别，不再变孤儿单。
  [FIX-4] L4 保证金台账双向漏账修复：
          (a) 防守档位被吃到成交时，从 _wg_margin_ledger 扣减(预留→转真实仓位)；
          (b) reconcile_orphans 重启后重建 _wg_margin_ledger / _wg_last_quote_price，
              否则硬顶保护在重启后失效。
  [FIX-5] L3 跟随重挂逻辑修复：防守单始终以【均价 open_rate】为中心，原先用"现价漂移"
          触发重挂却仍围绕不变的均价重挂，属自相矛盾且高频空转。改为以"均价变化"
          触发重挂(即仅在 DCA 改变持仓成本后跟随)，语义与实现一致。
  注: [FIX-1](策略文件钩子未调 super 导致本 mixin 钩子全部失效) 需在【策略文件】侧修改，
      见本文件末尾附注。本文件已确保 mixin 侧钩子均正确 super()。

============================== 插针防御体系分层 ==============================
【L0 触发价类型】用 mark price 触发 TP/SL，从源头抵抗薄流动性插针误触发。
【L1 确认窗口(Debounce)】本地风控决策要求条件持续 ≥ confirm_seconds 才执行。
【L2 止损两级降落伞】stop-limit 主(限滑点) + 市价兜底(跳空时强制离场)。
【L3 防守单进化】多档阶梯 + 随机抖动抗定位 + 均价漂移后跟随重挂。
【L4 保证金隔离 + 被触发后处置】硬顶预算隔离；被吃到立刻挂紧止盈并记台账。
【L5 流动性熔断】盘口 spread 异常放大时延后本地兜底动作，避免最坏点位成交。
（各层详细设计说明同前版，为节省篇幅此处保留要点。）
"""

import logging
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional

from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    ccxt = None


class GateOCOFailsafeMixin:
    """需与 IStrategy 子类多重继承: class X(GateOCOFailsafeMixin, BaseStrategy)。"""

    # ================= OCO 结构性开关 =================
    oco_enabled = True
    oco_tp_pct = 0.06
    oco_sl_buffer = 1.15
    OCO_COID_PREFIX = "t-oco"
    _oco_ex = None

    # [FIX-2] SL 距离兜底与硬上限：马丁把 self.stoploss 设到极深(如 -0.99)时不能拿它当锚
    oco_sl_fallback_dist = 0.20         # 取不到中断线时的回退 SL 距离(20%)，而非 self.stoploss
    oco_sl_max_dist = 0.50              # SL 距离硬上限，永不允许算出 >50% 的荒谬触发价

    # ---- L0: 触发价类型 ----
    oco_trigger_price_type = "mark"
    oco_trigger_price_type_raw = 1      # gate 原生: 0=last,1=mark,2=index

    # ---- L2: 止损两级降落伞 ----
    oco_sl_limit_slippage_pct = 0.010
    oco_gap_fallback_buffer = 0.02

    # ---- L1: 确认窗口 ----
    default_confirm_seconds = 8

    # ---- L3: 插针防守单(Wick Guard) ----
    wick_guard_enabled = True
    wick_guard_rungs = ((1.0, 0.45), (1.6, 0.35), (2.4, 0.20))
    wick_guard_distance_pct = 0.12
    wick_guard_atr_mult = 4.0
    wick_guard_amount_ratio = 0.30
    wick_guard_jitter_pct = 0.15
    wick_guard_requote_drift_pct = 0.03
    WG_COID_PREFIX = "t-wg"

    # ---- L4: 保证金隔离 ----
    WICK_GUARD_MARGIN_RATIO = 0.05
    wg_post_trigger_tp_pct = 0.02

    # ---- L5: 流动性熔断 ----
    depth_check_enabled = True
    depth_spike_multiplier = 3.0
    depth_history_len = 60

    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._breach_since: dict = {}
        self._wg_last_quote_price: dict = {}
        self._wg_margin_ledger: dict = defaultdict(float)
        self._wg_captured_positions: dict = defaultdict(list)
        self._spread_history: dict = defaultdict(lambda: deque(maxlen=self.depth_history_len))

    # ------------------------------------------------------------------
    def _oco_exchange(self):
        if not (self.oco_enabled or self.wick_guard_enabled) or ccxt is None:
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

    # ---- OCO clientOrderId 编解码 ----
    @classmethod
    def _oco_coid(cls, trade_id: int, leg: str) -> str:
        return f"{cls.OCO_COID_PREFIX}_{trade_id}_{leg}"[:28]

    @classmethod
    def _parse_coid(cls, coid: str):
        if not coid or not coid.startswith(cls.OCO_COID_PREFIX):
            return None
        try:
            body = coid[len(cls.OCO_COID_PREFIX):].strip("_")
            tid_str, leg = body.split("_", 1)
            return int(tid_str), leg
        except (ValueError, AttributeError):
            return None

    # ---- Wick Guard clientOrderId 编解码 ----
    # leg 形如 'dn0','up1'(防守单) 或 'tp_dn0'(被吃到后挂的紧止盈单，见 FIX-3)
    @classmethod
    def _wg_coid(cls, trade_id: int, leg: str) -> str:
        return f"{cls.WG_COID_PREFIX}_{trade_id}_{leg}"[:28]

    @classmethod
    def _parse_wg_coid(cls, coid: str):
        if not coid or not coid.startswith(cls.WG_COID_PREFIX):
            return None
        try:
            body = coid[len(cls.WG_COID_PREFIX):].strip("_")
            tid_str, leg = body.split("_", 1)
            return int(tid_str), leg
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _is_capture_tp_leg(leg: str) -> bool:
        """[FIX-3] 判断某个 wg leg 是否是'被吃到后的紧止盈单'(不能当普通防守单撤)。"""
        return isinstance(leg, str) and leg.startswith("tp_")

    # ==================================================================
    # L1: 确认窗口(Debounce)
    # ==================================================================
    def confirm_breach(self, trade: Trade, key: str, breached: bool,
                       current_time: datetime, confirm_seconds: Optional[int] = None) -> bool:
        ck = (trade.id, key)
        if not breached:
            self._breach_since.pop(ck, None)
            return False
        first_seen = self._breach_since.get(ck)
        if first_seen is None:
            self._breach_since[ck] = current_time
            return False
        window = confirm_seconds if confirm_seconds is not None else self.default_confirm_seconds
        return (current_time - first_seen).total_seconds() >= window

    # ==================================================================
    # L5: 流动性熔断
    # ==================================================================
    def _sample_spread(self, pair: str) -> Optional[float]:
        ex = self._oco_exchange()
        if ex is None:
            return None
        try:
            ob = ex.fetch_order_book(pair, limit=5)
            bid = ob["bids"][0][0] if ob.get("bids") else None
            ask = ob["asks"][0][0] if ob.get("asks") else None
            if not bid or not ask or bid <= 0:
                return None
            spread_pct = (ask - bid) / bid
            self._spread_history[pair].append(spread_pct)
            return spread_pct
        except Exception as e:
            logger.debug("[L5] 拉取 %s 盘口失败: %s", pair, e)
            return None

    def is_orderbook_abnormal(self, pair: str) -> bool:
        if not self.depth_check_enabled:
            return False
        cur = self._sample_spread(pair)
        hist = self._spread_history.get(pair)
        if cur is None or not hist or len(hist) < 10:
            return False
        sorted_hist = sorted(hist)
        median = sorted_hist[len(sorted_hist) // 2]
        if median <= 0:
            return False
        return cur > median * self.depth_spike_multiplier

    def risk_gate(self, trade: Trade, key: str, breached: bool, current_time: datetime,
                 confirm_seconds: Optional[int] = None, bypass_depth_check: bool = False) -> bool:
        if not self.confirm_breach(trade, key, breached, current_time, confirm_seconds):
            return False
        if not bypass_depth_check and self.is_orderbook_abnormal(trade.pair):
            logger.info("[L5] %s 盘口异常(疑似插针瞬间)，延后执行 %s", trade.pair, key)
            return False
        return True

    # ==================================================================
    # [FIX-2] OCO 止损距离计算
    # ==================================================================
    def _oco_sl_distance(self, trade: Trade) -> float:
        """
        [FIX-2 修正] 失效保险 SL 应落在: 设计中断线之外(更深一点)、强平安全线之内。
        做法: 把"想要的 SL 距离"作为 wanted_level 交给 risk_ext._safe_interrupt_level,
        由它用真实 liquidationPrice 夹紧(max(wanted, -liq*LIQ_SAFETY_RATIO)),
        保证本 SL 永远早于强平, 又不会被马丁的深 stoploss 算出负价。
        """
        # 设计中断线(相对均价的负向回撤): 策略可暴露 oco_design_interrupt_level(属性或方法)
        design = getattr(self, "oco_design_interrupt_level", None)
        if callable(design):
            try:
                design = design(trade)
            except Exception:
                design = None
        if design not in (None, 0):
            base = abs(float(design)) * self.oco_sl_buffer   # 比中断线再深一点做兜底
        else:
            base = self.oco_sl_fallback_dist                 # 取不到设计线时的合理回退

        wanted = -abs(base)
        dist = abs(base)
        if hasattr(self, "_safe_interrupt_level"):
            try:
                # 正确传两个参数; 返回被强平安全线夹紧后的生效负向水平
                eff = self._safe_interrupt_level(trade, wanted)
                dist = abs(float(eff))
            except Exception as e:
                logger.debug("[OCO] _safe_interrupt_level 调用失败, 用回退距离: %s", e)

        dist = min(dist, self.oco_sl_max_dist)
        if dist <= 0:
            dist = min(self.oco_sl_fallback_dist, self.oco_sl_max_dist)
        return dist

    # ==================================================================
    # OCO 括号单：L0 + L2
    # ==================================================================
    def place_bracket(self, trade: Trade) -> None:
        ex = self._oco_exchange()
        if ex is None or not self.oco_enabled:
            logger.info("[OCO] dry-run/未启用, 跳过挂括号: %s", trade.pair)
            return

        self.cancel_bracket(trade)

        avg, amount, sym = trade.open_rate, trade.amount, trade.pair
        sl_dist = self._oco_sl_distance(trade)     # [FIX-2] 不再用 abs(self.stoploss)
        if trade.is_short:
            tp_price = avg * (1 - self.oco_tp_pct)
            sl_trigger = avg * (1 + sl_dist)
            sl_limit = sl_trigger * (1 + self.oco_sl_limit_slippage_pct)
        else:
            tp_price = avg * (1 + self.oco_tp_pct)
            sl_trigger = avg * (1 - sl_dist)
            sl_limit = sl_trigger * (1 - self.oco_sl_limit_slippage_pct)

        # [FIX-2] 双重兜底：即便上面出了意外，也绝不发出 <=0 的价格
        if tp_price <= 0 or sl_trigger <= 0 or sl_limit <= 0:
            logger.error("[OCO] %s 计算出非法价格(tp=%.8g sl_trigger=%.8g sl_limit=%.8g)，"
                         "放弃挂括号，请检查中断线/均价", sym, tp_price, sl_trigger, sl_limit)
            return

        side = self._close_side(trade)
        trigger_params = {
            "triggerPriceType": self.oco_trigger_price_type,
            "price_type": self.oco_trigger_price_type_raw,
        }

        ids = {}
        try:
            tp_o = ex.create_order(sym, "market", side, amount, None, {
                "reduceOnly": True, "takeProfitPrice": tp_price,
                "clientOrderId": self._oco_coid(trade.id, "tp"),
                **trigger_params,
            })
            ids["tp"] = tp_o["id"]

            sl_o = ex.create_order(sym, "limit", side, amount, sl_limit, {
                "reduceOnly": True, "stopLossPrice": sl_trigger,
                "clientOrderId": self._oco_coid(trade.id, "sl"),
                **trigger_params,
            })
            ids["sl"] = sl_o["id"]
            ids["sl_trigger_price"] = sl_trigger
            trade.set_custom_data("oco_orders", ids)
            logger.info("[OCO] 已挂括号(mark触发) %s tp=%.8g sl_trigger=%.8g sl_limit=%.8g (sl_dist=%.2f%%)",
                        sym, tp_price, sl_trigger, sl_limit, sl_dist * 100)
        except Exception as e:
            for leg, oid in ids.items():
                if leg == "sl_trigger_price":
                    continue
                try:
                    ex.cancel_order(oid, sym, params={"stop": True})
                except Exception as cancel_error:
                    logger.debug("[OCO] 回滚括号单 %s 失败: %s", oid, cancel_error)
            trade.set_custom_data("oco_orders", {})
            logger.warning("[OCO] 挂括号失败 %s: %s", sym, e)

    def cancel_bracket(self, trade: Trade) -> None:
        ex = self._oco_exchange()
        if ex is None:
            return
        ids = trade.get_custom_data("oco_orders") or {}
        for leg, oid in ids.items():
            if leg == "sl_trigger_price":
                continue
            try:
                ex.cancel_order(oid, trade.pair, params={"stop": True})
            except Exception as e:
                logger.debug("[OCO] 撤单 %s(%s) 失败(可能已成交/已撤): %s", oid, leg, e)
        trade.set_custom_data("oco_orders", {})

    def _force_market_fallback(self, trade: Trade, leg_oid: str) -> None:
        ex = self._oco_exchange()
        try:
            ex.cancel_order(leg_oid, trade.pair, params={"stop": True})
        except Exception:
            pass
        side = self._close_side(trade)
        try:
            ex.create_order(trade.pair, "market", side, trade.amount, None, {"reduceOnly": True})
            logger.warning("[L2] %s 检测到SL跳空未成交，已市价强制兜底离场", trade.pair)
        except Exception as e:
            logger.error("[L2] 市价兜底离场失败 %s: %s", trade.pair, e)

    def sync_brackets(self) -> None:
        ex = self._oco_exchange()
        if ex is None:
            return
        for trade in Trade.get_open_trades():
            ids = trade.get_custom_data("oco_orders") or {}
            if not ids:
                continue
            triggered_leg = None
            for leg, oid in ids.items():
                if leg == "sl_trigger_price":
                    continue
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
                continue

            # --- L2: 跳空兜底检测 ---
            sl_trigger = ids.get("sl_trigger_price")
            sl_oid = ids.get("sl")
            if sl_trigger and sl_oid:
                try:
                    ticker = ex.fetch_ticker(trade.pair)
                    last = ticker.get("last") or ticker.get("close")
                except Exception:
                    last = None
                if last:
                    gapped = (not trade.is_short and last < sl_trigger * (1 - self.oco_gap_fallback_buffer)) or \
                             (trade.is_short and last > sl_trigger * (1 + self.oco_gap_fallback_buffer))
                    if gapped:
                        self._force_market_fallback(trade, sl_oid)
                        trade.set_custom_data("oco_orders", {})
                        trade.set_custom_data("oco_triggered", "sl")

    # ==================================================================
    # L3+L4: 插针防守单(Wick Guard)
    # ==================================================================
    def _wg_distance_pct(self, pair: str) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df) > 0 and "atr_pct" in df.columns:
                atr_pct = float(df["atr_pct"].iloc[-1])
                if atr_pct > 0:
                    return max(atr_pct * self.wick_guard_atr_mult, self.wick_guard_distance_pct)
        except Exception:
            pass
        return self.wick_guard_distance_pct

    def _wg_total_budget_amount(self, trade: Trade) -> float:
        base_amount = trade.amount * self.wick_guard_amount_ratio
        try:
            total_stake = self.wallets.get_total_stake_amount()
        except Exception:
            total_stake = None
        if total_stake:
            margin_cap = total_stake * self.WICK_GUARD_MARGIN_RATIO
            used = self._wg_margin_ledger[trade.pair]
            remaining_margin = max(margin_cap - used, 0)
            lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
            max_amount_by_margin = (remaining_margin * lev) / max(trade.open_rate, 1e-9)
            if max_amount_by_margin <= 0:
                logger.warning("[L4] %s 防守单保证金已达硬顶(%.2f)，跳过本次挂单", trade.pair, margin_cap)
                return 0.0
            base_amount = min(base_amount, max_amount_by_margin)
        return base_amount

    def place_wick_guards(self, trade: Trade, force: bool = False) -> None:
        """
        L3: 在【均价】上下按阶梯挂多档限价单(非reduceOnly，独立对冲仓位)。
        L4: 总量受保证金硬顶约束，并做台账记录。
        ⚠️ 前提：交易所账户必须开启双向持仓(hedge mode)，否则这两笔反向单会与
        主仓净持仓相抵。上线前务必先小仓验证 positionMode 与成交效果。
        """
        ex = self._oco_exchange()
        if ex is None or not self.wick_guard_enabled:
            logger.info("[WG] dry-run/未启用, 跳过挂防守单: %s", trade.pair)
            return

        # [FIX-3/4] 仅撤销普通防守档位，保留"被吃到后的紧止盈单"(它对应真实持仓，不能被误撤)
        self.cancel_wick_guards(trade, keep_capture_tp=True)

        avg, sym = trade.open_rate, trade.pair
        total_amount = self._wg_total_budget_amount(trade)
        if total_amount <= 0:
            return
        base_distance = self._wg_distance_pct(sym)

        # 保留已存在的紧止盈单条目，重挂只覆盖普通档位
        existing = trade.get_custom_data("wg_orders") or {}
        ids = {leg: info for leg, info in existing.items() if self._is_capture_tp_leg(leg)}

        placed_notional = 0.0
        try:
            for i, (dist_mult, ratio) in enumerate(self.wick_guard_rungs):
                jitter = 1 + random.uniform(-self.wick_guard_jitter_pct, self.wick_guard_jitter_pct)
                distance = base_distance * dist_mult * jitter
                rung_amount = total_amount * ratio
                if rung_amount <= 0:
                    continue

                dn_price = avg * (1 - distance)
                up_price = avg * (1 + distance)
                if dn_price <= 0 or up_price <= 0:
                    continue

                long_o = ex.create_order(sym, "limit", "buy", rung_amount, dn_price, {
                    "reduceOnly": False,
                    "clientOrderId": self._wg_coid(trade.id, f"dn{i}"),
                })
                ids[f"dn{i}"] = {"id": long_o["id"], "price": dn_price, "amount": rung_amount, "side": "buy"}

                short_o = ex.create_order(sym, "limit", "sell", rung_amount, up_price, {
                    "reduceOnly": False,
                    "clientOrderId": self._wg_coid(trade.id, f"up{i}"),
                })
                ids[f"up{i}"] = {"id": short_o["id"], "price": up_price, "amount": rung_amount, "side": "sell"}

                placed_notional += rung_amount * avg * 2
                logger.info("[WG] %s 阶梯#%d dn=%.8g up=%.8g amount=%.8g(distance=%.2f%%)",
                            sym, i, dn_price, up_price, rung_amount, distance * 100)

            trade.set_custom_data("wg_orders", ids)
            self._wg_last_quote_price[trade.id] = avg
            self._wg_margin_ledger[sym] += placed_notional / max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
        except Exception as e:
            for leg, info in ids.items():
                if self._is_capture_tp_leg(leg):
                    continue
                try:
                    ex.cancel_order(info["id"], sym)
                except Exception as cancel_error:
                    logger.debug("[WG] 回滚防守单 %s 失败: %s", info.get("id"), cancel_error)
            # 回滚时保留紧止盈单条目
            trade.set_custom_data("wg_orders",
                                  {leg: info for leg, info in ids.items() if self._is_capture_tp_leg(leg)})
            logger.warning("[WG] 挂防守单失败 %s: %s", sym, e)

    def cancel_wick_guards(self, trade: Trade, keep_capture_tp: bool = False) -> None:
        """
        撤销未成交的阶梯防守单，并释放 L4 保证金台账占用。
        [FIX-3] keep_capture_tp=True 时保留"被吃到后的紧止盈单"(tp_* 前缀)，
                因为它对应一笔真实持仓，撤掉会留下裸仓。仅在"跟随重挂"时使用。
                主交易退出(confirm_trade_exit)时应传 False，整体清理。
        """
        ex = self._oco_exchange()
        if ex is None:
            return
        ids = trade.get_custom_data("wg_orders") or {}
        released_notional = 0.0
        remained = {}
        for leg, info in ids.items():
            if keep_capture_tp and self._is_capture_tp_leg(leg):
                remained[leg] = info
                continue
            oid = info.get("id") if isinstance(info, dict) else info
            try:
                ex.cancel_order(oid, trade.pair)
            except Exception as e:
                logger.debug("[WG] 撤防守单 %s(%s) 失败(可能已成交/已撤): %s", oid, leg, e)
            # 只有普通防守档位才占用了 L4 预留台账，紧止盈单不计入预留
            if isinstance(info, dict) and not self._is_capture_tp_leg(leg):
                released_notional += info.get("amount", 0) * info.get("price", 0) * 2 \
                    if False else info.get("amount", 0) * info.get("price", 0)
        if released_notional:
            lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
            self._wg_margin_ledger[trade.pair] = max(
                0.0, self._wg_margin_ledger[trade.pair] - released_notional / lev
            )
        trade.set_custom_data("wg_orders", remained)
        if not remained:
            self._wg_last_quote_price.pop(trade.id, None)

    def manual_cancel_wick_guards(self, pair: str) -> None:
        for trade in Trade.get_trades_proxy(pair=pair, is_open=True):
            self.cancel_wick_guards(trade, keep_capture_tp=False)
            logger.info("[WG] 人工撤销防守单: %s trade_id=%s", pair, trade.id)

    def _handle_wg_fill(self, trade: Trade, leg: str, info: dict) -> dict:
        """
        L4: 某档防守单被吃到后的处置。
        [FIX-3] 关键修复：
          (a) reduceOnly=True —— hedge 模式下，用反向 side + reduceOnly 才是"平掉这笔
              意外接盘的仓位"；原先 reduceOnly=False 会在对侧开出全新仓位，锁不住任何东西。
          (b) 补 clientOrderId(带 wg 前缀 tp_*) —— 使这张紧止盈单能被 reconcile_orphans
              识别；否则重启后无前缀无法辨认，成为无法清理的孤儿单。
        返回一个 wg_orders 条目(leg='tp_<原leg>')，由调用方并入 still_open，
        使其纳入统一的轮询/重建/清理生命周期。
        为什么不做反手/持有等更复杂处理：插针后最优应对高度依赖临场判断，
        自动化在极端时刻风险更大，故只做"止盈锁定 + 台账标记留人工决策"。
        """
        ex = self._oco_exchange()
        fill_price = info["price"]
        amount = info["amount"]
        opened_side = info["side"]                       # 'buy' 或 'sell'
        close_side = "sell" if opened_side == "buy" else "buy"
        tp_price = fill_price * (1 + self.wg_post_trigger_tp_pct) if opened_side == "buy" \
            else fill_price * (1 - self.wg_post_trigger_tp_pct)
        tp_leg = f"tp_{leg}"
        try:
            tp_o = ex.create_order(trade.pair, "limit", close_side, amount, tp_price, {
                "reduceOnly": True,                                   # [FIX-3a]
                "clientOrderId": self._wg_coid(trade.id, tp_leg),     # [FIX-3b]
            })
            entry = {"id": tp_o.get("id"), "price": tp_price, "amount": amount,
                     "side": close_side, "capture_side": opened_side, "fill_price": fill_price}
            self._wg_captured_positions[trade.id].append({
                "leg": leg, "side": opened_side, "amount": amount,
                "fill_price": fill_price, "tp_order_id": tp_o.get("id"), "tp_price": tp_price,
            })
            logger.warning("[WG][L4] %s 防守单%s被插针吃到！已挂 reduceOnly 紧止盈 %.8g 锁定处置，"
                           "请人工核查该笔意外仓位(trade_id=%s)", trade.pair, leg, tp_price, trade.id)
            return {tp_leg: entry}
        except Exception as e:
            logger.error("[WG][L4] 被触发档位挂止盈失败 %s leg=%s: %s", trade.pair, leg, e)
            return {}

    def sync_wick_guards(self, trade: Trade) -> None:
        """
        每个 bot_loop 轮询：
          - 检测各普通防守档是否被插针吃到 → 触发 L4 处置(挂紧止盈+标记+台账扣减)；
          - 检测紧止盈单是否成交 → 从台账清除；
          - L3 跟随：仅当【均价】相对上次挂单均价漂移超阈值(即发生了 DCA)才重挂。
        """
        ex = self._oco_exchange()
        if ex is None:
            return
        ids = trade.get_custom_data("wg_orders") or {}
        if ids:
            still_open = {}
            newly_captured = {}
            for leg, info in ids.items():
                oid = info.get("id") if isinstance(info, dict) else info
                try:
                    o = ex.fetch_order(oid, trade.pair)
                except Exception:
                    still_open[leg] = info
                    continue
                filled = o.get("status") in ("closed", "finished", "filled")

                if self._is_capture_tp_leg(leg):
                    # 这是被吃到后的紧止盈单
                    if filled:
                        logger.info("[WG][L4] %s 紧止盈单 %s 已成交，意外仓位已了结", trade.pair, leg)
                        # 不需要再保留；台账里对应记录留存供审计
                    else:
                        still_open[leg] = info
                    continue

                # 普通防守档
                if filled:
                    # [FIX-4a] 被吃到 → 预留保证金转为真实仓位，从台账扣减该档预留
                    if isinstance(info, dict):
                        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
                        released = info.get("amount", 0) * info.get("price", 0)
                        self._wg_margin_ledger[trade.pair] = max(
                            0.0, self._wg_margin_ledger[trade.pair] - released / lev
                        )
                    newly_captured.update(self._handle_wg_fill(trade, leg, info))
                else:
                    still_open[leg] = info
            still_open.update(newly_captured)
            trade.set_custom_data("wg_orders", still_open)

        # --- L3: 跟随重挂(修复版) ---
        # [FIX-5] 防守单始终以均价 open_rate 为中心分布。原实现用"现价漂移"触发，
        #         但重挂时中心仍是不变的均价 → 只是空转 churn。改为以"均价变化"触发：
        #         仅当 DCA 加仓改变了持仓成本时，才有必要围绕新均价重新分布防守单。
        last_quote = self._wg_last_quote_price.get(trade.id)
        current_avg = trade.open_rate
        if last_quote and current_avg > 0:
            drift = abs(current_avg - last_quote) / last_quote
            if drift > self.wick_guard_requote_drift_pct:
                logger.info("[WG][L3] %s 均价漂移 %.2f%%(DCA后)，跟随重挂防守单",
                            trade.pair, drift * 100)
                self.place_wick_guards(trade, force=True)

    # ------------------------------------------------------------------
    def reconcile_orphans(self) -> None:
        """
        bot_start 时清理 OCO 触发单 + Wick Guard 限价单 两类孤儿单，
        并重建内存态映射与台账。
        [FIX-4b] 重启后必须重建 _wg_margin_ledger / _wg_last_quote_price，
                 否则台账归零 → L4 硬顶失效 → 可突破预算继续加挂防守单。
        """
        ex = self._oco_exchange()
        if ex is None:
            logger.info("[OCO/WG] dry-run/未启用, 跳过孤儿单清理")
            return

        try:
            open_trades = list(Trade.get_open_trades())
        except Exception:
            open_trades = []
        open_trade_ids = {t.id for t in open_trades}
        trade_by_id = {t.id: t for t in open_trades}

        pairs = set(self.config.get("exchange", {}).get("pair_whitelist", []))
        if hasattr(self, "proxy_pair"):
            pairs.add(self.proxy_pair)

        rebuilt_oco: dict = {}
        rebuilt_wg: dict = {}
        orphan_cnt = 0

        for p in pairs:
            # --- OCO 触发单 ---
            try:
                stop_orders = ex.fetch_open_orders(p, params={"stop": True})
            except Exception as e:
                logger.debug("[OCO] reconcile 拉取 %s 触发单失败: %s", p, e)
                stop_orders = []
            for o in stop_orders:
                coid = o.get("clientOrderId") or (o.get("info") or {}).get("text", "")
                parsed = self._parse_coid(coid)
                if not parsed:
                    continue
                tid, leg = parsed
                if tid not in open_trade_ids:
                    try:
                        ex.cancel_order(o["id"], p, params={"stop": True})
                        orphan_cnt += 1
                        logger.warning("[OCO] 清理孤儿单 trade=%s leg=%s id=%s", tid, leg, o["id"])
                    except Exception as e:
                        logger.warning("[OCO] 撤孤儿单失败 id=%s: %s", o["id"], e)
                else:
                    rebuilt_oco.setdefault(tid, {})[leg] = o["id"]

            # --- Wick Guard 限价单(含普通防守档 dn*/up* 与紧止盈单 tp_*) ---
            try:
                limit_orders = ex.fetch_open_orders(p)
            except Exception as e:
                logger.debug("[WG] reconcile 拉取 %s 限价单失败: %s", p, e)
                limit_orders = []
            for o in limit_orders:
                coid = o.get("clientOrderId") or (o.get("info") or {}).get("text", "")
                parsed_wg = self._parse_wg_coid(coid)
                if not parsed_wg:
                    continue
                tid, leg = parsed_wg
                if tid not in open_trade_ids:
                    try:
                        ex.cancel_order(o["id"], p)
                        orphan_cnt += 1
                        logger.warning("[WG] 清理孤儿防守单 trade=%s leg=%s id=%s", tid, leg, o["id"])
                    except Exception as e:
                        logger.warning("[WG] 撤孤儿防守单失败 id=%s: %s", o["id"], e)
                else:
                    rebuilt_wg.setdefault(tid, {})[leg] = {
                        "id": o["id"], "price": o.get("price"), "amount": o.get("amount"),
                        "side": o.get("side"),
                    }

        # --- 重建映射 + [FIX-4b] 重建 L4 台账 ---
        rebuilt_cnt = 0
        # 先清空台账再按交易所真实挂单重算，避免与旧内存态叠加
        self._wg_margin_ledger.clear()
        for tid, legs in rebuilt_wg.items():
            t = trade_by_id.get(tid)
            if t is None:
                continue
            lev = max(getattr(t, "leverage", 1.0) or 1.0, 1.0)
            for leg, info in legs.items():
                if self._is_capture_tp_leg(leg):
                    continue  # 紧止盈单不计入防守预留
                price = info.get("price") or 0
                amount = info.get("amount") or 0
                self._wg_margin_ledger[t.pair] += (price * amount) / lev
            # 重建"上次挂单均价"用于 trailing 判断，避免重启后立刻误触发重挂
            self._wg_last_quote_price[tid] = t.open_rate

        for t in open_trades:
            changed = False
            if t.id in rebuilt_oco:
                t.set_custom_data("oco_orders", rebuilt_oco[t.id])
                changed = True
            if t.id in rebuilt_wg:
                t.set_custom_data("wg_orders", rebuilt_wg[t.id])
                changed = True
            if changed:
                rebuilt_cnt += 1
        logger.info("[OCO/WG] reconcile 完成: 清理孤儿 %d 条, 重建映射 %d 个 trade, "
                    "重建保证金台账 %d 个 pair", orphan_cnt, rebuilt_cnt, len(self._wg_margin_ledger))

    # ------------------------------------------------------------------
    # freqtrade 回调钩子(mixin 侧均正确 super()，勿在策略侧覆盖后漏调 super — 见 FIX-1 附注)
    # ------------------------------------------------------------------
    def order_filled(self, pair, trade, order, current_time, **kwargs):
        try:
            is_entry = (order.ft_order_side == trade.entry_side) and order.status == "closed"
            if is_entry:
                self.place_bracket(trade)
                self.place_wick_guards(trade)
        except Exception as e:
            logger.warning("[OCO/WG] order_filled 挂括号/防守单异常: %s", e)
        return super().order_filled(pair, trade, order, current_time, **kwargs)

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time, **kwargs) -> bool:
        self.cancel_bracket(trade)
        self.cancel_wick_guards(trade, keep_capture_tp=False)  # 整体清理，含紧止盈单
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
        for trade in Trade.get_open_trades():
            self.sync_wick_guards(trade)

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