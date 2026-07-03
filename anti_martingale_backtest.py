# -*- coding: utf-8 -*-
"""
Meme_反马丁_趋势.py 自定义回测引擎 v2 (BUG修复版)
====================================
全币种 x 全周期（9 x 3 = 27 组）独立运行。
输出：绩效指标 / 盈亏曲线 / 最大浮亏 / 加仓触发记录 / 爆仓风险统计。
"""

import json
import os
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 策略参数（与策略文件保持一致） ──────────────────────────────────────
EMA_FAST = 21
EMA_SLOW = 55
EMA_TREND = 200
ADX_PERIOD = 14
ADX_THRESHOLD = 25
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_MULT = 1.3
ATR_PERIOD = 14
AM_PROFIT_STEPS = [0.04, 0.09, 0.16]   # 加仓浮盈阈值
AM_STAKE_RATIO = 0.6                     # 金字塔递减比例
CHANDELIER_ATR_MULT = 3.0               # 吊灯止损倍数
HARD_STOPLOSS = -0.18                    # 极端兜底
STARTUP_CANDLES = 220                    # 预热K线数（跳过前N根）

# ── 资金管理 ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1000.0     # USDT
STAKE_AMOUNT = 50.0          # 每次底仓投入（USDT）
FEE_RATE = 0.001             # 0.1% 手续费

DATA_DIR = "E:/source/freqtrade/user_data/data/gate"

COINS = {
    "主流稳定": ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"],
    "中端中型": ["XCN_USDT"],
    "高波动妖币": ["H_USDT", "VELVET_USDT", "BEAT_USDT", "COAI_USDT"],
}
TIMEFRAMES = ["1m", "5m", "15m"]


# ─────────────────────────────────────────────────────────────────────────
# 技术指标（纯 pandas/numpy）
# ─────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    dm_plus[dm_plus < dm_minus] = 0.0
    dm_minus[dm_minus < dm_plus] = 0.0
    atr_s = tr.ewm(span=period, adjust=False).mean()
    eps = 1e-9
    di_p = 100 * dm_plus.ewm(span=period, adjust=False).mean() / (atr_s + eps)
    di_m = 100 * dm_minus.ewm(span=period, adjust=False).mean() / (atr_s + eps)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m + eps)
    return dx.ewm(span=period, adjust=False).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["adx"]  = calc_adx(df["high"], df["low"], df["close"], ADX_PERIOD)
    df["atr"]  = calc_atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["donchian_up"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["donchian_dn"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────
# 交易数据结构
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Entry:
    price: float
    stake: float      # USDT 买入金额（已从 capital 扣除）
    amount: float     # 买入数量（费后）
    time: pd.Timestamp
    entry_n: int      # 1=底仓, 2/3/4=加仓


@dataclass
class Trade:
    pair: str
    entries: List[Entry] = field(default_factory=list)
    max_price: float = 0.0     # 跟踪开仓以来最高价（用于吊灯止损）
    is_open: bool = True
    close_price: float = 0.0
    close_time: Optional[pd.Timestamp] = None
    close_reason: str = ""
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    add_records: List[dict] = field(default_factory=list)
    max_drawdown_pct: float = 0.0   # 开仓期间最大浮亏%（负数）

    @property
    def total_stake(self) -> float:
        return sum(e.stake for e in self.entries)

    @property
    def total_amount(self) -> float:
        return sum(e.amount for e in self.entries)

    @property
    def avg_price(self) -> float:
        ta = self.total_amount
        if ta == 0:
            return 0.0
        return sum(e.price * e.amount for e in self.entries) / ta

    def current_profit_pct(self, price: float) -> float:
        ap = self.avg_price
        return (price / ap - 1) if ap > 0 else 0.0

    def unrealized_pnl(self, price: float) -> float:
        """按当前价格计算持仓浮盈（USDT），含卖出手续费"""
        return self.total_amount * price * (1 - FEE_RATE) - self.total_stake

    def chandelier_stop_price(self, atr_val: float) -> float:
        return self.max_price - atr_val * CHANDELIER_ATR_MULT

    def trend_ok(self, row) -> bool:
        return (row["ema_fast"] > row["ema_slow"]
                and row["ema_slow"] > row["ema_trend"]
                and row["adx"] > ADX_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────
# 核心回测引擎
# ─────────────────────────────────────────────────────────────────────────

def run_backtest(pair: str, timeframe: str) -> dict:
    filepath = f"{DATA_DIR}/{pair}-{timeframe}.feather"
    if not os.path.exists(filepath):
        return {"error": f"文件不存在: {filepath}"}

    df = pd.read_feather(filepath).sort_values("date").reset_index(drop=True)
    df = compute_indicators(df)

    capital = INITIAL_CAPITAL
    equity_curve: list = []
    closed_trades: List[Trade] = []
    active_trade: Optional[Trade] = None
    all_add_records: list = []
    blowup_events: list = []

    for i in range(STARTUP_CANDLES, len(df)):
        row  = df.iloc[i]
        ts   = row["date"]
        price = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0

        # ── 权益计算 ────────────────────────────────────────────────────
        # total_assets = 可用现金 + 持仓市值
        if active_trade:
            holding_value = active_trade.total_amount * price * (1 - FEE_RATE)
            cur_equity = capital + holding_value
            # 更新最高价
            if high > active_trade.max_price:
                active_trade.max_price = high
            # 更新单笔最大浮亏
            low_profit_pct = active_trade.current_profit_pct(low)
            if low_profit_pct < active_trade.max_drawdown_pct:
                active_trade.max_drawdown_pct = low_profit_pct
            # 爆仓风险检测：持仓浮亏超过账户总权益 20%
            unreal = active_trade.unrealized_pnl(price)
            loss_pct = unreal / INITIAL_CAPITAL
            if loss_pct < -0.20:
                blowup_events.append({
                    "time": str(ts),
                    "pair": pair,
                    "tf": timeframe,
                    "price": round(price, 6),
                    "unrealized_loss_pct": round(loss_pct * 100, 2),
                    "total_stake": round(active_trade.total_stake, 2),
                    "n_entries": len(active_trade.entries),
                })
        else:
            cur_equity = capital

        equity_curve.append({"time": str(ts), "equity": round(cur_equity, 4)})

        # ── 持仓管理（退出 + 加仓） ──────────────────────────────────
        if active_trade:
            # 1. 退出检查
            stop_px = active_trade.chandelier_stop_price(atr_val) if atr_val > 0 else 0.0
            hard_stop_px = active_trade.avg_price * (1 + HARD_STOPLOSS)
            exit_reason = None
            exit_price  = price

            if atr_val > 0 and low <= stop_px:
                exit_price  = max(stop_px, low)
                exit_reason = "chandelier_stop"
            elif low <= hard_stop_px:
                exit_price  = hard_stop_px
                exit_reason = "hard_stoploss"
            elif (row["ema_fast"] < row["ema_slow"]
                  or row["close"] < row["donchian_dn"]):
                exit_price  = price
                exit_reason = "trend_reversal"

            if exit_reason:
                sell_value = active_trade.total_amount * exit_price * (1 - FEE_RATE)
                pnl_usdt   = sell_value - active_trade.total_stake
                pnl_pct    = pnl_usdt / active_trade.total_stake * 100
                active_trade.is_open      = False
                active_trade.close_price  = exit_price
                active_trade.close_time   = ts
                active_trade.close_reason = exit_reason
                active_trade.pnl_usdt     = round(pnl_usdt, 4)
                active_trade.pnl_pct      = round(pnl_pct, 4)
                capital += sell_value   # 归还卖出所得
                capital = max(capital, 0.0)
                closed_trades.append(active_trade)
                active_trade = None
                continue   # 当前bar退出，下一bar再检查入场

            # 2. 反马丁加仓
            n_entries = len(active_trade.entries)
            if n_entries <= len(AM_PROFIT_STEPS):
                threshold = AM_PROFIT_STEPS[n_entries - 1]
                cur_profit = active_trade.current_profit_pct(price)
                if cur_profit >= threshold and active_trade.trend_ok(row):
                    base_stake = active_trade.entries[0].stake
                    add_stake  = base_stake * (AM_STAKE_RATIO ** n_entries)
                    add_stake  = min(add_stake, capital * 0.95)
                    if add_stake >= 1.0:
                        add_amount = add_stake * (1 - FEE_RATE) / price
                        capital   -= add_stake
                        new_entry  = Entry(
                            price=price, stake=add_stake,
                            amount=add_amount, time=ts,
                            entry_n=n_entries + 1,
                        )
                        active_trade.entries.append(new_entry)
                        rec = {
                            "time": str(ts),
                            "pair": pair,
                            "tf": timeframe,
                            "entry_n": n_entries + 1,
                            "add_price": round(price, 6),
                            "add_stake_usdt": round(add_stake, 4),
                            "profit_at_add_pct": round(cur_profit * 100, 2),
                            "threshold_pct": round(threshold * 100, 2),
                            "total_stake_usdt": round(active_trade.total_stake, 4),
                            "avg_price": round(active_trade.avg_price, 6),
                        }
                        active_trade.add_records.append(rec)
                        all_add_records.append(rec)

        # ── 入场逻辑 ────────────────────────────────────────────────────
        if active_trade is None:
            # NaN 检查
            if any(np.isnan(row[c]) for c in
                   ["ema_fast","ema_slow","ema_trend","adx","donchian_up","vol_ma"]):
                continue

            entry_signal = (
                row["ema_fast"]  > row["ema_slow"]
                and row["ema_slow"]  > row["ema_trend"]
                and row["adx"]       > ADX_THRESHOLD
                and row["close"]     > row["donchian_up"]
                and row["volume"]    > row["vol_ma"] * VOL_MULT
            )
            if entry_signal:
                stake = min(STAKE_AMOUNT, capital * 0.95)
                if stake >= 1.0:
                    amount = stake * (1 - FEE_RATE) / price
                    capital -= stake
                    entry = Entry(price=price, stake=stake,
                                  amount=amount, time=ts, entry_n=1)
                    active_trade = Trade(pair=pair)
                    active_trade.entries.append(entry)
                    active_trade.max_price = high

    # ── 强制平仓（数据末尾） ─────────────────────────────────────────────
    if active_trade and active_trade.is_open:
        last = df.iloc[-1]
        ep   = float(last["close"])
        sv   = active_trade.total_amount * ep * (1 - FEE_RATE)
        pnl  = sv - active_trade.total_stake
        active_trade.is_open      = False
        active_trade.close_price  = ep
        active_trade.close_time   = last["date"]
        active_trade.close_reason = "end_of_data"
        active_trade.pnl_usdt     = round(pnl, 4)
        active_trade.pnl_pct      = round(pnl / active_trade.total_stake * 100, 4)
        capital += sv
        closed_trades.append(active_trade)

    # ── 绩效统计 ─────────────────────────────────────────────────────────
    n_trades = len(closed_trades)
    wins     = [t for t in closed_trades if t.pnl_usdt > 0]
    losses   = [t for t in closed_trades if t.pnl_usdt <= 0]
    n_win    = len(wins)
    n_loss   = len(losses)

    win_rate  = n_win / n_trades * 100 if n_trades else 0.0
    total_pnl = sum(t.pnl_usdt for t in closed_trades)
    total_ret = total_pnl / INITIAL_CAPITAL * 100

    avg_win  = float(np.mean([t.pnl_usdt for t in wins]))   if wins   else 0.0
    avg_loss = float(np.mean([t.pnl_usdt for t in losses])) if losses else 0.0

    gross_win  = sum(t.pnl_usdt for t in wins)
    gross_loss = abs(sum(t.pnl_usdt for t in losses))
    profit_factor = gross_win / (gross_loss + 1e-9)

    # 最大回撤（资金曲线）
    eq_series = pd.Series([e["equity"] for e in equity_curve], dtype=float)
    roll_max  = eq_series.cummax()
    dd_series = (eq_series - roll_max) / roll_max * 100
    max_dd    = float(dd_series.min())

    # 最恶单笔持仓浮亏
    worst_trade_dd = min((t.max_drawdown_pct * 100 for t in closed_trades), default=0.0)

    # 简化夏普
    rets   = [t.pnl_pct / 100 for t in closed_trades]
    sharpe = (float(np.mean(rets)) / (float(np.std(rets)) + 1e-9) * math.sqrt(max(n_trades,1))
              if n_trades > 1 else 0.0)

    # 退出原因分布
    exit_dist = {}
    for t in closed_trades:
        exit_dist[t.close_reason] = exit_dist.get(t.close_reason, 0) + 1

    metrics = {
        "pair": pair,
        "timeframe": timeframe,
        "n_trades": n_trades,
        "n_win": n_win,
        "n_loss": n_loss,
        "win_rate_pct": round(win_rate, 2),
        "total_pnl_usdt": round(total_pnl, 4),
        "total_return_pct": round(total_ret, 2),
        "final_capital_usdt": round(INITIAL_CAPITAL + total_pnl, 4),
        "avg_win_usdt": round(avg_win, 4),
        "avg_loss_usdt": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "worst_trade_drawdown_pct": round(worst_trade_dd, 2),
        "n_add_orders": len(all_add_records),
        "n_blowup_events": len(blowup_events),
        "max_blowup_loss_pct": round(
            min((b["unrealized_loss_pct"] for b in blowup_events), default=0.0), 2),
        "exit_distribution": exit_dist,
    }

    trades_detail = []
    for t in closed_trades:
        trades_detail.append({
            "open_time":   str(t.entries[0].time),
            "close_time":  str(t.close_time),
            "n_entries":   len(t.entries),
            "avg_price":   round(t.avg_price, 6),
            "close_price": round(t.close_price, 6),
            "total_stake": round(t.total_stake, 4),
            "pnl_usdt":    t.pnl_usdt,
            "pnl_pct":     t.pnl_pct,
            "max_dd_pct":  round(t.max_drawdown_pct * 100, 2),
            "close_reason": t.close_reason,
        })

    return {
        "metrics":      metrics,
        "equity_curve": equity_curve,
        "trades":       trades_detail,
        "add_records":  all_add_records,
        "blowup_events": blowup_events,
    }


# ─────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    coin_group_map = {}
    for group, coins in COINS.items():
        for c in coins:
            coin_group_map[c] = group

    all_coins = [c for coins in COINS.values() for c in coins]
    total = len(all_coins) * len(TIMEFRAMES)
    done  = 0

    for pair in all_coins:
        all_results[pair] = {}
        for tf in TIMEFRAMES:
            done += 1
            result = run_backtest(pair, tf)
            all_results[pair][tf] = result
            if "error" in result:
                print(f"[{done:02d}/{total}] {pair} @{tf}: ERROR {result['error']}")
            else:
                m = result["metrics"]
                print(
                    f"[{done:02d}/{total}] {pair} @{tf}: "
                    f"trades={m['n_trades']} win={m['win_rate_pct']}% "
                    f"ret={m['total_return_pct']}% mdd={m['max_drawdown_pct']}% "
                    f"adds={m['n_add_orders']} blowup={m['n_blowup_events']}"
                )

    out_json = "E:/source/freqtrade/anti_martingale_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"results": all_results, "coin_group_map": coin_group_map},
            f, ensure_ascii=False, indent=2
        )
    print(f"\n[OK] JSON saved: {out_json}")
    return all_results, coin_group_map


if __name__ == "__main__":
    results, group_map = main()
