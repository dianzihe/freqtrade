# -*- coding: utf-8 -*-
"""
星河量化策略 · 全币种全周期批量回测脚本 (独立引擎版)
======================================================
绕过 freqtrade 内部引擎的复杂初始化依赖，直接用 pandas 实现策略逻辑回测。

回测矩阵: 9 币种 × 3 K线周期 = 27 组独立回测
输出: HTML 报告 + JSON 结果 + 每组合盈亏曲线图

用法:
    .venv/Scripts/python batch_backtest_xinghe.py
"""

import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# 路径配置
# ==============================================================================
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "user_data" / "data" / "gate"
OUTPUT_DIR = PROJECT_DIR / "user_data" / "backtest_results" / "xinghe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_CATEGORIES = {
    "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中端中型币种": ["XCN/USDT"],
    "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

TIMEFRAMES = ["1m", "5m", "15m"]
TIMERANGE_START = "2026-06-24"
TIMERANGE_END = "2026-06-29"

# ==============================================================================
# 策略参数 (与 GateXingheGridProStrategy 默认值完全一致)
# ==============================================================================
PARAMS = {
    # 趋势/指标
    "ema_fast_period": 20,
    "ema_slow_period": 60,
    "atr_period": 14,
    "volatility_period": 20,
    "vote_threshold": 2,           # 三票共识门槛: 2票同向即开仓

    # ATR 可交易区间
    "min_atr_pct": 0.0006,         # 死水下界
    "max_atr_pct": 0.055,          # 极端行情上界

    # 网格步长 (影响 DCA 补仓密度)
    "min_grid_step_pct": 0.006,
    "max_grid_step_pct": 0.060,
    "atr_grid_multiplier": 2.0,
    "grid_growth_per_entry": 0.25,

    # 出场
    "tp_roi": 0.025,               # 止盈目标
    "deep_loss_fuse": -0.22,       # 中断平仓熔丝
    "time_stop_candles": 240,      # 时间止损 (K线数)

    # 仓位
    "stake_amount": 200,           # 单笔保证金 (USDT)
    "dry_run_wallet": 2000,        # 初始权益
    "max_open_trades": 3,          # 最大同时持仓
    "max_entry_position_adjustment": 5,  # DCA 最大补仓次数
    "dca_multipliers": [1.3, 1.6, 2.0, 2.4, 3.0],

    # 手续费
    "fee": 0.001,                  # 0.1% (现货)

    # 预热
    "startup_candle_count": 240,
}

# ==============================================================================
# 指标计算 (与策略 populate_indicators 一致)
# ==============================================================================

def compute_atr_pct(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR 占价格百分比 (EWM Wilder 平滑)"""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return (atr / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """计算所有策略指标 (完全对应 populate_indicators)"""
    ema_f = params["ema_fast_period"]
    ema_s = params["ema_slow_period"]
    atr_p = params["atr_period"]
    vol_p = params["volatility_period"]

    df["ema_fast"] = df["close"].ewm(span=ema_f, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_s, adjust=False).mean()
    df["atr_pct"] = compute_atr_pct(df, atr_p)
    df["volatility_pct"] = df["close"].pct_change().rolling(vol_p).std().fillna(0)
    df["vol_baseline"] = df["volatility_pct"].rolling(60).mean()

    # 三票制
    df["ema_vote"] = np.select(
        [df["ema_fast"] > df["ema_slow"], df["ema_fast"] < df["ema_slow"]],
        [1, -1], default=0)

    df["atr_vote"] = np.select([
        (df["atr_pct"] > df["atr_pct"].shift(1)) & (df["close"] > df["open"]),
        (df["atr_pct"] > df["atr_pct"].shift(1)) & (df["close"] < df["open"]),
    ], [1, -1], default=0)

    df["volatility_vote"] = np.select([
        (df["volatility_pct"] > df["vol_baseline"]) & (df["close"] > df["ema_fast"]),
        (df["volatility_pct"] > df["vol_baseline"]) & (df["close"] < df["ema_fast"]),
    ], [1, -1], default=0)

    vote_sum = df["ema_vote"] + df["atr_vote"] + df["volatility_vote"]
    th = params["vote_threshold"]
    df["trend_signal"] = np.select([vote_sum >= th, vote_sum <= -th], [1, -1], default=0)

    # 风险保险丝
    df["risk_fuse"] = (
        (df["atr_pct"] < params["min_atr_pct"]) | (df["atr_pct"] > params["max_atr_pct"])
    ).astype(int)

    # 网格步长 (基于实时 ATR)
    raw_step = df["atr_pct"] * params["atr_grid_multiplier"]
    df["grid_step_pct"] = raw_step.clip(
        lower=params["min_grid_step_pct"], upper=params["max_grid_step_pct"])

    return df


# ==============================================================================
# 回测引擎
# ==============================================================================

@pd.api.extensions.register_dataframe_accessor("bt")
class BacktestAccessor:
    """DataFrame 回测访问器: df.bt.simulate(pair, name, params)"""

    def __init__(self, df):
        self._df = df

    def simulate(self, pair: str, params: dict) -> dict:
        """运行回测模拟，返回结果字典"""
        df = self._df.copy()
        df = compute_indicators(df, params)

        wallet = params["dry_run_wallet"]
        stake_amount = params["stake_amount"]
        max_open = params["max_open_trades"]
        fee = params["fee"]
        tp_roi = params["tp_roi"]
        deep_loss_fuse = params["deep_loss_fuse"]
        time_stop = params["time_stop_candles"]
        max_dca = params["max_entry_position_adjustment"]
        dca_mults = params["dca_multipliers"]
        startup = params["startup_candle_count"]

        # ---- 交易状态 ----
        positions: List[dict] = []          # 当前持仓列表
        closed_trades: List[dict] = []       # 已平仓记录
        equity_curve = []                    # 权益曲线
        dca_events = []                      # DCA 触发记录

        peak_equity = wallet
        max_drawdown = 0.0
        max_floating_dd = 0.0

        # ---- 遍历每根 K 线 ----
        for i in range(startup, len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            price = row["close"]
            high = row["high"]
            low = row["low"]
            trend = row["trend_signal"]
            fuse = row["risk_fuse"]
            atr_pct = row["atr_pct"]
            grid_step = row["grid_step_pct"]

            # ===== A. 检查现有持仓的出场条件 =====
            to_close = []
            for p_idx, pos in enumerate(positions):
                pos["bars_held"] += 1
                # 当前浮盈 (按 close 计算)
                if pos["side"] == "long":
                    pos["current_profit"] = (price - pos["avg_entry"]) / pos["avg_entry"]
                else:
                    pos["current_profit"] = (pos["avg_entry"] - price) / pos["avg_entry"]

                # 止盈检查 (用 high/low 检测触发)
                tp_triggered = False
                sl_triggered = False
                exit_price = price
                exit_reason = ""

                if pos["side"] == "long":
                    tp_price = pos["avg_entry"] * (1 + tp_roi)
                    sl_price = pos["avg_entry"] * (1 + deep_loss_fuse)
                    if high >= tp_price:
                        tp_triggered = True
                        exit_price = tp_price
                        exit_reason = "tp_roi"
                    elif low <= sl_price:
                        sl_triggered = True
                        exit_price = sl_price
                        exit_reason = "deep_loss_fuse"
                else:
                    tp_price = pos["avg_entry"] * (1 - tp_roi)
                    sl_price = pos["avg_entry"] * (1 - deep_loss_fuse)
                    if low <= tp_price:
                        tp_triggered = True
                        exit_price = tp_price
                        exit_reason = "tp_roi"
                    elif high >= sl_price:
                        sl_triggered = True
                        exit_price = sl_price
                        exit_reason = "deep_loss_fuse"

                # 信号反转 / 风险熔丝
                if not tp_triggered and not sl_triggered:
                    if pos["side"] == "long" and (trend == -1 or fuse == 1):
                        exit_price = price
                        exit_reason = "trend_reversal_or_risk"
                    elif pos["side"] == "short" and (trend == 1 or fuse == 1):
                        exit_price = price
                        exit_reason = "trend_reversal_or_risk"
                    # 时间止损
                    elif pos["bars_held"] >= time_stop and pos["current_profit"] <= 0:
                        exit_price = price
                        exit_reason = "time_stop"

                if exit_reason:
                    # 计算出场
                    pos["exit_price"] = exit_price
                    pos["exit_time"] = current_time
                    pos["exit_reason"] = exit_reason
                    if pos["side"] == "long":
                        profit = (exit_price - pos["avg_entry"]) / pos["avg_entry"]
                    else:
                        profit = (pos["avg_entry"] - exit_price) / pos["avg_entry"]
                    profit -= fee * 2  # 进出各一次手续费
                    pos["profit_ratio"] = profit
                    pos["profit_abs"] = pos["total_stake"] * profit
                    wallet += pos["profit_abs"]
                    closed_trades.append(pos)
                    to_close.append(p_idx)
                    # DCA 未补完的记录
                    if pos.get("dca_count", 0) > 0:
                        dca_events.append({
                            "open_time": pos["open_time"],
                            "dca_count": pos["dca_count"],
                            "exit_reason": exit_reason,
                            "final_profit": profit,
                        })

            # 移除已平仓 (倒序)
            for p_idx in sorted(to_close, reverse=True):
                positions.pop(p_idx)

            # ===== B. 检查 DCA 加仓 =====
            for pos in positions:
                if pos["dca_count"] >= max_dca:
                    continue
                # 检查浮亏区间
                if pos["current_profit"] >= 0 or pos["current_profit"] <= deep_loss_fuse:
                    continue
                # 逆势幅度
                if pos["side"] == "long":
                    adverse = (pos["avg_entry"] - price) / pos["avg_entry"]
                else:
                    adverse = (price - pos["avg_entry"]) / pos["avg_entry"]
                if adverse <= 0:
                    continue
                # 网格阈值 (随 DCA 次数递增)
                entry_count = pos["dca_count"]
                dca_threshold = grid_step * (1 + entry_count * params["grid_growth_per_entry"])
                dca_threshold = max(params["min_grid_step_pct"], min(dca_threshold, params["max_grid_step_pct"]))
                if adverse < dca_threshold:
                    continue
                # 执行 DCA
                idx = min(entry_count, len(dca_mults) - 1)
                dca_stake = stake_amount * dca_mults[idx]
                dca_amount = dca_stake / price
                if pos["side"] == "long":
                    total_cost = pos["total_stake"] + dca_stake
                    pos["avg_entry"] = total_cost / (pos["amount"] + dca_amount)
                else:
                    total_cost = pos["total_stake"] + dca_stake
                    pos["avg_entry"] = total_cost / (pos["amount"] + dca_amount)
                pos["amount"] += dca_amount
                pos["total_stake"] += dca_stake
                pos["dca_count"] += 1
                pos["dca_times"].append(current_time)
                dca_events.append({
                    "open_time": pos["open_time"],
                    "dca_count": pos["dca_count"],
                    "adverse": round(adverse, 6),
                    "threshold": round(dca_threshold, 6),
                    "stake": round(dca_stake, 2),
                })

            # ===== C. 检查开仓信号 =====
            if len(positions) >= max_open:
                continue
            if fuse == 1:
                continue
            if trend not in [1, -1]:
                continue

            # 做多信号 (现货模式仅做多)
            if trend == 1:
                amount = stake_amount / price
                pos = {
                    "open_time": current_time,
                    "open_price": price,
                    "side": "long",
                    "amount": amount,
                    "avg_entry": price,
                    "total_stake": stake_amount,
                    "bars_held": 0,
                    "current_profit": 0.0,
                    "dca_count": 0,
                    "dca_times": [],
                }
                positions.append(pos)

            # ===== D. 更新权益曲线 =====
            equity = wallet + sum(
                p["total_stake"] * (1 + p["current_profit"]) for p in positions
            )
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
            if dd > max_floating_dd:
                max_floating_dd = dd
            equity_curve.append({
                "time": current_time,
                "equity": equity,
                "drawdown": dd,
            })

        # ---- 强制平仓未平仓位 ----
        if positions and len(df) > startup:
            last_price = df.iloc[-1]["close"]
            last_time = df.index[-1]
            for pos in positions:
                if pos["side"] == "long":
                    profit = (last_price - pos["avg_entry"]) / pos["avg_entry"]
                else:
                    profit = (pos["avg_entry"] - last_price) / pos["avg_entry"]
                profit -= fee * 2
                pos["profit_ratio"] = profit
                pos["profit_abs"] = pos["total_stake"] * profit
                pos["exit_price"] = last_price
                pos["exit_time"] = last_time
                pos["exit_reason"] = "force_close_eod"
                wallet += pos["profit_abs"]
                closed_trades.append(pos)

        # ---- 汇总统计 ----
        n_trades = len(closed_trades)
        wins = [t for t in closed_trades if t["profit_ratio"] > 0]
        losses = [t for t in closed_trades if t["profit_ratio"] <= 0]
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

        total_profit_pct = (wallet - params["dry_run_wallet"]) / params["dry_run_wallet"] * 100
        avg_profit = np.mean([t["profit_ratio"] for t in closed_trades]) * 100 if n_trades > 0 else 0
        best = max([t["profit_ratio"] for t in closed_trades]) * 100 if n_trades > 0 else 0
        worst = min([t["profit_ratio"] for t in closed_trades]) * 100 if n_trades > 0 else 0

        # 盈亏比
        gross_profit = sum(t["profit_abs"] for t in wins)
        gross_loss = abs(sum(t["profit_abs"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

        # 爆仓风险评估: 单笔亏损>50% 的次数
        liq_risk_score = sum(1 for t in losses if t["profit_ratio"] < -0.50)
        liquidation_trades = [t for t in losses if t["profit_ratio"] < -0.50]

        # DCA 统计
        total_dca = len(dca_events)
        avg_dca_per_trade = total_dca / n_trades if n_trades > 0 else 0

        # 最大持仓时长
        holding_bars = [t["bars_held"] for t in closed_trades]

        return {
            "pair": pair,
            "n_trades": n_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "total_profit_pct": round(total_profit_pct, 2),
            "total_profit_abs": round(wallet - params["dry_run_wallet"], 2),
            "final_wallet": round(wallet, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "max_floating_dd_pct": round(max_floating_dd * 100, 2),
            "avg_profit_pct": round(avg_profit, 2),
            "best_trade_pct": round(best, 2),
            "worst_trade_pct": round(worst, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 99.99,
            "dca_total": total_dca,
            "dca_avg_per_trade": round(avg_dca_per_trade, 2),
            "liquidation_risk_score": liq_risk_score,
            "liquidation_trades_count": len(liquidation_trades),
            "liquidation_risk_level": (
                "极高风险" if liq_risk_score >= 3 else
                "高风险" if liq_risk_score >= 2 else
                "中等风险" if liq_risk_score >= 1 else
                "安全"
            ),
            "avg_holding_bars": round(np.mean(holding_bars), 1) if holding_bars else 0,
            "max_holding_bars": max(holding_bars) if holding_bars else 0,
            "equity_curve": equity_curve,
            "closed_trades": closed_trades,
            "dca_events": dca_events,
        }


# ==============================================================================
# HTML 报告生成 (含 Chart.js 盈亏曲线图)
# ==============================================================================

def generate_html_report(all_results: List[dict]) -> str:
    html_path = OUTPUT_DIR / "xinghe_backtest_report.html"

    # 构建 chart.js 数据 JSON
    chart_data = {}
    for r in all_results:
        if not r.get("success"):
            continue
        key = f"{r['pair'].replace('/', '_')}_{r['timeframe']}"
        eq = r.get("equity_curve", [])
        if eq:
            chart_data[key] = {
                "pair": r["pair"],
                "timeframe": r["timeframe"],
                "category": r["category"],
                "timeline": [e["time"].strftime("%m-%d %H:%M") if hasattr(e["time"], "strftime") else str(e["time"]) for e in eq],
                "equity": [e["equity"] for e in eq],
                "drawdown": [e["drawdown"] * 100 for e in eq],
            }

    chart_json = json.dumps(chart_data, ensure_ascii=False)

    # 构建表格行
    rows_html = ""
    for r in all_results:
        cat = r.get("category", "N/A")
        pair = r.get("pair", "N/A")
        tf = r.get("timeframe", "N/A")
        success = r.get("success", False)

        if not success:
            rows_html += f"""
            <tr class="error-row">
                <td>{cat}</td><td>{pair}</td><td>{tf}</td>
                <td colspan="16" class="error-cell">{r.get('error', '未知')}</td>
            </tr>"""
            continue

        n_trades = r.get("n_trades", 0)
        wins = r.get("wins", 0)
        losses = r.get("losses", 0)
        wr = r.get("win_rate", 0)
        profit_pct = r.get("total_profit_pct", 0)
        dd_pct = r.get("max_drawdown_pct", 0)
        float_dd = r.get("max_floating_dd_pct", 0)
        pf = r.get("profit_factor", 0)
        best = r.get("best_trade_pct", 0)
        worst = r.get("worst_trade_pct", 0)
        dca = r.get("dca_total", 0)
        liq = r.get("liquidation_risk_level", "N/A")
        avg_hold = r.get("avg_holding_bars", 0)

        profit_color = "#ff4444" if profit_pct >= 0 else "#00aa44"
        wr_color = "#ff4444" if wr >= 50 else "#00aa44"
        dd_color = "#00aa44" if dd_pct < 20 else ("#ffaa00" if dd_pct < 40 else "#ff4444")

        rows_html += f"""
        <tr>
            <td class="cat-cell">{cat}</td>
            <td class="pair-cell">{pair}</td>
            <td class="tf-cell">{tf}</td>
            <td class="num-cell">{n_trades}</td>
            <td class="num-cell">{wins}</td>
            <td class="num-cell">{losses}</td>
            <td class="num-cell" style="color:{wr_color}">{wr:.1f}%</td>
            <td class="num-cell" style="color:{profit_color}">{profit_pct:+.2f}%</td>
            <td class="num-cell" style="color:{dd_color}">{dd_pct:.1f}%</td>
            <td class="num-cell">{float_dd:.1f}%</td>
            <td class="num-cell">{pf:.2f}</td>
            <td class="num-cell" style="color:{'#ff4444' if best>=0 else '#00aa44'}">{best:+.2f}%</td>
            <td class="num-cell" style="color:{'#00aa44' if worst>=0 else '#ff4444'}">{worst:+.2f}%</td>
            <td class="num-cell">{dca}</td>
            <td class="num-cell">{avg_hold:.0f}</td>
            <td class="num-cell risk-{'high' if '高' in liq else 'low'}">{liq}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>星河量化 · 全币种全周期批量回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; color: #ff6b6b; margin-bottom: 8px; font-size: 28px; }}
h2 {{ color: #ffd93d; margin: 30px 0 15px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
h3 {{ color: #4ecdc4; margin: 20px 0 10px; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 8px; }}
.meta {{ text-align: center; color: #666; font-size: 13px; margin-bottom: 30px; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }}
th {{ background: #16213e; color: #ffd93d; padding: 10px 4px; text-align: center; border: 1px solid #333; position: sticky; top: 0; white-space: nowrap; }}
td {{ padding: 7px 4px; text-align: center; border: 1px solid #333; }}
tr:hover {{ background: #16213e55; }}
.cat-cell {{ text-align: left; font-weight: bold; color: #ccc; }}
.pair-cell {{ font-weight: bold; color: #4ecdc4; }}
.tf-cell {{ color: #ffd93d; }}
.num-cell {{ font-family: 'Consolas', monospace; font-size: 12px; }}
.error-row {{ background: #3a1a1a; }}
.error-cell {{ color: #ff4444; }}
.risk-high {{ color: #ff4444; font-weight: bold; }}
.risk-low {{ color: #00aa44; }}
.summary-box {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.stat-card {{ background: #16213e; border-radius: 8px; padding: 14px; text-align: center; border: 1px solid #333; }}
.stat-card .label {{ color: #888; font-size: 12px; margin-bottom: 6px; }}
.stat-card .value {{ font-size: 22px; font-weight: bold; }}
.note {{ background: #16213e; border-left: 3px solid #ffd93d; padding: 12px 16px; margin: 16px 0; border-radius: 4px; font-size: 13px; color: #ccc; }}
.footer {{ text-align: center; color: #555; margin-top: 40px; font-size: 12px; }}
.chart-container {{ width: 100%; max-width: 900px; margin: 20px auto; background: #16213e; border-radius: 8px; padding: 16px; border: 1px solid #333; }}
canvas {{ width: 100% !important; }}
.chart-selector {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
.chart-btn {{ padding: 6px 14px; border: 1px solid #444; border-radius: 4px; background: #1a1a2e; color: #ccc; cursor: pointer; font-size: 12px; }}
.chart-btn:hover {{ background: #333; }}
.chart-btn.active {{ background: #ff6b6b; color: #fff; border-color: #ff6b6b; }}
.tab-container {{ margin: 20px 0; }}
.tab-buttons {{ display: flex; gap: 4px; margin-bottom: 0; }}
.tab-btn {{ padding: 8px 20px; border: 1px solid #333; border-bottom: none; border-radius: 6px 6px 0 0; background: #16213e; color: #888; cursor: pointer; font-size: 13px; }}
.tab-btn.active {{ background: #1a1a2e; color: #ffd93d; font-weight: bold; }}
.tab-content {{ display: none; background: #1a1a2e; border: 1px solid #333; border-radius: 0 6px 6px 6px; padding: 15px; overflow-x: auto; }}
.tab-content.active {{ display: block; }}
.legend {{ display: flex; gap: 15px; justify-content: center; margin-bottom: 10px; font-size: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.collapsible {{ cursor: pointer; }}
.collapsible:hover {{ color: #ffd93d; }}
</style>
</head>
<body>

<h1>星河量化策略 · 批量回测报告</h1>
<div class="subtitle">三票制混合趋势 + ATR 自适应网格 DCA · 现货模式 (仅做多)</div>
<div class="meta">
    生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
    区间: {TIMERANGE_START} ~ {TIMERANGE_END} |
    初始: {PARAMS['dry_run_wallet']} USDT | 仓位: {PARAMS['stake_amount']} USDT |
    总组数: {len(all_results)}
</div>

<div class="note">
    <strong>策略配置:</strong>
    ema_fast={PARAMS['ema_fast_period']}, ema_slow={PARAMS['ema_slow_period']}, atr_period={PARAMS['atr_period']},
    vote_threshold={PARAMS['vote_threshold']}, min_atr_pct={PARAMS['min_atr_pct']}, max_atr_pct={PARAMS['max_atr_pct']},
    tp_roi={PARAMS['tp_roi']}, deep_loss_fuse={PARAMS['deep_loss_fuse']},
    DCA max={PARAMS['max_entry_position_adjustment']}次, 手续费={PARAMS['fee']*100}%.
    <br>
    <strong>说明:</strong> 原策略为 Gate 永续合约设计(含做空/杠杆/OCO括号单)，当前使用现货数据回测仅测试做多信号。
    策略核心(三票制趋势判断 + ATR自适应DCA补仓 + 风控熔断)在现货模式全部保留。
</div>

<h2>一、全局汇总</h2>
<div class="summary-box">
    <div class="stat-card"><div class="label">测试组数</div><div class="value" style="color:#ffd93d">{len(all_results)}</div></div>
    <div class="stat-card"><div class="label">成功组数</div><div class="value" style="color:#00aa44">{sum(1 for r in all_results if r.get('success'))}</div></div>
    <div class="stat-card"><div class="label">总交易次数</div><div class="value" style="color:#4ecdc4">{sum(r.get('n_trades', 0) for r in all_results)}</div></div>
    <div class="stat-card"><div class="label">总DCA次数</div><div class="value" style="color:#ffd93d">{sum(r.get('dca_total', 0) for r in all_results)}</div></div>
    <div class="stat-card"><div class="label">总盈亏</div><div class="value" style="color:{'#ff4444' if sum(r.get('total_profit_abs', 0) for r in all_results) >= 0 else '#00aa44'}">{sum(r.get('total_profit_abs', 0) for r in all_results):+.0f} USDT</div></div>
</div>

<h2>二、盈亏曲线图</h2>
<div class="chart-selector" id="chartSelector"></div>
<div class="chart-container">
    <canvas id="equityChart" height="300"></canvas>
</div>
<div class="legend" id="chartLegend"></div>

<h2>三、回测明细 ({len(all_results)} 组)</h2>
<div class="tab-container">
<div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab('all')">全部</button>
    <button class="tab-btn" onclick="switchTab('stable')">稳定主流币</button>
    <button class="tab-btn" onclick="switchTab('mid')">中端中型币种</button>
    <button class="tab-btn" onclick="switchTab('meme')">高波动妖币</button>
</div>
<div class="tab-content active" id="tab-all">
<table>
<thead><tr>
    <th>分类</th><th>币种</th><th>周期</th><th>交易</th><th>盈利</th><th>亏损</th><th>胜率</th>
    <th>总盈亏</th><th>回撤</th><th>浮亏</th><th>盈亏比</th><th>最佳</th><th>最差</th>
    <th>DCA</th><th>均持K</th><th>爆仓风险</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<div class="tab-content" id="tab-stable"></div>
<div class="tab-content" id="tab-mid"></div>
<div class="tab-content" id="tab-meme"></div>
</div>

<div class="footer">星河量化批量回测系统 · 自建引擎 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<script>
// Chart.js 盈亏曲线
const allData = {chart_json};
let activeChart = null;

// 构建选择器
const selector = document.getElementById('chartSelector');
const keys = Object.keys(allData);
keys.forEach((key, idx) => {{
    const d = allData[key];
    const btn = document.createElement('button');
    btn.className = 'chart-btn' + (idx === 0 ? ' active' : '');
    btn.textContent = d.pair + ' @ ' + d.timeframe;
    btn.onclick = () => {{
        document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        drawChart(d);
    }};
    selector.appendChild(btn);
}});

function drawChart(d) {{
    if (activeChart) activeChart.destroy();
    const ctx = document.getElementById('equityChart').getContext('2d');
    const timeline = d.timeline;
    // 抽样显示标签
    const step = Math.max(1, Math.floor(timeline.length / 15));
    activeChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: timeline,
            datasets: [
                {{
                    label: '权益 (USDT)',
                    data: d.equity,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255,107,107,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y',
                }},
                {{
                    label: '回撤 (%)',
                    data: d.drawdown,
                    borderColor: '#4ecdc4',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y1',
                }}
            ]
        }},
        options: {{
            responsive: true,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ labels: {{ color: '#ccc' }} }},
                title: {{ display: true, text: d.pair + ' @ ' + d.timeframe + ' (' + d.category + ')', color: '#ffd93d' }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#888', maxTicksLimit: 15, callback: function(v,i) {{ return i % step === 0 ? this.getLabelForValue(v) : ''; }} }} }},
                y: {{
                    type: 'linear', position: 'left',
                    title: {{ display: true, text: '权益 (USDT)', color: '#ff6b6b' }},
                    ticks: {{ color: '#ff6b6b' }}, grid: {{ color: '#333' }}
                }},
                y1: {{
                    type: 'linear', position: 'right',
                    title: {{ display: true, text: '回撤 (%)', color: '#4ecdc4' }},
                    ticks: {{ color: '#4ecdc4' }}, grid: {{ display: false }},
                    min: Math.min(...d.drawdown) - 2,
                    max: 0,
                }}
            }}
        }}
    }});
    document.getElementById('chartLegend').innerHTML =
        '<div class="legend-item"><span class="legend-dot" style="background:#ff6b6b"></span>权益</div>' +
        '<div class="legend-item"><span class="legend-dot" style="background:#4ecdc4"></span>回撤</div>';
}}

// 初始绘制
if (keys.length > 0) drawChart(allData[keys[0]]);

// Tab 切换
function switchTab(tab) {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    const target = document.getElementById('tab-' + tab);
    if (target) {{
        if (tab === 'all') {{
            target.innerHTML = document.getElementById('tab-all').innerHTML;
        }} else {{
            // 筛选对应分类的行
            const catMap = {{'stable': '稳定主流币', 'mid': '中端中型币种', 'meme': '高波动妖币'}};
            const cat = catMap[tab];
            const table = document.getElementById('tab-all').querySelector('table').cloneNode(true);
            const rows = table.querySelectorAll('tbody tr');
            let visible = 0;
            rows.forEach(r => {{
                const firstCell = r.querySelector('.cat-cell');
                if (firstCell && firstCell.textContent !== cat && !r.classList.contains('error-row')) {{
                    r.style.display = 'none';
                }} else {{
                    visible++;
                }}
            }});
            target.innerHTML = '';
            target.appendChild(table);
        }}
        target.classList.add('active');
    }}
}}
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return str(html_path)


# ==============================================================================
# 主流程
# ==============================================================================

def run_single(pair: str, timeframe: str, params: dict) -> dict:
    """运行单个回测"""
    pair_safe = pair.replace("/", "_")
    data_path = DATA_DIR / f"{pair_safe}-{timeframe}.feather"

    if not data_path.exists():
        return {"success": False, "pair": pair, "timeframe": timeframe,
                "error": f"数据文件不存在"}

    try:
        df = pd.read_feather(data_path)

        # 设置时间索引
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")

        # 裁剪时间范围
        start = pd.Timestamp(TIMERANGE_START, tz="utc")
        end = pd.Timestamp(TIMERANGE_END, tz="utc")
        df = df[(df.index >= start) & (df.index <= end)]

        # 确保必要列存在并标准化列名
        needed_cols = ["open", "high", "low", "close", "volume"]
        df.columns = [c.lower() for c in df.columns]
        for col in needed_cols:
            if col not in df.columns:
                return {"success": False, "pair": pair, "timeframe": timeframe,
                        "error": f"缺少列: {col}"}

        if len(df) < params["startup_candle_count"] + 20:
            return {"success": False, "pair": pair, "timeframe": timeframe,
                    "error": f"数据不足 {len(df)} bars (需要>{params['startup_candle_count']})"}

        # 运行回测
        result = df.bt.simulate(pair, params)
        result["success"] = True
        result["timeframe"] = timeframe
        result["data_bars"] = len(df)
        return result

    except Exception as e:
        import traceback
        return {"success": False, "pair": pair, "timeframe": timeframe,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    print("=" * 70)
    print("  星河量化 · 全币种全周期批量回测 (独立引擎)")
    print("=" * 70)
    print(f"  区间: {TIMERANGE_START} ~ {TIMERANGE_END}")
    print(f"  资金: {PARAMS['dry_run_wallet']} USDT | 仓位: {PARAMS['stake_amount']} USDT")
    total_runs = sum(len(v) for v in PAIR_CATEGORIES.values()) * len(TIMEFRAMES)
    print(f"  总计: {total_runs} 组 (9币种 × 3周期)")
    print("=" * 70)

    all_results = []
    current = 0

    for cat_name, pairs in PAIR_CATEGORIES.items():
        print(f"\n{'=' * 55}")
        print(f"  ▸ {cat_name} ({len(pairs)} 币种)")
        print(f"{'=' * 55}")
        for pair in pairs:
            for tf in TIMEFRAMES:
                current += 1
                print(f"  [{current}/{total_runs}] {pair} @ {tf} ... ", end="", flush=True)

                result = run_single(pair, tf, PARAMS.copy())
                result["category"] = cat_name

                if result["success"]:
                    n = result["n_trades"]
                    pnl = result["total_profit_pct"]
                    wr = result["win_rate"]
                    dca = result["dca_total"]
                    print(f"✓ {n}笔 盈亏{pnl:+.2f}% 胜率{wr:.1f}% DCA×{dca}")
                else:
                    print(f"✗ {result.get('error', '失败')}")

                all_results.append(result)

    # 保存 JSON (不含 equity_curve 和 trades 以减少体积)
    json_path = OUTPUT_DIR / "all_results.json"
    slim_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ("equity_curve", "closed_trades", "dca_events")}
        slim_results.append(sr)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON (摘要): {json_path}")

    # 保存完整结果(含曲线和交易数据)
    full_json_path = OUTPUT_DIR / "all_results_full.json"
    # 清理不可序列化的时间戳
    for r in all_results:
        if "equity_curve" in r:
            for e in r["equity_curve"]:
                if hasattr(e.get("time"), "isoformat"):
                    e["time"] = e["time"].isoformat()
        if "closed_trades" in r:
            for t in r["closed_trades"]:
                for k in ("open_time", "exit_time"):
                    if k in t and hasattr(t[k], "isoformat"):
                        t[k] = t[k].isoformat()
                if "dca_times" in t:
                    t["dca_times"] = [dt.isoformat() if hasattr(dt, "isoformat") else str(dt) for dt in t["dca_times"]]
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON (完整): {full_json_path}")

    # 生成 HTML
    html_path = generate_html_report(all_results)
    print(f"  HTML: {html_path}")

    success = sum(1 for r in all_results if r["success"])
    print(f"\n  完成: {success}/{total_runs} 组成功")
    print("=" * 70)
    return html_path


if __name__ == "__main__":
    main()
