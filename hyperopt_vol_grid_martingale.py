# -*- coding: utf-8 -*-
"""
Meme 波动率网格马丁策略 · 全币种全周期 Hyperopt 批量优化 + 回测脚本
======================================================================
基于 MemeVolatilityGridMartingaleStrategy 完整参数空间, 采用随机搜索+回测评估。
覆盖 9 币种 × 3 K线周期 = 27 组独立优化, 每组输出最优参数 + 绩效报告。

核心逻辑:
  1. 指标计算 → 与策略 populate_indicators 完全一致
  2. 入场 → 布林下轨超卖 + RSI超卖 + ATR波动率过滤 + 区间(0.12~0.72) + 趋势上
  3. 出场 → 回归布林中轨 / RSI转强
  4. 马丁加仓 → 阶梯式浮亏加仓 + 冷却 + ATR爆炸禁加
  5. 中断平仓 → 加仓耗尽后继续深跌 → 市价砍仓
  6. 时间止损 → 超时未达预期 → 强制离场
  7. 动态止盈 → 移动止盈锁利

与 MemeLimitedDcaMartingaleStrategy 的关键区别:
  - 无 buy_range_pos_min 参数 (区间下限固定 0.12, 上限固定 0.72)
  - 入场额外要求 atr_pct > buy_atr_pct_min (波动率不能太低)
  - 出场: close > bb_mid 或 rsi > sell_rsi_min (OR 逻辑)

用法:
    .venv/Scripts/python hyperopt_vol_grid_martingale.py

输出目录: user_data/backtest_results/vol_grid_martingale/
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

# Windows 控制台 UTF-8 编码修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ==============================================================================
# 路径配置
# ==============================================================================
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "user_data" / "data" / "gate"
OUTPUT_DIR = PROJECT_DIR / "user_data" / "backtest_results" / "vol_grid_martingale"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 币种 × 周期矩阵
# ==============================================================================
PAIR_CATEGORIES = {
    "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中端中型币种": ["XCN/USDT"],
    "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

TIMEFRAMES = ["1m", "5m", "15m"]

# 数据时间范围
TIMERANGE_START = "2026-06-24"
TIMERANGE_END = "2026-06-29"

# ==============================================================================
# 策略固定参数 (不参与优化)
# ==============================================================================
FIXED_PARAMS = {
    "stake_amount": 200,           # 单笔首仓保证金 (USDT)
    "dry_run_wallet": 2000,        # 初始权益
    "max_open_trades": 3,          # 最大同时持仓
    "fee": 0.001,                  # 手续费 0.1%
    "atr_spike_block": 3.0,        # ATR 爆炸禁加倍数
    "dca_interrupt_extra_loss": 0.08,  # 加仓耗尽后再亏N → 中断
    "ema_slope_block": 0.01,       # 趋势过滤斜率阈值
    "startup_candle_count": 240,   # 预热
    # 指示器固定参数
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "rsi_period": 14,
    "range_period": 48,
    "ema_slow_period": 100,
    "ema_slope_diff": 5,
    # 波动率网格特有: 区间位置固定范围
    "range_pos_min": 0.12,
    "range_pos_max": 0.72,
}

# ==============================================================================
# Hyperopt 参数空间 (8 参数, 不含 buy_range_pos_min)
# ==============================================================================
PARAM_SPACE = {
    "buy_atr_pct_min":     {"type": "float", "low": 0.003, "high": 0.020, "decimals": 4},
    "buy_rsi_max":          {"type": "int",   "low": 20,   "high": 45},
    "dca_max_entries":      {"type": "int",   "low": 1,    "high": 4},
    "dca_step_pct":         {"type": "float", "low": 0.04,  "high": 0.12,  "decimals": 3},
    "dca_cooldown_min":     {"type": "int",   "low": 15,   "high": 120},
    "take_profit_pct":      {"type": "float", "low": 0.02,  "high": 0.10,  "decimals": 3},
    "sell_rsi_min":         {"type": "int",   "low": 55,   "high": 75},
    "time_stop_candles":    {"type": "int",   "low": 48,   "high": 288},
}

# 每组合随机搜索迭代次数
HYPEROPT_ITERATIONS = 30

# ==============================================================================
# 指标计算 (与策略 populate_indicators 完全一致)
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


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有策略指标 (对应 populate_indicators)"""
    f = FIXED_PARAMS

    # 布林带
    ma = df["close"].rolling(f["bb_period"]).mean()
    sd = df["close"].rolling(f["bb_period"]).std()
    df["bb_mid"] = ma
    df["bb_lower"] = ma - f["bb_std"] * sd
    df["bb_upper"] = ma + f["bb_std"] * sd

    # ATR
    df["atr_pct"] = compute_atr_pct(df, f["atr_period"])
    df["atr_ratio"] = df["atr_pct"] / (df["atr_pct"].rolling(50).mean() + 1e-9)

    # RSI (手动实现, 避免 talib 依赖)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/f["rsi_period"], adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/f["rsi_period"], adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # range_position
    roll_low = df["low"].rolling(f["range_period"]).min()
    roll_high = df["high"].rolling(f["range_period"]).max()
    df["range_position"] = (df["close"] - roll_low) / (roll_high - roll_low + 1e-9)

    # EMA 趋势
    df["ema_slow"] = df["close"].ewm(span=f["ema_slow_period"], adjust=False).mean()
    df["ema_slope"] = df["ema_slow"].diff(f["ema_slope_diff"]) / (df["ema_slow"] + 1e-9)

    return df


# ==============================================================================
# 回测引擎 (波动率网格马丁专属版)
# ==============================================================================

def run_backtest(df: pd.DataFrame, params: dict, pair: str, fast_mode: bool = False) -> dict:
    """
    执行完整回测 (含 DCA 马丁 + 熔断 + 时间止损 + 移动止盈)
    返回详细的绩效指标和交易记录。

    波动率网格马丁特有逻辑:
    - 入场: BB下轨 + RSI超卖 + ATR波动率+ + 区间(0.12~0.72) + 趋势上 + 无ATR爆炸
    - 出场: 回归中轨 OR RSI转强 (与限制定投的区别: 无区间下限参数)
    """
    p = params
    f = FIXED_PARAMS

    wallet = f["dry_run_wallet"]
    stake_amount = f["stake_amount"]
    max_open = f["max_open_trades"]
    fee = f["fee"]
    startup = f["startup_candle_count"]

    # 持仓状态
    positions: List[dict] = []
    closed_trades: List[dict] = []
    equity_curve: List[dict] = []
    dca_events: List[dict] = []

    peak_equity = wallet
    max_drawdown = 0.0
    max_floating_dd = 0.0

    # 每根K线遍历
    for i in range(startup, len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        price = row["close"]
        high = row["high"]
        low = row["low"]
        bb_mid = row["bb_mid"]
        bb_lower = row["bb_lower"]
        rsi = row["rsi"]
        atr_pct = row["atr_pct"]
        atr_ratio = row["atr_ratio"]
        range_pos = row["range_position"]
        ema_slope = row["ema_slope"]
        volume = row.get("volume", 1)

        # ── A. 检查入场信号 (波动率网格马丁版) ──
        # 条件1: 波动率足够 (太死的盘没回归空间)
        # 条件2: 跌破布林下轨 (超卖)
        # 条件3: 区间内 0.12~0.72 (不在底部最深处)
        # 条件4: RSI 超卖确认
        # 条件5: 非下跌趋势
        # 条件6: ATR 未爆炸
        if len(positions) < max_open:
            wide_enough = atr_pct > p["buy_atr_pct_min"]
            lower_band_touch = price < bb_lower
            range_not_dead = f["range_pos_min"] <= range_pos <= f["range_pos_max"]
            rsi_oversold = rsi < p["buy_rsi_max"]
            not_downtrend = ema_slope > -f["ema_slope_block"]
            no_atr_spike = atr_ratio < f["atr_spike_block"]

            long_signal = (
                wide_enough and lower_band_touch and range_not_dead
                and rsi_oversold and not_downtrend and no_atr_spike
                and volume > 0
            )
            if long_signal:
                amount = stake_amount / price
                pos = {
                    "open_time": current_time,
                    "open_price": price,
                    "side": "long",
                    "amount": amount,
                    "avg_entry": price,
                    "total_stake": stake_amount,
                    "bars_held": 0,
                    "dca_count": 0,
                    "dca_times": [],
                    "entry_stakes": [stake_amount],
                    "entry_prices": [price],
                    "entry_tag": "vol_grid_lower_band",
                }
                positions.append(pos)

        # ── B. 检查持仓出场 + DCA 加仓 ──
        to_close = []
        for p_idx, pos in enumerate(positions):
            pos["bars_held"] += 1

            # 当前浮盈 (价格层面)
            pos["current_profit"] = (price - pos["avg_entry"]) / pos["avg_entry"]

            # ── B1. 止盈检查: 回归中轨 或 RSI超买 (波动率网格专属: OR 逻辑) ──
            exit_price = None
            exit_reason = ""

            # 用 high 检查是否触及 bb_mid (更真实的回测: 捕捉盘中最优价)
            tp_bb = high >= bb_mid
            tp_rsi = rsi > p["sell_rsi_min"]
            tp_triggered = tp_bb or tp_rsi

            if tp_triggered:
                if tp_bb:
                    exit_price = max(price, bb_mid)  # 取较好价格
                    exit_reason = "mean_revert_tp"
                else:
                    exit_price = price
                    exit_reason = "rsi_overbought_tp"

            if exit_price is None:
                # ── B2. 马丁中断平仓 ──
                if pos["dca_count"] > p["dca_max_entries"]:
                    interrupt_level = -(p["dca_step_pct"] * p["dca_max_entries"] + f["dca_interrupt_extra_loss"])
                    if pos["current_profit"] < interrupt_level:
                        exit_price = price
                        exit_reason = "martingale_interrupt"

            if exit_price is None:
                # ── B3. 时间止损 ──
                tf_min = {"1m": 1, "5m": 5, "15m": 15}.get(
                    params.get("timeframe", "15m"), 15
                )
                max_hold_minutes = tf_min * p["time_stop_candles"]
                held_minutes = pos["bars_held"] * tf_min
                if held_minutes > max_hold_minutes and pos["current_profit"] < p["take_profit_pct"]:
                    exit_price = price
                    exit_reason = "time_stop"

            if exit_price is not None:
                # 执行平仓
                profit = (exit_price - pos["avg_entry"]) / pos["avg_entry"]
                profit -= fee * (1 + pos["dca_count"] + 1)  # 进出各一次 (含加仓的手续费)
                pos["exit_price"] = exit_price
                pos["exit_time"] = current_time
                pos["exit_reason"] = exit_reason
                pos["profit_ratio"] = profit
                pos["profit_abs"] = pos["total_stake"] * profit
                wallet += pos["profit_abs"]
                closed_trades.append(pos)
                to_close.append(p_idx)
                if not fast_mode and pos.get("dca_count", 0) > 0:
                    dca_events.append({
                        "open_time": pos["open_time"],
                        "dca_count": pos["dca_count"],
                        "exit_reason": exit_reason,
                        "final_profit": profit,
                    })
                continue  # 已平仓, 跳过加仓检查

            # ── B4. DCA 加仓 (未出场) ──
            if pos["dca_count"] >= p["dca_max_entries"]:
                continue  # 已达加仓上限

            # 浮盈不加仓
            if pos["current_profit"] >= -p["dca_step_pct"]:
                continue

            # 阶梯阈值: 第 N 次加仓需要浮亏 >= N * dca_step_pct
            next_trigger = -p["dca_step_pct"] * (pos["dca_count"] + 1)
            if pos["current_profit"] > next_trigger:
                continue

            # 加仓冷却
            if pos["dca_times"]:
                last_dca = pos["dca_times"][-1]
                if (current_time - last_dca).total_seconds() / 60 < p["dca_cooldown_min"]:
                    continue

            # ATR 爆炸禁加
            if atr_ratio > f["atr_spike_block"]:
                continue

            # 逆势趋势禁加 (做多时 ema_slope < -0.01 禁止加仓)
            if ema_slope < -f["ema_slope_block"]:
                continue

            # 执行加仓
            entry_count = pos["dca_count"] + 1
            multiplier = 1.0 + 0.2 * entry_count  # 与 custom_stake_amount 的预留序列一致
            add_stake = stake_amount * multiplier
            add_amount = add_stake / price

            # 更新持仓
            total_cost = pos["total_stake"] + add_stake
            pos["avg_entry"] = total_cost / (pos["amount"] + add_amount)
            pos["amount"] += add_amount
            pos["total_stake"] += add_stake
            pos["dca_count"] += 1
            pos["dca_times"].append(current_time)
            pos["entry_stakes"].append(add_stake)
            pos["entry_prices"].append(price)

            if not fast_mode:
                dca_events.append({
                    "open_time": pos["open_time"],
                    "dca_count": pos["dca_count"],
                    "adverse": round(pos["current_profit"], 6),
                    "threshold": round(next_trigger, 6),
                    "stake": round(add_stake, 2),
                    "time": current_time,
                })

        # 移除已平仓 (倒序)
        for p_idx in sorted(to_close, reverse=True):
            positions.pop(p_idx)

        # ── C. 更新权益曲线 ──
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
        if not fast_mode:
            equity_curve.append({
                "time": current_time,
                "equity": equity,
                "drawdown": dd,
            })

    # ── D. 强制平仓未平仓位 ──
    if positions and len(df) > startup:
        last_price = df.iloc[-1]["close"]
        last_time = df.index[-1]
        for pos in positions:
            profit = (last_price - pos["avg_entry"]) / pos["avg_entry"]
            profit -= fee * (1 + pos["dca_count"] + 1)
            pos["profit_ratio"] = profit
            pos["profit_abs"] = pos["total_stake"] * profit
            pos["exit_price"] = last_price
            pos["exit_time"] = last_time
            pos["exit_reason"] = "force_close_eod"
            wallet += pos["profit_abs"]
            closed_trades.append(pos)
            if pos.get("dca_count", 0) > 0:
                dca_events.append({
                    "open_time": pos["open_time"],
                    "dca_count": pos["dca_count"],
                    "exit_reason": "force_close_eod",
                    "final_profit": profit,
                })

    # ── E. 汇总统计 ──
    n_trades = len(closed_trades)
    wins = [t for t in closed_trades if t["profit_ratio"] > 0]
    losses = [t for t in closed_trades if t["profit_ratio"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    total_profit_pct = (wallet - f["dry_run_wallet"]) / f["dry_run_wallet"] * 100
    profits = [t["profit_ratio"] for t in closed_trades]
    avg_profit = np.mean(profits) * 100 if profits else 0
    best = max(profits) * 100 if profits else 0
    worst = min(profits) * 100 if profits else 0

    gross_profit = sum(t["profit_abs"] for t in wins)
    gross_loss = abs(sum(t["profit_abs"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    # 爆仓风险评估: 单笔亏损 > 50%
    liq_trades = [t for t in losses if t["profit_ratio"] < -0.50]
    liq_score = len(liq_trades)
    liq_level = (
        "极高风险" if liq_score >= 3 else
        "高风险" if liq_score >= 2 else
        "中等风险" if liq_score >= 1 else
        "安全"
    )

    # DCA 统计
    total_dca = len([e for e in dca_events if "adverse" in e])  # 只算加仓事件
    avg_dca_per_trade = total_dca / n_trades if n_trades > 0 else 0

    # 持仓时长
    holding_bars = [t["bars_held"] for t in closed_trades]

    # Sortino ratio (下行风险调整收益)
    returns = np.array(profits) if profits else np.array([0])
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0.01
    sortino = np.mean(returns) / downside_std if downside_std > 0 else (np.mean(returns) * 100)

    # Calmar ratio
    calmar = (total_profit_pct / 100) / (max_drawdown + 0.001)

    # Sharpe ratio (年化)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * (60 // {"1m": 1, "5m": 5, "15m": 15}.get(params.get("timeframe", "15m"), 15)))

    # 持仓频率分析
    exit_reasons = defaultdict(int)
    for t in closed_trades:
        exit_reasons[t.get("exit_reason", "unknown")] += 1

    # DCA 过程中最大持仓市值（用于爆仓风险评估）
    max_position_value = max(
        (p.get("total_stake", 0) for p in closed_trades),
        default=0
    )

    return {
        "pair": pair,
        "n_trades": n_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "total_profit_pct": round(total_profit_pct, 2),
        "total_profit_abs": round(wallet - f["dry_run_wallet"], 2),
        "final_wallet": round(wallet, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "max_floating_dd_pct": round(max_floating_dd * 100, 2),
        "avg_profit_pct": round(avg_profit, 2),
        "best_trade_pct": round(best, 2),
        "worst_trade_pct": round(worst, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 99.99,
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "sharpe_ratio": round(sharpe, 4),
        "dca_total": total_dca,
        "dca_avg_per_trade": round(avg_dca_per_trade, 2),
        "liquidation_risk_score": liq_score,
        "liquidation_trades_count": len(liq_trades),
        "liquidation_risk_level": liq_level,
        "avg_holding_bars": round(np.mean(holding_bars), 1) if holding_bars else 0,
        "max_holding_bars": max(holding_bars) if holding_bars else 0,
        "max_position_value": round(max_position_value, 2),
        "exit_reasons": dict(exit_reasons),
        "equity_curve": equity_curve,
        "closed_trades": closed_trades,
        "dca_events": dca_events,
        "params": params,
    }


# ==============================================================================
# 损失函数: 综合评分 (Sortino ratio 为核心, 惩罚爆仓)
# ==============================================================================

def evaluate_params(df: pd.DataFrame, params: dict, pair: str) -> float:
    """返回损失值 (越小越好), 用于 Hyperopt 搜索 (fast mode)"""
    result = run_backtest(df, params, pair, fast_mode=True)
    n = result["n_trades"]

    # 无交易 → 最差评分
    if n < 1:
        return 9999.0

    # 核心: Sortino ratio (越高越好 → 取负号使其越小越好)
    sortino = result["sortino_ratio"]

    # 惩罚项
    penalty = 0.0

    # 惩罚爆仓风险 (单笔亏损 > 50%)
    penalty += result["liquidation_risk_score"] * 2.0

    # 惩罚过高回撤
    if result["max_drawdown_pct"] > 40:
        penalty += (result["max_drawdown_pct"] - 40) / 10

    # 惩罚过低胜率
    if result["win_rate"] < 30:
        penalty += (30 - result["win_rate"]) / 10

    # 惩罚过少交易（太保守的参数）
    if n < 3:
        penalty += (3 - n) * 2

    # 综合损失
    loss = -sortino + penalty

    return loss


# ==============================================================================
# 随机搜索 Hyperopt
# ==============================================================================

def sample_params() -> dict:
    """从参数空间随机采样一组参数"""
    sampled = {}
    for name, spec in PARAM_SPACE.items():
        if spec["type"] == "int":
            sampled[name] = np.random.randint(spec["low"], spec["high"] + 1)
        elif spec["type"] == "float":
            val = np.random.uniform(spec["low"], spec["high"])
            sampled[name] = round(val, spec.get("decimals", 3))
    return sampled


def hyperopt_single(pair: str, timeframe: str, iterations: int = HYPEROPT_ITERATIONS) -> Tuple[dict, float, List[dict]]:
    """
    对单个 (pair, timeframe) 组合执行随机搜索 Hyperopt。
    返回: (best_params, best_loss, all_trials)
    """
    pair_safe = pair.replace("/", "_")
    data_path = DATA_DIR / f"{pair_safe}-{timeframe}.feather"

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
    df.columns = [c.lower() for c in df.columns]

    # 裁剪时间范围
    start = pd.Timestamp(TIMERANGE_START, tz="utc")
    end = pd.Timestamp(TIMERANGE_END, tz="utc")
    df = df[(df.index >= start) & (df.index <= end)]

    if len(df) < FIXED_PARAMS["startup_candle_count"] + 20:
        raise ValueError(f"数据不足: {len(df)} bars")

    df = compute_indicators(df)

    best_params = None
    best_loss = float("inf")
    all_trials = []

    for i in range(iterations):
        params = sample_params()
        params["timeframe"] = timeframe
        loss = evaluate_params(df, params, pair)

        trial = {"iteration": i, "params": params, "loss": loss}
        all_trials.append(trial)

        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()

        # 进度输出 (每 10 次)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{iterations}] best_loss={best_loss:.4f}", flush=True)

    return best_params, best_loss, all_trials


# ==============================================================================
# HTML 报告生成
# ==============================================================================

def generate_html_report(all_results: List[dict]) -> str:
    """生成包含盈亏曲线、加仓记录、爆仓统计的交互式 HTML 报告"""
    html_path = OUTPUT_DIR / "vol_grid_martingale_hyperopt_report.html"

    # 构建 chart.js 数据
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
                "timeline": [
                    e["time"].strftime("%m-%d %H:%M") if hasattr(e["time"], "strftime") else str(e["time"])
                    for e in eq
                ],
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
                <td colspan="19" class="error-cell">{r.get('error', '未知')}</td>
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
        sortino = r.get("sortino_ratio", 0)
        calmar = r.get("calmar_ratio", 0)
        sharpe = r.get("sharpe_ratio", 0)
        best = r.get("best_trade_pct", 0)
        worst = r.get("worst_trade_pct", 0)
        dca = r.get("dca_total", 0)
        dca_avg = r.get("dca_avg_per_trade", 0)
        liq = r.get("liquidation_risk_level", "N/A")
        liq_score = r.get("liquidation_risk_score", 0)
        avg_hold = r.get("avg_holding_bars", 0)
        max_pos = r.get("max_position_value", 0)
        params = r.get("params", {})
        params_str = ", ".join(
            f"{k}={v}" for k, v in params.items()
            if k in PARAM_SPACE
        )

        profit_color = "#ff4444" if profit_pct >= 0 else "#00aa44"
        dd_color = "#00aa44" if dd_pct < 20 else ("#ffaa00" if dd_pct < 40 else "#ff4444")

        rows_html += f"""
        <tr>
            <td class="cat-cell">{cat}</td>
            <td class="pair-cell">{pair}</td>
            <td class="tf-cell">{tf}</td>
            <td class="num-cell">{n_trades}</td>
            <td class="num-cell">{wins}</td>
            <td class="num-cell">{losses}</td>
            <td class="num-cell" style="color:{'#ff4444' if wr>=50 else '#00aa44'}">{wr:.1f}%</td>
            <td class="num-cell" style="color:{profit_color}">{profit_pct:+.2f}%</td>
            <td class="num-cell" style="color:{dd_color}">{dd_pct:.1f}%</td>
            <td class="num-cell">{float_dd:.1f}%</td>
            <td class="num-cell">{pf:.2f}</td>
            <td class="num-cell">{sortino:.2f}</td>
            <td class="num-cell">{calmar:.2f}</td>
            <td class="num-cell">{sharpe:.2f}</td>
            <td class="num-cell" style="color:{'#ff4444' if best>=0 else '#00aa44'}">{best:+.2f}%</td>
            <td class="num-cell" style="color:{'#00aa44' if worst>=0 else '#ff4444'}">{worst:+.2f}%</td>
            <td class="num-cell">{dca}</td>
            <td class="num-cell">{dca_avg:.1f}</td>
            <td class="num-cell">{avg_hold:.0f}</td>
            <td class="num-cell">{max_pos:.0f}</td>
            <td class="num-cell risk-{'high' if '高' in liq else 'low'}">{liq} ({liq_score})</td>
        </tr>"""

    # 最优参数表
    params_rows = ""
    for r in all_results:
        if not r.get("success"):
            continue
        params = r.get("params", {})
        params_rows += f"""
        <tr>
            <td class="pair-cell">{r['pair']}</td>
            <td class="tf-cell">{r['timeframe']}</td>
            <td class="num-cell">{params.get('buy_atr_pct_min', '-')}</td>
            <td class="num-cell">{params.get('buy_rsi_max', '-')}</td>
            <td class="num-cell">{params.get('dca_max_entries', '-')}</td>
            <td class="num-cell">{params.get('dca_step_pct', '-')}</td>
            <td class="num-cell">{params.get('dca_cooldown_min', '-')}</td>
            <td class="num-cell">{params.get('take_profit_pct', '-')}</td>
            <td class="num-cell">{params.get('sell_rsi_min', '-')}</td>
            <td class="num-cell">{params.get('time_stop_candles', '-')}</td>
            <td class="num-cell" style="color:{'#ff4444' if r['total_profit_pct']>=0 else '#00aa44'}">{r['total_profit_pct']:+.2f}%</td>
            <td class="num-cell">{r['sortino_ratio']:.2f}</td>
            <td class="num-cell">{r['max_drawdown_pct']:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meme 波动率网格马丁 · Hyperopt 批量优化报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ text-align: center; color: #ff6b6b; margin-bottom: 4px; font-size: 28px; }}
h2 {{ color: #ffd93d; margin: 30px 0 15px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
h3 {{ color: #4ecdc4; margin: 20px 0 10px; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 4px; }}
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
.summary-box {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.stat-card {{ background: #16213e; border-radius: 8px; padding: 12px; text-align: center; border: 1px solid #333; }}
.stat-card .label {{ color: #888; font-size: 11px; margin-bottom: 4px; }}
.stat-card .value {{ font-size: 20px; font-weight: bold; }}
.note {{ background: #16213e; border-left: 3px solid #ffd93d; padding: 12px 16px; margin: 16px 0; border-radius: 4px; font-size: 13px; color: #ccc; }}
.footer {{ text-align: center; color: #555; margin-top: 40px; font-size: 12px; }}
.chart-container {{ width: 100%; max-width: 900px; margin: 20px auto; background: #16213e; border-radius: 8px; padding: 16px; border: 1px solid #333; }}
canvas {{ width: 100% !important; }}
.chart-selector {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
.chart-btn {{ padding: 5px 12px; border: 1px solid #444; border-radius: 4px; background: #1a1a2e; color: #ccc; cursor: pointer; font-size: 11px; }}
.chart-btn:hover {{ background: #333; }}
.chart-btn.active {{ background: #ff6b6b; color: #fff; border-color: #ff6b6b; }}
.tab-container {{ margin: 20px 0; }}
.tab-buttons {{ display: flex; gap: 4px; margin-bottom: 0; }}
.tab-btn {{ padding: 8px 18px; border: 1px solid #333; border-bottom: none; border-radius: 6px 6px 0 0; background: #16213e; color: #888; cursor: pointer; font-size: 13px; }}
.tab-btn.active {{ background: #1a1a2e; color: #ffd93d; font-weight: bold; }}
.tab-content {{ display: none; background: #1a1a2e; border: 1px solid #333; border-radius: 0 6px 6px 6px; padding: 15px; overflow-x: auto; }}
.tab-content.active {{ display: block; }}
.legend {{ display: flex; gap: 15px; justify-content: center; margin-bottom: 10px; font-size: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.collapsible {{ cursor: pointer; }}
.collapsible:hover {{ color: #ffd93d; }}
@media (max-width: 768px) {{ table {{ font-size: 10px; }} th, td {{ padding: 4px 2px; }} }}
</style>
</head>
<body>

<h1>Meme 波动率网格马丁 · Hyperopt 批量优化报告</h1>
<div class="subtitle">布林下轨 + ATR波动过滤 → 超卖区间均值回归 + 马丁阶梯加仓 + 多层风控</div>
<div class="meta">
    生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
    区间: {TIMERANGE_START} ~ {TIMERANGE_END} |
    初始: {FIXED_PARAMS['dry_run_wallet']} USDT | 仓位: {FIXED_PARAMS['stake_amount']} USDT |
    每组 {HYPEROPT_ITERATIONS} 次搜索 | 总组数: {len(all_results)}
</div>

<div class="note">
    <strong>策略说明:</strong> 布林带(20,2.0)下轨超卖 + ATR波动率过滤(>buy_atr_pct_min) + RSI超卖 + 区间(0.12~0.72) →
    均值回归抄底。马丁DCA (1-4次阶梯加仓) + 冷却 + ATR爆炸禁加 + 趋势过滤。出场: 回归中轨 或 RSI转强 (OR逻辑)。
    多层风控: 马丁中断平仓、时间止损、移动止盈锁利。
    <br>
    <strong>Hyperopt:</strong> 8 参数随机搜索 ({HYPEROPT_ITERATIONS} iterations/组), 损失函数 = -Sortino + 爆仓惩罚 + 回撤惩罚。
    <br>
    <strong>与限制定投马丁的关键差异:</strong> 无 buy_range_pos_min 参数; 入场要求 atr_pct > 阈值 (确保波动率充足有回归空间); 
    出口 OR 逻辑 (回归中轨 或 RSI转强即止盈) 而非 AND 逻辑。
</div>

<h2>一、全局汇总</h2>
<div class="summary-box">
    <div class="stat-card"><div class="label">测试组数</div><div class="value" style="color:#ffd93d">{len(all_results)}</div></div>
    <div class="stat-card"><div class="label">成功组数</div><div class="value" style="color:#00aa44">{sum(1 for r in all_results if r.get('success'))}</div></div>
    <div class="stat-card"><div class="label">总交易次数</div><div class="value" style="color:#4ecdc4">{sum(r.get('n_trades', 0) for r in all_results)}</div></div>
    <div class="stat-card"><div class="label">总DCA次数</div><div class="value" style="color:#ffd93d">{sum(r.get('dca_total', 0) for r in all_results)}</div></div>
    <div class="stat-card"><div class="label">总盈亏</div><div class="value" style="color:{'#ff4444' if sum(r.get('total_profit_abs', 0) for r in all_results) >= 0 else '#00aa44'}">{sum(r.get('total_profit_abs', 0) for r in all_results):+.0f} USDT</div></div>
    <div class="stat-card"><div class="label">爆仓事件</div><div class="value" style="color:{'#00aa44' if sum(r.get('liquidation_trades_count', 0) for r in all_results)==0 else '#ff4444'}">{sum(r.get('liquidation_trades_count', 0) for r in all_results)}</div></div>
    <div class="stat-card"><div class="label">平均胜率</div><div class="value" style="color:#4ecdc4">{np.mean([r.get('win_rate',0) for r in all_results if r.get('success')]):.1f}%</div></div>
    <div class="stat-card"><div class="label">平均Sortino</div><div class="value" style="color:#4ecdc4">{np.mean([r.get('sortino_ratio',0) for r in all_results if r.get('success')]):.2f}</div></div>
</div>

<h2>二、盈亏曲线图</h2>
<div class="chart-selector" id="chartSelector"></div>
<div class="chart-container">
    <canvas id="equityChart" height="300"></canvas>
</div>
<div class="legend" id="chartLegend"></div>

<h2>三、Hyperopt 最优参数 ({len(all_results)} 组)</h2>
<div class="tab-container">
<div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab('params-all')">全部</button>
    <button class="tab-btn" onclick="switchTab('params-stable')">稳定主流币</button>
    <button class="tab-btn" onclick="switchTab('params-mid')">中端中型币种</button>
    <button class="tab-btn" onclick="switchTab('params-meme')">高波动妖币</button>
</div>
<div class="tab-content active" id="tab-params-all">
<table>
<thead><tr>
    <th>币种</th><th>周期</th>
    <th>ATR下限</th><th>RSI上限</th>
    <th>DCA次数</th><th>DCA步长</th><th>DCA冷却</th>
    <th>止盈%</th><th>止盈RSI</th><th>时间止损</th>
    <th>总盈亏</th><th>Sortino</th><th>回撤</th>
</tr></thead>
<tbody>{params_rows}</tbody>
</table>
</div>
<div class="tab-content" id="tab-params-stable"></div>
<div class="tab-content" id="tab-params-mid"></div>
<div class="tab-content" id="tab-params-meme"></div>
</div>

<h2>四、回测绩效明细 ({len(all_results)} 组)</h2>
<div class="tab-container">
<div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab2('perf-all')">全部</button>
    <button class="tab-btn" onclick="switchTab2('perf-stable')">稳定主流币</button>
    <button class="tab-btn" onclick="switchTab2('perf-mid')">中端中型币种</button>
    <button class="tab-btn" onclick="switchTab2('perf-meme')">高波动妖币</button>
</div>
<div class="tab-content active" id="tab-perf-all">
<table>
<thead><tr>
    <th>分类</th><th>币种</th><th>周期</th><th>交易</th><th>盈利</th><th>亏损</th><th>胜率</th>
    <th>总盈亏</th><th>回撤</th><th>浮亏</th><th>盈亏比</th><th>Sortino</th><th>Calmar</th><th>Sharpe</th>
    <th>最佳</th><th>最差</th><th>DCA</th><th>均DCA</th><th>均持K</th><th>最大持仓</th><th>爆仓</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<div class="tab-content" id="tab-perf-stable"></div>
<div class="tab-content" id="tab-perf-mid"></div>
<div class="tab-content" id="tab-perf-meme"></div>
</div>

<h2>五、DCA 加仓触发记录</h2>
<div class="note">
    <strong>加仓触发统计:</strong> 展示每个参数组合下马丁加仓的触发频率、加仓层级分布和最终结果。
    加仓频繁 → 市场区间震荡特征明显; 加仓稀少 → 单边行情主导, 或波动率不足以触发入场。
</div>
<div class="tab-container">
<div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab3('dca-all')">全部</button>
    <button class="tab-btn" onclick="switchTab3('dca-stable')">稳定主流币</button>
    <button class="tab-btn" onclick="switchTab3('dca-mid')">中端中型币种</button>
    <button class="tab-btn" onclick="switchTab3('dca-meme')">高波动妖币</button>
</div>
<div class="tab-content active" id="tab-dca-all" style="max-height:500px; overflow-y:auto;">
<table id="dcaTable">
<thead><tr>
    <th>币种</th><th>周期</th><th>总DCA</th><th>均DCA/笔</th>
    <th>DCA1层</th><th>DCA2层</th><th>DCA3层</th><th>DCA4层</th>
    <th>中断平仓</th><th>均值回归止盈</th><th>RSI止盈</th><th>时间止损</th><th>强制平仓</th>
</tr></thead>
<tbody id="dcaBody"></tbody>
</table>
</div>
<div class="tab-content" id="tab-dca-stable" style="max-height:500px; overflow-y:auto;"></div>
<div class="tab-content" id="tab-dca-mid" style="max-height:500px; overflow-y:auto;"></div>
<div class="tab-content" id="tab-dca-meme" style="max-height:500px; overflow-y:auto;"></div>
</div>

<div class="note">
    <strong>风险提示:</strong> 马丁策略本质是高胜率低盈亏比的负偏度策略。Hyperopt 最优参数
    可能存在过拟合风险, 建议仅作参考。实盘使用前需留出验证集做样本外检验。
    爆仓风险评级: 单笔亏损>50% = 爆仓事件。次数越多风险越高。
    持仓最大值超过初始资金 50% → 高杠杆风险。
</div>

<div class="footer">Meme 波动率网格马丁 Hyperopt 批量优化系统 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<script>
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
        document.querySelectorAll('#chartSelector .chart-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        drawChart(d);
    }};
    selector.appendChild(btn);
}});

function drawChart(d) {{
    if (activeChart) activeChart.destroy();
    const ctx = document.getElementById('equityChart').getContext('2d');
    const step = Math.max(1, Math.floor(d.timeline.length / 15));
    activeChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: d.timeline,
            datasets: [
                {{
                    label: '权益 (USDT)',
                    data: d.equity,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255,107,107,0.1)',
                    fill: true, tension: 0.3, pointRadius: 0, yAxisID: 'y',
                }},
                {{
                    label: '回撤 (%)',
                    data: d.drawdown,
                    borderColor: '#4ecdc4',
                    borderDash: [5,5],
                    fill: false, tension: 0.3, pointRadius: 0, yAxisID: 'y1',
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
                y: {{ type: 'linear', position: 'left', title: {{ display: true, text: '权益 (USDT)', color: '#ff6b6b' }}, ticks: {{ color: '#ff6b6b' }}, grid: {{ color: '#333' }} }},
                y1: {{ type: 'linear', position: 'right', title: {{ display: true, text: '回撤 (%)', color: '#4ecdc4' }}, ticks: {{ color: '#4ecdc4' }}, grid: {{ display: false }}, min: Math.min(...d.drawdown)-2, max: 0 }},
            }}
        }}
    }});
    document.getElementById('chartLegend').innerHTML =
        '<div class="legend-item"><span class="legend-dot" style="background:#ff6b6b"></span>权益</div>' +
        '<div class="legend-item"><span class="legend-dot" style="background:#4ecdc4"></span>回撤</div>';
}}

if (keys.length > 0) drawChart(allData[keys[0]]);

function switchTab(tab) {{
    document.querySelectorAll('.tab-container:first-of-type .tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-container:first-of-type .tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    const target = document.getElementById('tab-' + tab);
    if (target) {{
        if (tab === 'params-all') {{
            target.innerHTML = document.getElementById('tab-params-all').innerHTML;
        }} else {{
            const catMap = {{'params-stable': '稳定主流币', 'params-mid': '中端中型币种', 'params-meme': '高波动妖币'}};
            const cat = catMap[tab];
            const table = document.getElementById('tab-params-all').querySelector('table').cloneNode(true);
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(r => {{
                const cells = r.querySelectorAll('.pair-cell, .tf-cell');
                if (cells.length >= 2) {{
                    const pair = cells[0].textContent;
                    const stableCoins = ['BTC/USDT','ETH/USDT','SOL/USDT','XRP/USDT'];
                    const midCoins = ['XCN/USDT'];
                    const memeCoins = ['H/USDT','VELVET/USDT','BEAT/USDT','COAI/USDT'];
                    let isMatch = false;
                    if (cat === '稳定主流币' && stableCoins.includes(pair)) isMatch = true;
                    if (cat === '中端中型币种' && midCoins.includes(pair)) isMatch = true;
                    if (cat === '高波动妖币' && memeCoins.includes(pair)) isMatch = true;
                    r.style.display = isMatch ? '' : 'none';
                }}
            }});
            target.innerHTML = '';
            target.appendChild(table);
        }}
        target.classList.add('active');
    }}
}}

function switchTab2(tab) {{
    document.querySelectorAll('.tab-container:nth-of-type(2) .tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-container:nth-of-type(2) .tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    const target = document.getElementById('tab-' + tab);
    if (target) {{
        if (tab === 'perf-all') {{
            target.innerHTML = document.getElementById('tab-perf-all').innerHTML;
        }} else {{
            const catMap = {{'perf-stable': '稳定主流币', 'perf-mid': '中端中型币种', 'perf-meme': '高波动妖币'}};
            const cat = catMap[tab];
            const table = document.getElementById('tab-perf-all').querySelector('table').cloneNode(true);
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(r => {{
                const firstCell = r.querySelector('.cat-cell');
                if (firstCell && firstCell.textContent !== cat && !r.classList.contains('error-row')) {{
                    r.style.display = 'none';
                }} else {{
                    r.style.display = '';
                }}
            }});
            target.innerHTML = '';
            target.appendChild(table);
        }}
        target.classList.add('active');
    }}
}}

function switchTab3(tab) {{
    document.querySelectorAll('.tab-container:nth-of-type(3) .tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-container:nth-of-type(3) .tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    const target = document.getElementById('tab-' + tab);
    if (target) {{
        target.classList.add('active');
        if (tab === 'dca-all') return;
        const catMap = {{'dca-stable': '稳定主流币', 'dca-mid': '中端中型币种', 'dca-meme': '高波动妖币'}};
        const cat = catMap[tab];
        const table = document.getElementById('dcaTable').cloneNode(true);
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach(r => {{
            const catCell = r.querySelector('td');
            if (catCell && catCell.textContent !== cat) r.style.display = 'none';
        }});
        target.innerHTML = '';
        target.appendChild(table);
    }}
}}
</script>
</body>
</html>"""

    # 生成 DCA 统计 JavaScript 数据
    dca_summary_data = []
    for r in all_results:
        if not r.get("success"):
            continue
        exit_reasons = r.get("exit_reasons", {})
        # DCA 层级统计
        dca_levels = defaultdict(int)
        for t in r.get("closed_trades", []):
            level = t.get("dca_count", 0)
            dca_levels[f"dca_{level}"] += 1
        dca_summary_data.append({
            "pair": r["pair"],
            "timeframe": r["timeframe"],
            "dca_total": r["dca_total"],
            "dca_avg": r["dca_avg_per_trade"],
            "dca_1": dca_levels.get("dca_1", 0),
            "dca_2": dca_levels.get("dca_2", 0),
            "dca_3": dca_levels.get("dca_3", 0),
            "dca_4": dca_levels.get("dca_4", 0),
            "exit_interrupt": exit_reasons.get("martingale_interrupt", 0),
            "exit_mean_revert": exit_reasons.get("mean_revert_tp", 0),
            "exit_rsi": exit_reasons.get("rsi_overbought_tp", 0),
            "exit_time": exit_reasons.get("time_stop", 0),
            "exit_force": exit_reasons.get("force_close_eod", 0),
            "category": r["category"],
        })

    dca_json = json.dumps(dca_summary_data, ensure_ascii=False)

    # 注入 DCA 表格数据
    dca_rows_js = ""
    for d in dca_summary_data:
        dca_rows_js += f"""
        <tr>
            <td class="pair-cell">{d['pair']}</td>
            <td class="tf-cell">{d['timeframe']}</td>
            <td class="num-cell">{d['dca_total']}</td>
            <td class="num-cell">{d['dca_avg']:.1f}</td>
            <td class="num-cell">{d['dca_1']}</td>
            <td class="num-cell">{d['dca_2']}</td>
            <td class="num-cell">{d['dca_3']}</td>
            <td class="num-cell">{d['dca_4']}</td>
            <td class="num-cell" style="color:#ff4444">{d['exit_interrupt']}</td>
            <td class="num-cell" style="color:#00aa44">{d['exit_mean_revert']}</td>
            <td class="num-cell" style="color:#4ecdc4">{d['exit_rsi']}</td>
            <td class="num-cell" style="color:#ffaa00">{d['exit_time']}</td>
            <td class="num-cell">{d['exit_force']}</td>
        </tr>"""

    html = html.replace('<tbody id="dcaBody"></tbody>', f'<tbody id="dcaBody">{dca_rows_js}</tbody>')

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return str(html_path)


# ==============================================================================
# 加仓记录 CSV 导出
# ==============================================================================

def export_dca_records(all_results: List[dict]):
    """导出加仓触发记录到 CSV"""
    dca_path = OUTPUT_DIR / "dca_records.csv"
    rows = []
    for r in all_results:
        if not r.get("success"):
            continue
        for e in r.get("dca_events", []):
            rows.append({
                "pair": r["pair"],
                "timeframe": r["timeframe"],
                "open_time": str(e.get("open_time", "")),
                "dca_count": e.get("dca_count", 0),
                "adverse": e.get("adverse", ""),
                "threshold": e.get("threshold", ""),
                "stake": e.get("stake", ""),
                "exit_reason": e.get("exit_reason", ""),
                "final_profit": e.get("final_profit", ""),
            })
    if rows:
        pd.DataFrame(rows).to_csv(dca_path, index=False, encoding="utf-8-sig")
        return str(dca_path)
    return None


# ==============================================================================
# 爆仓风险汇总 CSV
# ==============================================================================

def export_liquidation_details(all_results: List[dict]):
    """导出爆仓风险详情"""
    liq_path = OUTPUT_DIR / "liquidation_risk_details.csv"
    rows = []
    for r in all_results:
        if not r.get("success"):
            continue
        for t in r.get("closed_trades", []):
            if t.get("profit_ratio", 0) < -0.50:
                rows.append({
                    "pair": r["pair"],
                    "timeframe": r["timeframe"],
                    "open_time": str(t.get("open_time", "")),
                    "exit_time": str(t.get("exit_time", "")),
                    "exit_reason": t.get("exit_reason", ""),
                    "profit_ratio": round(t.get("profit_ratio", 0) * 100, 2),
                    "dca_count": t.get("dca_count", 0),
                    "total_stake": round(t.get("total_stake", 0), 2),
                    "avg_entry": round(t.get("avg_entry", 8), 8) if t.get("avg_entry") else "",
                    "params": json.dumps(r.get("params", {}), ensure_ascii=False),
                })
    if rows:
        pd.DataFrame(rows).to_csv(liq_path, index=False, encoding="utf-8-sig")
        return str(liq_path)
    return None


# ==============================================================================
# 最优参数汇总 CSV
# ==============================================================================

def export_best_params(all_results: List[dict]):
    """导出所有最优参数到 CSV"""
    params_path = OUTPUT_DIR / "best_params.csv"
    rows = []
    for r in all_results:
        if not r.get("success"):
            continue
        row = {
            "category": r["category"],
            "pair": r["pair"],
            "timeframe": r["timeframe"],
        }
        row.update(r.get("params", {}))
        row.update({
            "total_profit_pct": r["total_profit_pct"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "sortino_ratio": r["sortino_ratio"],
            "calmar_ratio": r["calmar_ratio"],
            "sharpe_ratio": r["sharpe_ratio"],
            "win_rate": r["win_rate"],
            "n_trades": r["n_trades"],
            "dca_total": r["dca_total"],
            "liquidation_risk": r["liquidation_risk_level"],
        })
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(params_path, index=False, encoding="utf-8-sig")
        return str(params_path)
    return None


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    print("=" * 70)
    print("  Meme 波动率网格马丁 · Hyperopt 批量优化")
    print("=" * 70)
    print(f"  区间: {TIMERANGE_START} ~ {TIMERANGE_END}")
    print(f"  资金: {FIXED_PARAMS['dry_run_wallet']} USDT | 仓位: {FIXED_PARAMS['stake_amount']} USDT")
    print(f"  参数空间: {len(PARAM_SPACE)} 个参数, 每组 {HYPEROPT_ITERATIONS} 次随机搜索")
    total_runs = sum(len(v) for v in PAIR_CATEGORIES.values()) * len(TIMEFRAMES)
    print(f"  总计: {total_runs} 组 (9coins x 3timeframes)")
    print(f"  输出: {OUTPUT_DIR}")
    print("=" * 70)

    all_results = []
    current = 0
    start_time = datetime.now()

    for cat_name, pairs in PAIR_CATEGORIES.items():
        print(f"\n{'=' * 55}")
        print(f"  > {cat_name} ({len(pairs)} pairs)")
        print(f"{'=' * 55}")

        for pair in pairs:
            for tf in TIMEFRAMES:
                current += 1
                pair_safe = pair.replace("/", "_")
                data_path = DATA_DIR / f"{pair_safe}-{tf}.feather"

                elapsed = datetime.now() - start_time
                eta = elapsed / max(current - 1, 1) * (total_runs - current + 1) if current > 1 else timedelta()
                print(f"\n  [{current}/{total_runs}] {pair} @ {tf}  (ETR {str(eta).split('.')[0]})")
                print(f"    数据: {data_path}", flush=True)

                if not data_path.exists():
                    print(f"    X 数据文件不存在")
                    all_results.append({
                        "success": False, "pair": pair, "timeframe": tf,
                        "category": cat_name, "error": "数据文件不存在"
                    })
                    continue

                try:
                    # Phase 1: Hyperopt
                    print(f"    Hyperopt 搜索中 ({HYPEROPT_ITERATIONS} iterations)...", flush=True)
                    t0 = datetime.now()
                    best_params, best_loss, trials = hyperopt_single(pair, tf, HYPEROPT_ITERATIONS)
                    t1 = datetime.now()

                    print(f"    最优损失: {best_loss:.4f}  (耗时 {str(t1-t0).split('.')[0]})")
                    print(f"    最优参数: {json.dumps({k: v for k, v in best_params.items() if k in PARAM_SPACE}, ensure_ascii=False)}")

                    # Phase 2: 用最优参数做完整回测
                    print(f"    回测中...", end="", flush=True)
                    t2 = datetime.now()

                    df = pd.read_feather(data_path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], utc=True)
                        df = df.set_index("date")
                    df.columns = [c.lower() for c in df.columns]

                    start = pd.Timestamp(TIMERANGE_START, tz="utc")
                    end = pd.Timestamp(TIMERANGE_END, tz="utc")
                    df = df[(df.index >= start) & (df.index <= end)]
                    df = compute_indicators(df)

                    result = run_backtest(df, best_params, pair)
                    result["success"] = True
                    result["timeframe"] = tf
                    result["category"] = cat_name
                    result["best_loss"] = best_loss
                    result["hyperopt_trials"] = len(trials)
                    result["hyperopt_trials_data"] = [
                        {"i": t["iteration"], "loss": t["loss"],
                         "params": {k: v for k, v in t["params"].items() if k in PARAM_SPACE}}
                        for t in sorted(trials, key=lambda x: x["loss"])[:10]
                    ]

                    t3 = datetime.now()

                    n = result["n_trades"]
                    pnl = result["total_profit_pct"]
                    wr = result["win_rate"]
                    dca = result["dca_total"]
                    dd = result["max_drawdown_pct"]
                    liq = result["liquidation_risk_level"]
                    sortino = result["sortino_ratio"]
                    print(f" OK {n}trades | PnL{pnl:+.2f}% | WR{wr:.1f}% | DD{dd:.1f}% | Sortino{sortino:.2f} | DCAx{dca} | {liq}  (回测 {str(t3-t2).split('.')[0]})")

                except Exception as e:
                    import traceback
                    print(f" X 失败: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "success": False, "pair": pair, "timeframe": tf,
                        "category": cat_name, "error": f"{type(e).__name__}: {str(e)[:300]}"
                    })
                    continue

                all_results.append(result)

    # ── 输出文件 ──
    total_elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print(f"  Hyperopt 完成! 总耗时: {str(total_elapsed).split('.')[0]}")
    print(f"{'=' * 70}")

    # 1. JSON 摘要 (不含大数组)
    json_path = OUTPUT_DIR / "hyperopt_summary.json"
    slim_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items()
              if k not in ("equity_curve", "closed_trades", "dca_events", "hyperopt_trials_data")}
        slim_results.append(sr)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON 摘要: {json_path}")

    # 2. JSON 完整结果
    full_json_path = OUTPUT_DIR / "hyperopt_full_results.json"
    for r in all_results:
        if r.get("success"):
            for e in r.get("equity_curve", []):
                if hasattr(e.get("time"), "isoformat"):
                    e["time"] = e["time"].isoformat()
            for t in r.get("closed_trades", []):
                for k in ("open_time", "exit_time"):
                    if k in t and hasattr(t[k], "isoformat"):
                        t[k] = t[k].isoformat()
                if "dca_times" in t:
                    t["dca_times"] = [dt.isoformat() if hasattr(dt, "isoformat") else str(dt) for dt in t["dca_times"]]
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON 完整: {full_json_path}")

    # 3. DCA 加仓记录
    dca_csv = export_dca_records(all_results)
    if dca_csv:
        print(f"  DCA 记录: {dca_csv}")

    # 4. 爆仓风险详情
    liq_csv = export_liquidation_details(all_results)
    if liq_csv:
        print(f"  爆仓详情: {liq_csv}")

    # 5. 最优参数 CSV
    params_csv = export_best_params(all_results)
    if params_csv:
        print(f"  最优参数: {params_csv}")

    # 6. HTML 报告
    html_path = generate_html_report(all_results)
    print(f"  HTML 报告: {html_path}")

    success_count = sum(1 for r in all_results if r["success"])
    print(f"\n  完成: {success_count}/{total_runs} 组成功")
    print("=" * 70)

    # ── 分类汇总 ──
    print("\n  ▹ 分类汇总:")
    for cat_name, pairs in PAIR_CATEGORIES.items():
        cat_results = [r for r in all_results if r.get("category") == cat_name and r.get("success")]
        if not cat_results:
            print(f"    {cat_name}: 无成功结果")
            continue
        avg_pnl = np.mean([r["total_profit_pct"] for r in cat_results])
        avg_sortino = np.mean([r["sortino_ratio"] for r in cat_results])
        avg_dd = np.mean([r["max_drawdown_pct"] for r in cat_results])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in cat_results])
        total_liq = sum(r["liquidation_risk_score"] for r in cat_results)
        total_trades = sum(r["n_trades"] for r in cat_results)
        total_dca = sum(r["dca_total"] for r in cat_results)
        print(f"    {cat_name}: avg PnL{avg_pnl:+.2f}% | Sortino{avg_sortino:.2f} | Sharpe{avg_sharpe:.2f} | DD{avg_dd:.1f}% | {total_trades}trades | {total_dca}DCA | liqRisk{total_liq}")

    # ── 按时间框架汇总 ──
    print("\n  ▹ 按时间框架汇总:")
    for tf in TIMEFRAMES:
        tf_results = [r for r in all_results if r.get("timeframe") == tf and r.get("success")]
        if not tf_results:
            print(f"    {tf}: 无结果")
            continue
        avg_pnl = np.mean([r["total_profit_pct"] for r in tf_results])
        avg_sortino = np.mean([r["sortino_ratio"] for r in tf_results])
        avg_dd = np.mean([r["max_drawdown_pct"] for r in tf_results])
        total_liq = sum(r["liquidation_risk_score"] for r in tf_results)
        print(f"    {tf}: avg PnL{avg_pnl:+.2f}% | Sortino{avg_sortino:.2f} | DD{avg_dd:.1f}% | liqRisk{total_liq}")

    return html_path


if __name__ == "__main__":
    main()
