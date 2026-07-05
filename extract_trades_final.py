#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从所有 backtest-result zip 文件中提取交易明细。
生成：盈亏曲线 CSV、DCA 加仓记录、爆仓风险统计。
"""
import zipfile, json, re, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

BASE = Path(r"F:\source\freqtrade")
BT_DIR = BASE / "user_data" / "backtest_results"
OUT_DIR = BASE / "deliverables" / "backtest_meme_limited_20260705_211205"


def extract_all():
    """遍历所有 zip，按 (pair, timeframe) 分组提取 trades。"""
    zips = sorted(BT_DIR.glob("backtest-result-2026-07-05_*.zip"))
    grouped = {}

    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                # 读 config 获取 timeframe
                config_file = [n for n in z.namelist() if "config" in n and n.endswith(".json")]
                tf = "unknown"
                if config_file:
                    cfg = json.loads(z.read(config_file[0]))
                    tf = cfg.get("timeframe", "unknown")

                # 读 results JSON 获取 trades
                json_file = [n for n in z.namelist()
                             if n.endswith(".json") and "config" not in n and "strat" not in n]
                if not json_file:
                    continue
                data = json.loads(z.read(json_file[0]))

                if "strategy" not in data:
                    continue
                for strat_name, strat_data in data["strategy"].items():
                    trades = strat_data.get("trades", [])
                    for t in trades:
                        pair = t.get("pair", "unknown")
                        key = f"{pair}_{tf}"
                        grouped.setdefault(key, []).append(t)
        except Exception as e:
            print(f"  ✗ {zp.name}: {e}")

    return grouped


def detect_dca_entries(trades):
    """分析每个 trade 的 DCA 加仓情况。"""
    dca_records = []
    for t in trades:
        orders = t.get("orders", [])
        entry_orders = [o for o in orders if o.get("ft_is_entry")]
        n_entries = len(entry_orders)
        if n_entries > 1:
            dca_records.append({
                "pair": t["pair"],
                "open_date": t["open_date"],
                "close_date": t["close_date"],
                "n_entries": n_entries,
                "total_stake": t.get("max_stake_amount", t["stake_amount"]),
                "initial_stake": t["stake_amount"],
                "profit_ratio": t["profit_ratio"],
                "profit_abs": t["profit_abs"],
                "exit_reason": t.get("exit_reason", ""),
            })
    return dca_records


def build_equity_curve(trades):
    """按平仓时间排序，构建累计盈亏曲线。"""
    if not trades:
        return [], [], [], []
    
    sorted_trades = sorted(trades, key=lambda x: x["close_date"])
    
    dates = []
    cum_profit = []
    drawdowns = []
    balances = []
    balance = 1000.0  # 起始余额
    peak = balance
    
    for t in sorted_trades:
        profit = t["profit_abs"]
        balance += profit
        dates.append(t["close_date"])
        cum_profit.append(balance - 1000.0)
        balances.append(balance)
        if balance > peak:
            peak = balance
        dd = (balance - peak) / peak * 100
        drawdowns.append(abs(dd))
    
    return dates, cum_profit, drawdowns, balances


def calc_risk_stats(trades):
    """爆仓风险统计。"""
    if not trades:
        return {}
    
    sorted_trades = sorted(trades, key=lambda x: x["close_date"])
    
    # 最大浮亏
    balance = 1000.0
    peak = balance
    max_dd_pct = 0
    max_dd_abs = 0
    
    for t in sorted_trades:
        balance += t["profit_abs"]
        if balance > peak:
            peak = balance
        dd_pct = (peak - balance) / peak * 100
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_abs = peak - balance
    
    # 连续亏损
    max_consec_loss = 0
    consec_loss = 0
    max_consec_loss_pct = 0.0
    consec_loss_pct = 0.0
    
    for t in sorted_trades:
        if t["profit_ratio"] < 0:
            consec_loss += 1
            consec_loss_pct += t["profit_ratio"] * 100
        else:
            if consec_loss > max_consec_loss:
                max_consec_loss = consec_loss
                max_consec_loss_pct = consec_loss_pct
            consec_loss = 0
            consec_loss_pct = 0.0
    
    if consec_loss > max_consec_loss:
        max_consec_loss = consec_loss
        max_consec_loss_pct = consec_loss_pct
    
    # 极端回撤事件（单笔亏损 > 5%）
    extreme_draws = [t for t in sorted_trades if t["profit_ratio"] < -0.05]
    
    return {
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dd_abs": round(max_dd_abs, 4),
        "max_consec_losses": max_consec_loss,
        "max_consec_loss_pct": round(abs(max_consec_loss_pct), 2),
        "extreme_loss_events": len(extreme_draws),
        "extreme_loss_total": round(sum(t["profit_ratio"] * 100 for t in extreme_draws), 2),
        "worst_trade_pct": round(min(t["profit_ratio"] * 100 for t in sorted_trades), 2) if sorted_trades else 0,
    }


def main():
    print("=" * 70)
    print("提取交易明细 & 生成深度报告")
    print("=" * 70)
    
    grouped = extract_all()
    print(f"提取到 {len(grouped)} 个分组")
    
    # 构建 pair_timeframe 列表
    PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
             "XCN/USDT", "H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"]
    TFS = ["1m", "5m", "15m"]
    
    all_dca = []
    all_equity_data = {}
    all_risk = []
    
    for pair in PAIRS:
        for tf in TFS:
            key = f"{pair}_{tf}"
            trades = grouped.get(key, [])
            if not trades:
                # 尝试其他 format
                for k, v in grouped.items():
                    if pair in k and tf in k:
                        trades = v
                        break
            
            if not trades:
                continue
            
            n = len(trades)
            total_profit = sum(t["profit_abs"] for t in trades)
            
            # DCA 记录
            dca = detect_dca_entries(trades)
            all_dca.extend(dca)
            
            # 盈亏曲线
            dates, cum_profit, drawdowns, balances = build_equity_curve(trades)
            all_equity_data[f"{pair}_{tf}"] = {
                "pair": pair, "tf": tf,
                "dates": dates, "cum_profit": cum_profit,
                "drawdowns": drawdowns, "balances": balances,
            }
            
            # 风险统计
            risk = calc_risk_stats(trades)
            risk["pair"] = pair
            risk["timeframe"] = tf
            risk["n_trades"] = n
            risk["total_profit_pct"] = round(total_profit / 1000 * 100, 2)
            all_risk.append(risk)
            
            # DCA 统计
            n_dca = len(dca)
            dca_rate = n_dca / n * 100 if n > 0 else 0
            print(f"  {pair:12s} {tf:4s} | {n:3d} trades | "
                  f"DCA={n_dca:2d}({dca_rate:4.1f}%) | "
                  f"profit={total_profit:+.3f} USDT | "
                  f"maxDD={risk['max_dd_pct']:5.2f}% | "
                  f"consecLoss={risk['max_consec_losses']}")
    
    # 保存 DCA 记录
    import csv
    dca_file = OUT_DIR / "dca_records.csv"
    if all_dca:
        with open(dca_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["币种", "开仓时间", "平仓时间", "加仓次数", "总仓位", "初始仓位",
                        "收益比%", "收益USDT", "平仓原因"])
            for r in all_dca:
                w.writerow([r["pair"], r["open_date"], r["close_date"],
                            r["n_entries"], r["total_stake"], r["initial_stake"],
                            round(r["profit_ratio"]*100, 2), round(r["profit_abs"], 4),
                            r["exit_reason"]])
    print(f"\n✅ DCA 记录: {dca_file} ({len(all_dca)} 条)")
    
    # 保存风险统计
    risk_file = OUT_DIR / "risk_stats.csv"
    if all_risk:
        with open(risk_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["币种", "周期", "交易数", "总收益%", "最大回撤%", "最大回撤USDT",
                        "最大连亏数", "最大连亏%", "极端亏损事件数", "极端亏损总%", "最差单笔%"])
            for r in all_risk:
                w.writerow([r["pair"], r["timeframe"], r["n_trades"],
                            r["total_profit_pct"], r["max_dd_pct"], r["max_dd_abs"],
                            r["max_consec_losses"], r["max_consec_loss_pct"],
                            r["extreme_loss_events"], r["extreme_loss_total"],
                            r["worst_trade_pct"]])
    print(f"✅ 风险统计: {risk_file} ({len(all_risk)} 条)")
    
    # 保存盈亏曲线 JSON（供 HTML 图表使用）
    import json as _json
    equity_file = OUT_DIR / "equity_curves.json"
    with open(equity_file, "w", encoding="utf-8") as f:
        _json.dump(all_equity_data, f, ensure_ascii=False, default=str)
    print(f"✅ 盈亏曲线: {equity_file}")
    
    return all_risk, all_dca, all_equity_data


if __name__ == "__main__":
    main()
