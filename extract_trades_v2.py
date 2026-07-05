#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从正确的 zip 文件提取各 (pair, tf) 的交易明细。
生成盈亏曲线、DCA 记录、风险统计。
"""
import zipfile, json, csv
from pathlib import Path
from datetime import datetime
import numpy as np

BASE = Path(r"F:\source\freqtrade")
BT_DIR = BASE / "user_data" / "backtest_results"
OUT_DIR = BASE / "deliverables" / "backtest_meme_limited_20260705_211205"

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
         "XCN/USDT", "H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"]
TFS = ["1m", "5m", "15m"]


def get_zip_for(pair, tf):
    """找到对应 (pair, tf) 的 zip 文件。"""
    zips = sorted(BT_DIR.glob("backtest-result-2026-07-05_*.zip"))
    for zp in reversed(zips):  # 从最新的开始找
        try:
            with zipfile.ZipFile(zp) as z:
                # 读 config
                cfgs = [n for n in z.namelist() if "config" in n and n.endswith(".json")]
                if not cfgs:
                    continue
                cfg = json.loads(z.read(cfgs[0]))
                if cfg.get("timeframe") != tf:
                    continue

                # 读 trades JSON
                jsons = [n for n in z.namelist()
                         if n.endswith(".json") and "config" not in n]
                if not jsons:
                    continue
                data = json.loads(z.read(jsons[0]))
                for sname, sdata in data.get("strategy", {}).items():
                    trades = sdata.get("trades", [])
                    if trades and trades[0].get("pair") == pair:
                        return trades
        except:
            continue
    return []


def detect_dca_entries(trades):
    """检测 DCA 加仓。"""
    records = []
    for t in trades:
        orders = t.get("orders", [])
        entry_orders = [o for o in orders if o.get("ft_is_entry")]
        n = len(entry_orders)
        if n > 1:
            records.append({
                "pair": t["pair"],
                "open_date": t["open_date"],
                "close_date": t["close_date"],
                "n_entries": n,
                "max_stake": t.get("max_stake_amount", t["stake_amount"]),
                "initial_stake": t["stake_amount"],
                "profit_pct": round(t["profit_ratio"] * 100, 2),
                "profit_usdt": round(t["profit_abs"], 4),
                "exit_reason": t.get("exit_reason", ""),
            })
    return records


def build_equity(trades):
    """从 trades 构建盈亏曲线。"""
    if not trades:
        return [], [], []
    sorted_t = sorted(trades, key=lambda x: x["close_date"])
    dates, cum_profit, drawdowns = [], [], []
    balance = 1000.0
    peak = balance
    for t in sorted_t:
        balance += t["profit_abs"]
        dates.append(t["close_date"])
        cum_profit.append(round(balance - 1000, 4))
        if balance > peak:
            peak = balance
        dd = round((peak - balance) / peak * 100, 2)
        drawdowns.append(dd)
    return dates, cum_profit, drawdowns


def calc_risk(trades):
    """爆仓风险统计。"""
    if not trades:
        return {}
    sorted_t = sorted(trades, key=lambda x: x["close_date"])
    balance = 1000.0
    peak = balance
    max_dd_pct = 0
    for t in sorted_t:
        balance += t["profit_abs"]
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100
        if dd > max_dd_pct:
            max_dd_pct = dd

    # 连续亏损
    max_cl = 0
    max_cl_pct = 0.0
    cl = 0
    cl_pct = 0.0
    for t in sorted_t:
        if t["profit_ratio"] < 0:
            cl += 1
            cl_pct += t["profit_ratio"] * 100
        else:
            if cl > max_cl:
                max_cl, max_cl_pct = cl, round(abs(cl_pct), 2)
            cl, cl_pct = 0, 0.0
    if cl > max_cl:
        max_cl, max_cl_pct = cl, round(abs(cl_pct), 2)

    # 极端亏损 (>5%)
    extreme = [t for t in sorted_t if t["profit_ratio"] < -0.05]
    
    total_profit = sum(t["profit_abs"] for t in sorted_t)

    return {
        "max_dd_pct": round(max_dd_pct, 2),
        "max_consec_losses": max_cl,
        "max_consec_loss_pct": max_cl_pct,
        "extreme_events": len(extreme),
        "extreme_total_pct": round(sum(t["profit_ratio"] * 100 for t in extreme), 2),
        "worst_trade_pct": round(min(t["profit_ratio"] * 100 for t in sorted_t), 2) if sorted_t else 0,
        "total_profit_usdt": round(total_profit, 4),
    }


def main():
    print("=" * 70)
    print("正确提取交易明细 (每个 zip 对应一个 backtest)")
    print("=" * 70)

    all_dca = []
    all_risk = []
    all_equity = {}  # key: "pair_tf" -> {dates, cum_profit, drawdowns}

    for pair in PAIRS:
        for tf in TFS:
            trades = get_zip_for(pair, tf)
            n = len(trades)
            if n == 0:
                continue

            # DCA
            dca = detect_dca_entries(trades)
            all_dca.extend(dca)

            # 盈亏曲线
            dates, cum_profits, dds = build_equity(trades)
            all_equity[f"{pair}_{tf}"] = {
                "pair": pair, "tf": tf,
                "dates": [str(d) for d in dates],
                "cum_profit": cum_profits,
                "drawdowns": dds,
            }

            # 风险统计
            risk = calc_risk(trades)
            risk["pair"] = pair
            risk["timeframe"] = tf
            risk["n_trades"] = n
            all_risk.append(risk)

            dca_rate = len(dca) / n * 100 if n > 0 else 0
            print(f"  {pair:12s} {tf:4s} | {n:3d} trades | "
                  f"DCA={len(dca):2d}({dca_rate:4.1f}%) | "
                  f"profit={risk['total_profit_usdt']:+.3f} USDT | "
                  f"maxDD={risk['max_dd_pct']:5.2f}% | "
                  f"consecLoss={risk['max_consec_losses']}")

    # 保存 DCA 记录
    dca_file = OUT_DIR / "dca_records.csv"
    if all_dca:
        with open(dca_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["币种", "开仓时间", "平仓时间", "加仓次数", "最大仓位",
                        "初始仓位", "收益%", "收益USDT", "平仓原因"])
            for r in all_dca:
                w.writerow([r[k] for k in ["pair","open_date","close_date",
                            "n_entries","max_stake","initial_stake",
                            "profit_pct","profit_usdt","exit_reason"]])
    print(f"\n✅ DCA 记录: {dca_file} ({len(all_dca)} 条)")

    # 保存风险统计
    risk_file = OUT_DIR / "risk_stats.csv"
    if all_risk:
        with open(risk_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["币种","周期","交易数","总收益USDT","最大回撤%",
                        "最大连亏数","最大连亏%","极端亏损次数","极端亏损总%",
                        "最差单笔%"])
            for r in all_risk:
                w.writerow([r[k] for k in ["pair","timeframe","n_trades",
                            "total_profit_usdt","max_dd_pct","max_consec_losses",
                            "max_consec_loss_pct","extreme_events",
                            "extreme_total_pct","worst_trade_pct"]])
    print(f"✅ 风险统计: {risk_file} ({len(all_risk)} 条)")

    # 保存盈亏曲线 JSON（供 HTML 图表用）
    with open(OUT_DIR / "equity_curves.json", "w", encoding="utf-8") as f:
        json.dump(all_equity, f, ensure_ascii=False)
    print(f"✅ 盈亏曲线: {OUT_DIR / 'equity_curves.json'}")

    return all_risk, all_dca, all_equity


if __name__ == "__main__":
    main()
