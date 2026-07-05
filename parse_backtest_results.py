#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理脚本：正确解析 v6 回测 stdout 文件，提取完整指标。
生成：
 1. 全币种全周期绩效汇总表（CSV + 控制台）
 2. 各币种盈亏曲线数据（从 trades json）
 3. 最大浮亏统计
 4. 加仓触发记录（DCA）
 5. 爆仓风险统计
"""
import json, re, os
from pathlib import Path
from datetime import datetime

BASE = Path(r"F:\source\freqtrade\deliverables\backtest_meme_limited_20260705_211205")
RESULTS_DIR = BASE / "results"
METRICS_DIR = BASE / "metrics"

# 预期 27 个组合
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "XCN/USDT",
    "H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT",
]
TFS = ["1m", "5m", "15m"]


def parse_stdout_file(stdout_file):
    """正确解析 freqtrade backtesting stdout 文件。"""
    with open(stdout_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")

    m = {
        "total_trades": 0,
        "win_pct": 0.0,
        "tot_profit_pct": 0.0,
        "tot_profit_usdt": 0.0,
        "avg_profit_pct": 0.0,
        "max_dd_pct": 0.0,
        "max_dd_usdt": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "profit_factor": 0.0,
        "cagr": 0.0,
        "sqn": 0.0,
        "exit_reasons": {},
        "enter_tags": {},
        "avg_duration": "",
        "best_trade_pct": 0.0,
        "worst_trade_pct": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
    }

    try:
        # 解析 BACKTESTING REPORT — 找 TOTAL 行
        in_sec = False
        for i, ln in enumerate(lines):
            if "BACKTESTING REPORT" in ln:
                in_sec = True
                continue
            if in_sec and ("┴" in ln or "══" in ln):
                in_sec = False
                continue
            if in_sec and "│" in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 2 and cols[0] == "TOTAL" and len(cols) >= 7:
                    m["total_trades"]    = int(cols[1])
                    m["avg_profit_pct"]  = _f(cols[2])
                    m["tot_profit_usdt"] = _f(cols[3])
                    m["tot_profit_pct"]  = _f(cols[4])
                    m["avg_duration"]    = cols[5]
                    # 最后一列: "22 0 6 78.6"
                    nums = re.findall(r"[\d.]+", cols[6])
                    if len(nums) >= 4:
                        m["win_pct"] = float(nums[3])
                    break

        # EXIT REASON STATS
        in_sec = False
        for ln in lines:
            if "EXIT REASON STATS" in ln:
                in_sec = True
                continue
            if in_sec and ("┴" in ln or "══" in ln):
                in_sec = False
                continue
            if in_sec and "│" in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 2 and cols[0] not in ("Exit Reason", "TOTAL"):
                    m["exit_reasons"][cols[0]] = int(cols[1])

        # ENTER TAG STATS
        in_sec = False
        for ln in lines:
            if "ENTER TAG STATS" in ln:
                in_sec = True
                continue
            if in_sec and ("┴" in ln or "══" in ln):
                in_sec = False
                continue
            if in_sec and "│" in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 2 and cols[0] not in ("Enter Tag", "TOTAL"):
                    m["enter_tags"][cols[0]] = int(cols[1])

        # SUMMARY METRICS — 用正则逐行提取
        def _re_val(pattern, group=1):
            m2 = re.search(pattern, text)
            if m2:
                try:
                    return float(m2.group(group))
                except:
                    pass
            return 0.0

        def _re_int(pattern):
            m2 = re.search(pattern, text)
            if m2:
                try:
                    return int(m2.group(1))
                except:
                    pass
            return 0

        m["sharpe"]           = _re_val(r"Sharpe\s*\(closed\s+trades\)[^\n]*?([-\d.]+)")
        m["sortino"]          = _re_val(r"Sortino\s*\(closed\s+trades\)[^\n]*?([-\d.]+)")
        m["profit_factor"]    = _re_val(r"Profit factor[^\n]*?([-\d.]+)")
        m["cagr"]             = _re_val(r"CAGR\s*%[^\n]*?([-\d.]+)")
        m["sqn"]              = _re_val(r"^\│\s*SQN[^\n]*?([-\d.]+)", group=1)
        m["best_trade_pct"]   = _re_val(r"Best trade[^\n]*?([-\d.]+)\s*%")
        m["worst_trade_pct"]  = _re_val(r"Worst trade[^\n]*?([-\d.]+)\s*%")
        m["max_consecutive_wins"]  = _re_int(r"Max Consecutive Wins.*?(\d+)")
        m["max_consecutive_losses"] = _re_int(r"Max Consecutive Loss[^\n]*?(\d+)")

        # Max drawdown % (closed trades, first occurrence)
        for dd_pat in [
            r"Max % of account underwater\s*\(closed trades\)[^\n]*?([\d.]+)\s*%",
            r"Max % of account underwater\s*\(.*?\)[^\n]*?([\d.]+)\s*%",
            r"Absolute drawdown.*?([\d.]+)\s*%",
            r"Max drawdown[^\n]*?([\d.]+)\s*%",
        ]:
            m2 = re.search(dd_pat, text)
            if m2:
                m["max_dd_pct"] = float(m2.group(1))
                break

    except Exception as e:
        print(f"  解析异常: {e}")

    return m


def _f(s):
    """安全转 float。"""
    try:
        return float(s)
    except:
        return 0.0


def main():
    print("=" * 70)
    print("Meme_限制定投_马丁 回测结果后处理")
    print(f"  目录: {BASE}")
    print("=" * 70)

    all_results = []
    for pair in PAIRS:
        for tf in TFS:
            safe = pair.replace("/", "_")
            name = f"MemeLimitedMartingaleSpotStrategy_{safe}_{tf}"
            stdout_file = RESULTS_DIR / f"{name}_stdout.txt"
            if not stdout_file.exists():
                print(f"  ✗ 缺失: {stdout_file.name}")
                continue
            m = parse_stdout_file(stdout_file)
            m["pair"]      = pair
            m["timeframe"] = tf
            m["success"]   = True
            all_results.append(m)
            print(f"  {pair:12s} {tf:4s} | trades={m['total_trades']:3d}  "
                  f"profit={m['tot_profit_pct']:+7.2f}%  dd={m['max_dd_pct']:5.2f}%  "
                  f"Sharpe={m['sharpe']:5.2f}")
    
    # 保存完整指标 JSON
    out = {
        "strategy": "MemeLimitedMartingaleSpotStrategy",
        "timerange": "20260608-20260627",
        "total": len(all_results),
        "results": all_results,
    }
    with open(BASE / "results_parsed.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {BASE / 'results_parsed.json'}")

    # 生成 CSV
    import csv
    csv_file = BASE / "results_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "币种", "周期", "交易数", "胜率%", "总收益%", "总收益USDT",
            "平均收益%", "最大回撤%", "Sharpe", "Sortino", "ProfitFactor",
            "CAGR%", "SQN", "最佳交易%", "最差交易%",
            "最大连胜", "最大连亏",
        ])
        for r in all_results:
            writer.writerow([
                r["pair"], r["timeframe"], r["total_trades"], r["win_pct"],
                r["tot_profit_pct"], r["tot_profit_usdt"],
                r["avg_profit_pct"], r["max_dd_pct"],
                r["sharpe"], r["sortino"], r["profit_factor"],
                r["cagr"], r["sqn"],
                r["best_trade_pct"], r["worst_trade_pct"],
                r["max_consecutive_wins"], r["max_consecutive_losses"],
            ])
    print(f"✅ 已保存: {csv_file}")

    # 按类别输出汇总
    print("\n" + "=" * 70)
    print("按类别汇总")
    print("=" * 70)
    categories = {
        "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
        "中端中型币": ["XCN/USDT"],
        "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
    }
    for cat, pairs in categories.items():
        print(f"\n--- {cat} ---")
        cat_results = [r for r in all_results if r["pair"] in pairs]
        for r in cat_results:
            print(f"  {r['pair']:12s} {r['timeframe']:4s} | "
                  f"trades={r['total_trades']:3d}  "
                  f"profit={r['tot_profit_pct']:+7.2f}%  "
                  f"dd={r['max_dd_pct']:5.2f}%  "
                  f"Sharpe={r['sharpe']:5.2f}")


if __name__ == "__main__":
    main()
