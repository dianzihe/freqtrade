#!/usr/bin/env python3
"""
回测结果解析 + HTML 报告生成
从 backtest_logs/ 目录下的日志文件解析结果，生成报告
"""
import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "user_data" / "backtest_logs"
RESULTS_PATH = PROJECT_DIR / "user_data" / "backtest_results.json"

CATEGORIES = {
    "stable": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LTC/USDT", "HYPE/USDT"],
    "mid":    ["XCN/USDT", "IP/USDT", "BAS/USDT", "PEAQ/USDT", "TA/USDT"],
    "meme":   ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT", "ALLO/USDT", "DN/USDT", "STG/USDT"],
}


def parse_log(log_path: Path) -> dict | None:
    """Parse a backtest log file and extract metrics."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        stdout = f.read()

    # Check for fatal error
    if "Fatal exception" in stdout or "ERROR" in stdout:
        error_lines = [l.strip() for l in stdout.split("\n")
                       if "ERROR" in l or "Exception" in l or "Fatal" in l]
        return {"error": "\n".join(error_lines[-3:])}

    # Parse strategy name and timeframe from filename
    fname = log_path.stem  # e.g. "ChanlunCenterBreakoutStrategy_1m"
    parts = fname.rsplit("_", 1)
    if len(parts) == 2:
        strategy, timeframe = parts[0], parts[1]
    else:
        strategy = fname
        timeframe = "?"

    result = {
        "strategy": strategy,
        "timeframe": timeframe,
        "pair_results": {},
        "error": None,
    }

    # Parse per-pair results from summary table
    for line in stdout.split("\n"):
        # Match pair lines: │  BTC/USDT  │   285  │  -2.15%  │ ...
        m = re.search(r"│\s*(\S+/USDT)\s*│\s*(\d+)\s*│\s*([-\d.]+)%?", line)
        if m:
            pair = m.group(1).replace("_", "/")
            result["pair_results"][pair] = {
                "trades": int(m.group(2)),
                "profit_pct": float(m.group(3)),
            }

    if not result["pair_results"]:
        return {"error": "No pair results found - probably failed"}

    # Parse TOTAL row
    m = re.search(r"TOTAL\s*│\s*(\d+)\s*│\s*([-\d.]+)%", stdout)
    if m:
        result["total_trades"] = int(m.group(1))
        result["total_profit_pct"] = float(m.group(2))

    # Win rate
    m = re.search(r"Win\s*/\s*Loss\s*│\s*(\d+)\s*/\s*(\d+)", stdout)
    if m:
        wins = int(m.group(1))
        losses = int(m.group(2))
        if wins + losses > 0:
            result["win_rate"] = wins / (wins + losses)

    # Max drawdown
    m = re.search(r"Max\s*drawdown.*?([-\d.]+)%", stdout)
    if m:
        result["max_drawdown_pct"] = float(m.group(1))

    # Sharpe / Sortino
    for key, pat in [("sharpe", r"Sharpe.*?([-\d.]+)"),
                      ("sortino", r"Sortino.*?([-\d.]+)"),
                      ("profit_factor", r"Profit\s*factor.*?([\d.]+)")]:
        m = re.search(pat, stdout)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass

    return result


def aggregate_by_category(results: list) -> list:
    """Aggregate pair results by category."""
    aggregated = []
    for r in results:
        if r.get("error") or not r.get("pair_results"):
            aggregated.append(r)
            continue

        cat_results = {}
        for cat_key, cat_pairs in CATEGORIES.items():
            cat_pr = {p: r["pair_results"][p] for p in cat_pairs
                       if p in r["pair_results"]}
            if not cat_pr:
                continue
            cat_trades = sum(v["trades"] for v in cat_pr.values())
            # Weighted profit by trades
            if cat_trades > 0:
                cat_profit = sum(v["profit_pct"] * v["trades"] for v in cat_pr.values()) / cat_trades
            else:
                cat_profit = 0
            cat_results[cat_key] = {
                "pairs": list(cat_pr.keys()),
                "trades": cat_trades,
                "profit_pct": cat_profit,
            }

        r["category_results"] = cat_results
        aggregated.append(r)

    return aggregated


def build_html(results: list) -> str:
    """Generate HTML report."""
    # Filter successful results
    ok = [r for r in results if not r.get("error") and r.get("category_results")]
    failed = [r for r in results if r.get("error")]

    # Sort by profit
    ok.sort(key=lambda r: -abs(r.get("total_profit_pct") or 0))

    now = Path(__file__).stat().st_mtime
    from datetime import datetime
    now_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M")

    # Color for profit
    def profit_color(v):
        if v is None:
            return "#888"
        try:
            return "#ef4444" if float(v) > 0 else "#22c55e"  # China: red=up
        except (ValueError, TypeError):
            return "#888"

    def fmt(v, suffix="", is_pct=False):
        if v is None:
            return "—"
        try:
            return f"{float(v):.2f}{suffix}"
        except (ValueError, TypeError):
            return str(v)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Freqtrade 回测报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, sans-serif; background:#f5f5f5; padding:20px; }}
h1 {{ text-align:center; margin-bottom:5px; }}
.sub {{ text-align:center; color:#666; margin-bottom:20px; font-size:13px; }}
.section {{ background:#fff; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:16px; overflow:hidden; }}
.header {{ background:#1e293b; color:#fff; padding:10px 16px; font-size:15px; font-weight:600; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#f8fafc; padding:8px 10px; text-align:left; color:#64748b; border-bottom:2px solid #e2e8f0; font-size:12px; }}
td {{ padding:7px 10px; border-bottom:1px solid #f1f5f9; }}
tr:hover td {{ background:#f8fafc; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }}
.b-stable {{ background:#dbeafe; color:#1e40af; }}
.b-mid {{ background:#fef3c7; color:#92400e; }}
.b-meme {{ background:#fce7f3; color:#9d174d; }}
.b-1m {{ background:#e0e7ff; color:#3730a3; }}
.b-15m {{ background:#d1fae5; color:#065f46; }}
.p-pos {{ color:#ef4444; font-weight:600; }}
.p-neg {{ color:#22c55e; font-weight:600; }}
</style>
</head>
<body>
<h1>Freqtrade 批量回测报告</h1>
<p class="sub">数据源: Gate.io Spot · 时间范围: 2026-06-08 ~ 2026-06-21 (13天) · 生成: {now_str}</p>
<div class="section">
<div class="header">回测结果汇总 ({len(ok)} 成功 / {len(failed)} 失败)</div>
<table>
<thead><tr>
<th>策略</th><th>时间框架</th><th>类别</th><th>币种数</th><th>交易数</th><th>总收益率</th><th>胜率</th><th>最大回撤</th><th>夏普</th>
</tr></thead>
<tbody>
"""

    for r in ok:
        strategy = r["strategy"]
        tf = r["timeframe"]
        tf_badge = "b-1m" if tf == "1m" else "b-15m"

        cat_results = r.get("category_results", {})
        for cat_key in ["stable", "mid", "meme"]:
            if cat_key not in cat_results:
                continue
            cr = cat_results[cat_key]
            cat_badge = f"b-{cat_key}"
            profit = cr.get("profit_pct")
            profit_class = "p-pos" if profit and profit > 0 else "p-neg" if profit and profit < 0 else ""
            profit_str = fmt(profit, suffix="%") if profit is not None else "—"

            html += f"""<tr>
<td>{strategy}</td>
<td><span class="badge {tf_badge}">{tf}</span></td>
<td><span class="badge {cat_badge}">{cat_key}</span></td>
<td>{len(cr.get('pairs', []))}</td>
<td>{cr.get('trades', 0)}</td>
<td class="{profit_class}">{profit_str}</td>
<td>{fmt(r.get('win_rate'), suffix='%')}</td>
<td>{fmt(r.get('max_drawdown_pct'), suffix='%')}</td>
<td>{fmt(r.get('sharpe'))}</td>
</tr>
"""

    html += """</tbody></table></div>"""

    # Failed section
    if failed:
        html += f"""<div class="section">
<div class="header" style="background:#7f1d1d;">失败的回测 ({len(failed)})</div>
<table><thead><tr><th>策略</th><th>时间框架</th><th>错误</th></tr></thead><tbody>
"""
        for r in failed:
            html += f"""<tr><td>{r.get('strategy','?')}</td><td>{r.get('timeframe','?')}</td>
<td style="color:#dc2626;font-size:12px;">{str(r.get('error',''))[:200]}</td></tr>
"""
        html += "</tbody></table></div>"

    html += "</body></html>"

    return html


def main():
    if not LOG_DIR.exists():
        print(f"日志目录不存在: {LOG_DIR}")
        return

    # Find all log files
    log_files = list(LOG_DIR.glob("*.log"))
    # Filter to v4 format: {strategy}_{timeframe}.log (no category in name)
    log_files = [f for f in log_files if "_" in f.stem and f.stem.count("_") == 1]
    # Also accept files with strategy_timeframe format
    log_files = [f for f in log_files if not any(cat in f.stem for cat in ["稳定型", "中间币", "妖币"])]

    if not log_files:
        print(f"没有找到回测日志文件 (v4格式) 在 {LOG_DIR}")
        print("已有的日志文件:")
        for f in list(LOG_DIR.glob("*.log"))[:10]:
            print(f"  {f.name}")
        return

    print(f"找到 {len(log_files)} 个日志文件，开始解析...")

    results = []
    for log_path in sorted(log_files):
        print(f"解析: {log_path.name}...")
        r = parse_log(log_path)
        if r:
            results.append(r)

    print(f"解析完成: {len(results)} 个结果 ({sum(1 for r in results if not r.get('error'))} 成功)")

    # Aggregate by category
    results = aggregate_by_category(results)

    # Save JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"timestamp": "", "results": results}, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {RESULTS_PATH}")

    # Generate HTML
    html = build_html(results)
    report_path = PROJECT_DIR / "user_data" / "backtest_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    main()
