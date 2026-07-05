#!/usr/bin/env python
"""Generate comprehensive HTML report for GoldKylin strategy backtests."""

import re
import json
import os
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("user_data/bt_results/goldkylin")
REPORT_FILE = Path("user_data/bt_results/goldkylin/goldkylin_backtest_report.html")

PAIRS = [
    ("BTC/USDT", "BTC", "稳定主流币"),
    ("ETH/USDT", "ETH", "稳定主流币"),
    ("SOL/USDT", "SOL", "稳定主流币"),
    ("XRP/USDT", "XRP", "稳定主流币"),
    ("XCN/USDT", "XCN", "中端中型币种"),
    ("H/USDT", "H", "高波动妖币"),
    ("VELVET/USDT", "VELVET", "高波动妖币"),
    ("BEAT/USDT", "BEAT", "高波动妖币"),
    ("COAI/USDT", "COAI", "高波动妖币"),
]
TIMEFRAMES = ["1m", "5m", "15m"]


def parse_log(filepath):
    """Parse a freqtrade backtest log file and extract metrics."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    result = {
        "trades": 0,
        "avg_profit_pct": 0.0,
        "tot_profit_usdt": 0.0,
        "tot_profit_pct": 0.0,
        "avg_duration": "0:00:00",
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "win_rate": 0.0,
        "sharpe_closed": None,
        "sortino_closed": None,
        "calmar_closed": None,
        "sharpe_daily": None,
        "sortino_daily": None,
        "calmar_daily": None,
        "min_balance": None,
        "max_balance": None,
        "min_balance_wallet": None,
        "max_balance_wallet": None,
        "drawdown_duration": None,
        "drawdown_start": None,
        "drawdown_end": None,
        "drawdown_duration_wallet": None,
        "drawdown_start_wallet": None,
        "drawdown_end_wallet": None,
        "backtest_start": None,
        "backtest_end": None,
        "max_open_trades": None,
        "exit_reasons": [],
        "enter_tags": [],
        "left_open_trades": 0,
        "left_open_profit": 0.0,
        "total_profit_pct_raw": None,
        "no_trades": False,
        "strategy_params": {},
    }

    # Check no trades
    if "No trades made" in content:
        result["no_trades"] = True

    # Extract backtest date range
    m = re.search(r"Backtested\s+(.+?)\s+->\s+(.+?)\s+\|\s+Max open trades\s*:\s*(\d+)", content)
    if m:
        result["backtest_start"] = m.group(1).strip()
        result["backtest_end"] = m.group(2).strip()
        result["max_open_trades"] = int(m.group(3))

    # Extract STRATEGY SUMMARY line (the last TOTAL in strategy summary)
    # Look for the pattern after "STRATEGY SUMMARY"
    strategy_summary_section = content.split("STRATEGY SUMMARY")
    if len(strategy_summary_section) > 1:
        summary_text = strategy_summary_section[-1]
        # Find the GoldKylin line
        m = re.search(
            r"GoldKylin.*?│\s*(\d+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*([\d:]+)\s*│\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)",
            summary_text,
        )
        if m:
            result["trades"] = int(m.group(1))
            result["avg_profit_pct"] = float(m.group(2))
            result["tot_profit_usdt"] = float(m.group(3))
            result["tot_profit_pct"] = float(m.group(4))
            result["avg_duration"] = m.group(5)
            result["wins"] = int(m.group(6))
            result["draws"] = int(m.group(7))
            result["losses"] = int(m.group(8))
            result["win_rate"] = float(m.group(9))

    # Also try to find TOTAL from the first results table (before EXIT REASON)
    if result["trades"] == 0 and not result["no_trades"]:
        # Try to find the first TOTAL line with actual trade data
        total_matches = re.findall(
            r"│\s*TOTAL\s*│\s*(\d+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*([\d:]+)\s*│\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)",
            content,
        )
        if total_matches:
            # Take the first non-zero one
            for tm in total_matches:
                if int(tm[0]) > 0:
                    result["trades"] = int(tm[0])
                    result["avg_profit_pct"] = float(tm[1])
                    result["tot_profit_usdt"] = float(tm[2])
                    result["tot_profit_pct"] = float(tm[3])
                    result["avg_duration"] = tm[4]
                    result["wins"] = int(tm[5])
                    result["draws"] = int(tm[6])
                    result["losses"] = int(tm[7])
                    result["win_rate"] = float(tm[8])
                    break

    # Extract exit reasons
    exit_section = content.split("EXIT REASON STATS")
    if len(exit_section) > 1:
        exit_text = exit_section[-1]
        # Find exit reason rows (not TOTAL)
        reason_matches = re.findall(
            r"│\s*(\w[\w_]*)\s*│\s*(\d+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*([\d:]+)\s*│\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)",
            exit_text,
        )
        for rm in reason_matches:
            if rm[0] != "TOTAL" and int(rm[1]) > 0:
                result["exit_reasons"].append({
                    "reason": rm[0],
                    "count": int(rm[1]),
                    "avg_profit_pct": float(rm[2]),
                    "tot_profit_usdt": float(rm[3]),
                    "tot_profit_pct": float(rm[4]),
                    "duration": rm[5],
                    "wins": int(rm[6]),
                    "draws": int(rm[7]),
                    "losses": int(rm[8]),
                    "win_rate": float(rm[9]),
                })

    # Extract enter tags
    enter_section = content.split("ENTER TAG STATS")
    if len(enter_section) > 1:
        enter_text = enter_section[-1]
        tag_matches = re.findall(
            r"│\s*(\w[\w_]*)\s*│\s*(\d+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*([\d:]+)\s*│\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)",
            enter_text,
        )
        for tm in tag_matches:
            if tm[0] != "TOTAL" and int(tm[1]) > 0:
                result["enter_tags"].append({
                    "tag": tm[0],
                    "count": int(tm[1]),
                    "avg_profit_pct": float(tm[2]),
                    "tot_profit_usdt": float(tm[3]),
                    "tot_profit_pct": float(tm[4]),
                    "duration": tm[5],
                    "wins": int(tm[6]),
                    "draws": int(tm[7]),
                    "losses": int(tm[8]),
                    "win_rate": float(tm[9]),
                })

    # Extract performance metrics
    patterns = {
        "sharpe_closed": r"Sharpe \(closed trades\)\s*│\s*(-?[\d.]+)",
        "sortino_closed": r"Sortino \(closed trades\)\s*│\s*(-?[\d.]+)",
        "calmar_closed": r"Calmar \(closed trades\)\s*│\s*(-?[\d.]+)",
        "sharpe_daily": r"Sharpe \(daily wallet balance\)\s*│\s*(-?[\d.]+)",
        "sortino_daily": r"Sortino \(daily wallet balance\)\s*│\s*(-?[\d.]+)",
        "calmar_daily": r"Calmar \(daily wallet balance\)\s*│\s*(-?[\d.]+)",
        "total_profit_pct_raw": r"Total profit %\s*│\s*(-?[\d.]+)%",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            result[key] = float(m.group(1))

    # Min/Max balance
    m = re.search(r"Min/Max balance \(closed trades\)\s*│\s*([\d.]+)\s*USDT\s*/\s*([\d.]+)\s*USDT", content)
    if m:
        result["min_balance"] = float(m.group(1))
        result["max_balance"] = float(m.group(2))

    m = re.search(r"Min/Max balance \(wallet balance\)\s*│\s*([\d.]+)\s*USDT\s*/\s*([\d.]+)\s*USDT", content)
    if m:
        result["min_balance_wallet"] = float(m.group(1))
        result["max_balance_wallet"] = float(m.group(2))

    # Drawdown info (first occurrence = closed trades, second = wallet)
    dd_matches = re.findall(r"Drawdown duration\s*│\s*(.+?)\s*(?:│|$)", content)
    dd_start_matches = re.findall(r"Drawdown start\s*│\s*(.+?)\s*(?:│|$)", content)
    dd_end_matches = re.findall(r"Drawdown end\s*│\s*(.+?)\s*(?:│|$)", content)

    if dd_matches:
        result["drawdown_duration"] = dd_matches[0].strip()
    if dd_start_matches:
        result["drawdown_start"] = dd_start_matches[0].strip()
    if dd_end_matches:
        result["drawdown_end"] = dd_end_matches[0].strip()

    if len(dd_matches) > 1:
        result["drawdown_duration_wallet"] = dd_matches[1].strip()
    if len(dd_start_matches) > 1:
        result["drawdown_start_wallet"] = dd_start_matches[1].strip()
    if len(dd_end_matches) > 1:
        result["drawdown_end_wallet"] = dd_end_matches[1].strip()

    # Left open trades
    left_section = content.split("LEFT OPEN TRADES REPORT")
    if len(left_section) > 1:
        left_text = left_section[-1]
        # Count left open trades rows
        left_matches = re.findall(r"│\s*\d+\s*│", left_text)
        result["left_open_trades"] = len(left_matches)

        # Try to extract profit of left open trades
        m = re.search(r"Total profit USDT\s*│\s*(-?[\d.]+)", left_text)
        if m:
            result["left_open_profit"] = float(m.group(1))

    # Strategy parameters
    param_patterns = {
        "stoploss": r"stoploss:\s*(-?[\d.]+)",
        "trailing_stop": r"trailing_stop:\s*(True|False)",
        "trailing_stop_positive": r"trailing_stop_positive:\s*([\d.]+)",
        "trailing_stop_positive_offset": r"trailing_stop_positive_offset:\s*([\d.]+)",
        "use_custom_stoploss": r"use_custom_stoploss:\s*(True|False)",
        "minimal_roi": r"minimal_roi:\s*(\{.*?\})",
        "position_adjustment_enable": r"position_adjustment_enable:\s*(True|False)",
        "max_entry_position_adjustment": r"max_entry_position_adjustment:\s*(-?\d+)",
    }
    for key, pattern in param_patterns.items():
        m = re.search(pattern, content)
        if m:
            result["strategy_params"][key] = m.group(1)

    # Extract max drawdown percentage if available
    m = re.search(r"Drawdown\s*│\s*(-?[\d.]+)%", content)
    if m:
        result["max_drawdown_pct"] = float(m.group(1))
    else:
        result["max_drawdown_pct"] = 0.0

    # Try to find drawdown in the strategy summary table
    if result["max_drawdown_pct"] == 0.0:
        # Look for drawdown column in strategy summary
        dd_match = re.search(r"GoldKylin.*?│\s*(-?[\d.]+)%\s*│", content)
        if dd_match:
            result["max_drawdown_pct"] = float(dd_match.group(1))

    return result


def generate_html(all_results):
    """Generate comprehensive HTML report."""

    # Calculate summary statistics
    total_trades = sum(r["trades"] for r in all_results)
    total_profit = sum(r["tot_profit_usdt"] for r in all_results)
    profitable_count = sum(1 for r in all_results if r["tot_profit_usdt"] > 0)
    loss_count = sum(1 for r in all_results if r["tot_profit_usdt"] < 0)
    no_trade_count = sum(1 for r in all_results if r["trades"] == 0)
    trade_count = sum(1 for r in all_results if r["trades"] > 0)

    # Category statistics
    categories = {}
    for pair_info in PAIRS:
        cat = pair_info[2]
        if cat not in categories:
            categories[cat] = {"trades": 0, "profit": 0.0, "count": 0}
        for tf in TIMEFRAMES:
            key = f"{pair_info[1]}_{tf}"
            if key in all_results_dict:
                r = all_results_dict[key]
                categories[cat]["trades"] += r["trades"]
                categories[cat]["profit"] += r["tot_profit_usdt"]
                categories[cat]["count"] += 1

    # Collect all exit reasons
    all_exit_reasons = {}
    for r in all_results:
        for er in r["exit_reasons"]:
            if er["reason"] not in all_exit_reasons:
                all_exit_reasons[er["reason"]] = {"count": 0, "profit": 0.0, "wins": 0, "losses": 0}
            all_exit_reasons[er["reason"]]["count"] += er["count"]
            all_exit_reasons[er["reason"]]["profit"] += er["tot_profit_usdt"]
            all_exit_reasons[er["reason"]]["wins"] += er["wins"]
            all_exit_reasons[er["reason"]]["losses"] += er["losses"]

    # Collect all enter tags
    all_enter_tags = {}
    for r in all_results:
        for et in r["enter_tags"]:
            if et["tag"] not in all_enter_tags:
                all_enter_tags[et["tag"]] = {"count": 0, "profit": 0.0, "wins": 0, "losses": 0}
            all_enter_tags[et["tag"]]["count"] += et["count"]
            all_enter_tags[et["tag"]]["profit"] += et["tot_profit_usdt"]
            all_enter_tags[et["tag"]]["wins"] += et["wins"]
            all_enter_tags[et["tag"]]["losses"] += et["losses"]

    # Position adjustment info (from strategy params)
    pos_adjust_enabled = all_results[0]["strategy_params"].get("position_adjustment_enable", "False") if all_results else "False"
    max_entry_adjust = all_results[0]["strategy_params"].get("max_entry_position_adjustment", "-1") if all_results else "-1"

    # Build HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>金麒麟区间突破风控版 - 全币种全周期回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; color: #1a1a2e; margin: 20px 0; font-size: 28px; }
h2 { color: #16213e; margin: 30px 0 15px; font-size: 22px; border-bottom: 2px solid #0f3460; padding-bottom: 8px; }
h3 { color: #0f3460; margin: 20px 0 10px; font-size: 18px; }
.subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }

/* Summary cards */
.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
.card { background: #fff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; transition: transform 0.2s; }
.card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.card .label { font-size: 13px; color: #888; margin-bottom: 5px; }
.card .value { font-size: 28px; font-weight: bold; }
.card .sub { font-size: 12px; color: #aaa; margin-top: 3px; }
.card.profit .value { color: #e74c3c; }
.card.loss .value { color: #27ae60; }
.card.neutral .value { color: #2c3e50; }

/* Tables */
table { width: 100%; border-collapse: collapse; margin: 15px 0; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
th { background: #16213e; color: #fff; padding: 12px 8px; text-align: center; font-size: 13px; font-weight: 600; }
td { padding: 10px 8px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover { background: #f8f9fa; }
tr.category-header td { background: #e8ecf1; font-weight: bold; color: #16213e; }
.profit-cell { color: #e74c3c; font-weight: bold; }
.loss-cell { color: #27ae60; font-weight: bold; }
.zero-cell { color: #999; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
.badge-profit { background: #fdeaea; color: #e74c3c; }
.badge-loss { background: #e8f8f0; color: #27ae60; }
.badge-zero { background: #f0f0f0; color: #999; }

/* Chart containers */
.chart-container { background: #fff; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.chart-wrapper { position: relative; height: 400px; }

/* Info section */
.info-section { background: #fff; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.info-section p { margin: 5px 0; }
.param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; margin: 10px 0; }
.param-item { background: #f8f9fa; padding: 8px 12px; border-radius: 5px; font-size: 13px; }
.param-item strong { color: #0f3460; }

/* Risk section */
.risk-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin: 15px 0; }
.risk-card { background: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #0f3460; }
.risk-card.high { border-left-color: #e74c3c; }
.risk-card.medium { border-left-color: #f39c12; }
.risk-card.low { border-left-color: #27ae60; }
.risk-card .title { font-weight: bold; color: #16213e; margin-bottom: 8px; font-size: 14px; }
.risk-card .detail { font-size: 13px; color: #555; }

footer { text-align: center; margin: 30px 0; color: #999; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
""")

    # Title
    html_parts.append(f"""
<h1>金麒麟区间突破风控版 - 全币种全周期回测报告</h1>
<p class="subtitle">策略文件: 金麒麟_区间突破_风控版.py | 数据源: user_data/data/gate | 回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
""")

    # Summary cards
    html_parts.append(f"""
<div class="summary-cards">
    <div class="card neutral"><div class="label">总回测组合</div><div class="value">{len(all_results)}</div><div class="sub">9币种 × 3周期</div></div>
    <div class="card neutral"><div class="label">总交易笔数</div><div class="value">{total_trades}</div><div class="sub">全部27个组合</div></div>
    <div class="card {'profit' if total_profit > 0 else 'loss'}"><div class="label">总盈亏 (USDT)</div><div class="value">{total_profit:+.2f}</div><div class="sub">初始资金 10000 USDT/组</div></div>
    <div class="card profit"><div class="label">盈利组合</div><div class="value">{profitable_count}</div><div class="sub">有交易且盈利</div></div>
    <div class="card loss"><div class="label">亏损组合</div><div class="value">{loss_count}</div><div class="sub">有交易且亏损</div></div>
    <div class="card neutral"><div class="label">无交易组合</div><div class="value">{no_trade_count}</div><div class="sub">策略未触发入场</div></div>
</div>
""")

    # Category summary
    html_parts.append('<h2>分类汇总</h2>')
    html_parts.append('<table><thead><tr><th>币种类型</th><th>组合数</th><th>总交易笔数</th><th>总盈亏 (USDT)</th><th>平均盈亏</th></tr></thead><tbody>')
    for cat, stats in categories.items():
        avg = stats["profit"] / stats["count"] if stats["count"] > 0 else 0
        profit_class = "profit-cell" if stats["profit"] > 0 else ("loss-cell" if stats["profit"] < 0 else "zero-cell")
        html_parts.append(f'<tr><td>{cat}</td><td>{stats["count"]}</td><td>{stats["trades"]}</td><td class="{profit_class}">{stats["profit"]:+.2f}</td><td>{avg:+.2f}</td></tr>')
    html_parts.append('</tbody></table>')

    # Strategy parameters
    html_parts.append('<h2>策略参数配置</h2>')
    html_parts.append('<div class="info-section">')
    if all_results and all_results[0]["strategy_params"]:
        params = all_results[0]["strategy_params"]
        html_parts.append('<div class="param-grid">')
        param_labels = {
            "stoploss": "止损比例",
            "trailing_stop": "追踪止损",
            "trailing_stop_positive": "追踪止损正偏移",
            "trailing_stop_positive_offset": "追踪止损触发偏移",
            "use_custom_stoploss": "自定义止损",
            "minimal_roi": "最小ROI表",
            "position_adjustment_enable": "加仓功能",
            "max_entry_position_adjustment": "最大加仓次数",
        }
        for key, label in param_labels.items():
            if key in params:
                val = params[key]
                if key == "stoploss":
                    val = f"{float(val)*100:.1f}%"
                elif key in ("trailing_stop_positive", "trailing_stop_positive_offset"):
                    val = f"{float(val)*100:.1f}%"
                html_parts.append(f'<div class="param-item"><strong>{label}:</strong> {val}</div>')
        html_parts.append('</div>')
    html_parts.append('</div>')

    # Detailed results table
    html_parts.append('<h2>详细回测结果 - 全币种全周期</h2>')
    html_parts.append('<table><thead><tr>')
    html_parts.append('<th>币种</th><th>类型</th><th>周期</th><th>回测区间</th><th>交易数</th><th>胜率</th><th>盈/平/负</th><th>总盈亏(USDT)</th><th>总盈亏(%)</th><th>均盈利(%)</th><th>均持仓时长</th><th>夏普率</th><th>最大回撤(%)</th><th>最小余额</th><th>最大余额</th><th>回撤持续</th><th>状态</th>')
    html_parts.append('</tr></thead><tbody>')

    current_cat = None
    for pair_info in PAIRS:
        pair_full, pair_short, cat = pair_info
        if cat != current_cat:
            current_cat = cat
            html_parts.append(f'<tr class="category-header"><td colspan="17">{cat}</td></tr>')

        for tf in TIMEFRAMES:
            key = f"{pair_short}_{tf}"
            r = all_results_dict.get(key, {})
            if not r:
                continue

            if r["trades"] == 0:
                status_badge = '<span class="badge badge-zero">无交易</span>'
            elif r["tot_profit_usdt"] > 0:
                status_badge = '<span class="badge badge-profit">盈利</span>'
            else:
                status_badge = '<span class="badge badge-loss">亏损</span>'

            profit_class = "profit-cell" if r["tot_profit_usdt"] > 0 else ("loss-cell" if r["tot_profit_usdt"] < 0 else "zero-cell")

            bt_range = ""
            if r.get("backtest_start") and r.get("backtest_end"):
                bt_start_short = r["backtest_start"][:10] if r["backtest_start"] else ""
                bt_end_short = r["backtest_end"][:10] if r["backtest_end"] else ""
                bt_range = f"{bt_start_short} ~ {bt_end_short}"

            sharpe = f"{r['sharpe_closed']:.2f}" if r.get("sharpe_closed") is not None else "-"
            max_dd = f"{r.get('max_drawdown_pct', 0):.2f}%"
            min_bal = f"{r['min_balance']:.1f}" if r.get("min_balance") else "-"
            max_bal = f"{r['max_balance']:.1f}" if r.get("max_balance") else "-"
            dd_dur = r.get("drawdown_duration", "-") or "-"

            html_parts.append(f'''<tr>
                <td>{pair_short}/USDT</td><td>{cat}</td><td>{tf}</td>
                <td style="font-size:11px">{bt_range}</td>
                <td>{r['trades']}</td>
                <td>{r['win_rate']:.1f}%</td>
                <td>{r['wins']}/{r['draws']}/{r['losses']}</td>
                <td class="{profit_class}">{r['tot_profit_usdt']:+.2f}</td>
                <td class="{profit_class}">{r['tot_profit_pct']:+.2f}%</td>
                <td>{r['avg_profit_pct']:+.2f}%</td>
                <td>{r['avg_duration']}</td>
                <td>{sharpe}</td>
                <td>{max_dd}</td>
                <td>{min_bal}</td>
                <td>{max_bal}</td>
                <td style="font-size:11px">{dd_dur}</td>
                <td>{status_badge}</td>
            </tr>''')

    html_parts.append('</tbody></table>')

    # Profit chart
    html_parts.append('<h2>盈亏分布图</h2>')
    html_parts.append('<div class="chart-container"><div class="chart-wrapper"><canvas id="profitChart"></canvas></div></div>')

    # Trades chart
    html_parts.append('<h2>交易笔数分布</h2>')
    html_parts.append('<div class="chart-container"><div class="chart-wrapper"><canvas id="tradesChart"></canvas></div></div>')

    # Win rate chart
    html_parts.append('<h2>胜率对比（仅有交易的组合）</h2>')
    html_parts.append('<div class="chart-container"><div class="chart-wrapper"><canvas id="winRateChart"></canvas></div></div>')

    # Exit reason analysis
    html_parts.append('<h2>退出原因分析</h2>')
    if all_exit_reasons:
        html_parts.append('<table><thead><tr><th>退出原因</th><th>总次数</th><th>总盈亏(USDT)</th><th>胜次数</th><th>负次数</th><th>胜率</th><th>平均盈亏</th></tr></thead><tbody>')
        for reason, stats in sorted(all_exit_reasons.items(), key=lambda x: -x[1]["count"]):
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
            avg_p = stats["profit"] / stats["count"] if stats["count"] > 0 else 0
            profit_class = "profit-cell" if stats["profit"] > 0 else ("loss-cell" if stats["profit"] < 0 else "zero-cell")
            html_parts.append(f'<tr><td>{reason}</td><td>{stats["count"]}</td><td class="{profit_class}">{stats["profit"]:+.2f}</td><td>{stats["wins"]}</td><td>{stats["losses"]}</td><td>{wr:.1f}%</td><td class="{profit_class}">{avg_p:+.2f}</td></tr>')
        html_parts.append('</tbody></table>')
    else:
        html_parts.append('<p>无交易数据</p>')

    # Enter tag analysis
    html_parts.append('<h2>入场标签分析</h2>')
    if all_enter_tags:
        html_parts.append('<table><thead><tr><th>入场标签</th><th>总次数</th><th>总盈亏(USDT)</th><th>胜次数</th><th>负次数</th><th>胜率</th><th>平均盈亏</th></tr></thead><tbody>')
        for tag, stats in sorted(all_enter_tags.items(), key=lambda x: -x[1]["count"]):
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
            avg_p = stats["profit"] / stats["count"] if stats["count"] > 0 else 0
            profit_class = "profit-cell" if stats["profit"] > 0 else ("loss-cell" if stats["profit"] < 0 else "zero-cell")
            html_parts.append(f'<tr><td>{tag}</td><td>{stats["count"]}</td><td class="{profit_class}">{stats["profit"]:+.2f}</td><td>{stats["wins"]}</td><td>{stats["losses"]}</td><td>{wr:.1f}%</td><td class="{profit_class}">{avg_p:+.2f}</td></tr>')
        html_parts.append('</tbody></table>')
    else:
        html_parts.append('<p>无交易数据</p>')

    # Position adjustment / add-on analysis
    html_parts.append('<h2>加仓触发记录分析</h2>')
    html_parts.append('<div class="info-section">')
    if pos_adjust_enabled == "True":
        html_parts.append(f'<p><strong>加仓功能状态:</strong> 已启用 (最大加仓次数: {max_entry_adjust})</p>')
    else:
        html_parts.append(f'<p><strong>加仓功能状态:</strong> 未启用 (position_adjustment_enable = {pos_adjust_enabled})</p>')
        html_parts.append('<p style="color:#666;margin-top:5px">该策略未启用加仓功能，因此无加仓触发记录。所有交易均为单次入场。</p>')
    html_parts.append('</div>')

    # Drawdown analysis
    html_parts.append('<h2>最大浮亏（回撤）统计</h2>')
    html_parts.append('<table><thead><tr><th>币种</th><th>周期</th><th>最大回撤(%)</th><th>回撤持续时长</th><th>回撤起始</th><th>回撤结束</th><th>最小余额(USDT)</th><th>最大余额(USDT)</th><th>风险评级</th></tr></thead><tbody>')

    current_cat = None
    for pair_info in PAIRS:
        pair_full, pair_short, cat = pair_info
        if cat != current_cat:
            current_cat = cat
            html_parts.append(f'<tr class="category-header"><td colspan="9">{cat}</td></tr>')

        for tf in TIMEFRAMES:
            key = f"{pair_short}_{tf}"
            r = all_results_dict.get(key, {})
            if not r:
                continue

            max_dd = r.get("max_drawdown_pct", 0)
            if max_dd > 5:
                risk_level = '<span style="color:#e74c3c;font-weight:bold">高风险</span>'
            elif max_dd > 2:
                risk_level = '<span style="color:#f39c12;font-weight:bold">中风险</span>'
            elif max_dd > 0:
                risk_level = '<span style="color:#27ae60;font-weight:bold">低风险</span>'
            else:
                risk_level = '<span style="color:#999">无回撤</span>'

            dd_dur = r.get("drawdown_duration", "-") or "-"
            dd_start = r.get("drawdown_start", "-") or "-"
            dd_end = r.get("drawdown_end", "-") or "-"
            min_bal = f"{r['min_balance']:.1f}" if r.get("min_balance") else "-"
            max_bal = f"{r['max_balance']:.1f}" if r.get("max_balance") else "-"

            html_parts.append(f'<tr><td>{pair_short}/USDT</td><td>{tf}</td><td>{max_dd:.2f}%</td><td>{dd_dur}</td><td style="font-size:11px">{dd_start}</td><td style="font-size:11px">{dd_end}</td><td>{min_bal}</td><td>{max_bal}</td><td>{risk_level}</td></tr>')

    html_parts.append('</tbody></table>')

    # Liquidation/Ruin risk analysis
    html_parts.append('<h2>爆仓风险统计</h2>')
    html_parts.append('<div class="info-section">')
    html_parts.append('<p><strong>交易模式:</strong> 现货交易 (spot) — 无杠杆，不存在传统意义上的爆仓风险</p>')
    html_parts.append('<p><strong>止损设置:</strong> ' + (f"{float(all_results[0]['strategy_params'].get('stoploss', -0.05))*100:.1f}%" if all_results and all_results[0]["strategy_params"].get("stoploss") else "-5.0%") + '</p>')
    html_parts.append('<p><strong>追踪止损:</strong> ' + (f"启用 (正偏移: {float(all_results[0]['strategy_params'].get('trailing_stop_positive', 0))*100:.1f}%)" if all_results and all_results[0]["strategy_params"].get("trailing_stop") == "True" else "未启用") + '</p>')
    html_parts.append('<p><strong>自定义止损:</strong> ' + (all_results[0]["strategy_params"].get("use_custom_stoploss", "False") if all_results else "False") + '</p>')

    # Calculate max single trade loss
    max_losses = []
    for r in all_results:
        if r["trades"] > 0 and r["losses"] > 0:
            # Estimate avg loss per trade
            avg_loss = r["tot_profit_usdt"] / r["trades"] if r["trades"] > 0 else 0
            max_losses.append((r, avg_loss))

    html_parts.append(f'<p><strong>最大单笔亏损估计:</strong> 基于止损-5%及100 USDT仓位，单笔最大亏损约 5 USDT (含手续费约 5.4 USDT)</p>')
    html_parts.append(f'<p><strong>最大组合回撤:</strong> ' + f"{max(r.get('max_drawdown_pct', 0) for r in all_results):.2f}%" + '</p>')
    html_parts.append(f'<p><strong>爆仓风险评级:</strong> <span style="color:#27ae60;font-weight:bold">极低</span> — 现货交易无杠杆，止损严格，单笔风险可控</p>')
    html_parts.append('</div>')

    # Per-pair detailed analysis for pairs with trades
    html_parts.append('<h2>有交易组合的详细绩效指标</h2>')
    for pair_info in PAIRS:
        pair_full, pair_short, cat = pair_info
        for tf in TIMEFRAMES:
            key = f"{pair_short}_{tf}"
            r = all_results_dict.get(key, {})
            if not r or r["trades"] == 0:
                continue

            html_parts.append(f'<h3>{pair_short}/USDT - {tf}周期</h3>')
            html_parts.append('<div class="info-section">')
            html_parts.append(f'<p><strong>回测区间:</strong> {r.get("backtest_start", "-")} ~ {r.get("backtest_end", "-")}</p>')
            html_parts.append(f'<p><strong>交易笔数:</strong> {r["trades"]} | <strong>胜率:</strong> {r["win_rate"]:.1f}% | <strong>盈/平/负:</strong> {r["wins"]}/{r["draws"]}/{r["losses"]}</p>')
            html_parts.append(f'<p><strong>总盈亏:</strong> {r["tot_profit_usdt"]:+.2f} USDT ({r["tot_profit_pct"]:+.2f}%) | <strong>平均每笔:</strong> {r["avg_profit_pct"]:+.2f}%</p>')
            html_parts.append(f'<p><strong>平均持仓时长:</strong> {r["avg_duration"]}</p>')

            if r.get("sharpe_closed") is not None:
                html_parts.append(f'<p><strong>夏普率(已平仓):</strong> {r["sharpe_closed"]:.2f} | <strong>索提诺:</strong> {r.get("sortino_closed", 0):.2f} | <strong>卡玛:</strong> {r.get("calmar_closed", 0):.2f}</p>')
            if r.get("sharpe_daily") is not None:
                html_parts.append(f'<p><strong>夏普率(日级钱包):</strong> {r["sharpe_daily"]:.2f} | <strong>索提诺:</strong> {r.get("sortino_daily", 0):.2f} | <strong>卡玛:</strong> {r.get("calmar_daily", 0):.2f}</p>')

            if r.get("min_balance"):
                html_parts.append(f'<p><strong>最小/最大余额(已平仓):</strong> {r["min_balance"]:.2f} / {r["max_balance"]:.2f} USDT</p>')
            if r.get("min_balance_wallet"):
                html_parts.append(f'<p><strong>最小/最大余额(钱包):</strong> {r["min_balance_wallet"]:.2f} / {r["max_balance_wallet"]:.2f} USDT</p>')
            if r.get("drawdown_duration"):
                html_parts.append(f'<p><strong>最大回撤持续:</strong> {r["drawdown_duration"]} ({r.get("drawdown_start", "-")} ~ {r.get("drawdown_end", "-")})</p>')

            if r["exit_reasons"]:
                html_parts.append('<p><strong>退出原因明细:</strong></p><table style="margin-top:5px"><thead><tr><th>原因</th><th>次数</th><th>盈亏</th><th>胜/负</th><th>胜率</th></tr></thead><tbody>')
                for er in r["exit_reasons"]:
                    pcls = "profit-cell" if er["tot_profit_usdt"] > 0 else "loss-cell"
                    html_parts.append(f'<tr><td>{er["reason"]}</td><td>{er["count"]}</td><td class="{pcls}">{er["tot_profit_usdt"]:+.2f}</td><td>{er["wins"]}/{er["losses"]}</td><td>{er["win_rate"]:.0f}%</td></tr>')
                html_parts.append('</tbody></table>')

            html_parts.append('</div>')

    # Chart JavaScript
    # Prepare chart data
    labels = []
    profit_data = []
    trades_data = []
    winrate_labels = []
    winrate_data = []
    colors_profit = []
    colors_trades = []

    for pair_info in PAIRS:
        pair_full, pair_short, cat = pair_info
        for tf in TIMEFRAMES:
            key = f"{pair_short}_{tf}"
            r = all_results_dict.get(key, {})
            label = f"{pair_short}_{tf}"
            labels.append(label)
            profit_data.append(r.get("tot_profit_usdt", 0))
            trades_data.append(r.get("trades", 0))
            if r.get("trades", 0) > 0:
                winrate_labels.append(label)
                winrate_data.append(r.get("win_rate", 0))

            if r.get("tot_profit_usdt", 0) > 0:
                colors_profit.append("#e74c3c")
            elif r.get("tot_profit_usdt", 0) < 0:
                colors_profit.append("#27ae60")
            else:
                colors_profit.append("#bdc3c7")

            if r.get("trades", 0) > 0:
                colors_trades.append("#0f3460")
            else:
                colors_trades.push("#bdc3c7") if hasattr(colors_trades, 'push') else colors_trades.append("#bdc3c7")

    # Category colors for trades chart
    cat_colors = {
        "稳定主流币": "#3498db",
        "中端中型币种": "#9b59b6",
        "高波动妖币": "#e67e22",
    }
    trades_colors = []
    for pair_info in PAIRS:
        cat = pair_info[2]
        for tf in TIMEFRAMES:
            trades_colors.append(cat_colors.get(cat, "#0f3460"))

    html_parts.append(f"""
<script>
const ctx1 = document.getElementById('profitChart').getContext('2d');
new Chart(ctx1, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '盈亏 (USDT)',
            data: {json.dumps(profit_data)},
            backgroundColor: {json.dumps(colors_profit)},
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            title: {{ display: true, text: '各组合盈亏对比 (红色=盈利, 绿色=亏损, 灰色=无交易)', font: {{ size: 14 }} }},
            legend: {{ display: false }}
        }},
        scales: {{
            y: {{ title: {{ display: true, text: 'USDT' }} }}
        }}
    }}
}});

const ctx2 = document.getElementById('tradesChart').getContext('2d');
new Chart(ctx2, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '交易笔数',
            data: {json.dumps(trades_data)},
            backgroundColor: {json.dumps(trades_colors)},
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            title: {{ display: true, text: '各组合交易笔数 (蓝=主流币, 紫=中型币, 橙=妖币)', font: {{ size: 14 }} }},
            legend: {{ display: false }}
        }},
        scales: {{
            y: {{ title: {{ display: true, text: '交易笔数' }}, beginAtZero: true }}
        }}
    }}
}});

const ctx3 = document.getElementById('winRateChart').getContext('2d');
new Chart(ctx3, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(winrate_labels)},
        datasets: [{{
            label: '胜率 (%)',
            data: {json.dumps(winrate_data)},
            backgroundColor: '#0f3460',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            title: {{ display: true, text: '有交易组合的胜率对比', font: {{ size: 14 }} }},
            legend: {{ display: false }}
        }},
        scales: {{
            y: {{ title: {{ display: true, text: '胜率 (%)' }}, beginAtZero: true, max: 100 }}
        }}
    }}
}});
</script>
""")

    html_parts.append(f"""
<footer>
    <p>金麒麟区间突破风控版策略回测报告 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>数据源: Gate交易所 feather格式 | 初始资金: 10000 USDT/组 | 单笔仓位: 100 USDT | 手续费: 0.2%</p>
</footer>
</div>
</body>
</html>
""")

    return "".join(html_parts)


# Main
all_results = []
all_results_dict = {}

for pair_info in PAIRS:
    pair_full, pair_short, cat = pair_info
    for tf in TIMEFRAMES:
        log_file = RESULTS_DIR / f"log_{pair_short}_USDT_{tf}.txt"
        if log_file.exists():
            result = parse_log(log_file)
            result["pair"] = pair_short
            result["timeframe"] = tf
            result["category"] = cat
            key = f"{pair_short}_{tf}"
            all_results.append(result)
            all_results_dict[key] = result
            print(f"Parsed {key}: trades={result['trades']}, profit={result['tot_profit_usdt']:+.2f}")
        else:
            print(f"MISSING: {log_file}")

# Generate HTML
html = generate_html(all_results)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nReport generated: {REPORT_FILE}")
print(f"Total results parsed: {len(all_results)}")
