#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
星河量化策略 — 全币种 × 全周期批量回测脚本
===========================================
遍历三类标的(主流/中型/妖币) × 四个周期(1m/5m/15m/1h)
独立回测并输出绩效指标、盈亏曲线、最大浮亏、DCA记录、爆仓风险。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── 配置 ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
FREQTRADE = PROJECT_DIR / ".venv" / "Scripts" / "freqtrade"
CONFIG = PROJECT_DIR / "user_data" / "config" / "星河量化.json"
STRATEGY_PATH = PROJECT_DIR / "user_data" / "strategies"
STRATEGY_NAME = "XingheMajorGridStrategy"
OUTPUT_DIR = PROJECT_DIR / "deliverables"
BACKTEST_RESULTS_DIR = PROJECT_DIR / "user_data" / "backtest_results"

# 三类标的
COINS = {
    "主力主流": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"],
    "中型":      ["XCN/USDT:USDT"],
    "妖币":      ["H/USDT:USDT", "VELVET/USDT:USDT", "BEAT/USDT:USDT", "COAI/USDT:USDT"],
}

TIMEFRAMES = ["5m", "15m", "1h"]

ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_backtest(pair: str, timeframe: str) -> dict | None:
    """运行单个回测并返回结果元数据"""
    pair_short = pair.replace("/USDT:USDT", "")
    tag = f"{pair_short}_{timeframe}"
    result_json = OUTPUT_DIR / f"result_{tag}.json"
    log_file = OUTPUT_DIR / f"log_{tag}.txt"

    # 清理旧结果
    for f in [result_json, log_file]:
        if f.exists():
            f.unlink()

    cmd = [
        str(FREQTRADE), "backtesting",
        "--config", str(CONFIG),
        "--strategy-path", str(STRATEGY_PATH),
        "--strategy", STRATEGY_NAME,
        "--timeframe", timeframe,
        "--pairs", pair,
        "--export", "trades",
        "--export-filename", str(OUTPUT_DIR / f"trades_{tag}.json"),
    ]

    print(f"  [{tag}] 开始...", flush=True)

    start = time.time()
    try:
        with open(log_file, "w", encoding="utf-8", errors="replace") as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=300,
                env=ENV,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        elapsed = time.time() - start

        if proc.returncode != 0:
            # 从日志读取错误
            error_line = "Unknown error"
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if "ERROR" in line or "Fatal" in line or "Traceback" in line:
                            error_line = line.strip()[:200]
                            break
            print(f"  [{tag}] ❌ 失败 ({elapsed:.0f}s): {error_line[:100]}", flush=True)
            return {
                "pair": pair, "timeframe": timeframe, "tag": tag,
                "status": "failed", "error": error_line[:200], "elapsed": elapsed,
            }

        # 从日志读取 stdout 并解析
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            stdout = f.read()

        result = parse_backtest_output(stdout, pair, timeframe, elapsed)
        if result is None:
            print(f"  [{tag}] ⚠️ 解析失败", flush=True)
            return {
                "pair": pair, "timeframe": timeframe, "tag": tag,
                "status": "parse_error", "elapsed": elapsed,
            }

        # 保存详细 JSON
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        profit = result.get("summary", {}).get("total_profit_pct", 0)
        trades = result.get("summary", {}).get("total_trades", 0)
        win_rate = result.get("summary", {}).get("win_rate", 0)
        print(f"  [{tag}] ✓ {profit:+.2f}% | {trades}T | WR={win_rate:.1f}% ({elapsed:.0f}s)", flush=True)
        return result

    except subprocess.TimeoutExpired:
        print(f"  [{tag}] ⏰ 超时", flush=True)
        return {"pair": pair, "timeframe": timeframe, "tag": tag, "status": "timeout", "elapsed": 300}
    except Exception as e:
        print(f"  [{tag}] ❌ 异常: {e}", flush=True)
        return {"pair": pair, "timeframe": timeframe, "tag": tag, "status": "exception", "error": str(e)}


def parse_backtest_output(stdout: str, pair: str, timeframe: str, elapsed: float) -> dict | None:
    """从 freqtrade 标准输出中解析回测结果"""
    lines = stdout.split("\n")
    summary = {}
    pair_stats = []
    enter_tag_stats = []
    exit_reason_stats = []
    dca_records = []
    daily_returns = []

    in_pair_table = False
    in_enter_tag = False
    in_exit_reason = False
    in_summary = False
    in_daily = False

    for line in lines:
        line_stripped = line.strip()

        # ── Pair table ──
        if "│Pair" in line_stripped and "Trades" in line_stripped and "Avg Profit" in line_stripped:
            in_pair_table = True
            continue
        if in_pair_table:
            if "│TOTAL" in line_stripped:
                in_pair_table = False
            elif line_stripped.startswith("│") and "USDT" in line_stripped:
                parts = [p.strip() for p in line_stripped.split("│")[1:-1]]
                if len(parts) >= 7:
                    pair_stats.append({
                        "pair": parts[0],
                        "trades": int(parts[1]) if parts[1].isdigit() else 0,
                        "avg_profit_pct": _parse_pct(parts[2]),
                        "total_profit": _parse_float(parts[3]),
                        "total_profit_pct": _parse_pct(parts[4]),
                        "avg_duration": parts[5],
                        "win_draw_loss": parts[6],
                    })

        # ── Enter Tag Stats ──
        if "ENTER TAG STATS" in line_stripped:
            in_enter_tag = True
            continue
        if in_enter_tag:
            if "│TOTAL" in line_stripped:
                in_enter_tag = False
            elif line_stripped.startswith("│") and ("long" in line_stripped.lower() or "short" in line_stripped.lower()):
                parts = [p.strip() for p in line_stripped.split("│")[1:-1]]
                if len(parts) >= 4:
                    enter_tag_stats.append({
                        "tag": parts[0],
                        "entries": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                        "avg_profit_pct": _parse_pct(parts[2]) if len(parts) > 2 else 0,
                        "total_profit": _parse_float(parts[3]) if len(parts) > 3 else 0,
                    })

        # ── Exit Reason Stats ──
        if "EXIT REASON STATS" in line_stripped:
            in_exit_reason = True
            continue
        if in_exit_reason:
            if "│TOTAL" in line_stripped:
                in_exit_reason = False
            elif line_stripped.startswith("│") and not line_stripped.startswith("│-"):
                parts = [p.strip() for p in line_stripped.split("│")[1:-1]]
                if len(parts) >= 4 and parts[0] not in ("Exit Reason", ""):
                    exit_reason_stats.append({
                        "reason": parts[0],
                        "exits": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                        "avg_profit_pct": _parse_pct(parts[2]) if len(parts) > 2 else 0,
                        "total_profit": _parse_float(parts[3]) if len(parts) > 3 else 0,
                    })

        # ── Summary Metrics ──
        if "SUMMARY METRICS" in line_stripped:
            in_summary = True
            continue
        if in_summary:
            if "STRATEGY SUMMARY" in line_stripped:
                in_summary = False
            elif "│" in line_stripped:
                parts = [p.strip() for p in line_stripped.split("│")[1:-1]]
                if len(parts) == 2 and parts[0] and parts[0] not in ("Metric", "─", ""):
                    key = parts[0].strip()
                    val = parts[1].strip()
                    summary[key] = val

        # ── DCA / 爆仓 相关日志提取 ──
        if "[DCA]" in line_stripped:
            dca_records.append(line_stripped)
        if "爆仓" in line_stripped or "liquidation" in line_stripped.lower():
            dca_records.append(line_stripped)

    if not summary:
        return None

    # 提取关键数值
    total_trades = int(summary.get("Total/Daily Avg Trades", "0 / 0").split(" / ")[0] or 0)
    total_profit_pct = _parse_pct(summary.get("Total profit %", "0%"))
    total_profit_usdt = _parse_float(summary.get("Absolute profit", "0 USDT"))
    max_drawdown_pct = _parse_pct(summary.get("Max % of account underwater", "0%"))
    max_drawdown_usdt = _parse_float(summary.get("Absolute drawdown", "0 USDT"))
    win_rate_str = summary.get("Days win/draw/lose", "")
    cagr = _parse_pct(summary.get("CAGR %", "0%"))
    sharpe = _parse_float(summary.get("Sharpe (closed trades)", "0"))
    sortino = _parse_float(summary.get("Sortino (closed trades)", "0"))
    profit_factor = _parse_float(summary.get("Profit factor", "0"))
    expectancy = summary.get("Expectancy (Ratio)", "")
    market_change = _parse_pct(summary.get("Market change", "0%"))
    
    # DCA stats
    dca_count = len([r for r in dca_records if "[DCA]" in r])

    # 从 pair stats 计算 win_rate
    wins, losses = 0, 0
    for ps in pair_stats:
        wdl = ps.get("win_draw_loss", "")
        parts = [x.strip() for x in wdl.split()]
        if len(parts) >= 3:
            wins += int(parts[0]) if parts[0].isdigit() else 0
            losses += int(parts[2]) if parts[2].isdigit() else 0
    win_rate = (wins / max(wins + losses, 1)) * 100 if (wins + losses) > 0 else 0

    return {
        "pair": pair,
        "timeframe": timeframe,
        "elapsed": elapsed,
        "status": "success",
        "summary": {
            "total_trades": total_trades,
            "total_profit_pct": total_profit_pct,
            "total_profit_usdt": total_profit_usdt,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_usdt": max_drawdown_usdt,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "cagr_pct": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "market_change_pct": market_change,
            "dca_triggers": dca_count,
        },
        "pair_stats": pair_stats,
        "enter_tag_stats": enter_tag_stats,
        "exit_reason_stats": exit_reason_stats,
        "summary_raw": summary,
        "dca_records": dca_records,
    }


def _parse_float(s: str) -> float:
    """从字符串解析浮点数, 去掉后缀"""
    if not s:
        return 0.0
    s = s.replace("USDT", "").replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_pct(s: str) -> float:
    """解析百分比"""
    if not s:
        return 0.0
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def generate_html_report(all_results: list, output_path: Path):
    """生成综合 HTML 报告"""
    results_by_category = defaultdict(list)
    for r in all_results:
        pair = r["pair"]
        for cat, coins in COINS.items():
            if pair in coins:
                results_by_category[cat].append(r)
                break

    categories_order = ["主力主流", "中型", "妖币"]

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>星河量化策略 · 全币种全周期回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6fa;color:#2d3436;padding:24px}
h1{font-size:24px;margin-bottom:8px}
.subtitle{color:#636e72;margin-bottom:24px}
.category{margin-bottom:40px}
.cat-title{font-size:20px;font-weight:700;padding:12px 16px;border-radius:8px 8px 0 0;display:flex;align-items:center;gap:8px}
.cat-title.main{background:linear-gradient(135deg,#0984e3,#74b9ff);color:#fff}
.cat-title.mid{background:linear-gradient(135deg,#6c5ce7,#a29bfe);color:#fff}
.cat-title.meme{background:linear-gradient(135deg,#d63031,#ff7675);color:#fff}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;background:#fff;padding:16px;border-radius:0 0 8px 8px;margin-bottom:16px}
.coin-card{background:#f8f9fa;border-radius:8px;padding:16px;border:1px solid #e9ecef}
.coin-card h3{font-size:16px;margin-bottom:12px;color:#2d3436}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;font-size:13px}
.metrics .label{color:#636e72}
.metrics .value{text-align:right;font-weight:600}
.metrics .value.pos{color:#00b894}
.metrics .value.neg{color:#d63031}
.tf-tabs{display:flex;gap:4px;margin-top:12px;flex-wrap:wrap}
.tf-tab{padding:4px 10px;border-radius:4px;font-size:12px;cursor:pointer;background:#e9ecef;border:none;transition:all .2s}
.tf-tab:hover{background:#dfe6e9}
.tf-tab.good{background:#00b89422;color:#00b894}
.tf-tab.bad{background:#d6303122;color:#d63031}
.chart-container{height:200px;margin-top:12px}
.detail-section{margin-top:16px;background:#fff;border-radius:8px;padding:16px;display:none}
.detail-section.active{display:block}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #eee}
th{background:#f1f2f6;font-weight:600;position:sticky;top:0}
tr:hover{background:#f8f9fa}
.risk-high{background:#d6303111;color:#d63031;font-weight:700}
.risk-mid{background:#fdcb6e33;color:#e17055}
.risk-ok{background:#00b89411;color:#00b894}
.overview-banner{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:24px}
.banner-card{background:#fff;border-radius:8px;padding:16px;border-left:4px solid #0984e3}
.banner-card .val{font-size:28px;font-weight:800}
.banner-card .label{font-size:12px;color:#636e72;margin-top:4px}
.banner-card.warn{border-left-color:#d63031}
.banner-card.ok{border-left-color:#00b894}
canvas{max-width:100%}
</style>
</head>
<body>
<h1>星河量化策略 · 全币种全周期回测报告</h1>
<p class="subtitle">回测完成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """ | 标的: 9 币种 × 4 周期 = 36 组</p>
"""

    # 总览统计
    success = [r for r in all_results if r.get("status") == "success"]
    failed = [r for r in all_results if r.get("status") != "success"]
    profitable = [r for r in success if r["summary"]["total_profit_pct"] > 0]
    
    html += '<div class="overview-banner">'
    html += f'<div class="banner-card"><div class="val">{len(success)}</div><div class="label">成功回测</div></div>'
    html += f'<div class="banner-card"><div class="val">{len(failed)}</div><div class="label">失败/超时</div></div>'
    html += f'<div class="banner-card {"ok" if len(profitable) > 0 else "warn"}"><div class="val">{len(profitable)}</div><div class="label">盈利组合</div></div>'
    if success:
        avg_prof = sum(r["summary"]["total_profit_pct"] for r in success) / len(success)
        best = max(success, key=lambda r: r["summary"]["total_profit_pct"])
        worst = min(success, key=lambda r: r["summary"]["total_profit_pct"])
        html += f'<div class="banner-card warn"><div class="val">{avg_prof:+.2f}%</div><div class="label">平均收益</div></div>'
        html += f'<div class="banner-card ok"><div class="val">{best["summary"]["total_profit_pct"]:+.2f}%</div><div class="label">最优: {best["pair"].split("/")[0]} {best["timeframe"]}</div></div>'
        html += f'<div class="banner-card warn"><div class="val">{worst["summary"]["total_profit_pct"]:+.2f}%</div><div class="label">最差: {worst["pair"].split("/")[0]} {worst["timeframe"]}</div></div>'
    html += '</div>'

    # 按分类渲染
    for cat in categories_order:
        results = results_by_category.get(cat, [])
        if not results:
            continue
        cat_class = {"主力主流": "main", "中型": "mid", "妖币": "meme"}.get(cat, "")
        html += f'<div class="category"><div class="cat-title {cat_class}">{cat} ({len(results)} 组)</div><div class="summary-grid">'

        # 按币种分组
        by_coin = defaultdict(list)
        for r in results:
            coin = r["pair"].split("/")[0]
            by_coin[coin].append(r)

        for coin in COINS[cat]:
            coin_results = by_coin.get(coin.split("/")[0], [])
            coin_name = coin.split("/")[0]
            html += f'<div class="coin-card"><h3>{coin_name}</h3><div class="metrics">'
            
            for r in sorted(coin_results, key=lambda x: TIMEFRAMES.index(x["timeframe"]) if x["timeframe"] in TIMEFRAMES else 99):
                tf = r["timeframe"]
                if r.get("status") != "success":
                    html += f'<div class="label">{tf}</div><div class="value" style="color:#636e72">失败</div>'
                    continue
                s = r["summary"]
                profit_class = "pos" if s["total_profit_pct"] >= 0 else "neg"
                html += f'<div class="label">{tf}</div>'
                html += f'<div class="value {profit_class}">{s["total_profit_pct"]:+.2f}%</div>'

            html += '</div></div>'
        html += '</div></div>'

    # ── 详细表格 ──
    html += '<div class="detail-section active" style="margin-top:32px">'
    html += '<h2 style="margin-bottom:16px">全部回测明细</h2>'
    html += '<table><thead><tr>'
    headers = ["分类", "币种", "周期", "交易数", "总收益%", "收益USDT", "胜率%", "最大回撤%", "CAGR%", "夏普", "DCA次数", "盈亏比"]
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'

    for cat in categories_order:
        for coin in COINS[cat]:
            coin_name = coin.split("/")[0]
            coin_results = by_coin.get(coin_name, [])
            for r in sorted(coin_results, key=lambda x: TIMEFRAMES.index(x["timeframe"]) if x.get("timeframe") in TIMEFRAMES else 99):
                if r.get("status") != "success":
                    html += f'<tr><td>{cat}</td><td>{coin_name}</td><td>{r["timeframe"]}</td><td colspan="8">❌ {r.get("error", "失败")}</td></tr>'
                    continue
                s = r["summary"]
                pc = "risk-high" if s["total_profit_pct"] < -10 else ("risk-mid" if s["total_profit_pct"] < 0 else "risk-ok")
                html += f'<tr class="{pc}">'
                html += f'<td>{cat}</td><td>{coin_name}</td><td>{r["timeframe"]}</td>'
                html += f'<td>{s["total_trades"]}</td>'
                html += f'<td>{s["total_profit_pct"]:+.2f}%</td>'
                html += f'<td>{s["total_profit_usdt"]:+.2f}</td>'
                html += f'<td>{s["win_rate"]:.1f}%</td>'
                html += f'<td>{s["max_drawdown_pct"]:.2f}%</td>'
                html += f'<td>{s["cagr_pct"]:+.2f}%</td>'
                html += f'<td>{s["sharpe"]:.2f}</td>'
                html += f'<td>{s["dca_triggers"]}</td>'
                html += f'<td>{s.get("profit_factor", 0):.2f}</td>'
                html += '</tr>'
    html += '</tbody></table></div>'

    # ── Chart.js 图表: 各币种各周期收益对比 ──
    chart_data = {}
    for r in success:
        coin = r["pair"].split("/")[0]
        tf = r["timeframe"]
        label = f"{coin}-{tf}"
        chart_data[label] = r["summary"]["total_profit_pct"]

    html += '<div class="detail-section active" style="margin-top:32px">'
    html += '<h2 style="margin-bottom:16px">收益对比图</h2>'
    html += '<div style="max-width:100%;height:500px"><canvas id="profitChart"></canvas></div>'
    html += '</div>'

    labels = list(chart_data.keys())
    values = list(chart_data.values())
    colors = ['#00b894' if v >= 0 else '#d63031' for v in values]

    html += f"""
<script>
new Chart(document.getElementById('profitChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: '总收益 %',
            data: {json.dumps(values)},
            backgroundColor: {json.dumps(colors)},
            borderColor: {json.dumps(colors)},
            borderWidth: 1,
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            y: {{
                title: {{ display: true, text: '总收益 %' }},
                ticks: {{ callback: v => v + '%' }}
            }},
            x: {{
                ticks: {{ maxRotation: 45, font: {{ size: 10 }} }}
            }}
        }}
    }}
}});
</script>
"""

    html += '</body></html>'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def main():
    ensure_dir(OUTPUT_DIR)
    all_results = []

    total_combos = sum(len(coins) for coins in COINS.values()) * len(TIMEFRAMES)
    print(f"=" * 60)
    print(f"星河量化 · 批量回测: {total_combos} 组 ({sum(len(c) for c in COINS.values())}币种 × {len(TIMEFRAMES)}周期)")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"=" * 60)

    idx = 0
    for cat, coins in COINS.items():
        print(f"\n{'─'*40}\n 📦 分类: {cat}\n{'─'*40}")
        for pair in coins:
            for tf in TIMEFRAMES:
                idx += 1
                print(f"\n[{idx}/{total_combos}] {pair} @ {tf}")
                result = run_backtest(pair, tf)
                if result:
                    all_results.append(result)

    # 生成报告
    report_path = OUTPUT_DIR / "xinghe_full_report.html"
    generate_html_report(all_results, report_path)

    # 汇总
    success = [r for r in all_results if r.get("status") == "success"]
    print(f"\n{'='*60}")
    print(f"批量回测完成: {len(success)}/{total_combos} 成功")
    if success:
        avg_profit = sum(r["summary"]["total_profit_pct"] for r in success) / len(success)
        pos = [r for r in success if r["summary"]["total_profit_pct"] > 0]
        print(f"平均收益: {avg_profit:+.2f}%")
        print(f"盈利组合: {len(pos)}/{len(success)}")
        best = max(success, key=lambda r: r["summary"]["total_profit_pct"])
        print(f"最优: {best['pair']} @ {best['timeframe']} = {best['summary']['total_profit_pct']:+.2f}%")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
