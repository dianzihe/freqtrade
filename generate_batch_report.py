#!/usr/bin/env python3
"""
批量回测综合报告生成器
读取 master_results.json，生成交互式 HTML 绩效报告
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_DIR = Path(r"F:\source\freqtrade")
RESULTS_FILE = PROJECT_DIR / "deliverables" / "backtest_batch_20260705" / "summary" / "master_results.json"
OUTPUT_HTML = PROJECT_DIR / "deliverables" / "backtest_batch_20260705" / "backtest_report.html"

COIN_CATEGORIES = {
    "主流": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中型": ["XCN/USDT"],
    "妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

TIMEFRAMES = ["1m", "5m", "15m"]


def get_category(pair):
    for cat, pairs in COIN_CATEGORIES.items():
        if pair in pairs:
            return cat
    return "其他"


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(result):
    m = result.get("metrics", {})
    td = result.get("trade_details", {})
    return {
        "strategy": result.get("strategy_name", ""),
        "tf": result.get("timeframe", ""),
        "pair": result.get("pair", ""),
        "category": get_category(result.get("pair", "")),
        "success": result.get("success", False),
        "elapsed": result.get("elapsed_sec", 0),
        "trades": m.get("total_trades", 0),
        "winning": m.get("winning_trades", 0),
        "losing": m.get("losing_trades", 0),
        "win_rate": m.get("win_rate", 0),
        "total_profit_pct": m.get("total_profit_pct", 0),
        "total_profit_abs": m.get("total_profit_abs", 0),
        "avg_profit_pct": m.get("avg_profit_pct", 0),
        "max_dd_pct": m.get("max_drawdown_pct", 0),
        "max_dd_abs": m.get("max_drawdown_abs", 0),
        "profit_factor": m.get("profit_factor", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "avg_duration": m.get("avg_duration", ""),
        "position_adds": td.get("position_adds", 0),
        "liquidation_risk": td.get("liquidation_risk", 0),
        "max_concurrent": td.get("max_concurrent_positions", 0),
        "entry_signals": td.get("entry_signals", 0),
        "exit_signals": td.get("exit_signals", 0),
        "error": result.get("error", result.get("stderr", "")),
    }


def build_html(results):
    metrics_list = [extract_metrics(r) for r in results]
    total = len(metrics_list)
    success = sum(1 for m in metrics_list if m["success"])
    failed = total - success
    has_trades = sum(1 for m in metrics_list if m["trades"] > 0)

    # Build summary by strategy
    strat_summary = defaultdict(lambda: {"total": 0, "success": 0, "profitable": 0, "total_trades": 0, "best_profit": -999, "worst_profit": 999, "best_pair": "", "worst_pair": ""})
    for m in metrics_list:
        s = strat_summary[m["strategy"]]
        s["total"] += 1
        if m["success"]:
            s["success"] += 1
        if m["total_profit_pct"] > 0:
            s["profitable"] += 1
        s["total_trades"] += m["trades"]
        if m["success"] and m["total_profit_pct"] > s["best_profit"]:
            s["best_profit"] = m["total_profit_pct"]
            s["best_pair"] = f"{m['pair']} ({m['tf']})"
        if m["success"] and m["total_profit_pct"] < s["worst_profit"]:
            s["worst_profit"] = m["total_profit_pct"]
            s["worst_pair"] = f"{m['pair']} ({m['tf']})"

    # Generate ranking table
    rank_rows = []
    for m in metrics_list:
        if m["success"] and m["trades"] > 0:
            rank_rows.append(m)
    rank_rows.sort(key=lambda x: x["total_profit_pct"], reverse=True)
    top10 = rank_rows[:10]
    bottom10 = rank_rows[-10:]

    # Generate chart data by category
    cat_data = defaultdict(lambda: defaultdict(list))
    for m in metrics_list:
        if m["success"]:
            cat = m["category"]
            cat_data[cat][m["tf"]].append({
                "strategy": m["strategy"],
                "pair": m["pair"].replace("/", "_"),
                "profit": m["total_profit_pct"],
                "trades": m["trades"],
                "win_rate": m["win_rate"],
                "max_dd": m["max_dd_pct"],
                "sharpe": m["sharpe"],
                "adds": m["position_adds"],
                "liq": m["liquidation_risk"],
            })

    # Generate heatmap data: strategy x pair for each timeframe
    heatmaps = {}
    for tf in TIMEFRAMES:
        strategies_seen = set()
        pairs_seen = set()
        grid = {}
        for m in metrics_list:
            if m["tf"] == tf and m["success"]:
                key = (m["strategy"], m["pair"])
                grid[key] = {
                    "profit": m["total_profit_pct"],
                    "trades": m["trades"],
                    "win_rate": m["win_rate"],
                    "max_dd": m["max_dd_pct"],
                }
                strategies_seen.add(m["strategy"])
                pairs_seen.add(m["pair"])
        heatmaps[tf] = {
            "strategies": sorted(strategies_seen),
            "pairs": sorted(pairs_seen),
            "grid": {f"{s}|{p}": grid.get((s, p), {}) for s in strategies_seen for p in pairs_seen},
        }

    # Liquidation risk summary
    liq_summary = defaultdict(lambda: {"total_trades": 0, "liq_events": 0, "total_adds": 0})
    for m in metrics_list:
        key = f"{m['strategy']}|{m['tf']}|{m['category']}"
        liq_summary[key]["total_trades"] += m["trades"]
        liq_summary[key]["liq_events"] += m["liquidation_risk"]
        liq_summary[key]["total_adds"] += m["position_adds"]

    # Build HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全策略批量回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f6fa; color: #2d3436; padding: 20px; }}
h1 {{ text-align: center; color: #1a1a2e; margin-bottom: 5px; font-size: 24px; }}
.subtitle {{ text-align: center; color: #636e72; margin-bottom: 30px; font-size: 14px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
.card {{ background: #fff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.card .num {{ font-size: 32px; font-weight: 700; }}
.card .label {{ font-size: 13px; color: #636e72; margin-top: 5px; }}
.card.green .num {{ color: #00b894; }}
.card.red .num {{ color: #d63031; }}
.card.blue .num {{ color: #0984e3; }}
.section {{ background: #fff; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 18px; margin-bottom: 15px; color: #1a1a2e; border-bottom: 2px solid #0984e3; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 6px; text-align: right; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; color: #636e72; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
td:first-child, th:first-child {{ text-align: left; }}
.profit-pos {{ color: #d63031; font-weight: 600; }}
.profit-neg {{ color: #00b894; font-weight: 600; }}
.warn {{ color: #e17055; }}
.danger {{ color: #d63031; font-weight: 700; }}
.chart-wrap {{ width: 100%; max-height: 400px; margin: 15px 0; }}
.filter-bar {{ display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }}
.filter-bar select, .filter-bar button {{ padding: 6px 12px; border: 1px solid #ddd; border-radius: 5px; background: #fff; font-size: 13px; cursor: pointer; }}
.filter-bar button.active {{ background: #0984e3; color: #fff; border-color: #0984e3; }}
.tab-nav {{ display: flex; gap: 5px; margin-bottom: 15px; flex-wrap: wrap; }}
.tab-nav button {{ padding: 8px 16px; border: 1px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer; font-size: 13px; }}
.tab-nav button.active {{ background: #0984e3; color: #fff; border-color: #0984e3; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.heatmap-cell {{ display: inline-block; width: 60px; height: 30px; text-align: center; line-height: 30px; font-size: 11px; border-radius: 3px; color: #fff; margin: 1px; }}
.heatmap-cell.empty {{ background: #eee; color: #999; }}
.tooltip {{ position: relative; cursor: help; }}
.tooltip:hover .tooltip-text {{ visibility: visible; }}
.tooltip-text {{ visibility: hidden; position: absolute; background: #333; color: #fff; padding: 5px 10px; border-radius: 4px; font-size: 11px; white-space: nowrap; z-index: 10; bottom: 100%; left: 50%; transform: translateX(-50%); }}
.conf-table {{ max-height: 500px; overflow-y: auto; }}
</style>
</head>
<body>

<h1>全策略批量回测报告</h1>
<div class="subtitle">数据源: gate.io | 回测区间: 2026-06-08 ~ 2026-06-27 | 生成时间: {now}</div>

<div class="summary-cards">
    <div class="card blue"><div class="num">{total}</div><div class="label">回测组合总数</div></div>
    <div class="card green"><div class="num">{success}</div><div class="label">成功完成</div></div>
    <div class="card red"><div class="num">{failed}</div><div class="label">失败</div></div>
    <div class="card green"><div class="num">{has_trades}</div><div class="label">有交易记录</div></div>
    <div class="card blue"><div class="num">{len(strat_summary)}</div><div class="label">策略数量</div></div>
</div>

<div class="section">
    <h2>策略整体表现汇总</h2>
    <table>
        <thead><tr>
            <th>策略</th><th>回测数</th><th>成功</th><th>盈利组合</th><th>总交易数</th><th>最佳收益%</th><th>最佳组合</th><th>最差收益%</th><th>最差组合</th>
        </tr></thead>
        <tbody>
"""
    for sname in sorted(strat_summary.keys()):
        s = strat_summary[sname]
        best_class = "profit-pos" if s["best_profit"] > 0 else "profit-neg"
        worst_class = "profit-pos" if s["worst_profit"] > 0 else "profit-neg"
        html += f"""            <tr>
                <td>{sname}</td>
                <td>{s['total']}</td>
                <td>{s['success']}</td>
                <td>{s['profitable']}</td>
                <td>{s['total_trades']}</td>
                <td class="{best_class}">{s['best_profit']:.2f}%</td>
                <td style="font-size:11px">{s['best_pair']}</td>
                <td class="{worst_class}">{s['worst_profit']:.2f}%</td>
                <td style="font-size:11px">{s['worst_pair']}</td>
            </tr>
"""
    html += """        </tbody>
    </table>
</div>

<div class="section">
    <h2>TOP 10 最佳收益组合</h2>
    <table>
        <thead><tr><th>排名</th><th>策略</th><th>周期</th><th>币种</th><th>类别</th><th>交易数</th><th>胜率%</th><th>总收益%</th><th>最大回撤%</th><th>盈亏比</th><th>夏普</th><th>加仓</th></tr></thead>
        <tbody>
"""
    for i, m in enumerate(top10):
        pf_class = "profit-pos" if m["total_profit_pct"] > 0 else "profit-neg"
        html += f"""            <tr>
                <td>{i+1}</td>
                <td>{m['strategy']}</td>
                <td>{m['tf']}</td>
                <td>{m['pair']}</td>
                <td>{m['category']}</td>
                <td>{m['trades']}</td>
                <td>{m['win_rate']:.1f}</td>
                <td class="{pf_class}">{m['total_profit_pct']:.2f}%</td>
                <td>{m['max_dd_pct']:.2f}%</td>
                <td>{m['profit_factor']:.2f}</td>
                <td>{m['sharpe']:.2f}</td>
                <td>{m['position_adds']}</td>
            </tr>
"""
    html += """        </tbody>
    </table>
</div>

<div class="section">
    <h2>BOTTOM 10 最差收益组合</h2>
    <table>
        <thead><tr><th>排名</th><th>策略</th><th>周期</th><th>币种</th><th>类别</th><th>交易数</th><th>胜率%</th><th>总收益%</th><th>最大回撤%</th><th>盈亏比</th><th>夏普</th><th>加仓</th></tr></thead>
        <tbody>
"""
    for i, m in enumerate(bottom10):
        pf_class = "profit-pos" if m["total_profit_pct"] > 0 else "profit-neg"
        html += f"""            <tr>
                <td>{i+1}</td>
                <td>{m['strategy']}</td>
                <td>{m['tf']}</td>
                <td>{m['pair']}</td>
                <td>{m['category']}</td>
                <td>{m['trades']}</td>
                <td>{m['win_rate']:.1f}</td>
                <td class="{pf_class}">{m['total_profit_pct']:.2f}%</td>
                <td>{m['max_dd_pct']:.2f}%</td>
                <td>{m['profit_factor']:.2f}</td>
                <td>{m['sharpe']:.2f}</td>
                <td>{m['position_adds']}</td>
            </tr>
"""
    html += """        </tbody>
    </table>
</div>

<div class="section">
    <h2>加仓触发记录 & 爆仓风险统计</h2>
    <table>
        <thead><tr><th>策略</th><th>周期</th><th>币种类别</th><th>总交易数</th><th>加仓触发</th><th>加仓率</th><th>爆仓风险</th><th>风险率</th></tr></thead>
        <tbody>
"""
    for key in sorted(liq_summary.keys()):
        s = liq_summary[key]
        parts = key.split("|")
        add_rate = (s["total_adds"] / s["total_trades"] * 100) if s["total_trades"] > 0 else 0
        liq_rate = (s["liq_events"] / s["total_trades"] * 100) if s["total_trades"] > 0 else 0
        liq_class = "danger" if liq_rate > 5 else ("warn" if liq_rate > 1 else "")
        html += f"""            <tr>
                <td>{parts[0]}</td>
                <td>{parts[1]}</td>
                <td>{parts[2]}</td>
                <td>{s['total_trades']}</td>
                <td>{s['total_adds']}</td>
                <td>{add_rate:.1f}%</td>
                <td class="{liq_class}">{s['liq_events']}</td>
                <td class="{liq_class}">{liq_rate:.1f}%</td>
            </tr>
"""
    html += """        </tbody>
    </table>
</div>
"""
    # Heatmap sections for each timeframe
    for tf in TIMEFRAMES:
        hm = heatmaps[tf]
        strategies = hm["strategies"]
        pairs = hm["pairs"]
        grid = hm["grid"]
        if not strategies or not pairs:
            continue

        # Profit heatmap
        html += f"""<div class="section">
    <h2>收益热力图 - {tf}</h2>
    <p style="color:#636e72;font-size:12px;margin-bottom:10px">红色=盈利，绿色=亏损，颜色越深越大。空白=无交易/失败。</p>
    <table style="font-size:11px">
        <thead><tr><th>策略 / 币种</th>"""
        for p in pairs:
            html += f"<th>{p.replace('/USDT','')}</th>"
        html += "</tr></thead><tbody>"
        for s in strategies:
            html += f"<tr><td style='font-weight:600'>{s}</td>"
            for p in pairs:
                cell = grid.get(f"{s}|{p}", {})
                profit = cell.get("profit", 0)
                trades = cell.get("trades", 0)
                if trades == 0:
                    html += "<td style='text-align:center;color:#ccc'>-</td>"
                else:
                    # Color scale: green for negative (loss), red for positive (profit) - Chinese convention
                    if profit >= 0:
                        intensity = min(abs(profit) / 20, 1)  # Cap at 20%
                        r, g, b = 214, int(48 * (1 - intensity)), int(49 * (1 - intensity))
                    else:
                        intensity = min(abs(profit) / 20, 1)
                        r, g, b = int(0 * (1 - intensity)), int(184 * (1 - intensity)), int(148 * (1 - intensity))
                    bg = f"rgb({r},{g},{b})"
                    text_color = "#fff" if intensity > 0.3 else "#333"
                    html += f"""<td style="background:{bg};color:{text_color};text-align:center;font-size:10px" title="{s} | {p} | 收益={profit:.1f}% | 交易={trades}">{profit:.1f}%</td>"""
            html += "</tr>"
        html += "</tbody></table></div>"

    # Profit chart by category
    html += """<div class="section">
    <h2>收益分布对比图</h2>
    <div class="tab-nav">
"""
    for i, cat in enumerate(["主流", "中型", "妖币"]):
        active = "active" if i == 0 else ""
        html += f"""        <button class="tab-btn {active}" onclick="switchChartTab('{cat}')">{cat}</button>
"""
    html += """    </div>
"""
    for i, cat in enumerate(["主流", "中型", "妖币"]):
        active = "active" if i == 0 else ""
        html += f"""    <div id="chart-{cat}" class="tab-content {active}">
        <div class="chart-wrap"><canvas id="canvas-{cat}"></canvas></div>
    </div>
"""

    # Chart data as JS
    html += """</div>

<script>
// Chart data
const chartData = {
"""
    for cat in ["主流", "中型", "妖币"]:
        html += f'    "{cat}": {{\n'
        cd = cat_data[cat]
        for tf in TIMEFRAMES:
            items = cd.get(tf, [])
            # Group by strategy
            strat_groups = defaultdict(list)
            for item in items:
                strat_groups[item["strategy"]].append(item["profit"])
            labels = sorted(strat_groups.keys())
            avg_profits = [sum(v) / len(v) for v in [strat_groups[s] for s in labels]]
            html += f'        "{tf}": {{ labels: {json.dumps(labels, ensure_ascii=False)}, data: {json.dumps([round(p, 2) for p in avg_profits])} }},\n'
        html += "    },\n"
    html += """};

let currentCat = '主流';
let chartInstance = null;

function drawChart(cat) {
    const ctx = document.getElementById('canvas-' + cat).getContext('2d');
    if (chartInstance) chartInstance.destroy();
    const cd = chartData[cat];
    const colors = ['#d63031', '#0984e3', '#00b894'];
    const datasets = Object.keys(cd).map((tf, i) => ({
        label: tf,
        data: cd[tf].data,
        backgroundColor: colors[i] + '80',
        borderColor: colors[i],
        borderWidth: 1
    }));

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: cd['1m'].labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: cat + '币种 - 各策略平均收益对比', font: { size: 14 } },
                legend: { position: 'top' }
            },
            scales: {
                y: { title: { display: true, text: '平均收益 %' } }
            }
        }
    });
}

function switchChartTab(cat) {
    currentCat = cat;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('chart-' + cat).classList.add('active');
    drawChart(cat);
}

document.addEventListener('DOMContentLoaded', () => drawChart('主流'));
</script>

<div class="section">
    <h2>完整回测明细表</h2>
    <div class="filter-bar">
        <select id="filter-strategy" onchange="applyFilter()">
            <option value="">全部策略</option>
"""
    all_strategies = sorted(set(m["strategy"] for m in metrics_list))
    for s in all_strategies:
        html += f'            <option value="{s}">{s}</option>\n'
    html += """        </select>
        <select id="filter-tf" onchange="applyFilter()">
            <option value="">全部周期</option>
"""
    for tf in TIMEFRAMES:
        html += f'            <option value="{tf}">{tf}</option>\n'
    html += """        </select>
        <select id="filter-cat" onchange="applyFilter()">
            <option value="">全部类别</option>
            <option value="主流">主流</option>
            <option value="中型">中型</option>
            <option value="妖币">妖币</option>
        </select>
        <button onclick="resetFilter()">重置筛选</button>
        <span id="filter-count" style="margin-left:auto;color:#636e72;font-size:13px"></span>
    </div>
    <div class="conf-table">
    <table id="detail-table">
        <thead><tr>
            <th>策略</th><th>周期</th><th>币种</th><th>类别</th><th>状态</th><th>交易数</th><th>胜率%</th><th>总收益%</th><th>平均收益%</th><th>最大回撤%</th><th>盈亏比</th><th>夏普</th><th>加仓</th><th>爆仓风险</th><th>最大并发</th><th>耗时s</th>
        </tr></thead>
        <tbody id="detail-body">
"""
    for m in sorted(metrics_list, key=lambda x: x["total_profit_pct"], reverse=True):
        pf_class = "profit-pos" if m["total_profit_pct"] > 0 else "profit-neg"
        status = "成功" if m["success"] else "失败"
        liq_class = "danger" if m["liquidation_risk"] > 0 else ""
        html += f"""            <tr data-strategy="{m['strategy']}" data-tf="{m['tf']}" data-cat="{m['category']}">
                <td>{m['strategy']}</td>
                <td>{m['tf']}</td>
                <td>{m['pair']}</td>
                <td>{m['category']}</td>
                <td>{status}</td>
                <td>{m['trades']}</td>
                <td>{m['win_rate']:.1f}</td>
                <td class="{pf_class}">{m['total_profit_pct']:.2f}%</td>
                <td class="{pf_class}">{m['avg_profit_pct']:.2f}%</td>
                <td>{m['max_dd_pct']:.2f}%</td>
                <td>{m['profit_factor']:.2f}</td>
                <td>{m['sharpe']:.2f}</td>
                <td>{m['position_adds']}</td>
                <td class="{liq_class}">{m['liquidation_risk']}</td>
                <td>{m['max_concurrent']}</td>
                <td>{m['elapsed']:.0f}</td>
            </tr>
"""
    html += """        </tbody>
    </table>
    </div>
</div>

<script>
function applyFilter() {
    const strategy = document.getElementById('filter-strategy').value;
    const tf = document.getElementById('filter-tf').value;
    const cat = document.getElementById('filter-cat').value;
    const rows = document.querySelectorAll('#detail-body tr');
    let count = 0;
    rows.forEach(row => {
        const show = (!strategy || row.dataset.strategy === strategy) &&
                     (!tf || row.dataset.tf === tf) &&
                     (!cat || row.dataset.cat === cat);
        row.style.display = show ? '' : 'none';
        if (show) count++;
    });
    document.getElementById('filter-count').textContent = `显示 ${count} 条`;
}
function resetFilter() {
    document.getElementById('filter-strategy').value = '';
    document.getElementById('filter-tf').value = '';
    document.getElementById('filter-cat').value = '';
    applyFilter();
}
applyFilter();
</script>

</body>
</html>"""
    return html


def main():
    if not RESULTS_FILE.exists():
        print(f"结果文件不存在: {RESULTS_FILE}")
        print("请先运行 batch_backtest_all_8strategies.py")
        return

    print(f"读取结果: {RESULTS_FILE}")
    results = load_results()
    print(f"共 {len(results)} 条回测记录")

    html = build_html(results)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"报告已生成: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
