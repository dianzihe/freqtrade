"""
Generate intraday analysis HTML report from JSON data.
Separated to avoid large string issues in main analysis script.
"""

import json
import pandas as pd
from pathlib import Path

INPUT_JSON = Path('E:/source/freqtrade/user_data/intraday_analysis_data.json')
OUTPUT_HTML = Path('E:/source/freqtrade/user_data/intraday_analysis_report.html')


def generate_html_report(data):
    """Generate the full HTML report."""
    hourly = pd.DataFrame(data['hourly_returns'])
    sessions = data['sessions']
    early_pred = data['early_hour_predictions']
    vol_analysis = data['volatility_analysis']

    # Hourly chart data
    hours_labels = [f"{int(h)}:00" for h in hourly['hour']]
    returns_data = hourly['avg_return_pct'].tolist()
    return_colors = ['#ef4444' if v > 0 else '#22c55e' for v in returns_data]

    winrate_data = [0] * 24
    for item in data['hourly_winrate']:
        winrate_data[int(item['hour'])] = item['win_rate_pct']

    volatility_data = [0] * 24
    for item in data['hourly_volatility']:
        volatility_data[int(item['hour'])] = item['avg_volatility_pct']

    volume_data = [0] * 24
    for item in data['hourly_volume']:
        volume_data[int(item['hour'])] = item['avg_volume']

    # Session chart data
    sessions_labels = json.dumps([s['session'] for s in sessions], ensure_ascii=False)
    sessions_returns = json.dumps([s['avg_cumulative_return_pct'] for s in sessions])
    sessions_winrate = json.dumps([s['win_rate_pct'] for s in sessions])

    # Early hour chart data
    early_labels = json.dumps([p['n_hours'] for p in early_pred])
    early_corr = json.dumps([p['correlation'] for p in early_pred])
    early_acc = json.dumps([p['direction_accuracy_pct'] for p in early_pred])

    # Insight values (JSON serializes keys as strings)
    best_hour = int(data['best_hour'])
    worst_hour = int(data['worst_hour'])
    best_hour_val = hourly[hourly['hour'] == best_hour]['avg_return_pct'].values[0]
    worst_hour_val = hourly[hourly['hour'] == worst_hour]['avg_return_pct'].values[0]
    wr_best = winrate_data[best_hour]
    wr_worst = winrate_data[worst_hour]
    beijing_best = (best_hour + 8) % 24
    beijing_worst = (worst_hour + 8) % 24
    # Find the earliest N hours with accuracy > 60% (meaningful prediction)
    meaningful_n = next((p for p in early_pred if p['direction_accuracy_pct'] > 60), early_pred[3])
    dir_consistency = vol_analysis['direction_consistency_pct']
    vol_peak = volume_data.index(max(volume_data))
    early_4h_corr = early_pred[3]['correlation']
    early_4h_acc = early_pred[3]['direction_accuracy_pct']
    early_8h_corr = early_pred[7]['correlation']
    early_8h_acc = early_pred[7]['direction_accuracy_pct']

    # Build session table rows
    session_rows = ''
    for s in sessions:
        cls = 'positive' if s['avg_cumulative_return_pct'] > 0 else 'negative'
        session_rows += f'''<tr>
            <td>{s['session']}</td>
            <td>{s['hours']}</td>
            <td class="{cls}">{s['avg_cumulative_return_pct']:.4f}%</td>
            <td>{s['std_return_pct']:.4f}%</td>
            <td>{s['win_rate_pct']:.1f}%</td>
            <td>{s['count']:,}</td>
        </tr>
'''

    # Build all chart JS
    chart_js = f'''
<script>
const gridColor = 'rgba(48, 54, 61, 0.8)';
const textColor = '#8b949e';

// Hourly Return Chart
new Chart(document.getElementById('chartHourlyReturn'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(hours_labels)},
        datasets: [{{
            label: '平均收益率 (%)',
            data: {json.dumps(returns_data)},
            backgroundColor: {json.dumps(return_colors)},
            borderColor: {json.dumps(return_colors)},
            borderWidth: 1,
            borderRadius: 4
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, maxRotation: 0 }} }},
            y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toFixed(4) + '%' }} }}
        }}
    }}
}});

// Win Rate Chart
new Chart(document.getElementById('chartWinRate'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(hours_labels)},
        datasets: [{{
            label: '胜率 (%)',
            data: {json.dumps(winrate_data)},
            backgroundColor: 'rgba(88, 166, 255, 0.7)',
            borderColor: '#58a6ff',
            borderWidth: 1,
            borderRadius: 4
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, maxRotation: 0 }} }},
            y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toFixed(1) + '%' }}, min: 45 }}
        }}
    }}
}});

// Volatility Chart
new Chart(document.getElementById('chartVolatility'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(hours_labels)},
        datasets: [{{
            label: '平均振幅 (%)',
            data: {json.dumps(volatility_data)},
            backgroundColor: 'rgba(240, 136, 62, 0.7)',
            borderColor: '#f0883e',
            borderWidth: 1,
            borderRadius: 4
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, maxRotation: 0 }} }},
            y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toFixed(2) + '%' }} }}
        }}
    }}
}});

// Volume Chart
new Chart(document.getElementById('chartVolume'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(hours_labels)},
        datasets: [{{
            label: '平均成交量',
            data: {json.dumps(volume_data)},
            backgroundColor: 'rgba(63, 185, 80, 0.5)',
            borderColor: '#3fb950',
            borderWidth: 1,
            borderRadius: 4
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, maxRotation: 0 }} }},
            y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor }} }}
        }}
    }}
}});

// Sessions Chart
new Chart(document.getElementById('chartSessions'), {{
    type: 'bar',
    data: {{
        labels: {sessions_labels},
        datasets: [
            {{
                label: '平均累计收益 (%)',
                data: {sessions_returns},
                backgroundColor: 'rgba(239, 68, 68, 0.6)',
                borderColor: '#ef4444',
                borderWidth: 1,
                borderRadius: 4,
                yAxisID: 'y'
            }},
            {{
                label: '胜率 (%)',
                data: {sessions_winrate},
                type: 'line',
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: '#58a6ff',
                yAxisID: 'y1'
            }}
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, maxRotation: 30 }} }},
            y: {{ type: 'linear', position: 'left', grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toFixed(3) + '%' }}, title: {{ display: true, text: '收益 %', color: textColor }} }},
            y1: {{ type: 'linear', position: 'right', grid: {{ display: false }}, ticks: {{ color: '#58a6ff', callback: v => v.toFixed(1) + '%' }}, min: 45, max: 60, title: {{ display: true, text: '胜率 %', color: '#58a6ff' }} }}
        }}
    }}
}});

// Early Hour Correlation
new Chart(document.getElementById('chartEarlyCorr'), {{
    type: 'line',
    data: {{
        labels: {early_labels},
        datasets: [{{
            label: '前N小时与全日收益相关性',
            data: {early_corr},
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            borderWidth: 2.5,
            fill: true,
            pointRadius: 5,
            pointBackgroundColor: '#58a6ff',
            tension: 0.3
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor }}, title: {{ display: true, text: 'N 小时', color: textColor }} }},
            y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor }}, title: {{ display: true, text: 'Pearson 相关系数', color: textColor }}, min: 0 }}
        }}
    }}
}});

// Early Hour Accuracy
new Chart(document.getElementById('chartEarlyAcc'), {{
    type: 'line',
    data: {{
        labels: {early_labels},
        datasets: [{{
            label: '方向预测准确率 (%)',
            data: {early_acc},
            borderColor: '#3fb950',
            backgroundColor: 'rgba(63, 185, 80, 0.1)',
            borderWidth: 2.5,
            fill: true,
            pointRadius: 5,
            pointBackgroundColor: '#3fb950',
            tension: 0.3
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor }}, title: {{ display: true, text: 'N 小时', color: textColor }} }},
            y: {{ grid: {{ color: gridColor }}, ticks: {{ color: textColor, callback: v => v.toFixed(1) + '%' }}, title: {{ display: true, text: '准确率 %', color: textColor }}, min: 45 }}
        }}
    }}
}});
</script>
'''

    # Build the full HTML
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>加密货币日内时间段趋势分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 24px; }}
h1 {{ text-align: center; color: #58a6ff; margin-bottom: 8px; font-size: 28px; }}
.subtitle {{ text-align: center; color: #8b949e; margin-bottom: 32px; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
.full {{ grid-column: 1 / -1; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }}
.card h2 {{ color: #e6edf3; font-size: 16px; margin-bottom: 16px; border-bottom: 1px solid #21262d; padding-bottom: 10px; }}
.chart-wrap {{ position: relative; width: 100%; }}
.chart-wrap canvas {{ width: 100% !important; }}
.insight {{ background: #1a2332; border-left: 3px solid #58a6ff; padding: 12px 16px; border-radius: 6px; margin-top: 12px; font-size: 13px; line-height: 1.6; }}
.insight.warn {{ border-left-color: #f0883e; background: #1f1a16; }}
.insight.good {{ border-left-color: #3fb950; background: #162318; }}
.insight strong {{ color: #e6edf3; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }}
th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #21262d; }}
th {{ color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
th:first-child, td:first-child {{ text-align: left; }}
.positive {{ color: #ef4444; }}
.negative {{ color: #22c55e; }}
.summary-box {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; text-align: center; }}
.stat-card .value {{ font-size: 28px; font-weight: 700; }}
.stat-card .label {{ color: #8b949e; font-size: 12px; margin-top: 4px; }}
</style>
</head>
<body>

<h1>加密货币日内时间段趋势分析报告</h1>
<p class="subtitle">基于 11,213,289 条 1分钟K线 | 531 个交易对 | 6 个交易所 | 2026-06-08 ~ 2026-06-21 UTC</p>

<div class="summary-box">
    <div class="stat-card">
        <div class="value" style="color:#58a6ff">{best_hour}:00</div>
        <div class="label">最佳交易时段 (UTC)</div>
    </div>
    <div class="stat-card">
        <div class="value" style="color:#f0883e">{worst_hour}:00</div>
        <div class="label">最差交易时段 (UTC)</div>
    </div>
    <div class="stat-card">
        <div class="value" style="color:#ef4444">{early_4h_acc:.1f}%</div>
        <div class="label">前4小时方向预测准确率</div>
    </div>
    <div class="stat-card">
        <div class="value" style="color:#3fb950">{dir_consistency:.1f}%</div>
        <div class="label">上午→下午方向一致性</div>
    </div>
</div>

<div class="grid">
    <div class="card">
        <h2>各时段平均收益率 (UTC)</h2>
        <div class="chart-wrap"><canvas id="chartHourlyReturn"></canvas></div>
    </div>
    <div class="card">
        <h2>各时段胜率 (上涨概率 %)</h2>
        <div class="chart-wrap"><canvas id="chartWinRate"></canvas></div>
    </div>
</div>

<div class="grid">
    <div class="card">
        <h2>各时段波动率 (平均振幅 %)</h2>
        <div class="chart-wrap"><canvas id="chartVolatility"></canvas></div>
    </div>
    <div class="card">
        <h2>各时段成交量分布</h2>
        <div class="chart-wrap"><canvas id="chartVolume"></canvas></div>
    </div>
</div>

<div class="grid">
    <div class="card full">
        <h2>交易时段综合表现</h2>
        <div class="chart-wrap"><canvas id="chartSessions"></canvas></div>
    </div>
</div>

<div class="grid">
    <div class="card">
        <h2>前N小时 vs 全日收益相关性</h2>
        <div class="chart-wrap"><canvas id="chartEarlyCorr"></canvas></div>
        <div class="insight">
            <strong>解读：</strong>相关性越高，说明前N小时的走势对全天走势的预测能力越强。
            当相关性 > 0.6 时，前N小时的方向可作为全天方向的重要参考。<br>
            当前 4小时相关性: <strong>{early_4h_corr:.3f}</strong> | 8小时相关性: <strong>{early_8h_corr:.3f}</strong>
        </div>
    </div>
    <div class="card">
        <h2>前N小时方向预测准确率</h2>
        <div class="chart-wrap"><canvas id="chartEarlyAcc"></canvas></div>
        <div class="insight good">
            <strong>解读：</strong>如果前N小时上涨/下跌，全天最终也同向的概率。
            > 60% 表明有显著的预测价值。<br>
            当前 4小时准确率: <strong>{early_4h_acc:.1f}%</strong> | 8小时准确率: <strong>{early_8h_acc:.1f}%</strong>
        </div>
    </div>
</div>

<div class="grid">
    <div class="card full">
        <h2>交易时段详细数据</h2>
        <table>
            <thead>
                <tr><th>时段</th><th>UTC时间</th><th>平均累计收益(%)</th><th>标准差(%)</th><th>胜率(%)</th><th>样本数</th></tr>
            </thead>
            <tbody>
{session_rows}
            </tbody>
        </table>
    </div>
</div>

<div class="card" style="margin-bottom:24px">
    <h2>关键发现与交易建议</h2>
    <div class="insight good">
        <strong>1. 最佳交易时段：</strong>UTC {best_hour}:00，平均收益率 {best_hour_val:.4f}%，胜率 {wr_best:.1f}%。<br>
        该时段对应北京时间 {beijing_best}:00（UTC+8），建议重点关注此时间窗口的交易机会。
    </div>
    <div class="insight warn">
        <strong>2. 风险时段：</strong>UTC {worst_hour}:00，平均收益率 {worst_hour_val:.4f}%，胜率仅 {wr_worst:.1f}%。<br>
        该时段对应北京时间 {beijing_worst}:00，建议减少仓位或避免在此时间段开仓。
    </div>
    <div class="insight">
        <strong>3. 预测能力：</strong>前 {meaningful_n["n_hours"]} 小时走势即可对全天方向做出有效预测，准确率达 {meaningful_n["direction_accuracy_pct"]:.1f}%，相关性 {meaningful_n["correlation"]:.3f}。<br>
        随着时间推进预测精度持续提升：4小时→{early_4h_acc:.1f}%，8小时→{early_8h_acc:.1f}%，12小时→{early_pred[11]["direction_accuracy_pct"]:.1f}%。
    </div>
    <div class="insight good">
        <strong>4. 上午→下午持续性：</strong>上午（0-8 UTC）收益与下午（8-16 UTC）收益高度相关（r={vol_analysis["morning_afternoon_return_corr"]:.3f}），<br>
        但方向一致率仅 {dir_consistency:.1f}%，说明趋势在方向层面不具有简单的跟随性，但涨幅/跌幅的幅度高度一致。
    </div>
    <div class="insight">
        <strong>5. 成交量规律：</strong>UTC {vol_peak}:00 前后为成交量峰值时段，对应欧美重叠交易时间。<br>
        高成交量时段通常伴随更大的波动，适合短线交易；低成交量时段更适合观望。
    </div>
</div>

{chart_js}

</body>
</html>'''

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report saved to: {OUTPUT_HTML}")


if __name__ == '__main__':
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    generate_html_report(data)
