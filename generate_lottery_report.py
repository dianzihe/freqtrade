"""
Generate HTML report for Lottery Ticket Momentum backtest on Gate data.
"""
import zipfile
import json
import os
import sys
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# Load backtest results
result_zip = 'user_data/backtest_results/backtest-result-2026-07-02_17-01-32.zip'
with zipfile.ZipFile(result_zip) as z:
    with z.open('backtest-result-2026-07-02_17-01-32.json') as f:
        data = json.load(f)

strat_name = list(data['strategy'].keys())[0]
strat_data = data['strategy'][strat_name]
trades = strat_data['trades']
df = pd.DataFrame(trades)

# Convert dates
df['open_date'] = pd.to_datetime(df['open_date'])
df['close_date'] = pd.to_datetime(df['close_date'])
df['profit_pct'] = df['profit_ratio'] * 100
df['duration_hours'] = df['trade_duration'] / 60.0

# Separate closed and open trades
closed = df[~df['is_open']].copy()
open_trades = df[df['is_open']].copy()

# Summary metrics
total_trades = len(closed)
wins = closed[closed['profit_abs'] > 0]
losses = closed[closed['profit_abs'] <= 0]
win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
total_profit = closed['profit_abs'].sum()
total_profit_pct = total_profit / 1000 * 100  # relative to 1000 USDT starting balance
avg_profit_pct = closed['profit_pct'].mean()
best_trade = closed.loc[closed['profit_pct'].idxmax()] if len(closed) > 0 else None
worst_trade = closed.loc[closed['profit_pct'].idxmin()] if len(closed) > 0 else None
profit_factor = wins['profit_abs'].sum() / abs(losses['profit_abs'].sum()) if len(losses) > 0 and losses['profit_abs'].sum() != 0 else float('inf')
max_consec_wins = 0
max_consec_losses = 0
current_wins = 0
current_losses = 0
for _, t in closed.sort_values('close_date').iterrows():
    if t['profit_abs'] > 0:
        current_wins += 1
        current_losses = 0
        max_consec_wins = max(max_consec_wins, current_wins)
    else:
        current_losses += 1
        current_wins = 0
        max_consec_losses = max(max_consec_losses, current_losses)

# Entry tag analysis
tag_stats = closed.groupby('enter_tag').agg(
    trades=('profit_abs', 'count'),
    total_profit=('profit_abs', 'sum'),
    avg_profit_pct=('profit_pct', 'mean'),
    wins=('profit_abs', lambda x: (x > 0).sum()),
    losses=('profit_abs', lambda x: (x <= 0).sum()),
    avg_duration_h=('duration_hours', 'mean'),
).reset_index()
tag_stats['win_rate'] = tag_stats['wins'] / tag_stats['trades'] * 100

# Exit reason analysis
exit_stats = closed.groupby('exit_reason').agg(
    trades=('profit_abs', 'count'),
    total_profit=('profit_abs', 'sum'),
    avg_profit_pct=('profit_pct', 'mean'),
    wins=('profit_abs', lambda x: (x > 0).sum()),
    losses=('profit_abs', lambda x: (x <= 0).sum()),
    avg_duration_h=('duration_hours', 'mean'),
).reset_index()
exit_stats['win_rate'] = exit_stats['wins'] / exit_stats['trades'] * 100

# Pair analysis
pair_stats = closed.groupby('pair').agg(
    trades=('profit_abs', 'count'),
    total_profit=('profit_abs', 'sum'),
    avg_profit_pct=('profit_pct', 'mean'),
).reset_index().sort_values('total_profit')

# Daily PnL
closed['date'] = closed['close_date'].dt.date
daily_pnl = closed.groupby('date')['profit_abs'].sum().reset_index()
daily_pnl['cumulative'] = daily_pnl['profit_abs'].cumsum()

# Build HTML
html_parts = []
html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lottery Ticket Momentum V3 - Gate 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card-bg: #1a1d28;
    --border: #2a2e3c;
    --text: #e0e3eb;
    --text-dim: #8b8fa3;
    --red: #ef4444;
    --green: #22c55e;
    --blue: #3b82f6;
    --yellow: #eab308;
    --purple: #a855f7;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; padding: 20px; line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; color: var(--text); }}
  h2 {{ font-size: 22px; margin: 30px 0 15px; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  h3 {{ font-size: 16px; margin: 20px 0 10px; color: var(--text-dim); }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 25px; font-size: 14px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .card-label {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
  .card-value {{ font-size: 28px; font-weight: 700; }}
  .card-sub {{ font-size: 13px; color: var(--text-dim); margin-top: 4px; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .neutral {{ color: var(--text); }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }}
  th {{ background: var(--card-bg); padding: 12px 15px; text-align: left; font-weight: 600; color: var(--text-dim); border-bottom: 2px solid var(--border); }}
  td {{ padding: 10px 15px; border-bottom: 1px solid var(--border); }}
  tr:hover {{ background: rgba(255,255,255,0.03); }}
  .chart-container {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; position: relative; }}
  .chart-wrapper {{ position: relative; height: 350px; }}
  .warning {{ background: rgba(234, 179, 8, 0.1); border: 1px solid var(--yellow); border-radius: 8px; padding: 15px; margin: 15px 0; color: var(--yellow); font-size: 14px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .tag-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
  .tag-sniper {{ background: rgba(168, 85, 247, 0.2); color: var(--purple); }}
  .tag-breakout {{ background: rgba(59, 130, 246, 0.2); color: var(--blue); }}
  .tag-pullback {{ background: rgba(249, 115, 22, 0.2); color: var(--orange); }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>Lottery Ticket Momentum V3.1 — Gate 回测报告</h1>
<p class="subtitle">策略: LotteryTicketMomentumV3 | 交易所: Gate (现货) | 数据: 2026-06-24 ~ 2026-07-02 | 有效回测: 2026-06-25 09:20 ~ 2026-07-02 00:00 (约6.5天) | 交易对: 195个 USDT 现货对</p>

<div class="warning">
  <strong>⚠️ 窗口过短警告:</strong> 策略文档明确指出 "5天窗口无法评估彩票策略（右尾赢家出现概率极低）"。当前仅约6.5天数据，样本极小，以下结果仅供参考，不可作为策略有效性的最终判断。
</div>
""")

# Summary cards
html_parts.append(f"""
<h2>核心指标</h2>
<div class="cards">
  <div class="card">
    <div class="card-label">总交易数</div>
    <div class="card-value neutral">{total_trades}</div>
    <div class="card-sub">已平仓 + {len(open_trades)} 笔未平仓</div>
  </div>
  <div class="card">
    <div class="card-label">总盈亏</div>
    <div class="card-value {'neg' if total_profit < 0 else 'pos'}">{total_profit:+.3f} USDT</div>
    <div class="card-sub">{total_profit_pct:+.2f}% (初始1000 USDT)</div>
  </div>
  <div class="card">
    <div class="card-label">胜率</div>
    <div class="card-value {'neg' if win_rate < 50 else 'pos'}">{win_rate:.1f}%</div>
    <div class="card-sub">{len(wins)} 胜 / {len(losses)} 负</div>
  </div>
  <div class="card">
    <div class="card-label">盈亏比</div>
    <div class="card-value {'neg' if profit_factor < 1 else 'pos'}">{profit_factor:.2f}</div>
    <div class="card-sub">总盈利 / 总亏损</div>
  </div>
  <div class="card">
    <div class="card-label">平均每笔收益</div>
    <div class="card-value {'neg' if avg_profit_pct < 0 else 'pos'}">{avg_profit_pct:+.2f}%</div>
    <div class="card-sub">平均持仓 {closed['duration_hours'].mean():.1f}h</div>
  </div>
  <div class="card">
    <div class="card-label">最大连续亏损</div>
    <div class="card-value neg">{max_consec_losses}</div>
    <div class="card-sub">最大连续盈利 {max_consec_wins}</div>
  </div>
  <div class="card">
    <div class="card-label">最佳交易</div>
    <div class="card-value pos">+{best_trade['profit_pct']:.1f}%</div>
    <div class="card-sub">{best_trade['pair']}</div>
  </div>
  <div class="card">
    <div class="card-label">最差交易</div>
    <div class="card-value neg">{worst_trade['profit_pct']:.1f}%</div>
    <div class="card-sub">{worst_trade['pair']}</div>
  </div>
</div>
""")

# Daily PnL chart
daily_labels = [str(d) for d in daily_pnl['date']]
daily_values = daily_pnl['profit_abs'].tolist()
cum_values = daily_pnl['cumulative'].tolist()

html_parts.append(f"""
<h2>每日盈亏</h2>
<div class="chart-container">
  <div class="chart-wrapper">
    <canvas id="dailyChart"></canvas>
  </div>
</div>

<h2>累计盈亏曲线</h2>
<div class="chart-container">
  <div class="chart-wrapper">
    <canvas id="cumChart"></canvas>
  </div>
</div>
""")

# Entry tag analysis
html_parts.append("<h2>入场信号分析</h2>")
html_parts.append("""<table>
<thead><tr><th>入场标签</th><th>交易数</th><th>总盈亏 (USDT)</th><th>平均收益 (%)</th><th>胜率</th><th>平均持仓 (h)</th></tr></thead>
<tbody>""")
tag_colors = {'lottery_sniper_a': 'tag-sniper', 'lottery_breakout_b': 'tag-breakout', 'lottery_pullback_c': 'tag-pullback'}
for _, row in tag_stats.iterrows():
    tag = row['enter_tag']
    badge_class = tag_colors.get(tag, '')
    cls = 'pos' if row['total_profit'] > 0 else 'neg'
    html_parts.append(f"""<tr>
      <td><span class="tag-badge {badge_class}">{tag}</span></td>
      <td>{int(row['trades'])}</td>
      <td class="{cls}">{row['total_profit']:+.3f}</td>
      <td class="{cls}">{row['avg_profit_pct']:+.2f}%</td>
      <td>{row['win_rate']:.1f}% ({int(row['wins'])}W/{int(row['losses'])}L)</td>
      <td>{row['avg_duration_h']:.1f}</td>
    </tr>""")
html_parts.append("</tbody></table>")

# Exit reason analysis
html_parts.append("<h2>退出原因分析</h2>")
html_parts.append("""<table>
<thead><tr><th>退出原因</th><th>交易数</th><th>总盈亏 (USDT)</th><th>平均收益 (%)</th><th>胜率</th><th>平均持仓 (h)</th></tr></thead>
<tbody>""")
for _, row in exit_stats.iterrows():
    cls = 'pos' if row['total_profit'] > 0 else 'neg'
    html_parts.append(f"""<tr>
      <td>{row['exit_reason']}</td>
      <td>{int(row['trades'])}</td>
      <td class="{cls}">{row['total_profit']:+.3f}</td>
      <td class="{cls}">{row['avg_profit_pct']:+.2f}%</td>
      <td>{row['win_rate']:.1f}% ({int(row['wins'])}W/{int(row['losses'])}L)</td>
      <td>{row['avg_duration_h']:.1f}</td>
    </tr>""")
html_parts.append("</tbody></table>")

# Entry tag pie chart
tag_labels = tag_stats['enter_tag'].tolist()
tag_values = tag_stats['trades'].tolist()

# Exit reason distribution
exit_labels = exit_stats['exit_reason'].tolist()
exit_values = exit_stats['trades'].tolist()

html_parts.append(f"""
<div class="grid-2">
  <div class="chart-container">
    <h3>入场信号分布</h3>
    <div class="chart-wrapper">
      <canvas id="tagPieChart"></canvas>
    </div>
  </div>
  <div class="chart-container">
    <h3>退出原因分布</h3>
    <div class="chart-wrapper">
      <canvas id="exitPieChart"></canvas>
    </div>
  </div>
</div>
""")

# Pair analysis
html_parts.append("<h2>交易对表现</h2>")
html_parts.append("""<table>
<thead><tr><th>交易对</th><th>交易数</th><th>总盈亏 (USDT)</th><th>平均收益 (%)</th></tr></thead>
<tbody>""")
for _, row in pair_stats.iterrows():
    cls = 'pos' if row['total_profit'] > 0 else 'neg'
    html_parts.append(f"""<tr>
      <td>{row['pair']}</td>
      <td>{int(row['trades'])}</td>
      <td class="{cls}">{row['total_profit']:+.3f}</td>
      <td class="{cls}">{row['avg_profit_pct']:+.2f}%</td>
    </tr>""")
html_parts.append("</tbody></table>")

# Individual trades table
html_parts.append("<h2>详细交易记录</h2>")
html_parts.append("""<table>
<thead><tr><th>#</th><th>交易对</th><th>入场标签</th><th>开仓时间</th><th>平仓时间</th><th>持仓(h)</th><th>收益(%)</th><th>盈亏(USDT)</th><th>退出原因</th></tr></thead>
<tbody>""")
for i, (_, t) in enumerate(closed.sort_values('open_date').iterrows(), 1):
    cls = 'pos' if t['profit_abs'] > 0 else 'neg'
    tag = t['enter_tag']
    badge_class = tag_colors.get(tag, '')
    html_parts.append(f"""<tr>
      <td>{i}</td>
      <td>{t['pair']}</td>
      <td><span class="tag-badge {badge_class}">{tag}</span></td>
      <td>{t['open_date'].strftime('%m-%d %H:%M')}</td>
      <td>{t['close_date'].strftime('%m-%d %H:%M')}</td>
      <td>{t['duration_hours']:.1f}</td>
      <td class="{cls}">{t['profit_pct']:+.2f}%</td>
      <td class="{cls}">{t['profit_abs']:+.4f}</td>
      <td>{t['exit_reason']}</td>
    </tr>""")
html_parts.append("</tbody></table>")

# Open trades
if len(open_trades) > 0:
    html_parts.append("<h2>未平仓交易</h2>")
    html_parts.append("""<table>
<thead><tr><th>交易对</th><th>入场标签</th><th>开仓时间</th><th>开仓价</th><th>当前盈亏(%)</th><th>盈亏(USDT)</th></tr></thead>
<tbody>""")
    for _, t in open_trades.iterrows():
        cls = 'pos' if t['profit_abs'] > 0 else 'neg'
        tag = t['enter_tag']
        badge_class = tag_colors.get(tag, '')
        html_parts.append(f"""<tr>
          <td>{t['pair']}</td>
          <td><span class="tag-badge {badge_class}">{tag}</span></td>
          <td>{t['open_date'].strftime('%m-%d %H:%M')}</td>
          <td>{t['open_rate']:.6f}</td>
          <td class="{cls}">{t['profit_pct']:+.2f}%</td>
          <td class="{cls}">{t['profit_abs']:+.4f}</td>
        </tr>""")
    html_parts.append("</tbody></table>")

# Analysis section
html_parts.append("""
<h2>分析与结论</h2>
<div class="card" style="padding: 25px;">
  <h3>策略表现概述</h3>
  <p style="margin-bottom: 15px;">在约6.5天的回测窗口中，策略共触发 <strong>31笔已平仓交易</strong>，总亏损 <strong>-0.967 USDT (-0.10%)</strong>。
  胜率仅 <strong>25.8%</strong>，盈亏比 <strong>0.17</strong>，远低于盈利所需水平。21笔交易触发止损（-6%），仅5笔通过棘轮止损获利退出。</p>

  <h3>入场信号对比</h3>
  <ul style="margin-bottom: 15px; padding-left: 20px;">
    <li><strong>lottery_breakout_b</strong> (17笔): 触发最多但表现最差，23.5%胜率，总亏 -0.452 USDT</li>
    <li><strong>lottery_sniper_a</strong> (8笔): 25.0%胜率，总亏 -0.295 USDT，6笔止损</li>
    <li><strong>lottery_pullback_c</strong> (6笔): 胜率最高 33.3%，但4笔止损仍导致 -0.219 USDT 亏损</li>
  </ul>

  <h3>退出机制分析</h3>
  <ul style="margin-bottom: 15px; padding-left: 20px;">
    <li><strong>止损 (stop_loss):</strong> 21笔全部亏损，共 -1.128 USDT — 止损触发率67.7%，说明入场后大部分行情不利</li>
    <li><strong>棘轮止损 (trailing_stop_loss):</strong> 5笔全部盈利，共 +0.147 USDT — 棘轮机制在有利行情中有效锁定利润</li>
    <li><strong>强制退出 (force_exit):</strong> 5笔（含未平仓），3盈2亏，+0.015 USDT — 时间止损触发正常</li>
  </ul>

  <h3>关键问题</h3>
  <ul style="margin-bottom: 15px; padding-left: 20px;">
    <li><strong>样本不足:</strong> 6.5天/31笔交易远不足以评估彩票策略。策略需要3-6个月含妖币行情的数据</li>
    <li><strong>止损率过高:</strong> 67.7%的交易触发-6%止损，说明入场时机或阈值可能需要优化</li>
    <li><strong>盈亏比失衡:</strong> 0.17的盈亏比意味着赢的钱远不够弥补亏损，需要更大的"彩票"赢家</li>
    <li><strong>最佳交易仅+14.78%:</strong> 彩票策略依赖偶尔的暴涨（+50%~+200%），但本窗口未出现此类行情</li>
    <li><strong>101个信号被拒:</strong> 票数限制和冷却期工作正常，但也意味着很多潜在机会被过滤</li>
  </ul>

  <h3>建议</h3>
  <ul style="padding-left: 20px;">
    <li>使用更长的时间窗口（至少3个月）重新回测，包含牛市/妖币行情</li>
    <li>考虑对 breakout_b 信号提高阈值或增加额外过滤条件，降低假突破入场</li>
    <li>评估是否需要放宽每日票数限制（当前6票/天），以增加样本量</li>
    <li>棘轮止损表现良好，可考虑在更多盈利区间提前触发棘轮</li>
    <li>考虑加入趋势强度过滤（如 BTC 日线级别），在明确牛市时加大信号灵敏度</li>
  </ul>
</div>
""")

html_parts.append(f"""
<div class="footer">
  回测报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 策略: LotteryTicketMomentumV3 | 数据源: Gate 现货
</div>
</div>

<script>
// Daily PnL Bar Chart
new Chart(document.getElementById('dailyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(daily_labels)},
    datasets: [{{
      label: '每日盈亏 (USDT)',
      data: {json.dumps([round(v, 4) for v in daily_values])},
      backgroundColor: {json.dumps(['#22c55e' if v >= 0 else '#ef4444' for v in daily_values])},
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b8fa3' }}, grid: {{ color: '#2a2e3c' }} }},
      y: {{ ticks: {{ color: '#8b8fa3' }}, grid: {{ color: '#2a2e3c' }} }}
    }}
  }}
}});

// Cumulative PnL Line Chart
new Chart(document.getElementById('cumChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(daily_labels)},
    datasets: [{{
      label: '累计盈亏 (USDT)',
      data: {json.dumps([round(v, 4) for v in cum_values])},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 5,
      pointBackgroundColor: '#3b82f6',
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b8fa3' }}, grid: {{ color: '#2a2e3c' }} }},
      y: {{ ticks: {{ color: '#8b8fa3' }}, grid: {{ color: '#2a2e3c' }} }}
    }}
  }}
}});

// Entry Tag Pie Chart
new Chart(document.getElementById('tagPieChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(tag_labels)},
    datasets: [{{
      data: {json.dumps([int(v) for v in tag_values])},
      backgroundColor: ['#a855f7', '#3b82f6', '#f97316'],
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b8fa3', padding: 15 }} }}
    }}
  }}
}});

// Exit Reason Pie Chart
new Chart(document.getElementById('exitPieChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(exit_labels)},
    datasets: [{{
      data: {json.dumps([int(v) for v in exit_values])},
      backgroundColor: ['#ef4444', '#22c55e', '#eab308'],
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b8fa3', padding: 15 }} }}
    }}
  }}
}});
</script>
</body>
</html>
""")

# Write HTML
output_path = 'lottery_ticket_gate_report.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))

print(f'Report generated: {output_path}')
print(f'Total trades: {total_trades}')
print(f'Total profit: {total_profit:+.3f} USDT ({total_profit_pct:+.2f}%)')
print(f'Win rate: {win_rate:.1f}%')
print(f'Profit factor: {profit_factor:.2f}')
