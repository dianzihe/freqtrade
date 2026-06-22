"""
Intraday Trading Pattern Analysis
Analyze relationship between fixed time periods and daily trading trends.
Uses 1-minute OHLCV data across multiple exchanges and pairs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

BASE = Path('E:/source/freqtrade/user_data/data')
OUTPUT_HTML = Path('E:/source/freqtrade/user_data/intraday_analysis_report.html')
OUTPUT_JSON = Path('E:/source/freqtrade/user_data/intraday_analysis_data.json')

# ─── 1. Load all data ───────────────────────────────────────────────

def load_all_data():
    """Load all feather files from all exchanges."""
    all_files = sorted(BASE.rglob('*.feather'))
    print(f"Found {len(all_files)} feather files")
    
    dfs = []
    stats = {'exchanges': defaultdict(int), 'pairs': []}
    
    for fp in all_files:
        try:
            df = pd.read_feather(fp)
            exchange = fp.parent.name
            pair = fp.stem.replace('-1m', '')
            
            df['date'] = pd.to_datetime(df['date'])
            df['exchange'] = exchange
            df['pair'] = pair
            df.set_index('date', inplace=True)
            dfs.append(df[['open', 'high', 'low', 'close', 'volume', 'pair', 'exchange']])
            stats['exchanges'][exchange] += 1
            stats['pairs'].append(pair)
        except Exception as e:
            print(f"  Skip {fp.name}: {e}")
    
    combined = pd.concat(dfs, axis=0)
    print(f"Total rows: {len(combined):,}")
    print(f"Exchanges: {dict(stats['exchanges'])}")
    print(f"Unique pairs: {combined['pair'].nunique()}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")
    return combined

# ─── 2. Hourly analysis ─────────────────────────────────────────────

def compute_hourly_returns(df):
    """Compute returns by hour of day (UTC)."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['date_only'] = df.index.date
    
    # For each pair, each day, compute hourly OHLC
    grouped = df.groupby(['pair', 'date_only', 'hour'])
    
    hourly = grouped.agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    
    hourly['hour_return'] = (hourly['close'] - hourly['open']) / hourly['open'] * 100
    hourly['hour_high_low'] = (hourly['high'] - hourly['low']) / hourly['open'] * 100
    
    return hourly

def analyze_hourly_patterns(hourly_df):
    """Analyze patterns by hour of day."""
    results = {}
    
    # Average return by hour
    hour_return = hourly_df.groupby('hour')['hour_return'].agg(['mean', 'std', 'count'])
    hour_return.columns = ['avg_return_pct', 'std_return_pct', 'sample_count']
    
    # Win rate by hour (positive return probability)
    hour_winrate = hourly_df.groupby('hour')['hour_return'].apply(
        lambda x: (x > 0).mean() * 100
    ).reset_index(name='win_rate_pct')
    
    # Volatility by hour
    hour_vol = hourly_df.groupby('hour')['hour_high_low'].mean().reset_index(name='avg_volatility_pct')
    
    # Volume by hour
    hour_vol_avg = hourly_df.groupby('hour')['volume'].mean().reset_index(name='avg_volume')
    
    results['hourly_returns'] = hour_return.reset_index().to_dict('records')
    results['hourly_winrate'] = hour_winrate.to_dict('records')
    results['hourly_volatility'] = hour_vol.to_dict('records')
    results['hourly_volume'] = hour_vol_avg.to_dict('records')
    
    # Best and worst hours
    results['best_hour'] = hour_return['avg_return_pct'].idxmax()
    results['worst_hour'] = hour_return['avg_return_pct'].idxmin()
    results['most_volatile_hour'] = hour_vol['avg_volatility_pct'].idxmax()
    
    print(f"\n=== Hourly Patterns ===")
    print(f"Best hour (highest return): {results['best_hour']}:00 UTC ({hour_return.loc[results['best_hour'], 'avg_return_pct']:.4f}%)")
    print(f"Worst hour (lowest return): {results['worst_hour']}:00 UTC ({hour_return.loc[results['worst_hour'], 'avg_return_pct']:.4f}%)")
    print(f"Most volatile hour: {results['most_volatile_hour']}:00 UTC")
    
    return results

# ─── 3. Session analysis ────────────────────────────────────────────

def analyze_sessions(hourly_df):
    """Analyze returns by trading session (UTC time)."""
    results = {}
    
    sessions = {
        'Asia Open (0-3 UTC)': (0, 3),
        'Asia Morning (3-6 UTC)': (3, 6),
        'Asia Late (6-9 UTC)': (6, 9),
        'Europe Open (7-10 UTC)': (7, 10),
        'Europe Morning (10-13 UTC)': (10, 13),
        'Europe-US Overlap (13-16 UTC)': (13, 16),
        'US Open (13-16 UTC)': (13, 16),
        'US Afternoon (16-20 UTC)': (16, 20),
        'US Close (20-23 UTC)': (20, 23),
        'Overnight (23-0 UTC)': (23, 24),
    }
    
    session_data = []
    for name, (start, end) in sessions.items():
        mask = (hourly_df['hour'] >= start) & (hourly_df['hour'] < end)
        session_returns = hourly_df.loc[mask].groupby(['pair', 'date_only'])['hour_return'].sum()
        
        row = {
            'session': name,
            'hours': f'{start:02d}:00-{end:02d}:00',
            'avg_cumulative_return_pct': session_returns.mean(),
            'std_return_pct': session_returns.std(),
            'win_rate_pct': (session_returns > 0).mean() * 100,
            'count': len(session_returns)
        }
        session_data.append(row)
    
    results['sessions'] = session_data
    
    print(f"\n=== Session Analysis ===")
    for s in session_data:
        print(f"  {s['session']}: avg={s['avg_cumulative_return_pct']:.4f}%, win_rate={s['win_rate_pct']:.1f}%")
    
    return results

# ─── 4. First N hours vs full day ───────────────────────────────────

def analyze_early_hour_predictive_power(df):
    """Analyze if early hours predict full-day direction."""
    results = {}
    
    df = df.copy()
    df['date_only'] = df.index.date
    df['hour'] = df.index.hour
    
    # Daily returns per pair
    daily = df.groupby(['pair', 'date_only']).agg(
        daily_open=('open', 'first'),
        daily_close=('close', 'last'),
    ).reset_index()
    daily['daily_return'] = (daily['daily_close'] - daily['daily_open']) / daily['daily_open'] * 100
    daily['daily_direction'] = daily['daily_return'] > 0
    
    # For each N hours (1 to 12), compute early return and check correlation
    predictions = []
    for n_hours in range(1, 13):
        early_returns = []
        
        for (pair, date_only), group in df.groupby(['pair', 'date_only']):
            early_data = group[group['hour'] < n_hours]
            if len(early_data) == 0:
                continue
            
            early_open = early_data['open'].iloc[0]
            early_close = early_data['close'].iloc[-1]
            early_return = (early_close - early_open) / early_open * 100
            
            early_returns.append({
                'pair': pair,
                'date_only': date_only,
                'early_return': early_return,
                'n_hours': n_hours
            })
        
        early_df = pd.DataFrame(early_returns)
        early_df = early_df.merge(daily[['pair', 'date_only', 'daily_return', 'daily_direction']],
                                   on=['pair', 'date_only'])
        
        # Correlation between early return and full day return
        corr = early_df['early_return'].corr(early_df['daily_return'])
        
        # Direction accuracy: does early direction match full day direction?
        early_df['early_direction'] = early_df['early_return'] > 0
        direction_match = (early_df['early_direction'] == early_df['daily_direction']).mean() * 100
        
        predictions.append({
            'n_hours': n_hours,
            'correlation': corr,
            'direction_accuracy_pct': direction_match,
            'sample_count': len(early_df)
        })
    
    results['early_hour_predictions'] = predictions
    
    print(f"\n=== Early Hour Predictive Power ===")
    for p in predictions:
        print(f"  First {p['n_hours']}h: corr={p['correlation']:.4f}, accuracy={p['direction_accuracy_pct']:.1f}%")
    
    return results

# ─── 5. Volatility regime analysis ──────────────────────────────────

def analyze_volatility_patterns(hourly_df, df):
    """Analyze if morning volatility predicts afternoon trend."""
    results = {}
    
    df = df.copy()
    df['date_only'] = df.index.date
    df['hour'] = df.index.hour
    
    daily_data = []
    for (pair, date_only), group in df.groupby(['pair', 'date_only']):
        morning = group[(group['hour'] >= 0) & (group['hour'] < 8)]
        afternoon = group[(group['hour'] >= 8) & (group['hour'] < 16)]
        evening = group[(group['hour'] >= 16) & (group['hour'] < 24)]
        
        if len(morning) == 0 or len(afternoon) == 0:
            continue
        
        # Morning metrics
        morning_ret = (morning['close'].iloc[-1] - morning['open'].iloc[0]) / morning['open'].iloc[0] * 100
        morning_vol = (morning['high'].max() - morning['low'].min()) / morning['open'].iloc[0] * 100
        morning_range = morning['high'].max() - morning['low'].min()
        
        # Afternoon metrics
        if len(afternoon) > 0:
            aft_ret = (afternoon['close'].iloc[-1] - afternoon['open'].iloc[0]) / afternoon['open'].iloc[0] * 100
        else:
            aft_ret = np.nan
        
        # Full day
        full_ret = (group['close'].iloc[-1] - group['open'].iloc[0]) / group['open'].iloc[0] * 100
        
        daily_data.append({
            'pair': pair,
            'date': date_only,
            'morning_return': morning_ret,
            'morning_volatility': morning_vol,
            'afternoon_return': aft_ret,
            'full_day_return': full_ret
        })
    
    daily_df = pd.DataFrame(daily_data)
    
    # Correlation: morning return vs afternoon return
    valid = daily_df.dropna(subset=['morning_return', 'afternoon_return'])
    if len(valid) > 10:
        corr_morning_aft = valid['morning_return'].corr(valid['afternoon_return'])
    else:
        corr_morning_aft = np.nan
    
    # Does morning volatility predict afternoon absolute return?
    valid2 = daily_df.dropna(subset=['morning_volatility', 'afternoon_return'])
    if len(valid2) > 10:
        corr_vol_aft_abs = valid2['morning_volatility'].corr(valid2['afternoon_return'].abs())
    else:
        corr_vol_aft_abs = np.nan
    
    results['volatility_analysis'] = {
        'morning_afternoon_return_corr': corr_morning_aft,
        'morning_vol_to_afternoon_abs_return_corr': corr_vol_aft_abs,
    }
    
    # Direction consistency: if morning is up, does afternoon follow?
    valid['morning_up'] = valid['morning_return'] > 0
    valid['afternoon_up'] = valid['afternoon_return'] > 0
    valid['same_direction'] = valid['morning_up'] == valid['afternoon_up']
    direction_consistency = valid['same_direction'].mean() * 100
    
    results['volatility_analysis']['direction_consistency_pct'] = direction_consistency
    results['volatility_analysis']['morning_samples'] = len(valid)
    
    print(f"\n=== Volatility / Session Continuity ===")
    print(f"  Morning→Afternoon return corr: {corr_morning_aft:.4f}")
    print(f"  Morning vol → Afternoon abs return corr: {corr_vol_aft_abs:.4f}")
    print(f"  Direction consistency (morning→afternoon): {direction_consistency:.1f}%")
    
    return results

# ─── 6. Cross-exchange comparison ───────────────────────────────────

def compare_exchanges(hourly_df, df):
    """Compare patterns across exchanges."""
    results = {}
    
    df = df.copy()
    df['date_only'] = df.index.date
    
    # Daily returns by exchange
    daily_by_exchange = df.groupby(['exchange', 'pair', 'date_only']).agg(
        daily_open=('open', 'first'),
        daily_close=('close', 'last')
    ).reset_index()
    daily_by_exchange['daily_return'] = (daily_by_exchange['daily_close'] - daily_by_exchange['daily_open']) / daily_by_exchange['daily_open'] * 100
    
    exchange_stats = daily_by_exchange.groupby('exchange')['daily_return'].agg(['mean', 'std', 'count'])
    results['exchange_daily'] = exchange_stats.reset_index().to_dict('records')
    
    print(f"\n=== Exchange Comparison ===")
    for _, row in exchange_stats.iterrows():
        print(f"  {row.name}: avg_daily={row['mean']:.4f}%, std={row['std']:.4f}%")
    
    return results

# ─── 7. Generate HTML Report ────────────────────────────────────────

def generate_html_report(all_results, df_stats):
    """Generate interactive HTML visualization report."""
    
    hourly = pd.DataFrame(all_results['hourly_returns'])
    sessions = all_results['sessions']
    early_pred = all_results['early_hour_predictions']
    vol_analysis = all_results['volatility_analysis']
    
    # Prepare chart data
    hours_labels = [f"{int(h)}:00" for h in hourly['hour']]
    returns_data = hourly['avg_return_pct'].tolist()
    winrate_data = [0] * 24
    for item in all_results['hourly_winrate']:
        winrate_data[int(item['hour'])] = item['win_rate_pct']
    volatility_data = [0] * 24
    for item in all_results['hourly_volatility']:
        volatility_data[int(item['hour'])] = item['avg_volatility_pct']
    volume_data = [0] * 24
    for item in all_results['hourly_volume']:
        volume_data[int(item['hour'])] = item['avg_volume']
    
    sessions_labels = json.dumps([s['session'] for s in sessions])
    sessions_returns = json.dumps([s['avg_cumulative_return_pct'] for s in sessions])
    sessions_winrate = json.dumps([s['win_rate_pct'] for s in sessions])
    
    early_labels = json.dumps([p['n_hours'] for p in early_pred])
    early_corr = json.dumps([p['correlation'] for p in early_pred])
    early_acc = json.dumps([p['direction_accuracy_pct'] for p in early_pred])
    
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
.highlight {{ background: #1a2332; }}
.summary-box {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; text-align: center; }}
.stat-card .value {{ font-size: 28px; font-weight: 700; }}
.stat-card .label {{ color: #8b949e; font-size: 12px; margin-top: 4px; }}
</style>
</head>
<body>

<h1>加密货币日内时间段趋势分析报告</h1>
<p class="subtitle">分析数据: {df_stats['total_rows']:,} 条 1分钟K线 | {df_stats['pairs']} 个交易对 | {df_stats['exchanges']} 个交易所 | {df_stats['date_start']} ~ {df_stats['date_end']} UTC</p>

<div class="summary-box">
    <div class="stat-card">
        <div class="value" style="color:#58a6ff">{all_results['best_hour']}:00</div>
        <div class="label">最佳交易时段 (UTC)</div>
    </div>
    <div class="stat-card">
        <div class="value" style="color:#f0883e">{all_results['worst_hour']}:00</div>
        <div class="label">最差交易时段 (UTC)</div>
    </div>
    <div class="stat-card">
        <div class="value" style="color:#ef4444">{early_pred[3]['direction_accuracy_pct']:.1f}%</div>
        <div class="label">前4小时方向预测准确率</div>
    </div>
    <div class="stat-card">
        <div class="value" style="color:#3fb950">{vol_analysis['direction_consistency_pct']:.1f}%</div>
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
            当前 4小时相关性: <strong>{early_pred[3]['correlation']:.3f}</strong> | 8小时相关性: <strong>{early_pred[7]['correlation']:.3f}</strong>
        </div>
    </div>
    <div class="card">
        <h2>前N小时方向预测准确率</h2>
        <div class="chart-wrap"><canvas id="chartEarlyAcc"></canvas></div>
        <div class="insight good">
            <strong>解读：</strong>如果前N小时上涨/下跌，全天最终也同向的概率。
            > 60% 表明有显著的预测价值。<br>
            当前 4小时准确率: <strong>{early_pred[3]['direction_accuracy_pct']:.1f}%</strong> | 8小时准确率: <strong>{early_pred[7]['direction_accuracy_pct']:.1f}%</strong>
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
'''
    
    for s in sessions:
        ret_class = 'positive' if s['avg_cumulative_return_pct'] > 0 else 'negative'
        html += f'''<tr>
            <td>{s['session']}</td>
            <td>{s['hours']}</td>
            <td class="{ret_class}">{s['avg_cumulative_return_pct']:.4f}%</td>
            <td>{s['std_return_pct']:.4f}%</td>
            <td>{s['win_rate_pct']:.1f}%</td>
            <td>{s['count']:,}</td>
        </tr>\n'''
    
    html += '''</tbody></table>
    </div>
</div>

<div class="card" style="margin-bottom:24px">
    <h2>关键发现与交易建议</h2>
    <div class="insight good">
        <strong>1. 最佳交易时段：</strong>UTC {hourly_return_best}:00，平均收益率 {hourly_return_best_val:.4f}%，胜率 {winrate_best_val:.1f}%。<br>
        该时段对应北京时间 {beijing_best}:00（UTC+8），建议重点关注此时间窗口的交易机会。
    </div>
    <div class="insight warn">
        <strong>2. 风险时段：</strong>UTC {hourly_return_worst}:00，平均收益率 {hourly_return_worst_val:.4f}%，胜率仅 {winrate_worst_val:.1f}%。<br>
        该时段对应北京时间 {beijing_worst}:00，建议减少仓位或避免在此时间段开仓。
    </div>
    <div class="insight">
        <strong>3. 预测能力：</strong>前 {best_n_hours} 小时走势对全天方向预测准确率达 {best_accuracy:.1f}%，相关性 {best_corr:.3f}。<br>
        可在每日前 {best_n_hours} 小时后观察趋势，若此时段大幅上涨/下跌，全天大概率延续该方向。
    </div>
    <div class="insight good">
        <strong>4. 上午→下午持续性：</strong>上午（0-8 UTC）方向与下午（8-16 UTC）方向一致的概率为 {direction_consistency_pct:.1f}%，<br>
        说明日内趋势具有一定的持续性，趋势交易策略在日内级别有效。
    </div>
    <div class="insight">
        <strong>5. 成交量规律：</strong>UTC {volume_peak_hour}:00 前后为成交量峰值时段，对应全球主要市场重叠交易时间。<br>
        高成交量时段通常伴随更大的波动，适合短线交易；低成交量时段更适合观望。
    </div>
</div>

<script>
// Color helpers
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
            backgroundColor: {json.dumps(['#ef4444' if v > 0 else '#22c55e' for v in returns_data])},
            borderColor: {json.dumps(['#ef4444' if v > 0 else '#22c55e' for v in returns_data])},
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

</body>
</html>'''
    
    # Fill in template variables
    # Compute insight values
    best_hour_val = hourly[hourly['hour'] == all_results['best_hour']]['avg_return_pct'].values[0]
    worst_hour_val = hourly[hourly['hour'] == int(all_results['worst_hour'])]['avg_return_pct'].values[0]
    wr_best = winrate_data[all_results['best_hour']]
    wr_worst = winrate_data[int(all_results['worst_hour'])]
    beijing_best = (all_results['best_hour'] + 8) % 24
    beijing_worst = (int(all_results['worst_hour']) + 8) % 24
    best_n = max(early_pred, key=lambda x: x['direction_accuracy_pct'])
    vol_peak = volume_data.index(max(volume_data))
    
    # Build insight HTML with actual values
    insight_html = f'''
<div class="card" style="margin-bottom:24px">
    <h2>关键发现与交易建议</h2>
    <div class="insight good">
        <strong>1. 最佳交易时段：</strong>UTC {all_results["best_hour"]}:00，平均收益率 {best_hour_val:.4f}%，胜率 {wr_best:.1f}%。<br>
        该时段对应北京时间 {beijing_best}:00（UTC+8），建议重点关注此时间窗口的交易机会。
    </div>
    <div class="insight warn">
        <strong>2. 风险时段：</strong>UTC {all_results["worst_hour"]}:00，平均收益率 {worst_hour_val:.4f}%，胜率仅 {wr_worst:.1f}%。<br>
        该时段对应北京时间 {beijing_worst}:00，建议减少仓位或避免在此时间段开仓。
    </div>
    <div class="insight">
        <strong>3. 预测能力：</strong>前 {best_n["n_hours"]} 小时走势对全天方向预测准确率达 {best_n["direction_accuracy_pct"]:.1f}%，相关性 {best_n["correlation"]:.3f}。<br>
        可在每日前 {best_n["n_hours"]} 小时后观察趋势，若此时段大幅上涨/下跌，全天大概率延续该方向。
    </div>
    <div class="insight good">
        <strong>4. 上午→下午持续性：</strong>上午（0-8 UTC）方向与下午（8-16 UTC）方向一致的概率为 {vol_analysis["direction_consistency_pct"]:.1f}%，<br>
        说明日内趋势具有一定的持续性，趋势交易策略在日内级别有效。
    </div>
    <div class="insight">
        <strong>5. 成交量规律：</strong>UTC {vol_peak}:00 前后为成交量峰值时段，对应全球主要市场重叠交易时间。<br>
        高成交量时段通常伴随更大的波动，适合短线交易；低成交量时段更适合观望。
    </div>
</div>'''
    
    # Replace the placeholder insight block with actual values
    html = html.replace('''<div class="card" style="margin-bottom:24px">
    <h2>关键发现与交易建议</h2>
    <div class="insight good">
        <strong>1. 最佳交易时段：</strong>UTC {hourly_return_best}:00，平均收益率 {hourly_return_best_val:.4f}%，胜率 {winrate_best_val:.1f}%。<br>
        该时段对应北京时间 {beijing_best}:00（UTC+8），建议重点关注此时间窗口的交易机会。
    </div>
    <div class="insight warn">
        <strong>2. 风险时段：</strong>UTC {hourly_return_worst}:00，平均收益率 {hourly_return_worst_val:.4f}%，胜率仅 {winrate_worst_val:.1f}%。<br>
        该时段对应北京时间 {beijing_worst}:00，建议减少仓位或避免在此时间段开仓。
    </div>
    <div class="insight">
        <strong>3. 预测能力：</strong>前 {best_n_hours} 小时走势对全天方向预测准确率达 {best_accuracy:.1f}%，相关性 {best_corr:.3f}。<br>
        可在每日前 {best_n_hours} 小时后观察趋势，若此时段大幅上涨/下跌，全天大概率延续该方向。
    </div>
    <div class="insight good">
        <strong>4. 上午→下午持续性：</strong>上午（0-8 UTC）方向与下午（8-16 UTC）方向一致的概率为 {direction_consistency_pct:.1f}%，<br>
        说明日内趋势具有一定的持续性，趋势交易策略在日内级别有效。
    </div>
    <div class="insight">
        <strong>5. 成交量规律：</strong>UTC {volume_peak_hour}:00 前后为成交量峰值时段，对应全球主要市场重叠交易时间。<br>
        高成交量时段通常伴随更大的波动，适合短线交易；低成交量时段更适合观望。
    </div>
</div>''', insight_html)
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nHTML report saved to: {OUTPUT_HTML}")

# ─── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Intraday Trading Pattern Analysis")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_all_data()
    
    # Compute hourly returns
    print("\n[2/6] Computing hourly returns...")
    hourly_df = compute_hourly_returns(df)
    
    # Analyze hourly patterns
    print("\n[3/6] Analyzing hourly patterns...")
    hourly_results = analyze_hourly_patterns(hourly_df)
    
    # Session analysis
    print("\n[4/6] Analyzing trading sessions...")
    session_results = analyze_sessions(hourly_df)
    
    # Early hour predictive power
    print("\n[5/6] Analyzing early hour predictive power...")
    early_results = analyze_early_hour_predictive_power(df)
    
    # Volatility analysis
    print("\n[6/6] Analyzing volatility patterns...")
    vol_results = analyze_volatility_patterns(hourly_df, df)
    
    # Exchange comparison
    exchange_results = compare_exchanges(hourly_df, df)
    
    # Combine all results
    all_results = {
        **hourly_results,
        **session_results,
        **early_results,
        **vol_results,
        **exchange_results
    }
    
    # Save JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON data saved to: {OUTPUT_JSON}")
    
    # Generate report
    df_stats = {
        'total_rows': len(df),
        'pairs': df['pair'].nunique(),
        'exchanges': df['exchange'].nunique(),
        'date_start': str(df.index.min().date()),
        'date_end': str(df.index.max().date()),
    }
    
    generate_html_report(all_results, df_stats)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
