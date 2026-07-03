"""
Gate Moji-Coin Analysis: pre-pump feature mining & algorithm design
Identify coins with >20% intraday gain, analyze pre-pump characteristics, design detection algorithm
"""

import os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("E:/source/freqtrade/user_data/data/gate")
OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/moji-coin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_5m_data():
    files = sorted(DATA_DIR.glob("*-5m.feather"))
    coin_data = {}
    for f in files:
        symbol = f.name.replace("-5m.feather", "")
        df = pd.read_feather(f)
        df['symbol'] = symbol
        df = df.set_index('date').sort_index()
        df['day'] = df.index.date
        coin_data[symbol] = df
    return coin_data


def compute_daily_stats(coin_data):
    records = []
    for symbol, df in coin_data.items():
        for day, grp in df.groupby('day'):
            if len(grp) < 12:
                continue
            op = grp['open'].iloc[0]
            cl = grp['close'].iloc[-1]
            hi = grp['high'].max()
            lo = grp['low'].min()
            records.append({
                'symbol': symbol, 'day': day,
                'open': op, 'close': cl, 'high': hi, 'low': lo,
                'daily_change_pct': (cl - op) / op * 100,
                'max_intraday_gain_pct': (hi - op) / op * 100,
                'max_intraday_drop_pct': (lo - op) / op * 100,
                'total_volume': grp['volume'].sum(),
                'mean_volume': grp['volume'].mean(),
                'num_bars': len(grp),
            })
    return pd.DataFrame(records)


def find_moji_coins(daily_df, threshold=20.0):
    moji = daily_df[daily_df['max_intraday_gain_pct'] >= threshold].copy()
    moji = moji.sort_values('max_intraday_gain_pct', ascending=False)
    return moji


def find_pump_start(df_5m, day, threshold_pct=5.0):
    day_df = df_5m[df_5m['day'] == day]
    if len(day_df) == 0:
        return None
    day_open = day_df['open'].iloc[0]
    cummax_gain = ((day_df['high'].cummax() - day_open) / day_open * 100)
    trigger = cummax_gain[cummax_gain >= threshold_pct]
    if len(trigger) == 0:
        return None
    return trigger.index[0]


def compute_pre_pump_features(df_5m, pump_start_time, lookback_hours=6):
    lookback_bars = lookback_hours * 12
    loc = df_5m.index.get_loc(pump_start_time)
    start_loc = max(0, loc - lookback_bars)
    pre = df_5m.iloc[start_loc:loc]
    if len(pre) < 12:
        return None

    f = {}
    f['pre_bars'] = len(pre)
    f['pre_open'] = pre['open'].iloc[0]
    f['pre_close'] = pre['close'].iloc[-1]
    f['pre_change_pct'] = (f['pre_close'] - f['pre_open']) / f['pre_open'] * 100
    f['pre_high'] = pre['high'].max()
    f['pre_low'] = pre['low'].min()
    f['pre_range_pct'] = (f['pre_high'] - f['pre_low']) / f['pre_low'] * 100

    lr = np.log(pre['close'] / pre['close'].shift(1)).dropna()
    f['pre_volatility'] = lr.std() * np.sqrt(12 * 24) if len(lr) > 1 else 0

    f['pre_total_volume'] = pre['volume'].sum()
    f['pre_mean_volume'] = pre['volume'].mean()
    f['pre_volume_std'] = pre['volume'].std()
    f['pre_volume_skew'] = pre['volume'].skew() if len(pre) > 2 else 0

    last1h = pre.iloc[-12:]
    earlier = pre.iloc[:-12] if len(pre) > 12 else pre
    f['volume_ratio_last1h_vs_avg'] = (
        last1h['volume'].mean() / earlier['volume'].mean()
        if len(earlier) > 0 and earlier['volume'].mean() > 0 else 0
    )

    if len(pre) >= 24:
        sma12 = pre['close'].rolling(12).mean().dropna()
        if len(sma12) >= 2 and sma12.iloc[-24] if len(sma12) >= 24 else 0 != 0:
            f['sma12_slope'] = (sma12.iloc[-1] - sma12.iloc[min(23, len(sma12)-1)]) / sma12.iloc[min(23, len(sma12)-1)] * 100
        else:
            f['sma12_slope'] = 0
    else:
        f['sma12_slope'] = 0

    last_n = min(6, len(pre))
    lc = pre['close'].iloc[-last_n:].values
    f['staircase_score'] = sum(1 for i in range(1, len(lc)) if lc[i] >= lc[i-1]) / max(1, len(lc)-1)

    lv = pre['volume'].iloc[-last_n:].values
    f['volume_staircase_score'] = sum(1 for i in range(1, len(lv)) if lv[i] >= lv[i-1]) / max(1, len(lv)-1)

    if len(pre) >= 24:
        ranges = (pre['high'] - pre['low']) / pre['low'] * 100
        r1 = ranges.iloc[:len(ranges)//2].mean()
        r2 = ranges.iloc[len(ranges)//2:].mean()
        f['range_compression_ratio'] = r2 / r1 if r1 > 0 else 0
    else:
        f['range_compression_ratio'] = 0

    return f


def analyze_moji_pre_features(coin_data, moji_df, lookback_hours=6):
    recs = []
    for _, row in moji_df.iterrows():
        sym = row['symbol']
        day = row['day']
        if sym not in coin_data:
            continue
        df = coin_data[sym]
        ps = find_pump_start(df, day, threshold_pct=5.0)
        if ps is None:
            continue
        feats = compute_pre_pump_features(df, ps, lookback_hours)
        if feats is None:
            continue
        feats['symbol'] = sym
        feats['pump_day'] = str(day)
        feats['pump_gain_pct'] = row['max_intraday_gain_pct']
        feats['pump_start_time'] = str(ps)
        recs.append(feats)
    return pd.DataFrame(recs)


def compute_control_features(coin_data, daily_df, moji_df, sample_per_day=20, lookback_hours=6):
    moji_set = set(zip(moji_df['symbol'], moji_df['day'].astype(str)))
    non_moji = daily_df[~daily_df.apply(lambda r: (r['symbol'], str(r['day'])) in moji_set, axis=1)]
    non_moji = non_moji[non_moji['num_bars'] >= 200].copy()
    if len(non_moji) == 0:
        return pd.DataFrame()
    sample = non_moji.sample(n=min(sample_per_day * 5, len(non_moji)), random_state=42)

    recs = []
    for _, row in sample.iterrows():
        sym = row['symbol']
        day = row['day']
        if sym not in coin_data:
            continue
        df = coin_data[sym]
        day_df = df[df['day'] == day]
        if len(day_df) == 0:
            continue
        rl = min(len(day_df) - 1, 72)
        vs = day_df.index[rl]
        feats = compute_pre_pump_features(df, vs, lookback_hours)
        if feats is None:
            continue
        feats['symbol'] = sym
        feats['pump_day'] = str(day)
        feats['pump_gain_pct'] = row['max_intraday_gain_pct']
        feats['pump_start_time'] = str(vs)
        feats['is_moji'] = False
        recs.append(feats)
    return pd.DataFrame(recs)


def compare_features(moji_f, control_f):
    moji_f['is_moji'] = True
    cols = [
        'pre_change_pct', 'pre_range_pct', 'pre_volatility',
        'pre_total_volume', 'volume_ratio_last1h_vs_avg',
        'sma12_slope', 'staircase_score', 'volume_staircase_score',
        'range_compression_ratio', 'pre_volume_skew',
    ]
    comp = {}
    for c in cols:
        mv = moji_f[c].dropna()
        cv = control_f[c].dropna()
        if len(mv) > 0 and len(cv) > 0:
            comp[c] = {
                'moji_mean': mv.mean(), 'moji_median': mv.median(),
                'control_mean': cv.mean(), 'control_median': cv.median(),
                'diff_pct': ((mv.mean() - cv.mean()) / abs(cv.mean()) * 100) if cv.mean() != 0 else float('inf'),
            }
    return pd.DataFrame(comp).T


def design_moji_detector(comp_df, moji_f):
    med = moji_f.describe().T['50%']
    thresholds = {}
    rules = []

    vr = max(float(med.get('volume_ratio_last1h_vs_avg', 2)), 1.5)
    thresholds['volume_ratio'] = vr
    rules.append({'id': 'R1', 'name': 'Volume Spike', 'desc': f'Last 1h volume >= {vr:.1f}x of earlier 5h', 'cond': f'volume_ratio_last1h_vs_avg >= {vr:.1f}', 'weight': 25})

    ss = max(float(med.get('staircase_score', 0.6)), 0.5)
    thresholds['staircase_score'] = ss
    rules.append({'id': 'R2', 'name': 'Price Staircase', 'desc': f'Last 6 bars closing-up ratio >= {ss:.2f}', 'cond': f'staircase_score >= {ss:.2f}', 'weight': 20})

    vs = max(float(med.get('volume_staircase_score', 0.5)), 0.4)
    thresholds['volume_staircase_score'] = vs
    rules.append({'id': 'R3', 'name': 'Volume Staircase', 'desc': f'Last 6 bars volume-up ratio >= {vs:.2f}', 'cond': f'volume_staircase_score >= {vs:.2f}', 'weight': 20})

    rc = float(med.get('range_compression_ratio', 1.0))
    thresholds['range_compression_ratio'] = rc
    rules.append({'id': 'R4', 'name': 'Range Expansion', 'desc': f'Recent half range/earlier half >= {rc:.2f} (range expanding)', 'cond': f'range_compression_ratio >= {rc:.2f}', 'weight': 15})

    pc = float(med.get('pre_change_pct', 2))
    thresholds['pre_change_pct'] = pc
    rules.append({'id': 'R5', 'name': 'Mild Accumulation', 'desc': f'Pre-pump 6h cum. change >= {pc:.2f}%', 'cond': f'pre_change_pct >= {pc:.2f}', 'weight': 10})

    pv = float(med.get('pre_volatility', 0.5))
    thresholds['pre_volatility'] = pv
    rules.append({'id': 'R6', 'name': 'Volatility Compression', 'desc': f'Pre-pump 6h annualized vol <= {pv:.4f}', 'cond': f'pre_volatility <= {pv:.4f}', 'weight': 10})

    return {
        'name': 'MojiCoinDetector v1',
        'rules': rules,
        'thresholds': thresholds,
        'scoring': {'method': 'weighted_sum', 'trigger_threshold': 60,
                    'levels': {'strong': 'score >= 80, 3+ rules', 'medium': 'score >= 60, 2+ rules', 'weak': 'score >= 40, 1+ rules'}}
    }


def build_html(data):
    """Build self-contained HTML report from data dict (avoids f-string issues with CSS/JS curly braces)"""

    # Helper to escape for HTML
    def esc(s):
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # 1. Moji table
    moji_rows = ""
    for row in data['moji_list']:
        moji_rows += "<tr><td>{sym}</td><td>{day}</td><td>{gain:.1f}%</td><td>{op:.6f}</td><td>{hi:.6f}</td><td>{vol:.2f}</td></tr>".format(
            sym=esc(row['symbol']), day=esc(row['day']),
            gain=row['max_intraday_gain_pct'], op=row['open'], hi=row['high'], vol=row['total_volume'])

    # 2. Comparison table
    comp_rows = ""
    for feat, vals in data['comparison'].items():
        dc = "green" if vals['diff_pct'] > 30 else "red" if vals['diff_pct'] < -30 else "#8b949e"
        comp_rows += "<tr><td>{f}</td><td>{m:.4f}</td><td>{c:.4f}</td><td style='color:{dc};font-weight:bold'>{d:.1f}%</td></tr>".format(
            f=esc(feat), m=vals['moji_mean'], c=vals['control_mean'], d=vals['diff_pct'], dc=dc)

    # 3. Rules table
    rules_rows = ""
    for r in data['algo_rules']:
        rules_rows += "<tr><td>{id}</td><td><b>{n}</b></td><td>{d}</td><td>{c}</td><td>{w}%</td></tr>".format(
            id=r['id'], n=esc(r['name']), d=esc(r['desc']), c=esc(r['cond']), w=r['weight'])

    # 4. Pseudocode
    pseudo_lines = []
    t = data['thresholds']
    pseudo_lines.append("def moji_coin_detector(kline_5m, lookback_hours=6):")
    pseudo_lines.append("    bars = lookback_hours * 12  # 72 bars")
    pseudo_lines.append("    pre_df = kline_5m.tail(bars)")
    pseudo_lines.append("    score = 0; triggered = []")
    pseudo_lines.append("")
    pseudo_lines.append("    # R1: Volume Spike")
    pseudo_lines.append("    vol_ratio = last_1h_volume / earlier_5h_volume")
    pseudo_lines.append(f"    if vol_ratio >= {t.get('volume_ratio',1.5):.1f}: score += 25; triggered.append('R1')")
    pseudo_lines.append("")
    pseudo_lines.append("    # R2: Price Staircase")
    pseudo_lines.append("    stair_pct = count(close[i] >= close[i-1]) / 5")
    pseudo_lines.append(f"    if stair_pct >= {t.get('staircase_score',0.5):.2f}: score += 20; triggered.append('R2')")
    pseudo_lines.append("")
    pseudo_lines.append("    # R3: Volume Staircase")
    pseudo_lines.append("    vol_stair_pct = count(vol[i] >= vol[i-1]) / 5")
    pseudo_lines.append(f"    if vol_stair_pct >= {t.get('volume_staircase_score',0.4):.2f}: score += 20; triggered.append('R3')")
    pseudo_lines.append("")
    pseudo_lines.append("    # R4: Range Expansion")
    pseudo_lines.append("    range_ratio = recent_range_avg / earlier_range_avg")
    pseudo_lines.append(f"    if range_ratio >= {t.get('range_compression_ratio',1.0):.2f}: score += 15; triggered.append('R4')")
    pseudo_lines.append("")
    pseudo_lines.append("    # R5: Mild Accumulation")
    pseudo_lines.append("    cum_change = (latest_close - first_open) / first_open * 100")
    pseudo_lines.append(f"    if cum_change >= {t.get('pre_change_pct',2):.2f}: score += 10; triggered.append('R5')")
    pseudo_lines.append("")
    pseudo_lines.append("    # R6: Volatility Compression")
    pseudo_lines.append("    volatility = log_returns.std() * sqrt(288)")
    pseudo_lines.append(f"    if volatility <= {t.get('pre_volatility',0.5):.4f}: score += 10; triggered.append('R6')")
    pseudo_lines.append("")
    pseudo_lines.append("    if score >= 80 and len(triggered) >= 3: level = 'STRONG'")
    pseudo_lines.append("    elif score >= 60 and len(triggered) >= 2: level = 'MEDIUM'")
    pseudo_lines.append("    elif score >= 40: level = 'WEAK'")
    pseudo_lines.append("    else: level = 'NONE'")
    pseudo_lines.append("")
    pseudo_lines.append("    return {'score': score, 'level': level, 'triggered_rules': triggered}")
    pseudocode_html = esc("\n".join(pseudo_lines))

    # 5. Chart data as JSON
    gain_json = json.dumps(data['gain_data'])
    radar_labels_json = json.dumps(data['radar_labels'])
    moji_radar_json = json.dumps(data['moji_radar_norm'])
    control_radar_json = json.dumps(data['control_radar_norm'])

    price_datasets_json = json.dumps(data['price_datasets'])
    price_dates_json = json.dumps(data['price_dates'])

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Gate Moji-Coin Analysis Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "PingFang SC", "Segoe UI", sans-serif; max-width: 1080px; margin: 0 auto; padding: 24px; background: #0d1117; color: #c9d1d9; }
.hero { background: linear-gradient(135deg, #ff6b35 0%, #f7931a 50%, #e85d04 100%); color: white; padding: 40px; border-radius: 16px; margin-bottom: 32px; text-align: center; }
.hero h1 { font-size: 36px; margin-bottom: 8px; }
.hero .subtitle { font-size: 16px; opacity: 0.9; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px; }
.stat-box { background: rgba(255,255,255,0.15); padding: 16px; border-radius: 10px; text-align: center; }
.stat-box .num { font-size: 28px; font-weight: bold; }
.stat-box .label { font-size: 13px; opacity: 0.8; }
.card { background: #161b22; border: 1px solid #30363d; padding: 24px; border-radius: 12px; margin-bottom: 24px; }
.card h2 { color: #f0883e; margin-bottom: 16px; font-size: 20px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #30363d; font-size: 14px; }
th { color: #f0883e; font-weight: 600; background: #1c2128; }
tr:hover { background: #1c2128; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
.tag-green { background: #238636; color: white; }
.tag-red { background: #da3633; color: white; }
.tag-yellow { background: #d29922; color: black; }
.code-block { background: #1c2128; border: 1px solid #30363d; padding: 16px; border-radius: 8px; font-family: "Cascadia Code", monospace; font-size: 13px; overflow-x: auto; white-space: pre; }
.disclaimer { color: #8b949e; font-size: 13px; padding: 16px; border-top: 1px solid #30363d; margin-top: 32px; text-align: center; }
.si { color: #8b949e; margin-bottom: 12px; font-size: 14px; }
</style>
</head>
<body>

<div class="hero">
    <h1>&#x1F525; Gate Moji-Coin Analysis</h1>
    <div class="subtitle">Intraday gain >20%: pre-pump feature mining &amp; detection algorithm</div>
    <div class="stats-grid">
        <div class="stat-box"><div class="num">""" + str(data['n_coins']) + """</div><div class="label">Coins Analyzed</div></div>
        <div class="stat-box"><div class="num">""" + str(data['n_moji']) + """</div><div class="label">Moji Events</div></div>
        <div class="stat-box"><div class="num">""" + f"{data['max_gain']:.0f}" + """%</div><div class="label">Max Gain</div></div>
        <div class="stat-box"><div class="num">""" + str(data['n_rules']) + """</div><div class="label">Detection Rules</div></div>
    </div>
</div>

<div class="card">
    <h2>&#x1F4CB; Moji-Coin Events (Intraday Gain &ge;20%)</h2>
    <div class="si">Coins with max intraday gain exceeding 20% in a single day</div>
    <table>
        <thead><tr><th>Symbol</th><th>Date</th><th>Max Gain</th><th>Open</th><th>High</th><th>Volume</th></tr></thead>
        <tbody>""" + moji_rows + """</tbody>
    </table>
</div>

<div class="card">
    <h2>&#x1F4CA; Gain Distribution</h2>
    <canvas id="gainDistChart" height="300"></canvas>
</div>

<div class="card">
    <h2>&#x1F50D; Pre-Pump Feature Comparison (Moji vs Normal)</h2>
    <div class="si">6-hour pre-pump features: moji fingerprint analysis</div>
    <table>
        <thead><tr><th>Feature</th><th>Moji Mean</th><th>Control Mean</th><th>Diff</th></tr></thead>
        <tbody>""" + comp_rows + """</tbody>
    </table>
    <canvas id="radarChart" height="350"></canvas>
</div>

<div class="card">
    <h2>&#x1F9E0; Detection Algorithm: """ + esc(data['algo_name']) + """</h2>
    <div class="si">
        Accumulation-Breakout pattern detection<br>
        <span class="tag tag-yellow">5min</span>
        <span class="tag tag-green">6h lookback</span>
        <span class="tag tag-red">Weighted score</span>
    </div>
    <table>
        <thead><tr><th>ID</th><th>Name</th><th>Description</th><th>Condition</th><th>Weight</th></tr></thead>
        <tbody>""" + rules_rows + """</tbody>
    </table>
    <div class="si" style="margin-top:16px">
        <b>Scoring:</b> weighted sum, trigger at &ge;""" + str(data['trigger_threshold']) + """<br>
        &bull; Strong: &ge;80 + 3+ rules<br>
        &bull; Medium: &ge;60 + 2+ rules<br>
        &bull; Weak: &ge;40 + 1+ rules
    </div>
</div>

<div class="card">
    <h2>&#x1F4C8; Top Moji-Coin Price Trend (Recent 2 Days)</h2>
    <canvas id="priceChart" height="350"></canvas>
</div>

<div class="card">
    <h2>&#x1F4BB; Algorithm Pseudocode</h2>
    <div class="code-block">""" + pseudocode_html + """</div>
</div>

<div class="disclaimer">
    &#x26A0;&#xFE0F; This analysis is based on Gate exchange public K-line data, generated by AI, for reference only, not investment advice.<br>
    Moji-coins are extremely volatile and risky. Do not chase blindly. Investment has risks, decisions require caution.<br>
    Data source: Gate 5m K-line &middot; Analysis date: 2026-07-02
</div>

<script>
const gains = """ + gain_json + """;
const gainBuckets = {};
gains.forEach(g => {
    const bucket = g < 30 ? '20-30%' : g < 50 ? '30-50%' : g < 100 ? '50-100%' : '100%+';
    gainBuckets[bucket] = (gainBuckets[bucket] || 0) + 1;
});
new Chart(document.getElementById('gainDistChart').getContext('2d'), {
    type: 'bar',
    data: {
        labels: Object.keys(gainBuckets),
        datasets: [{ label: 'Moji Coins', data: Object.values(gainBuckets),
            backgroundColor: ['#f0883e','#da3633','#a371f7','#238636'], borderRadius: 6 }]
    },
    options: { responsive: true,
        plugins: { title: { display: true, text: 'Moji-Coin Gain Distribution', color: '#c9d1d9' }},
        scales: { x: { ticks: { color: '#8b949e' }}, y: { ticks: { color: '#8b949e' }} }
    }
});

const radarLabels = """ + radar_labels_json + """;
const mojiNorm = """ + moji_radar_json + """;
const ctrlNorm = """ + control_radar_json + """;
new Chart(document.getElementById('radarChart').getContext('2d'), {
    type: 'radar',
    data: {
        labels: radarLabels,
        datasets: [
            { label: 'Moji', data: mojiNorm, borderColor: '#f0883e', backgroundColor: 'rgba(240,136,62,0.2)', pointBackgroundColor: '#f0883e' },
            { label: 'Normal', data: ctrlNorm, borderColor: '#8b949e', backgroundColor: 'rgba(139,148,158,0.2)', pointBackgroundColor: '#8b949e' }
        ]
    },
    options: { responsive: true,
        scales: { r: { beginAtZero: true, max: 10, grid: { color: '#30363d' },
            pointLabels: { color: '#c9d1d9', font: { size: 11 }}, ticks: { color: '#8b949e' }}},
        plugins: { title: { display: true, text: 'Pre-Pump Feature Radar', color: '#c9d1d9' }}
    }
});

const pDates = """ + price_dates_json + """;
const pSets = """ + price_datasets_json + """;
const colors = ['#f0883e','#a371f7','#238636','#58a6ff','#d29922'];
const ds = pSets.map((d,i) => ({ label: d.label, data: d.data, borderColor: colors[i], backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0 }));
new Chart(document.getElementById('priceChart').getContext('2d'), {
    type: 'line',
    data: { labels: pDates, datasets: ds },
    options: { responsive: true,
        plugins: { title: { display: true, text: 'Top Moji-Coin Recent Price', color: '#c9d1d9' }},
        scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 12 }}, y: { ticks: { color: '#8b949e' }}},
        elements: { line: { tension: 0.2 }}
    }
});
</script>
</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print("Gate Moji-Coin Analysis: Pre-Pump Feature Mining")
    print("=" * 60)

    print("\n[Step 1] Loading 5m K-line data...")
    coin_data = load_all_5m_data()
    print(f"  Loaded {len(coin_data)} coins")

    print("\n[Step 2] Computing daily stats, finding moji coins (>20%)...")
    daily_df = compute_daily_stats(coin_data)
    print(f"  {len(daily_df)} day-records total")

    moji_df = find_moji_coins(daily_df, threshold=20.0)
    print(f"  Moji events: {len(moji_df)}")
    if len(moji_df) > 0:
        print(f"  Max gain: {moji_df['max_intraday_gain_pct'].max():.1f}%")
        print(f"  Unique symbols: {moji_df['symbol'].nunique()}")
        for _, row in moji_df.iterrows():
            print(f"    {row['symbol']} | {row['day']} | {row['max_intraday_gain_pct']:.1f}%")

    print("\n[Step 3] Analyzing pre-pump features (6h lookback)...")
    if len(moji_df) > 0:
        moji_f = analyze_moji_pre_features(coin_data, moji_df, lookback_hours=6)
        print(f"  Extracted features: {len(moji_f)} events")
        if len(moji_f) > 0:
            for col in ['pre_change_pct', 'volume_ratio_last1h_vs_avg', 'staircase_score',
                        'volume_staircase_score', 'range_compression_ratio', 'pre_volatility']:
                if col in moji_f.columns:
                    v = moji_f[col].dropna()
                    print(f"    {col}: mean={v.mean():.4f}, median={v.median():.4f}")
    else:
        moji_f = pd.DataFrame()
        print("  No moji events found")

    print("\n[Step 4] Computing control group features...")
    control_f = compute_control_features(coin_data, daily_df, moji_df, sample_per_day=20, lookback_hours=6)
    print(f"  Control samples: {len(control_f)}")

    print("\n[Step 5] Feature comparison...")
    if len(moji_f) > 0 and len(control_f) > 0:
        comp_df = compare_features(moji_f, control_f)
        print(f"  {len(comp_df)} features compared")
        for idx, row in comp_df.iterrows():
            print(f"    {idx}: moji={row['moji_mean']:.4f}, ctrl={row['control_mean']:.4f}, diff={row['diff_pct']:.1f}%")
    else:
        comp_df = pd.DataFrame()

    print("\n[Step 6] Designing detection algorithm...")
    if len(comp_df) > 0:
        algo = design_moji_detector(comp_df, moji_f)
        print(f"  Algorithm: {algo['name']}")
        for r in algo['rules']:
            print(f"    {r['id']} {r['name']}: {r['cond']} (weight {r['weight']})")
    else:
        algo = {'name': 'N/A', 'rules': [], 'thresholds': {}, 'scoring': {'trigger_threshold': 0}}

    print("\n[Step 7] Building HTML report...")

    # Prepare chart data
    moji_list = moji_df.head(30).to_dict('records') if len(moji_df) > 0 else []
    # comp_df.to_dict() returns {col: {row_idx: val}} not {row_idx: {col: val}}
    # We need orient='index' to get {feature: {moji_mean: ..., control_mean: ..., diff_pct: ...}}
    comparison = comp_df.to_dict(orient='index') if len(comp_df) > 0 else {}
    # comp_df.to_dict() returns {feature: {col: val}} structure
    # e.g. {'pre_change_pct': {'moji_mean': 3.88, 'control_mean': -0.21, ...}}

    gain_data = moji_df['max_intraday_gain_pct'].values.tolist() if len(moji_df) > 0 else []

    radar_labels = list(comparison.keys()) if comparison else []
    moji_radar_raw = [comparison[k].get('moji_mean', 0) for k in radar_labels]
    control_radar_raw = [comparison[k].get('control_mean', 0) for k in radar_labels]
    scale_max = max(max(abs(v) for v in moji_radar_raw) if moji_radar_raw else 1,
                    max(abs(v) for v in control_radar_raw) if control_radar_raw else 1, 1)
    moji_radar_norm = [abs(v)/scale_max * 10 for v in moji_radar_raw]
    control_radar_norm = [abs(v)/scale_max * 10 for v in control_radar_raw]

    top_moji_sym = moji_df['symbol'].head(5).tolist() if len(moji_df) > 0 else []
    price_datasets = []
    price_dates = []
    for sym in top_moji_sym:
        if sym in coin_data:
            df = coin_data[sym].tail(576)
            price_datasets.append({'label': sym, 'data': df['close'].values.tolist()})
            if len(price_dates) == 0:
                price_dates = [str(d) for d in df.index.tolist()]

    data = {
        'n_coins': len(coin_data), 'n_moji': len(moji_df),
        'max_gain': moji_df['max_intraday_gain_pct'].max() if len(moji_df) > 0 else 0,
        'n_rules': len(algo['rules']),
        'moji_list': moji_list, 'comparison': comparison,
        'gain_data': gain_data,
        'radar_labels': radar_labels,
        'moji_radar_norm': moji_radar_norm, 'control_radar_norm': control_radar_norm,
        'price_datasets': price_datasets, 'price_dates': price_dates,
        'algo_name': algo['name'], 'algo_rules': algo['rules'],
        'thresholds': algo['thresholds'],
        'trigger_threshold': algo['scoring']['trigger_threshold'],
    }

    html = build_html(data)
    report_path = OUTPUT_DIR / "gate-moji-coin-analysis-2026-07-02.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Report saved: {report_path}")

    # Save CSV data
    if len(moji_f) > 0:
        moji_f.to_csv(OUTPUT_DIR / "moji_features.csv", index=False)
    if len(control_f) > 0:
        control_f.to_csv(OUTPUT_DIR / "control_features.csv", index=False)
    if len(comp_df) > 0:
        comp_df.to_csv(OUTPUT_DIR / "feature_comparison.csv")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    return moji_df, moji_f, comp_df, algo


if __name__ == "__main__":
    main()
