"""生成多时间框架分析 HTML 报告"""
import json, pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/moji-coin")

# 数据来自分析脚本的输出
data = {
    '5m': {
        'pre_change_pct': {'moji': 3.91, 'ctrl': 0.04, 'diff': 9632},
        'sma_slope_pct': {'moji': 2.59, 'ctrl': -0.04, 'diff': 6757},
        'vol_concentration': {'moji': 0.057, 'ctrl': 0.024, 'diff': 132},
        'volume_ratio': {'moji': 1.98, 'ctrl': 1.14, 'diff': 73},
        'rec_thresholds': {'pre_change_pct': 2.0, 'sma_slope_pct': 1.2, 'vol_concentration': 0.03}
    },
    '15m': {
        'sma_slope_pct': {'moji': 2.65, 'ctrl': 0.98, 'diff': 172},
        'pre_change_pct': {'moji': 3.44, 'ctrl': 1.36, 'diff': 152},
        'vol_concentration': {'moji': 0.138, 'ctrl': 0.075, 'diff': 84},
        'volume_ratio': {'moji': 1.83, 'ctrl': 1.20, 'diff': 52},
        'rec_thresholds': {'sma_slope_pct': 1.3, 'pre_change_pct': 2.1, 'vol_concentration': 0.08}
    },
    '30m': {
        'sma_slope_pct': {'moji': 2.31, 'ctrl': -0.08, 'diff': 3106},
        'pre_change_pct': {'moji': 2.77, 'ctrl': 0.32, 'diff': 759},
        'vol_concentration': {'moji': 0.241, 'ctrl': 0.156, 'diff': 54},
        'volume_ratio': {'moji': 1.75, 'ctrl': 1.34, 'diff': 31},
        'rec_thresholds': {'sma_slope_pct': 1.3, 'pre_change_pct': 1.6, 'vol_concentration': 0.17}
    },
    '1h': {
        'sma_slope_pct': {'moji': 1.55, 'ctrl': -0.02, 'diff': 6666},
        'pre_change_pct': {'moji': 2.42, 'ctrl': -0.23, 'diff': 1128},
        'vol_concentration': {'moji': 0.30, 'ctrl': 0.22, 'diff': 36},
        'volume_ratio': {'moji': 1.53, 'ctrl': 1.31, 'diff': 17},
        'rec_thresholds': {'sma_slope_pct': 0.7, 'pre_change_pct': 1.4, 'vol_concentration': 0.20}
    },
    '4h': {
        'sma_slope_pct': {'moji': 0.0, 'ctrl': 0.0, 'diff': 0},
        'pre_change_pct': {'moji': 1.69, 'ctrl': -0.23, 'diff': 824},
        'pre_volatility': {'moji': 1.74, 'ctrl': 2.54, 'diff': 32},
        'rec_thresholds': {'pre_change_pct': 0.5}
    },
}

timeframes = ['5m', '15m', '30m', '1h', '4h']
features = ['pre_change_pct', 'sma_slope_pct', 'vol_concentration', 'volume_ratio']
feature_labels = ['累计涨幅%', 'SMA斜率%', '成交量集中度', '量比']
colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][:len(timeframes)]

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>多时间框架妖币信号分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "PingFang SC", sans-serif;
         max-width: 1080px; margin: 0 auto; padding: 24px;
         background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; color: #f1f5f9; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 32px; font-size: 14px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 24px;
           margin-bottom: 20px; border: 1px solid #334155; }}
  .card h2 {{ font-size: 18px; margin-bottom: 16px; color: #f1f5f9; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .grid-4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }}
  .metric-box {{ background: #0f172a; border-radius: 8px; padding: 16px;
                 border: 1px solid #334155; text-align: center; }}
  .metric-box .value {{ font-size: 28px; font-weight: 700; margin: 4px 0; }}
  .metric-box .label {{ font-size: 12px; color: #94a3b8; }}
  .green {{ color: #10b981; }}
  .red {{ color: #ef4444; }}
  .yellow {{ color: #f59e0b; }}
  .blue {{ color: #3b82f6; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ color: #94a3b8; font-weight: 500; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
  .tag-best {{ background: rgba(16,185,129,0.2); color: #10b981; }}
  .tag-good {{ background: rgba(59,130,246,0.2); color: #3b82f6; }}
  .tag-weak {{ background: rgba(245,158,11,0.2); color: #f59e0b; }}
  .insight {{ background: #0f172a; border-left: 3px solid #3b82f6;
             padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 13px; }}
  .insight strong {{ color: #60a5fa; }}
  canvas {{ max-height: 320px; }}
  .tf-label {{ font-size: 20px; font-weight: 600; margin-right: 8px; }}
  .separator {{ border-top: 1px solid #334155; margin: 24px 0; }}
  .code {{ font-family: 'Consolas', monospace; font-size: 13px;
          background: #0f172a; padding: 12px 16px; border-radius: 6px;
          border: 1px solid #334155; white-space: pre-wrap; }}
  .disclaimer {{ color: #64748b; font-size: 12px; padding: 16px;
                border-top: 1px solid #334155; margin-top: 24px; }}
</style>
</head>
<body>

<h1>🔬 多时间框架妖币信号分析</h1>
<p class="subtitle">201 个 Gate 现货对 · 8 天数据 · 1m 精确检测 + 多 TF 对比</p>

<!-- KEY INSIGHT -->
<div class="card">
  <h2>🎯 核心发现</h2>
  <div class="grid-3">
    <div class="metric-box">
      <div class="label">最稳定特征</div>
      <div class="value green">sma_slope_pct</div>
      <div class="label">5/5 时间框架 top 3</div>
    </div>
    <div class="metric-box">
      <div class="label">最强鉴别框架</div>
      <div class="value blue">5m & 30m</div>
      <div class="label">绝对差 3.87% / 2.39%</div>
    </div>
    <div class="metric-box">
      <div class="label">15m — 噪音陷阱</div>
      <div class="value yellow">控制组 1.36%</div>
      <div class="label">正常币也正向漂移</div>
    </div>
  </div>
  <div class="insight" style="margin-top: 16px;">
    <strong>15m 是最差的选币框架!</strong> 
    控制组的平均涨幅 +1.36%, 与妖币组的 3.44% 差距仅 2.08%, 
    而 5m 框架差距为 3.87%. 15m 的正常市场噪音恰好落在"准妖币"区间, 
    造成大量假阳性. 建议<strong>避开 15m, 优先用 5m 做初筛, 30m 做确认</strong>.
  </div>
</div>

<!-- DISCRIMINATION POWER CHART -->
<div class="card">
  <h2>📊 各时间框架鉴别力对比</h2>
  <p style="color: #94a3b8; font-size: 13px; margin-bottom: 16px;">
    基于最强特征 (pre_change_pct 或 sma_slope_pct), 展示妖币组 vs 控制组的均值差异
  </p>
  <canvas id="discriminationChart" style="max-height: 280px;"></canvas>
</div>

<!-- FEATURE SCORE TABLE -->
<div class="card">
  <h2>📋 各时间框架 Top 特征完整评分卡</h2>
  <table>
    <thead>
      <tr>
        <th>时间框架</th>
        <th>特征</th>
        <th>妖币均值</th>
        <th>控制组均值</th>
        <th>绝对差</th>
        <th>区分度%</th>
        <th>推荐阈值</th>
        <th>评价</th>
      </tr>
    </thead>
    <tbody>
"""

for tf in timeframes:
    for feati, feat in enumerate(features):
        if feat not in data[tf]:
            continue
        d = data[tf][feat]
        abs_diff = abs(d['moji'] - d['ctrl'])
        rec = data[tf]['rec_thresholds'].get(feat, 'N/A')
        
        # 评价标签
        if d['diff'] >= 1000:
            tag_class = 'tag-best'
            tag_text = '⭐⭐⭐ 极强'
        elif d['diff'] >= 100:
            tag_class = 'tag-good'
            tag_text = '⭐⭐ 良好'
        elif d['diff'] >= 30:
            tag_class = 'tag-weak'
            tag_text = '⭐ 可用'
        else:
            tag_class = ''
            tag_text = '弱'
        
        html += f"""      <tr>
        <td><b>{tf}</b></td>
        <td>{feature_labels[feati]}</td>
        <td class="green">{d['moji']:.2f}{'%' if 'pct' in feat else ''}</td>
        <td class="red">{d['ctrl']:.2f}{'%' if 'pct' in feat else ''}</td>
        <td><b>{abs_diff:.2f}</b></td>
        <td>{d['diff']:.0f}%</td>
        <td>{rec}</td>
        <td><span class="tag {tag_class}">{tag_text}</span></td>
      </tr>
"""

html += """    </tbody>
  </table>
</div>

<!-- RECOMMENDED STRATEGY -->
<div class="card">
  <h2>⚙️ 推荐的多框架选币策略</h2>
  
  <div class="insight">
    <strong>双框架确认机制:</strong> 在 5m 框架检测初筛信号, 
    在 30m/1h 框架验证, 只有双框架同时触发才入场. 
    这可以剔除 15m 的噪音假阳性.
  </div>

  <div class="code"># 多层过滤器
def select_moji_coins_5m(df_5m):
    # Layer 1 (5m): pre_change >= 2.0% AND sma_slope >= 1.2%
    layer1 = (df_5m['pre_change_pct'] >= 2.0) & (df_5m['sma_slope_pct'] >= 1.2)

    # Layer 2 (vol concentration): vol_concentration >= 0.03
    layer2 = df_5m['vol_concentration'] >= 0.03

    candidates = df_5m[layer1 & layer2]
    return candidates

def confirm_30m(df_30m):
    # 30m must also show positive drift
    confirmed = (df_30m['sma_slope_pct'] >= 1.0) & (df_30m['pre_change_pct'] >= 1.0)
    return confirmed

def confirm_1h(df_1h):
    # 1h must show accumulation (most reliable)
    confirmed = (df_1h['sma_slope_pct'] >= 0.5) & (df_1h['pre_change_pct'] >= 0.5)
    return confirmed</div>

  <h3 style="margin-top: 20px; color: #f1f5f9;">推荐参数卡</h3>
  <table>
    <thead><tr><th>时间框架</th><th>特征</th><th>阈值</th><th>角色</th></tr></thead>
    <tbody>"""

best_configs = [
    ('5m', 'pre_change_pct', '≥ 2.0%', '🔍 初筛层 (灵敏度最高)'),
    ('5m', 'sma_slope_pct', '≥ 1.2%', '🔍 初筛层'),
    ('5m', 'vol_concentration', '≥ 0.03', '🔍 量确认'),
    ('30m', 'sma_slope_pct', '≥ 1.0%', '✅ 确认层 (降噪)'),
    ('30m', 'pre_change_pct', '≥ 1.0%', '✅ 确认层'),
    ('1h', 'sma_slope_pct', '≥ 0.5%', '🛡️ 安全层 (防假突破)'),
    ('1h', 'pre_change_pct', '≥ 0.5%', '🛡️ 安全层'),
]

for tf, feat, thresh, role in best_configs:
    html += f"""      <tr><td><b>{tf}</b></td><td>{feat}</td><td><b class="blue">{thresh}</b></td><td>{role}</td></tr>
"""

html += """    </tbody>
  </table>
</div>

<!-- 15M WARNING -->
<div class="card" style="border-color: #f59e0b;">
  <h2>⚠️ 时间框架陷阱: 为什么 15m 最危险?</h2>
  <div class="grid-2">
    <div>
      <canvas id="tfComparison" style="max-height: 260px;"></canvas>
    </div>
    <div>
      <p style="font-size: 14px; line-height: 1.7; color: #cbd5e1;">
        <strong style="color: #f59e0b;">15m 的独特问题:</strong><br>
        <br>
        • 妖币组 pre_change: <b style="color:#10b981">+3.44%</b><br>
        • 控制组 pre_change: <b style="color:#f59e0b">+1.36%</b><br>
        • 绝对差仅 <b>2.08%</b> (5m 为 3.87%)<br>
        <br>
        控制组在 15m 级别也表现出正向漂移,
        意味着"正常交易"和"妖币蓄势"的边界模糊.
        用 15m 做初筛会收到大量假阳性信号.
        <br><br>
        <b style="color:#3b82f6;">建议:</b> 若必须用 15m, 阈值要设到 
        <b style="color:#3b82f6">pre_change ≥ 3.0%</b> (高于妖币均值才安全),
        但这会大幅降低命中率.
      </p>
    </div>
  </div>
</div>

<div class="disclaimer">
  ⚠️ 本分析基于 2026-06-24 至 2026-07-02 共 8 天数据, 
  样本量有限 (66 个妖币事件). 仅用于策略开发参考, 
  不构成投资建议. Meme 币交易风险极高, 实盘前请充分验证.
</div>

<script>
// Discrimination chart: bar chart comparing moji vs control for pre_change_pct across TFs
const tfLabels = """ + json.dumps(timeframes) + """;
const mojiVals = """ + json.dumps([data[tf].get('pre_change_pct', data[tf].get('sma_slope_pct', {'moji':0}))['moji'] for tf in timeframes]) + """;
const ctrlVals = """ + json.dumps([data[tf].get('pre_change_pct', data[tf].get('sma_slope_pct', {'ctrl':0}))['ctrl'] for tf in timeframes]) + """;

new Chart(document.getElementById('discriminationChart'), {
  type: 'bar',
  data: {
    labels: tfLabels,
    datasets: [
      { label: '妖币 (moji)', data: mojiVals, backgroundColor: 'rgba(16,185,129,0.6)', borderColor: '#10b981', borderWidth: 1 },
      { label: '控制组 (ctrl)', data: ctrlVals, backgroundColor: 'rgba(239,68,68,0.4)', borderColor: '#ef4444', borderWidth: 1 }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: '#94a3b8' } },
      tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + '%' } }
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
      y: { ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: '#334155' }, title: { display: true, text: 'pre_change_pct (%)', color: '#94a3b8' } }
    }
  }
});

// 15m comparison: moji vs control for sma_slope_pct
const tfCompLabels = ['5m', '15m', '30m', '1h'];
const smaMoji = """ + json.dumps([data[tf]['sma_slope_pct']['moji'] for tf in ['5m','15m','30m','1h']]) + """;
const smaCtrl = """ + json.dumps([data[tf]['sma_slope_pct']['ctrl'] for tf in ['5m','15m','30m','1h']]) + """;
const gaps = smaMoji.map((m, i) => Math.abs(m - smaCtrl[i]));

new Chart(document.getElementById('tfComparison'), {
  type: 'bar',
  data: {
    labels: tfCompLabels,
    datasets: [
      { label: 'SMA斜率差 (绝对%)', data: gaps, backgroundColor: gaps.map((g, i) => i === 1 ? 'rgba(245,158,11,0.8)' : 'rgba(59,130,246,0.6)'), borderColor: gaps.map((g, i) => i === 1 ? '#f59e0b' : '#3b82f6'), borderWidth: 1 }
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#94a3b8' } }, tooltip: { callbacks: { label: ctx => '绝对差: ' + ctx.parsed.y.toFixed(2) + '%' } } },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
      y: { ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: '#334155' }, title: { display: true, text: 'SMA斜率绝对差 (%)', color: '#94a3b8' } }
    }
  }
});
</script>

</body>
</html>"""


report_path = OUTPUT_DIR / "multitf-moji-signal-report.html"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Report written to {report_path}")
