"""
生成插针分析 HTML 可视化报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

FORMATTED = Path("user_data/orderbook_data/formatted")


def analyze_events(wicks, snaps, pair):
    results = []
    for _, w in wicks.iterrows():
        ts = w["timestamp"]
        et = w["event_type"]
        mag = w["magnitude_pct"]
        side = "ask" if et == "up_wick" else "bid"
        
        mag_match = abs(snaps["event_magnitude_pct"] - mag) < 0.001
        type_match = snaps["event_type"] == et
        ev = snaps[mag_match & type_match].sort_values("seconds_from_event")
        
        if len(ev) < 10:
            continue
        
        before = ev[(ev["seconds_from_event"] >= -25) & (ev["seconds_from_event"] <= -5)]
        after = ev[(ev["seconds_from_event"] >= 5) & (ev["seconds_from_event"] <= 25)]
        
        if len(before) < 3 or len(after) < 3:
            continue
        
        col = f"{side}_depth_0_1pct"
        bv = before[before[col].notna()]
        av = after[after[col].notna()]
        if len(bv) == 0 or len(av) == 0:
            continue
        
        br = bv.iloc[-1]
        ar = av.iloc[0]
        
        levels = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        level_data = []
        breached = False
        max_breach = 0
        for pct in levels:
            cn = f"{side}_depth_{str(pct).replace('.', '_')}pct"
            if cn not in br.index:
                continue
            d_b = br[cn]
            d_a = ar[cn]
            if pd.isna(d_b) or pd.isna(d_a) or d_b <= 0:
                continue
            consumed = max(d_b - d_a, 0)
            pct_c = consumed / d_b * 100
            is_b = pct_c > 50
            if is_b:
                breached = True
                max_breach = max(max_breach, pct)
            level_data.append({
                "level": pct, "before": float(d_b), "after": float(d_a),
                "consumed": float(consumed), "pct": round(pct_c, 1), "breached": is_b
            })
        
        if not level_data:
            continue
        
        total_c = sum(ld["consumed"] for ld in level_data)
        close_p = float(br.get("close", 0))
        high_p = float(ev["high"].max()) if "high" in ev else 0
        low_p = float(ev["low"].min()) if "low" in ev else 0
        
        results.append({
            "pair": pair,
            "timestamp": str(ts),
            "event_type": et,
            "magnitude_pct": round(mag, 3),
            "side": side,
            "close": close_p,
            "high": high_p,
            "low": low_p,
            "total_consumed": total_c,
            "breached": breached,
            "max_breach": max_breach,
            "levels": level_data,
        })
    
    results.sort(key=lambda x: (x["max_breach"], x["magnitude_pct"]), reverse=True)
    return results


def build_html_section(pair, results):
    """Build HTML section for one pair"""
    breached = sum(1 for r in results if r["breached"])
    up = sum(1 for r in results if r["event_type"] == "up_wick")
    down = sum(1 for r in results if r["event_type"] == "down_wick")
    
    if not results:
        return ""
    
    max_mag = max(r["magnitude_pct"] for r in results)
    max_c = max(r["total_consumed"] for r in results)
    avg_c = np.mean([r["total_consumed"] for r in results])
    
    # Top 5 breached events detail
    breached_events = [r for r in results if r["breached"]]
    detail_html = ""
    for r in breached_events[:3]:
        tag_cls = "tag-down" if r["event_type"] == "down_wick" else "tag-up"
        event_cn = "向上插针" if r["event_type"] == "up_wick" else "向下插针"
        detail_html += f"""
    <div class="card" style="margin: 15px 0;">
      <h3>{r['pair']} | {r['timestamp']} | <span class="tag {tag_cls}">{event_cn}</span> | 波动幅度: {r['magnitude_pct']:.3f}%</h3>
      <p style="color: #8b949e; margin: 5px 0;">
        价格: ${r['close']:,.2f} | 区间: ${r['low']:,.2f} - ${r['high']:,.2f} | 方向: {r['side']}
      </p>
      <table style="margin: 10px 0;">
      <tr><th>深度档位</th><th>事件前 (USDT)</th><th>事件后 (USDT)</th><th>消耗 (USDT)</th><th>消耗%</th><th>状态</th></tr>"""
        for ld in r["levels"]:
            if ld["breached"]:
                cls = 'class="breach"'
                status = '<span class="tag tag-breach">击穿</span>'
            elif ld["pct"] > 30:
                cls = ""
                status = '<span class="heavy">重耗</span>'
            elif ld["pct"] > 10:
                cls = ""
                status = '<span class="mild">消耗</span>'
            else:
                cls = ""
                status = '<span style="color:#484f58">完好</span>'
            detail_html += f"""
      <tr {cls}>
        <td>{ld['level']:.1f}%</td>
        <td>{ld['before']:,.0f}</td>
        <td>{ld['after']:,.0f}</td>
        <td>{ld['consumed']:,.0f}</td>
        <td>{ld['pct']:.1f}%</td>
        <td>{status}</td>
      </tr>"""
        detail_html += "\n      </table>\n    </div>"
    
    # Table rows
    table_rows = ""
    for i, r in enumerate(results[:15]):
        cls = "breach" if r["breached"] else ""
        lvl_str = f"{r['max_breach']:.1f}%" if r["breached"] else "-"
        n_breach = sum(1 for ld in r["levels"] if ld["breached"])
        tag_cls = "tag-down" if r["event_type"] == "down_wick" else "tag-up"
        type_cn = "向下插针" if r["event_type"] == "down_wick" else "向上插针"
        table_rows += f"""
    <tr class="{cls}">
      <td>{i+1}</td>
      <td>{r['timestamp']}</td>
      <td><span class="tag {tag_cls}">{type_cn}</span></td>
      <td>{r['magnitude_pct']:.3f}%</td>
      <td>${r['close']:,.2f}</td>
      <td>{r['total_consumed']:,.0f}</td>
      <td>{lvl_str}</td>
      <td>{n_breach}</td>
    </tr>"""
    
    pair_short = pair.split("/")[0]
    
    html = f"""
<h2>{pair} 插针分析</h2>

<div class="summary-grid">
  <div class="card"><div class="value">{len(results)}</div><div class="label">插针事件</div></div>
  <div class="card danger"><div class="value">{breached}</div><div class="label">L2 击穿</div></div>
  <div class="card"><div class="value">{max_mag:.3f}%</div><div class="label">最大幅度</div></div>
  <div class="card warn"><div class="value">${max_c/1e6:.2f}M</div><div class="label">最大消耗</div></div>
</div>

<div class="note">
  <b>{pair}</b>: 共 {len(results)} 次插针（上插针 {up} 次，下插针 {down} 次），
  其中 {breached} 次击穿 L2 订单簿（{breached/max(len(results),1)*100:.0f}%）。
  平均每次消耗深度 ${avg_c:,.0f} USDT。
</div>

<h3>TOP 15 事件</h3>
<table>
<tr><th>#</th><th>时间 (UTC)</th><th>类型</th><th>幅度%</th><th>价格</th><th>消耗 USDT</th><th>最深击穿</th><th>击穿档数</th></tr>
{table_rows}
</table>

<div id="chart_{pair_short}_mag" class="chart-container" style="height:300px;">
  <canvas id="c_{pair_short}_mag"></canvas>
</div>
<div id="chart_{pair_short}_scatter" class="chart-container" style="height:350px;">
  <canvas id="c_{pair_short}_scatter"></canvas>
</div>
"""
    
    if breached_events:
        html += f"""
<h3>最严重 L2 击穿事件</h3>
{detail_html}
"""
    
    return html


def build_report():
    btc_wicks = pd.read_csv(FORMATTED / "BTC_USDT" / "wick_events.csv")
    eth_wicks = pd.read_csv(FORMATTED / "ETH_USDT" / "wick_events.csv")
    btc_snaps = pd.read_parquet(FORMATTED / "BTC_USDT" / "wick_depth_snapshots.parquet")
    eth_snaps = pd.read_parquet(FORMATTED / "ETH_USDT" / "wick_depth_snapshots.parquet")
    
    btc_wicks["timestamp"] = pd.to_datetime(btc_wicks["timestamp"])
    eth_wicks["timestamp"] = pd.to_datetime(eth_wicks["timestamp"])
    
    btc_results = analyze_events(btc_wicks, btc_snaps, "BTC/USDT")
    eth_results = analyze_events(eth_wicks, eth_snaps, "ETH/USDT")
    
    all_results = btc_results + eth_results
    total_breached = sum(1 for r in all_results if r["breached"])
    total_consumed = sum(r["total_consumed"] for r in all_results)
    
    # Prepare chart data
    btc_mags = [r["magnitude_pct"] for r in btc_results]
    btc_consumed_k = [r["total_consumed"] / 1000 for r in btc_results]
    btc_colors = ['#f85149' if r["breached"] else '#3fb950' for r in btc_results]
    
    eth_mags = [r["magnitude_pct"] for r in eth_results]
    eth_consumed_k = [r["total_consumed"] / 1000 for r in eth_results]
    eth_colors = ['#f85149' if r["breached"] else '#3fb950' for r in eth_results]
    
    # Pre-compute all chart JSON
    btc_bar_labels = json.dumps([r['timestamp'][-8:] for r in btc_results[:30]])
    btc_bar_data = json.dumps(btc_mags[:30])
    btc_bar_colors = json.dumps(['#3fb950' if r['event_type']=='down_wick' else '#f85149' for r in btc_results[:30]])
    
    btc_scatter_data = json.dumps([{"x": btc_mags[i], "y": btc_consumed_k[i]} for i in range(len(btc_mags))])
    btc_scatter_colors = json.dumps(btc_colors)
    
    eth_bar_labels = json.dumps([r['timestamp'][-8:] for r in eth_results[:30]])
    eth_bar_data = json.dumps(eth_mags[:30])
    eth_bar_colors = json.dumps(['#3fb950' if r['event_type']=='down_wick' else '#f85149' for r in eth_results[:30]])
    
    eth_scatter_data = json.dumps([{"x": eth_mags[i], "y": eth_consumed_k[i]} for i in range(len(eth_mags))])
    eth_scatter_colors = json.dumps(eth_colors)
    
    # Hour distribution for ETH
    eth_hours = {}
    for r in eth_results:
        h = r["timestamp"][:13]
        eth_hours[h] = eth_hours.get(h, 0) + 1
    
    # Build all sections
    btc_html = build_html_section("BTC/USDT", btc_results)
    eth_html = build_html_section("ETH/USDT", eth_results)
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gate.io 插针事件 L2 订单簿深度分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 8px; }}
h2 {{ color: #f0883e; font-size: 20px; margin: 30px 0 15px; border-bottom: 2px solid #30363d; padding-bottom: 8px; }}
h3 {{ color: #d2a8ff; font-size: 16px; margin: 20px 0 10px; }}
.subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 25px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin: 20px 0; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
.card .value {{ font-size: 32px; font-weight: bold; color: #58a6ff; }}
.card .label {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
.card.danger .value {{ color: #f85149; }}
.card.warn .value {{ color: #f0883e; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
th {{ background: #21262d; color: #c9d1d9; padding: 8px 10px; text-align: left; border: 1px solid #30363d; }}
td {{ padding: 7px 10px; border: 1px solid #30363d; }}
tr:hover {{ background: #1c2129; }}
.breach {{ background: #3d1f1f !important; }}
.breach td {{ color: #f85149; font-weight: bold; }}
.heavy {{ color: #f0883e; }}
.mild {{ color: #d2a8ff; }}
.chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 15px 0; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.tag-up {{ background: #3d1f1f; color: #f85149; }}
.tag-down {{ background: #1f3d1f; color: #3fb950; }}
.tag-breach {{ background: #f85149; color: #fff; }}
.note {{ background: #1c2129; border-left: 3px solid #58a6ff; padding: 12px 16px; margin: 15px 0; font-size: 13px; color: #8b949e; }}
.footer {{ text-align: center; color: #484f58; font-size: 12px; margin-top: 40px; padding: 20px 0; border-top: 1px solid #21262d; }}
</style>
</head>
<body>
<div class="container">

<h1>Gate.io 插针事件 L2 订单簿深度分析报告</h1>
<p class="subtitle">数据范围: 2026-06-26 ~ 2026-06-30 | 插针阈值: 0.5% | 窗口: 60秒 | 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="summary-grid">
  <div class="card"><div class="value">{len(all_results)}</div><div class="label">插针事件总数</div></div>
  <div class="card danger"><div class="value">{total_breached}</div><div class="label">L2 订单簿击穿</div></div>
  <div class="card"><div class="value">${total_consumed/1e6:.1f}M</div><div class="label">累计深度消耗 (USDT)</div></div>
  <div class="card warn"><div class="value">{total_breached/max(len(all_results),1)*100:.0f}%</div><div class="label">击穿率</div></div>
</div>

<div class="note">
  <b>核心发现:</b> 在 {len(all_results)} 次插针事件中（BTC {len(btc_results)} 次，ETH {len(eth_results)} 次），
  有 {total_breached} 次（{total_breached/max(len(all_results),1)*100:.0f}%）击穿了 L2 订单簿（任一档位深度消耗 >50%）。
  BTC 全部为向下插针（共 {len(btc_results)} 次），ETH 则同时存在向上（{sum(1 for r in eth_results if r['event_type']=='up_wick')} 次）和向下（{sum(1 for r in eth_results if r['event_type']=='down_wick')} 次）插针。
  最严重的一次击穿消耗了 {max(r['total_consumed'] for r in all_results):,.0f} USDT 的深度。
</div>

{btc_html}
{eth_html}

<div class="footer">
  <p>由 orderbook_formatter.py + wick_report.py 生成 | Gate.io 现货订单簿数据 | 2026-06-26 ~ 2026-06-30 UTC</p>
  <p>插针检测: 60秒滑动窗口, 0.5% 幅度阈值, 30% 回撤要求 | 击穿定义: 任一档位深度消耗 >50%</p>
</div>

</div>

<script>
(function() {{
// BTC Magnitude
new Chart(document.getElementById('c_BTC_mag'), {{
  type: 'bar',
  data: {{
    labels: """ + btc_bar_labels + """,
    datasets: [{
      label: 'Magnitude %',
      data: """ + btc_bar_data + """,
      backgroundColor: """ + btc_bar_colors + """,
    }]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'BTC/USDT 插针幅度分布', color: '#c9d1d9' }} }},
    scales: {{
      y: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
      x: {{ ticks: {{ color: '#8b949e', maxRotation: 45, autoSkip: true }} }}
    }}
  }}
}});

// BTC Scatter
new Chart(document.getElementById('c_BTC_scatter'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'BTC/USDT',
      data: """ + btc_scatter_data + """,
      backgroundColor: """ + btc_scatter_colors + """,
      pointRadius: 5,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      title: {{ display: true, text: 'BTC/USDT 深度消耗 vs 波动幅度', color: '#c9d1d9' }},
      tooltip: {{ callbacks: {{ label: function(c) {{ return '幅度: ' + c.raw.x.toFixed(3) + '% | 消耗: $' + c.raw.y.toFixed(0) + 'K'; }} }} }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: '波动幅度 %', color: '#8b949e' }}, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
      y: {{ title: {{ display: true, text: '深度消耗 (K USDT)', color: '#8b949e' }}, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }}
  }}
}});

// ETH Magnitude
new Chart(document.getElementById('c_ETH_mag'), {{
  type: 'bar',
  data: {{
    labels: """ + eth_bar_labels + """,
    datasets: [{
      label: '幅度 %',
      data: """ + eth_bar_data + """,
      backgroundColor: """ + eth_bar_colors + """,
    }]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: 'ETH/USDT 插针幅度分布', color: '#c9d1d9' }} }},
    scales: {{
      y: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
      x: {{ ticks: {{ color: '#8b949e', maxRotation: 45, autoSkip: true }} }}
    }}
  }}
}});

// ETH Scatter
new Chart(document.getElementById('c_ETH_scatter'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'ETH/USDT',
      data: """ + eth_scatter_data + """,
      backgroundColor: """ + eth_scatter_colors + """,
      pointRadius: 5,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      title: {{ display: true, text: 'ETH/USDT 深度消耗 vs 波动幅度', color: '#c9d1d9' }},
      tooltip: {{ callbacks: {{ label: function(c) {{ return '幅度: ' + c.raw.x.toFixed(3) + '% | 消耗: $' + c.raw.y.toFixed(0) + 'K'; }} }} }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Magnitude %', color: '#8b949e' }}, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
      y: {{ title: {{ display: true, text: 'Depth Consumed (K USDT)', color: '#8b949e' }}, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }}
  }}
}});
}})();
</script>

</body>
</html>"""
    
    output_path = FORMATTED / "wick_analysis_report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


if __name__ == "__main__":
    build_report()
