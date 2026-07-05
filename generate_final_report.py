# -*- coding: utf-8 -*-
"""
生成 Meme_限制定投_马丁 最终回测报告 HTML。
含: 绩效指标表、类别汇总、风险分析、关键发现。
"""
import json
from pathlib import Path
from datetime import datetime

BASE = Path(r"F:\source\freqtrade")
OUT_DIR = BASE / "deliverables" / "backtest_meme_limited_20260705_211205"

with open(OUT_DIR / "results_parsed.json") as f:
    data = json.load(f)

results = data["results"]

# 按类别分组
cats = {
    "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中端中型币": ["XCN/USDT"],
    "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

# ── helpers ──

def color(v, typ="profit", dark=True):
    """根据数值返回颜色。"""
    if typ == "profit":
        if v > 0.05:
            return "#00cc66" if dark else "#006600"
        elif v > 0:
            return "#66ff99" if dark else "#338800"
        elif v > -0.3:
            return "#ffcc00" if dark else "#cc8800"
        else:
            return "#ff4444" if dark else "#cc0000"
    elif typ == "sharpe":
        if v > 3:
            return "#00cc66" if dark else "#006600"
        elif v > 1:
            return "#66ff99" if dark else "#338800"
        elif v > -1:
            return "#ffcc00" if dark else "#cc8800"
        else:
            return "#ff4444" if dark else "#cc0000"
    elif typ == "dd":
        if v < 0.5:
            return "#00cc66" if dark else "#006600"
        elif v < 1:
            return "#ffcc00" if dark else "#cc8800"
        else:
            return "#ff4444" if dark else "#cc0000"
    return "#e0e0e0" if dark else "#333"


def sigfmt(v, fmt_str="+.2f"):
    """格式化为带符号字符串。"""
    if v == 0:
        return "0.00"
    return f"{v:{fmt_str}}"


def _r(pair, tf):
    for r in results:
        if r["pair"] == pair and r["timeframe"] == tf:
            return r
    return None


# ── build HTML ──

def build_row(r, dark=True):
    """构建一行表格。"""
    return f"""
        <tr>
            <td>{r['pair']}</td>
            <td>{r['timeframe']}</td>
            <td>{r['total_trades']}</td>
            <td style="color:{color(r['win_pct'], 'profit', dark)}">{r['win_pct']:.1f}%</td>
            <td style="color:{color(r['tot_profit_pct'], 'profit', dark)};font-weight:bold">{sigfmt(r['tot_profit_pct'])}%</td>
            <td>{r['tot_profit_usdt']:+.3f}</td>
            <td style="color:{color(r['max_dd_pct'], 'dd', dark)}">{r['max_dd_pct']:.2f}%</td>
            <td style="color:{color(r['sharpe'], 'sharpe', dark)}">{r['sharpe']:.2f}</td>
            <td style="color:{color(r['sortino'], 'sharpe', dark)}">{r['sortino']:.2f}</td>
            <td>{r['profit_factor']:.2f}</td>
            <td>{r['cagr']:.2f}%</td>
            <td>{r.get('max_consecutive_wins', 0)}</td>
            <td>{r.get('max_consecutive_losses', 0)}</td>
        </tr>"""


def build_tf_row(tf, results_for_tf):
    """构建一个周期的汇总行。"""
    total_trades = sum(r["total_trades"] for r in results_for_tf)
    total_profit = sum(r["tot_profit_usdt"] for r in results_for_tf)
    n_pos = sum(1 for r in results_for_tf if r["tot_profit_pct"] > 0)
    return f"""
    <tr style="background:#1a1a3a; font-weight:bold">
        <td colspan="2">{tf} 周期汇总</td>
        <td>{total_trades}</td>
        <td>—</td>
        <td style="color:{'#00cc66' if total_profit > 0 else '#ff4444'}">{total_profit:+.3f} USDT</td>
        <td>{total_profit:+.3f}</td>
        <td>—</td>
        <td>—</td>
        <td>—</td>
        <td>—</td>
        <td>—</td>
        <td>盈利 {n_pos}/{len(results_for_tf)} 组</td>
        <td></td>
    </tr>"""


# ── 分析总结 ──

# 找出最佳/最差组合
best = max(results, key=lambda r: r["tot_profit_pct"])
worst = min(results, key=lambda r: r["tot_profit_pct"])

# 按周期统计
tfs = ["1m", "5m", "15m"]
tf_stats = {}
for tf in tfs:
    tf_r = [r for r in results if r["timeframe"] == tf]
    tf_stats[tf] = {
        "total": sum(r["total_trades"] for r in tf_r),
        "profit": sum(r["tot_profit_usdt"] for r in tf_r),
        "n_profitable": sum(1 for r in tf_r if r["tot_profit_pct"] > 0),
    }

# 按类别统计
cat_stats = {}
for cat, pairs in cats.items():
    cat_r = [r for r in results if r["pair"] in pairs]
    cat_stats[cat] = {
        "total": sum(r["total_trades"] for r in cat_r),
        "profit": sum(r["tot_profit_usdt"] for r in cat_r),
        "n_profitable": sum(1 for r in cat_r if r["tot_profit_pct"] > 0),
    }

now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meme_限制定投_马丁 · 全品种全周期回测报告</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

    .header {{ background: linear-gradient(135deg, #1a237e, #0d47a1); color: white; padding: 30px 40px; border-radius: 12px; margin-bottom: 24px; }}
    .header h1 {{ font-size: 28px; }}
    .header .meta {{ opacity: 0.8; font-size: 14px; margin-top: 8px; }}

    .card {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}

    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
    .kpi {{ text-align: center; padding: 16px; border-radius: 8px; }}
    .kpi .value {{ font-size: 28px; font-weight: bold; }}
    .kpi .label {{ font-size: 13px; color: #666; margin-top: 4px; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ background: #1a237e; color: white; padding: 10px 8px; text-align: center; position: sticky; top: 0; z-index: 1; }}
    td {{ padding: 8px; text-align: center; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #e8eaf6; }}

    h2 {{ font-size: 20px; color: #1a237e; margin-bottom: 16px; border-left: 4px solid #1a237e; padding-left: 12px; }}
    h3 {{ font-size: 16px; color: #333; margin: 12px 0 8px; }}

    .findings {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .finding {{ padding: 16px; border-radius: 8px; }}
    .finding.good {{ background: #e8f5e9; border-left: 4px solid #4caf50; }}
    .finding.bad {{ background: #fbe9e7; border-left: 4px solid #f44336; }}
    .finding.info {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}

    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
    .tag.good {{ background: #c8e6c9; color: #2e7d32; }}
    .tag.bad {{ background: #ffcdd2; color: #c62828; }}
    .tag.neutral {{ background: #fff9c4; color: #f57f17; }}

    .section-cat {{ margin: 8px 0; padding: 6px 12px; background: #e8eaf6; font-weight: bold; font-size: 14px; }}

    @media (max-width: 768px) {{
        .findings {{ grid-template-columns: 1fr; }}
        table {{ font-size: 11px; }}
    }}
</style>
</head>
<body>

<div class="container">

<div class="header">
    <h1>Meme_限制定投_马丁 全品种全周期回测报告</h1>
    <div class="meta">
        策略: MemeLimitedMartingaleSpotStrategy &nbsp;|&nbsp;
        回测周期: 2026-06-08 ~ 2026-06-27 (19天) &nbsp;|&nbsp;
        数据源: Gate.io Spot &nbsp;|&nbsp;
        生成时间: {now}
    </div>
</div>

<!-- KPI 卡片 -->
<div class="card">
    <h2>核心指标概览</h2>
    <div class="kpi-grid">
        <div class="kpi" style="background:#e8f5e9">
            <div class="value" style="color:#2e7d32">{sum(1 for r in results if r['tot_profit_pct']>0)}</div>
            <div class="label">盈利组合数 / {len(results)}</div>
        </div>
        <div class="kpi" style="background:#e3f2fd">
            <div class="value" style="color:#1565c0">{best['pair'].split('/')[0]} {best['timeframe']}</div>
            <div class="label">最佳组合 ({best['tot_profit_pct']:+.2f}%)</div>
        </div>
        <div class="kpi" style="background:#fbe9e7">
            <div class="value" style="color:#c62828">{worst['pair'].split('/')[0]} {worst['timeframe']}</div>
            <div class="label">最差组合 ({worst['tot_profit_pct']:+.2f}%)</div>
        </div>
        <div class="kpi" style="background:#fff3e0">
            <div class="value" style="color:#e65100">{sum(r['total_trades'] for r in results)}</div>
            <div class="label">总交易次数</div>
        </div>
        <div class="kpi" style="background:#f3e5f5">
            <div class="value" style="color:{'#2e7d32' if sum(r['tot_profit_usdt'] for r in results) > 0 else '#c62828'}">{sum(r['tot_profit_usdt'] for r in results):+.3f}</div>
            <div class="label">总收益 USDT</div>
        </div>
    </div>
</div>

<!-- 按周期汇总 -->
<div class="card">
    <h2>按周期汇总</h2>
    <table>
        <tr><th>周期</th><th>总交易数</th><th>总收益 USDT</th><th>盈利组数</th><th>评价</th></tr>
"""

for tf in tfs:
    s = tf_stats[tf]
    tag_cls = "good" if s["profit"] > 0 else "bad"
    review = "推荐 ✓" if s["profit"] > 0 and s["n_profitable"] >= 5 else \
             "可用 ⚠" if s["profit"] > -1 else "不推荐 ✗"
    html += f"""
        <tr>
            <td style="font-weight:bold">{tf}</td>
            <td>{s['total']}</td>
            <td style="color:{'#2e7d32' if s['profit'] > 0 else '#c62828'}">{s['profit']:+.3f}</td>
            <td>{s['n_profitable']}/9</td>
            <td><span class="tag {tag_cls}">{review}</span></td>
        </tr>"""

html += """
    </table>
</div>

<!-- 按类别汇总 -->
<div class="card">
    <h2>按币种类别汇总</h2>
    <table>
        <tr><th>类别</th><th>总交易数</th><th>总收益 USDT</th><th>盈利组数</th><th>评价</th></tr>
"""

for cat in ["稳定主流币", "中端中型币", "高波动妖币"]:
    s = cat_stats[cat]
    tag_cls = "good" if s["profit"] > 0 else "bad"
    review = "稳定盈利" if s["profit"] > 2 else ("小幅盈利" if s["profit"] > 0 else "亏损")
    html += f"""
        <tr>
            <td style="font-weight:bold">{cat}</td>
            <td>{s['total']}</td>
            <td style="color:{'#2e7d32' if s['profit'] > 0 else '#c62828'}">{s['profit']:+.3f}</td>
            <td>{s['n_profitable']}/{'12' if cat != '中端中型币' else '3'}</td>
            <td><span class="tag {tag_cls}">{review}</span></td>
        </tr>"""

html += """
    </table>
</div>

<!-- 完整指标表 -->
<div class="card">
    <h2>全币种全周期绩效指标</h2>
    <div style="overflow-x:auto; max-height:700px; overflow-y:auto">
    <table>
        <tr>
            <th>币种</th><th>周期</th><th>交易数</th><th>胜率%</th>
            <th>总收益%</th><th>收益USDT</th><th>最大回撤%</th>
            <th>Sharpe</th><th>Sortino</th><th>PF</th><th>CAGR%</th>
            <th>连胜</th><th>连亏</th>
        </tr>
"""

# 按类别显示
for cat, pairs in cats.items():
    html += f'<tr><td colspan="13" class="section-cat">{cat}</td></tr>'
    for pair in pairs:
        for tf in tfs:
            r = _r(pair, tf)
            if r:
                html += build_row(r, dark=False)

html += """
    </table>
    </div>
</div>

<!-- 关键发现与分析 -->
<div class="card">
    <h2>关键发现</h2>
    <div class="findings">
        <div class="finding good">
            <strong>最佳策略：H/USDT 5m</strong><br>
            28笔交易，+0.26%，Sharpe=5.62，Sortino=7.72<br>
            胜率 78.6%，最大回撤仅 0.19%
        </div>
        <div class="finding good">
            <strong>次优组合：COAI/USDT 5m</strong><br>
            14笔交易，+0.12%，Sharpe=7.76，Sortino=12.26<br>
            胜率 78.6%，回撤仅 0.10%
        </div>
        <div class="finding bad">
            <strong>1m 周期过频交易</strong><br>
            H 1m: 125笔/-0.06%, VELVET 1m: 101笔/-0.80%<br>
            高频交易+手续费侵蚀利润
        </div>
        <div class="finding bad">
            <strong>妖币分化严重</strong><br>
            VELVET 全部亏损（-0.06%~-0.80%）<br>
            BEAT 1m/5m 亏损，仅 15m 微盈（+0.12%）
        </div>
        <div class="finding info">
            <strong>稳定币交易信号极少</strong><br>
            BTC/ETH/SOL/XRP 在 1m/5m 几乎无信号<br>
            策略不适合低波动大市值币种
        </div>
        <div class="finding info">
            <strong>15m 周期更稳定</strong><br>
            大部分币种在 15m 有正向收益<br>
            ETH 15m: +0.03%, BEAT 15m: +0.12%
        </div>
    </div>
</div>

<!-- 风险统计 -->
<div class="card">
    <h2>风险统计</h2>
    <table>
        <tr>
            <th>币种</th><th>周期</th><th>交易数</th>
            <th>总收益%</th><th>最大回撤%</th>
            <th>最大连胜</th><th>最大连亏</th>
            <th>Sharpe</th><th>Sortino</th>
        </tr>
"""

for r in results:
    if r["total_trades"] == 0:
        continue
    risk_level = "🟢" if r["max_dd_pct"] < 0.3 else ("🟡" if r["max_dd_pct"] < 1 else "🔴")
    html += f"""
        <tr>
            <td>{r['pair']}</td>
            <td>{r['timeframe']}</td>
            <td>{r['total_trades']}</td>
            <td style="color:{color(r['tot_profit_pct'], 'profit', False)}">{sigfmt(r['tot_profit_pct'])}%</td>
            <td style="color:{color(r['max_dd_pct'], 'dd', False)}">{r['max_dd_pct']:.2f}% {risk_level}</td>
            <td>{r.get('max_consecutive_wins', 0)}</td>
            <td>{r.get('max_consecutive_losses', 0)}</td>
            <td>{r['sharpe']:.2f}</td>
            <td>{r['sortino']:.2f}</td>
        </tr>"""

html += """
    </table>
</div>

<!-- 优化建议 -->
<div class="card">
    <h2>策略优化建议</h2>
    <div class="findings">
        <div class="finding info">
            <strong>1. 聚焦 5m 周期</strong><br>
            5m 是 sweet spot：既有足够信号，又不过频。建议默认时间框架改为 5m。
        </div>
        <div class="finding info">
            <strong>2. 按币种定制参数</strong><br>
            妖币波动更大，可考虑调整布林带宽度、RSI 阈值。稳定币可放弃。
        </div>
        <div class="finding info">
            <strong>3. DCA 机制待优化</strong><br>
            当前回测 DCA 触发极少（<5%），说明加仓条件过严。可放宽触发阈值。
        </div>
        <div class="finding info">
            <strong>4. 增加波动率过滤</strong><br>
            VELVET 和 BEAT 回撤过大，建议加入 ATR 或市场状态过滤，避免高波动时逆势加仓。
        </div>
    </div>
</div>

<p style="text-align:center; color:#999; font-size:12px; margin-top:30px; padding:20px;">
    生成脚本: generate_final_report.py &nbsp;|&nbsp;
    数据源: F:\\source\\freqtrade\\deliverables\\backtest_meme_limited_20260705_211205\\
</p>

</div>
</body>
</html>
"""

# 写文件
html_file = OUT_DIR / "backtest_report.html"
html_file.write_text(html, encoding="utf-8")
print(f"✅ HTML 报告已保存: {html_file}")
print(f"\n报告内容:")
print(f"  - 27 个回测完整绩效指标")
print(f"  - 按周期/类别/币种三维汇总")
print(f"  - 核心 KPI 卡片")
print(f"  - 关键发现与分析")
print(f"  - 风险统计与优化建议")
print(f"  - CSV 数据文件: results_summary.csv, risk_stats.csv")
