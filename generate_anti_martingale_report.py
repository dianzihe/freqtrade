#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Meme_反马丁_趋势 回测 HTML 报告
"""

import json
from pathlib import Path

DATA_PATH = "E:/source/freqtrade/anti_martingale_results.json"
OUT_PATH  = "E:/source/freqtrade/anti_martingale_report.html"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

results_raw  = raw["results"]
coin_group   = raw["coin_group_map"]

COINS_ORDER = ["BTC_USDT","ETH_USDT","SOL_USDT","XRP_USDT",
               "XCN_USDT","H_USDT","VELVET_USDT","BEAT_USDT","COAI_USDT"]
TFS = ["1m","5m","15m"]

GROUP_COLOR = {
    "主流稳定":   "#3b82f6",
    "中端中型":   "#f59e0b",
    "高波动妖币": "#ef4444",
}

# ── 提取结构化数据 ─────────────────────────────────────────────────────
all_metrics = []
equity_data = {}
add_records_all = []
blowup_all = []
trade_detail_all = []

for pair in COINS_ORDER:
    equity_data[pair] = {}
    for tf in TFS:
        r = results_raw.get(pair, {}).get(tf, {})
        if "metrics" not in r:
            continue
        m = r["metrics"]
        m["group"] = coin_group.get(pair, "未知")
        all_metrics.append(m)
        equity_data[pair][tf] = r.get("equity_curve", [])
        for rec in r.get("add_records", []):
            add_records_all.append(rec)
        for ev in r.get("blowup_events", []):
            blowup_all.append(ev)
        for t in r.get("trades", []):
            t["pair"] = pair
            t["tf"] = tf
            t["group"] = coin_group.get(pair, "未知")
            trade_detail_all.append(t)

# ── 构建 Chart.js 数据 ────────────────────────────────────────────────

def eq_chart_data(pair, tf):
    ec = equity_data.get(pair, {}).get(tf, [])
    if not ec:
        return "[]", "[]"
    labels = [e["time"][11:16] for e in ec[::max(1,len(ec)//200)]]
    values = [e["equity"]      for e in ec[::max(1,len(ec)//200)]]
    return json.dumps(labels), json.dumps(values)

# ── 构建汇总表格行 ─────────────────────────────────────────────────────

def fmt_pct(v, plus=False):
    sign = "+" if (plus and v > 0) else ""
    color = "#22c55e" if v > 0 else ("#ef4444" if v < 0 else "#94a3b8")
    return f'<span style="color:{color}">{sign}{v:.2f}%</span>'

def fmt_n(v):
    return f'<span style="color:#e2e8f0">{v}</span>'

table_rows = ""
for m in all_metrics:
    pair = m["pair"]
    tf   = m["timeframe"]
    grp  = m["group"]
    gc   = GROUP_COLOR.get(grp, "#94a3b8")
    exit_dist = m.get("exit_distribution", {})
    exit_str = " | ".join(f"{k}:{v}" for k,v in exit_dist.items())
    table_rows += f"""
    <tr class="metric-row" data-group="{grp}" data-tf="{tf}" data-pair="{pair}">
      <td><span style="color:{gc};font-weight:600">{grp}</span></td>
      <td style="font-weight:700;color:#e2e8f0">{pair.replace('_USDT','')}</td>
      <td><span class="tf-badge tf-{tf}">{tf}</span></td>
      <td>{fmt_n(m['n_trades'])}</td>
      <td>{fmt_pct(m['win_rate_pct'])}</td>
      <td>{fmt_pct(m['total_return_pct'], plus=True)}</td>
      <td>{fmt_pct(m['max_drawdown_pct'])}</td>
      <td>{fmt_pct(m['worst_trade_drawdown_pct'])}</td>
      <td style="color:#e2e8f0">{m['profit_factor']:.3f}</td>
      <td style="color:#e2e8f0">{m['sharpe']:.3f}</td>
      <td style="color:#a78bfa">{m['n_add_orders']}</td>
      <td style="color:{'#ef4444' if m['n_blowup_events']>0 else '#94a3b8'}">{m['n_blowup_events']}</td>
      <td style="font-size:11px;color:#94a3b8;max-width:180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{exit_str}</td>
    </tr>"""

# ── 盈亏曲线 charts JSON ─────────────────────────────────────────────
import json as _json
equity_json = {}
for pair in COINS_ORDER:
    equity_json[pair] = {}
    for tf in TFS:
        ec = equity_data.get(pair, {}).get(tf, [])
        step = max(1, len(ec) // 300)
        equity_json[pair][tf] = {
            "labels": [e["time"][5:16] for e in ec[::step]],
            "values": [e["equity"] for e in ec[::step]],
        }

# ── 加仓记录表格 ─────────────────────────────────────────────────────
add_rows = ""
for rec in add_records_all[:500]:
    add_rows += f"""
    <tr>
      <td style="color:#94a3b8;font-size:12px">{rec['time'][5:16]}</td>
      <td style="color:#e2e8f0;font-weight:600">{rec['pair'].replace('_USDT','')}</td>
      <td><span class="tf-badge tf-{rec['tf']}">{rec['tf']}</span></td>
      <td style="color:#e2e8f0">{rec['entry_n']}</td>
      <td style="color:#e2e8f0">{rec['add_price']}</td>
      <td style="color:#22c55e">+{rec['profit_at_add_pct']}%</td>
      <td style="color:#94a3b8">{rec['threshold_pct']}%</td>
      <td style="color:#a78bfa">{rec['add_stake_usdt']} USDT</td>
      <td style="color:#e2e8f0">{rec['total_stake_usdt']} USDT</td>
    </tr>"""

if not add_rows:
    add_rows = '<tr><td colspan="9" style="text-align:center;color:#64748b;padding:20px">暂无加仓记录</td></tr>'

# ── 爆仓风险表格 ─────────────────────────────────────────────────────
blowup_rows = ""
for ev in blowup_all:
    blowup_rows += f"""
    <tr>
      <td style="color:#94a3b8;font-size:12px">{ev['time'][5:16]}</td>
      <td style="color:#e2e8f0;font-weight:600">{ev['pair'].replace('_USDT','')}</td>
      <td><span class="tf-badge tf-{ev['tf']}">{ev['tf']}</span></td>
      <td style="color:#ef4444;font-weight:700">{ev['unrealized_loss_pct']}%</td>
      <td style="color:#e2e8f0">{ev['price']}</td>
      <td style="color:#f59e0b">{ev['total_stake']} USDT</td>
      <td style="color:#e2e8f0">{ev['n_entries']}</td>
    </tr>"""

if not blowup_rows:
    blowup_rows = '<tr><td colspan="7" style="text-align:center;color:#22c55e;padding:20px">无爆仓风险事件 (浮亏均未超过20%)</td></tr>'

# ── 交易明细（取最大亏损10笔 + 最大盈利10笔） ────────────────────────
sorted_by_pnl = sorted(trade_detail_all, key=lambda x: x["pnl_usdt"])
worst10 = sorted_by_pnl[:10]
best10  = sorted_by_pnl[-10:][::-1]

def trade_row(t, highlight=None):
    color = "#22c55e" if t["pnl_usdt"] > 0 else "#ef4444"
    gc = GROUP_COLOR.get(t.get("group",""), "#94a3b8")
    return f"""<tr>
      <td style="color:{gc};font-size:12px">{t.get('group','')}</td>
      <td style="color:#e2e8f0;font-weight:600">{t['pair'].replace('_USDT','')}</td>
      <td><span class="tf-badge tf-{t['tf']}">{t['tf']}</span></td>
      <td style="color:#94a3b8;font-size:12px">{str(t['open_time'])[5:16]}</td>
      <td style="color:#94a3b8;font-size:12px">{str(t['close_time'])[5:16]}</td>
      <td style="color:#e2e8f0">{t['n_entries']}</td>
      <td style="color:{color};font-weight:700">{t['pnl_usdt']:+.4f} U</td>
      <td style="color:{color}">{t['pnl_pct']:+.2f}%</td>
      <td style="color:#ef4444">{t['max_dd_pct']:.2f}%</td>
      <td style="color:#94a3b8;font-size:11px">{t['close_reason']}</td>
    </tr>"""

worst_rows = "".join(trade_row(t) for t in worst10) or '<tr><td colspan="10" style="text-align:center;color:#64748b;padding:16px">无</td></tr>'
best_rows  = "".join(trade_row(t) for t in best10)  or '<tr><td colspan="10" style="text-align:center;color:#64748b;padding:16px">无</td></tr>'

# ── 汇总卡片数据 ──────────────────────────────────────────────────────
total_trades_all = sum(m["n_trades"] for m in all_metrics)
avg_win_rate = sum(m["win_rate_pct"] for m in all_metrics if m["n_trades"]>0) / max(sum(1 for m in all_metrics if m["n_trades"]>0),1)
total_adds   = sum(m["n_add_orders"] for m in all_metrics)
total_blowup = sum(m["n_blowup_events"] for m in all_metrics)
best_ret_m   = max(all_metrics, key=lambda m: m["total_return_pct"]) if all_metrics else {}
worst_ret_m  = min(all_metrics, key=lambda m: m["total_return_pct"]) if all_metrics else {}
worst_dd_m   = min(all_metrics, key=lambda m: m["max_drawdown_pct"]) if all_metrics else {}

# ── 生成 HTML ─────────────────────────────────────────────────────────
equity_json_str = _json.dumps(equity_json, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meme 反马丁趋势策略 — 全币种回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}}
  .header{{background:linear-gradient(135deg,#1e3a5f 0%,#0f172a 100%);padding:32px 40px;border-bottom:1px solid #1e293b}}
  .header h1{{font-size:26px;font-weight:700;color:#f1f5f9;letter-spacing:-0.5px}}
  .header p{{color:#64748b;margin-top:6px;font-size:14px}}
  .badge-row{{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}}
  .badge{{padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}}
  .badge-blue{{background:#1e40af;color:#93c5fd}}
  .badge-amber{{background:#78350f;color:#fde68a}}
  .badge-red{{background:#7f1d1d;color:#fca5a5}}
  .badge-purple{{background:#4c1d95;color:#c4b5fd}}
  .container{{max-width:1600px;margin:0 auto;padding:24px 20px}}
  .section-title{{font-size:16px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #1e293b}}
  /* 统计卡片 */
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}}
  .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 16px;position:relative;overflow:hidden}}
  .card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
  .card.blue::before{{background:linear-gradient(90deg,#3b82f6,#60a5fa)}}
  .card.green::before{{background:linear-gradient(90deg,#22c55e,#4ade80)}}
  .card.red::before{{background:linear-gradient(90deg,#ef4444,#f87171)}}
  .card.amber::before{{background:linear-gradient(90deg,#f59e0b,#fbbf24)}}
  .card.purple::before{{background:linear-gradient(90deg,#a855f7,#c084fc)}}
  .card-label{{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px}}
  .card-value{{font-size:28px;font-weight:700;color:#f1f5f9;line-height:1}}
  .card-sub{{font-size:12px;color:#475569;margin-top:6px}}
  /* 过滤器 */
  .filter-bar{{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap;align-items:center}}
  .filter-btn{{padding:6px 14px;border-radius:20px;border:1px solid #334155;background:#1e293b;color:#94a3b8;cursor:pointer;font-size:13px;transition:all .2s}}
  .filter-btn:hover,.filter-btn.active{{background:#334155;color:#e2e8f0;border-color:#475569}}
  .filter-btn.active{{background:#1d4ed8;border-color:#3b82f6;color:#eff6ff}}
  /* 表格 */
  .table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid #1e293b;margin-bottom:32px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#1e293b;color:#64748b;padding:10px 14px;text-align:left;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;white-space:nowrap}}
  td{{padding:9px 14px;border-bottom:1px solid #1e293b;vertical-align:middle}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#1e293b44}}
  .tf-badge{{padding:2px 8px;border-radius:8px;font-size:11px;font-weight:700}}
  .tf-1m{{background:#1d4ed8;color:#bfdbfe}}
  .tf-5m{{background:#065f46;color:#a7f3d0}}
  .tf-15m{{background:#7c2d12;color:#fed7aa}}
  /* 图表区域 */
  .chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px;margin-bottom:32px}}
  .chart-card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px}}
  .chart-title{{font-size:13px;font-weight:600;color:#94a3b8;margin-bottom:12px}}
  .chart-select{{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}}
  .cs-btn{{padding:3px 10px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#64748b;cursor:pointer;font-size:12px}}
  .cs-btn.active{{background:#1d4ed8;border-color:#3b82f6;color:#eff6ff}}
  canvas{{max-height:220px}}
  /* tabs */
  .tabs{{display:flex;gap:0;margin-bottom:0;border-bottom:1px solid #1e293b}}
  .tab{{padding:10px 20px;cursor:pointer;font-size:14px;color:#64748b;border-bottom:2px solid transparent;transition:all .2s}}
  .tab:hover{{color:#94a3b8}}
  .tab.active{{color:#3b82f6;border-bottom-color:#3b82f6}}
  .tab-content{{display:none;padding-top:20px}}
  .tab-content.active{{display:block}}
  /* 热力图 */
  .heatmap{{display:grid;gap:2px}}
  .hm-cell{{padding:8px;border-radius:6px;font-size:12px;text-align:center;font-weight:600;cursor:default;transition:all .2s}}
  .hm-cell:hover{{filter:brightness(1.2)}}
  /* 响应式 */
  @media(max-width:768px){{
    .header{{padding:20px}}
    .chart-grid{{grid-template-columns:1fr}}
    .cards{{grid-template-columns:repeat(2,1fr)}}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>⚡ Meme 反马丁趋势策略 — 全量回测报告</h1>
  <p>策略文件：Meme_反马丁_趋势.py &nbsp;|&nbsp; 数据源：Gate.io &nbsp;|&nbsp;
     回测区间：2026-06-24 ~ 2026-06-29 &nbsp;|&nbsp; 初始资金：1,000 USDT</p>
  <div class="badge-row">
    <span class="badge badge-blue">主流稳定：BTC ETH SOL XRP</span>
    <span class="badge badge-amber">中端中型：XCN</span>
    <span class="badge badge-red">高波动妖币：H VELVET BEAT COAI</span>
    <span class="badge badge-purple">周期：1m / 5m / 15m</span>
  </div>
</div>

<div class="container">

  <!-- 统计卡片 -->
  <div class="cards">
    <div class="card blue">
      <div class="card-label">总交易笔数</div>
      <div class="card-value">{total_trades_all}</div>
      <div class="card-sub">27组合计</div>
    </div>
    <div class="card {'green' if avg_win_rate >= 40 else 'red'}">
      <div class="card-label">平均胜率</div>
      <div class="card-value">{avg_win_rate:.1f}%</div>
      <div class="card-sub">所有有效组合</div>
    </div>
    <div class="card green">
      <div class="card-label">最佳回测组</div>
      <div class="card-value" style="font-size:18px;color:#4ade80">{best_ret_m.get('pair','').replace('_USDT','')} @{best_ret_m.get('timeframe','')}</div>
      <div class="card-sub">收益 <span style="color:#22c55e">+{best_ret_m.get('total_return_pct',0):.2f}%</span></div>
    </div>
    <div class="card red">
      <div class="card-label">最差回测组</div>
      <div class="card-value" style="font-size:18px;color:#f87171">{worst_ret_m.get('pair','').replace('_USDT','')} @{worst_ret_m.get('timeframe','')}</div>
      <div class="card-sub">收益 <span style="color:#ef4444">{worst_ret_m.get('total_return_pct',0):.2f}%</span></div>
    </div>
    <div class="card red">
      <div class="card-label">最大回撤</div>
      <div class="card-value" style="font-size:22px;color:#f87171">{worst_dd_m.get('max_drawdown_pct',0):.2f}%</div>
      <div class="card-sub">{worst_dd_m.get('pair','').replace('_USDT','')} @{worst_dd_m.get('timeframe','')}</div>
    </div>
    <div class="card purple">
      <div class="card-label">加仓触发次数</div>
      <div class="card-value" style="color:#c084fc">{total_adds}</div>
      <div class="card-sub">反马丁金字塔</div>
    </div>
    <div class="card {'red' if total_blowup > 0 else 'green'}">
      <div class="card-label">爆仓风险事件</div>
      <div class="card-value" style="color:{'#f87171' if total_blowup > 0 else '#4ade80'}">{total_blowup}</div>
      <div class="card-sub">浮亏 > 20% 初始资金</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" onclick="switchTab('metrics')">绩效汇总</div>
    <div class="tab" onclick="switchTab('equity')">盈亏曲线</div>
    <div class="tab" onclick="switchTab('heatmap')">收益热力图</div>
    <div class="tab" onclick="switchTab('trades')">典型交易</div>
    <div class="tab" onclick="switchTab('adds')">加仓记录</div>
    <div class="tab" onclick="switchTab('blowup')">爆仓风险</div>
  </div>

  <!-- Tab 1: 绩效汇总 -->
  <div id="tab-metrics" class="tab-content active">
    <div class="filter-bar">
      <span style="color:#64748b;font-size:13px">筛选：</span>
      <button class="filter-btn active" onclick="filterTable('all',this)">全部</button>
      <button class="filter-btn" onclick="filterTable('主流稳定',this)">主流稳定</button>
      <button class="filter-btn" onclick="filterTable('中端中型',this)">中端中型</button>
      <button class="filter-btn" onclick="filterTable('高波动妖币',this)">高波动妖币</button>
      <button class="filter-btn" onclick="filterTable('tf-1m',this)">1m</button>
      <button class="filter-btn" onclick="filterTable('tf-5m',this)">5m</button>
      <button class="filter-btn" onclick="filterTable('tf-15m',this)">15m</button>
    </div>
    <div class="table-wrap">
      <table id="metrics-table">
        <thead>
          <tr>
            <th>分组</th><th>币种</th><th>周期</th><th>交易数</th>
            <th>胜率</th><th>总收益%</th><th>最大回撤</th><th>单笔最大浮亏</th>
            <th>盈亏比</th><th>夏普</th><th>加仓次数</th><th>爆仓风险</th><th>退出原因分布</th>
          </tr>
        </thead>
        <tbody id="metrics-tbody">
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Tab 2: 盈亏曲线 -->
  <div id="tab-equity" class="tab-content">
    <div class="chart-grid" id="equity-grid"></div>
  </div>

  <!-- Tab 3: 收益热力图 -->
  <div id="tab-heatmap" class="tab-content">
    <p style="color:#64748b;font-size:13px;margin-bottom:16px">各币种×周期组合的总收益%，红=亏损，绿=盈利，深色=量级更大</p>
    <div id="heatmap-container" style="overflow-x:auto"></div>
  </div>

  <!-- Tab 4: 典型交易 -->
  <div id="tab-trades" class="tab-content">
    <h3 style="color:#ef4444;font-size:14px;margin-bottom:12px">最大亏损 Top 10</h3>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>分组</th><th>币种</th><th>周期</th><th>开仓时间</th><th>平仓时间</th>
          <th>加仓次数</th><th>盈亏(USDT)</th><th>盈亏%</th><th>最大浮亏</th><th>退出原因</th>
        </tr></thead>
        <tbody>{worst_rows}</tbody>
      </table>
    </div>
    <h3 style="color:#22c55e;font-size:14px;margin:24px 0 12px">最大盈利 Top 10</h3>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>分组</th><th>币种</th><th>周期</th><th>开仓时间</th><th>平仓时间</th>
          <th>加仓次数</th><th>盈亏(USDT)</th><th>盈亏%</th><th>最大浮亏</th><th>退出原因</th>
        </tr></thead>
        <tbody>{best_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Tab 5: 加仓记录 -->
  <div id="tab-adds" class="tab-content">
    <p style="color:#64748b;font-size:13px;margin-bottom:16px">
      反马丁金字塔加仓触发记录（浮盈达阈值且趋势确认后加仓）
    </p>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>时间</th><th>币种</th><th>周期</th><th>第N次</th>
          <th>加仓价</th><th>当时浮盈</th><th>触发阈值</th><th>加仓金额</th><th>累计投入</th>
        </tr></thead>
        <tbody>{add_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Tab 6: 爆仓风险 -->
  <div id="tab-blowup" class="tab-content">
    <p style="color:#64748b;font-size:13px;margin-bottom:16px">
      爆仓风险事件：持仓浮亏超过初始资金 20% 的时刻（总计 {total_blowup} 次）
    </p>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>时间</th><th>币种</th><th>周期</th><th>浮亏%</th>
          <th>当时价格</th><th>持仓成本</th><th>加仓数</th>
        </tr></thead>
        <tbody>{blowup_rows}</tbody>
      </table>
    </div>
  </div>

</div><!-- /container -->

<script>
const EQUITY_DATA = {equity_json_str};
const COINS_ORDER = {json.dumps(COINS_ORDER)};
const TFS = ["1m","5m","15m"];
const GROUP_COLOR = {json.dumps(GROUP_COLOR)};
const ALL_METRICS = {json.dumps(all_metrics)};

// ── Tab 切换 ──────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'equity' && !window._equityBuilt) {{ buildEquityCharts(); window._equityBuilt=true; }}
  if (name === 'heatmap' && !window._heatmapBuilt) {{ buildHeatmap(); window._heatmapBuilt=true; }}
}}

// ── 表格过滤 ──────────────────────────────────────────────────────────
function filterTable(filter, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#metrics-tbody .metric-row').forEach(row => {{
    if (filter === 'all') {{ row.style.display=''; return; }}
    const grp = row.dataset.group;
    const tf  = row.dataset.tf;
    if (filter.startsWith('tf-')) {{
      row.style.display = ('tf-'+tf === filter) ? '' : 'none';
    }} else {{
      row.style.display = (grp === filter) ? '' : 'none';
    }}
  }});
}}

// ── 盈亏曲线 ──────────────────────────────────────────────────────────
const chartRefs = {{}};
function buildEquityCharts() {{
  const grid = document.getElementById('equity-grid');
  grid.innerHTML = '';
  COINS_ORDER.forEach(pair => {{
    const shortName = pair.replace('_USDT','');
    const div = document.createElement('div');
    div.className = 'chart-card';
    div.innerHTML = `
      <div class="chart-title">&#x1F4C8; ${{shortName}} — 权益曲线</div>
      <div class="chart-select" id="cs-${{shortName}}">
        ${{TFS.map(tf => `<button class="cs-btn ${{tf==='5m'?'active':''}}" onclick="switchEqTF('${{shortName}}','${{tf}}',this)">${{tf}}</button>`).join('')}}
      </div>
      <canvas id="chart-${{shortName}}"></canvas>`;
    grid.appendChild(div);
    const ctx = document.getElementById('chart-'+shortName).getContext('2d');
    const d = EQUITY_DATA[pair]['5m'] || {{labels:[],values:[]}};
    chartRefs[shortName] = new Chart(ctx, buildChartConfig(d.labels, d.values, shortName));
  }});
}}

function buildChartConfig(labels, values, title) {{
  const color = values[values.length-1] >= values[0] ? '#22c55e' : '#ef4444';
  return {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [{{
        label: '权益(USDT)',
        data: values,
        borderColor: color,
        backgroundColor: color+'22',
        borderWidth: 1.5,
        fill: true,
        pointRadius: 0,
        tension: 0.3,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      plugins: {{ legend: {{ display:false }}, tooltip: {{ mode:'index', intersect:false }} }},
      scales: {{
        x: {{ ticks: {{ color:'#475569', maxTicksLimit:6, maxRotation:0 }}, grid: {{ color:'#1e293b' }} }},
        y: {{ ticks: {{ color:'#475569' }}, grid: {{ color:'#1e293b44' }} }}
      }}
    }}
  }};
}}

function switchEqTF(shortName, tf, btn) {{
  const pair = shortName+'_USDT';
  const d = EQUITY_DATA[pair]?.[tf] || {{labels:[],values:[]}};
  const chart = chartRefs[shortName];
  if (!chart) return;
  const color = d.values.length && d.values[d.values.length-1] >= d.values[0] ? '#22c55e' : '#ef4444';
  chart.data.labels = d.labels;
  chart.data.datasets[0].data = d.values;
  chart.data.datasets[0].borderColor = color;
  chart.data.datasets[0].backgroundColor = color+'22';
  chart.update();
  document.getElementById('cs-'+shortName).querySelectorAll('.cs-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}}

// ── 热力图 ─────────────────────────────────────────────────────────────
function buildHeatmap() {{
  const container = document.getElementById('heatmap-container');
  // 构建数据
  const metricMap = {{}};
  ALL_METRICS.forEach(m => {{ metricMap[m.pair+'-'+m.timeframe] = m; }});

  let html = '<table style="border-collapse:separate;border-spacing:4px;min-width:600px">';
  html += '<tr><th style="color:#64748b;padding:8px;text-align:left;font-size:12px">币种</th>';
  TFS.forEach(tf => {{
    html += `<th style="color:#94a3b8;padding:8px;text-align:center;font-size:13px;font-weight:700">${{tf}}</th>`;
  }});
  html += '</tr>';

  COINS_ORDER.forEach(pair => {{
    const shortName = pair.replace('_USDT','');
    html += `<tr><td style="color:#e2e8f0;padding:6px 12px;font-weight:600;white-space:nowrap">${{shortName}}</td>`;
    TFS.forEach(tf => {{
      const key = pair+'-'+tf;
      const m = metricMap[key];
      if (!m || m.n_trades === 0) {{
        html += '<td style="background:#1e293b;border-radius:8px;padding:12px;text-align:center;color:#334155;font-size:12px">—</td>';
        return;
      }}
      const ret = m.total_return_pct;
      const wr  = m.win_rate_pct;
      let bg, fc;
      if (ret > 2)       {{ bg='#14532d'; fc='#4ade80'; }}
      else if (ret > 0.5) {{ bg='#166534'; fc='#86efac'; }}
      else if (ret > 0)  {{ bg='#15803d'; fc='#bbf7d0'; }}
      else if (ret > -0.5){{ bg='#7f1d1d'; fc='#fca5a5'; }}
      else if (ret > -1.5){{ bg='#991b1b'; fc='#f87171'; }}
      else                {{ bg='#7f1d1d'; fc='#ef4444'; }}
      html += `<td style="background:${{bg}};border-radius:8px;padding:10px;text-align:center;cursor:default" title="${{pair}} @${{tf}}: ${{ret}}% | 胜率${{wr}}%">
        <div style="color:${{fc}};font-weight:700;font-size:14px">${{ret > 0 ? '+' : ''}}${{ret}}%</div>
        <div style="color:${{fc}}aa;font-size:11px">${{m.n_trades}}笔 ${{wr}}%胜</div>
      </td>`;
    }});
    html += '</tr>';
  }});
  html += '</table>';
  container.innerHTML = html;
}}
</script>
</body>
</html>
"""

Path(OUT_PATH).write_text(html, encoding="utf-8")
print(f"[OK] Report saved: {OUT_PATH}")
