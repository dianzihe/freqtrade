# -*- coding: utf-8 -*-
"""
微结构信号模块 L2 数据回测脚本
================================
数据来源: user_data/orderbook_data/gate/spot/BTC_USDT/l2 (parquet增量更新流)
逻辑:
  1. 加载所有 parquet 文件 → 重建订单簿快照
  2. 按 1 分钟聚合 → LOB 特征 K 线
  3. 调用 add_lob_regime_signals() 生成 ch1-4 + signal_trigger
  4. 提取信号触发点，计算信号后 15 分钟行情
  5. 输出 HTML 报告
"""

import ast
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径配置 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
L2_DATA_DIR = ROOT / "user_data/orderbook_data/gate/spot/BTC_USDT/l2"
STRATEGIES_DIR = ROOT / "user_data/strategies"
OUTPUT_HTML = ROOT / "microstructure_backtest_report.html"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STRATEGIES_DIR))


# ════════════════════════════════════════════════════════════════════════════
#  Step 1: 加载原始 L2 增量数据
# ════════════════════════════════════════════════════════════════════════════

def load_all_parquet(data_dir: Path) -> pd.DataFrame:
    """递归加载目录下所有 parquet 文件并合并。"""
    files = sorted(glob.glob(str(data_dir / "**/*.parquet"), recursive=True))
    print(f"[INFO] 找到 {len(files)} 个 parquet 文件，正在加载...")
    parts = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            parts.append(df)
        except Exception as e:
            print(f"[WARN] 跳过文件 {fp}: {e}")
    if not parts:
        raise RuntimeError("没有找到有效的 parquet 文件！")
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.sort_values("exchange_time_ms").reset_index(drop=True)
    print(f"[INFO] 原始 L2 记录共 {len(raw):,} 行")
    return raw


# ════════════════════════════════════════════════════════════════════════════
#  Step 2: 解析 LOB 并重建每行的订单簿快照
#          → 逐行维护 bid/ask 字典，生成"快照流"
# ════════════════════════════════════════════════════════════════════════════

def parse_levels(value) -> list:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    if isinstance(value, list):
        return value
    return []


def rebuild_lob_stream(raw: pd.DataFrame, max_levels: int = 25) -> pd.DataFrame:
    """
    从增量更新流重建订单簿快照，每条记录生成一个快照。
    返回包含以下列的 DataFrame:
        timestamp_ms, best_bid, best_ask, spread_ratio,
        bid_depth_25, ask_depth_25
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}

    records = []
    for _, row in raw.iterrows():
        ts = int(row["exchange_time_ms"])

        # 应用增量更新（quantity=0 表示删除该档位）
        for level in parse_levels(row["bid_updates"]):
            try:
                p, q = float(level[0]), float(level[1])
                if q <= 0:
                    bids.pop(p, None)
                else:
                    bids[p] = q
            except Exception:
                continue

        for level in parse_levels(row["ask_updates"]):
            try:
                p, q = float(level[0]), float(level[1])
                if q <= 0:
                    asks.pop(p, None)
                else:
                    asks[p] = q
            except Exception:
                continue

        if not bids or not asks:
            continue

        # 排序取 top N
        sorted_bids = sorted(bids.items(), reverse=True)[:max_levels]
        sorted_asks = sorted(asks.items())[:max_levels]

        best_bid = sorted_bids[0][0]
        best_ask = sorted_asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        spread_ratio = (best_ask - best_bid) / mid if mid > 0 else 0.0

        bid_depth = sum(p * q for p, q in sorted_bids)
        ask_depth = sum(p * q for p, q in sorted_asks)

        records.append({
            "timestamp_ms": ts,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_ratio": spread_ratio,
            "bid_depth_25": bid_depth,
            "ask_depth_25": ask_depth,
        })

    snap = pd.DataFrame(records)
    print(f"[INFO] 重建快照 {len(snap):,} 条（去除空盘面后）")
    return snap


# ════════════════════════════════════════════════════════════════════════════
#  Step 3: 按 1 分钟聚合 → OHLCV + LOB 特征
# ════════════════════════════════════════════════════════════════════════════

def aggregate_to_1m(snap: pd.DataFrame) -> pd.DataFrame:
    """
    将快照流聚合为 1 分钟 K 线，包含 OHLCV 代理 + LOB 衍生特征。
    mid_price 作为 close（orderbook 中间价，比成交价更稳定）。
    """
    snap["datetime"] = pd.to_datetime(snap["timestamp_ms"], unit="ms", utc=True)
    snap = snap.set_index("datetime")

    price = snap["mid_price"]
    spread_ratio = snap["spread_ratio"]
    bid_depth = snap["bid_depth_25"]
    ask_depth = snap["ask_depth_25"]

    df_1m = pd.DataFrame({
        "open":  price.resample("1min").first(),
        "high":  price.resample("1min").max(),
        "low":   price.resample("1min").min(),
        "close": price.resample("1min").last(),
        "volume": (bid_depth + ask_depth).resample("1min").mean(),  # 近似流动性代理

        # LOB 特征列（信号模块直接使用）
        "best_bid":      snap["best_bid"].resample("1min").last(),
        "best_ask":      snap["best_ask"].resample("1min").last(),
        "spread_ratio":  spread_ratio.resample("1min").mean(),
        "bid_depth_25":  bid_depth.resample("1min").mean(),
        "ask_depth_25":  ask_depth.resample("1min").mean(),

        # 深度不平衡（正 = 买方占优，负 = 卖方占优）
        "depth_imbalance": (bid_depth - ask_depth).resample("1min").mean(),
    }).dropna(subset=["close"])

    # 前向填充 OHLC 防止空洞
    for col in ["open", "high", "low", "close"]:
        df_1m[col] = df_1m[col].ffill()

    df_1m = df_1m.reset_index()
    df_1m = df_1m.rename(columns={"datetime": "date"})
    print(f"[INFO] 1分钟K线 {len(df_1m)} 根，时间范围: {df_1m['date'].iloc[0]} → {df_1m['date'].iloc[-1]}")
    return df_1m


# ════════════════════════════════════════════════════════════════════════════
#  Step 4: 运行微结构信号模块
# ════════════════════════════════════════════════════════════════════════════

def run_signal_module(df_1m: pd.DataFrame) -> pd.DataFrame:
    from 微结构_信号模块 import add_lob_regime_signals

    # 信号模块通过 bid_depth_10/ask_depth_10 计算深度
    df_1m["bid_depth_10"] = df_1m["bid_depth_25"]
    df_1m["ask_depth_10"] = df_1m["ask_depth_25"]

    df_signals = add_lob_regime_signals(
        df_1m,
        lookback_period=24,
        volatility_window=20,
        threshold_percentile=88,
        confirmation_bars=2,
        min_signal_strength=0.35,
        signal_valid_bars=20,
    )
    triggers = df_signals["signal_trigger"].sum()
    print(f"[INFO] 信号触发次数: {triggers}")
    return df_signals


# ════════════════════════════════════════════════════════════════════════════
#  Step 5: 提取信号 + 计算后15分钟行情
# ════════════════════════════════════════════════════════════════════════════

def extract_signal_outcomes(df: pd.DataFrame, forward_bars: int = 15) -> pd.DataFrame:
    """
    提取每个 signal_trigger=1 的点，计算其后 forward_bars 根 K 线的行情。
    """
    signal_rows = df[df["signal_trigger"] == 1].copy()
    if len(signal_rows) == 0:
        print("[WARN] 没有触发任何信号！")
        return pd.DataFrame()

    results = []
    for idx in signal_rows.index:
        row = df.loc[idx]
        signal_price = row["close"]

        # 未来 15 根 1 分钟 K 线
        future_slice = df.loc[idx + 1 : idx + forward_bars]

        if len(future_slice) == 0:
            future_high = np.nan
            future_low = np.nan
            future_close = np.nan
            max_gain_pct = np.nan
            max_drop_pct = np.nan
            close_change_pct = np.nan
            direction = "N/A"
        else:
            future_high = future_slice["close"].max()
            future_low = future_slice["close"].min()
            future_close = future_slice["close"].iloc[-1]
            max_gain_pct = (future_high - signal_price) / signal_price * 100
            max_drop_pct = (future_low - signal_price) / signal_price * 100
            close_change_pct = (future_close - signal_price) / signal_price * 100

            # 方向判断：15分钟后收盘涨跌 + 极值对比
            if abs(close_change_pct) < 0.05:
                direction = "横盘"
            elif close_change_pct > 0:
                direction = "⬆ 上涨"
            else:
                direction = "⬇ 下跌"

        results.append({
            "信号时间(UTC)":       row["date"].strftime("%Y-%m-%d %H:%M"),
            "信号时价格":           round(signal_price, 2),
            "ch1_波动率熵":         round(float(row["ch1_vol_entropy"]), 4),
            "ch2_深度侵蚀":         round(float(row["ch2_depth_erosion"]), 4),
            "ch3_价差漂移":         round(float(row["ch3_spread_drift"]), 4),
            "ch4_订单流":           round(float(row["ch4_order_flow"]), 4),
            "综合信号强度":          round(float(row["composite_smooth"]), 4),
            "自适应阈值":            round(float(row.get("adaptive_threshold", 0)), 4),
            "15min后最高价":         round(float(future_high), 2) if not np.isnan(future_high) else "N/A",
            "15min后最低价":         round(float(future_low), 2) if not np.isnan(future_low) else "N/A",
            "15min后收盘价":         round(float(future_close), 2) if not np.isnan(future_close) else "N/A",
            "最大涨幅%":            round(float(max_gain_pct), 3) if not np.isnan(max_gain_pct) else "N/A",
            "最大跌幅%":            round(float(max_drop_pct), 3) if not np.isnan(max_drop_pct) else "N/A",
            "15min收盘变化%":        round(float(close_change_pct), 3) if not np.isnan(close_change_pct) else "N/A",
            "行情方向":              direction,
        })

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════════
#  Step 6: 生成 HTML 报告
# ════════════════════════════════════════════════════════════════════════════

def build_html_report(df_signals: pd.DataFrame, outcomes: pd.DataFrame, output_path: Path):
    """生成完整的 HTML 报告，包含统计摘要、通道时序图、信号明细表。"""

    # ---- 统计摘要 ----
    total_signals = len(outcomes) if len(outcomes) > 0 else 0
    if total_signals > 0 and "15min收盘变化%" in outcomes.columns:
        valid = outcomes[outcomes["15min收盘变化%"] != "N/A"]
        up_count = len(valid[valid["行情方向"] == "⬆ 上涨"])
        down_count = len(valid[valid["行情方向"] == "⬇ 下跌"])
        flat_count = len(valid[valid["行情方向"] == "横盘"])
        win_rate = round(up_count / len(valid) * 100, 1) if len(valid) > 0 else 0
        avg_change = round(pd.to_numeric(valid["15min收盘变化%"]).mean(), 3)
    else:
        up_count = down_count = flat_count = win_rate = avg_change = 0

    # ---- 时间轴 JS 数据 ----
    # 采样最近 500 根 K 线避免过大
    plot_df = df_signals.tail(min(len(df_signals), 500)).copy()
    plot_df["date_str"] = plot_df["date"].dt.strftime("%m-%d %H:%M")

    def js_arr(series):
        return "[" + ",".join(str(round(float(v), 4)) if not pd.isna(v) else "null" for v in series) + "]"

    labels = "[" + ",".join(f'"{s}"' for s in plot_df["date_str"]) + "]"
    ch1_data = js_arr(plot_df["ch1_vol_entropy"])
    ch2_data = js_arr(plot_df["ch2_depth_erosion"])
    ch3_data = js_arr(plot_df["ch3_spread_drift"])
    ch4_data = js_arr(plot_df["ch4_order_flow"].abs())
    comp_data = js_arr(plot_df["composite_smooth"])
    thresh_data = js_arr(plot_df["adaptive_threshold"])
    price_data = js_arr(plot_df["close"])

    # 信号点标注
    signal_points = []
    sig_df = plot_df[plot_df["signal_trigger"] == 1]
    local_indices = [list(plot_df.index).index(i) for i in sig_df.index if i in list(plot_df.index)]
    for li in local_indices:
        signal_points.append({"x": li, "y": float(round(plot_df.iloc[li]["composite_smooth"], 4))})
    signal_js = str(signal_points).replace("'", '"')

    price_signal_points = []
    for li in local_indices:
        price_signal_points.append({"x": li, "y": float(round(plot_df.iloc[li]["close"], 2))})
    price_signal_js = str(price_signal_points).replace("'", '"')

    # ---- 信号明细表 HTML ----
    if len(outcomes) > 0:
        rows_html = ""
        for _, r in outcomes.iterrows():
            direction = r["行情方向"]
            if "上涨" in str(direction):
                dir_class = "up"
                dir_bg = "#1a3a1a"
            elif "下跌" in str(direction):
                dir_class = "down"
                dir_bg = "#3a1a1a"
            else:
                dir_class = "flat"
                dir_bg = "#2a2a1a"

            chg = r["15min收盘变化%"]
            if chg != "N/A":
                chg_color = "#26a69a" if float(chg) >= 0 else "#ef5350"
                chg_str = f'<span style="color:{chg_color};font-weight:600">{chg:+.3f}%</span>'
            else:
                chg_str = '<span style="color:#888">N/A</span>'

            mg = r["最大涨幅%"]
            md = r["最大跌幅%"]
            mg_str = f'<span style="color:#26a69a">+{mg:.3f}%</span>' if mg != "N/A" else "N/A"
            md_str = f'<span style="color:#ef5350">{md:.3f}%</span>' if md != "N/A" else "N/A"

            # ch4 可以是负数
            ch4_val = r["ch4_订单流"]
            ch4_color = "#ef5350" if float(ch4_val) < 0 else "#888"

            rows_html += f"""
            <tr style="background:{dir_bg}">
                <td>{r["信号时间(UTC)"]}</td>
                <td>{r["信号时价格"]:,}</td>
                <td><div class="bar-cell"><div class="bar" style="width:{min(r['ch1_波动率熵']*100,100):.1f}%;background:#7b68ee"></div><span>{r['ch1_波动率熵']:.4f}</span></div></td>
                <td><div class="bar-cell"><div class="bar" style="width:{min(r['ch2_深度侵蚀']*20,100):.1f}%;background:#ffa726"></div><span>{r['ch2_深度侵蚀']:.4f}</span></div></td>
                <td><div class="bar-cell"><div class="bar" style="width:{min(r['ch3_价差漂移']*20,100):.1f}%;background:#42a5f5"></div><span>{r['ch3_价差漂移']:.4f}</span></div></td>
                <td style="color:{ch4_color}">{r['ch4_订单流']:.4f}</td>
                <td>{r['综合信号强度']:.4f}</td>
                <td>{r['自适应阈值']:.4f}</td>
                <td>{r['15min后收盘价']}</td>
                <td>{mg_str}</td>
                <td>{md_str}</td>
                <td>{chg_str}</td>
                <td><span class="{dir_class}-tag">{direction}</span></td>
            </tr>"""
    else:
        rows_html = '<tr><td colspan="13" style="text-align:center;color:#888;padding:40px">没有检测到信号触发</td></tr>'

    # ---- 完整 HTML ────────────────────────────────────────────────────────
    data_range = f"{df_signals['date'].iloc[0].strftime('%Y-%m-%d %H:%M')} → {df_signals['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC/USDT 微结构信号回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: "Segoe UI", "PingFang SC", sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; color: #e6edf3; }}
  .subtitle {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }}
  .section-title {{ font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin: 28px 0 14px; padding-left: 10px; border-left: 3px solid #58a6ff; }}

  /* 统计卡片 */
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px 22px; flex: 1; min-width: 140px; }}
  .card .label {{ font-size: 0.75rem; color: #8b949e; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; color: #58a6ff; }}
  .card .value.up {{ color: #26a69a; }}
  .card .value.down {{ color: #ef5350; }}
  .card .value.neutral {{ color: #ffa726; }}

  /* 图表容器 */
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  .chart-box canvas {{ max-height: 220px; }}

  /* 表格 */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  thead th {{ background: #21262d; color: #8b949e; font-weight: 600; padding: 10px 12px; text-align: left; white-space: nowrap; border-bottom: 2px solid #30363d; }}
  tbody td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; white-space: nowrap; }}
  tbody tr:hover {{ background: #1c2128; }}

  .bar-cell {{ display: flex; align-items: center; gap: 8px; min-width: 120px; }}
  .bar {{ height: 6px; border-radius: 3px; min-width: 2px; }}

  .up-tag {{ background: #1a3a1a; color: #26a69a; border: 1px solid #26a69a; border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; }}
  .down-tag {{ background: #3a1a1a; color: #ef5350; border: 1px solid #ef5350; border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; }}
  .flat-tag {{ background: #2a2a1a; color: #ffa726; border: 1px solid #ffa726; border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; }}

  /* 信号通道说明 */
  .legend-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0 16px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #8b949e; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
</style>
</head>
<body>

<h1>📊 BTC/USDT 微结构信号回测报告</h1>
<div class="subtitle">数据来源: Gate.io L2 订单簿 | 时间窗口: {data_range} | 1分钟K线聚合 | 信号模块: arXiv:2604.20949</div>

<!-- 统计卡片 -->
<div class="section-title">📈 统计摘要</div>
<div class="cards">
  <div class="card"><div class="label">数据K线总数</div><div class="value neutral">{len(df_signals)}</div></div>
  <div class="card"><div class="label">信号触发次数</div><div class="value">{total_signals}</div></div>
  <div class="card"><div class="label">上涨 / 下跌 / 横盘</div><div class="value" style="font-size:1.2rem">{up_count} / {down_count} / {flat_count}</div></div>
  <div class="card"><div class="label">15min后收盘上涨率</div><div class="value {'up' if win_rate >= 50 else 'down'}">{win_rate}%</div></div>
  <div class="card"><div class="label">15min平均收盘变化</div><div class="value {'up' if avg_change >= 0 else 'down'}">{avg_change:+.3f}%</div></div>
</div>

<!-- 通道信号图 -->
<div class="section-title">🔬 四通道信号强度（最近500根K线）</div>
<div class="legend-row">
  <div class="legend-item"><div class="legend-dot" style="background:#7b68ee"></div>ch1 波动率熵</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ffa726"></div>ch2 深度侵蚀</div>
  <div class="legend-item"><div class="legend-dot" style="background:#42a5f5"></div>ch3 价差漂移</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef5350"></div>ch4 |订单流|</div>
  <div class="legend-item"><div class="legend-dot" style="background:#26a69a"></div>综合信号 (MAX)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ffd700"></div>自适应阈值</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff6b6b;border:2px solid white"></div>信号触发点</div>
</div>
<div class="chart-box">
  <canvas id="channelChart"></canvas>
</div>

<!-- 价格图 -->
<div class="section-title">💰 BTC/USDT 中间价走势（最近500根K线）</div>
<div class="chart-box">
  <canvas id="priceChart"></canvas>
</div>

<!-- 信号明细表 -->
<div class="section-title">📋 信号触发明细（共 {total_signals} 次）</div>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>信号时间(UTC)</th>
      <th>信号价格($)</th>
      <th>ch1 波动率熵</th>
      <th>ch2 深度侵蚀</th>
      <th>ch3 价差漂移</th>
      <th>ch4 订单流</th>
      <th>综合强度</th>
      <th>自适应阈值</th>
      <th>15min后收盘</th>
      <th>最大涨幅</th>
      <th>最大跌幅</th>
      <th>15min变化</th>
      <th>行情方向</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>

<!-- 说明 -->
<div style="margin-top: 32px; padding: 16px; background: #161b22; border: 1px solid #30363d; border-radius: 10px; font-size: 0.8rem; color: #8b949e; line-height: 1.7;">
  <strong style="color:#e6edf3">📌 通道含义说明</strong><br>
  <b style="color:#7b68ee">ch1 波动率熵</b>：三尺度波动率分歧熵，高值=多时间尺度波动正在分化，市场内部结构变化中。<br>
  <b style="color:#ffa726">ch2 深度侵蚀</b>：订单簿前25档总深度的 z-score 标准化（负向），高值=流动性快速萎缩，做市商撤单。<br>
  <b style="color:#42a5f5">ch3 价差漂移</b>：相对价差的 z-score，高值=bid-ask 价差相对历史均值扩大，流动性恶化信号。<br>
  <b style="color:#ef5350">ch4 订单流</b>：净买卖量的 z-score，负值=净卖压大，正值=净买压大。取绝对值参与 MAX 聚合。<br>
  <b style="color:#26a69a">综合信号</b>：四通道 MAX 聚合后的3根K线平滑值。超过自适应阈值且连续上升时触发 signal_trigger=1。<br>
  <br>
  <b>⚠️ 声明：</b> 本报告仅作为微观结构分析参考，不构成交易建议。信号发出后行情方向由多空力量博弈决定，需结合趋势判断使用。
</div>

<script>
// ── 四通道图 ────────────────────────────────────────────────────────────────
const labels = {labels};
const ch1 = {ch1_data};
const ch2 = {ch2_data};
const ch3 = {ch3_data};
const ch4 = {ch4_data};
const comp = {comp_data};
const thresh = {thresh_data};
const signalPoints = {signal_js};

const channelCtx = document.getElementById("channelChart").getContext("2d");
new Chart(channelCtx, {{
  type: "line",
  data: {{
    labels,
    datasets: [
      {{ label: "ch1 波动率熵",  data: ch1,   borderColor: "#7b68ee", borderWidth: 1, pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: "ch2 深度侵蚀",  data: ch2,   borderColor: "#ffa726", borderWidth: 1, pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: "ch3 价差漂移",  data: ch3,   borderColor: "#42a5f5", borderWidth: 1, pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: "ch4 |订单流|",  data: ch4,   borderColor: "#ef5350", borderWidth: 1, pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: "综合信号 (MAX)", data: comp,  borderColor: "#26a69a", borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false }},
      {{ label: "自适应阈值",     data: thresh, borderColor: "#ffd700", borderWidth: 1.5, borderDash: [4,4], pointRadius: 0, tension: 0, fill: false }},
      {{ label: "信号触发点", data: signalPoints, type: "scatter",
         pointBackgroundColor: "#ff6b6b", pointBorderColor: "#fff", pointRadius: 8, pointBorderWidth: 2,
         showLine: false }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: true,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: "#8b949e", font: {{ size: 11 }} }} }},
      tooltip: {{ backgroundColor: "#21262d", titleColor: "#e6edf3", bodyColor: "#c9d1d9", borderColor: "#30363d", borderWidth: 1 }}
    }},
    scales: {{
      x: {{ ticks: {{ color: "#8b949e", maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ color: "#21262d" }} }},
      y: {{ ticks: {{ color: "#8b949e", font: {{ size: 10 }} }}, grid: {{ color: "#21262d" }} }}
    }}
  }}
}});

// ── 价格图 ──────────────────────────────────────────────────────────────────
const priceData = {price_data};
const priceSignalPoints = {price_signal_js};

const priceCtx = document.getElementById("priceChart").getContext("2d");
new Chart(priceCtx, {{
  type: "line",
  data: {{
    labels,
    datasets: [
      {{ label: "BTC 中间价", data: priceData, borderColor: "#58a6ff", borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: {{ target: "origin", above: "rgba(88,166,255,0.04)" }} }},
      {{ label: "信号触发",   data: priceSignalPoints, type: "scatter",
         pointBackgroundColor: "#ff6b6b", pointBorderColor: "#fff", pointRadius: 8, pointBorderWidth: 2, showLine: false }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: true,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: "#8b949e", font: {{ size: 11 }} }} }},
      tooltip: {{ backgroundColor: "#21262d", titleColor: "#e6edf3", bodyColor: "#c9d1d9", borderColor: "#30363d", borderWidth: 1,
        callbacks: {{ label: (ctx) => ctx.dataset.label + ": $" + (ctx.raw?.y ?? ctx.raw).toLocaleString() }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: "#8b949e", maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ color: "#21262d" }} }},
      y: {{ ticks: {{ color: "#8b949e", font: {{ size: 10 }}, callback: (v) => "$" + v.toLocaleString() }}, grid: {{ color: "#21262d" }} }}
    }}
  }}
}});
</script>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[INFO] HTML 报告已生成: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BTC/USDT 微结构信号模块 L2 回测")
    print("=" * 60)

    # 1. 加载原始数据
    raw = load_all_parquet(L2_DATA_DIR)

    # 2. 重建订单簿快照流
    print("[INFO] 正在重建订单簿快照（较慢，约1-3分钟）...")
    snap = rebuild_lob_stream(raw)

    # 3. 聚合为 1 分钟 K 线
    df_1m = aggregate_to_1m(snap)

    # 4. 运行信号模块
    df_signals = run_signal_module(df_1m)

    # 5. 提取信号结果
    outcomes = extract_signal_outcomes(df_signals, forward_bars=15)
    if len(outcomes) > 0:
        print("\n[RESULT] 信号触发汇总：")
        print(outcomes[["信号时间(UTC)", "信号时价格", "ch1_波动率熵", "ch2_深度侵蚀",
                         "ch3_价差漂移", "ch4_订单流", "综合信号强度", "行情方向", "15min收盘变化%"]].to_string(index=False))

    # 6. 生成报告
    build_html_report(df_signals, outcomes, OUTPUT_HTML)
    print(f"\n[DONE] 报告路径: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
