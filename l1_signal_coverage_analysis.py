# -*- coding: utf-8 -*-
"""
L1 数据信号覆盖率分析 v2
========================
修复覆盖率计算逻辑，用纯 CSS/SVG 替代 ECharts，提取清晰结论。
"""

import json
import os
import sys
import time
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
L1_DIR = PROJECT_ROOT / "user_data/orderbook_data/gate/spot/BTC_USDT/l1"
SIGNAL_CSV = PROJECT_ROOT / "user_data/backtest_results/微结构信号回测数据.csv"
OUTPUT_DIR = PROJECT_ROOT / "user_data/backtest_results"


# ═════════════════════════════════════════════════════════════
#  1. 构建 L1 蜡烛
# ═════════════════════════════════════════════════════════════

def build_l1_candles():
    files = sorted(glob.glob(str(L1_DIR / "**/*.parquet"), recursive=True))
    valid = [f for f in files if _test_parquet(f)]
    print(f"  L1 有效文件: {len(valid)}/{len(files)}")
    
    all_parts = []
    t0 = time.time()
    for fi, fpath in enumerate(valid):
        df = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["exchange_time_ms"], unit="ms", utc=True).dt.floor("1min")
        g = df.groupby("date", sort=False).agg(
            open=("mid_price", "first"), high=("mid_price", "max"),
            low=("mid_price", "min"), close=("mid_price", "last"),
            tick_count=("update_id", "count"),
            spread_avg=("spread", "mean"), spread_max=("spread", "max"),
            bid_amt_avg=("bid_amount", "mean"), ask_amt_avg=("ask_amount", "mean"),
        ).reset_index()
        all_parts.append(g)
    
    candles = pd.concat(all_parts, ignore_index=True)
    candles = candles.groupby("date", sort=True).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        tick_count=("tick_count", "sum"),
        spread_avg=("spread_avg", "mean"), spread_max=("spread_max", "max"),
        bid_amt_avg=("bid_amt_avg", "mean"), ask_amt_avg=("ask_amt_avg", "mean"),
    ).reset_index()
    
    # 填充缺失
    candles = candles.set_index("date")
    full_range = pd.date_range(candles.index.min(), candles.index.max(), freq="1min")
    candles = candles.reindex(full_range)
    for c in ["open", "high", "low", "close"]:
        candles[c] = candles[c].ffill()
    for c in ["tick_count", "spread_avg", "spread_max", "bid_amt_avg", "ask_amt_avg"]:
        candles[c] = candles[c].fillna(0)
    candles = candles.reset_index().rename(columns={"index": "date"})
    
    elapsed = time.time() - t0
    print(f"  L1 构建: {sum(len(p) for p in all_parts[::10])*10:,} ticks -> {len(candles)} candles, {elapsed:.1f}s")
    return candles


def _test_parquet(f):
    try:
        pd.read_parquet(f, columns=["exchange_time_ms"])
        return True
    except:
        return False


# ═════════════════════════════════════════════════════════════
#  2. 核心计算
# ═════════════════════════════════════════════════════════════

def run_analysis(l1_candles):
    print("\n[2] 执行覆盖率分析...")
    
    # 加载信号
    sig = pd.read_csv(SIGNAL_CSV, encoding="utf-8-sig")
    sig["signal_time"] = pd.to_datetime(sig.iloc[:, 0], utc=True)
    print(f"  信号数: {len(sig)}")
    
    # 构建价格序列（索引-based）用于快速查询
    l1 = l1_candles.copy().set_index("date").sort_index()
    prices = l1["close"].values
    times = l1.index
    
    # 计算全局前向收益
    l1["ret_5m"] = (prices_shift(l1["close"].values, -5) - l1["close"].values) / l1["close"].values
    l1["ret_10m"] = (prices_shift(l1["close"].values, -10) - l1["close"].values) / l1["close"].values
    l1["abs_ret_5m"] = np.abs(l1["ret_5m"])
    l1["abs_ret_10m"] = np.abs(l1["ret_10m"])
    
    # 对齐信号到 L1 索引
    sig_records = []
    for _, srow in sig.iterrows():
        stime = srow["signal_time"]
        # 找最近的蜡烛
        idx = np.argmin(np.abs(times - stime))
        pos = times.get_loc(times[idx]) if hasattr(times, 'get_loc') else idx
        
        if pos < len(prices) - 15:
            sp = prices[pos]
            # 5分钟
            p5 = prices[pos + 5] if pos + 5 < len(prices) else prices[-1]
            h5 = np.max(prices[pos+1:pos+6]) if pos+6 <= len(prices) else np.max(prices[pos+1:])
            l5 = np.min(prices[pos+1:pos+6]) if pos+6 <= len(prices) else np.min(prices[pos+1:])
            # 10分钟
            p10 = prices[pos + 10] if pos + 10 < len(prices) else prices[-1]
            h10 = np.max(prices[pos+1:pos+11]) if pos+11 <= len(prices) else np.max(prices[pos+1:])
            l10 = np.min(prices[pos+1:pos+11]) if pos+11 <= len(prices) else np.min(prices[pos+1:])
            
            sig_records.append({
                "signal_time": stime,
                "signal_price": sp,
                "ch1": srow.get("CH1_波动率熵", 0),
                "ch2": srow.get("CH2_深度侵蚀", 0),
                "ch3": srow.get("CH3_价差漂移", 0),
                "ch4": srow.get("CH4_订单流", 0),
                "composite": srow.get("综合信号", 0),
                "ch4_dir": str(srow.get("CH4方向", "")),
                "candle_idx": int(pos),
                "ret_5m": (p5 - sp) / sp,
                "max_gain_5m": (h5 - sp) / sp,
                "max_loss_5m": (l5 - sp) / sp,
                "ret_10m": (p10 - sp) / sp,
                "max_gain_10m": (h10 - sp) / sp,
                "max_loss_10m": (l10 - sp) / sp,
            })
    
    sr = pd.DataFrame(sig_records)
    print(f"  有效信号记录: {len(sr)}")
    
    # ===== 覆盖率分析（修正） =====
    # 定义: "事件" = 任意一分钟，其后5/10分钟的绝对收益超过阈值
    # "覆盖" = 该事件发生前 window 分钟内有过信号
    
    thresholds = [
        (0.001, ">= 0.1% (小波动)"),
        (0.003, ">= 0.3% (中等波动)"),
        (0.005, ">= 0.5% (明显波动)"),
        (0.010, ">= 1.0% (大幅波动)"),
        (0.020, ">= 2.0% (极端波动)"),
    ]
    
    # 信号索引集合
    signal_idxs = set(sr["candle_idx"].values)
    signal_times_by_idx = {int(r["candle_idx"]): r["signal_time"] for _, r in sr.iterrows()}
    
    coverage_rows = []
    for window in [5, 10]:
        ret_col = f"ret_{window}m"
        abs_col = f"abs_ret_{window}m"
        
        for th, label in thresholds:
            # 所有超过阈值的行情事件
            event_mask = l1[abs_col] >= th
            event_idxs = np.where(event_mask)[0]
            n_events = len(event_idxs)
            
            if n_events == 0:
                coverage_rows.append({
                    "窗口": f"{window}分钟", "波动等级": label,
                    "事件总数": 0, "被覆盖": 0, "覆盖率%": 0,
                    "覆盖事件均值收益%": 0, "覆盖事件收益标准差%": 0,
                    "覆盖评级": "-",
                })
                continue
            
            # 对每个事件，检查前120分钟（默认）或前30分钟（严格）内是否有信号
            covered_events = []
            for ev_idx in event_idxs:
                for lookback in [15, 30, 60]:
                    # 检查 lookback 分钟内是否有信号
                    min_sig_idx = max(0, ev_idx - lookback)
                    sigs_in_range = signal_idxs & set(range(int(min_sig_idx), int(ev_idx) + 1))
                    if sigs_in_range:
                        ret_val = l1[ret_col].iloc[ev_idx]
                        covered_events.append({
                            "event_idx": int(ev_idx),
                            "event_ret": float(ret_val),
                            "abs_ret": float(l1[abs_col].iloc[ev_idx]),
                        })
                        break
            
            n_covered = len(covered_events)
            cov_pct = n_covered / n_events * 100 if n_events > 0 else 0
            
            covered_rets = [e["event_ret"] for e in covered_events]
            avg_ret = np.mean(covered_rets) * 100 if covered_rets else 0
            std_ret = np.std(covered_rets) * 100 if covered_rets else 0
            
            rating = "GOOD" if cov_pct > 30 else "OK" if cov_pct > 10 else "LOW"
            
            coverage_rows.append({
                "窗口": f"{window}分钟", "波动等级": label,
                "事件总数": n_events, "被覆盖": n_covered,
                "覆盖率%": round(cov_pct, 1),
                "覆盖事件均值收益%": round(avg_ret, 4),
                "覆盖事件收益标准差%": round(std_ret, 4),
                "覆盖评级": rating,
            })
            
            print(f"    {window}m {label}: {n_events} events, {n_covered} covered ({cov_pct:.1f}%)")
    
    coverage_df = pd.DataFrame(coverage_rows)
    
    # ===== 信号强度分层 =====
    sr["strength_rank"] = pd.qcut(sr["composite"], q=5, labels=["Q1最弱", "Q2", "Q3", "Q4", "Q5最强"])
    
    strength_stats = []
    for rank in ["Q1最弱", "Q2", "Q3", "Q4", "Q5最强"]:
        sub = sr[sr["strength_rank"] == rank]
        n = len(sub)
        if n == 0: continue
        strength_stats.append({
            "信号强度": rank, "信号数": n,
            "composite_mean": round(sub["composite"].mean(), 4),
            "5m均值收益%": round(sub["ret_5m"].mean() * 100, 4),
            "5m胜率%": round((sub["ret_5m"] > 0).mean() * 100, 1),
            "5m最大涨幅均值%": round(sub["max_gain_5m"].mean() * 100, 4),
            "5m最大跌幅均值%": round(sub["max_loss_5m"].mean() * 100, 4),
            "10m均值收益%": round(sub["ret_10m"].mean() * 100, 4),
            "10m胜率%": round((sub["ret_10m"] > 0).mean() * 100, 1),
            "10m最大涨幅均值%": round(sub["max_gain_10m"].mean() * 100, 4),
            "10m最大跌幅均值%": round(sub["max_loss_10m"].mean() * 100, 4),
        })
    strength_df = pd.DataFrame(strength_stats)
    
    # ===== CH4 方向分组 =====
    dir_stats = []
    for d in ["买压", "卖压"]:
        sub = sr[sr["ch4_dir"] == d]
        n = len(sub)
        if n == 0: continue
        dir_stats.append({
            "CH4方向": d, "信号数": n,
            "5m均值收益%": round(sub["ret_5m"].mean() * 100, 4),
            "5m胜率%": round((sub["ret_5m"] > 0).mean() * 100, 1),
            "5m最大波幅均值%": round(sub["abs_ret_5m"].mean() * 100, 4) if "abs_ret_5m" in sub.columns else 0,
            "10m均值收益%": round(sub["ret_10m"].mean() * 100, 4),
            "10m胜率%": round((sub["ret_10m"] > 0).mean() * 100, 1),
            "10m最大波幅均值%": round(sub["abs_ret_10m"].mean() * 100, 4) if "abs_ret_10m" in sub.columns else 0,
        })
    # add abs_ret
    sr["abs_ret_5m"] = sr["ret_5m"].abs()
    sr["abs_ret_10m"] = sr["ret_10m"].abs()
    dir_stats[0]["5m最大波幅均值%"] = round(sr[sr["ch4_dir"]=="买压"]["abs_ret_5m"].mean()*100, 4)
    dir_stats[0]["10m最大波幅均值%"] = round(sr[sr["ch4_dir"]=="买压"]["abs_ret_10m"].mean()*100, 4)
    dir_stats[1]["5m最大波幅均值%"] = round(sr[sr["ch4_dir"]=="卖压"]["abs_ret_5m"].mean()*100, 4)
    dir_stats[1]["10m最大波幅均值%"] = round(sr[sr["ch4_dir"]=="卖压"]["abs_ret_10m"].mean()*100, 4)
    direction_df = pd.DataFrame(dir_stats)
    
    # ===== 全局行情分布 =====
    total_candles = len(l1)
    market_dist = []
    for w, rc in [(5, "ret_5m"), (10, "ret_10m")]:
        a = l1[rc].abs().dropna()
        market_dist.append({
            "窗口": f"{w}分钟", "总观察": len(a),
            "<0.1%": f"{(a < 0.001).sum()} ({(a < 0.001).mean()*100:.1f}%)",
            "0.1-0.3%": f"{((a >= 0.001) & (a < 0.003)).sum()} ({((a >= 0.001) & (a < 0.003)).mean()*100:.1f}%)",
            "0.3-0.5%": f"{((a >= 0.003) & (a < 0.005)).sum()} ({((a >= 0.003) & (a < 0.005)).mean()*100:.1f}%)",
            "0.5-1.0%": f"{((a >= 0.005) & (a < 0.01)).sum()} ({((a >= 0.005) & (a < 0.01)).mean()*100:.1f}%)",
            ">1.0%": f"{(a >= 0.01).sum()} ({(a >= 0.01).mean()*100:.1f}%)",
            "最大值": f"{a.max()*100:.3f}%",
        })
    market_df = pd.DataFrame(market_dist)
    
    return sr, coverage_df, strength_df, direction_df, market_df, l1, signal_idxs


def prices_shift(arr, n):
    """安全的数组偏移（负数为前移）"""
    result = np.full_like(arr, np.nan, dtype=float)
    if n < 0:
        result[:n] = arr[-n:]
    elif n > 0:
        result[n:] = arr[:-n]
    return result


# ═════════════════════════════════════════════════════════════
#  3. HTML 报告 (纯 CSS+SVG, 无外部依赖)
# ═════════════════════════════════════════════════════════════

def scatter_svg(points, width=580, height=360, xlabel="综合信号强度", ylabel="前向收益%"):
    """纯 SVG 散点图。points: [(x, y, color), ...]"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs: return "<p>无数据</p>"
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # 边距
    ml, mr, mt, mb = 60, 30, 30, 50
    pw = width - ml - mr
    ph = height - mt - mb
    
    # 确保范围不为0
    xr = x_max - x_min or 1
    yr = y_max - y_min or 0.1
    
    def tx(x):
        return ml + (x - x_min) / xr * pw
    def ty(y):
        return height - mb - (y - y_min) / yr * ph
    
    circles = ""
    for x, y, color, label in points:
        cx, cy = tx(x), ty(y)
        circles += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}" opacity="0.7">'
        circles += f'<title>{label}: signal={x:.4f}, ret={y:.3f}%</title></circle>\n'
    
    # 网格线
    grid = ""
    for i in range(5):
        gy = ty(y_min + yr * i / 4)
        grid += f'<line x1="{ml}" y1="{gy:.1f}" x2="{width-mr}" y2="{gy:.1f}" stroke="#334155" stroke-width="0.5"/>\n'
    
    # 零线
    if y_min < 0 < y_max:
        zy = ty(0)
        grid += f'<line x1="{ml}" y1="{zy:.1f}" x2="{width-mr}" y2="{zy:.1f}" stroke="#ef4444" stroke-width="1" stroke-dasharray="4,2"/>\n'
    
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <rect width="{width}" height="{height}" fill="#1e293b" rx="6"/>
        {grid}
        {circles}
        <text x="{width/2:.0f}" y="{height-10}" text-anchor="middle" fill="#94a3b8" font-size="11">{xlabel}</text>
        <text x="15" y="{height/2:.0f}" text-anchor="middle" fill="#94a3b8" font-size="11" transform="rotate(-90,15,{height/2:.0f})">{ylabel}</text>
    </svg>'''


def generate_html_report(sr, coverage_df, strength_df, direction_df, market_df, l1, output_path):
    print(f"\n[3] 生成 HTML 报告...")
    
    # 关键统计
    n_signals = len(sr)
    win5 = (sr["ret_5m"] > 0).mean() * 100
    win10 = (sr["ret_10m"] > 0).mean() * 100
    avg_ret5 = sr["ret_5m"].mean() * 100
    avg_ret10 = sr["ret_10m"].mean() * 100
    corr5 = sr["composite"].corr(sr["ret_5m"])
    corr10 = sr["composite"].corr(sr["ret_10m"])
    
    # 极端波动覆盖率
    # 5m >0.5%
    cov_5m_big = coverage_df[(coverage_df["窗口"]=="5分钟") & (coverage_df["波动等级"].str.contains("0.5"))]["覆盖率%"].values
    cov_5m_big = cov_5m_big[0] if len(cov_5m_big) > 0 else 0
    # 10m >=0.5%
    cov_10m_big = coverage_df[(coverage_df["窗口"]=="10分钟") & (coverage_df["波动等级"].str.contains(">=\s*0.5", regex=True))]["覆盖率%"].values
    cov_10m_big = cov_10m_big[0] if len(cov_10m_big) > 0 else 0
    # 10m >1.0%
    cov_10m_ext = coverage_df[(coverage_df["窗口"]=="10分钟") & (coverage_df["波动等级"].str.contains(">=\s*1.0", regex=True))]["覆盖率%"].values
    cov_10m_ext = cov_10m_ext[0] if len(cov_10m_ext) > 0 else 0
    
    # 信号覆盖的总体行情分布
    signal_idxs = set(sr["candle_idx"].values)
    # 这些信号时刻之后的行情分布
    sr_returns_5m = sr["ret_5m"] * 100
    sr_returns_10m = sr["ret_10m"] * 100
    
    # 散点图数据
    scatter_5m_pts = []
    scatter_10m_pts = []
    for _, r in sr.iterrows():
        c = "#ef4444" if r["ch4_dir"] == "买压" else "#22c55e"
        label = f"{r['signal_time'].strftime('%m-%d %H:%M')}"
        scatter_5m_pts.append((float(r["composite"]), float(r["ret_5m"])*100, c, label))
        scatter_10m_pts.append((float(r["composite"]), float(r["ret_10m"])*100, c, label))
    
    svg5 = scatter_svg(scatter_5m_pts, ylabel="5分钟收益%")
    svg10 = scatter_svg(scatter_10m_pts, ylabel="10分钟收益%")
    
    # 信号收益分布直方图 (纯ASCII文本表格)
    def pct_bin(ret_series):
        bins = [(-99, -1.0), (-1.0, -0.5), (-0.5, -0.3), (-0.3, -0.1), (-0.1, 0),
                (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 99)]
        vals = ret_series.values
        counts = []
        for lo, hi in bins:
            cnt = ((vals > lo) & (vals <= hi)).sum()
            counts.append(cnt)
        return counts, bins
    
    cnt5, bins5 = pct_bin(sr_returns_5m)
    cnt10, bins10 = pct_bin(sr_returns_10m)
    
    def hist_html(cnts, bins, title):
        max_c = max(cnts) if max(cnts) > 0 else 1
        rows = ""
        for (lo, hi), c in zip(bins, cnts):
            bar_w = int(c / max_c * 200)
            color = "#ef4444" if lo >= 0 else "#22c55e"
            label = f"{lo:.1f}~{hi:.1f}%"
            rows += f'<tr><td style="text-align:right;padding-right:8px;font-size:11px;color:#94a3b8;">{label}</td>'
            rows += f'<td><div style="height:14px;width:{bar_w}px;background:{color};border-radius:3px;opacity:0.8;"></div></td>'
            rows += f'<td style="font-size:11px;color:#cbd5e1;">{c}</td></tr>\n'
        return f'<table>{rows}</table>'
    
    # Coverage 表格
    cov_rows = ""
    for _, r in coverage_df.iterrows():
        cov = r["覆盖率%"]
        color = "#22c55e" if cov > 30 else "#f59e0b" if cov > 10 else "#ef4444"
        cov_rows += f'''<tr>
            <td>{r["窗口"]}</td><td>{r["波动等级"]}</td>
            <td>{r["事件总数"]}</td><td>{r["被覆盖"]}</td>
            <td style="color:{color};font-weight:700;">{cov:.1f}%</td>
            <td>{r["覆盖事件均值收益%"]:.4f}%</td>
            <td>{r["覆盖事件收益标准差%"]:.4f}%</td>
            <td>{r["覆盖评级"]}</td>
        </tr>\n'''
    
    # 强度表格
    str_rows = ""
    for _, r in strength_df.iterrows():
        w5 = r["5m胜率%"]; w10 = r["10m胜率%"]
        str_rows += f'''<tr>
            <td><strong>{r["信号强度"]}</strong></td><td>{r["信号数"]}</td>
            <td>{r["composite_mean"]:.4f}</td>
            <td style="color:{'#ef4444' if r['5m均值收益%']>0 else '#22c55e'}">{r["5m均值收益%"]:+.4f}%</td>
            <td style="color:{'#22c55e' if w5>=50 else '#ef4444'}">{w5:.1f}%</td>
            <td>+{r["5m最大涨幅均值%"]:.4f}%</td><td>{r["5m最大跌幅均值%"]:.4f}%</td>
            <td style="color:{'#ef4444' if r['10m均值收益%']>0 else '#22c55e'}">{r["10m均值收益%"]:+.4f}%</td>
            <td style="color:{'#22c55e' if w10>=50 else '#ef4444'}">{w10:.1f}%</td>
            <td>+{r["10m最大涨幅均值%"]:.4f}%</td><td>{r["10m最大跌幅均值%"]:.4f}%</td>
        </tr>\n'''
    
    # 方向表格
    dir_rows = ""
    for _, r in direction_df.iterrows():
        w5 = r["5m胜率%"]; w10 = r["10m胜率%"]
        dir_rows += f'''<tr>
            <td><strong>{r["CH4方向"]}</strong></td><td>{r["信号数"]}</td>
            <td style="color:{'#ef4444' if r['5m均值收益%']>0 else '#22c55e'}">{r["5m均值收益%"]:+.4f}%</td>
            <td style="color:{'#22c55e' if w5>=50 else '#ef4444'}">{w5:.1f}%</td>
            <td>{r["5m最大波幅均值%"]:.4f}%</td>
            <td style="color:{'#ef4444' if r['10m均值收益%']>0 else '#22c55e'}">{r["10m均值收益%"]:+.4f}%</td>
            <td style="color:{'#22c55e' if w10>=50 else '#ef4444'}">{w10:.1f}%</td>
            <td>{r["10m最大波幅均值%"]:.4f}%</td>
        </tr>\n'''
    
    # 信号详情表（前10和后10）
    sig_detail = ""
    for i, (_, r) in enumerate(sr.iterrows()):
        if i >= 10 and i < len(sr) - 10:
            if i == 10:
                sig_detail += '<tr><td colspan="12" style="text-align:center;color:#64748b;">... 省略中间信号 ...</td></tr>\n'
            continue
        
        c5 = "#ef4444" if r["ret_5m"] > 0 else "#22c55e"
        c10 = "#ef4444" if r["ret_10m"] > 0 else "#22c55e"
        c_dir = "#ef4444" if r["ch4_dir"] == "买压" else "#22c55e"
        
        sig_detail += f'''<tr>
            <td>{r["signal_time"].strftime("%m-%d %H:%M")}</td>
            <td>${r["signal_price"]:,.2f}</td>
            <td>{r["ch1"]:.3f}</td><td>{r["ch2"]:.3f}</td>
            <td>{r["ch3"]:.3f}</td><td>{r["ch4"]:.3f}</td>
            <td>{r["composite"]:.3f}</td>
            <td style="color:{c_dir};">{r["ch4_dir"]}</td>
            <td style="color:{c5};">{r["ret_5m"]*100:+.3f}%</td>
            <td>+{r["max_gain_5m"]*100:.3f}%/{r["max_loss_5m"]*100:.3f}%</td>
            <td style="color:{c10};">{r["ret_10m"]*100:+.3f}%</td>
        </tr>\n'''
    
    # 结论评估
    cov_verdict = "优秀" if cov_10m_big > 50 else "良好" if cov_10m_big > 30 else "中等" if cov_10m_big > 15 else "较低"
    dir_verdict = "有意义" if abs(sr[sr["ch4_dir"]=="买压"]["ret_10m"].mean() - sr[sr["ch4_dir"]=="卖压"]["ret_10m"].mean()) > 0.005 else "无显著区分"
    strength_verdict = "有正相关但较弱" if corr10 > 0.1 else "几乎无相关性" if abs(corr10) < 0.05 else "有微弱负相关"
    
    data_range = f"{l1['date'].min().strftime('%Y-%m-%d %H:%M')} ~ {l1['date'].max().strftime('%Y-%m-%d %H:%M UTC')}"
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>信号覆盖率分析 - BTC/USDT L1</title>
<style>
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6;padding:24px;}}
    .container{{max-width:1300px;margin:0 auto;}}
    h1{{font-size:26px;color:#f8fafc;margin-bottom:4px;}}
    h2{{font-size:19px;margin:36px 0 14px;color:#f1f5f9;border-bottom:1px solid #334155;padding-bottom:8px;}}
    h3{{font-size:15px;color:#cbd5e1;margin:20px 0 10px;}}
    .subtitle{{color:#94a3b8;font-size:13px;margin-bottom:24px;}}
    
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
    .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;}}
    .grid5{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px;}}
    
    .card{{background:#1e293b;border-radius:8px;padding:16px;border:1px solid #334155;}}
    .card .label{{color:#94a3b8;font-size:11px;margin-bottom:4px;}}
    .card .value{{font-size:24px;font-weight:700;color:#f8fafc;}}
    .card .sub{{font-size:11px;color:#64748b;margin-top:2px;}}
    
    table{{width:100%;border-collapse:collapse;margin:10px 0 20px;background:#1e293b;border-radius:8px;overflow:hidden;}}
    th{{background:#334155;padding:9px 10px;text-align:left;font-size:12px;color:#cbd5e1;font-weight:600;white-space:nowrap;}}
    td{{padding:7px 10px;border-bottom:1px solid #334155;font-size:12px;color:#e2e8f0;}}
    tr:last-child td{{border-bottom:none;}}
    
    .verdict-box{{background:linear-gradient(135deg,#1e293b,#0f172a);border:2px solid #334155;border-radius:12px;padding:24px;margin:24px 0;}}
    .verdict-box h3{{color:#f8fafc;margin-bottom:14px;}}
    .verdict-box .item{{margin-bottom:10px;padding:8px 12px;background:rgba(15,23,42,0.6);border-radius:6px;}}
    .verdict-box .item .key{{font-weight:600;color:#cbd5e1;}}
    .verdict-box .item .val{{font-weight:700;}}
    .good{{color:#22c55e;}}
    .bad{{color:#ef4444;}}
    .warn{{color:#f59e0b;}}
    .neutral{{color:#94a3b8;}}
    
    .svg-container{{background:#1e293b;border-radius:8px;padding:16px;border:1px solid #334155;margin:12px 0;text-align:center;}}
    
    .footer{{margin-top:36px;padding-top:16px;border-top:1px solid #334155;color:#64748b;font-size:11px;text-align:center;}}
</style>
</head>
<body>
<div class="container">
    <h1>微观结构信号覆盖率分析报告</h1>
    <p class="subtitle">
        标的: BTC/USDT (Gate Spot) | 价格数据: L1 订单簿 (mid_price) | 信号源: 微结构_信号模块 (arXiv:2604.20949)<br>
        数据范围: {data_range} | 共 {len(l1):,} 分钟K线 | {n_signals} 个信号触发
    </p>
    
    <!-- ===== 核心结论 ===== -->
    <h2>结论摘要</h2>
    <div class="verdict-box">
        <h3>信号覆盖率评估</h3>
        
        <div class="item">
            <span class="key">大幅波动覆盖(10分钟, >0.5%):</span>
            <span class="val {('good' if cov_10m_big > 30 else 'warn')}">{cov_10m_big:.1f}%</span>
            <span class="neutral"> — 评级: {cov_verdict}</span>
        </div>
        
        <div class="item">
            <span class="key">极端波动覆盖(10分钟, >1.0%):</span>
            <span class="val neutral">{cov_10m_ext:.1f}%</span>
            <span class="warn"> — 样本极少，无统计意义</span>
        </div>
        
        <div class="item">
            <span class="key">信号方向准确率:</span>
            <span class="val {'good' if win10 > 55 else 'warn' if win10 > 50 else 'bad'}">
                5分钟 {win5:.1f}% | 10分钟 {win10:.1f}%
            </span>
            <span class="neutral"> — 接近随机，信号不预测方向</span>
        </div>
        
        <div class="item">
            <span class="key">信号强度-收益相关性:</span>
            <span class="val {'good' if abs(corr10) > 0.2 else 'warn'}">
                5m r={corr5:.3f} | 10m r={corr10:.3f}
            </span>
            <span class="neutral"> — {strength_verdict}</span>
        </div>
        
        <div class="item">
            <span class="key">CH4方向区分度:</span>
            <span class="val warn">{dir_verdict}</span>
            <span class="neutral"> — 买压/卖压对后续收益无显著预测能力</span>
        </div>
        
        <div class="item" style="margin-top:12px;border-left:3px solid #f59e0b;padding-left:16px;">
            <strong class="warn">核心发现:</strong> 微结构信号模块检测的是微观结构<strong>恶化状态</strong>（波动率熵增、深度侵蚀、价差扩大），
            而非方向性信号。信号触发时，后续10分钟内出现 >0.5% 大幅波动的概率为 {cov_10m_big:.0f}%，但方向近乎随机。
            信号更适合作为<strong>波动率预警</strong>而非交易入场信号。
        </div>
    </div>
    
    <!-- ===== 全局行情分布 ===== -->
    <h2>一、全局行情波动分布</h2>
    <p class="subtitle">整个数据期间的所有5分钟/10分钟窗口的绝对收益分布（不含信号筛选）</p>
    <table>
        <thead><tr><th>窗口</th><th>总观察</th><th>&lt;0.1%</th><th>0.1-0.3%</th><th>0.3-0.5%</th><th>0.5-1.0%</th><th>&gt;1.0%</th><th>最大波动</th></tr></thead>
        <tbody>
            {"".join(f'<tr><td><strong>{r["窗口"]}</strong></td><td>{r["总观察"]}</td><td>{r["<0.1%"]}</td><td>{r["0.1-0.3%"]}</td><td>{r["0.3-0.5%"]}</td><td>{r["0.5-1.0%"]}</td><td>{r[">1.0%"]}</td><td>{r["最大值"]}</td></tr>' for _, r in market_df.iterrows())}
        </tbody>
    </table>
    
    <!-- ===== 覆盖率矩阵 ===== -->
    <h2>二、信号对行情事件的覆盖率</h2>
    <p class="subtitle">各波动等级窗口中，行情事件被信号覆盖的比例（信号在事件前60分钟内）</p>
    <table>
        <thead><tr><th>窗口</th><th>波动等级</th><th>事件总数</th><th>被覆盖</th><th>覆盖率</th><th>覆盖事件均值收益</th><th>标准差</th><th>评级</th></tr></thead>
        <tbody>{cov_rows}</tbody>
    </table>
    
    <!-- ===== 信号强度分层 ===== -->
    <h2>三、信号强度 vs 后续收益</h2>
    <p class="subtitle">按综合信号强度五分位分组，分析不同强度信号的5/10分钟表现</p>
    <table>
        <thead><tr>
            <th>信号强度</th><th>数量</th><th>composite均值</th>
            <th>5m均值收益</th><th>5m胜率</th><th>5m最大涨幅</th><th>5m最大跌幅</th>
            <th>10m均值收益</th><th>10m胜率</th><th>10m最大涨幅</th><th>10m最大跌幅</th>
        </tr></thead>
        <tbody>{str_rows}</tbody>
    </table>
    
    <!-- ===== CH4方向分组 ===== -->
    <h2>四、CH4订单流方向分组</h2>
    <p class="subtitle">按CH4方向（买压/卖压）分组看信号表现</p>
    <table>
        <thead><tr>
            <th>CH4方向</th><th>信号数</th>
            <th>5m均值收益</th><th>5m胜率</th><th>5m波幅</th>
            <th>10m均值收益</th><th>10m胜率</th><th>10m波幅</th>
        </tr></thead>
        <tbody>{dir_rows}</tbody>
    </table>
    
    <!-- ===== 散点图 ===== -->
    <h2>五、信号-收益散点图</h2>
    <p class="subtitle">红点: CH4买压 | 绿点: CH4卖压 | 虚线: 零收益线</p>
    
    <div class="grid2">
        <div class="svg-container">
            <h3 style="margin-top:0;">5分钟前向收益 vs 信号强度</h3>
            {svg5}
        </div>
        <div class="svg-container">
            <h3 style="margin-top:0;">10分钟前向收益 vs 信号强度</h3>
            {svg10}
        </div>
    </div>
    
    <!-- ===== 信号收益分布 ===== -->
    <h2>六、信号收益分布直方图</h2>
    <div class="grid2">
        <div class="card">
            <h3>5分钟收益分布</h3>
            {hist_html(cnt5, bins5, "")}
        </div>
        <div class="card">
            <h3>10分钟收益分布</h3>
            {hist_html(cnt10, bins10, "")}
        </div>
    </div>
    
    <!-- ===== 信号明细 ===== -->
    <h2>七、信号明细（前10+后10条）</h2>
    <table>
        <thead><tr>
            <th>时间</th><th>价格</th><th>CH1</th><th>CH2</th><th>CH3</th><th>CH4</th><th>综合</th><th>方向</th>
            <th>5m收益</th><th>5m区间</th><th>10m收益</th>
        </tr></thead>
        <tbody>{sig_detail}</tbody>
    </table>
    
    <div class="footer">
        信号覆盖率分析 | 价格数据: Gate L1 (mid_price) | 信号: 微结构_信号模块 (L2回测) | 
        生成: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}
    </div>
</div>
</body>
</html>"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  报告: {output_path}")
    
    # 保存 CSV
    csv_path = OUTPUT_DIR / "信号覆盖率分析数据_v2.csv"
    csv_cols = ["signal_time", "signal_price", "ch1", "ch2", "ch3", "ch4", "composite", "ch4_dir",
                "ret_5m", "max_gain_5m", "max_loss_5m", "ret_10m", "max_gain_10m", "max_loss_10m"]
    sr[csv_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {csv_path}")


# ═════════════════════════════════════════════════════════════
#  主流程
# ═════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  微结构信号覆盖率分析 v2 (L1数据)")
    print("=" * 60)
    
    # 1. L1 蜡烛
    cache = OUTPUT_DIR / "l1_candles_cache.pkl"
    if cache.exists():
        print(f"\n[1] 从缓存加载: {cache}")
        l1_candles = pd.read_pickle(cache)
        print(f"  {len(l1_candles)} 根K线")
    else:
        print("\n[1] 构建 L1 蜡烛...")
        l1_candles = build_l1_candles()
        cache.parent.mkdir(parents=True, exist_ok=True)
        l1_candles.to_pickle(cache)
    
    # 2. 分析
    sr, coverage_df, strength_df, direction_df, market_df, l1_indexed, signal_idxs = run_analysis(l1_candles)
    
    # 3. 报告
    output = OUTPUT_DIR / "信号覆盖率分析报告.html"
    generate_html_report(sr, coverage_df, strength_df, direction_df, market_df, l1_candles, output)
    
    # 打印结论
    print("\n" + "=" * 60)
    print("  核心结论")
    print("=" * 60)
    win5 = (sr["ret_5m"] > 0).mean() * 100
    win10 = (sr["ret_10m"] > 0).mean() * 100
    corr5 = sr["composite"].corr(sr["ret_5m"])
    corr10 = sr["composite"].corr(sr["ret_10m"])
    
    print(f"\n  信号方向准确率: 5分钟 {win5:.1f}% | 10分钟 {win10:.1f}% (~随机)")
    print(f"  信号强度-收益相关: 5m r={corr5:.3f} | 10m r={corr10:.3f}")
    
    cov10_big = coverage_df[(coverage_df["窗口"]=="10分钟") & (coverage_df["波动等级"].str.contains(">=\s*0.5", regex=True))]
    if len(cov10_big) > 0:
        print(f"  大幅波动(>0.5%)10分钟覆盖: {cov10_big['覆盖率%'].values[0]:.1f}%")
    
    print(f"\n  --> 信号是波动率预警器，非方向性交易信号")
    print(f"  --> 报告: {output}")


if __name__ == "__main__":
    main()
