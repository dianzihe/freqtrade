# -*- coding: utf-8 -*-
"""
微结构信号模块回测脚本
======================
使用 Gate L2 订单簿数据对 "微结构_信号模块.py" 进行回测。
重建完整订单簿 -> 构建1分钟K线 -> 生成信号 -> 分析15分钟后续行情 -> 输出HTML报告
"""

import json
import os
import sys
import time
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# 项目路径
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "user_data/orderbook_data/gate/spot/BTC_USDT/l2"
STRATEGY_DIR = PROJECT_ROOT / "user_data/strategies"
OUTPUT_DIR = PROJECT_ROOT / "user_data/backtest_results"

# 确保能导入信号模块
sys.path.insert(0, str(PROJECT_ROOT))
from user_data.strategies.微结构_信号模块 import add_lob_regime_signals

LOB_SNAPSHOT_LEVELS = 25  # 信号模块取 top25 档位


# ═════════════════════════════════════════════════════════════
#  1. 加载并重建订单簿
# ═════════════════════════════════════════════════════════════

def load_all_parquet_files():
    """加载所有有效的 parquet 文件路径（按时间排序）。"""
    files = sorted(glob.glob(str(DATA_DIR / "**/*.parquet"), recursive=True))
    valid = []
    for f in files:
        try:
            # 只检查文件头，不全读
            pq_file = pd.read_parquet(f, columns=["exchange_time_ms"])
            valid.append(f)
        except Exception:
            print(f"  [跳过] 损坏文件: {os.path.basename(f)}")
    print(f"  有效文件: {len(valid)}/{len(files)}")
    return valid


def parse_l2_updates(value):
    """解析 L2 更新数据，返回 [(price, amount), ...] 列表。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(value, list):
        return value
    return []


def reconstruct_orderbook(files):
    """
    逐文件、逐 tick 重建订单簿，在每分钟边界采样快照。
    
    返回:
        minute_records: list of dict, 每条记录对应1分钟K线 + L2快照
    """
    bid_book = {}  # {price: amount}
    ask_book = {}
    
    # 当前分钟的聚合状态
    current_minute = None
    minute_open = None
    minute_high = None
    minute_low = None
    minute_close = None
    minute_volume = 0  # update count proxy
    minute_tick_count = 0
    minute_bid_updates_total = 0
    minute_ask_updates_total = 0
    minute_mid_prices = []  # 用于计算波动率
    
    minute_records = []
    
    total_rows = 0
    start_time = time.time()
    
    for file_idx, filepath in enumerate(files):
        df = pd.read_parquet(filepath)
        total_rows += len(df)
        
        # 提取列数据为列表，加速访问
        timestamps = df["exchange_time_ms"].values
        bid_updates_col = df["bid_updates"].values
        ask_updates_col = df["ask_updates"].values
        bid_counts = df["bid_update_count"].values
        ask_counts = df["ask_update_count"].values
        
        # 判断第一行是否为完整快照（>20档）
        first_bid = parse_l2_updates(bid_updates_col[0])
        if len(first_bid) > 20:
            # 完整快照，重置订单簿
            bid_book = {}
            ask_book = {}
        
        for i in range(len(df)):
            ts_ms = int(timestamps[i])
            ts_sec = ts_ms // 1000
            minute_key = ts_sec // 60  # 分钟序号
            
            # 解析并应用 bid 更新
            bid_updates = parse_l2_updates(bid_updates_col[i])
            for level in bid_updates:
                try:
                    price = float(level[0])
                    amount = float(level[1])
                    if amount > 0:
                        bid_book[price] = amount
                    elif price in bid_book:
                        del bid_book[price]
                except (TypeError, ValueError, IndexError):
                    continue
            
            # 解析并应用 ask 更新
            ask_updates = parse_l2_updates(ask_updates_col[i])
            for level in ask_updates:
                try:
                    price = float(level[0])
                    amount = float(level[1])
                    if amount > 0:
                        ask_book[price] = amount
                    elif price in ask_book:
                        del ask_book[price]
                except (TypeError, ValueError, IndexError):
                    continue
            
            # 计算当前 mid price
            if bid_book and ask_book:
                best_bid = max(bid_book.keys())
                best_ask = min(ask_book.keys())
                mid_price = (best_bid + best_ask) / 2.0
            else:
                continue  # 没有有效报价，跳过
            
            # 分钟边界检测
            if current_minute is None:
                current_minute = minute_key
                minute_open = mid_price
                minute_high = mid_price
                minute_low = mid_price
                minute_close = mid_price
                minute_volume = 0
                minute_tick_count = 0
                minute_bid_updates_total = 0
                minute_ask_updates_total = 0
                minute_mid_prices = [mid_price]
            elif minute_key != current_minute:
                # 保存上一分钟的数据 + 当前订单簿快照
                record = _build_minute_record(
                    current_minute, minute_open, minute_high, minute_low, minute_close,
                    minute_volume, minute_tick_count,
                    minute_bid_updates_total, minute_ask_updates_total,
                    bid_book, ask_book,
                )
                if record:
                    minute_records.append(record)
                
                # 开始新分钟
                current_minute = minute_key
                minute_open = mid_price
                minute_high = mid_price
                minute_low = mid_price
                minute_close = mid_price
                minute_volume = 0
                minute_tick_count = 0
                minute_bid_updates_total = 0
                minute_ask_updates_total = 0
                minute_mid_prices = [mid_price]
            
            # 更新当前分钟聚合
            minute_high = max(minute_high, mid_price)
            minute_low = min(minute_low, mid_price)
            minute_close = mid_price
            minute_volume += int(bid_counts[i]) + int(ask_counts[i])
            minute_tick_count += 1
            minute_bid_updates_total += int(bid_counts[i])
            minute_ask_updates_total += int(ask_counts[i])
            minute_mid_prices.append(mid_price)
        
        # 进度报告
        elapsed = time.time() - start_time
        speed = total_rows / elapsed if elapsed > 0 else 0
        print(f"  [{file_idx+1}/{len(files)}] {os.path.basename(filepath)} | "
              f"累计 {total_rows:,} ticks | {speed:,.0f} ticks/s | "
              f"{len(minute_records)} 分钟K线")
    
    # 保存最后一分钟
    if current_minute is not None:
        record = _build_minute_record(
            current_minute, minute_open, minute_high, minute_low, minute_close,
            minute_volume, minute_tick_count,
            minute_bid_updates_total, minute_ask_updates_total,
            bid_book, ask_book,
        )
        if record:
            minute_records.append(record)
    
    elapsed = time.time() - start_time
    print(f"\n  重建完成: {total_rows:,} ticks -> {len(minute_records)} 分钟K线 | 耗时 {elapsed:.1f}s")
    
    return minute_records


def _build_minute_record(minute_key, open_price, high, low, close,
                          volume, tick_count, bid_updates_total, ask_updates_total,
                          bid_book, ask_book):
    """构建一条分钟K线记录，包含L2快照。"""
    if not bid_book or not ask_book:
        return None
    
    # 取 top25 档位快照
    bid_snapshot = sorted(bid_book.items(), key=lambda x: -x[0])[:LOB_SNAPSHOT_LEVELS]
    ask_snapshot = sorted(ask_book.items(), key=lambda x: x[0])[:LOB_SNAPSHOT_LEVELS]
    
    # 转为 [[price, amount], ...] 格式（与信号模块兼容）
    bid_list = [[str(p), str(a)] for p, a in bid_snapshot]
    ask_list = [[str(p), str(a)] for p, a in ask_snapshot]
    
    # 计算深度和价差
    best_bid = bid_snapshot[0][0]
    best_ask = ask_snapshot[0][0]
    mid = (best_bid + best_ask) / 2.0
    
    bid_depth_10 = sum(a for _, a in bid_snapshot[:10])
    ask_depth_10 = sum(a for _, a in ask_snapshot[:10])
    
    # 订单流不平衡作为 buy_volume_ratio 代理
    total_depth = bid_depth_10 + ask_depth_10
    if total_depth > 0:
        imbalance = (bid_depth_10 - ask_depth_10) / total_depth
        buy_volume_ratio = (imbalance + 1.0) / 2.0  # 映射到 [0, 1]
    else:
        buy_volume_ratio = 0.5
    
    # 用 update count 作为 volume 代理
    vol = max(volume, 1)
    buy_vol = buy_volume_ratio * vol
    sell_vol = (1.0 - buy_volume_ratio) * vol
    
    ts = pd.Timestamp(minute_key * 60, unit="s", tz="UTC")
    
    return {
        "date": ts,
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(vol),
        "trade_count": int(tick_count),
        "buy_volume": float(buy_vol),
        "sell_volume": float(sell_vol),
        "buy_volume_ratio": float(buy_volume_ratio),
        "bid_updates": bid_list,
        "ask_updates": ask_list,
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "mid_price": float(mid),
        "bid_depth_10": float(bid_depth_10),
        "ask_depth_10": float(ask_depth_10),
    }


# ═════════════════════════════════════════════════════════════
#  2. 生成信号
# ═════════════════════════════════════════════════════════════

def generate_signals(candles_df):
    """调用信号模块生成 ch1-ch4 和 signal_trigger。"""
    print("\n[2] 生成微结构信号...")
    
    df = add_lob_regime_signals(
        candles_df,
        lookback_period=24,
        volatility_window=20,
        threshold_percentile=88,
        confirmation_bars=2,
        min_signal_strength=0.40,
        signal_valid_bars=25,
    )
    
    # 确保列名标准化
    channel_cols = {
        "ch1": "ch1_vol_entropy",
        "ch2": "ch2_depth_erosion",
        "ch3": "ch3_spread_drift",
        "ch4": "ch4_order_flow",
    }
    
    signal_count = int(df["signal_trigger"].sum())
    print(f"  信号触发次数: {signal_count}")
    print(f"  K线总数: {len(df)}")
    print(f"  信号占比: {signal_count/len(df)*100:.2f}%")
    
    return df


# ═════════════════════════════════════════════════════════════
#  3. 分析信号后的15分钟行情
# ═════════════════════════════════════════════════════════════

def analyze_forward_price_action(df, forward_minutes=15):
    """
    对每个信号触发点，分析后续 forward_minutes 分钟内的行情。
    """
    print(f"\n[3] 分析信号后 {forward_minutes} 分钟行情...")
    
    signals = df[df["signal_trigger"] == 1].copy()
    
    if len(signals) == 0:
        print("  无信号触发，跳过分析。")
        return []
    
    results = []
    
    for idx in signals.index:
        pos = df.index.get_loc(idx)
        signal_row = df.loc[idx]
        signal_price = signal_row["close"]
        signal_time = signal_row["date"]
        
        # 提取信号时刻的通道值
        ch1_val = signal_row.get("ch1_vol_entropy", 0)
        ch2_val = signal_row.get("ch2_depth_erosion", 0)
        ch3_val = signal_row.get("ch3_spread_drift", 0)
        ch4_val = signal_row.get("ch4_order_flow", 0)
        composite = signal_row.get("composite_smooth", 0)
        threshold = signal_row.get("adaptive_threshold", 0)
        
        # 获取后续 forward_minutes 根K线
        end_pos = min(pos + forward_minutes + 1, len(df))
        forward = df.iloc[pos + 1 : end_pos]
        
        if len(forward) == 0:
            continue
        
        # 计算后续行情指标
        forward_highs = forward["high"].values
        forward_lows = forward["low"].values
        forward_closes = forward["close"].values
        
        max_high = float(np.max(forward_highs))
        min_low = float(np.min(forward_lows))
        final_close = float(forward_closes[-1])
        
        max_gain_pct = (max_high - signal_price) / signal_price * 100
        max_loss_pct = (min_low - signal_price) / signal_price * 100
        net_change_pct = (final_close - signal_price) / signal_price * 100
        
        # 每分钟收盘价序列
        minute_prices = []
        for m in range(forward_minutes):
            if m < len(forward):
                price = float(forward_closes[m])
                change_pct = (price - signal_price) / signal_price * 100
                minute_prices.append({
                    "minute": m + 1,
                    "price": price,
                    "change_pct": change_pct,
                })
        
        # 判断方向
        if net_change_pct > 0.01:
            direction = "上涨"
        elif net_change_pct < -0.01:
            direction = "下跌"
        else:
            direction = "震荡"
        
        # 判断信号有效性（假设信号预测下跌，即"微观结构恶化"）
        # 但实际方向取决于 ch4 的符号
        ch4_direction = "卖压" if ch4_val < 0 else "买压"
        
        result = {
            "signal_time": signal_time,
            "signal_price": signal_price,
            "ch1": ch1_val,
            "ch2": ch2_val,
            "ch3": ch3_val,
            "ch4": ch4_val,
            "composite": composite,
            "threshold": threshold,
            "ch4_direction": ch4_direction,
            "max_high": max_high,
            "min_low": min_low,
            "final_close": final_close,
            "max_gain_pct": max_gain_pct,
            "max_loss_pct": max_loss_pct,
            "net_change_pct": net_change_pct,
            "direction": direction,
            "minute_prices": minute_prices,
            "forward_count": len(forward),
        }
        results.append(result)
    
    print(f"  分析完成: {len(results)} 个信号")
    
    # 统计
    if results:
        up_count = sum(1 for r in results if r["net_change_pct"] > 0)
        down_count = sum(1 for r in results if r["net_change_pct"] < 0)
        flat_count = len(results) - up_count - down_count
        avg_gain = np.mean([r["max_gain_pct"] for r in results])
        avg_loss = np.mean([r["max_loss_pct"] for r in results])
        avg_net = np.mean([r["net_change_pct"] for r in results])
        
        print(f"  后续上涨: {up_count} | 下跌: {down_count} | 震荡: {flat_count}")
        print(f"  平均最大涨幅: {avg_gain:.3f}% | 平均最大跌幅: {avg_loss:.3f}%")
        print(f"  平均净变化: {avg_net:.3f}%")
    
    return results


# ═════════════════════════════════════════════════════════════
#  4. 生成 HTML 报告
# ═════════════════════════════════════════════════════════════

def generate_html_report(df, signals, candles_count, output_path):
    """生成包含信号详情和行情分析的 HTML 报告。"""
    print(f"\n[4] 生成 HTML 报告...")
    
    data_start = df["date"].min().strftime("%Y-%m-%d %H:%M UTC")
    data_end = df["date"].max().strftime("%Y-%m-%d %H:%M UTC")
    
    # 统计
    up_count = sum(1 for s in signals if s["net_change_pct"] > 0)
    down_count = sum(1 for s in signals if s["net_change_pct"] < 0)
    flat_count = len(signals) - up_count - down_count
    
    avg_gain = np.mean([s["max_gain_pct"] for s in signals]) if signals else 0
    avg_loss = np.mean([s["max_loss_pct"] for s in signals]) if signals else 0
    avg_net = np.mean([s["net_change_pct"] for s in signals]) if signals else 0
    
    # 构建信号表格行
    signal_rows_html = ""
    for i, s in enumerate(signals):
        # 方向颜色
        if s["net_change_pct"] > 0:
            dir_color = "#ef4444"  # 涨=红
            dir_bg = "rgba(239,68,68,0.08)"
        elif s["net_change_pct"] < 0:
            dir_color = "#22c55e"  # 跌=绿
            dir_bg = "rgba(34,197,94,0.08)"
        else:
            dir_color = "#94a3b8"
            dir_bg = "rgba(148,163,184,0.08)"
        
        # 15分钟走势迷你图 (SVG sparkline)
        sparkline = _generate_sparkline_svg(s["minute_prices"], s["signal_price"])
        
        # 每分钟价格变化
        minute_changes = " | ".join(
            f"+{m}m: {mp['change_pct']:+.3f}%" for m, mp in enumerate(s["minute_prices"])
        )
        
        signal_rows_html += f"""
        <div class="signal-card" style="border-left: 4px solid {dir_color}; background: {dir_bg};">
            <div class="signal-header">
                <span class="signal-num">#{i+1}</span>
                <span class="signal-time">{s['signal_time'].strftime('%Y-%m-%d %H:%M')} UTC</span>
                <span class="signal-price">BTC ${s['signal_price']:,.2f}</span>
                <span class="signal-dir" style="color:{dir_color};">{s['direction']} {s['net_change_pct']:+.3f}%</span>
            </div>
            <div class="signal-body">
                <div class="signal-channels">
                    <div class="ch-box">
                        <div class="ch-label">CH1 波动率熵</div>
                        <div class="ch-value">{s['ch1']:.4f}</div>
                        <div class="ch-bar"><div class="ch-fill" style="width:{min(s['ch1']*100,100):.0f}%;background:#f59e0b;"></div></div>
                    </div>
                    <div class="ch-box">
                        <div class="ch-label">CH2 深度侵蚀</div>
                        <div class="ch-value">{s['ch2']:.4f}</div>
                        <div class="ch-bar"><div class="ch-fill" style="width:{min(s['ch2']*100,100):.0f}%;background:#ef4444;"></div></div>
                    </div>
                    <div class="ch-box">
                        <div class="ch-label">CH3 价差漂移</div>
                        <div class="ch-value">{s['ch3']:.4f}</div>
                        <div class="ch-bar"><div class="ch-fill" style="width:{min(s['ch3']*100,100):.0f}%;background:#8b5cf6;"></div></div>
                    </div>
                    <div class="ch-box">
                        <div class="ch-label">CH4 订单流({s['ch4_direction']})</div>
                        <div class="ch-value">{s['ch4']:.4f}</div>
                        <div class="ch-bar"><div class="ch-fill" style="width:{min(abs(s['ch4'])*100,100):.0f}%;background:#3b82f6;"></div></div>
                    </div>
                    <div class="ch-box">
                        <div class="ch-label">综合信号</div>
                        <div class="ch-value">{s['composite']:.4f}</div>
                        <div class="ch-bar"><div class="ch-fill" style="width:{min(s['composite']*100,100):.0f}%;background:#ec4899;"></div></div>
                    </div>
                </div>
                <div class="signal-forward">
                    <div class="forward-stats">
                        <div class="stat"><span class="stat-label">最大涨幅</span><span class="stat-val" style="color:#ef4444;">+{s['max_gain_pct']:.3f}%</span></div>
                        <div class="stat"><span class="stat-label">最大跌幅</span><span class="stat-val" style="color:#22c55e;">{s['max_loss_pct']:.3f}%</span></div>
                        <div class="stat"><span class="stat-label">15分钟净变化</span><span class="stat-val" style="color:{dir_color};">{s['net_change_pct']:+.3f}%</span></div>
                        <div class="stat"><span class="stat-label">最高价</span><span class="stat-val">${s['max_high']:,.2f}</span></div>
                        <div class="stat"><span class="stat-label">最低价</span><span class="stat-val">${s['min_low']:,.2f}</span></div>
                        <div class="stat"><span class="stat-label">15分钟后收盘</span><span class="stat-val">${s['final_close']:,.2f}</span></div>
                    </div>
                    <div class="sparkline-container">{sparkline}</div>
                </div>
                <div class="minute-breakdown">{minute_changes}</div>
            </div>
        </div>
        """
    
    # 构建信号分布统计
    ch_stats = ""
    for ch_name, col in [("CH1 波动率熵", "ch1_vol_entropy"),
                          ("CH2 深度侵蚀", "ch2_depth_erosion"),
                          ("CH3 价差漂移", "ch3_spread_drift"),
                          ("CH4 订单流", "ch4_order_flow"),
                          ("综合信号", "composite_smooth")]:
        signal_vals = df.loc[df["signal_trigger"] == 1, col] if col in df.columns else pd.Series()
        if len(signal_vals) > 0:
            ch_stats += f"""
            <tr>
                <td>{ch_name}</td>
                <td>{signal_vals.mean():.4f}</td>
                <td>{signal_vals.max():.4f}</td>
                <td>{signal_vals.min():.4f}</td>
                <td>{signal_vals.std():.4f}</td>
            </tr>
            """
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>微结构信号回测报告 - BTC/USDT</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
            background: #0f172a; color: #e2e8f0; line-height: 1.6;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 28px; margin-bottom: 8px; color: #f8fafc; }}
        h2 {{ font-size: 20px; margin: 32px 0 16px; color: #f1f5f9; 
             border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 24px; }}
        
        .summary-grid {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 24px;
        }}
        .summary-card {{ 
            background: #1e293b; border-radius: 8px; padding: 16px;
            border: 1px solid #334155;
        }}
        .summary-card .label {{ color: #94a3b8; font-size: 12px; margin-bottom: 4px; }}
        .summary-card .value {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
        .summary-card .sub {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
        
        .stats-table {{ 
            width: 100%; border-collapse: collapse; margin-bottom: 24px;
            background: #1e293b; border-radius: 8px; overflow: hidden;
        }}
        .stats-table th {{ 
            background: #334155; padding: 10px 14px; text-align: left;
            font-size: 13px; color: #cbd5e1; font-weight: 600;
        }}
        .stats-table td {{ 
            padding: 8px 14px; border-bottom: 1px solid #334155;
            font-size: 13px; color: #e2e8f0;
        }}
        .stats-table tr:last-child td {{ border-bottom: none; }}
        
        .signal-card {{ 
            border-radius: 8px; padding: 16px; margin-bottom: 16px;
            border: 1px solid #334155;
        }}
        .signal-header {{ 
            display: flex; align-items: center; gap: 16px; margin-bottom: 12px;
            flex-wrap: wrap;
        }}
        .signal-num {{ 
            background: #334155; color: #f8fafc; padding: 2px 10px;
            border-radius: 4px; font-weight: 700; font-size: 14px;
        }}
        .signal-time {{ font-size: 15px; font-weight: 600; color: #f1f5f9; }}
        .signal-price {{ font-size: 14px; color: #94a3b8; }}
        .signal-dir {{ font-size: 14px; font-weight: 700; margin-left: auto; }}
        
        .signal-body {{ display: flex; flex-direction: column; gap: 12px; }}
        
        .signal-channels {{ 
            display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px;
        }}
        .ch-box {{ background: rgba(15,23,42,0.5); border-radius: 6px; padding: 8px; }}
        .ch-label {{ font-size: 11px; color: #94a3b8; margin-bottom: 2px; }}
        .ch-value {{ font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
        .ch-bar {{ height: 4px; background: #334155; border-radius: 2px; overflow: hidden; }}
        .ch-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
        
        .signal-forward {{ 
            display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap;
        }}
        .forward-stats {{ 
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
            flex: 1; min-width: 300px;
        }}
        .stat {{ background: rgba(15,23,42,0.5); border-radius: 6px; padding: 6px 10px; }}
        .stat-label {{ display: block; font-size: 11px; color: #94a3b8; }}
        .stat-val {{ font-size: 14px; font-weight: 600; }}
        
        .sparkline-container {{ flex-shrink: 0; }}
        
        .minute-breakdown {{ 
            font-size: 11px; color: #64748b; line-height: 1.8;
            font-family: "Cascadia Code", "Fira Code", monospace;
            background: rgba(15,23,42,0.5); border-radius: 6px; padding: 8px;
        }}
        
        .footer {{ 
            margin-top: 32px; padding-top: 16px; border-top: 1px solid #334155;
            color: #64748b; font-size: 12px; text-align: center;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>微结构信号回测报告</h1>
    <p class="subtitle">
        标的: BTC/USDT (Gate Spot) | 数据源: L2 订单簿 | 时间框架: 1分钟 |
        信号模块: 微结构_信号模块.py<br>
        数据范围: {data_start} ~ {data_end} | 共 {candles_count:,} 根K线
    </p>
    
    <h2>概览</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="label">信号触发次数</div>
            <div class="value">{len(signals)}</div>
            <div class="sub">占比 {len(signals)/candles_count*100:.2f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">后续上涨</div>
            <div class="value" style="color:#ef4444;">{up_count}</div>
            <div class="sub">{up_count/len(signals)*100:.0f}% (15分钟后) </div>
        </div>
        <div class="summary-card">
            <div class="label">后续下跌</div>
            <div class="value" style="color:#22c55e;">{down_count}</div>
            <div class="sub">{down_count/len(signals)*100:.0f}% (15分钟后)</div>
        </div>
        <div class="summary-card">
            <div class="label">平均最大涨幅</div>
            <div class="value" style="color:#ef4444;">+{avg_gain:.3f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">平均最大跌幅</div>
            <div class="value" style="color:#22c55e;">{avg_loss:.3f}%</div>
        </div>
        <div class="summary-card">
            <div class="label">平均净变化</div>
            <div class="value" style="color:{'#ef4444' if avg_net>0 else '#22c55e'};">{avg_net:+.3f}%</div>
            <div class="sub">15分钟后</div>
        </div>
    </div>
    
    <h2>信号通道统计</h2>
    <table class="stats-table">
        <thead>
            <tr><th>通道</th><th>均值</th><th>最大值</th><th>最小值</th><th>标准差</th></tr>
        </thead>
        <tbody>{ch_stats}</tbody>
    </table>
    
    <h2>信号详情（共 {len(signals)} 个）</h2>
    {signal_rows_html if signal_rows_html else '<p style="color:#94a3b8;padding:24px;text-align:center;">无信号触发</p>'}
    
    <div class="footer">
        微结构信号回测报告 | 基于论文 arXiv:2604.20949 | 
        生成时间: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}
    </div>
</div>
</body>
</html>"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  报告已保存: {output_path}")


def _generate_sparkline_svg(minute_prices, signal_price):
    """生成15分钟价格走势的 SVG sparkline。"""
    if not minute_prices:
        return ""
    
    width = 280
    height = 60
    padding = 8
    
    prices = [mp["price"] for mp in minute_prices]
    changes = [mp["change_pct"] for mp in minute_prices]
    
    min_change = min(min(changes), -0.01)
    max_change = max(max(changes), 0.01)
    range_val = max_change - min_change
    if range_val < 0.001:
        range_val = 0.01
    
    n = len(prices)
    x_step = (width - 2 * padding) / max(n - 1, 1)
    
    points = []
    for i, ch in enumerate(changes):
        x = padding + i * x_step
        y = height - padding - ((ch - min_change) / range_val) * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")
    
    # 基准线（0%）
    zero_y = height - padding - ((0 - min_change) / range_val) * (height - 2 * padding)
    
    # 颜色：上涨红，下跌绿
    final_change = changes[-1] if changes else 0
    color = "#ef4444" if final_change > 0 else "#22c55e" if final_change < 0 else "#94a3b8"
    
    poly_points = " ".join(points)
    area_points = f"{padding},{zero_y:.1f} {poly_points} {padding + (n-1)*x_step:.1f},{zero_y:.1f}"
    
    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="sparkGrad{i}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.3"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
            </linearGradient>
        </defs>
        <line x1="{padding}" y1="{zero_y:.1f}" x2="{width-padding}" y2="{zero_y:.1f}" 
              stroke="#475569" stroke-width="0.5" stroke-dasharray="2,2"/>
        <polygon points="{area_points}" fill="url(#sparkGrad{i})"/>
        <polyline points="{poly_points}" fill="none" stroke="{color}" stroke-width="1.5"/>
        <text x="{width-padding}" y="{padding+4}" text-anchor="end" 
              fill="{color}" font-size="10" font-family="monospace">
            {final_change:+.3f}%
        </text>
    </svg>"""
    
    return svg


# ═════════════════════════════════════════════════════════════
#  主流程
# ═════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  微结构信号模块回测")
    print("  数据: Gate L2 订单簿 / BTC_USDT")
    print("=" * 60)
    
    # 1. 加载文件
    print("\n[1a] 加载 parquet 文件...")
    files = load_all_parquet_files()
    if not files:
        print("错误: 无有效数据文件")
        return
    
    # 1b. 重建订单簿
    # 缓存机制：如果已有缓存的 pickle，直接加载
    cache_path = PROJECT_ROOT / "user_data/backtest_results/lob_candles_cache.pkl"
    if cache_path.exists():
        print(f"\n[1b] 从缓存加载K线数据: {cache_path}")
        candles_df = pd.read_pickle(cache_path)
        print(f"  加载 {len(candles_df)} 根K线")
    else:
        print("\n[1b] 重建订单簿并构建1分钟K线...")
        minute_records = reconstruct_orderbook(files)
        if not minute_records:
            print("错误: 未能构建K线数据")
            return

        # 构建 DataFrame
        candles_df = pd.DataFrame(minute_records)
        candles_df = candles_df.sort_values("date").reset_index(drop=True)

        # 填充缺失分钟（如果时间不连续，用前值填充）
        candles_df = candles_df.set_index("date")
        full_range = pd.date_range(candles_df.index.min(), candles_df.index.max(), freq="1min")
        candles_df = candles_df.reindex(full_range)
        # 前向填充 OHLCV
        for col in ["open", "high", "low", "close", "volume", "buy_volume", "sell_volume",
                     "buy_volume_ratio", "bid_updates", "ask_updates", "best_bid", "best_ask",
                     "mid_price", "bid_depth_10", "ask_depth_10"]:
            if col in candles_df.columns:
                candles_df[col] = candles_df[col].ffill()
        candles_df = candles_df.reset_index().rename(columns={"index": "date"})
        candles_df["trade_count"] = candles_df["trade_count"].fillna(0).astype(int)

        # 保存缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        candles_df.to_pickle(cache_path)
        print(f"  缓存已保存: {cache_path}")
    
    print(f"  K线总数(含填充): {len(candles_df)}")
    print(f"  时间范围: {candles_df['date'].min()} ~ {candles_df['date'].max()}")
    print(f"  价格范围: ${candles_df['close'].min():,.2f} ~ ${candles_df['close'].max():,.2f}")
    
    # 2. 生成信号
    df = generate_signals(candles_df)
    
    # 3. 分析信号后行情
    signals = analyze_forward_price_action(df, forward_minutes=15)
    
    # 4. 生成报告
    output_path = OUTPUT_DIR / "微结构信号回测报告.html"
    generate_html_report(df, signals, len(df), output_path)
    
    # 5. 导出 CSV
    if signals:
        csv_data = []
        for s in signals:
            row = {
                "信号时间": s["signal_time"].strftime("%Y-%m-%d %H:%M UTC"),
                "信号价格": s["signal_price"],
                "CH1_波动率熵": round(s["ch1"], 6),
                "CH2_深度侵蚀": round(s["ch2"], 6),
                "CH3_价差漂移": round(s["ch3"], 6),
                "CH4_订单流": round(s["ch4"], 6),
                "综合信号": round(s["composite"], 6),
                "自适应阈值": round(s["threshold"], 6),
                "CH4方向": s["ch4_direction"],
                "最大涨幅%": round(s["max_gain_pct"], 4),
                "最大跌幅%": round(s["max_loss_pct"], 4),
                "15分钟净变化%": round(s["net_change_pct"], 4),
                "最高价": s["max_high"],
                "最低价": s["min_low"],
                "15分钟后收盘": s["final_close"],
                "方向": s["direction"],
            }
            for mp in s["minute_prices"]:
                row[f"+{mp['minute']}m价格"] = round(mp["price"], 2)
                row[f"+{mp['minute']}m变化%"] = round(mp["change_pct"], 4)
            csv_data.append(row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = OUTPUT_DIR / "微结构信号回测数据.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  CSV 已保存: {csv_path}")
    
    print("\n" + "=" * 60)
    print("  回测完成!")
    print(f"  报告: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
