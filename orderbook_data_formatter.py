"""
Gate 订单簿数据规整与插针检测工具 (v3)
=============================================
简洁高效的实现:
  1. 从 L1 tick 生成多粒度 OHLCV
  2. 在 1min OHLCV 上检测插针(粗筛) → tick 级精确定位
  3. L2 快照+增量订单簿深度重建
  4. 生成滚动最大回撤/偏移指标(马丁策略核心关注)

L2 格式: 快照(first_update_id==update_id) + 增量 diff
"""

import argparse, json, os, sys, glob
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path("user_data/orderbook_data/gate/spot")
OUTPUT_ROOT = Path("user_data/orderbook_data/formatted")

# 插针参数 — 基于实际数据分布调整
WICK_MIN_PCT = 0.1       # 1min K线最小波幅阈值%
WICK_RETRACE = 0.40      # 回归比例
WICK_LOOKAHEAD = 5       # 峰值后查找回归的最大分钟数

PAIRS = ["BTC_USDT", "ETH_USDT"]
TIMEFRAMES = ["1s", "5s", "10s", "30s", "1min", "5min", "15min", "1h"]


# ============================================================
# 数据加载
# ============================================================

def load_l1_tick(pair: str) -> pd.DataFrame | None:
    """合并所有 L1 文件为 tick 级 DataFrame (datetime 索引)"""
    l1_dir = DATA_ROOT / pair / "l1"
    if not l1_dir.exists():
        return None

    files = sorted(l1_dir.rglob("*.parquet"))
    print(f"  L1: {len(files)} 文件")
    dfs, bad = [], 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            bad += 1
    if bad:
        print(f"  跳过 {bad} 损坏文件")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["dt"] = pd.to_datetime(df["exchange_time_ms"], unit="ms", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    print(f"  {len(df):,} ticks, {df.index.min()} ~ {df.index.max()}")
    return df


def load_l2_all(pair: str) -> pd.DataFrame | None:
    """合并所有 L2 文件"""
    l2_dir = DATA_ROOT / pair / "l2"
    if not l2_dir.exists():
        return None
    files = sorted(l2_dir.rglob("*.parquet"))
    print(f"  L2: {len(files)} 文件")
    dfs, bad = [], 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            bad += 1
    if bad:
        print(f"  跳过 {bad} 损坏文件")
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["dt"] = pd.to_datetime(df["exchange_time_ms"], unit="ms", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    df["is_snap"] = df["first_update_id"] == df["update_id"]
    n_snap = df["is_snap"].sum()
    print(f"  {len(df):,} 行 (快照:{n_snap:,} 增量:{len(df)-n_snap:,})")
    return df


# ============================================================
# OHLCV 生成
# ============================================================

def build_ohlcv(tick: pd.DataFrame, tf: str) -> pd.DataFrame:
    """tick → OHLCV"""
    r = tick["mid_price"].resample(tf).ohlc().dropna()
    ohlcv = pd.DataFrame(index=r.index)
    ohlcv["open"] = r["open"]
    ohlcv["high"] = r["high"]
    ohlcv["low"] = r["low"]
    ohlcv["close"] = r["close"]
    # 成交量
    if "bid_amount" in tick.columns and "ask_amount" in tick.columns:
        ohlcv["volume"] = (tick["bid_amount"] + tick["ask_amount"]).resample(tf).sum()
    # 买卖压力比
    if "bid_amount" in tick.columns:
        bv = tick["bid_amount"].resample(tf).sum()
        av = tick["ask_amount"].resample(tf).sum() if "ask_amount" in tick.columns else 0
        ohlcv["buy_ratio"] = (bv / (bv + av + 1e-10)).fillna(0.5)
    # 价差
    if "spread" in tick.columns:
        ohlcv["avg_spread"] = tick["spread"].resample(tf).mean()
    return ohlcv


# ============================================================
# 滚动偏移指标 (马丁策略核心)
# ============================================================

def build_excursion_metrics(tick: pd.DataFrame) -> pd.DataFrame:
    """
    计算滚动最大不利偏移指标。
    对马丁策略来说，这是最直接的风险度量:
      - 在给定窗口内，价格反向波动的最大幅度
      - 可以用来估算触发加仓/爆仓的概率
    """
    price = tick["mid_price"].values
    index = tick.index
    n = len(price)

    # 估算 1 分钟对应的 tick 数
    time_ms = tick["exchange_time_ms"].values
    avg_tick_ms = np.median(np.diff(time_ms))
    ticks_per_min = max(1, int(60000 / avg_tick_ms))
    ticks_per_5min = ticks_per_min * 5

    results = []
    # 每 30 秒采样一个点
    step = max(1, ticks_per_min // 2)
    for start in range(0, n - ticks_per_min, step):
        # 1 分钟窗口
        end_1m = min(n, start + ticks_per_min)
        window_p = price[start:end_1m]
        ref = price[start]

        upward_excursion = (np.max(window_p) - ref) / ref * 100
        downward_excursion = (ref - np.min(window_p)) / ref * 100

        # 5 分钟窗口
        end_5m = min(n, start + ticks_per_5min)
        window_p5 = price[start:end_5m]
        upward_5m = (np.max(window_p5) - ref) / ref * 100
        downward_5m = (ref - np.min(window_p5)) / ref * 100

        results.append({
            "dt": index[start],
            "price": price[start],
            "up_1m_pct": round(upward_excursion, 4),
            "down_1m_pct": round(downward_excursion, 4),
            "up_5m_pct": round(upward_5m, 4),
            "down_5m_pct": round(downward_5m, 4),
        })

    return pd.DataFrame(results).set_index("dt")


# ============================================================
# 插针检测 (1min OHLCV 粗筛 → tick 精确定位)
# ============================================================

def detect_wicks(
    tick: pd.DataFrame,
    min_pct: float = WICK_MIN_PCT,
    retrace_ratio: float = WICK_RETRACE,
    lookahead_min: int = WICK_LOOKAHEAD,
) -> pd.DataFrame:
    """
    插针检测:
      1. 在 1min OHLCV 上找波幅 > min_pct% 的 K 线 (粗筛)
      2. 根据 K 线形态判断方向:
         - 下影线长 (close > open) → 向下插针 (买方收复)
         - 上影线长 (close < open) → 向上插针 (卖方收复)
      3. 在 tick 数据中精确定位峰值和回归点
    """
    if len(tick) == 0:
        return pd.DataFrame()

    # 1. 1min OHLCV 粗筛
    ohlcv = build_ohlcv(tick, "1min")

    # 标记 wick 候选
    ohlcv["range_pct"] = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"] * 100
    ohlcv["upper_shadow"] = (ohlcv["high"] - ohlcv[["open", "close"]].max(axis=1)) / ohlcv["close"] * 100
    ohlcv["lower_shadow"] = (ohlcv[["open", "close"]].min(axis=1) - ohlcv["low"]) / ohlcv["close"] * 100

    candidates = ohlcv[ohlcv["range_pct"] > min_pct].copy()
    if len(candidates) == 0:
        print(f"  候选 wick: 0 (1min 波幅 > {min_pct}%)")
        return pd.DataFrame()

    # 方向判断
    candidates["direction"] = "up"   # 默认向上插针(上影线)
    # 下影线更长 → 向下插针
    down_mask = candidates["lower_shadow"] > candidates["upper_shadow"]
    candidates.loc[down_mask, "direction"] = "down"

    print(f"  候选 wick: {len(candidates)} (1min 波幅 > {min_pct}%)")

    # 2. Tick 级精确定位
    price = tick["mid_price"].values
    ask_p = tick["ask_price"].values if "ask_price" in tick.columns else None
    bid_p = tick["bid_price"].values if "bid_price" in tick.columns else None
    spread_p = tick["spread"].values if "spread" in tick.columns else None
    time_ms = tick["exchange_time_ms"].values
    index = tick.index

    avg_tick_ms = np.median(np.diff(time_ms))
    ticks_per_min = int(60000 / avg_tick_ms)

    events = []
    for _, row in candidates.iterrows():
        cand_dt = row.name
        direction = row["direction"]

        # Tick 索引范围: 当前分钟 ± 缓冲
        start_dt = cand_dt - pd.Timedelta(seconds=30)
        end_dt = cand_dt + pd.Timedelta(seconds=lookahead_min * 60 + 30)

        i_start = np.searchsorted(time_ms, int(start_dt.timestamp() * 1000))
        i_end = min(len(price) - 1, np.searchsorted(time_ms, int(end_dt.timestamp() * 1000)))

        if i_end <= i_start:
            continue

        window_p = price[i_start:i_end + 1]
        window_t = time_ms[i_start:i_end + 1]
        window_idx = index[i_start:i_end + 1]

        # 找基准 (前 30 秒中位数)
        mid = i_start + ticks_per_min // 4
        if mid > i_end:
            mid = i_start + 1
        baseline = np.median(price[i_start:mid])

        if direction == "up":
            peak_rel = np.argmax(window_p)
            peak_price = window_p[peak_rel]
            deviation = (peak_price - baseline) / baseline * 100
        else:
            peak_rel = np.argmin(window_p)
            peak_price = window_p[peak_rel]
            deviation = (baseline - peak_price) / baseline * 100

        if deviation < min_pct:
            continue

        peak_dt = window_idx[peak_rel]
        peak_t = window_t[peak_rel]

        # 找回归点
        deviation_abs = abs(peak_price - baseline)
        recovered = False
        recovery_dt = None
        recovery_price = None

        for j in range(peak_rel + 1, len(window_p)):
            remaining = abs(window_p[j] - baseline)
            if remaining < deviation_abs * retrace_ratio:
                recovered = True
                recovery_dt = window_idx[j]
                recovery_price = window_p[j]
                break

        # 峰值的 ask/bid/spread
        abs_peak = peak_rel + i_start
        events.append({
            "start_dt": cand_dt,
            "peak_dt": peak_dt,
            "recovery_dt": recovery_dt,
            "direction": direction,
            "baseline": round(float(baseline), 2),
            "start_price": round(float(window_p[0]), 2),
            "peak_price": round(float(peak_price), 2),
            "recovery_price": round(float(recovery_price), 2) if recovery_price else None,
            "deviation_pct": round(float(deviation), 4),
            "duration_to_peak_ms": int(peak_t - window_t[0]),
            "duration_to_recovery_ms": (
                int(recovery_dt.timestamp() * 1000 - peak_t) if recovery_dt else None
            ),
            "recovered": recovered,
            "peak_ask": round(float(ask_p[abs_peak]), 2) if ask_p is not None else None,
            "peak_bid": round(float(bid_p[abs_peak]), 2) if bid_p is not None else None,
            "peak_spread": round(float(spread_p[abs_peak]), 4) if spread_p is not None else None,
            "upper_shadow_pct": round(float(row["upper_shadow"]), 4),
            "lower_shadow_pct": round(float(row["lower_shadow"]), 4),
        })

    wicks = pd.DataFrame(events)
    if len(wicks) > 0:
        wicks = wicks.sort_values("start_dt").reset_index(drop=True)
        wicks["wick_id"] = range(1, len(wicks) + 1)
    return wicks


# ============================================================
# L2 订单簿重建 (快照+增量)
# ============================================================

def rebuild_orderbook(l2: pd.DataFrame, target_dt: pd.Timestamp) -> dict:
    """在 target_dt 重建订单簿深度"""
    if l2 is None or len(l2) == 0:
        return {}

    # 找最近快照
    snaps = l2[(l2["is_snap"]) & (l2["dt"] <= target_dt) & (l2["bid_update_count"] > 0)]
    if len(snaps) == 0:
        return {}
    snap_row = snaps.iloc[-1]

    try:
        bid_snap = json.loads(snap_row["bid_updates"])
        ask_snap = json.loads(snap_row["ask_updates"])
    except Exception:
        return {}

    book_b, book_a = {}, {}
    for p, a in bid_snap:
        if float(a) > 0:
            book_b[float(p)] = float(a)
    for p, a in ask_snap:
        if float(a) > 0:
            book_a[float(p)] = float(a)

    # 应用增量
    incrs = l2[(~l2["is_snap"]) & (l2["dt"] > snap_row["dt"]) & (l2["dt"] <= target_dt)]
    updates = 0
    for _, r in incrs.iterrows():
        try:
            for side, book in [("bid_updates", book_b), ("ask_updates", book_a)]:
                for p, a in json.loads(r[side]):
                    if float(a) == 0:
                        book.pop(float(p), None)
                    else:
                        book[float(p)] = float(a)
                    updates += 1
        except Exception:
            pass

    bids = sorted(book_b.items(), key=lambda x: -x[0])[:20]
    asks = sorted(book_a.items(), key=lambda x: x[0])[:20]

    # 深度指标
    b5 = sum(a for _, a in bids[:5])
    a5 = sum(a for _, a in asks[:5])
    b20 = sum(a for _, a in bids)
    a20 = sum(a for _, a in asks)
    imb = b5 / (b5 + a5 + 1e-10)

    s0 = (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else None

    return {
        "bid_depth_5": round(b5, 4), "ask_depth_5": round(a5, 4),
        "bid_depth_20": round(b20, 4), "ask_depth_20": round(a20, 4),
        "imbalance": round(imb, 4), "spread_bps": round(s0, 2) if s0 else None,
        "n_bids": len(bids), "n_asks": len(asks), "updates_applied": updates,
    }


# ============================================================
# 主流程
# ============================================================

def process(pair: str, min_pct: float, retrace: float, lookahead: int, skip_l2: bool):
    print(f"\n{'='*60}\n  {pair}\n{'='*60}")

    # 1. 加载
    tick = load_l1_tick(pair)
    if tick is None:
        return

    pstats = {
        "min": float(tick["mid_price"].min()),
        "max": float(tick["mid_price"].max()),
        "mean": float(tick["mid_price"].mean()),
        "std": float(tick["mid_price"].std()),
    }
    pstats["range_pct"] = round((pstats["max"] - pstats["min"]) / pstats["mean"] * 100, 2)

    l2 = None
    if not skip_l2:
        l2 = load_l2_all(pair)

    # 2. 检测插针
    print(f"\n[插针检测] 1min 波幅>{min_pct}% 回归>{retrace*100}%")
    wicks = detect_wicks(tick, min_pct, retrace, lookahead)

    if len(wicks) > 0:
        up_n = (wicks["direction"] == "up").sum()
        print(f"  结果: {len(wicks)} 个 (上{up_n}/下{len(wicks)-up_n}) "
              f"回归率={(wicks['recovered']==True).sum()/len(wicks)*100:.0f}%")
        print(f"  偏离: 中位={wicks['deviation_pct'].median():.3f}% "
              f"最大={wicks['deviation_pct'].max():.3f}%")
    else:
        print("  结果: 0 个")

    # 3. 深度重建
    depth_rows = []
    if l2 is not None and len(wicks) > 0:
        print(f"\n[深度重建] 对 {min(200, len(wicks))} 个 wick 重建深度...")
        top_w = wicks.nlargest(200, "deviation_pct")
        for i, (_, w) in enumerate(top_w.iterrows()):
            if i % 50 == 0:
                print(f"  {i}/{len(top_w)}")
            for label, dt in [("peak", w["peak_dt"]), ("start", w["start_dt"])]:
                ob = rebuild_orderbook(l2, dt)
                if ob:
                    depth_rows.append({
                        "wick_id": w["wick_id"], "point": label, "dt": str(dt),
                        "direction": w["direction"], "deviation_pct": w["deviation_pct"],
                        **ob,
                    })

    # 4. 滚动偏移指标
    print(f"\n[偏移指标] 计算滚动最大不利偏移...")
    exc = build_excursion_metrics(tick)
    print(f"  {len(exc)} 个采样点")
    exc_up = exc["up_1m_pct"]
    exc_dn = exc["down_1m_pct"]
    print(f"  1min 上涨偏移: P50={exc_up.median():.3f}% P95={exc_up.quantile(0.95):.3f}% Max={exc_up.max():.3f}%")
    print(f"  1min 下跌偏移: P50={exc_dn.median():.3f}% P95={exc_dn.quantile(0.95):.3f}% Max={exc_dn.max():.3f}%")

    # 5. OHLCV
    print(f"\n[OHLCV] 生成多粒度 K 线...")
    ohlcv_map = {}
    for tf in TIMEFRAMES:
        o = build_ohlcv(tick, tf)
        ohlcv_map[tf] = o
        print(f"  {tf:>6s}: {len(o):>8,} 根")

    # ── 保存 ──
    out = OUTPUT_ROOT / pair
    out.mkdir(parents=True, exist_ok=True)

    # tick
    tick_path = out / "tick_l1.parquet"
    tick.to_parquet(tick_path)
    print(f"\n[保存] tick_l1.parquet ({tick_path.stat().st_size/1024/1024:.1f}MB)")

    # wicks
    if len(wicks) > 0:
        wicks.to_parquet(out / "wicks.parquet")
        wc = wicks.copy()
        for c in wc.select_dtypes(include=["datetime64"]).columns:
            wc[c] = wc[c].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        wc.to_csv(out / "wicks.csv", index=False, encoding="utf-8-sig")
        print(f"[保存] wicks.parquet + wicks.csv ({len(wicks)}条)")

    # depth
    if depth_rows:
        dd = pd.DataFrame(depth_rows)
        dd.to_parquet(out / "wick_depth.parquet")
        dd.to_csv(out / "wick_depth.csv", index=False, encoding="utf-8-sig")
        print(f"[保存] wick_depth.parquet + wick_depth.csv ({len(dd)}条)")

    # excursion
    exc.to_parquet(out / "excursion_metrics.parquet")
    print(f"[保存] excursion_metrics.parquet ({len(exc)}采样点)")

    # ohlcv
    for tf, o in ohlcv_map.items():
        o.to_parquet(out / f"ohlcv_{tf}.parquet")

    # summary
    w_summary = {
        "pair": pair, "total_wicks": len(wicks),
        "up": int((wicks["direction"] == "up").sum()) if len(wicks) > 0 else 0,
        "down": int((wicks["direction"] == "down").sum()) if len(wicks) > 0 else 0,
        "recovered": int(wicks["recovered"].sum()) if len(wicks) > 0 else 0,
        "recovery_pct": round(wicks["recovered"].sum() / len(wicks) * 100, 1) if len(wicks) > 0 else 0,
        "deviation_median": round(float(wicks["deviation_pct"].median()), 4) if len(wicks) > 0 else 0,
        "deviation_max": round(float(wicks["deviation_pct"].max()), 4) if len(wicks) > 0 else 0,
        "excursion_1m_up_p95": round(float(exc_up.quantile(0.95)), 4),
        "excursion_1m_up_max": round(float(exc_up.max()), 4),
        "excursion_1m_dn_p95": round(float(exc_dn.quantile(0.95)), 4),
        "excursion_1m_dn_max": round(float(exc_dn.max()), 4),
        "price_stats": pstats,
        "time_range": {"start": str(tick.index.min()), "end": str(tick.index.max())},
        "params": {"min_pct": min_pct, "retrace": retrace, "lookahead_min": lookahead},
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(w_summary, f, indent=2, ensure_ascii=False, default=str)

    return w_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--min-pct", type=float, default=WICK_MIN_PCT)
    parser.add_argument("--retrace", type=float, default=WICK_RETRACE)
    parser.add_argument("--lookahead", type=int, default=WICK_LOOKAHEAD)
    parser.add_argument("--skip-l2", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gate 订单簿数据规整 v3")
    print("=" * 60)
    print(f"  源: {DATA_ROOT.resolve()}")
    print(f"  输出: {OUTPUT_ROOT.resolve()}")
    print(f"  插针: >{args.min_pct}% | 回归≤{args.retrace*100}% | 前瞻{args.lookahead}min")
    print(f"  交易对: {args.pair or '全部'}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = [args.pair] if args.pair else PAIRS
    summaries = []

    for p in pairs:
        s = process(p, args.min_pct, args.retrace, args.lookahead, args.skip_l2)
        if s:
            summaries.append(s)

    # 总汇
    master = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "pairs": summaries,
    }
    with open(OUTPUT_ROOT / "all_summary.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}\n  完成! {OUTPUT_ROOT.resolve()}\n{'='*60}")
    for s in summaries:
        print(f"\n  [{s['pair']}]")
        print(f"    插针: {s['total_wicks']} (上{s['up']}/下{s['down']}) 回归率:{s['recovery_pct']}%")
        print(f"    偏移(P95): 涨{s['excursion_1m_up_p95']:.3f}% 跌{s['excursion_1m_dn_p95']:.3f}%")
        ps = s["price_stats"]
        print(f"    价格: {ps['min']:.2f} - {ps['max']:.2f} ({ps['range_pct']}%)")


if __name__ == "__main__":
    main()
