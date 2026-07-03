"""
Gate.io 订单簿数据格式化工具
=============================

将原始 L1（最优买卖价）+ L2（快照+增量深度）数据规整为统一格式，
用于马丁策略回测，重点关注插针/爆仓场景。

L2 数据特点：
- 每小时文件第一条记录是 50 档快照
- 后续记录是增量 diff：[["price", "amount"], ...]，amount=0 表示该价位撤销

输出：
  formatted/{pair}/
    orderbook_states.parquet   - 每次 L2 更新时的订单簿状态（流式写入）
    ohlcv_1s.parquet           - 1秒 OHLCV K线 + 深度特征
    wick_events.csv            - 插针/爆仓事件目录
    wick_depth_snapshots.parquet - 插针前后深度快照
"""

import os
import sys
import time
import datetime
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Optional

import orjson
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# 配置
# ============================================================

DATA_ROOT = Path("user_data/orderbook_data/gate/spot")
OUTPUT_ROOT = Path("user_data/orderbook_data/formatted")
PAIRS = ["BTC_USDT", "ETH_USDT"]
DEPTH_LEVELS_PCT = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
WICK_THRESHOLD_PCT = 0.5
WICK_WINDOW_SEC = 60
WICK_REVERSAL_PCT = 0.3
MARTINGALE_LEVERAGES = [3, 5, 10, 20, 50, 100]
DEPTH_SAMPLE_INTERVAL_MS = 200  # 每 200ms 计算一次完整深度
CHUNK_SIZE_FILES = 15  # 每批次处理的文件数


# ============================================================
# L2 订单簿重建器
# ============================================================

class OrderBookReconstructor:
    """从快照+增量更新重建完整订单簿"""

    def __init__(self):
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_id: int = 0
        self.is_initialized: bool = False

    def apply_snapshot(self, bid_updates, ask_updates):
        """应用全量快照"""
        self.bids.clear()
        self.asks.clear()
        for price_str, amount_str in bid_updates:
            amount = float(amount_str)
            if amount > 0:
                self.bids[float(price_str)] = amount
        for price_str, amount_str in ask_updates:
            amount = float(amount_str)
            if amount > 0:
                self.asks[float(price_str)] = amount
        self.is_initialized = True

    def apply_incremental(self, bid_updates, ask_updates):
        """应用增量更新"""
        for price_str, amount_str in bid_updates:
            price = float(price_str)
            amount = float(amount_str)
            if amount <= 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = amount
        for price_str, amount_str in ask_updates:
            price = float(price_str)
            amount = float(amount_str)
            if amount <= 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = amount

    def apply_update(self, row):
        """应用一行 L2 数据，自动判断快照/增量"""
        bid_updates = row["bid_updates"]
        ask_updates = row["ask_updates"]
        if isinstance(bid_updates, str):
            bid_updates = ororjson.loads(bid_updates)
        if isinstance(ask_updates, str):
            ask_updates = orjson.loads(ask_updates)

        bid_count = row["bid_update_count"]
        ask_count = row["ask_update_count"]
        is_snapshot = (bid_count == 50 and ask_count == 50 and
                       (not self.is_initialized or bid_count > 40))

        if is_snapshot:
            self.apply_snapshot(bid_updates, ask_updates)
        else:
            self.apply_incremental(bid_updates, ask_updates)

        self.last_update_id = row["update_id"]

    def apply_update_arrays(self, bid_updates, ask_updates, bid_count, ask_count, update_id):
        """快速版本：接收已解析的数组，避免 JSON 解析开销"""
        is_snapshot = (bid_count == 50 and ask_count == 50 and
                       (not self.is_initialized or bid_count > 40))

        if is_snapshot:
            self.apply_snapshot(bid_updates, ask_updates)
        else:
            self.apply_incremental(bid_updates, ask_updates)

        self.last_update_id = update_id

    def get_best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        bb, ba = self.get_best_bid(), self.get_best_ask()
        return (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

    def get_spread(self) -> Optional[float]:
        bb, ba = self.get_best_bid(), self.get_best_ask()
        return ba - bb if (bb is not None and ba is not None) else None

    def get_depth_at_level(self, side: str, mid_price: float, pct: float) -> float:
        """计算 side 在 mid ± pct% 范围内的累计深度（quote 计价）"""
        if side == "bid":
            target = mid_price * (1 - pct / 100)
            return sum(p * a for p, a in self.bids.items() if p >= target)
        else:
            target = mid_price * (1 + pct / 100)
            return sum(p * a for p, a in self.asks.items() if p <= target)

    def get_state_light(self) -> dict:
        """获取轻量状态（无深度计算，极快）"""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is None or ba is None:
            return None
        return {
            "best_bid": bb,
            "best_ask": ba,
            "mid_price": (bb + ba) / 2.0,
            "spread": ba - bb,
            "n_bid_levels": len(self.bids),
            "n_ask_levels": len(self.asks),
        }

    def get_state_full(self) -> dict:
        """获取完整状态（含深度、挂单墙、失衡度）"""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb is None or ba is None:
            return None

        mid = (bb + ba) / 2.0
        state = {
            "best_bid": bb,
            "best_ask": ba,
            "mid_price": mid,
            "spread": ba - bb,
            "n_bid_levels": len(self.bids),
            "n_ask_levels": len(self.asks),
        }

        # 深度计算（只在必要时进行）
        for pct in DEPTH_LEVELS_PCT:
            key = str(pct).replace(".", "_")
            state[f"bid_depth_{key}pct"] = self.get_depth_at_level("bid", mid, pct)
            state[f"ask_depth_{key}pct"] = self.get_depth_at_level("ask", mid, pct)

        # 挂单墙
        if self.bids:
            max_bid = max(self.bids.items(), key=lambda x: x[0] * x[1])
            state["bid_wall_price"] = max_bid[0]
            state["bid_wall_amount"] = max_bid[0] * max_bid[1]
        else:
            state["bid_wall_price"] = np.nan
            state["bid_wall_amount"] = np.nan

        if self.asks:
            max_ask = max(self.asks.items(), key=lambda x: x[0] * x[1])
            state["ask_wall_price"] = max_ask[0]
            state["ask_wall_amount"] = max_ask[0] * max_ask[1]
        else:
            state["ask_wall_price"] = np.nan
            state["ask_wall_amount"] = np.nan

        bd1 = state.get("bid_depth_1_0pct", 0) or 0
        ad1 = state.get("ask_depth_1_0pct", 0) or 0
        total = bd1 + ad1
        state["depth_imbalance"] = (bd1 - ad1) / total if total > 0 else 0

        return state

    def get_state(self, mid_price: Optional[float] = None) -> dict:
        """兼容旧接口，默认调用 get_state_light"""
        return self.get_state_light()


# ============================================================
# 插针/爆仓检测器
# ============================================================

class WickDetector:
    """基于 1秒 OHLCV 检测插针事件"""

    def __init__(self, threshold_pct=WICK_THRESHOLD_PCT,
                 window_sec=WICK_WINDOW_SEC, reversal_pct=WICK_REVERSAL_PCT):
        self.threshold_pct = threshold_pct
        self.window_sec = window_sec
        self.reversal_pct = reversal_pct

    def detect(self, ohlcv: pd.DataFrame, pair_name="") -> pd.DataFrame:
        """
        从 OHLCV 检测插针。

        逻辑：
        - 滑动窗口内 high/low 偏离 mid > threshold%
        - 收盘价回撤超过波动幅度的 reversal_pct
        """
        events = []
        ws = int(self.window_sec)
        o = ohlcv["open"].values
        h = ohlcv["high"].values
        l = ohlcv["low"].values
        c = ohlcv["close"].values
        idx = ohlcv.index

        for i in range(ws, len(ohlcv)):
            w_high = h[i - ws : i + 1].max()
            w_low = l[i - ws : i + 1].min()
            w_open = o[i - ws]
            w_close = c[i]
            mid_ref = (w_open + w_close) / 2.0
            if mid_ref <= 0:
                continue

            high_pct = (w_high - mid_ref) / mid_ref * 100
            low_pct = (mid_ref - w_low) / mid_ref * 100

            # 上插针
            if high_pct > self.threshold_pct and w_high > mid_ref:
                rev = (w_high - w_close) / (w_high - mid_ref)
                if rev > self.reversal_pct:
                    events.append({
                        "pair": pair_name, "timestamp": idx[i],
                        "event_type": "up_wick",
                        "magnitude_pct": round(high_pct, 4),
                        "reversal_ratio": round(rev, 4),
                        "window_open": w_open, "window_high": w_high,
                        "window_low": w_low, "window_close": w_close,
                    })

            # 下插针
            if low_pct > self.threshold_pct and mid_ref > w_low:
                rev = (w_close - w_low) / (mid_ref - w_low)
                if rev > self.reversal_pct:
                    events.append({
                        "pair": pair_name, "timestamp": idx[i],
                        "event_type": "down_wick",
                        "magnitude_pct": round(low_pct, 4),
                        "reversal_ratio": round(rev, 4),
                        "window_open": w_open, "window_high": w_high,
                        "window_low": w_low, "window_close": w_close,
                    })

        return pd.DataFrame(events)


# ============================================================
# 文件收集与验证
# ============================================================

def collect_files(data_root, pair, level):
    """收集指定 pair/level 的所有 parquet 文件，按路径排序"""
    path = data_root / pair / level
    files = []
    for root, dirs, fs in os.walk(str(path)):
        for f in fs:
            if f.endswith(".parquet"):
                files.append(Path(root) / f)
    files.sort()
    return files


def is_valid_parquet(fpath):
    """检查 parquet 文件是否可读"""
    try:
        pf = pq.ParquetFile(fpath)
        return pf.metadata.num_rows > 0
    except Exception:
        return False


# ============================================================
# 流式 L2 订单簿重建 + 状态写入
# ============================================================

def reconstruct_l2_streaming(l2_files, output_path, pair_name):
    """
    流式重建 L2 订单簿并写入 parquet。
    使用预解析 JSON + 数组访问优化性能（~1.8M rows/s）。
    """
    print(f"  [{pair_name}] L2 files: {len(l2_files)}")

    chunk_dir = output_path.parent / f"_tmp_{pair_name}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    ob = OrderBookReconstructor()
    states_batch = []
    chunk_idx = 0
    total_rows = 0
    skipped_rows = 0
    t_start = time.time()

    for fi, fpath in enumerate(l2_files):
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            print(f"    Skip corrupted: {fpath.name}")
            skipped_rows += 1
            continue

        n = len(df)

        # 预解析 JSON 列（矢量化，快）
        if isinstance(df["bid_updates"].iloc[0], str):
            bid_parsed = df["bid_updates"].apply(orjson.loads).values
        else:
            bid_parsed = df["bid_updates"].values
        if isinstance(df["ask_updates"].iloc[0], str):
            ask_parsed = df["ask_updates"].apply(orjson.loads).values
        else:
            ask_parsed = df["ask_updates"].values

        bid_counts = df["bid_update_count"].values
        ask_counts = df["ask_update_count"].values
        update_ids = df["update_id"].values
        timestamps = df["exchange_time_ms"].values

        # 数组遍历（比 iterrows 快 200x）
        last_full_ms = 0
        for i in range(n):
            try:
                ob.apply_update_arrays(
                    bid_parsed[i], ask_parsed[i],
                    int(bid_counts[i]), int(ask_counts[i]),
                    int(update_ids[i])
                )
                ts = int(timestamps[i])

                # 大部分 tick 只取轻量状态，每 DEPTH_SAMPLE_INTERVAL_MS 取一次完整深度
                if ts - last_full_ms >= DEPTH_SAMPLE_INTERVAL_MS:
                    state = ob.get_state_full()
                    last_full_ms = ts
                else:
                    state = ob.get_state_light()

                if state is not None:
                    state["timestamp_ms"] = ts
                    state["update_id"] = int(update_ids[i])
                    states_batch.append(state)
                    total_rows += 1
            except Exception:
                skipped_rows += 1

        # 每 N 个文件刷盘
        if (fi + 1) % CHUNK_SIZE_FILES == 0 and states_batch:
            _flush_chunk(states_batch, chunk_dir, chunk_idx)
            states_batch = []
            chunk_idx += 1
            gc.collect()

            elapsed = time.time() - t_start
            print(f"    [{pair_name}] {fi+1}/{len(l2_files)} files, {total_rows:,} rows, "
                  f"{elapsed:.0f}s, ~{total_rows/max(elapsed,1):.0f} rows/s")

    # 最后一批
    if states_batch:
        _flush_chunk(states_batch, chunk_dir, chunk_idx)
        chunk_idx += 1
        states_batch = None
        gc.collect()

    elapsed = time.time() - t_start
    print(f"    [{pair_name}] Done L2: {total_rows:,} rows in {elapsed:.0f}s "
          f"({total_rows/max(elapsed,1):.0f} rows/s)")

    # 合并 chunk
    print(f"    [{pair_name}] Merging {chunk_idx} chunks...")
    _merge_chunks(chunk_dir, output_path)

    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    return total_rows


def _flush_chunk(states, chunk_dir, chunk_idx):
    """将状态列表写入 parquet chunk"""
    df = pd.DataFrame(states)
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
    df.to_parquet(chunk_path, index=False)


def _merge_chunks(chunk_dir, output_path):
    """合并所有临时 chunk 到最终文件"""
    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
    tables = []
    for cf in chunk_files:
        tables.append(pq.read_table(cf))
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_path, compression="snappy")
    print(f"    Merged {len(chunk_files)} chunks -> {output_path}")


# ============================================================
# OHLCV + 深度特征生成
# ============================================================

def generate_ohlcv(states_path, output_path, pair_name):
    """
    从 orderbook_states.parquet 生成 1秒 OHLCV + 深度特征。

    注意：orderbook_states 中只有部分行有深度数据（每 200ms 采样一次），
    OHLCV 从 mid_price 列生成，深度特征从有深度数据的行中 ffill。
    """
    print(f"  [{pair_name}] Generating 1s OHLCV from {states_path.name}...")

    # 一次性读取（orderbook_states 已去重，行数可控）
    df = pd.read_parquet(states_path)
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df.set_index("datetime", inplace=True)

    # OHLCV from mid_price
    ohlcv = df["mid_price"].resample("1s").agg(["first", "max", "min", "last"])
    ohlcv.columns = ["open", "high", "low", "close"]
    ohlcv["volume_ticks"] = df["mid_price"].resample("1s").count()

    # 基础字段
    for col in ["spread", "best_bid", "best_ask", "n_bid_levels", "n_ask_levels"]:
        if col in df.columns:
            ohlcv[col] = df[col].resample("1s").last()

    # 深度字段：只在有深度的行上操作
    depth_cols_present = [c for c in df.columns
                          if c.startswith(("bid_depth_", "ask_depth_", "depth_",
                                            "bid_wall_", "ask_wall_"))]
    if depth_cols_present:
        # 找到有深度数据的行
        depth_mask = df[depth_cols_present].notna().any(axis=1)
        depth_df = df.loc[depth_mask, depth_cols_present]
        if len(depth_df) > 0:
            depth_1s = depth_df.resample("1s").last()
            depth_1s = depth_1s.ffill()  # 填充没有深度数据的秒
            ohlcv = pd.concat([ohlcv, depth_1s], axis=1)

    # 衍生特征
    max_oc = ohlcv[["open", "close"]].max(axis=1)
    min_oc = ohlcv[["open", "close"]].min(axis=1)
    ohlcv["wick_up_pct"] = np.where(ohlcv["close"] > 0,
                                     (ohlcv["high"] - max_oc) / ohlcv["close"] * 100, 0)
    ohlcv["wick_down_pct"] = np.where(ohlcv["close"] > 0,
                                       (min_oc - ohlcv["low"]) / ohlcv["close"] * 100, 0)
    ohlcv["body_pct"] = np.where(ohlcv["close"] > 0,
                                  abs(ohlcv["close"] - ohlcv["open"]) / ohlcv["close"] * 100, 0)
    ohlcv["returns"] = ohlcv["close"].pct_change() * 100
    ohlcv["volatility_60s"] = ohlcv["returns"].rolling(60).std()

    for lev in MARTINGALE_LEVERAGES:
        ohlcv[f"liq_price_move_{lev}x_pct"] = 100.0 / lev

    ohlcv.to_parquet(output_path)
    print(f"    [{pair_name}] Saved: {output_path} ({len(ohlcv):,} rows)")
    return ohlcv


# ============================================================
# 插针检测 + 快照生成
# ============================================================

def detect_wicks_and_snapshots(ohlcv, pair_output, pair_name):
    """检测插针事件并生成前后深度快照"""
    print(f"  [{pair_name}] Detecting wick events...")

    detector = WickDetector()
    df_events = detector.detect(ohlcv.dropna(subset=["open", "high", "low", "close"]), pair_name)

    events_path = pair_output / "wick_events.csv"
    df_events.to_csv(events_path, index=False)
    print(f"    [{pair_name}] Detected {len(df_events)} wick events -> {events_path}")

    if len(df_events) > 0:
        up_n = (df_events["event_type"] == "up_wick").sum()
        down_n = (df_events["event_type"] == "down_wick").sum()
        print(f"    Up wicks: {up_n}, Down wicks: {down_n}")
        if len(df_events) > 0:
            print(f"    Max magnitude: {df_events['magnitude_pct'].max():.2f}%")

    # 插针前后深度快照
    if len(df_events) > 0:
        print(f"    [{pair_name}] Generating wick depth snapshots...")
        snapshots = []
        for _, event in df_events.iterrows():
            event_ts = event["timestamp"]
            ws = event_ts - pd.Timedelta(seconds=30)
            we = event_ts + pd.Timedelta(seconds=30)
            window = ohlcv.loc[ws:we].copy()
            if len(window) == 0:
                continue
            window["event_type"] = event["event_type"]
            window["event_magnitude_pct"] = event["magnitude_pct"]
            window["seconds_from_event"] = (window.index - event_ts).total_seconds()
            snapshots.append(window)

        if snapshots:
            df_snap = pd.concat(snapshots)
            snap_path = pair_output / "wick_depth_snapshots.parquet"
            df_snap.to_parquet(snap_path)
            print(f"    [{pair_name}] Saved: {snap_path} ({len(df_snap):,} rows)")

    return df_events


# ============================================================
# 主流程
# ============================================================

def process_pair(pair, skip_existing=False):
    """处理单个交易对的完整数据流程"""
    pair_output = OUTPUT_ROOT / pair
    states_path = pair_output / "orderbook_states.parquet"
    ohlcv_path = pair_output / "ohlcv_1s.parquet"

    if skip_existing and states_path.exists() and ohlcv_path.exists():
        print(f"  [{pair}] Skipping (outputs exist)")
        return None

    pair_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Processing: {pair}")
    print(f"{'='*60}")

    # Step 1: 收集文件
    l1_files = collect_files(DATA_ROOT, pair, "l1")
    l2_files_raw = collect_files(DATA_ROOT, pair, "l2")
    print(f"  [{pair}] Found {len(l1_files)} L1 files, {len(l2_files_raw)} L2 files")

    # 过滤损坏文件
    l2_files = [f for f in l2_files_raw if is_valid_parquet(f)]
    n_bad = len(l2_files_raw) - len(l2_files)
    if n_bad > 0:
        print(f"  [{pair}] {n_bad} corrupted L2 files skipped")

    # Step 2: L2 订单簿重建 + 流式写入
    print(f"\n  --- Step 1/3: Reconstructing L2 order book ---")
    n_states = reconstruct_l2_streaming(l2_files, states_path, pair)

    # Step 3: 生成 1秒 OHLCV
    print(f"\n  --- Step 2/3: Generating 1s OHLCV ---")
    ohlcv = generate_ohlcv(states_path, ohlcv_path, pair)

    # Step 4: 插针检测
    print(f"\n  --- Step 3/3: Wick detection ---")
    df_events = detect_wicks_and_snapshots(ohlcv, pair_output, pair)

    result = {
        "pair": pair,
        "orderbook_states": n_states,
        "ohlcv_rows": len(ohlcv),
        "wick_events": len(df_events),
    }

    print(f"\n  [{pair}] Complete!")
    print(f"    States: {n_states:,} | OHLCV: {len(ohlcv):,} rows | Wicks: {len(df_events)}")
    return result


def generate_report(results, output_root):
    """生成格式化摘要报告"""
    report_path = output_root / "format_summary.txt"
    lines = [
        "=" * 60,
        "Gate.io 订单簿数据格式化报告",
        "=" * 60,
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for r in results:
        if r is None:
            continue
        lines.append(f"--- {r['pair']} ---")
        lines.append(f"  订单簿状态记录: {r['orderbook_states']:,}")
        lines.append(f"  1秒 OHLCV 行数: {r['ohlcv_rows']:,}")
        lines.append(f"  插针事件:       {r['wick_events']:,}")
        lines.append("")

    lines += [
        "=" * 60,
        "输出文件:",
        "=" * 60,
        "  formatted/{pair}/",
        "    orderbook_states.parquet   - 订单簿状态（每次L2更新）",
        "    ohlcv_1s.parquet           - 1秒K线 + 深度特征",
        "    wick_events.csv            - 插针事件目录",
        "    wick_depth_snapshots.parquet - 插针前后深度快照",
        "",
        "=" * 60,
        "关键字段:",
        "=" * 60,
        "  mid_price, spread, best_bid/ask",
        "  bid_depth_Xpct / ask_depth_Xpct  - mid±X% 累计深度(USDT)",
        "  depth_imbalance  - 买卖失衡度(-1~1)",
        "  wick_up_pct / wick_down_pct  - 1秒K线影线比例",
        "  volatility_60s  - 60秒滚动波动率",
        "  liq_safe_Xx  - 深度是否覆盖X倍杠杆爆仓",
    ]
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report: {report_path}")


def main():
    global WICK_THRESHOLD_PCT, WICK_WINDOW_SEC

    parser = argparse.ArgumentParser(description="Gate.io Order Book Formatter")
    parser.add_argument("--pair", type=str, default=None,
                        help="Specific pair (BTC_USDT or ETH_USDT)")
    parser.add_argument("--wick-threshold", type=float, default=WICK_THRESHOLD_PCT,
                        help=f"Wick threshold %% (default: {WICK_THRESHOLD_PCT})")
    parser.add_argument("--wick-window", type=int, default=WICK_WINDOW_SEC,
                        help=f"Wick window seconds (default: {WICK_WINDOW_SEC})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output exists")
    args = parser.parse_args()

    WICK_THRESHOLD_PCT = args.wick_threshold
    WICK_WINDOW_SEC = args.wick_window

    pairs_to_process = [args.pair] if args.pair else PAIRS

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()

    for pair in pairs_to_process:
        r = process_pair(pair, args.skip_existing)
        results.append(r)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All done in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    generate_report(results, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
