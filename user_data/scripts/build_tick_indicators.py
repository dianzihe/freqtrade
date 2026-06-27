# -*- coding: utf-8 -*-
"""
从 Binance aggTrades 数据构建 5 分钟 OHLCV + 微结构指标

输入：user_data/tickdata/BTCUSDT-aggTrades-*.csv
输出：user_data/data/binance/BTC_USDT-5m.feather

指标计算（对应论文四个信号通道）：
  - trade_count: 每 5 分钟成交笔数
  - buy_volume_ratio: 买方成交量占比 → 通道 4：订单流（直接来自 B/S 标记）
  - price_std: 成交价格标准差 → 通道 3：价差代理（直接度量窗口内价格分散度）
  - trade_intensity: 每秒成交笔数 → 通道 2：深度侵蚀（活跃度下降 = 流动性撤出）
  - large_trade_ratio: 大单占比 → 通道 2 辅助（大单撤退 = 深度侵蚀先兆）

用法：
  python user_data/scripts/build_tick_indicators.py
  # 或指定月份：
  python user_data/scripts/build_tick_indicators.py --months 2026-05
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---- 配置 ----
TICK_DIR = Path("user_data/tickdata")
OUT_DIR = Path("user_data/data/binance")
OUTPUT_FILE = OUT_DIR / "BTC_USDT-5m.feather"
TIMEFRAME_MINUTES = 5
CHUNKSIZE = 3_000_000  # 每块 300 万行，约 250 MB
USD_MIN_THRESHOLD = 100  # 单笔成交低于此金额的视为碎股


def process_month(filepath: Path) -> pd.DataFrame:
    """
    处理单个月份的 aggTrades CSV。

    返回 DataFrame，包含 5 分钟 OHLCV + 微结构指标。
    使用分块读取 + 渐进聚合避免内存溢出。
    """
    print(f"  读取: {filepath.name} ({filepath.stat().st_size / 1e9:.1f} GB)")

    chunk_results = []
    total_rows = 0
    t_start = time.time()

    # ---- 第一轮：分块聚合 ----
    for chunk_idx, chunk in enumerate(
        pd.read_csv(filepath, chunksize=CHUNKSIZE, low_memory=False)
    ):
        total_rows += len(chunk)
        if chunk_idx % 5 == 0:
            print(f"    块 {chunk_idx}: {total_rows / 1e6:.1f}M 行已处理")

        # 解析时间戳（微秒 → datetime）
        chunk["dt"] = pd.to_datetime(chunk["datetime"], unit="us")
        # 向下取整到 5 分钟窗口
        chunk["window"] = chunk["dt"].dt.floor(f"{TIMEFRAME_MINUTES}min")

        # B/S 成交量分离
        chunk["is_buy"] = chunk["BS"] == "B"
        chunk["buy_vol"]  = np.where(chunk["is_buy"], chunk["volume"], 0.0)
        chunk["sell_vol"] = np.where(chunk["is_buy"], 0.0, chunk["volume"])

        # 美元成交额
        chunk["usd_vol"] = chunk["price"] * chunk["volume"]

        # 大单标记（单笔 ≥ 5000 USDT）
        chunk["is_large"] = chunk["usd_vol"] >= 5000
        chunk["large_vol"] = np.where(chunk["is_large"], chunk["volume"], 0.0)

        # 价格平方（用于计算方差）
        chunk["price_sq"] = chunk["price"] ** 2

        # ---- 向量化聚合 ----
        g = chunk.groupby("window", sort=False)

        agg = g.agg(
            # OHLCV 基础
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),

            # 微结构指标（可累加的聚合值）
            trade_count=("trade_id", "count"),
            buy_volume=("buy_vol", "sum"),
            sell_volume=("sell_vol", "sum"),
            usd_volume=("usd_vol", "sum"),
            large_volume=("large_vol", "sum"),

            # 价格统计辅助（用于跨块合并后精确计算方差）
            price_sum=("price", "sum"),
            price_sq_sum=("price_sq", "sum"),
            price_count=("price", "count"),
        ).reset_index()

        chunk_results.append(agg)

    t_read = time.time() - t_start
    print(f"    读取完成: {total_rows / 1e6:.1f}M 行, {t_read:.0f}s")

    if not chunk_results:
        print("    警告：无数据")
        return pd.DataFrame()

    # ---- 第二轮：跨块合并（同一 5 分钟窗口可能被分到两个块） ----
    monthly = pd.concat(chunk_results, ignore_index=True)

    final_g = monthly.groupby("window", sort=True)

    final = final_g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        usd_volume=("usd_volume", "sum"),
        large_volume=("large_volume", "sum"),
        price_sum=("price_sum", "sum"),
        price_sq_sum=("price_sq_sum", "sum"),
        price_count=("price_count", "sum"),
    ).reset_index()

    # ---- 第三轮：计算派生指标 ----
    total_vol = final["buy_volume"] + final["sell_volume"]

    # 通道 4：订单流 — 买方成交量占比（直接来自 B/S 标记，无代理误差）
    final["buy_volume_ratio"] = (final["buy_volume"] / total_vol.replace(0, np.nan))

    # 通道 3：价差代理 — 成交价格的标准差
    # Var(X) = E[X²] - E[X]²
    mean_price = final["price_sum"] / final["price_count"].replace(0, np.nan)
    mean_sq = final["price_sq_sum"] / final["price_count"].replace(0, np.nan)
    final["price_std"] = np.sqrt(np.maximum(0, mean_sq - mean_price**2))

    # 通道 2：深度侵蚀 — 成交活跃度（每秒笔数）
    final["trade_intensity"] = final["trade_count"] / (TIMEFRAME_MINUTES * 60)

    # 通道 2 辅助：大单占比（大单撤退 = 机构撤出 = 深度侵蚀先兆）
    final["large_trade_ratio"] = (final["large_volume"] / total_vol.replace(0, np.nan))

    # VWAP（加权均价）
    final["vwap"] = final["usd_volume"] / total_vol.replace(0, np.nan)

    # VWAP 偏差（当前收盘偏离 VWAP 的程度，方向性信号）
    final["vwap_deviation"] = (final["close"] - final["vwap"]) / final["vwap"].replace(0, np.nan)

    # ---- 格式化 ----
    # freqtrade 要求 date 列（带时区）
    final["date"] = final["window"].dt.tz_localize("UTC")
    final["date"] = final["date"].astype("datetime64[ms, UTC]")

    # 排序
    final = final.sort_values("date").reset_index(drop=True)

    # 选择最终列（OHLCV 标准列 + 微结构扩展列）
    output_cols = [
        "date", "open", "high", "low", "close", "volume",
        # 微结构扩展列（论文四通道的输入数据）
        "trade_count",           # 成交笔数
        "buy_volume_ratio",      # 买方占比（通道 4：订单流）
        "price_std",             # 价格标准差（通道 3：价差）
        "trade_intensity",       # 成交强度（通道 2：深度代理）
        "large_trade_ratio",     # 大单占比（通道 2 辅助）
        "vwap",                  # VWAP
        "vwap_deviation",        # VWAP 偏离
    ]
    final = final[output_cols]

    # 填充边界 NaN
    final = final.fillna(0.0)

    t_total = time.time() - t_start
    print(f"    产出 {len(final):,} 根 5m K 线, 耗时 {t_total:.0f}s")
    return final


def main():
    parser = argparse.ArgumentParser(description="从 aggTrades 构建 5m OHLCV + 微结构指标")
    parser.add_argument("--months", nargs="+", help="指定处理的月份，如 2026-05 2026-04")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="输出文件路径")
    args = parser.parse_args()

    # 收集需要处理的文件
    if args.months:
        files = [TICK_DIR / f"BTCUSDT-aggTrades-{m}.csv" for m in args.months]
    else:
        files = sorted(TICK_DIR.glob("BTCUSDT-aggTrades-*.csv"))

    existing = [f for f in files if f.exists()]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"警告：以下文件不存在: {[f.name for f in missing]}")

    if not existing:
        print("错误：没有可处理的文件")
        sys.exit(1)

    print(f"将处理 {len(existing)} 个文件，输入目录: {TICK_DIR}")
    print(f"输出: {args.output}")
    print()

    # 逐月处理
    all_data = []
    for f in existing:
        df = process_month(f)
        if len(df) > 0:
            all_data.append(df)

    if not all_data:
        print("错误：所有文件均无有效数据")
        sys.exit(1)

    # 合并所有月份
    print(f"\n合并 {len(all_data)} 个月数据...")
    combined = pd.concat(all_data, ignore_index=True)

    # 去重（月末/月初可能有重叠的 5 分钟窗口）
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)

    # 检查数据连续性
    expected_freq = pd.Timedelta(minutes=TIMEFRAME_MINUTES)
    gaps = combined["date"].diff()[1:] > expected_freq * 1.5
    if gaps.any():
        print(f"  发现 {gaps.sum()} 个时间缺口（> {TIMEFRAME_MINUTES * 1.5:.0f} 分钟）")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n保存到 {out_path}...")
    combined.to_feather(out_path)

    # 验证
    verify = pd.read_feather(out_path)
    print(f"\n===== 输出验证 =====")
    print(f"总 K 线数: {len(verify):,}")
    print(f"时间范围: {verify['date'].min()} ~ {verify['date'].max()}")
    print(f"列: {list(verify.columns)}")
    print(f"缺失值:\n{verify.isnull().sum()}")
    print(f"\n指标统计:")
    for col in ["buy_volume_ratio", "price_std", "trade_intensity", "large_trade_ratio"]:
        if col in verify.columns:
            print(f"  {col}: mean={verify[col].mean():.4f}, std={verify[col].std():.4f}, "
                  f"min={verify[col].min():.4f}, max={verify[col].max():.4f}")

    print(f"\n完成！文件: {out_path.resolve()}")
    print(f"文件大小: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
