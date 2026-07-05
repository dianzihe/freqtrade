"""
分析订单簿每秒深度消耗情况
读取 cex_tick_recorder 录制的 depth parquet 数据，计算每秒的买卖盘消耗量
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
CHINA_TZ = timezone(timedelta(hours=8))

def load_depth_data(pair: str, exchange: str) -> pd.DataFrame:
    """加载 depth parquet 数据"""
    depth_dir = Path(f"user_data/orderbook_data/{exchange}/spot/{pair}/depth")
    if not depth_dir.exists():
        raise FileNotFoundError(f"未找到 depth 数据: {depth_dir}")
    
    files = list(depth_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件: {depth_dir}")
    
    df = pd.read_parquet(depth_dir)
    print(f"  加载 {len(df)} 行 depth 数据（来自 {len(files)} 个文件）")
    return df

def compute_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """计算每秒深度消耗量"""
    # 按本地时间排序
    df = df.sort_values("local_time_ms").reset_index(drop=True)
    
    # 计算秒级时间戳
    df["second"] = (df["local_time_ms"] // 1000).astype(int)
    
    # 去重：同一秒内只保留最后一条（最新深度）
    df = df.drop_duplicates(subset=["second"], keep="last").reset_index(drop=True)
    
    # 计算消耗量（当前秒深度 - 上一秒深度）
    df["bid_depth_consumed"] = df["bid_depth"].shift(1) - df["bid_depth"]   # 正值=被消耗，负值=被补充
    df["ask_depth_consumed"] = df["ask_depth"].shift(1) - df["ask_depth"]
    df["total_depth_consumed"] = df["total_depth"].shift(1) - df["total_depth"]
    
    # 5bps 深度消耗
    df["bid_5bps_consumed"] = df["bid_depth_5bps"].shift(1) - df["bid_depth_5bps"]
    df["ask_5bps_consumed"] = df["ask_depth_5bps"].shift(1) - df["ask_depth_5bps"]
    
    # 时间字符串（本地时间）
    df["time_str"] = pd.to_datetime(df["second"], unit="s").dt.tz_localize(UTC).dt.tz_convert(CHINA_TZ).dt.strftime("%H:%M:%S")
    
    return df

def generate_report(df: pd.DataFrame, output_path: str = None):
    """生成深度消耗报告"""
    # 过滤掉 NaN（第一行）
    analysis = df.dropna(subset=["bid_depth_consumed"]).copy()
    
    if len(analysis) == 0:
        print("  ⚠️ 无有效消耗数据（至少需要 2 秒数据）")
        return
    
    print(f"\n{'='*80}")
    print(f"  订单簿深度消耗分析报告")
    print(f"{'='*80}")
    
    # 基本信息
    start_time = pd.to_datetime(analysis["second"].min(), unit="s").tz_localize(UTC).tz_convert(CHINA_TZ)
    end_time = pd.to_datetime(analysis["second"].max(), unit="s").tz_localize(UTC).tz_convert(CHINA_TZ)
    duration_sec = int(analysis["second"].max() - analysis["second"].min()) + 1
    
    print(f"\n📊 基本信息")
    print(f"  时间范围: {start_time.strftime('%H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')} (本地时间)")
    print(f"  数据时长: {duration_sec} 秒 ({duration_sec/60:.1f} 分钟)")
    print(f"  中价范围: {analysis['mid_price'].min():.2f} ~ {analysis['mid_price'].max():.2f} USDT")
    print(f"  平均价差: {analysis['spread'].mean():.2f} USDT ({analysis['spread'].mean()/analysis['mid_price'].mean()*10000:.1f} bps)")
    
    # 深度统计
    print(f"\n📊 深度统计（全档 20 档）")
    print(f"  买盘深度: 均值={analysis['bid_depth'].mean():.2f} BTC, 中位数={analysis['bid_depth'].median():.2f}, 最小={analysis['bid_depth'].min():.2f}, 最大={analysis['bid_depth'].max():.2f}")
    print(f"  卖盘深度: 均值={analysis['ask_depth'].mean():.2f} BTC, 中位数={analysis['ask_depth'].median():.2f}, 最小={analysis['ask_depth'].min():.2f}, 最大={analysis['ask_depth'].max():.2f}")
    print(f"  总深度:   均值={analysis['total_depth'].mean():.2f} BTC")
    
    print(f"\n📊 核心深度（距中价 5 bps 内）")
    print(f"  买盘 5bps: 均值={analysis['bid_depth_5bps'].mean():.4f} BTC, 中位数={analysis['bid_depth_5bps'].median():.4f}")
    print(f"  卖盘 5bps: 均值={analysis['ask_depth_5bps'].mean():.4f} BTC, 中位数={analysis['ask_depth_5bps'].median():.4f}")
    
    # 消耗量统计
    print(f"\n🔥 每秒深度消耗统计")
    buy_consumed = analysis["bid_depth_consumed"]
    sell_consumed = analysis["ask_depth_consumed"]
    
    print(f"  买盘消耗（被吃单消耗 = 正值）:")
    print(f"    平均消耗: {buy_consumed.mean():.4f} BTC/s")
    print(f"    最大消耗: {buy_consumed.max():.4f} BTC/s（最剧烈秒）")
    print(f"    最大补充: {buy_consumed.min():.4f} BTC/s（最大补充秒）")
    buy_consumed_pos = buy_consumed[buy_consumed > 0]
    if len(buy_consumed_pos) > 0:
        print(f"    消耗事件: {len(buy_consumed_pos)}/{len(analysis)} 秒（{len(buy_consumed_pos)/len(analysis)*100:.1f}%）")
        print(f"    消耗均值（仅消耗秒）: {buy_consumed_pos.mean():.4f} BTC/s")
    
    print(f"  卖盘消耗（被吃单消耗 = 正值）:")
    print(f"    平均消耗: {sell_consumed.mean():.4f} BTC/s")
    print(f"    最大消耗: {sell_consumed.max():.4f} BTC/s（最剧烈秒）")
    print(f"    最大补充: {sell_consumed.min():.4f} BTC/s（最大补充秒）")
    sell_consumed_pos = sell_consumed[sell_consumed > 0]
    if len(sell_consumed_pos) > 0:
        print(f"    消耗事件: {len(sell_consumed_pos)}/{len(analysis)} 秒（{len(sell_consumed_pos)/len(analysis)*100:.1f}%）")
        print(f"    消耗均值（仅消耗秒）: {sell_consumed_pos.mean():.4f} BTC/s")
    
    # 5bps 消耗
    buy_5bps_consumed = analysis["bid_5bps_consumed"]
    sell_5bps_consumed = analysis["ask_5bps_consumed"]
    
    print(f"\n🔥 核心深度消耗（5 bps 内）")
    print(f"  买盘 5bps 平均消耗: {buy_5bps_consumed.mean():.6f} BTC/s")
    print(f"  卖盘 5bps 平均消耗: {sell_5bps_consumed.mean():.6f} BTC/s")
    print(f"  买盘 5bps 最大消耗: {buy_5bps_consumed.max():.6f} BTC/s")
    print(f"  卖盘 5bps 最大消耗: {sell_5bps_consumed.max():.6f} BTC/s")
    
    # 失衡度
    print(f"\n⚖️  买卖盘失衡度")
    imbalance = analysis["imbalance"]
    print(f"  均值: {imbalance.mean():.4f}（正=买盘厚，负=卖盘厚）")
    print(f"  中位数: {imbalance.median():.4f}")
    print(f"  范围: [{imbalance.min():.4f}, {imbalance.max():.4f}]")
    imb_buy_heavy = (imbalance > 0.3).sum()
    imb_sell_heavy = (imbalance < -0.3).sum()
    print(f"  买盘偏厚(>0.3): {imb_buy_heavy}/{len(analysis)} 秒（{imb_buy_heavy/len(analysis)*100:.1f}%）")
    print(f"  卖盘偏厚(<-0.3): {imb_sell_heavy}/{len(analysis)} 秒（{imb_sell_heavy/len(analysis)*100:.1f}%）")
    
    # Top 消耗事件
    print(f"\n🚨 Top 10 买盘消耗事件（卖盘被吃）")
    top_buy = analysis.nlargest(10, "bid_depth_consumed")[["time_str", "mid_price", "bid_depth", "bid_depth_consumed", "ask_depth_consumed", "imbalance"]]
    for _, row in top_buy.iterrows():
        print(f"  {row['time_str']} | 中价={row['mid_price']:.2f} | 买耗={row['bid_depth_consumed']:.4f} | 卖耗={row['ask_depth_consumed']:.4f} | 失衡={row['imbalance']:.4f}")
    
    print(f"\n🚨 Top 10 卖盘消耗事件（买盘被吃）")
    top_sell = analysis.nlargest(10, "ask_depth_consumed")[["time_str", "mid_price", "bid_depth", "bid_depth_consumed", "ask_depth_consumed", "imbalance"]]
    for _, row in top_sell.iterrows():
        print(f"  {row['time_str']} | 中价={row['mid_price']:.2f} | 买耗={row['bid_depth_consumed']:.4f} | 卖耗={row['ask_depth_consumed']:.4f} | 失衡={row['imbalance']:.4f}")
    
    # 消耗量分布
    print(f"\n📈 消耗量分布")
    for label, col in [("买盘消耗", "bid_depth_consumed"), ("卖盘消耗", "ask_depth_consumed")]:
        s = analysis[col]
        bins = [-np.inf, -1.0, -0.1, -0.01, 0, 0.01, 0.1, 1.0, 10.0, np.inf]
        labels = ["<-1（巨量补充）", "-1~-0.1（大量补充）", "-0.1~-0.01", "~0", "0~0.01", "0.01~0.1", "0.1~1.0", "1.0~10.0", ">10（巨量消耗）"]
        counts = pd.cut(s, bins=bins, labels=labels).value_counts().sort_index()
        print(f"\n  {label}分布:")
        for bin_label, count in counts.items():
            bar = "█" * int(count / len(s) * 50)
            print(f"    {bin_label:20s}: {count:4d} ({count/len(s)*100:5.1f}%) {bar}")
    
    # 保存 CSV
    if output_path:
        analysis_out = analysis[[
            "time_str", "second", "mid_price", "spread",
            "bid_depth", "ask_depth", "total_depth",
            "bid_depth_consumed", "ask_depth_consumed", "total_depth_consumed",
            "bid_depth_5bps", "ask_depth_5bps",
            "bid_5bps_consumed", "ask_5bps_consumed",
            "imbalance"
        ]].copy()
        analysis_out.to_csv(output_path, index=False)
        print(f"\n💾 详细数据已保存至: {output_path}")
    
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="分析订单簿深度消耗")
    parser.add_argument("--exchange", default="binance", help="交易所名称")
    parser.add_argument("--pair", default=None, help="交易对（默认自动）")
    parser.add_argument("--output", default="user_data/orderbook_data/depth_consumption_report.csv", help="输出 CSV 路径")
    args = parser.parse_args()
    
    pair = args.pair or ("BTCUSDT" if args.exchange == "binance" else "BTC_USDT")
    
    print(f"加载 {args.exchange}/{pair} 的 depth 数据...")
    df = load_depth_data(pair, args.exchange)
    df = compute_consumption(df)
    generate_report(df, args.output)

if __name__ == "__main__":
    main()
