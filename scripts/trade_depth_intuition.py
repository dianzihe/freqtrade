"""
直观展示：每秒成交涉及/消耗了多少 L2 订单簿深度
核心思路：
  1. 逐笔成交的价格 vs 当时中价 = 价格冲击（bps）
  2. 价格冲击越大 = 成交穿透了越多档位深度
  3. 汇总到秒级：平均冲击、最大冲击、成交量加权冲击

输出：
  - 文字报告（分档统计）
  - HTML 交互图表（4 子图）
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent))

UTC = timezone.utc
CHINA_TZ = timezone(timedelta(hours=8))

def load_data(pair: str, exchange: str):
    from datetime import timezone, timedelta
    
    # Load trades
    trades_dir = Path(f"user_data/orderbook_data/{exchange}/spot/{pair}/trades")
    if not trades_dir.exists():
        raise FileNotFoundError(f"未找到 trades: {trades_dir}")
    trades_df = pd.read_parquet(trades_dir)
    print(f"  Trades: {len(trades_df)} 条")
    
    # Load depth (for mid price)
    depth_dir = Path(f"user_data/orderbook_data/{exchange}/spot/{pair}/depth")
    if not depth_dir.exists():
        raise FileNotFoundError(f"未找到 depth: {depth_dir}")
    depth_df = pd.read_parquet(depth_dir)
    print(f"  Depth:  {len(depth_df)} 行")
    
    return trades_df, depth_df

def analyze(pair: str, exchange: str, output_html: str = None):
    from datetime import timezone, timedelta
    
    print(f"加载 {exchange}/{pair} 数据...")
    trades_df, depth_df = load_data(pair, exchange)
    
    # 处理 depth：按秒去重，取中价
    depth_df = depth_df.sort_values("local_time_ms").reset_index(drop=True)
    depth_df["second"] = (depth_df["local_time_ms"] // 1000).astype(int)
    depth_sec = depth_df.drop_duplicates(subset=["second"], keep="last")[["second", "mid_price", "bid_depth", "ask_depth", "imbalance"]].copy()
    
    # 处理 trades：加秒级时间戳，计算中价（用 depth 数据插值）
    trades_df["second"] = (trades_df["exchange_time_ms"] // 1000).astype(int)
    
    # 将 depth 的中价按秒对齐到 trades
    trades_merged = pd.merge_asof(
        trades_df.sort_values("second"),
        depth_sec[["second", "mid_price"]].sort_values("second"),
        on="second",
        direction="nearest"
    )
    
    # 计算每笔成交的价格冲击（bps）
    # side: buy = 主动买入（吃卖盘），成交价 > 中价
    # side: sell = 主动卖出（吃买盘），成交价 < 中价
    trades_merged["price_impact_bps"] = (trades_merged["price"] - trades_merged["mid_price"]) / trades_merged["mid_price"] * 10000
    trades_merged["abs_impact_bps"] = trades_merged["price_impact_bps"].abs()
    
    # 按秒聚合
    sec_stats = trades_merged.groupby("second").agg(
        total_vol=("amount", "sum"),
        total_count=("amount", "count"),
        buy_vol=("amount", lambda x: x[trades_merged.loc[x.index, "side"] == "buy"].sum()),
        sell_vol=("amount", lambda x: x[trades_merged.loc[x.index, "side"] == "sell"].sum()),
        avg_impact_bps=("abs_impact_bps", "mean"),
        max_impact_bps=("abs_impact_bps", "max"),
        weighted_impact_bps=("abs_impact_bps", lambda x: (x * trades_merged.loc[x.index, "amount"]).sum() / x.sum() if x.sum() > 0 else 0),
    ).reset_index()
    
    # 合并 depth 信息
    sec_stats = pd.merge(sec_stats, depth_sec, on="second", how="left")
    
    # 时间字符串
    sec_stats["time_str"] = pd.to_datetime(sec_stats["second"], unit="s").dt.tz_localize(UTC).dt.tz_convert(CHINA_TZ).dt.strftime("%H:%M:%S")
    
    # 打印直观报告
    print_intuitive_report(sec_stats, pair, exchange)
    
    # 生成 HTML 图表
    if output_html:
        generate_intuitive_chart(sec_stats, output_html, pair, exchange)
    
    # 保存
    csv_path = f"user_data/orderbook_data/{exchange}/spot/{pair}/trade_impact_by_second.csv"
    sec_stats.to_csv(csv_path, index=False)
    print(f"\n详细数据: {csv_path}")
    
    return sec_stats

def print_intuitive_report(df: pd.DataFrame, pair: str, exchange: str):
    """打印直观报告"""
    print(f"\n{'='*70}")
    print(f"  {exchange.upper()}/{pair} — 每秒成交涉及多少 L2 深度？")
    print(f"{'='*70}")
    
    total_sec = len(df)
    total_trades = df["total_count"].sum()
    total_vol = df["total_vol"].sum()
    
    print(f"\n[概况]")
    print(f"  时长:     {total_sec} 秒 ({total_sec/60:.1f} 分钟)")
    print(f"  总成交:   {total_trades:.0f} 笔 / {total_vol:.2f} BTC")
    print(f"  均笔大小: {df['total_vol'].sum()/df['total_count'].sum()*1000:.1f} mBTC = {df['total_vol'].sum()/df['total_count'].sum()*100000000:.0f} sats")
    
    print(f"\n[每秒成交 vs 价格冲击]")
    print(f"  平均每秒成交:   {df['total_vol'].mean():.4f} BTC/s")
    print(f"  最大每秒成交:   {df['total_vol'].max():.4f} BTC/s")
    print(f"  平均价格冲击:   {df['avg_impact_bps'].mean():.2f} bps")
    print(f"  最大价格冲击:   {df['max_impact_bps'].max():.2f} bps")
    print(f"  成交量加权冲击: {df['weighted_impact_bps'].mean():.2f} bps")
    
    # 核心解读
    print(f"\n[核心解读：价格冲击 → 涉及多少档位]")
    print(f"  (Binance BTC/USDT 档位间隔 = 0.01 USDT = 0.016 bps)")
    print(f"  ------------------------------------------------------------")
    print(f"  冲击 < 0.1 bps  → 只碰最佳档（95%+ 成交在此档）")
    print(f"  冲击  0.1~1 bps → 涉及 2~5 档")
    print(f"  冲击  1~10 bps  → 涉及 5~50 档（吃穿部分深度）")
    print(f"  冲击 > 10 bps   → 涉及 50+ 档（大单/恐慌成交）")
    print(f"  ------------------------------------------------------------")
    
    # 按冲击分档
    print(f"\n[每秒冲击分档分布]")
    bins = [0, 0.1, 1, 10, 100, np.inf]
    labels = ["<0.1 bps (最佳档)", "0.1~1 bps (2~5档)", "1~10 bps (5~50档)", "10~100 bps (50~500档)", ">100 bps (500+档)"]
    df["impact_bin"] = pd.cut(df["avg_impact_bps"], bins=bins, labels=labels)
    counts = df["impact_bin"].value_counts().sort_index()
    for label, count in counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:25s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # 高冲击事件
    print(f"\n[高冲击事件 Top 10]")
    top = df.nlargest(10, "max_impact_bps")[["time_str", "total_vol", "total_count", "avg_impact_bps", "max_impact_bps", "mid_price"]]
    for _, row in top.iterrows():
        print(f"  {row['time_str']} | 成交={row['total_vol']:.4f} ({row['total_count']:.0f}笔) | 冲击={row['avg_impact_bps']:.2f}/{row['max_impact_bps']:.2f} bps | 中价={row['mid_price']:.1f}")
    
    # 按成交量分组的冲击
    print(f"\n[不同成交量的平均价格冲击]")
    vol_bins = [0, 0.001, 0.01, 0.1, 1, 10, np.inf]
    vol_labels = ["<0.001 BTC", "0.001~0.01 BTC", "0.01~0.1 BTC", "0.1~1 BTC", "1~10 BTC", ">10 BTC"]
    df["vol_bin"] = pd.cut(df["total_vol"], bins=vol_bins, labels=vol_labels)
    vol_impact = df.groupby("vol_bin", observed=True).agg(
        count=("total_vol", "count"),
        avg_impact=("avg_impact_bps", "mean"),
        max_impact=("max_impact_bps", "max"),
    )
    for label, row in vol_impact.iterrows():
        print(f"  {str(label):15s}: {int(row['count']):4d} 秒 | 平均冲击={row['avg_impact']:.2f} bps | 最大={row['max_impact']:.2f} bps")
    
    print(f"\n{'='*70}")
    print(f"  一句话总结:")
    avg_impact = df["avg_impact_bps"].mean()
    if avg_impact < 0.1:
        print(f"  大部分成交只在最佳档，市场冲击极小（{avg_impact:.3f} bps）")
    elif avg_impact < 1:
        print(f"  成交通常涉及 2~5 档深度，市场冲击较小（{avg_impact:.2f} bps）")
    elif avg_impact < 10:
        print(f"  成交经常吃穿多档，市场冲击中等（{avg_impact:.2f} bps）")
    else:
        print(f"  成交频繁吃穿大量档位，市场冲击大（{avg_impact:.2f} bps）")
    print(f"{'='*70}")

def generate_intuitive_chart(df: pd.DataFrame, output_path: str, pair: str, exchange: str):
    """生成 4 子图交互图表"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except ImportError:
        print("  [Warn] plotly 未安装，跳过图表")
        return
    
    df = df.copy()
    df["datetime_str"] = pd.to_datetime(df["second"], unit="s").dt.tz_localize(UTC).dt.tz_convert(CHINA_TZ).dt.strftime("%H:%M:%S")
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "每秒成交量（BTC）",
            "平均价格冲击（bps）",
            "最大单笔冲击（bps）",
            "买卖盘深度（BTC）",
        ],
        vertical_spacing=0.1,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )
    
    # 1. 成交量
    fig.add_trace(go.Bar(x=df["datetime_str"], y=df["total_vol"], name="成交量", marker_color="rgba(55,128,191,0.7)"), row=1, col=1)
    
    # 2. 平均冲击
    fig.add_trace(go.Scatter(x=df["datetime_str"], y=df["avg_impact_bps"], mode="markers+lines", name="平均冲击(bps)", marker=dict(size=5, color=df["avg_impact_bps"], colorscale="Viridis", showscale=True, colorbar=dict(title="bps", x=1.02))), row=2, col=1)
    
    # 3. 最大冲击
    fig.add_trace(go.Scatter(x=df["datetime_str"], y=df["max_impact_bps"], mode="markers", name="最大单笔冲击(bps)", marker=dict(size=5, color="red", opacity=0.7)), row=3, col=1)
    
    # 4. 深度
    fig.add_trace(go.Scatter(x=df["datetime_str"], y=df["bid_depth"], mode="lines", name="买盘深度", line=dict(color="red")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["datetime_str"], y=df["ask_depth"], mode="lines", name="卖盘深度", line=dict(color="green")), row=4, col=1)
    
    fig.update_layout(title=f"{exchange.upper()}/{pair} — 成交价格冲击分析（涉及多少 L2 深度）", height=1200, showlegend=True)
    fig.update_yaxes(title_text="BTC/s", row=1, col=1)
    fig.update_yaxes(title_text="bps", row=2, col=1)
    fig.update_yaxes(title_text="bps", row=3, col=1)
    fig.update_yaxes(title_text="BTC", row=4, col=1)
    
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"\n图表已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--pair", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    pair = args.pair or ("BTCUSDT" if args.exchange == "binance" else "BTC_USDT")
    output = args.output or f"user_data/orderbook_data/{args.exchange}/spot/{pair}/trade_impact_chart.html"
    
    analyze(pair, args.exchange, output)

if __name__ == "__main__":
    main()
