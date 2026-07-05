"""
直观展示：每秒成交涉及/消耗了多少 L2 订单簿深度
核心指标：
  - 每秒成交总量（BTC）
  - 每秒消耗的深度（BTC）→ 来自 depth 数据
  - 消耗比 = 深度消耗 / 成交总量
    - 比值高 → 大单吃穿了多档深度（市场冲击大）
    - 比值低 → 小单只碰了最佳档（市场冲击小）
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timezone, timedelta
import sys

# UTF-8 输出编码（修复 Windows 终端乱码）
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent))

UTC = timezone.utc
CHINA_TZ = timezone(timedelta(hours=8))

def load_trades(pair: str, exchange: str) -> pd.DataFrame:
    trades_dir = Path(f"user_data/orderbook_data/{exchange}/spot/{pair}/trades")
    if not trades_dir.exists():
        raise FileNotFoundError(f"未找到 trades 数据: {trades_dir}")
    files = list(trades_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件: {trades_dir}")
    df = pd.read_parquet(trades_dir)
    print(f"  加载 {len(df)} 条成交记录（来自 {len(files)} 个文件）")
    return df

def load_depth(pair: str, exchange: str) -> pd.DataFrame:
    depth_dir = Path(f"user_data/orderbook_data/{exchange}/spot/{pair}/depth")
    files = list(depth_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 depth 数据: {depth_dir}")
    df = pd.read_parquet(depth_dir)
    print(f"  加载 {len(df)} 行 depth 数据（来自 {len(files)} 个文件）")
    return df

def analyze_trade_depth_impact(pair: str, exchange: str, output_html: str = None):
    # 加载数据
    print(f"加载 {exchange}/{pair} 数据...")
    trades_df = load_trades(pair, exchange)
    depth_df = load_depth(pair, exchange)
    
    # 处理 trades：按秒聚合
    trades_df["second"] = (trades_df["exchange_time_ms"] // 1000).astype(int)
    # 判断买卖方向（简化：用 side 字段或 is_buyer_maker）
    if "is_buyer_maker" in trades_df.columns:
        trades_df["is_sell"] = trades_df["is_buyer_maker"]  # True = 主动卖出（吃买盘）
        trades_df["is_buy"] = ~trades_df["is_buyer_maker"]   # True = 主动买入（吃卖盘）
    elif "side" in trades_df.columns:
        trades_df["is_sell"] = trades_df["side"] == "sell"
        trades_df["is_buy"] = trades_df["side"] == "buy"
    else:
        # 无法判断方向，用总量
        trades_df["is_sell"] = False
        trades_df["is_buy"] = False
    
    # 每秒成交统计
    trades_sec = trades_df.groupby("second").agg(
        total_vol=("amount", "sum"),
        total_count=("amount", "count"),
        buy_vol=("amount", lambda x: x[trades_df.loc[x.index, "is_buy"]].sum() if "is_buy" in trades_df.columns else 0),
        sell_vol=("amount", lambda x: x[trades_df.loc[x.index, "is_sell"]].sum() if "is_sell" in trades_df.columns else 0),
    ).reset_index()
    
    # 处理 depth：按秒去重（取最后一笔）
    depth_df = depth_df.sort_values("local_time_ms").reset_index(drop=True)
    depth_df["second"] = (depth_df["local_time_ms"] // 1000).astype(int)
    depth_df = depth_df.drop_duplicates(subset=["second"], keep="last").reset_index(drop=True)
    
    # 计算深度消耗
    depth_df["bid_depth_consumed"] = depth_df["bid_depth"].shift(1) - depth_df["bid_depth"]
    depth_df["ask_depth_consumed"] = depth_df["ask_depth"].shift(1) - depth_df["ask_depth"]
    depth_df["total_depth_consumed"] = depth_df["bid_depth_consumed"].abs() + depth_df["ask_depth_consumed"].abs()
    # 净消耗（正值=被消耗，负值=被补充）
    depth_df["net_consumed"] = depth_df["bid_depth_consumed"].clip(lower=0) + depth_df["ask_depth_consumed"].clip(lower=0)
    
    # 合并
    merged = pd.merge(trades_sec, depth_df[["second", "mid_price", "bid_depth", "ask_depth",
                                             "bid_depth_consumed", "ask_depth_consumed",
                                             "total_depth_consumed", "net_consumed", "imbalance"]],
                      on="second", how="inner")
    
    merged = merged.dropna(subset=["total_depth_consumed"])
    
    if len(merged) == 0:
        print("  ⚠️ 无有效合并数据")
        return
    
    # 计算关键指标
    merged["consumed_per_btc"] = merged["total_depth_consumed"] / merged["total_vol"].replace(0, np.nan)
    merged["consumed_per_btc"] = merged["consumed_per_btc"].replace([np.inf, -np.inf], np.nan)
    merged["buy_consumed_per_btc"] = merged["bid_depth_consumed"].clip(lower=0) / merged["sell_vol"].replace(0, np.nan)
    merged["sell_consumed_per_btc"] = merged["ask_depth_consumed"].clip(lower=0) / merged["buy_vol"].replace(0, np.nan)
    
    # 时间字符串
    merged["time_str"] = pd.to_datetime(merged["second"], unit="s").dt.tz_localize(UTC).dt.tz_convert(CHINA_TZ).dt.strftime("%H:%M:%S")
    
    # 打印报告
    print_report(merged)
    
    # 生成 HTML 可视化
    if output_html:
        generate_html_chart(merged, output_html, pair, exchange)
    
    # 保存 CSV
    csv_path = f"user_data/orderbook_data/{exchange}/spot/{pair}/trade_depth_impact.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\n💾 详细数据已保存至: {csv_path}")
    
    return merged

def print_report(df: pd.DataFrame):
    """打印文字报告"""
    print(f"\n{'='*80}")
    print(f"  每秒成交 vs L2 深度消耗 -- 直观报告")
    print(f"{'='*80}")
    
    duration = len(df)
    print(f"\n[分析时长] {duration} 秒 ({duration/60:.1f} 分钟)")
    print(f"   时间范围: {df['time_str'].iloc[0]} ~ {df['time_str'].iloc[-1]}")
    
    # 成交概况
    print(f"\n📊 每秒成交概况")
    print(f"  平均成交量:   {df['total_vol'].mean():.4f} BTC/s")
    print(f"  最大成交量:   {df['total_vol'].max():.4f} BTC/s")
    print(f"  中位成交量:   {df['total_vol'].median():.4f} BTC/s")
    print(f"  P90 成交量:   {df['total_vol'].quantile(0.9):.4f} BTC/s")
    
    # 深度消耗概况
    print(f"\n🔥 每秒深度消耗概况")
    print(f"  平均消耗(总): {df['total_depth_consumed'].mean():.4f} BTC/s")
    print(f"  最大消耗(总): {df['total_depth_consumed'].max():.4f} BTC/s")
    print(f"  中位消耗(总): {df['total_depth_consumed'].median():.4f} BTC/s")
    
    # 核心指标：消耗比
    valid = df.dropna(subset=["consumed_per_btc"])
    print(f"\n💡 核心指标：成交 1 BTC 消耗多少深度？")
    print(f"  （消耗比 = 深度消耗 / 成交总量）")
    print(f"  -------------------------------------------------------")
    print(f"  均值:  每成交 1 BTC → 消耗 {valid['consumed_per_btc'].mean():.2f} BTC 深度")
    print(f"  中位数:每成交 1 BTC → 消耗 {valid['consumed_per_btc'].median():.2f} BTC 深度")
    print(f"  P10:   每成交 1 BTC → 消耗 {valid['consumed_per_btc'].quantile(0.1):.2f} BTC 深度")
    print(f"  P90:   每成交 1 BTC → 消耗 {valid['consumed_per_btc'].quantile(0.9):.2f} BTC 深度")
    print(f"  最大:  每成交 1 BTC → 消耗 {valid['consumed_per_btc'].max():.2f} BTC 深度")
    print(f"  -------------------------------------------------------")
    print(f"  解读:")
    print(f"    • 比值 ≈ 1 : 成交完全被深度吸收，1 BTC 成交消耗 1 BTC 深度（正常）")
    print(f"    • 比值 > 5 : 大单吃穿多档，市场冲击大（流动性差）")
    print(f"    • 比值 < 0.5: 小单只碰最佳档，市场冲击小（流动性好）")
    
    # 按消耗比分档
    print(f"\n📊 消耗比分布（每成交 1 BTC 消耗多少深度）")
    bins = [0, 0.5, 1, 2, 5, 10, 50, np.inf]
    labels = ["<0.5 (极小冲击)", "0.5~1 (小冲击)", "1~2", "2~5", "5~10 (大冲击)", "10~50 (很大冲击)", ">50 (极强冲击)"]
    valid["consumed_bin"] = pd.cut(valid["consumed_per_btc"], bins=bins, labels=labels)
    counts = valid["consumed_bin"].value_counts().sort_index()
    for label, count in counts.items():
        bar = "█" * int(count / len(valid) * 60)
        print(f"  {label:20s}: {count:4d} ({count/len(valid)*100:5.1f}%) {bar}")
    
    # 高冲击事件
    print(f"\n🚨 高市场冲击事件（消耗比 Top 10）")
    top = valid.nlargest(10, "consumed_per_btc")[["time_str", "mid_price", "total_vol", "total_depth_consumed", "consumed_per_btc", "buy_vol", "sell_vol"]]
    for _, row in top.iterrows():
        print(f"  {row['time_str']} | 中价={row['mid_price']:.1f} | 成交={row['total_vol']:.4f} | 消耗={row['total_depth_consumed']:.4f} | 比值={row['consumed_per_btc']:.1f}x | 买={row['buy_vol']:.4f} 卖={row['sell_vol']:.4f}")
    
    # 流动性好的时刻
    print(f"\n✅ 流动性好的时刻（消耗比最小 10 秒）")
    bottom = valid.nsmallest(10, "consumed_per_btc")[["time_str", "mid_price", "total_vol", "total_depth_consumed", "consumed_per_btc"]]
    for _, row in bottom.iterrows():
        print(f"  {row['time_str']} | 中价={row['mid_price']:.1f} | 成交={row['total_vol']:.4f} | 消耗={row['total_depth_consumed']:.4f} | 比值={row['consumed_per_btc']:.3f}x")
    
    print(f"\n{'='*80}")

def generate_html_chart(df: pd.DataFrame, output_path: str, pair: str, exchange: str):
    """生成交互式 HTML 图表"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except ImportError:
        print("  ⚠️ plotly 未安装，跳过 HTML 图表生成")
        print("     安装: pip install plotly")
        return
    
    # 准备数据
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["second"], unit="s").dt.tz_localize(UTC).dt.tz_convert(CHINA_TZ)
    df["datetime_str"] = df["datetime"].dt.strftime("%H:%M:%S")
    
    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "每秒成交量（BTC）",
            "每秒深度消耗（BTC）",
            "消耗比（深度消耗 / 成交总量）",
            "买卖盘深度",
        ],
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )
    
    # 1. 每秒成交量
    fig.add_trace(
        go.Bar(
            x=df["datetime_str"],
            y=df["total_vol"],
            name="成交总量 (BTC/s)",
            marker_color="rgba(55, 128, 191, 0.7)",
            hovertemplate="时间: %{x}<br>成交: %{y:.4f} BTC/s<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime_str"],
            y=df["buy_vol"],
            mode="lines",
            name="主动买入",
            line=dict(color="red", width=1),
            hovertemplate="主动买入: %{y:.4f} BTC/s<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime_str"],
            y=df["sell_vol"],
            mode="lines",
            name="主动卖出",
            line=dict(color="green", width=1),
            hovertemplate="主动卖出: %{y:.4f} BTC/s<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # 2. 深度消耗
    fig.add_trace(
        go.Bar(
            x=df["datetime_str"],
            y=df["bid_depth_consumed"].clip(lower=0),
            name="买盘消耗 (BTC/s)",
            marker_color="rgba(255, 100, 100, 0.7)",
            hovertemplate="买盘消耗: %{y:.4f} BTC/s<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["datetime_str"],
            y=df["ask_depth_consumed"].clip(lower=0),
            name="卖盘消耗 (BTC/s)",
            marker_color="rgba(100, 200, 100, 0.7)",
            hovertemplate="卖盘消耗: %{y:.4f} BTC/s<extra></extra>",
        ),
        row=2, col=1,
    )
    
    # 3. 消耗比
    valid = df.dropna(subset=["consumed_per_btc"])
    fig.add_trace(
        go.Scatter(
            x=valid["datetime_str"],
            y=valid["consumed_per_btc"],
            mode="markers+lines",
            name="消耗比 (x)",
            marker=dict(
                size=6,
                color=valid["consumed_per_btc"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="消耗比", x=1.02),
            ),
            line=dict(width=1),
            hovertemplate="时间: %{x}<br>消耗比: %{y:.2f}x<br>(每成交 1 BTC 消耗 %{y:.2f} BTC 深度)<extra></extra>",
        ),
        row=3, col=1,
    )
    
    # 4. 买卖盘深度
    fig.add_trace(
        go.Scatter(
            x=df["datetime_str"],
            y=df["bid_depth"],
            mode="lines",
            name="买盘深度 (BTC)",
            line=dict(color="red", width=1),
            hovertemplate="买盘深度: %{y:.2f} BTC<extra></extra>",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime_str"],
            y=df["ask_depth"],
            mode="lines",
            name="卖盘深度 (BTC)",
            line=dict(color="green", width=1),
            hovertemplate="卖盘深度: %{y:.2f} BTC<extra></extra>",
        ),
        row=4, col=1,
    )
    
    # 布局
    fig.update_layout(
        title=f"{exchange.upper()}/{pair} — 每秒成交 vs L2 深度消耗分析",
        height=1200,
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="时间 (本地)", row=4, col=1)
    fig.update_yaxes(title_text="BTC/s", row=1, col=1)
    fig.update_yaxes(title_text="BTC/s", row=2, col=1)
    fig.update_yaxes(title_text="消耗比 (x)", row=3, col=1)
    fig.update_yaxes(title_text="BTC", row=4, col=1)
    
    # 保存
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"\n📈 交互式图表已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="分析每秒成交涉及的 L2 深度")
    parser.add_argument("--exchange", default="binance", help="交易所名称")
    parser.add_argument("--pair", default=None, help="交易对")
    parser.add_argument("--output", default=None, help="输出 HTML 路径")
    args = parser.parse_args()
    
    pair = args.pair or ("BTCUSDT" if args.exchange == "binance" else "BTC_USDT")
    output = args.output or f"user_data/orderbook_data/{args.exchange}/spot/{pair}/trade_depth_impact.html"
    
    analyze_trade_depth_impact(pair, args.exchange, output)

if __name__ == "__main__":
    main()
