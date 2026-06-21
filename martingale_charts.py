"""
生成马丁策略分析的可视化图表
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import os

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

out_dir = "user_data/plots"
os.makedirs(out_dir, exist_ok=True)

# 加载数据
coins = ["H_USDT-1m", "VELVET_USDT-1m", "DN_USDT-1m", "BEAT_USDT-1m"]
coin_names = ["H", "VELVET", "DN", "BEAT"]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
data = {}

for fname, cname in zip(coins, coin_names):
    df = pd.read_feather(f"user_data/data/gate/{fname}.feather")
    df = df.set_index("date").sort_index()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    data[cname] = df

# ================================================================
# 图1: 四币价格走势对比（归一化）
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (cname, df) in enumerate(data.items()):
    ax = axes[i]
    norm_close = df["close"] / df["close"].iloc[0] * 100
    ax.fill_between(df.index, 100, norm_close, alpha=0.3, color=colors[i])
    ax.plot(df.index, norm_close, color=colors[i], linewidth=0.8)
    ax.plot(df.index, [100]*len(df), '--', color='gray', alpha=0.5)
    ax.set_title(f"{cname} 价格走势 (归一化)")
    ax.set_ylabel("Price (% of start)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.grid(True, alpha=0.3)

fig.suptitle("四只妖币价格走势对比 (06/08 - 06/20)", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{out_dir}/price_trends.png", bbox_inches='tight')
plt.close()
print("图1: price_trends.png 已生成")

# ================================================================
# 图2: 1小时波动率分布对比
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (cname, df) in enumerate(data.items()):
    ax = axes[i]
    hourly_ret = df["close"].resample("1h").last().pct_change().dropna() * 100
    # 过滤极端值用于直方图
    clean_ret = hourly_ret[(hourly_ret > -50) & (hourly_ret < 50)]
    ax.hist(clean_ret, bins=80, color=colors[i], alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(clean_ret.mean(), color='red', linestyle='--', label=f"Mean={clean_ret.mean():.2f}%")
    ax.set_title(f"{cname} 1小时收益率分布")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("1小时波动率分布", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{out_dir}/hourly_volatility.png", bbox_inches='tight')
plt.close()
print("图2: hourly_volatility.png 已生成")

# ================================================================
# 图3: 累积回撤图
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (cname, df) in enumerate(data.items()):
    ax = axes[i]
    close = df["close"]
    cummax = close.expanding().max()
    drawdown = (close - cummax) / cummax * 100
    ax.fill_between(df.index, 0, drawdown, alpha=0.4, color=colors[i])
    ax.plot(df.index, drawdown, color=colors[i], linewidth=0.5)
    ax.set_title(f"{cname} 累计回撤 (MaxDD={drawdown.min():.1f}%)")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylim(drawdown.min() * 1.2, 5)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

fig.suptitle("投资回撤曲线", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{out_dir}/drawdown_curves.png", bbox_inches='tight')
plt.close()
print("图3: drawdown_curves.png 已生成")

# ================================================================
# 图4: 马丁策略综合对比柱状图
# ================================================================
results = {
    "H": {"单向马丁": -19.50, "多空网格": -0.76, "DD": -27.22},
    "VELVET": {"单向马丁": 18.25, "多空网格": 60.20, "DD": -65.63},
    "DN": {"单向马丁": 33.70, "多空网格": 25.46, "DD": -44.17},
    "BEAT": {"单向马丁": 2.04, "多空网格": -14.07, "DD": -46.85},
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(coin_names))
width = 0.3

bars1 = ax.bar(x - width/2, [results[c]["单向马丁"] for c in coin_names],
               width, label='Standard Martingale', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, [results[c]["多空网格"] for c in coin_names],
               width, label='Grid Martingale', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Coin')
ax.set_ylabel('Return (%)')
ax.set_title('Martin Strategy Returns Comparison')
ax.set_xticks(x)
ax.set_xticklabels(coin_names)
ax.legend()
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + (2 if h>0 else -4),
            f'{h:+.1f}%', ha='center', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + (2 if h>0 else -4),
            f'{h:+.1f}%', ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(f"{out_dir}/strategy_comparison.png", bbox_inches='tight')
plt.close()
print("图4: strategy_comparison.png 已生成")

# ================================================================
# 图5: H币详细分析 - V形反转与波动
# ================================================================
h_df = data["H"]
h_close = h_df["close"]
h_1h = h_close.resample("1h").last()
h_1h_ret = h_1h.pct_change().dropna() * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# 上半: 价格 + 马丁网格
ax1.plot(h_df.index, h_df["close"], color='#e74c3c', linewidth=0.5, label='Close')
ax1.set_title("H Coin - Price with 5% Martin Grid Level")
ax1.set_ylabel("Price (USDT)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
init_price = h_close.iloc[0]
for level in np.arange(-0.8, 0.1, 0.05):
    ax1.axhline(init_price * (1 + level), color='gray', alpha=0.15, linewidth=0.5)
ax1.axhline(init_price, color='blue', linewidth=1, label=f'Entry={init_price:.4f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下半: 1小时收益率
ax2.fill_between(h_1h_ret.index, 0, h_1h_ret, alpha=0.5, color='#e74c3c')
ax2.bar(h_1h_ret.index, h_1h_ret, color=np.where(h_1h_ret > 0, 'red', 'green'),
        alpha=0.6, width=0.03)
ax2.set_title("H Coin - 1-Hour Returns")
ax2.set_ylabel("Return (%)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{out_dir}/h_coin_detail.png", bbox_inches='tight')
plt.close()
print("图5: h_coin_detail.png 已生成")

# ================================================================
# 图6: 波动率雷达图
# ================================================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

metrics = ['1m Volatility', '1h Volatility', 'Daily Volatility',
           'Max Drawdown(%)', 'Max Amplitude(%)', 'Trade Volume']

# 归一化到0-100
stats = {}
for cname, df in data.items():
    close = df["close"]
    returns = close.pct_change().dropna()
    cummax = close.expanding().max()
    dd = (close - cummax) / cummax
    d_ret = close.resample("1D").last().pct_change().dropna()
    h_ret = close.resample("1h").last().pct_change().dropna()
    stats[cname] = [
        returns.std() * 100 * 100,  # 放大
        h_ret.std() * 100,
        d_ret.std() * 100,
        abs(dd.min() * 100),
        (close.max() / close.min() - 1) * 100,
        (df["volume"] * close).mean() / 1000
    ]

# 归一化
all_vals = np.array([v for v in stats.values()])
max_vals = all_vals.max(axis=0)
for cname in stats:
    stats[cname] = [v/m * 100 for v, m in zip(stats[cname], max_vals)]

N = len(metrics)
angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (cname, color) in enumerate(zip(coin_names, colors)):
    values = stats[cname] + stats[cname][:1]
    ax.fill(angles, values, alpha=0.15, color=color)
    ax.plot(angles, values, 'o-', linewidth=2, label=cname, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=9)
ax.set_title("Multi-Dimensional Risk Comparison", fontsize=13, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_ylim(0, 100)

plt.tight_layout()
fig.savefig(f"{out_dir}/risk_radar.png", bbox_inches='tight')
plt.close()
print("图6: risk_radar.png 已生成")

# ================================================================
# 图7: DN币（最优表现）马丁策略详细模拟
# ================================================================
dn_df = data["DN"]
dn_close = dn_df["close"]

# 模拟马丁网格收益曲线
def simulate_martin_equity(df, grid_pct=0.05, max_layers=4, capital=1000):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    init = close[0]
    base_qty = capital * 0.3 / init
    cash = capital
    fee = 0.001

    pos_qty = 0
    pos_cost = 0
    last_entry_price = init
    layer = 0
    equity_curve = []
    trade_markers = []  # (idx, price, action)

    for i in range(n):
        c = close[i]
        h = high[i]
        l = low[i]

        # 止盈
        if pos_qty > 0:
            avg = pos_cost / pos_qty
            tp = avg * (1 + grid_pct)
            if h >= tp:
                sell = pos_qty * tp * (1 - fee)
                cash += sell
                profit = sell - pos_cost
                trade_markers.append((i, tp, 'sell', profit))
                pos_qty = 0
                pos_cost = 0
                layer = 0
                last_entry_price = tp

        # 首次建仓
        if pos_qty == 0:
            pos_qty = base_qty
            pos_cost = base_qty * c * (1 + fee)
            cash -= pos_cost
            last_entry_price = c
            layer = 1
            trade_markers.append((i, c, 'buy', 0))

        # 加仓
        elif layer < max_layers:
            trigger = last_entry_price * (1 - grid_pct)
            if l <= trigger:
                layer += 1
                mult = 2 ** (layer - 1)
                add_qty = base_qty * mult
                add_cost = add_qty * trigger * (1 + fee)
                if add_cost <= cash * 0.9:
                    cash -= add_cost
                    pos_qty += add_qty
                    pos_cost += add_cost
                    last_entry_price = trigger
                    trade_markers.append((i, trigger, 'add', mult))

        equity = cash + pos_qty * c
        equity_curve.append(equity)

    # 最终清算
    final_val = pos_qty * close[-1] * (1 - fee)
    equity_final = cash + final_val
    equity_curve[-1] = equity_final
    return equity_curve, trade_markers

# DN 币
eq_dn, trades_dn = simulate_martin_equity(dn_df)
# VELVET币
eq_vl, trades_vl = simulate_martin_equity(data["VELVET"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# DN
dn_cost = [0] * len(eq_dn)
ax1.plot(dn_df.index, eq_dn, color='#2ecc71', linewidth=1.5, label='Equity (Martin 5%)')
ax1.axhline(1000, color='gray', linestyle='--', alpha=0.5)
ax1.set_title(f"DN - Martin Strategy Equity Curve (5% grid, 4 layers)")
ax1.set_ylabel("Equity (USDT)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.legend()
ax1.grid(True, alpha=0.3)
for t in trades_dn:
    if t[2] in ('sell',):
        ax1.scatter(dn_df.index[t[0]], eq_dn[t[0]], color='red', s=20, zorder=5, alpha=0.7)
    elif t[2] in ('buy', 'add'):
        ax1.scatter(dn_df.index[t[0]], eq_dn[t[0]], color='blue', s=20, zorder=5, alpha=0.7)

# VELVET
ax2.plot(data["VELVET"].index, eq_vl, color='#3498db', linewidth=1.5, label='Equity (Martin 5%)')
ax2.axhline(1000, color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f"VELVET - Martin Strategy Equity Curve (5% grid, 4 layers)")
ax2.set_ylabel("Equity (USDT)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax2.legend()
ax2.grid(True, alpha=0.3)
for t in trades_vl:
    if t[2] in ('sell',):
        ax2.scatter(data["VELVET"].index[t[0]], eq_vl[t[0]], color='red', s=20, zorder=5, alpha=0.7)
    elif t[2] in ('buy', 'add'):
        ax2.scatter(data["VELVET"].index[t[0]], eq_vl[t[0]], color='blue', s=20, zorder=5, alpha=0.7)

plt.tight_layout()
fig.savefig(f"{out_dir}/equity_curves.png", bbox_inches='tight')
plt.close()
print("图7: equity_curves.png 已生成")

print("\n所有图表生成完成！路径: user_data/plots/")
for f in os.listdir(out_dir):
    print(f"  - {f}")
