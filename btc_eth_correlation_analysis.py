"""
BTC/ETH 5分钟涨跌一致性分析
分析 BTC 和 ETH 在 5 分钟维度上的价格变动是否趋于一致
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── 中文 & 暗色主题设置 ───
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')

# 中国股市惯例: 涨=红, 跌=绿
RED = '#EF5350'
GREEN = '#26A69A'
GOLD = '#FFD54F'
BLUE = '#42A5F5'
PURPLE = '#AB47BC'
ORANGE = '#FF7043'
CYAN = '#26C6DA'
WHITE = '#ECEFF1'
GRAY = '#78909C'

# ─── 1. 加载数据 ───
print("=" * 60)
print("BTC/ETH 5分钟涨跌一致性分析")
print("=" * 60)

btc_path = 'user_data/orderbook_data/formatted/BTC_USDT/ohlcv_5min.parquet'
eth_path = 'user_data/orderbook_data/formatted/ETH_USDT/ohlcv_5min.parquet'

btc = pd.read_parquet(btc_path)
eth = pd.read_parquet(eth_path)

# 确保 datetime 是索引且有时区感知
for df in [btc, eth]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

print(f"\nBTC 数据: {len(btc)} 行, {btc.index[0]} ~ {btc.index[-1]}")
print(f"ETH 数据: {len(eth)} 行, {eth.index[0]} ~ {eth.index[-1]}")

# ─── 2. 合并数据 ───
merged = pd.DataFrame({
    'btc_close': btc['close'],
    'eth_close': eth['close'],
    'btc_volume': btc['volume'],
    'eth_volume': eth['volume'],
}, index=btc.index).join(
    pd.DataFrame({'eth_close': eth['close']}, index=eth.index),
    how='inner', rsuffix='_eth'
)

# 去重列
merged = merged[['btc_close', 'eth_close', 'btc_volume', 'eth_volume']].dropna()

# 转换时区为北京时间
merged.index = merged.index.tz_convert('Asia/Shanghai')

print(f"对齐后数据: {len(merged)} 行")
print(f"时间范围: {merged.index[0]} ~ {merged.index[-1]}")

# ─── 3. 计算收益率 ───
merged['btc_ret'] = merged['btc_close'].pct_change() * 100  # 百分比
merged['eth_ret'] = merged['eth_close'].pct_change() * 100
merged['btc_dir'] = np.sign(merged['btc_ret'])
merged['eth_dir'] = np.sign(merged['eth_ret'])
merged['same_dir'] = (merged['btc_dir'] == merged['eth_dir']).astype(int)
merged['btc_cum'] = (1 + merged['btc_ret'] / 100).cumprod()
merged['eth_cum'] = (1 + merged['eth_ret'] / 100).cumprod()
merged = merged.dropna()

print(f"有效收益率: {len(merged)} 行")

# ─── 4. 核心统计 ───
ret_data = merged[['btc_ret', 'eth_ret']].dropna()

# Pearson 相关
pearson_r, pearson_p = stats.pearsonr(ret_data['btc_ret'], ret_data['eth_ret'])

# Spearman 相关
spearman_r, spearman_p = stats.spearmanr(ret_data['btc_ret'], ret_data['eth_ret'])

# 方向一致率
same_pct = merged['same_dir'].mean() * 100
# 排除平盘(0)的方向一致率
nonzero = merged[(merged['btc_dir'] != 0) & (merged['eth_dir'] != 0)]
same_nonzero_pct = nonzero['same_dir'].mean() * 100

# 回归
slope, intercept, r_value, p_value, std_err = stats.linregress(
    ret_data['btc_ret'], ret_data['eth_ret']
)

# Beta (ETH对BTC的敏感度)
beta = slope

print(f"\n{'='*40}")
print(f"核心统计结果")
print(f"{'='*40}")
print(f"Pearson 相关系数:  {pearson_r:.4f} (p={pearson_p:.2e})")
print(f"Spearman 相关系数: {spearman_r:.4f} (p={spearman_p:.2e})")
print(f"R² (决定系数):     {r_value**2:.4f}")
print(f"Beta (ETH/BTC):    {beta:.4f}")
print(f"方向一致率(全部):  {same_pct:.1f}%")
print(f"方向一致率(非零):  {same_nonzero_pct:.1f}%")
print(f"BTC 平均收益率:    {merged['btc_ret'].mean():.4f}%")
print(f"ETH 平均收益率:    {merged['eth_ret'].mean():.4f}%")
print(f"BTC 波动率(std):   {merged['btc_ret'].std():.4f}%")
print(f"ETH 波动率(std):   {merged['eth_ret'].std():.4f}%")

# ─── 5. 滚动相关性 ───
window = 48  # 4小时窗口 (48 x 5min)
merged['rolling_corr'] = merged['btc_ret'].rolling(window).corr(merged['eth_ret'])

# ─── 6. 绘图 ───
fig = plt.figure(figsize=(22, 18))
fig.suptitle('BTC/USDT vs ETH/USDT  5分钟涨跌一致性分析', fontsize=18, fontweight='bold', y=0.98)

# ── 图1: 归一化价格走势 ──
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(merged.index, merged['btc_cum'] / merged['btc_cum'].iloc[0] * 100,
         color=ORANGE, linewidth=1.2, alpha=0.9, label='BTC')
ax1.plot(merged.index, merged['eth_cum'] / merged['eth_cum'].iloc[0] * 100,
         color=CYAN, linewidth=1.2, alpha=0.9, label='ETH')
ax1.set_title('归一化累计收益率 (起点=100)', fontsize=13, fontweight='bold', color=WHITE)
ax1.legend(loc='upper left', fontsize=9)
ax1.axhline(y=100, color=GRAY, linewidth=0.5, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
ax1.grid(True, alpha=0.2)
ax1.set_ylabel('累计收益(%)')
ax1.tick_params(colors=GRAY)

# ── 图2: 收益率散点图 ──
ax2 = fig.add_subplot(3, 3, 2)
# 区分同向/反向
same = merged['same_dir'] == 1
diff = merged['same_dir'] == 0
ax2.scatter(merged.loc[same, 'btc_ret'], merged.loc[same, 'eth_ret'],
            c=GOLD, alpha=0.35, s=10, label=f'同向 ({same_pct:.0f}%)', edgecolors='none')
ax2.scatter(merged.loc[diff, 'btc_ret'], merged.loc[diff, 'eth_ret'],
            c=GRAY, alpha=0.2, s=10, label=f'反向 ({100-same_pct:.0f}%)', edgecolors='none')

# 回归线
x_range = np.linspace(merged['btc_ret'].min(), merged['btc_ret'].max(), 100)
ax2.plot(x_range, slope * x_range + intercept, color=RED, linewidth=2,
         label=f'回归: ETH={beta:.2f}×BTC+{intercept:.3f}\nR²={r_value**2:.3f}, ρ={pearson_r:.3f}')
ax2.axhline(y=0, color=GRAY, linewidth=0.5)
ax2.axvline(x=0, color=GRAY, linewidth=0.5)
ax2.set_xlabel('BTC 5分钟收益率 (%)', fontsize=11, color=GRAY)
ax2.set_ylabel('ETH 5分钟收益率 (%)', fontsize=11, color=GRAY)
ax2.set_title('5分钟收益率散点图', fontsize=13, fontweight='bold', color=WHITE)
ax2.legend(loc='upper left', fontsize=8, framealpha=0.7)
ax2.grid(True, alpha=0.2)
ax2.tick_params(colors=GRAY)

# ── 图3: 方向一致率分解 ──
ax3 = fig.add_subplot(3, 3, 3)
dir_labels = ['BTC涨 ETH涨', 'BTC跌 ETH跌', 'BTC涨 ETH跌', 'BTC跌 ETH涨']
btc_up_eth_up = ((merged['btc_ret'] > 0) & (merged['eth_ret'] > 0)).sum()
btc_down_eth_down = ((merged['btc_ret'] < 0) & (merged['eth_ret'] < 0)).sum()
btc_up_eth_down = ((merged['btc_ret'] > 0) & (merged['eth_ret'] < 0)).sum()
btc_down_eth_up = ((merged['btc_ret'] < 0) & (merged['eth_ret'] > 0)).sum()
dir_counts = [btc_up_eth_up, btc_down_eth_down, btc_up_eth_down, btc_down_eth_up]
dir_colors = [RED, GREEN, ORANGE, PURPLE]
total = sum(dir_counts)
dir_pcts = [c / total * 100 for c in dir_counts]

bars = ax3.barh(dir_labels, dir_pcts, color=dir_colors, edgecolor='#1a1a2e', linewidth=0.8)
for bar, pct, cnt in zip(bars, dir_pcts, dir_counts):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{pct:.1f}%  (n={cnt})', va='center', fontsize=10, color=WHITE)
ax3.set_title('涨跌方向分布', fontsize=13, fontweight='bold', color=WHITE)
ax3.set_xlabel('占比 (%)', fontsize=11, color=GRAY)
ax3.set_xlim(0, max(dir_pcts) * 1.3)
ax3.tick_params(colors=GRAY)

# ── 图4: 滚动相关性 ──
ax4 = fig.add_subplot(3, 3, 4)
ax4.fill_between(merged.index, merged['rolling_corr'], 0,
                 where=merged['rolling_corr'] >= 0,
                 color=RED, alpha=0.3, interpolate=True)
ax4.fill_between(merged.index, merged['rolling_corr'], 0,
                 where=merged['rolling_corr'] < 0,
                 color=GREEN, alpha=0.3, interpolate=True)
ax4.plot(merged.index, merged['rolling_corr'], color=WHITE, linewidth=1.2)
ax4.axhline(y=pearson_r, color=GOLD, linewidth=1.5, linestyle='--',
            label=f'整体 ρ={pearson_r:.3f}')
ax4.axhline(y=0, color=GRAY, linewidth=0.5)
ax4.set_title(f'滚动相关性 ({window}期 = 4小时)', fontsize=13, fontweight='bold', color=WHITE)
ax4.legend(loc='lower left', fontsize=9, framealpha=0.7)
ax4.set_ylabel('Pearson ρ', fontsize=11, color=GRAY)
ax4.set_ylim(-1, 1)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
ax4.grid(True, alpha=0.2)
ax4.tick_params(colors=GRAY)

# ── 图5: 收益率分布对比 ──
ax5 = fig.add_subplot(3, 3, 5)
bins = 50
ax5.hist(merged['btc_ret'], bins=bins, alpha=0.5, color=ORANGE, label='BTC',
         density=True, edgecolor='none')
ax5.hist(merged['eth_ret'], bins=bins, alpha=0.5, color=CYAN, label='ETH',
         density=True, edgecolor='none')
# KDE
from scipy.stats import gaussian_kde
btc_kde = gaussian_kde(merged['btc_ret'])
eth_kde = gaussian_kde(merged['eth_ret'])
x_kde = np.linspace(merged['btc_ret'].min(), merged['btc_ret'].max(), 200)
ax5.plot(x_kde, btc_kde(x_kde), color=ORANGE, linewidth=1.8)
ax5.plot(x_kde, eth_kde(x_kde), color=CYAN, linewidth=1.8)
ax5.axvline(x=0, color=GRAY, linewidth=0.5, linestyle='--')
ax5.set_title('5分钟收益率分布', fontsize=13, fontweight='bold', color=WHITE)
ax5.legend(loc='upper right', fontsize=9)
ax5.set_xlabel('收益率 (%)', fontsize=11, color=GRAY)
ax5.set_ylabel('概率密度', fontsize=11, color=GRAY)
ax5.tick_params(colors=GRAY)

# ── 图6: BTC/ETH 收益率的 2D 直方图 (热力图) ──
ax6 = fig.add_subplot(3, 3, 6)
h = ax6.hist2d(merged['btc_ret'], merged['eth_ret'],
               bins=[40, 40], cmap='YlOrRd', alpha=0.9,
               range=[[merged['btc_ret'].quantile(0.01), merged['btc_ret'].quantile(0.99)],
                      [merged['eth_ret'].quantile(0.01), merged['eth_ret'].quantile(0.99)]])
plt.colorbar(h[3], ax=ax6, label='频次')
ax6.plot(x_range, slope * x_range + intercept, color=BLUE, linewidth=2, linestyle='--',
         label=f'β={beta:.2f}')
ax6.axhline(y=0, color=GRAY, linewidth=0.5)
ax6.axvline(x=0, color=GRAY, linewidth=0.5)
ax6.set_xlabel('BTC 收益率 (%)', fontsize=11, color=GRAY)
ax6.set_ylabel('ETH 收益率 (%)', fontsize=11, color=GRAY)
ax6.set_title('收益率联合分布热力图', fontsize=13, fontweight='bold', color=WHITE)
ax6.legend(loc='upper left', fontsize=8, framealpha=0.7)

# ── 图7: 同向率随时间变化 ──
ax7 = fig.add_subplot(3, 3, 7)
# 按小时聚合方向一致率
merged['hour'] = merged.index.hour
hourly_same = merged.groupby('hour')['same_dir'].mean() * 100
hours = hourly_same.index.tolist()
colors_bar = [RED if v >= same_pct else GREEN for v in hourly_same.values]
ax7.bar(range(len(hours)), hourly_same.values, color=colors_bar, edgecolor='#1a1a2e', alpha=0.8)
ax7.axhline(y=same_pct, color=GOLD, linewidth=1.5, linestyle='--',
            label=f'总体={same_pct:.1f}%')
ax7.set_xticks(range(len(hours)))
ax7.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, fontsize=8)
ax7.set_title('各时段方向一致率 (北京时间)', fontsize=13, fontweight='bold', color=WHITE)
ax7.legend(loc='lower left', fontsize=9)
ax7.set_ylabel('一致率 (%)', fontsize=11, color=GRAY)
ax7.set_ylim(0, 100)
ax7.tick_params(colors=GRAY)

# ── 图8: 波动率对比 ──
ax8 = fig.add_subplot(3, 3, 8)
# 滚动波动率 (24期 = 2小时)
merged['btc_vol'] = merged['btc_ret'].rolling(24).std()
merged['eth_vol'] = merged['eth_ret'].rolling(24).std()
ax8.plot(merged.index, merged['btc_vol'], color=ORANGE, linewidth=1, alpha=0.8, label='BTC')
ax8.plot(merged.index, merged['eth_vol'], color=CYAN, linewidth=1, alpha=0.8, label='ETH')
ax8.set_title('滚动波动率 (24期=2小时)', fontsize=13, fontweight='bold', color=WHITE)
ax8.legend(loc='upper left', fontsize=9)
ax8.set_ylabel('波动率(std %)', fontsize=11, color=GRAY)
ax8.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
ax8.grid(True, alpha=0.2)
ax8.tick_params(colors=GRAY)

# ── 图9: 统计摘要面板 ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
╔══════════════════════════════════════╗
║         核心统计结果摘要              ║
╠══════════════════════════════════════╣
║                                      ║
║  📊 样本量: {len(merged):,} 个 5 分钟 K 线         ║
║  📅 时段: {merged.index[0].strftime('%m/%d %H:%M')} ~ {merged.index[-1].strftime('%m/%d %H:%M')}  ║
║                                      ║
║  🔗 Pearson  相关系数:  {pearson_r:>7.4f}         ║
║  🔗 Spearman 相关系数:  {spearman_r:>7.4f}         ║
║  📈 R² 决定系数:         {r_value**2:>7.4f}         ║
║  📐 Beta (ETH/BTC):      {beta:>7.4f}         ║
║                                      ║
║  ✅ 方向一致率 (全部):  {same_pct:>6.1f}%          ║
║  ✅ 方向一致率 (非零):  {same_nonzero_pct:>6.1f}%          ║
║                                      ║
║  🔴 同涨: {btc_up_eth_up:>5} ({dir_pcts[0]:.1f}%)                    ║
║  🟢 同跌: {btc_down_eth_down:>5} ({dir_pcts[1]:.1f}%)                    ║
║  🟠 背离: {btc_up_eth_down + btc_down_eth_up:>5} ({dir_pcts[2] + dir_pcts[3]:.1f}%)                    ║
║                                      ║
║  📊 BTC 波动: {merged['btc_ret'].std():.4f}% | ETH 波动: {merged['eth_ret'].std():.4f}%    ║
║                                      ║
╠══════════════════════════════════════╣
║  {'🟢 高度一致' if pearson_r > 0.7 else '🟡 中度相关' if pearson_r > 0.4 else '🔴 弱相关'}                                  ║
╚══════════════════════════════════════╝
"""
ax9.text(0.5, 0.5, summary_text, transform=ax9.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', color=WHITE,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor=GOLD, alpha=0.95))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('btc_eth_correlation_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print(f"\n图表已保存: btc_eth_correlation_analysis.png")

# ─── 7. 极端行情分析 ───
print(f"\n{'='*40}")
print(f"极端行情一致性分析")
print(f"{'='*40}")

# 定义极端行情: 收益率超过 ±0.5%
extreme_mask = (merged['btc_ret'].abs() > 0.5) | (merged['eth_ret'].abs() > 0.5)
extreme = merged[extreme_mask]

if len(extreme) > 0:
    extreme_same = extreme['same_dir'].mean() * 100
    print(f"极端行情次数 (>|0.5%|): {len(extreme)} ({len(extreme)/len(merged)*100:.1f}%)")
    print(f"极端行情方向一致率: {extreme_same:.1f}%")

    # BTC极端时ETH跟随
    btc_extreme = merged['btc_ret'].abs() > 0.5
    btc_ext_up = (merged['btc_ret'] > 0.5)
    btc_ext_down = (merged['btc_ret'] < -0.5)

    if btc_ext_up.sum() > 0:
        eth_follow_up = ((merged.loc[btc_ext_up, 'eth_ret'] > 0).sum() /
                         btc_ext_up.sum() * 100)
        print(f"BTC大涨>0.5%时 ETH跟随上涨: {eth_follow_up:.1f}% (n={btc_ext_up.sum()})")

    if btc_ext_down.sum() > 0:
        eth_follow_down = ((merged.loc[btc_ext_down, 'eth_ret'] < 0).sum() /
                           btc_ext_down.sum() * 100)
        print(f"BTC大跌<-0.5%时 ETH跟随下跌: {eth_follow_down:.1f}% (n={btc_ext_down.sum()})")

    # ETH极端时BTC跟随
    eth_extreme_up = (merged['eth_ret'] > 0.5)
    eth_extreme_down = (merged['eth_ret'] < -0.5)

    if eth_extreme_up.sum() > 0:
        btc_follow_up = ((merged.loc[eth_extreme_up, 'btc_ret'] > 0).sum() /
                         eth_extreme_up.sum() * 100)
        print(f"ETH大涨>0.5%时 BTC跟随上涨: {btc_follow_up:.1f}% (n={eth_extreme_up.sum()})")

    if eth_extreme_down.sum() > 0:
        btc_follow_down = ((merged.loc[eth_extreme_down, 'btc_ret'] < 0).sum() /
                           eth_extreme_down.sum() * 100)
        print(f"ETH大跌<-0.5%时 BTC跟随下跌: {btc_follow_down:.1f}% (n={eth_extreme_down.sum()})")

# ─── 8. 结论 ───
print(f"\n{'='*60}")
print(f"结论")
print(f"{'='*60}")

if pearson_r > 0.7:
    level = "高度相关"
    detail = "BTC和ETH在5分钟级别价格变动高度一致,几乎可以视为同一风险敞口"
elif pearson_r > 0.5:
    level = "显著相关"
    detail = "两者存在明显的联动关系,但仍有独立行情空间"
elif pearson_r > 0.3:
    level = "中度相关"
    detail = "有一定联动但不足以做跨品种套利"
else:
    level = "弱相关"
    detail = "两者的5分钟级别价格变动关系较弱"

print(f"相关程度: {level}")
print(f"详细评估: {detail}")
print(f"方向一致率 {same_pct:.1f}% 验证了上述判断")
print(f"Beta={beta:.2f} 说明ETH对BTC的敏感度为 {beta:.2f}倍")
print(f"\n图表文件: btc_eth_correlation_analysis.png")

plt.close()
