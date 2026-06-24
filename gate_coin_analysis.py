"""
Gate 交易所币种分类分析脚本
分类标准：
- 主流稳健币：高成交额 + 中低跳变/波动 + 趋势稳定
- 中间币：介于中间，流动性Beta
- 妖币：高成交额 + 高波动 + 放量/跳变明显
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = Path("E:/source/freqtrade/user_data/data/gate")

# 只取标准1分钟K线，排除 _USDT_USDT 这种异常命名
def get_1m_files():
    files = []
    for f in DATA_DIR.iterdir():
        if f.name.endswith("-1m.feather") and "_USDT_USDT" not in f.name:
            files.append(f)
    return files

def compute_coin_metrics(filepath):
    """计算单个币种的各项指标"""
    try:
        df = pd.read_feather(filepath)
        if df.empty or len(df) < 1000:
            return None

        # 基本列检查
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        # freqtrade feather 可能列名不同，检查一下
        col_map = {}
        for col in df.columns:
            for req in required:
                if req.lower() in col.lower():
                    col_map[req] = col
        if len(col_map) < 6:
            return None

        symbol = filepath.name.replace("_USDT-1m.feather", "").replace("-1m.feather", "")

        open_col = col_map['open']
        close_col = col_map['close']
        high_col = col_map['high']
        low_col = col_map['low']
        vol_col = col_map['volume']
        date_col = col_map['date']

        df['date'] = pd.to_datetime(df[date_col])
        df = df.sort_values('date').reset_index(drop=True)

        # 过滤掉成交量为0的行（无交易时段）
        df_active = df[df[vol_col] > 0].copy()
        if len(df_active) < 500:
            return None

        # === 1. 成交额指标 ===
        # 日均成交额（最近30天）
        df_active['trade_amount'] = df_active[close_col] * df_active[vol_col]
        df_active['day'] = df_active['date'].dt.date

        daily_amount = df_active.groupby('day')['trade_amount'].sum()
        # 最近30天日均
        recent_days = daily_amount.tail(30)
        avg_daily_amount = recent_days.mean()
        # 全期日均
        avg_daily_amount_all = daily_amount.mean()

        # === 2. 波动率指标 ===
        # 分钟收益率
        df_active['return'] = df_active[close_col].pct_change()
        returns = df_active['return'].dropna()

        # 日波动率（分钟收益标准差 × sqrt(1440)）
        min_vol = returns.std()
        daily_vol = min_vol * np.sqrt(1440)

        # 日内最大跳变（单分钟最大涨跌幅绝对值）
        max_jump = returns.abs().max()
        # 跳变频率：单分钟涨跌幅超过2%的占比
        jump_threshold = 0.02
        jump_freq = (returns.abs() > jump_threshold).mean()

        # === 3. 趋势稳定性 ===
        # 日K线
        daily_df = df_active.groupby('day').agg({
            open_col: 'first',
            high_col: 'max',
            low_col: 'min',
            close_col: 'last',
            vol_col: 'sum',
            'trade_amount': 'sum'
        }).reset_index()

        daily_close = daily_df[close_col]
        # 日收益率
        daily_returns = daily_close.pct_change().dropna()

        # 价格趋势线性度（日收盘价与线性拟合的R²）
        x = np.arange(len(daily_close))
        if len(x) < 10:
            trend_r2 = 0
        else:
            slope, intercept = np.polyfit(x, daily_close.values, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((daily_close.values - predicted) ** 2)
            ss_tot = np.sum((daily_close.values - np.mean(daily_close.values)) ** 2)
            trend_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # === 4. 放量突变指标 ===
        # 日成交额变异系数
        amount_cv = daily_amount.std() / daily_amount.mean() if daily_amount.mean() > 0 else 999

        # 放量突变频率（日成交额超过均值2倍的天数占比）
        volume_spike_freq = (daily_amount > daily_amount.mean() * 2).mean()

        # === 5. 高低价比（日内极差比）===
        daily_df['intraday_range'] = (daily_df[high_col] - daily_df[low_col]) / daily_df[close_col]
        avg_intraday_range = daily_df['intraday_range'].mean()

        # === 6. 数据天数 ===
        n_days = len(daily_amount)

        # 去掉前缀数字和后缀数字的干扰（如 2Z 等）
        clean_symbol = symbol.replace("_USDT", "")

        return {
            'symbol': clean_symbol,
            'avg_daily_amount': avg_daily_amount,
            'avg_daily_amount_all': avg_daily_amount_all,
            'daily_volatility': daily_vol,
            'max_jump_pct': max_jump * 100,
            'jump_freq_2pct': jump_freq * 100,
            'trend_r2': trend_r2,
            'amount_cv': amount_cv,
            'volume_spike_freq': volume_spike_freq * 100,
            'avg_intraday_range_pct': avg_intraday_range * 100,
            'n_days': n_days,
            'n_active_minutes': len(df_active),
            'recent_30d_avg_amount': recent_days.mean() if len(recent_days) > 0 else 0,
        }
    except Exception as e:
        return None

# 稳定币/锚定币名单（波动率极低，不是真正的"稳健币"）
STABLECOINS = {
    'FDUSD', 'USDC', 'USD1', 'TUSD', 'USDP', 'GUSD', 'PAXG', 'XAUT',
    'DAI', 'USDT', 'BUSD', 'FRAX', 'LUSD', 'USDD', 'CUSD',
}

def classify_coins(metrics_list):
    """根据指标对币种进行分类"""
    df = pd.DataFrame(metrics_list)

    # 过滤掉数据天数太少的币种（< 7天）
    df = df[df['n_days'] >= 7].copy()

    # 过滤掉稳定币/锚定币（日波动率 < 1% 的，或已知稳定币名单）
    df = df[(df['daily_volatility'] >= 0.01) & (~df['symbol'].isin(STABLECOINS))].copy()

    if len(df) == 0:
        return df

    # === 排名百分位计算 ===
    # 成交额排名（越高越大）
    df['amount_rank'] = df['avg_daily_amount'].rank(pct=True)
    # 波动率排名（越高越波动）
    df['vol_rank'] = df['daily_volatility'].rank(pct=True)
    # 跳变频率排名
    df['jump_rank'] = df['jump_freq_2pct'].rank(pct=True)
    # 放量突变频率排名
    df['spike_rank'] = df['volume_spike_freq'].rank(pct=True)
    # 趋势稳定性排名（R²越高越稳定）
    df['trend_rank'] = df['trend_r2'].rank(pct=True)
    # 日内极差排名
    df['range_rank'] = df['avg_intraday_range_pct'].rank(pct=True)

    # === 综合分类评分 ===
    # 稳健币评分：高成交额 + 低波动 + 低跳变 + 高趋势R²
    df['stable_score'] = (
        df['amount_rank'] * 0.25 +      # 成交额要高
        (1 - df['vol_rank']) * 0.25 +    # 波动要低
        (1 - df['jump_rank']) * 0.20 +   # 跳变要少
        df['trend_rank'] * 0.15 +        # 趋势要稳
        (1 - df['spike_rank']) * 0.15    # 放量突变要少
    )

    # 妖币评分：高成交额 + 高波动 + 高跳变 + 高放量突变
    df['wild_score'] = (
        df['amount_rank'] * 0.20 +       # 成交额要高
        df['vol_rank'] * 0.25 +          # 波动要高
        df['jump_rank'] * 0.25 +         # 跳变要多
        df['spike_rank'] * 0.20 +        # 放量突变要多
        df['range_rank'] * 0.10          # 日内极差要大
    )

    # 中间币评分：各项中等，成交额中等，波动中等
    # 用"偏离中间的程度"来衡量，越接近50%越好
    df['mid_score'] = (
        (1 - abs(df['amount_rank'] - 0.5) * 2) * 0.30 +  # 成交额居中
        (1 - abs(df['vol_rank'] - 0.5) * 2) * 0.25 +     # 波动居中
        (1 - abs(df['jump_rank'] - 0.5) * 2) * 0.20 +    # 跽变居中
        (1 - abs(df['trend_rank'] - 0.5) * 2) * 0.15 +   # 趋势居中
        (1 - abs(df['spike_rank'] - 0.5) * 2) * 0.10     # 放量居中
    )

    # 分类逻辑
    # 稳健币：成交额rank >= 0.5 + vol_rank <= 0.35 + 跳变少 + 趋势稳
    # 妖币：成交额rank >= 0.4 + vol_rank >= 0.65 + 高跳变 + 放量突变
    # 中间币：各项指标居中，成交额rank >= 0.25

    df['category'] = '未分类'

    # 稳健币：成交额要足够高 + 波动低 + 跳变少 + 趋势稳
    stable_mask = (
        (df['stable_score'] >= 0.60) &
        (df['amount_rank'] >= 0.50) &     # 成交额至少前50%
        (df['vol_rank'] <= 0.40) &         # 波动率在后40%
        (df['jump_freq_2pct'] <= 0.01)     # 2%跳变频率极低
    )
    df.loc[stable_mask, 'category'] = '主流稳健币'

    # 妖币：成交额也要有一定量（不然只是低流动性的杂币）+ 高波动 + 高跳变
    wild_mask = (
        (df['wild_score'] >= 0.55) &
        (df['amount_rank'] >= 0.40) &       # 成交额至少前40%
        (df['vol_rank'] >= 0.60) &           # 波动率在前60%
        (df['jump_freq_2pct'] >= 0.02)       # 跳变频率有一定量
    )
    df.loc[wild_mask, 'category'] = '妖币'

    # 中间币：不在以上两类 + 成交额居中 + 各项指标居中
    mid_mask = (
        (df['category'] == '未分类') &
        (df['mid_score'] >= 0.30) &
        (df['amount_rank'] >= 0.25) &        # 成交额至少前25%
        (df['amount_rank'] <= 0.85)          # 但不能太高（否则可能是稳健币）
    )
    df.loc[mid_mask, 'category'] = '中间币'

    return df

def main():
    files = get_1m_files()
    print(f"Found {len(files)} 1-minute feather files")

    # 并行计算指标
    metrics_list = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(compute_coin_metrics, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                metrics_list.append(result)

    print(f"Successfully computed metrics for {len(metrics_list)} coins")

    # 分类
    df = classify_coins(metrics_list)

    # 按类别输出
    categories = ['主流稳健币', '中间币', '妖币']

    results = {}
    for cat in categories:
        cat_df = df[df['category'] == cat].copy()
        if cat == '主流稳健币':
            cat_df = cat_df.sort_values('stable_score', ascending=False)
        elif cat == '妖币':
            cat_df = cat_df.sort_values('wild_score', ascending=False)
        else:
            cat_df = cat_df.sort_values('mid_score', ascending=False)

        top5 = cat_df.head(5)
        results[cat] = top5[['symbol', 'avg_daily_amount', 'daily_volatility',
                             'max_jump_pct', 'jump_freq_2pct', 'trend_r2',
                             'amount_cv', 'volume_spike_freq', 'avg_intraday_range_pct',
                             'stable_score', 'wild_score', 'mid_score', 'n_days']].to_dict('records')
        print(f"\n=== {cat} Top 5 ===")
        for row in top5.itertuples():
            print(f"  {row.symbol:12s} | 日均成交额: ${row.avg_daily_amount:>12,.0f} | 日波动率: {row.daily_volatility:.2%} | "
                  f"跳变2%频: {row.jump_freq_2pct:.3f}% | 趋势R²: {row.trend_r2:.3f} | "
                  f"放量突变: {row.volume_spike_freq:.2f}% | 日内极差: {row.avg_intraday_range_pct:.2f}% | "
                  f"稳健={row.stable_score:.3f} 妖={row.wild_score:.3f} 中={row.mid_score:.3f}")

    # 保存完整结果
    output_path = DATA_DIR.parent.parent / "gate_coin_classification.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # 保存完整分类结果
        full_df = df.sort_values('category')
        output_data = {
            'summary': {
                'total_coins': len(df),
                'stable_count': len(df[df['category'] == '主流稳健币']),
                'mid_count': len(df[df['category'] == '中间币']),
                'wild_count': len(df[df['category'] == '妖币']),
                'uncategorized': len(df[df['category'] == '未分类']),
            },
            'top5_stable': results.get('主流稳健币', []),
            'top5_mid': results.get('中间币', []),
            'top5_wild': results.get('妖币', []),
            'all_coins': full_df.to_dict('records')
        }
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # 也输出各分类的详细列表（前10）
    for cat in categories:
        cat_df = df[df['category'] == cat].copy()
        if cat == '主流稳健币':
            cat_df = cat_df.sort_values('stable_score', ascending=False)
        elif cat == '妖币':
            cat_df = cat_df.sort_values('wild_score', ascending=False)
        else:
            cat_df = cat_df.sort_values('mid_score', ascending=False)

        print(f"\n--- {cat} (共{len(cat_df)}个，前10) ---")
        for row in cat_df.head(10).itertuples():
            print(f"  {row.symbol:12s} | 成交额: ${row.avg_daily_amount:>12,.0f} | 波动率: {row.daily_volatility:.2%} | "
                  f"跳变: {row.jump_freq_2pct:.3f}% | R²: {row.trend_r2:.3f}")

if __name__ == '__main__':
    main()
