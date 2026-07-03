"""
多时间框架妖币信号分析
========================
从 1m 数据出发, 重采样到 5m/15m/30m/1h/4h/12h,
在每个时间框架上分析妖币 (>20%日内涨幅) 启动前的特征指纹,
找出最佳决策刻度。
"""

import os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("E:/source/freqtrade/user_data/data/gate")
OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/moji-coin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标时间框架
TIMEFRAMES = {
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '4h': '4h',
    '12h': '12h',
}

# 妖币阈值
MOJI_THRESHOLD = 20.0  # 日内最大涨幅 >= 20%


def load_all_1m_data():
    """加载全部 1m 数据, 这是最精确的检测粒度"""
    files = sorted(DATA_DIR.glob("*-1m.feather"))
    coin_data = {}
    for f in files:
        symbol = f.name.replace("-1m.feather", "")
        df = pd.read_feather(f)
        df = df.set_index('date').sort_index()
        coin_data[symbol] = df
    return coin_data


def resample_to_timeframes(df_1m):
    """从 1m OHLCV 数据重采样到所有目标时间框架"""
    results = {}
    for tf_name, tf_rule in TIMEFRAMES.items():
        # 如果已有原始数据文件 (5m/15m/1h), 优先使用
        fpath = DATA_DIR / f"{df_1m.attrs.get('symbol', '')}-{tf_name}.feather"
        # 统一重采样
        resampled = df_1m.resample(tf_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()
        results[tf_name] = resampled
    return results


def find_pump_events_1m(df_1m):
    """
    用 1m 数据精确检测日内涨幅 >= 20% 的妖币事件.
    返回: [(symbol, pump_day, pump_start_time, max_gain_pct), ...]
    """
    df = df_1m.copy()
    df['day'] = df.index.date
    events = []

    for day, grp in df.groupby('day'):
        if len(grp) < 60:  # 至少1小时数据
            continue
        day_open = grp['open'].iloc[0]
        cummax_high = grp['high'].cummax()
        max_gain = (cummax_high.max() - day_open) / day_open * 100

        if max_gain >= MOJI_THRESHOLD:
            # 找泵启动时间 (涨幅首次超过 5%)
            cum_gain = (cummax_high - day_open) / day_open * 100
            trigger_idx = cum_gain[cum_gain >= 5.0].index
            if len(trigger_idx) > 0:
                pump_start = trigger_idx[0]
                events.append({
                    'symbol': df_1m.attrs.get('symbol', 'unknown'),
                    'day': day,
                    'pump_start': pump_start,
                    'max_gain_pct': max_gain,
                })
    return events


def compute_pre_pump_features(df_tf, pump_start, lookback_periods):
    """
    计算 pump_start 之前的特征.
    lookback_periods: 不同时间框架使用不同 bar 数对应的回看窗口.
    """
    # 取 pump_start 之前的 lookback_periods 根 bar
    pump_loc = df_tf.index.get_indexer([pump_start], method='ffill')[0]
    start_loc = max(0, pump_loc - lookback_periods)
    pre = df_tf.iloc[start_loc:pump_loc]

    if len(pre) < max(3, lookback_periods // 4):
        return None

    f = {}
    f['n_bars'] = len(pre)
    f['pre_open'] = pre['open'].iloc[0]
    f['pre_close'] = pre['close'].iloc[-1]
    f['pre_change_pct'] = (f['pre_close'] - f['pre_open']) / f['pre_open'] * 100

    f['pre_high'] = pre['high'].max()
    f['pre_low'] = pre['low'].min()
    f['pre_range_pct'] = (f['pre_high'] - f['pre_low']) / f['pre_low'] * 100

    # 波动率 (年化, 直接根据 bar 间隔估算)
    lr = np.log(pre['close'] / pre['close'].shift(1)).dropna()
    if len(pre) > 1 and len(lr) > 0:
        time_span_h = (pre.index[-1] - pre.index[0]).total_seconds() / 3600
        if time_span_h > 0:
            bars_per_year = 365 * 24 / (time_span_h / len(pre))
            f['pre_volatility'] = lr.std() * np.sqrt(bars_per_year)
        else:
            f['pre_volatility'] = lr.std()
    else:
        f['pre_volatility'] = 0

    # 量比: 后半段 / 前半段
    half = len(pre) // 2
    if half >= 2:
        recent_vol = pre['volume'].iloc[-half:].mean()
        early_vol = pre['volume'].iloc[:half].mean()
        f['volume_ratio'] = recent_vol / early_vol if early_vol > 0 else 0
    else:
        f['volume_ratio'] = 1.0

    # SMA斜率
    sma_n = max(3, lookback_periods // 6)
    sma = pre['close'].rolling(sma_n).mean()
    if len(sma.dropna()) >= 2:
        sma_start = sma.dropna().iloc[0]
        sma_end = sma.dropna().iloc[-1]
        f['sma_slope_pct'] = (sma_end - sma_start) / sma_start * 100 if sma_start > 0 else 0
    else:
        f['sma_slope_pct'] = 0

    # 价格阶梯 (最后 6 根 bar)
    c = pre['close']
    if len(c) >= 7:
        steps = ((c.iloc[-1] >= c.iloc[-2]) + (c.iloc[-2] >= c.iloc[-3]) +
                 (c.iloc[-3] >= c.iloc[-4]) + (c.iloc[-4] >= c.iloc[-5]) +
                 (c.iloc[-5] >= c.iloc[-6]))
        f['staircase_score'] = steps / 5.0
    else:
        f['staircase_score'] = 0

    # 成交量集中度 (最后2根bar占总量的比例)
    if len(pre) >= 2:
        f['vol_concentration'] = (pre['volume'].iloc[-1] + pre['volume'].iloc[-2]) / max(pre['volume'].sum(), 1)
    else:
        f['vol_concentration'] = 0

    return f


def main():
    print("=" * 70)
    print("📊 多时间框架妖币信号分析")
    print("=" * 70)
    print()

    # Step 1: 加载 1m 数据
    print("[1/5] 加载 1m 数据...")
    coin_data_1m = load_all_1m_data()
    print(f"  加载了 {len(coin_data_1m)} 个币种的 1m 数据")

    # Step 2: 找全部妖币事件
    print("\n[2/5] 检测妖币事件 (日内涨幅 >= 20%)...")
    all_events = []
    for symbol, df_1m in coin_data_1m.items():
        df_1m.attrs['symbol'] = symbol
        events = find_pump_events_1m(df_1m)
        all_events.extend(events)

    print(f"  找到 {len(all_events)} 个妖币事件")
    events_df = pd.DataFrame(all_events)
    print(f"  涉及 {events_df['symbol'].nunique()} 个唯一币种")
    print(f"  日期范围: {events_df['day'].min()} ~ {events_df['day'].max()}")
    print(f"  最大涨幅: {events_df['max_gain_pct'].max():.1f}%")

    # Step 3: 每个时间框架分析
    print("\n[3/5] 逐时间框架分析预泵特征...")

    # 时间框架 → lookback (bars) 配置
    # 我们考察不同回看窗口: 固定 6h 和固定 24h 的 bar 数
    # 时间框架 → lookback (bars) 配置
    # 4h/12h 用 24h lookback 因为 6h 不够
    # 4h/12h 用 24h lookback 因为 6h bar 数不够, 最低需要 3根 bar
    tf_configs = {
        '5m':  {'lookback_6h': 72, 'lookback_24h': 288, 'rule': '5min'},
        '15m': {'lookback_6h': 24, 'lookback_24h': 96,  'rule': '15min'},
        '30m': {'lookback_6h': 12, 'lookback_24h': 48,  'rule': '30min'},
        '1h':  {'lookback_6h': 6,  'lookback_24h': 24,  'rule': '1h'},
        '4h':  {'lookback_12h': 3, 'lookback_24h': 6,   'rule': '4h'},
        '12h': {'lookback_12h': 1, 'lookback_24h': 2,   'rule': '12h'},
    }

    all_tf_features = {}  # tf_name -> {'feat_df', 'moji_mean', 'control_mean', ...}

    # 先建立 5m 数据 (已有)
    print("\n  正在为每个时间框架重采样并计算特征...")

    # 拿到所有已有数据
    for tf_name, cfg in tf_configs.items():
        print(f"\n  --- {tf_name} ---")

        # 如果有预计算的数据文件, 直接用
        data_pattern = f"*-{tf_name}.feather"
        has_files = list(DATA_DIR.glob(data_pattern))

        if has_files and tf_name in ['5m', '15m', '1h']:
            # 用已有数据
            print(f"    使用已有 {len(has_files)} 个 feather 文件")
            tf_data = {}
            for fp in has_files:
                sym = fp.name.replace(f"-{tf_name}.feather", "")
                df = pd.read_feather(fp).set_index('date').sort_index()
                tf_data[sym] = df
        else:
            # 需要重采样
            print(f"    从 1m 重采样...")
            tf_data = {}
            for symbol, df_1m in coin_data_1m.items():
                resampled = df_1m.resample(cfg['rule']).agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                tf_data[symbol] = resampled

        # 计算预泵特征 (用 6h lookback)
        moji_features = []
        control_features = []

        for evt in all_events:
            symbol = evt['symbol']
            pump_start = evt['pump_start']
            day = evt['day']

            if symbol not in tf_data:
                continue

            df_tf = tf_data[symbol]
            # 4h/12h 用 12h lookback, 其他用 6h
            lookback = cfg.get('lookback_12h', cfg.get('lookback_6h', 72))

            feat = compute_pre_pump_features(df_tf, pump_start, lookback)
            if feat is not None:
                feat['symbol'] = symbol
                feat['pump_day'] = str(day)
                feat['max_gain_pct'] = evt['max_gain_pct']
                feat['timeframe'] = tf_name
                moji_features.append(feat)

            # 对照组: 同一币种, 不同时间的随机样本
            # 取 pump 前 48h 的任意时间点 (避开 pump 日)
            df_no_pump = df_tf[df_tf.index.date != day]
            if len(df_no_pump) < lookback + 10:
                # 取同币种不同日的非妖币日
                all_days = sorted(set(df_tf.index.date))
                for d in all_days:
                    if d != day:
                        df_no_pump = df_tf[df_tf.index.date == d]
                        if len(df_no_pump) >= lookback + 10:
                            break

            if len(df_no_pump) >= lookback + 10:
                # 取随机起点
                rand_loc = np.random.randint(0, len(df_no_pump) - lookback - 1)
                fake_pump = df_no_pump.index[rand_loc + lookback]
                ctrl_feat = compute_pre_pump_features(df_tf, fake_pump, lookback)
                if ctrl_feat is not None:
                    ctrl_feat['symbol'] = symbol
                    ctrl_feat['timeframe'] = tf_name
                    control_features.append(ctrl_feat)

        moji_df = pd.DataFrame(moji_features)
        ctrl_df = pd.DataFrame(control_features)

        if len(moji_df) == 0 or len(ctrl_df) == 0:
            print(f"    警告: moji={len(moji_df)}, control={len(ctrl_df)}, 跳过")
            continue

        # 计算区分度
        numeric_cols = moji_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c in ctrl_df.columns and c not in
                        ['max_gain_pct', 'n_bars', 'pre_open']]

        comparison = []
        for col in numeric_cols:
            m_mean = moji_df[col].mean()
            m_med = moji_df[col].median()
            c_mean = ctrl_df[col].mean()
            c_med = ctrl_df[col].median()
            diff_pct = abs(m_mean - c_mean) / max(abs(c_mean), 1e-9) * 100 if abs(c_mean) > 1e-6 else float('inf')
            # 方向: 正值表示妖币特征更突出
            comparison.append({
                'feature': col,
                'moji_mean': m_mean,
                'moji_median': m_med,
                'control_mean': c_mean,
                'control_median': c_med,
                'diff_pct': diff_pct,
                'direction': '+' if m_mean > c_mean else '-',
            })

        comp_df = pd.DataFrame(comparison).sort_values('diff_pct', ascending=False)
        all_tf_features[tf_name] = {
            'moji_df': moji_df,
            'ctrl_df': ctrl_df,
            'comparison': comp_df,
        }

        print(f"    moji 样本: {len(moji_df)}, 对照: {len(ctrl_df)}")
        print(f"    Top 5 区分特征:")
        for _, row in comp_df.head(5).iterrows():
            print(f"      {row['feature']:25s}: {row['diff_pct']:8.0f}%  diff  "
                  f"(moji={row['moji_mean']:.4f}, ctrl={row['control_mean']:.4f})")

    # Step 4: 跨时间框架对比
    print("\n\n[4/5] 跨时间框架综合对比 (6h lookback)")
    print("=" * 70)

    # 提取每个框架的 top 特征
    summary_rows = []
    for tf_name, data in all_tf_features.items():
        comp = data['comparison']
        n_moji = len(data['moji_df'])
        for _, row in comp.head(3).iterrows():
            summary_rows.append({
                'timeframe': tf_name,
                'feature': row['feature'],
                'diff_pct': row['diff_pct'],
                'moji_mean': row['moji_mean'],
                'control_mean': row['control_mean'],
                'n_samples': n_moji,
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\n跨框架 Top 特征对比:")
    print(summary_df.to_string(index=False))

    # 关键对比: 每个框架中区分度最强的特征
    print("\n\n📊 各时间框架最强区分特征排名")
    print("-" * 50)

    tf_best = {}
    for tf_name, data in all_tf_features.items():
        best = data['comparison'].iloc[0]
        tf_best[tf_name] = {
            'feature': best['feature'],
            'diff_pct': best['diff_pct'],
            'moji_mean': best['moji_mean'],
            'control_mean': best['control_mean'],
        }
        print(f"  {tf_name:5s}: {best['feature']:20s} diff={best['diff_pct']:.0f}%  "
              f"(moji={best['moji_mean']:.4f}, ctrl={best['control_mean']:.4f})")

    # Step 5: 生成综合评分卡
    print("\n\n[5/5] 生成多时间框架选币评分卡")
    print("=" * 70)

    # 选择最佳时间框架组合
    # 找出每个时间框架下的最佳阈值
    print("\n建议选币参数 (基于各自时间框架):\n")

    for tf_name, data in all_tf_features.items():
        comp = data['comparison']
        print(f"--- {tf_name} ---")
        for _, row in comp.head(4).iterrows():
            # 推荐阈值为 moji 中位数和 ctrl 中位数的中点
            threshold = (row['moji_median'] + row['control_median']) / 2
            if abs(row['moji_median']) < 1e-6 and abs(row['control_median']) < 1e-6:
                threshold = row['moji_mean'] * 0.6
            print(f"  {row['feature']:20s}: 推荐阈值={threshold:.4f}  "
                  f"(moji={row['moji_mean']:.4f}, ctrl={row['control_mean']:.4f}, "
                  f"区分度={row['diff_pct']:.0f}%)")
        print()

    # 保存结果
    summary_df.to_csv(OUTPUT_DIR / 'multitf_summary.csv', index=False)

    # 保存每个时间框架的详细特征对比
    for tf_name, data in all_tf_features.items():
        data['comparison'].to_csv(OUTPUT_DIR / f'multitf_{tf_name}_features.csv', index=False)
        data['moji_df'].to_csv(OUTPUT_DIR / f'multitf_{tf_name}_moji.csv', index=False)

    print(f"详细数据已保存到 {OUTPUT_DIR}")

    # 最终建议
    print("\n" + "=" * 70)
    print("🎯 关键发现与建议")
    print("=" * 70)

    # 找出区分度最强的框架
    best_tf = max(tf_best.items(), key=lambda x: x[1]['diff_pct'])
    print(f"\n1. 最佳单一时间框架: {best_tf[0]} "
          f"(最强特征 {best_tf[1]['feature']}, 区分度 {best_tf[1]['diff_pct']:.0f}%)")

    # 统计各框架最强特征的区分度
    print("\n2. 区分度排序 (最强特征):")
    for tf, info in sorted(tf_best.items(), key=lambda x: x[1]['diff_pct'], reverse=True):
        bar = "█" * int(min(info['diff_pct'] / 100, 20))
        print(f"   {tf:5s}: {bar} {info['diff_pct']:.0f}% "
              f"({info['feature']})")

    # 跨框架一致性分析
    print("\n3. 哪些特征在所有时间框架中稳定出现?")
    all_features = set()
    for data in all_tf_features.values():
        all_features.update(data['comparison']['feature'].head(3).values)
    feature_tf_count = {}
    for feat in all_features:
        count = sum(1 for data in all_tf_features.values()
                    if feat in data['comparison']['feature'].values[:5])
        feature_tf_count[feat] = count

    for feat, count in sorted(feature_tf_count.items(), key=lambda x: x[1], reverse=True):
        marker = "⭐" if count >= 4 else ("✅" if count >= 3 else "  ")
        print(f"   {marker} {feat}: {count} 个时间框架 (共 {len(all_tf_features)} 个)")

    print("\n⚠️ 注意: 本分析基于 8 天数据窗口, 结果仅供参考.")


if __name__ == '__main__':
    main()
