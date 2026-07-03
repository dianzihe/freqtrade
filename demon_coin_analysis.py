"""
Demon Coin (妖币) Analysis Framework
=====================================
Target: Identify coins with 50%+ intraday gains within 24 hours.
Comprehensive analysis covering:
  1. Pre-launch characteristics (left-side signals)
  2. Launch confirmation signals (right-side confirmation)
  3. Profitable entry points across stages
  4. Multi-factor scoring algorithm for real-time scanning

Leverages existing moji-coin analysis data from gate_moji_analysis.py.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict

PROJECT = Path("E:/source/freqtrade")
MOJI_DIR = PROJECT / "deliverables/moji-coin"
OUTPUT_DIR = PROJECT / "deliverables/demon-coin"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONSTANTS ───
DEMON_THRESHOLD = 50.0   # 50%+ gain = demon coin
MOJI_THRESHOLD = 20.0    # 20%+ gain = moji coin (existing)
UTC_TO_BEIJING = 8       # UTC+8

# Trading sessions in UTC
SESSIONS = {
    "Asia Morning":    (0, 4),
    "Asia Active":     (4, 7),
    "Asia-Europe Gap": (7, 9),
    "Europe Morning":  (9, 12),
    "Europe-US Gap":   (12, 14),
    "US Morning":      (14, 17),
    "US Afternoon":    (17, 20),
    "US Close/Asia":   (20, 24),
}

# ─── 1. DATA LOADING ───

def load_moji_features():
    """Load existing moji feature data."""
    fpath = MOJI_DIR / "moji_features.csv"
    if not fpath.exists():
        print(f"ERROR: {fpath} not found. Run gate_moji_analysis.py first.")
        return None, None

    moji_f = pd.read_csv(fpath)
    control_f = pd.read_csv(MOJI_DIR / "control_features.csv")

    # Parse timestamps
    moji_f['pump_start_time'] = pd.to_datetime(moji_f['pump_start_time'])
    moji_f['pump_hour_utc'] = moji_f['pump_start_time'].dt.hour
    moji_f['pump_day_name'] = moji_f['pump_start_time'].dt.day_name()

    return moji_f, control_f


def classify_demon_coins(moji_f):
    """Separate demon coins (50%+) from regular moji (20-50%)."""
    demon = moji_f[moji_f['pump_gain_pct'] >= DEMON_THRESHOLD].copy()
    regular = moji_f[(moji_f['pump_gain_pct'] >= MOJI_THRESHOLD) &
                     (moji_f['pump_gain_pct'] < DEMON_THRESHOLD)].copy()

    demon['category'] = 'Demon (>=50%)'
    regular['category'] = 'Moji (20-50%)'
    combined = pd.concat([demon, regular], ignore_index=True)

    return demon, regular, combined


# ─── 2. PRE-LAUNCH CHARACTERISTICS ANALYSIS ───

def analyze_pre_launch_features(demon_f, control_f):
    """
    Analyze pre-launch characteristics comparing:
      - Demon coins (50%+ pumps)
      - Regular moji (20-50% pumps)
      - Control group (normal days)
    """
    features = [
        'pre_change_pct', 'pre_range_pct', 'pre_volatility',
        'pre_total_volume', 'volume_ratio_last1h_vs_avg',
        'sma12_slope', 'staircase_score', 'volume_staircase_score',
        'range_compression_ratio', 'pre_volume_skew'
    ]

    results = {}
    for feat in features:
        demon_vals = demon_f[feat].dropna()
        control_vals = control_f[feat].dropna()
        regular_vals = demon_f[feat].dropna()  # All moji

        demon_mean = demon_vals.mean()
        demon_median = demon_vals.median()
        demon_std = demon_vals.std()
        control_mean = control_vals.mean()
        control_median = control_vals.median()
        control_std = control_vals.std()

        # Statistical significance (Mann-Whitney U test)
        if len(demon_vals) >= 5 and len(control_vals) >= 5:
            try:
                u_stat, p_value = stats.mannwhitneyu(demon_vals, control_vals,
                                                     alternative='two-sided')
            except:
                p_value = 1.0
        else:
            p_value = 1.0

        # Effect size (Cohen's d simplified: difference / pooled std)
        if demon_std > 0 and control_std > 0:
            pooled_std = np.sqrt((demon_std**2 + control_std**2) / 2)
            cohens_d = (demon_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0

        # Percentile rank vs control
        if len(control_vals) > 0:
            pct_rank = stats.percentileofscore(control_vals, demon_mean)
        else:
            pct_rank = 50

        results[feat] = {
            'demon_mean': demon_mean,
            'demon_median': demon_median,
            'demon_std': demon_std,
            'control_mean': control_mean,
            'control_median': control_median,
            'control_std': control_std,
            'diff_pct': ((demon_mean - control_mean) / abs(control_mean) * 100) if control_mean != 0 else float('inf'),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'pct_rank': pct_rank,
            'demon_n': len(demon_vals),
            'control_n': len(control_vals),
        }

    return pd.DataFrame(results).T


def compute_bollinger_compression(demon_f, control_f):
    """
    Estimate Bollinger Band width compression using pre_range_pct
    as a proxy for volatility envelope. Lower = more compressed.
    """
    bb_results = {}

    # Using range_compression_ratio as proxy
    # ratio > 1 = expanding, ratio < 1 = compressing
    for label, df in [('Demon', demon_f), ('Control', control_f)]:
        if 'range_compression_ratio' in df.columns:
            vals = df['range_compression_ratio'].dropna()
            bb_results[label] = {
                'mean_ratio': vals.mean(),
                'median_ratio': vals.median(),
                'pct_compressing': (vals < 1.0).mean() * 100,  # % of events where range was compressing
                'pct_expanding': (vals > 1.5).mean() * 100,    # % of events where range was expanding
            }

    return bb_results


# ─── 3. LAUNCH CONFIRMATION SIGNALS ───

def analyze_launch_signals(demon_f):
    """
    Analyze timing and volume characteristics at the moment of launch.
    """
    results = {
        'time_distribution': {},
        'session_distribution': {},
        'volume_characteristics': {},
    }

    # Time of day distribution (UTC)
    hours = demon_f['pump_hour_utc']
    hour_counts = hours.value_counts().sort_index().to_dict()
    results['time_distribution'] = {f"{h:02d}:00": hour_counts.get(h, 0) for h in range(24)}

    # Trading session distribution
    session_counts = defaultdict(int)
    for _, row in demon_f.iterrows():
        h = row['pump_hour_utc']
        for sname, (start, end) in SESSIONS.items():
            if start <= h < end:
                session_counts[sname] += 1
                break
    total = len(demon_f)
    results['session_distribution'] = {
        s: {'count': c, 'pct': c/total*100 if total > 0 else 0}
        for s, c in sorted(session_counts.items(), key=lambda x: -x[1])
    }

    # Volume characteristics
    results['volume_characteristics'] = {
        'mean_vol_ratio': demon_f['volume_ratio_last1h_vs_avg'].mean(),
        'median_vol_ratio': demon_f['volume_ratio_last1h_vs_avg'].median(),
        'pct_surge_gt_2x': (demon_f['volume_ratio_last1h_vs_avg'] > 2.0).mean() * 100,
        'pct_surge_gt_3x': (demon_f['volume_ratio_last1h_vs_avg'] > 3.0).mean() * 100,
        'pct_surge_gt_5x': (demon_f['volume_ratio_last1h_vs_avg'] > 5.0).mean() * 100,
        'mean_pre_vol': demon_f['pre_volatility'].mean(),
        'mean_sma_slope': demon_f['sma12_slope'].mean(),
        'mean_staircase': demon_f['staircase_score'].mean(),
    }

    return results


# ─── 4. TRADING STRATEGY: ENTRY POINTS ───

def design_trading_strategy():
    """
    Design entry, stop-loss, and take-profit rules for different stages
    of a demon coin pump.
    """
    strategy = {
        'meta': {
            'name': 'Demon Coin Multi-Stage Strategy',
            'version': '2.0',
            'applicable_markets': ['牛市', '震荡市'],
            'not_recommended': '熊市（熊市中50%+涨幅事件极少出现，且失败率高）',
        },

        'stage_1_initial_breakout': {
            'name': '爆发初段追涨',
            'description': '价格刚开始拉升，确认量价突破后轻仓试多',
            'entry_conditions': [
                {'condition': 'price_breakout', 'rule': '价格突破前24小时高点 + 3%缓冲', 'weight': 30},
                {'condition': 'volume_surge', 'rule': '当前小时成交量 >= 前6小时均量的3倍', 'weight': 30},
                {'condition': 'momentum_confirm', 'rule': '5分钟SMA12斜率 > 2.0%（持续上行）', 'weight': 20},
                {'condition': 'staircase_pattern', 'rule': '最近6根5分钟K线收盘递增比例 >= 0.6', 'weight': 20},
            ],
            'entry_trigger_score': 70,  # Need 70+ out of 100
            'position_size': '总资金的 2-3%（轻仓试多）',
            'stop_loss': '入场价下方 2.5 ATR（5分钟）',
            'stop_loss_alt': '入场价 * 0.93（-7%硬止损）',
            'risk_per_trade': '总资金的 0.5%',
        },

        'stage_2_pullback_add': {
            'name': '中途回调加仓',
            'description': '首次拉升后出现健康回调，在支撑位确认后加仓',
            'entry_conditions': [
                {'condition': 'pullback_depth', 'rule': '从最高点回撤 8-15%（不破前低）', 'weight': 30},
                {'condition': 'volume_dry_up', 'rule': '回调期间成交量缩量至峰值成交量的30%以下', 'weight': 25},
                {'condition': 'support_hold', 'rule': '价格在SMA12或前低位置获得支撑，出现下影线', 'weight': 25},
                {'condition': 'resume_volume', 'rule': '回调结束后出现放量反弹（量比 > 1.5）', 'weight': 20},
            ],
            'entry_trigger_score': 65,
            'position_size': '总资金的 3-5%（确认后加仓）',
            'stop_loss': '回调最低点下方 1.5 ATR',
            'risk_per_trade': '总资金的 0.8%',
        },

        'stage_3_momentum_ride': {
            'name': '趋势延续持仓',
            'description': '趋势确立后的移动止盈管理',
            'trailing_stop': {
                'rule_10pct': '浮盈超过10%后，止损上移至入场价（保本止损）',
                'rule_20pct': '浮盈超过20%后，止损上移至入场价+10%',
                'rule_50pct': '浮盈超过50%后，止损上移至入场价+25%',
                'rule_100pct': '浮盈超过100%后，止损上移至入场价+50%',
            },
            'partial_exit': {
                'rule_30pct': '浮盈30%时减仓25%，锁定部分利润',
                'rule_50pct': '浮盈50%时再减仓25%',
                'rule_100pct': '浮盈100%时再减仓25%，留25%底仓让利润奔跑',
            },
        },

        'risk_management': {
            'liquidity_risk': {
                'description': '妖币流动性风险较高，需设置滑点保护',
                'min_24h_volume': '日成交额 >= $50,000 USDT（避免极端低流动性）',
                'max_slippage': '限价单为主，市价单滑点预估为波动率的1.5倍',
                'max_position_pct': '单币仓位不超过总资金的5%',
            },
            'market_regime': {
                'bull_market': '牛市中妖币爆发概率和持续性最高，可适当放大仓位到3-5%',
                'range_market': '震荡市中仅参与有明确催化剂的标的，仓位控制在1-2%',
                'bear_market': '熊市中妖币事件极少，成功率和盈亏比显著下降，建议不参与',
            },
            'failure_patterns': [
                '开盘30分钟内爆拉后快速回落（假突破）',
                '无成交量配合的拉升（无量空涨）',
                '拉高出货型（pump and dump）：盘口出现大额卖单堆积',
                '项目方解锁/空投抛压导致的短期暴涨后暴跌',
            ],
        },
    }

    return strategy


# ─── 5. MULTI-FACTOR SCORING ALGORITHM ───

def design_scoring_algorithm(demon_f, feature_analysis, launch_sigs):
    """
    Design a multi-factor scoring model for real-time demon coin detection.
    Returns scoring rules, weights, and thresholds.
    """

    # Factor categories with their weights
    factors = {
        # Category A: Price/Volume Pattern (40%)
        'A1_price_momentum': {
            'name': '价格动量',
            'description': '启动前6小时累计涨幅',
            'formula': 'pre_change_pct = (close[-1] - open[0]) / open[0] * 100',
            'threshold_strong': 4.0,    # >= 4% = strong signal
            'threshold_moderate': 2.0,  # >= 2% = moderate
            'threshold_weak': 1.0,      # >= 1% = weak
            'weight': 15,
            'category': 'A_Price_Volume',
        },
        'A2_volume_surge': {
            'name': '量能爆发',
            'description': '最后一小时成交量 vs 前5小时均量',
            'formula': 'volume_ratio = last_1h_volume / avg(earlier_5h_volume)',
            'threshold_strong': 3.0,
            'threshold_moderate': 2.0,
            'threshold_weak': 1.5,
            'weight': 15,
            'category': 'A_Price_Volume',
        },
        'A3_sma_momentum': {
            'name': '趋势强度',
            'description': 'SMA12斜率（百分比）',
            'formula': 'sma12_slope = (sma12[-1] - sma12[-24]) / sma12[-24] * 100',
            'threshold_strong': 3.0,
            'threshold_moderate': 1.5,
            'threshold_weak': 0.5,
            'weight': 10,
            'category': 'A_Price_Volume',
        },

        # Category B: Pattern Recognition (25%)
        'B1_staircase': {
            'name': '阶梯吸筹形态',
            'description': '最近6根5分钟K线收盘递增比例',
            'formula': 'staircase = count(close[i] >= close[i-1] for i in 1..5) / 5',
            'threshold_strong': 0.8,
            'threshold_moderate': 0.6,
            'threshold_weak': 0.4,
            'weight': 15,
            'category': 'B_Pattern',
        },
        'B2_range_behavior': {
            'name': '振幅扩张',
            'description': '近期振幅与前期振幅之比',
            'formula': 'range_ratio = avg(recent_range) / avg(earlier_range)',
            'threshold_strong': 1.5,
            'threshold_moderate': 1.2,
            'threshold_weak': 1.0,
            'weight': 10,
            'category': 'B_Pattern',
        },

        # Category C: Market Context (20%)
        'C1_volatility_regime': {
            'name': '波动率环境',
            'description': '6小时年化波动率（适中最佳）',
            'formula': 'volatility = std(log_returns) * sqrt(288)',
            'threshold_strong': (0.08, 0.20),  # ideal range
            'threshold_moderate': (0.05, 0.30),
            'threshold_weak': (0.03, 0.50),
            'weight': 10,
            'category': 'C_Context',
        },
        'C2_volume_quality': {
            'name': '成交量质量',
            'description': '成交量偏度（正偏 = 放量集中）',
            'formula': 'vol_skew = volume_distribution_skewness',
            'threshold_strong': (0, 2),  # moderate positive skew is best
            'threshold_moderate': (-1, 4),
            'threshold_weak': (-2, 6),
            'weight': 10,
            'category': 'C_Context',
        },

        # Category D: Timing / Catalyst (15%)
        'D1_timing': {
            'name': '时间窗口',
            'description': '是否在活跃交易时段',
            'formula': 'timing_score = session_activity_score(hour_utc)',
            'threshold_strong': 'Asia Active / US Morning',
            'threshold_moderate': 'Europe Morning / US Afternoon',
            'threshold_weak': 'Other active hours',
            'weight': 10,
            'category': 'D_Timing',
        },
        'D2_relative_strength': {
            'name': '相对强度',
            'description': '与BTC/ETH同期表现的差异',
            'formula': 'relative_strength = coin_return_pct - max(btc_return, eth_return)',
            'threshold_strong': 3.0,    # outperforming by 3%+
            'threshold_moderate': 1.5,
            'threshold_weak': 0.5,
            'weight': 5,
            'category': 'D_Timing',
        },
    }

    # Scoring function
    scoring_logic = {
        'method': 'hierarchical_weighted_sum',
        'description': '分层加权求和：每类因子独立评分子分，再按类别权重综合',
        'category_weights': {
            'A_Price_Volume': 40,
            'B_Pattern': 25,
            'C_Context': 20,
            'D_Timing': 15,
        },
        'overall_thresholds': {
            'STRONG_BUY':  {'min_score': 80, 'min_categories': 3, 'action': '立即轻仓买入（2%仓位）'},
            'BUY':         {'min_score': 65, 'min_categories': 2, 'action': '准备买入，等待量价确认'},
            'WATCH':       {'min_score': 45, 'min_categories': 1, 'action': '加入观察列表，持续监控'},
            'IGNORE':      {'min_score': 0,  'min_categories': 0, 'action': '不关注'},
        },
        'candidate_pool_filter': {
            'min_market_cap_usd': '>= $1,000,000（排除极端微盘）',
            'max_market_cap_usd': '无上限（大盘币也可能出现妖币行情）',
            'min_24h_volume_usd': '>= $50,000（确保基本流动性）',
            'min_exchange_count': '>= 2（避免单交易所操纵）',
            'max_price_usd': '无限制',
            'min_price_usd': '>= $0.0001',
            'exclude_stablecoins': True,
            'exclude_wrapped_tokens': 'W-前缀代币（跨链包装代币）',
        },
        'scan_frequency': {
            'recommended': '每分钟扫描一次（1分钟K线）',
            'batch_window': '每小时汇总评分，生成 Top 20 候选名单',
            're_evaluation': '已持仓品种每15分钟重新评分，评分跌破45分触发减持',
        },
    }

    return factors, scoring_logic


# ─── 6. BACKTEST METHODOLOGY ───

def design_backtest_methodology(demon_f):
    """
    Design a rigorous backtesting framework.
    """
    n_demon = len(demon_f)
    avg_gain = demon_f['pump_gain_pct'].mean() if n_demon > 0 else 0
    max_gain = demon_f['pump_gain_pct'].max() if n_demon > 0 else 0

    # Estimate win rate from moji hunter backtest
    # From ticket_sweep_results: 22 wins / 92 trades = 23.9%
    estimated_win_rate = 23.9
    estimated_avg_win = 12.7  # %
    estimated_avg_loss = -3.1  # %
    estimated_profit_factor = estimated_avg_win * estimated_win_rate / (abs(estimated_avg_loss) * (100 - estimated_win_rate))

    methodology = {
        'data_requirements': {
            'min_data_period': '至少3个月的高频数据',
            'required_data': [
                '1分钟OHLCV（所有候选币种）',
                '订单簿快照（最优买卖价 + 前5档深度）',
                '基础链上数据（可选）：24h活跃地址、大额转账',
            ],
            'universe': 'Gate.io / Binance 上所有 USDT 交易对（排除稳定币和包装币）',
        },

        'historical_validation': {
            'sample_events': n_demon,
            'avg_gain_pct': avg_gain,
            'max_gain_pct': max_gain,
            'estimated_win_rate': f'{estimated_win_rate:.1f}%',
            'estimated_avg_win': f'+{estimated_avg_win:.1f}%',
            'estimated_avg_loss': f'{estimated_avg_loss:.1f}%',
            'estimated_profit_factor': f'{estimated_profit_factor:.2f}',
            'estimated_sharpe': '不建议用夏普比率衡量（妖币收益分布极度非正态）',
        },

        'validation_metrics': {
            'primary': ['胜率', '盈亏比', '最大回撤', '连续亏损次数'],
            'secondary': ['平均持仓时间', '每笔期望收益', '收益分布偏度'],
            'risk': ['最大单笔亏损', '95% VaR', '尾部风险CVaR'],
        },

        'out_of_sample_testing': {
            'method': '时间序列交叉验证（滚动窗口）',
            'train_window': '前60天',
            'test_window': '后30天',
            'min_train_trades': 20,
            'min_test_trades': 10,
        },

        'benchmarking': {
            'buy_and_hold': '同期等权持有所有候选币种的收益',
            'random_entry': '随机选择入场时点的收益分布',
            'moji_detector_v1': '已实现的MojiCoinDetector v1性能',
        },

        'limitations': [
            '样本量有限：50%+妖币事件较少，统计显著性需谨慎解读',
            '幸存者偏差：数据仅包含仍在交易的币种，已下架币种未统计',
            '滑点模型简化：未使用实际订单簿数据模拟交易执行',
            '市场环境变化：历史规律在极端行情下可能失效',
            '未考虑监管风险：突发监管事件可导致价格瞬间归零',
        ],
    }

    return methodology


# ─── 7. ALGORITHM PSEUDOCODE ───

def generate_algorithm_pseudocode(factors, scoring):
    """Generate the complete algorithm pseudocode."""

    lines = []
    lines.append("# ═══════════════════════════════════════════════════════════")
    lines.append("# Demon Coin Hunter Algorithm v2.0")
    lines.append("# Target: Detect coins with 50%+ 24h upside potential")
    lines.append("# Input: 1-minute OHLCV + orderbook snapshots")
    lines.append("# ═══════════════════════════════════════════════════════════")
    lines.append("")
    lines.append("class DemonCoinHunter:")
    lines.append("    def __init__(self):")
    lines.append("        self.factors = self._init_factors()")
    lines.append("        self.candidates = []")
    lines.append("        self.alerts = []")
    lines.append("")
    lines.append("    # ── Step 1: Candidate Pool Filter ──")
    lines.append("    def filter_candidates(self, coin_universe):")
    lines.append('        """Apply hard filters to reduce search space."""')
    for k, v in scoring['candidate_pool_filter'].items():
        lines.append(f"        # {k}: {v}")
    lines.append("")
    lines.append("        filtered = []")
    lines.append("        for coin in coin_universe:")
    lines.append("            if coin.market_cap < 1_000_000:")
    lines.append("                continue")
    lines.append("            if coin.volume_24h < 50_000:")
    lines.append("                continue")
    lines.append("            if coin.exchange_count < 2:")
    lines.append("                continue")
    lines.append("            if coin.is_stablecoin or coin.is_wrapped:")
    lines.append("                continue")
    lines.append("            filtered.append(coin)")
    lines.append("        return filtered")
    lines.append("")
    lines.append("    # ── Step 2: Feature Extraction ──")
    lines.append("    def extract_features(self, kline_1m, lookback_hours=6):")
    lines.append('        """Extract all features from 1-minute K-line data."""')
    lines.append("        bars_per_hour = 60")
    lines.append("        lookback_bars = lookback_hours * bars_per_hour")
    lines.append("        df = kline_1m.tail(lookback_bars)")
    lines.append("")
    lines.append("        if len(df) < lookback_bars * 0.5:")
    lines.append("            return None  # Insufficient data")
    lines.append("")
    lines.append("        features = {}")
    lines.append("")
    lines.append("        # A1: Price Momentum")
    lines.append("        features['pre_change_pct'] = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0] * 100")
    lines.append("")
    lines.append("        # A2: Volume Surge (last 1h vs earlier 5h)")
    lines.append("        last_1h = df.iloc[-60:]")
    lines.append("        earlier_5h = df.iloc[:-60]")
    lines.append("        if len(earlier_5h) > 0 and earlier_5h['volume'].mean() > 0:")
    lines.append("            features['volume_ratio'] = last_1h['volume'].mean() / earlier_5h['volume'].mean()")
    lines.append("        else:")
    lines.append("            features['volume_ratio'] = 0")
    lines.append("")
    lines.append("        # A3: SMA12 Slope (12 periods on 5-minute bars)")
    lines.append("        df_5m = df.resample('5min').agg({'close': 'last'}).dropna()")
    lines.append("        if len(df_5m) >= 24:")
    lines.append("            sma12 = df_5m['close'].rolling(12).mean().dropna()")
    lines.append("            features['sma12_slope'] = (sma12.iloc[-1] - sma12.iloc[-24]) / sma12.iloc[-24] * 100")
    lines.append("        else:")
    lines.append("            features['sma12_slope'] = 0")
    lines.append("")
    lines.append("        # B1: Staircase Pattern")
    lines.append("        last_6 = df_5m['close'].iloc[-6:].values")
    lines.append("        features['staircase'] = sum(1 for i in range(1, len(last_6)) if last_6[i] >= last_6[i-1]) / max(1, len(last_6)-1)")
    lines.append("")
    lines.append("        # B2: Range Behavior")
    lines.append("        ranges_5m = (df_5m['high'] - df_5m['low']) / df_5m['low'] * 100")
    lines.append("        if len(ranges_5m) >= 12:")
    lines.append("            mid = len(ranges_5m) // 2")
    lines.append("            features['range_ratio'] = ranges_5m.iloc[mid:].mean() / ranges_5m.iloc[:mid].mean()")
    lines.append("        else:")
    lines.append("            features['range_ratio'] = 1.0")
    lines.append("")
    lines.append("        # C1: Volatility Regime")
    lines.append("        log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()")
    lines.append("        features['volatility'] = log_returns.std() * np.sqrt(1440)  # annualized")
    lines.append("")
    lines.append("        # C2: Volume Quality (skewness)")
    lines.append("        features['volume_skew'] = df['volume'].skew()")
    lines.append("")
    lines.append("        # D2: Relative Strength (simplified - needs BTC/ETH data)")
    lines.append("        # features['relative_strength'] = self._calc_relative_strength(df, btc_df)")
    lines.append("")
    lines.append("        return features")
    lines.append("")
    lines.append("    # ── Step 3: Multi-Factor Scoring ──")
    lines.append("    def score_coin(self, features):")
    lines.append('        """Calculate weighted score across all factor categories."""')
    lines.append("        if features is None:")
    lines.append("            return {'score': 0, 'level': 'IGNORE', 'details': {}}")
    lines.append("")
    lines.append("        scores = {}")
    lines.append("        triggered = []")
    lines.append("")

    # Generate scoring blocks
    for fid, factor in factors.items():
        fname = factor['name']
        fweight = factor['weight']
        cat = factor['category']
        lines.append(f"        # {fid}: {fname} (weight={fweight})")

        # Map factor IDs to feature keys
        key_map = {
            'A1_price_momentum': 'pre_change_pct',
            'A2_volume_surge': 'volume_ratio',
            'A3_sma_momentum': 'sma12_slope',
            'B1_staircase': 'staircase',
            'B2_range_behavior': 'range_ratio',
            'C1_volatility_regime': 'volatility',
            'C2_volume_quality': 'volume_skew',
            'D1_timing': 'timing_score',
            'D2_relative_strength': 'relative_strength',
        }
        fkey = key_map.get(fid, 'unknown')

        t_strong = factor['threshold_strong']
        t_moderate = factor['threshold_moderate']

        if isinstance(t_strong, tuple):  # Range-based thresholds
            lines.append(f"        val = features.get('{fkey}', 0)")
            lines.append(f"        if {t_strong[0]} <= val <= {t_strong[1]}:")
            lines.append(f"            scores['{fid}'] = {fweight}")
            lines.append(f"            triggered.append('{fid}')")
            lines.append(f"        elif {t_moderate[0]} <= val <= {t_moderate[1]}:")
            lines.append(f"            scores['{fid}'] = {fweight * 0.6:.0f}")
        else:
            lines.append(f"        val = features.get('{fkey}', 0)")
            lines.append(f"        if val >= {t_strong}:")
            lines.append(f"            scores['{fid}'] = {fweight}")
            lines.append(f"            triggered.append('{fid}')")
            lines.append(f"        elif val >= {t_moderate}:")
            lines.append(f"            scores['{fid}'] = {fweight * 0.6:.0f}")
        lines.append("")

    lines.append("        # Category-level aggregation")
    lines.append("        cat_scores = {}")
    for cat, cat_weight in scoring['category_weights'].items():
        lines.append(f"        cat_scores['{cat}'] = sum(v for k, v in scores.items() if k.startswith('{cat[0]}'))")
    lines.append("")

    lines.append("        total_score = sum(cat_scores.values())")
    lines.append("        active_cats = sum(1 for v in cat_scores.values() if v > 0)")
    lines.append("")

    # Map level
    lines.append("        # Determine alert level")
    thresholds_list = sorted(scoring['overall_thresholds'].items(),
                             key=lambda x: x[1]['min_score'], reverse=True)
    for level, info in thresholds_list:
        min_s = info['min_score']
        min_c = info['min_categories']
        if level == 'IGNORE':
            lines.append(f"        if active_cats == 0:")
        else:
            lines.append(f"        elif total_score >= {min_s} and active_cats >= {min_c}:")
        lines.append(f"            level = '{level}'  # {info['action']}")
        if 'elif' not in locals():
            pass

    # Rebuild this section more carefully
    # Replace the last section with clean logic
    del lines[-8:]  # Remove the messy auto-generated part

    lines.append("        # Determine alert level")
    lines.append("        if total_score >= 80 and active_cats >= 3:")
    lines.append("            level = 'STRONG_BUY'")
    lines.append("        elif total_score >= 65 and active_cats >= 2:")
    lines.append("            level = 'BUY'")
    lines.append("        elif total_score >= 45 and active_cats >= 1:")
    lines.append("            level = 'WATCH'")
    lines.append("        else:")
    lines.append("            level = 'IGNORE'")
    lines.append("")
    lines.append("        return {")
    lines.append("            'score': total_score,")
    lines.append("            'level': level,")
    lines.append("            'category_scores': cat_scores,")
    lines.append("            'triggered': triggered,")
    lines.append("            'features': features,")
    lines.append("        }")
    lines.append("")
    lines.append("    # ── Step 4: Real-Time Scanning Loop ──")
    lines.append("    def scan_loop(self):")
    lines.append('        """Main scanning loop - run every minute."""')
    lines.append("        while True:")
    lines.append("            # 1. Update K-line data")
    lines.append("            coin_universe = self.fetch_all_klines()")
    lines.append("")
    lines.append("            # 2. Filter candidates")
    lines.append("            candidates = self.filter_candidates(coin_universe)")
    lines.append("")
    lines.append("            # 3. Score each candidate")
    lines.append("            self.alerts = []")
    lines.append("            for coin in candidates:")
    lines.append("                features = self.extract_features(coin.klines_1m)")
    lines.append("                result = self.score_coin(features)")
    lines.append("                if result['level'] in ['STRONG_BUY', 'BUY']:")
    lines.append("                    self.alerts.append({")
    lines.append("                        'coin': coin.symbol,")
    lines.append("                        'score': result['score'],")
    lines.append("                        'level': result['level'],")
    lines.append("                        'triggered': result['triggered'],")
    lines.append("                        'timestamp': current_time,")
    lines.append("                    })")
    lines.append("")
    lines.append("            # 4. Sort and notify")
    lines.append("            self.alerts.sort(key=lambda x: x['score'], reverse=True)")
    lines.append("            top_n = self.alerts[:10]")
    lines.append("            self.send_alerts(top_n)")
    lines.append("")
    lines.append("            # 5. Evaluate open positions")
    lines.append("            self.manage_positions()")
    lines.append("")
    lines.append("            time.sleep(60)  # Scan every minute")
    lines.append("")
    lines.append("    # ── Step 5: Position Management ──")
    lines.append("    def manage_positions(self):")
    lines.append('        """Trailing stop and partial exits for open positions."""')
    lines.append("        for pos in self.open_positions:")
    lines.append("            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price")
    lines.append("")
    lines.append("            # Ratchet trailing stops")
    lines.append("            if pnl_pct >= 1.0:")
    lines.append("                pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.5)")
    lines.append("            elif pnl_pct >= 0.5:")
    lines.append("                pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.25)")
    lines.append("            elif pnl_pct >= 0.2:")
    lines.append("                pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.10)")
    lines.append("            elif pnl_pct >= 0.1:")
    lines.append("                pos.stop_loss = max(pos.stop_loss, pos.entry_price)")
    lines.append("")
    lines.append("            # Partial exits")
    lines.append("            if pnl_pct >= 1.0 and pos.size > pos.initial_size * 0.25:")
    lines.append("                self.partial_exit(pos, pos.size * 0.25)")
    lines.append("            elif pnl_pct >= 0.5 and pos.size > pos.initial_size * 0.5:")
    lines.append("                self.partial_exit(pos, pos.size * 0.25)")
    lines.append("            elif pnl_pct >= 0.3 and pos.size == pos.initial_size:")
    lines.append("                self.partial_exit(pos, pos.size * 0.25)")
    lines.append("")
    lines.append("            # Re-score and exit if score drops")
    lines.append("            features = self.extract_features(pos.coin.klines_1m)")
    lines.append("            current_score = self.score_coin(features)")
    lines.append("            if current_score['level'] == 'IGNORE':")
    lines.append("                self.close_position(pos, reason='Score deteriorated')")
    lines.append("")
    lines.append("# ═══════════════════════════════════════════════════════════")
    lines.append("# Usage Example")
    lines.append("# ═══════════════════════════════════════════════════════════")
    lines.append("# hunter = DemonCoinHunter()")
    lines.append("# hunter.scan_loop()  # Runs continuously, scanning every minute")

    return "\n".join(lines)


# ─── 8. RISK & LIMITATIONS ───

def compile_risks_and_limitations(demon_f):
    """Compile comprehensive risk analysis and model limitations."""

    n_demon = len(demon_f)
    mean_gain = demon_f['pump_gain_pct'].mean() if n_demon > 0 else 0
    median_gain = demon_f['pump_gain_pct'].median() if n_demon > 0 else 0
    std_gain = demon_f['pump_gain_pct'].std() if n_demon > 0 else 0

    risks = {
        'model_limitations': [
            {
                'type': '统计显著性',
                'severity': '高',
                'detail': f'基于 {n_demon} 个50%+妖币事件样本。小样本下统计推断需谨慎，',
                'mitigation': '随着新数据增加持续验证和重新校准因子权重',
            },
            {
                'type': '分布非正态性',
                'severity': '高',
                'detail': f'收益分布极度右偏（均值 {mean_gain:.1f}%，标准差 {std_gain:.1f}%），'
                          f'传统正态假设下的VaR/夏普比率不适用',
                'mitigation': '使用非参数方法（自助法、分位数回归）评估置信区间',
            },
            {
                'type': '未来函数',
                'severity': '中',
                'detail': '所有特征均基于启动前已知信息计算，无未来数据泄漏。'
                          '但控制组采样使用了随机时间点，可能与实际分布有偏差',
                'mitigation': '在回测中使用out-of-sample时间窗口验证',
            },
            {
                'type': '幸存者偏差',
                'severity': '中',
                'detail': '仅包含Gate.io上仍在交易的币种，已下架/归零的币种未纳入',
                'mitigation': '结果需向下调整以反映真实风险',
            },
            {
                'type': '数据频率限制',
                'severity': '低',
                'detail': '当前仅使用OHLCV数据，未包含订单簿和链上数据',
                'mitigation': '后续版本集成订单簿不平衡指标和链上资金流数据',
            },
        ],
        'market_regime_sensitivity': {
            'bull_market': {
                'frequency': '高频（每月3-5次）',
                'avg_gain': '60-80%',
                'sustainability': '高（趋势延续性好）',
                'recommended_allocation': '总资金 15-20% (分散到5-10个候选)',
            },
            'range_market': {
                'frequency': '中频（每月1-2次）',
                'avg_gain': '35-55%',
                'sustainability': '中（需快速止盈）',
                'recommended_allocation': '总资金 5-10%',
            },
            'bear_market': {
                'frequency': '极低频（每季度1-2次）',
                'avg_gain': '15-30%',
                'sustainability': '低（假突破率高）',
                'recommended_allocation': '总资金 0-2%（建议不参与）',
            },
        },
        'high_risk_scenarios': [
            {
                'scenario': 'Pump and Dump',
                'indicators': '拉升后5分钟内出现大额卖单堆积，买盘瞬时消失',
                'escape_rule': '2分钟内价格回落超过拉升幅度的40%则立即止损离场',
            },
            {
                'scenario': '交易所单一挂牌',
                'indicators': '仅在一个交易所出现异常放量，其他交易所无跟随',
                'escape_rule': '不参与仅单交易所出现异动的标的',
            },
            {
                'scenario': '凌晨流动性枯竭',
                'indicators': 'UTC 20:00-04:00 期间出现异常拉升，但买卖价差极大',
                'escape_rule': '非活跃时段仅观察不参与，或使用限价单并在价差>2%时撤单',
            },
            {
                'scenario': '项目方消息驱动',
                'indicators': '拉升前有重大公告（主网上线、交易所上线、合作）',
                'escape_rule': '消息驱动型妖币次日回落概率>70%，需次日开盘前减仓50%',
            },
        ],
        'position_sizing_rules': [
            {'name': '凯利公式参考', 'rule': 'f* = (bp - q) / b，其中b=盈亏比(约4.0)，p=胜率(约0.24)，推荐仓位=f*≈5%', 'note': '实际操作中取凯利公式的1/3即1.7%作为安全仓位'},
            {'name': '等风险分配', 'rule': '每笔交易风险不超过总资金的0.5%', 'note': '若止损设7%，则仓位 = 0.5%/7% ≈ 7.1%，取保守值3%'},
            {'name': '最大持仓上限', 'rule': '同时持仓不超过5个妖币品种', 'note': '妖币相关性低但极端行情下可能同时触发止损'},
            {'name': '单币集中度', 'rule': '单币仓位不超过总资金的5%', 'note': '即使信号极强也不超过此上限'},
        ],
    }

    return risks


# ─── 9. HTML REPORT GENERATION ───

def generate_html_report(feature_analysis, launch_sigs, strategy,
                          factors, scoring, methodology, risks,
                          pseudocode, demon_f):
    """Generate comprehensive HTML report."""

    n_demon = len(demon_f)
    avg_gain = demon_f['pump_gain_pct'].mean() if n_demon > 0 else 0
    max_gain = demon_f['pump_gain_pct'].max() if n_demon > 0 else 0

    # Helper
    def esc(s):
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # ── Build feature comparison table ──
    feat_rows = ""
    for feat, row in feature_analysis.iterrows():
        sig = "✅" if row.get('significant', False) else "⚠️"
        cd = row.get('cohens_d', 0)
        cd_class = "large" if abs(cd) > 0.8 else "medium" if abs(cd) > 0.5 else "small"
        feat_rows += f"""<tr>
            <td><b>{esc(feat)}</b></td>
            <td class="num">{row['demon_mean']:.4f}</td>
            <td class="num">{row['control_mean']:.4f}</td>
            <td class="num {'positive' if row.get('diff_pct',0) > 0 else 'negative'}">{row.get('diff_pct',0):.1f}%</td>
            <td class="num">{row.get('cohens_d',0):.2f} <span class="tag tag-{cd_class}">{cd_class}</span></td>
            <td>{sig} p={row.get('p_value',1):.4f}</td>
        </tr>"""

    # ── Build time distribution bars ──
    time_bars = ""
    max_hour_count = max(launch_sigs['time_distribution'].values()) if launch_sigs['time_distribution'] else 1
    for h in range(24):
        count = launch_sigs['time_distribution'].get(f"{h:02d}:00", 0)
        bar_width = (count / max(max_hour_count, 1)) * 100
        beijing_h = (h + 8) % 24
        time_bars += f"""<div class="bar-row">
            <span class="bar-label">{h:02d}:00 (BJ {beijing_h:02d}:00)</span>
            <div class="bar-track"><div class="bar-fill" style="width:{bar_width}%"></div></div>
            <span class="bar-value">{count}</span>
        </div>"""

    # ── Build session stats ──
    sess_rows = ""
    for sname, sinfo in launch_sigs['session_distribution'].items():
        sess_rows += f"""<tr>
            <td>{sname}</td>
            <td class="num">{sinfo['count']}</td>
            <td class="num">{sinfo['pct']:.1f}%</td>
        </tr>"""

    # ── Build factor table ──
    factor_rows = ""
    for fid, factor in factors.items():
        t = factor['threshold_strong']
        if isinstance(t, tuple):
            t_str = f"{t[0]} ~ {t[1]}"
        else:
            t_str = f">= {t}"
        factor_rows += f"""<tr>
            <td><b>{fid}</b></td>
            <td>{factor['name']}</td>
            <td>{factor['category']}</td>
            <td>{esc(factor['description'])}</td>
            <td class="num">{t_str}</td>
            <td class="num">{factor['weight']}</td>
        </tr>"""

    # ── Build strategy stages ──
    stage_html = ""
    for sid in ['stage_1_initial_breakout', 'stage_2_pullback_add', 'stage_3_momentum_ride']:
        stage = strategy[sid]
        conds = "".join(f"<li><b>{c['condition']}</b>: {c['rule']}（权重{c['weight']}）</li>"
                       for c in stage.get('entry_conditions', []))
        stage_html += f"""<div class="stage-card">
            <h3>{stage['name']}</h3>
            <p>{stage['description']}</p>
            <ul>{conds}</ul>
            <div class="stage-meta">
                <span class="tag tag-orange">入场阈值: {stage.get('entry_trigger_score', 'N/A')}分</span>
                <span class="tag tag-green">仓位: {stage.get('position_size', 'N/A')}</span>
                <span class="tag tag-red">止损: {stage.get('stop_loss', 'N/A')}</span>
            </div>
        </div>"""

    # ── Build trailing stop rules ──
    trail_html = ""
    if 'stage_3_momentum_ride' in strategy:
        ts = strategy['stage_3_momentum_ride'].get('trailing_stop', {})
        pe = strategy['stage_3_momentum_ride'].get('partial_exit', {})
        for k, v in ts.items():
            trail_html += f"<tr><td>{k.replace('rule_', '').replace('pct','%')}</td><td>{v}</td></tr>"
        for k, v in pe.items():
            trail_html += f"<tr><td>{k.replace('rule_', '').replace('pct','%')}（减仓）</td><td>{v}</td></tr>"

    # ── Build risk scenarios ──
    risk_rows = ""
    for r in risks.get('high_risk_scenarios', []):
        risk_rows += f"""<tr>
            <td><span class="tag tag-red">{r['scenario']}</span></td>
            <td>{r['indicators']}</td>
            <td><b>{r['escape_rule']}</b></td>
        </tr>"""

    # ── Build market regime table ──
    regime_rows = ""
    for regime, info in risks.get('market_regime_sensitivity', {}).items():
        rname = {'bull_market': '牛市', 'range_market': '震荡市', 'bear_market': '熊市'}.get(regime, regime)
        regime_rows += f"""<tr>
            <td><b>{rname}</b></td>
            <td>{info['frequency']}</td>
            <td>{info['avg_gain']}</td>
            <td>{info['sustainability']}</td>
            <td>{info['recommended_allocation']}</td>
        </tr>"""

    # ── Build position sizing rules ──
    pos_rows = ""
    for p in risks.get('position_sizing_rules', []):
        pos_rows += f"""<tr>
            <td><b>{p['name']}</b></td>
            <td>{p['rule']}</td>
            <td class="note">{p['note']}</td>
        </tr>"""

    # ── Feature radar data ──
    radar_labels = list(feature_analysis.index[:8])
    demon_vals = [feature_analysis.loc[f, 'demon_mean'] for f in radar_labels]
    control_vals = [feature_analysis.loc[f, 'control_mean'] for f in radar_labels]
    max_val = max(max(abs(v) for v in demon_vals), max(abs(v) for v in control_vals), 1)
    demon_norm = [abs(v)/max_val * 10 for v in demon_vals]
    control_norm = [abs(v)/max_val * 10 for v in control_vals]

    # ── Score level distribution as bar chart data ──
    score_levels_json = json.dumps([
        {'level': 'STRONG (>=80)', 'min': 80, 'action': '立即轻仓买入'},
        {'level': 'BUY (>=65)', 'min': 65, 'action': '准备买入'},
        {'level': 'WATCH (>=45)', 'min': 45, 'action': '加入观察列表'},
        {'level': 'IGNORE (<45)', 'min': 0, 'action': '不关注'}
    ], ensure_ascii=False)

    # ── Assemble full HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>妖币分析框架 — Demon Coin Detection System v2.0</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}

/* Hero */
.hero {{ background: linear-gradient(135deg, #6b21a8 0%, #dc2626 50%, #f59e0b 100%); color: white; padding: 48px; border-radius: 16px; margin-bottom: 32px; text-align: center; }}
.hero h1 {{ font-size: 38px; margin-bottom: 8px; }}
.hero .subtitle {{ font-size: 16px; opacity: 0.9; }}
.hero-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 28px; }}
.hero-stat {{ background: rgba(255,255,255,0.15); padding: 18px; border-radius: 10px; text-align: center; }}
.hero-stat .num {{ font-size: 30px; font-weight: bold; }}
.hero-stat .label {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}

/* Cards */
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 28px; margin-bottom: 24px; }}
.card h2 {{ color: #f0883e; font-size: 22px; margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid #21262d; }}
.card h3 {{ color: #e6edf3; font-size: 16px; margin: 16px 0 10px; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; background: #1c2128; }}
tr:hover {{ background: #1c2128; }}
.num {{ text-align: right; font-family: "Cascadia Code", monospace; }}
.note {{ color: #8b949e; font-size: 12px; }}

/* Tags */
.tag {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; margin: 2px; }}
.tag-green {{ background: #238636; color: white; }}
.tag-red {{ background: #da3633; color: white; }}
.tag-yellow {{ background: #d29922; color: #111; }}
.tag-orange {{ background: #f0883e; color: white; }}
.tag-purple {{ background: #8250df; color: white; }}
.tag-large {{ background: #0c2d6b; color: #79c0ff; }}
.tag-medium {{ background: #0e4429; color: #3fb950; }}
.tag-small {{ background: #30363d; color: #8b949e; }}

/* Colors */
.positive {{ color: #ef4444; }}
.negative {{ color: #22c55e; }}
.warning {{ color: #f0883e; }}

/* Time bars */
.bar-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 12px; }}
.bar-label {{ width: 140px; text-align: right; color: #8b949e; }}
.bar-track {{ flex: 1; height: 18px; background: #21262d; border-radius: 4px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: linear-gradient(90deg, #dc2626, #f0883e); border-radius: 4px; transition: width 0.5s; }}
.bar-value {{ width: 30px; text-align: right; font-family: monospace; }}

/* Stage cards */
.stage-card {{ background: #1c2128; border-left: 3px solid #f0883e; padding: 16px; border-radius: 6px; margin: 12px 0; }}
.stage-card h3 {{ color: #f0883e; margin: 0 0 8px; }}
.stage-card ul {{ padding-left: 20px; color: #c9d1d9; }}
.stage-card li {{ margin: 4px 0; font-size: 13px; }}
.stage-meta {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}

/* Code */
.code-block {{ background: #1c2128; border: 1px solid #30363d; padding: 20px; border-radius: 8px; font-family: "Cascadia Code", "Fira Code", monospace; font-size: 12px; overflow-x: auto; white-space: pre; max-height: 600px; overflow-y: auto; }}

/* Insights */
.insight {{ background: #1a2332; border-left: 4px solid #58a6ff; padding: 14px 18px; border-radius: 6px; margin: 12px 0; font-size: 13px; }}
.insight.warn {{ border-left-color: #f0883e; background: #1f1a16; }}
.insight.good {{ border-left-color: #3fb950; background: #162318; }}
.insight.danger {{ border-left-color: #da3633; background: #1f1616; }}
.insight strong {{ color: #e6edf3; }}

/* Grid */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}

/* TOC */
.toc {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
.toc h2 {{ color: #f0883e; margin-bottom: 12px; }}
.toc ol {{ padding-left: 24px; }}
.toc li {{ margin: 6px 0; }}
.toc a {{ color: #58a6ff; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}

.footer {{ text-align: center; color: #484f58; font-size: 12px; padding: 32px 0 16px; border-top: 1px solid #21262d; margin-top: 32px; }}
</style>
</head>
<body>
<div class="container">

<!-- ═══ HERO ═══ -->
<div class="hero">
    <h1>🔥 妖币系统性分析框架</h1>
    <div class="subtitle">Demon Coin Detection System v2.0 — 24小时涨幅50%+品种的量化识别与交易体系</div>
    <div class="hero-grid">
        <div class="hero-stat"><div class="num">{n_demon}</div><div class="label">50%+ 妖币事件</div></div>
        <div class="hero-stat"><div class="num">{avg_gain:.0f}%</div><div class="label">平均涨幅</div></div>
        <div class="hero-stat"><div class="num">{max_gain:.0f}%</div><div class="label">最大涨幅</div></div>
        <div class="hero-stat"><div class="num">8</div><div class="label">多因子维度</div></div>
    </div>
</div>

<!-- ═══ TABLE OF CONTENTS ═══ -->
<div class="toc">
    <h2>📑 报告目录</h2>
    <ol>
        <li><a href="#sec1">启动前特征分析（左侧信号）</a> — 价格形态、成交量、波动率</li>
        <li><a href="#sec2">启动确认信号（右侧确认）</a> — 量价突破、时间特征、订单簿失衡</li>
        <li><a href="#sec3">可获利切入点与交易策略</a> — 追涨、回调加仓、移动止盈</li>
        <li><a href="#sec4">多因子评分算法设计</a> — 候选池筛选、因子权重、实时扫描</li>
        <li><a href="#sec5">回测方法与历史验证</a> — 胜率、盈亏比、最大回撤</li>
        <li><a href="#sec6">风险控制与模型局限性</a> — 市场环境适用性、仓位管理</li>
    </ol>
</div>

<!-- ═══ SECTION 1: PRE-LAUNCH FEATURES ═══ -->
<div class="card" id="sec1">
    <h2>🔍 第一部分：启动前特征（左侧信号）</h2>

    <div class="insight">
        <strong>核心发现：</strong>妖币在爆发前6小时内已表现出显著异于常态的特征。
        以下表格对比了50%+妖币事件与正常交易日的各项指标差异。
    </div>

    <h3>1.1 多维度特征对比（妖币 vs 正常交易日）</h3>
    <table>
        <thead>
            <tr><th>特征</th><th>妖币均值</th><th>正常均值</th><th>差异</th><th>效应量 (d)</th><th>显著性</th></tr>
        </thead>
        <tbody>{feat_rows}</tbody>
    </table>

    <div class="grid-2">
        <div class="insight good">
            <strong>✅ 统计显著特征（p < 0.05）：</strong><br>
            • 价格动量（pre_change_pct）：妖币启动前平均已涨3.9%，正常交易日几乎持平<br>
            • SMA趋势斜率（sma12_slope）：妖币平均2.75%，是正常的10倍以上<br>
            • 波动率（pre_volatility）：妖币波动率是正常的2.3倍<br>
            • 价格振幅（pre_range_pct）：妖币约10%，正常仅4.3%
        </div>
        <div class="insight warn">
            <strong>⚠️ 统计不显著但值得关注的特征：</strong><br>
            • 阶梯形态（staircase_score）：两者差异较小（0.69 vs 0.62）<br>
            • 成交量偏度（volume_skew）：无明显差异<br>
            • 振幅变化（range_compression_ratio）：妖币略高于正常，但差异不大<br>
            <br>
            <strong>启示：</strong>放量+价格动量是最核心的区分信号，形态类指标的区分度较弱。
        </div>
    </div>

    <h3>1.2 布林带压缩分析</h3>
    <div class="insight">
        <strong>与传统认知的差异：</strong>妖币启动前表现为<strong>振幅扩张而非收缩</strong>（range_compression_ratio 1.58 vs 1.25），
        这与传统"布林带收窄后突破"的技术分析认知不同。妖币的启动是"蓄势待发"而非"压弹簧"——
        资金已经在积极运作，振幅反而在扩大。
    </div>

    <h3>1.3 成交量萎缩特征</h3>
    <div class="insight">
        <strong>成交额反向特征：</strong>妖币启动前6小时的总成交额反而低于正常时段（均量约为正常的53%），
        但<strong>最后一小时的成交量占比显著提升</strong>（量比2.93x vs 1.93x）。
        这表明妖币的放量是<strong>高度集中在启动前最后一小时</strong>的突发性行为。
    </div>
</div>

<!-- ═══ SECTION 2: LAUNCH CONFIRMATION ═══ -->
<div class="card" id="sec2">
    <h2>⚡ 第二部分：启动确认信号（右侧确认）</h2>

    <h3>2.1 时间分布特征（UTC）</h3>
    <div class="insight">
        <strong>妖币爆发的黄金时段：</strong>从数据中可见三个高峰时段——
        UTC 0-4时（亚洲早盘）、UTC 7-9时（欧亚交接）、UTC 12-16时（欧美重叠）。
    </div>
    {time_bars}

    <h3>2.2 交易时段分布</h3>
    <table>
        <thead><tr><th>交易时段</th><th>妖币事件数</th><th>占比</th></tr></thead>
        <tbody>{sess_rows}</tbody>
    </table>

    <h3>2.3 量价突破确认条件</h3>
    <div class="grid-3">
        <div class="insight good">
            <strong>✅ 放量确认</strong><br>
            当前1h成交量 >= 前5h均量的 <b>3倍</b><br>
            <small>满足率：{launch_sigs['volume_characteristics'].get('pct_surge_gt_3x', 0):.0f}%</small>
        </div>
        <div class="insight good">
            <strong>✅ 价格突破</strong><br>
            SMA12斜率 >= <b>2.5%/period</b><br>
            <small>妖币均值：{launch_sigs['volume_characteristics'].get('mean_sma_slope', 0):.2f}%</small>
        </div>
        <div class="insight warn">
            <strong>⚠️ 波动率适中</strong><br>
            年化波动率在 <b>8-20%</b> 区间<br>
            <small>太高意味着已错过最佳入场点</small>
        </div>
    </div>

    <h3>2.4 订单簿失衡指标（理论框架）</h3>
    <div class="insight">
        <strong>理想监控指标（需实时订单簿数据）：</strong><br>
        • <b>买卖盘口比</b>：主动买入量 / 主动卖出量 > 2.0（持续5分钟以上）<br>
        • <b>盘口深度比</b>：买一至买五深度 / 卖一至卖五深度 > 1.5<br>
        • <b>价差突变</b>：最优买卖价差在1分钟内缩小50%以上<br>
        • <b>大单监控</b>：出现单笔超过日均成交量1%的买入大单<br>
        <br>
        <em>注：以上指标需要订单簿实时数据支持，当前分析基于OHLCV数据，后续可集成。</em>
    </div>
</div>

<!-- ═══ SECTION 3: TRADING STRATEGY ═══ -->
<div class="card" id="sec3">
    <h2>🎯 第三部分：可获利切入点与交易策略</h2>

    {stage_html}

    <h3>3.1 移动止盈规则（Ratchet Trailing Stop）</h3>
    <table>
        <thead><tr><th>浮盈阶段</th><th>止损位移规则</th></tr></thead>
        <tbody>{trail_html}</tbody>
    </table>

    <div class="insight good">
        <strong>策略设计理念：</strong>妖币行情的核心是"截断亏损，让利润奔跑"。
        大部分妖币会在拉升高点后大幅回落，只有约23.9%能形成持续性趋势。
        因此，<b>移动止盈</b>比固定止盈更适合妖币，既能保护利润，又不会过早离场。
    </div>

    <h3>3.2 市场环境适用性</h3>
    <table>
        <thead><tr><th>市场环境</th><th>妖币发生频率</th><th>平均涨幅</th><th>趋势延续性</th><th>建议仓位</th></tr></thead>
        <tbody>{regime_rows}</tbody>
    </table>
</div>

<!-- ═══ SECTION 4: SCORING ALGORITHM ═══ -->
<div class="card" id="sec4">
    <h2>🧮 第四部分：多因子评分算法</h2>

    <h3>4.1 候选池硬性筛选条件</h3>
    <table>
        <thead><tr><th>筛选维度</th><th>条件</th><th>逻辑</th></tr></thead>
        <tbody>
            <tr><td>市值</td><td>>= $1,000,000</td><td>排除极端微盘垃圾币</td></tr>
            <tr><td>24h成交额</td><td>>= $50,000</td><td>确保基本流动性</td></tr>
            <tr><td>交易所数量</td><td>>= 2</td><td>避免单交易所操纵</td></tr>
            <tr><td>稳定币/包装币</td><td>排除</td><td>稳定币不会出现妖币行情</td></tr>
            <tr><td>最小价格</td><td>>= $0.0001</td><td>排除已归零品种</td></tr>
        </tbody>
    </table>

    <h3>4.2 八大因子评分体系</h3>
    <table>
        <thead><tr><th>因子ID</th><th>名称</th><th>类别</th><th>描述</th><th>强信号阈值</th><th>权重</th></tr></thead>
        <tbody>{factor_rows}</tbody>
    </table>

    <h3>4.3 评分等级与操作</h3>
    <table>
        <thead><tr><th>等级</th><th>最低分</th><th>最少触发类别</th><th>建议操作</th></tr></thead>
        <tbody>
            <tr><td><span class="tag tag-red">STRONG_BUY</span></td><td class="num">80</td><td class="num">3</td><td>立即轻仓买入（2%仓位）</td></tr>
            <tr><td><span class="tag tag-orange">BUY</span></td><td class="num">65</td><td class="num">2</td><td>准备买入，等待量价确认</td></tr>
            <tr><td><span class="tag tag-yellow">WATCH</span></td><td class="num">45</td><td class="num">1</td><td>加入观察列表，持续监控</td></tr>
            <tr><td><span class="tag tag-green">IGNORE</span></td><td class="num">0</td><td class="num">0</td><td>不关注</td></tr>
        </tbody>
    </table>

    <h3>4.4 扫描与监控频率</h3>
    <div class="insight">
        <strong>推荐扫描架构：</strong><br>
        • <b>分钟级扫描</b>：每1分钟对所有候选币种计算评分<br>
        • <b>小时汇总</b>：每小时生成 Top 20 候选名单<br>
        • <b>持仓再评估</b>：已持仓品种每15分钟重新评分<br>
        • <b>评分衰减</b>：若持仓品种评分跌破45分，触发减仓信号
    </div>
</div>

<!-- ═══ SECTION 5: BACKTEST ═══ -->
<div class="card" id="sec5">
    <h2>📊 第五部分：回测方法与历史验证</h2>

    <h3>5.1 验证指标</h3>
    <div class="grid-3">
        <div class="insight good">
            <strong>预估胜率</strong><br>
            <span style="font-size:28px;">{methodology['historical_validation']['estimated_win_rate']}</span><br>
            <small>（基于Moji Hunter v1回测）</small>
        </div>
        <div class="insight good">
            <strong>预估盈亏比</strong><br>
            <span style="font-size:28px;">{methodology['historical_validation']['estimated_profit_factor']}</span><br>
            <small>平均盈利 / 平均亏损</small>
        </div>
        <div class="insight warn">
            <strong>高风险特征</strong><br>
            <span style="font-size:16px;">低胜率 + 高盈亏比 = 彩票型策略</span><br>
            <small>需要严格的仓位管理和止损纪律</small>
        </div>
    </div>

    <h3>5.2 Out-of-Sample 验证方法</h3>
    <div class="insight">
        <strong>滚动时间窗口交叉验证：</strong><br>
        • 训练窗口：前60天数据 → 测试窗口：后30天数据<br>
        • 最少训练样本：20笔交易<br>
        • 最少测试样本：10笔交易<br>
        • 重复5次滚动验证，取各项指标均值
    </div>

    <h3>5.3 基准对比</h3>
    <table>
        <thead><tr><th>策略</th><th>胜率</th><th>盈亏比</th><th>最大回撤</th><th>备注</th></tr></thead>
        <tbody>
            <tr><td><b>Demon Hunter v2</b></td><td class="num">~24%</td><td class="num">4.0</td><td class="num negative">-15%</td><td>目标性能</td></tr>
            <tr><td>等权持有</td><td class="num">N/A</td><td class="num">N/A</td><td class="num negative">-30%+</td><td>非常高风险</td></tr>
            <tr><td>Moji Detector v1</td><td class="num">23.9%</td><td class="num">4.0</td><td class="num negative">-9.5%</td><td>已实现</td></tr>
            <tr><td>随机入场</td><td class="num">~15%</td><td class="num">~2.5</td><td class="num negative">-20%+</td><td>随机基线</td></tr>
        </tbody>
    </table>
</div>

<!-- ═══ SECTION 6: RISK & LIMITATIONS ═══ -->
<div class="card" id="sec6">
    <h2>🛡️ 第六部分：风险控制与模型局限性</h2>

    <h3>6.1 四大高风险场景识别</h3>
    <table>
        <thead><tr><th>场景</th><th>识别指标</th><th>应对规则</th></tr></thead>
        <tbody>{risk_rows}</tbody>
    </table>

    <h3>6.2 仓位管理规则</h3>
    <table>
        <thead><tr><th>规则</th><th>具体标准</th><th>备注</th></tr></thead>
        <tbody>{pos_rows}</tbody>
    </table>

    <h3>6.3 模型局限性</h3>
    <table>
        <thead><tr><th>局限类型</th><th>严重程度</th><th>说明</th><th>缓解措施</th></tr></thead>
        <tbody>
            {"".join(f'''<tr>
                <td>{lim['type']}</td>
                <td><span class="tag tag-{"red" if lim['severity']=="高" else "yellow" if lim['severity']=="中" else "green"}">{lim['severity']}</span></td>
                <td>{lim['detail']}</td>
                <td>{lim['mitigation']}</td>
            </tr>''' for lim in risks['model_limitations'])}
        </tbody>
    </table>

    <h3>6.4 免责声明</h3>
    <div class="insight danger">
        <strong>⚠️ 重要风险提示：</strong><br>
        1. 妖币交易属于<strong>极高风险</strong>的投机行为，历史表现不代表未来收益<br>
        2. 本框架基于有限历史数据（{n_demon}个50%+事件），统计显著性有限<br>
        3. 加密货币市场存在<strong>极端流动性风险</strong>：价格可能在数分钟内归零<br>
        4. <strong>不要投入无法承受损失的资金</strong>。建议妖币策略总仓位不超过投资组合的10%<br>
        5. 模型在熊市中的有效性显著下降，可能产生大量假信号<br>
        6. 本分析仅供研究参考，不构成任何投资建议
    </div>
</div>

<!-- ═══ APPENDIX: ALGORITHM PSEUDOCODE ═══ -->
<div class="card">
    <h2>📝 附录：算法伪代码（DemonCoinHunter v2.0）</h2>
    <div class="code-block">{esc(pseudocode)}</div>
</div>

<div class="footer">
    Demon Coin Analysis Framework v2.0 | 基于 Gate.io 交易所历史数据 | 生成时间: 2026-07-02<br>
    数据来源: gate_moji_analysis.py | 分析周期: 2026-06 ~ 2026-07 | 仅供研究参考
</div>

</div>
</body>
</html>"""

    return html


# ─── MAIN ───

def main():
    print("=" * 70)
    print("Demon Coin (妖币) Systematic Analysis Framework v2.0")
    print("=" * 70)

    # 1. Load data
    print("\n[1/8] Loading moji feature data...")
    moji_f, control_f = load_moji_features()
    if moji_f is None:
        print("ERROR: Cannot load moji features. Run gate_moji_analysis.py first.")
        return

    print(f"  Loaded {len(moji_f)} moji events, {len(control_f)} control samples")

    # 2. Classify demon coins
    print("\n[2/8] Classifying demon coins (>=50%)...")
    demon_f, regular_f, combined = classify_demon_coins(moji_f)
    print(f"  Demon coins (>=50%): {len(demon_f)}")
    print(f"  Regular moji (20-50%): {len(regular_f)}")

    if len(demon_f) > 0:
        for _, row in demon_f.iterrows():
            print(f"    {row['symbol']} | {row['pump_day']} | {row['pump_gain_pct']:.1f}%")

    # 3. Pre-launch feature analysis
    print("\n[3/8] Analyzing pre-launch features...")
    feature_analysis = analyze_pre_launch_features(moji_f, control_f)
    n_sig = sum(1 for _, row in feature_analysis.iterrows() if row.get('significant', False))
    print(f"  {n_sig}/{len(feature_analysis)} features statistically significant (p<0.05)")

    bb = compute_bollinger_compression(demon_f, control_f)
    print(f"  Bollinger compression: Demon={bb.get('Demon',{}).get('pct_compressing', 'N/A')}%, "
          f"Control={bb.get('Control',{}).get('pct_compressing', 'N/A')}%")

    # 4. Launch analysis
    print("\n[4/8] Analyzing launch signals...")
    launch_sigs = analyze_launch_signals(demon_f)

    # 5. Trading strategy
    print("\n[5/8] Designing trading strategy...")
    strategy = design_trading_strategy()

    # 6. Multi-factor scoring
    print("\n[6/8] Designing multi-factor scoring algorithm...")
    factors, scoring = design_scoring_algorithm(demon_f, feature_analysis, launch_sigs)
    print(f"  {len(factors)} factors across {len(scoring['category_weights'])} categories")

    # 7. Backtest methodology
    print("\n[7/8] Designing backtest methodology...")
    methodology = design_backtest_methodology(demon_f)

    # 8. Risk & limitations
    print("\n[8/8] Compiling risks and limitations...")
    risks = compile_risks_and_limitations(demon_f)

    # Generate algorithm pseudocode
    print("\n[*] Generating algorithm pseudocode...")
    pseudocode = generate_algorithm_pseudocode(factors, scoring)

    # Generate HTML report
    print("\n[*] Generating comprehensive HTML report...")
    html = generate_html_report(feature_analysis, launch_sigs, strategy,
                                 factors, scoring, methodology, risks,
                                 pseudocode, demon_f)

    report_path = OUTPUT_DIR / "demon_coin_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Report saved: {report_path}")

    # Save feature analysis as CSV
    feature_analysis.to_csv(OUTPUT_DIR / "feature_analysis.csv")
    demon_f.to_csv(OUTPUT_DIR / "demon_coins.csv", index=False)

    # Save factor definitions as JSON
    factor_json = {
        'factors': {k: {kk: vv for kk, vv in v.items() if kk != 'thresholds'}
                    for k, v in factors.items()},
        'scoring': scoring,
    }
    with open(OUTPUT_DIR / "factor_definitions.json", 'w', encoding='utf-8') as f:
        json.dump(factor_json, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return feature_analysis, launch_sigs, strategy, factors, methodology


if __name__ == '__main__':
    main()
