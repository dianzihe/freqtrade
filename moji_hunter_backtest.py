"""
MojiCoinHunter 独立回测验证脚本
=================================
直接用本地 feather 数据模拟 MojiCoinDetector 评分逻辑和交易管理,
无需连接 Gate API, 快速验证策略效果.

⚠️ 这是简化模拟, 不完全等同于 freqtrade 实际执行 (缺少撮合、滑点、手续费等细节)
"""

import os, json, glob, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DATA_DIR = Path("E:/source/freqtrade/user_data/data/gate")
OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/moji-coin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  MojiCoinDetector v2 参数 (与策略文件一致)
# ═══════════════════════════════════════════════════════════════════════

PARAMS = {
    # Rule weights and thresholds — v3: 更严格的阈值减少假阳性
    'R1_vol_ratio': {'threshold': 2.0, 'weight': 25},    # 量比 (moji 2.93, ctrl 1.93 → 阈值提高到2.0)
    'R2_sma_slope': {'threshold': 1.5, 'weight': 25},    # SMA斜率% (moji 2.75 → 提高到1.5, 加权)
    'R3_cum_change': {'threshold': 3.0, 'weight': 25},   # 累计涨幅% (moji 3.88 → 提高到3.0, 加权)
    'R4_range_pct': {'threshold': 8.0, 'weight': 15},    # 振幅% (moji 10.1 → 提高到8.0)
    'R5_range_ratio': {'threshold': 1.3, 'weight': 10},  # 振幅扩张 (moji 1.58 → 提高到1.3)
    'R6_stair_score': {'threshold': 0.80, 'weight': 0},  # 价格阶梯 (弱信号, 权重归零)
    'R7_vol_stair': {'threshold': 0.60, 'weight': 0},    # 量阶梯 (极弱信号, 权重归零)
    'score_threshold': 70,  # 只接受 strong 信号
    'min_rules': 2,
}

# 交易管理参数 — v3: 更保守的资金管理
TRADE_PARAMS = {
    'stake_per_trade': 1.0,     # $1/仓
    'max_open_trades': 20,      # 20仓上限
    'total_capital': 20.0,      # $20 总资金
    'stoploss_pct': -0.10,      # 10%硬止损 (放宽, meme币波动大)
    'commission_pct': 0.002,    # 0.2%手续费 (Gate现货)
    'max_hold_hours': 72,       # 3天最长持有
    'dud_early_hours': 8,       # 8h急跌止损 (放宽)
    'dud_half_day_hours': 12,   # 12h废票
    'dud_1day_hours': 24,       # 24h废票
    'dud_2day_hours': 48,       # 48h平庸
    # 棘轮止损
    'ratchet_10pct_protect': 0.0,   # 10%利润保本
    'ratchet_20pct_protect': 0.10,  # 20%利润保10%
    'ratchet_50pct_protect': 0.25,  # 50%利润保25%
    'ratchet_100pct_protect': 0.50, # 100%利润保50%
    'pullback_drawdown': 0.12,      # 从高点回撤12%退出 (收紧)
    'pullback_min_profit': 0.08,    # 回撤退出时利润至少8%
    'pullback_peak_min': 0.15,      # 历史高点至少15%
    'global_cooldown_min': 120,     # 全局冷却2小时
    'min_quote_volume_24h': 50000,  # 24h报价量至少$50k
}


# ═══════════════════════════════════════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════════════════════════════════════

def compute_moji_indicators(df):
    """在 5m DataFrame 上计算全部 MojiCoinDetector 指标"""
    df = df.copy()

    # SMA12
    df['sma12'] = df['close'].rolling(12, min_periods=6).mean()

    # R1: 量比 (最近12 bars / 之前60 bars)
    vol_recent = df['volume'].rolling(12, min_periods=3).mean()
    vol_earlier = df['volume'].shift(12).rolling(60, min_periods=24).mean()
    df['vol_ratio'] = (vol_recent / vol_earlier).replace([np.inf, -np.inf], 0).fillna(0)

    # R2: SMA12 斜率 (6h)
    sma_now = df['sma12']
    sma_6h = df['sma12'].shift(72)
    df['sma_slope_pct'] = ((sma_now - sma_6h) / sma_6h * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # R3: 累计涨幅 (6h)
    close_6h = df['close'].shift(72)
    df['cum_change_pct'] = ((df['close'] - close_6h) / close_6h * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # R4: 振幅 (6h)
    high_6h = df['high'].rolling(72, min_periods=36).max()
    low_6h = df['low'].rolling(72, min_periods=36).min()
    df['range_pct_6h'] = ((high_6h - low_6h) / low_6h * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # R5: 振幅扩张
    high_recent = df['high'].rolling(36, min_periods=18).max()
    low_recent = df['low'].rolling(36, min_periods=18).min()
    range_recent = (high_recent - low_recent) / low_recent.replace(0, np.nan)

    high_earlier = df['high'].shift(36).rolling(36, min_periods=18).max()
    low_earlier = df['low'].shift(36).rolling(36, min_periods=18).min()
    range_earlier = (high_earlier - low_earlier) / low_earlier.replace(0, np.nan)

    df['range_ratio'] = (range_recent / range_earlier).replace([np.inf, -np.inf], 0).fillna(0)

    # R6: 价格阶梯
    c = df['close']
    df['stair_score'] = (
        ((c.shift(-1) >= c.shift(-2)) if False else
         ((c >= c.shift(1)).astype(int) +
          (c.shift(1) >= c.shift(2)).astype(int) +
          (c.shift(2) >= c.shift(3)).astype(int) +
          (c.shift(3) >= c.shift(4)).astype(int) +
          (c.shift(4) >= c.shift(5)).astype(int))) / 5.0
    )
    # simpler approach
    df['stair_score'] = (
        ((c >= c.shift(1)).astype(int) +
         (c.shift(1) >= c.shift(2)).astype(int) +
         (c.shift(2) >= c.shift(3)).astype(int) +
         (c.shift(3) >= c.shift(4)).astype(int) +
         (c.shift(4) >= c.shift(5)).astype(int)) / 5.0
    )

    # R7: 量阶梯
    v = df['volume']
    df['vol_stair_score'] = (
        ((v >= v.shift(1)).astype(int) +
         (v.shift(1) >= v.shift(2)).astype(int) +
         (v.shift(2) >= v.shift(3)).astype(int) +
         (v.shift(3) >= v.shift(4)).astype(int) +
         (v.shift(4) >= v.shift(5)).astype(int)) / 5.0
    )

    return df


def compute_moji_score(df, params=PARAMS):
    """计算加权评分和触发规则数"""
    score = pd.Series(0.0, index=df.index)
    rules = pd.Series(0, index=df.index, dtype=int)

    rule_defs = [
        ('vol_ratio', 'R1_vol_ratio'),
        ('sma_slope_pct', 'R2_sma_slope'),
        ('cum_change_pct', 'R3_cum_change'),
        ('range_pct_6h', 'R4_range_pct'),
        ('range_ratio', 'R5_range_ratio'),
        ('stair_score', 'R6_stair_score'),
        ('vol_stair_score', 'R7_vol_stair'),
    ]

    for col, rule_key in rule_defs:
        threshold = params[rule_key]['threshold']
        weight = params[rule_key]['weight']
        hit = df[col] >= threshold
        score += hit.astype(float) * weight
        rules += hit.astype(int)

    return score, rules


# ═══════════════════════════════════════════════════════════════════════
#  交易模拟引擎
# ═══════════════════════════════════════════════════════════════════════

class TradeSimulator:
    """简化交易模拟器"""

    def __init__(self, params=TRADE_PARAMS, detector_params=PARAMS):
        self.trade_params = params
        self.detector_params = detector_params
        self.open_trades = {}  # pair -> trade_info
        self.closed_trades = []
        self.available_capital = params['total_capital']
        self.total_commission_paid = 0.0

    def try_enter(self, pair, timestamp, price, score, rules_triggered, signal_tag):
        """尝试开仓"""
        tp = self.trade_params

        if len(self.open_trades) >= tp['max_open_trades']:
            return False
        if self.available_capital < tp['stake_per_trade']:
            return False

        # 记录开仓
        stake = tp['stake_per_trade']
        commission = stake * tp['commission_pct']
        self.available_capital -= (stake + commission)
        self.total_commission_paid += commission

        self.open_trades[pair] = {
            'pair': pair,
            'entry_time': timestamp,
            'entry_price': price,
            'stake': stake,
            'amount': (stake - commission) / price,
            'score': score,
            'rules_triggered': rules_triggered,
            'signal_tag': signal_tag,
            'max_rate': price,
            'peak_profit_pct': 0.0,
        }
        return True

    def check_exit(self, pair, timestamp, price):
        """检查是否需要平仓"""
        trade = self.open_trades.get(pair)
        if trade is None:
            return None

        tp = self.trade_params
        entry_price = trade['entry_price']
        profit_pct = (price - entry_price) / entry_price

        # 更新峰值
        if price > trade['max_rate']:
            trade['max_rate'] = price
            trade['peak_profit_pct'] = (trade['max_rate'] - entry_price) / entry_price

        age_h = (timestamp - trade['entry_time']).total_seconds() / 3600.0

        # ── 硬止损 ──
        if profit_pct <= tp['stoploss_pct']:
            return ('stoploss', profit_pct)

        # ── 棘轮止损 ──
        if profit_pct >= 1.00:
            protect = tp['ratchet_100pct_protect']
            if profit_pct - protect < 0:
                return ('ratchet_100', profit_pct)
        if profit_pct >= 0.50:
            protect = tp['ratchet_50pct_protect']
            if profit_pct < protect:
                return ('ratchet_50', profit_pct)
        if profit_pct >= 0.20:
            protect = tp['ratchet_20pct_protect']
            if profit_pct < protect:
                return ('ratchet_20', profit_pct)
        if profit_pct >= 0.10:
            if profit_pct < tp['ratchet_10pct_protect']:
                return ('ratchet_10', profit_pct)

        # ── 时间止损 ──
        if age_h >= tp['dud_early_hours'] and profit_pct < -0.03:
            return ('dud_early', profit_pct)
        if age_h >= tp['dud_half_day_hours'] and profit_pct < 0:
            return ('stale_half_day', profit_pct)
        if age_h >= tp['dud_1day_hours'] and profit_pct < 0.01:
            return ('stale_dud', profit_pct)
        if age_h >= tp['dud_2day_hours'] and profit_pct < 0.05:
            return ('medium_2d', profit_pct)
        if age_h >= tp['max_hold_hours']:
            return ('max_hold_3d', profit_pct)

        # ── 从高点回撤 ──
        if trade['peak_profit_pct'] >= tp['pullback_peak_min'] and profit_pct >= tp['pullback_min_profit']:
            drawdown = trade['peak_profit_pct'] - profit_pct
            if drawdown >= tp['pullback_drawdown']:
                return ('pullback_exit', profit_pct)

        return None

    def close_trade(self, pair, timestamp, price, exit_reason, profit_pct):
        """平仓"""
        trade = self.open_trades.pop(pair)
        tp = self.trade_params

        # 计算收益
        stake_returned = trade['stake'] * (1 + profit_pct)
        commission_exit = stake_returned * tp['commission_pct']
        net_returned = stake_returned - commission_exit
        self.total_commission_paid += commission_exit
        self.available_capital += net_returned

        # 记录
        trade['exit_time'] = timestamp
        trade['exit_price'] = price
        trade['exit_reason'] = exit_reason
        trade['profit_pct'] = profit_pct * 100  # 转为百分比
        trade['net_profit_usd'] = net_returned - trade['stake']
        trade['hold_hours'] = (timestamp - trade['entry_time']).total_seconds() / 3600.0
        trade['peak_profit_pct'] *= 100  # 转为百分比

        self.closed_trades.append(trade)
        return trade


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════

def load_all_data():
    """加载全部 5m feather 数据"""
    files = sorted(DATA_DIR.glob("*-5m.feather"))
    coin_data = {}
    for f in files:
        symbol = f.name.replace("-5m.feather", "")
        df = pd.read_feather(f)
        df = df.set_index('date').sort_index()
        df = compute_moji_indicators(df)
        df['moji_score'], df['moji_rules'] = compute_moji_score(df)
        coin_data[symbol] = df
    return coin_data


def simulate_trading(coin_data):
    """模拟交易"""
    sim = TradeSimulator()
    tp = sim.trade_params

    # 获取全局时间线 (所有币共享时间轴)
    all_times = set()
    for symbol, df in coin_data.items():
        all_times.update(df.index)
    all_times = sorted(all_times)

    # 每5分钟扫描所有币
    entry_cooldown = {}  # pair -> last_entry_time
    last_global_entry = None
    global_cooldown = tp.get('global_cooldown_min', 120)

    for t in all_times:
        # ── 先检查现有仓位是否需要退出 ──
        for pair in list(sim.open_trades.keys()):
            df = coin_data.get(pair)
            if df is None:
                continue
            if t not in df.index:
                continue
            price = df.loc[t, 'close']
            result = sim.check_exit(pair, t, price)
            if result is not None:
                reason, profit_pct = result
                sim.close_trade(pair, t, price, reason, profit_pct)
                logger.debug(f"  EXIT {pair} @ {t}: reason={reason}, profit={profit_pct:.2%}")

        # ── 全局冷却 ──
        if last_global_entry and (t - last_global_entry).total_seconds() < global_cooldown * 60:
            continue

        # ── 扫描新入场信号 (只接受 strong) ──
        best_signal = None
        best_score = 0

        for symbol, df in coin_data.items():
            if symbol in sim.open_trades:
                continue

            # 冷却期
            last_entry = entry_cooldown.get(symbol)
            if last_entry and (t - last_entry).total_seconds() < global_cooldown * 60:
                continue

            if t not in df.index:
                continue

            row = df.loc[t]
            score = row['moji_score']
            rules = row['moji_rules']

            # 只接受 strong 信号 (score >= 70, 3+ rules)
            if score < tp.get('score_threshold', 70) or rules < tp.get('min_rules', 2):
                continue

            # 流动性过滤
            qvol = row.get('volume', 0) * row.get('close', 0)
            if qvol * 288 < tp.get('min_quote_volume_24h', 50000):
                continue

            price = row['close']
            if price < 0.0001 or price > 500:
                continue

            # 选评分最高的币优先入场
            if score > best_score:
                best_signal = (symbol, score, rules, price)

        # ── 入场 ──
        if best_signal is not None:
            symbol, score, rules, price = best_signal
            tag = 'moji_strong' if score >= 70 and rules >= 3 else 'moji_medium'
            entered = sim.try_enter(symbol, t, price, score, rules, tag)
            if entered:
                entry_cooldown[symbol] = t
                last_global_entry = t
                logger.info(f"  ENTER {symbol} @ {t}: tag={tag}, score={score:.0f}, rules={rules}")

    return sim


def generate_report(sim):
    """生成回测报告"""
    trades = sim.closed_trades

    if len(trades) == 0:
        print("\n⚠️ No trades executed! Signal threshold may be too high.")
        return

    # 统计
    df = pd.DataFrame(trades)

    total_trades = len(df)
    winning = df[df['profit_pct'] > 0]
    losing = df[df['profit_pct'] <= 0]
    win_rate = len(winning) / total_trades * 100

    avg_profit = df['profit_pct'].mean()
    avg_win = winning['profit_pct'].mean() if len(winning) > 0 else 0
    avg_loss = losing['profit_pct'].mean() if len(losing) > 0 else 0

    total_net = df['net_profit_usd'].sum()
    final_capital = sim.available_capital
    roi_pct = (final_capital - TRADE_PARAMS['total_capital']) / TRADE_PARAMS['total_capital'] * 100

    # 信号分布
    tag_counts = df['signal_tag'].value_counts()
    exit_counts = df['exit_reason'].value_counts()

    # 最佳/最差交易
    best_trade = df.loc[df['profit_pct'].idxmax()]
    worst_trade = df.loc[df['profit_pct'].idxmin()]

    print("\n" + "="*60)
    print("🔥 MojiCoinHunter 独立回测报告")
    print("="*60)
    print(f"  总交易数: {total_trades}")
    print(f"  赢率: {win_rate:.1f}% ({len(winning)}胜 / {len(losing)}负)")
    print(f"  平均利润: {avg_profit:.2f}%")
    print(f"  平均盈利: {avg_win:.2f}% (赢家)")
    print(f"  平均亏损: {avg_loss:.2f}% (输家)")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}x (avg_win/avg_loss)")
    print(f"  总净收益: $${total_net:.2f}")
    print(f"  最终资金: $${final_capital:.2f} (初始 $${TRADE_PARAMS['total_capital']:.0f})")
    print(f"  ROI: {roi_pct:.2f}%")
    print(f"  手续费总计: $${sim.total_commission_paid:.2f}")
    print()
    print(f"  信号分布:")
    for tag, count in tag_counts.items():
        pct = count / total_trades * 100
        avg_p = df[df['signal_tag'] == tag]['profit_pct'].mean()
        print(f"    {tag}: {count} ({pct:.1f}%) avg_profit={avg_p:.2f}%")
    print()
    print(f"  退出原因分布:")
    for reason, count in exit_counts.items():
        pct = count / total_trades * 100
        avg_p = df[df['exit_reason'] == reason]['profit_pct'].mean()
        print(f"    {reason}: {count} ({pct:.1f}%) avg_profit={avg_p:.2f}%")
    print()
    print(f"  🏆 最佳交易: {best_trade['pair']} profit={best_trade['profit_pct']:.2f}% "
          f"(score={best_trade['score']}, tag={best_trade['signal_tag']})")
    print(f"  💀 最差交易: {worst_trade['pair']} profit={worst_trade['profit_pct']:.2f}% "
          f"(score={worst_trade['score']}, tag={worst_trade['signal_tag']})")
    print()
    print(f"  平均持仓时长: {df['hold_hours'].mean():.1f}h")
    print(f"  平均峰值利润: {df['peak_profit_pct'].mean():.2f}%")
    print("="*60)

    # 保存详细交易记录
    df.to_csv(OUTPUT_DIR / 'moji_hunter_bt_trades.csv', index=False)
    logger.info(f"Saved trades to {OUTPUT_DIR / 'moji_hunter_bt_trades.csv'}")

    return df


def analyze_signal_quality(coin_data):
    """分析信号质量: 不同评分阈值下的命中率和收益"""
    print("\n📊 信号质量分析 (不同阈值)")

    thresholds = [40, 50, 60, 70, 80]

    for thresh in thresholds:
        signals = 0
        hit_moji = 0  # 在妖币日内触发
        # 加载妖币列表
        moji_df = pd.read_csv(OUTPUT_DIR / 'moji_features.csv')
        moji_days = set(zip(moji_df['symbol'], moji_df['pump_day']))

        for symbol, df in coin_data.items():
            score_col = df['moji_score']
            hits = score_col >= thresh
            signals += hits.sum()

            # 检查哪些信号出现在妖币日
            for idx in df.index[hits]:
                day = str(idx.date())
                if (symbol, day) in moji_days:
                    hit_moji += 1

        moji_hit_rate = hit_moji / max(signals, 1) * 100
        print(f"  阈值 >= {thresh}: {signals} 信号, 妖币日命中率 {moji_hit_rate:.1f}% ({hit_moji}/{signals})")


def main():
    logger.info("Loading data...")
    coin_data = load_all_data()
    logger.info(f"Loaded {len(coin_data)} coins")

    logger.info("Simulating trading...")
    sim = simulate_trading(coin_data)

    logger.info("Generating report...")
    trades_df = generate_report(sim)

    if trades_df is not None:
        analyze_signal_quality(coin_data)

    # 保存参数
    with open(OUTPUT_DIR / 'moji_hunter_params.json', 'w') as f:
        json.dump({'detector': PARAMS, 'trade': TRADE_PARAMS}, f, indent=2)


if __name__ == '__main__':
    main()
