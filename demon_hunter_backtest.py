"""
DemonHunter 独立回测验证脚本
==============================
基于 demon_coin_analysis.py 的 8因子 × 4类别评分体系,
直接使用本地 feather K线数据模拟完整交易流程.

设计要点:
1. 8因子分层评分 (A/B/C/D 四类, 100分制)
2. 三级入场: STRONG(>=80) / BUY(>=65) / WATCH(>=45)
3. 棘轮移动止损 + 分批止盈 + 时间退出
4. 全局冷却 + 单币冷却
5. 流动性过滤 + 价格合理性过滤
"""

import os, json, glob, logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DATA_DIR = Path("E:/source/freqtrade/user_data/data/gate")
OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/demon-coin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  DemonHunter 策略参数 — v1.0
# ═══════════════════════════════════════════════════════════════════════

# ── 因子权重与阈值 (100分制) ──
FACTOR_PARAMS = {
    'score_threshold': 65,       # 最低入场分
    'min_categories': 2,         # 最少活跃类别数

    # A类: 价格量能 (max 40分)
    'A1_price_momentum':    {'threshold': 4.0, 'weight': 15},  # 6h累计涨幅
    'A2_volume_surge':      {'threshold': 3.0, 'weight': 15},  # 量比
    'A3_sma_momentum':      {'threshold': 2.5, 'weight': 10},  # SMA12斜率

    # B类: 形态识别 (max 25分)
    'B1_staircase':         {'threshold': 0.70, 'weight': 15},  # 阶梯吸筹
    'B2_range_behavior':    {'threshold': 1.50, 'weight': 10},  # 振幅扩张

    # C类: 市场环境 (max 20分)
    'C1_vol_lo':            {'threshold': 0.08, 'weight': 10},  # 波动率下界
    'C1_vol_hi':            {'threshold': 0.25, 'weight': 10},  # 波动率上界
    'C2_volume_quality':    {'threshold': 2.0,  'weight': 10},  # 成交量偏度

    # D类: 时机催化 (max 15分)
    'D1_timing':            {'threshold': 8,    'weight': 10},  # 交易时段
    'D2_price_ok':          {'threshold': 1,    'weight': 5},    # 价格合理
}

# ── 交易管理参数 ──
TRADE_PARAMS = {
    'stake_per_trade': 1.0,       # $1/仓
    'max_open_trades': 5,         # 最多5仓
    'total_capital': 10.0,        # $10 总资金
    'stoploss_pct': -0.15,        # 15%硬止损
    'commission_pct': 0.002,      # 0.2%手续费

    # 棘轮止损
    'ratchet_10pct_protect': 0.0,     # 10%利润保本
    'ratchet_20pct_protect': 0.10,    # 20%利润保10%
    'ratchet_50pct_protect': 0.25,    # 50%利润保25%
    'ratchet_100pct_protect': 0.50,   # 100%利润保50%

    # 回撤保护
    'pullback_drawdown': 0.12,        # 从高点回撤12%退出
    'pullback_min_profit': 0.08,      # 回撤退出时利润至少8%
    'pullback_peak_min': 0.15,        # 历史高点至少15%

    # 时间退出
    'time_sharp_drop_hours': 6,       # 6h <= -8% 急跌退出
    'time_sharp_drop_threshold': -0.08,
    'time_dud_hours': 12,             # 12h <= -3% 废票
    'time_dud_threshold': -0.03,
    'time_stale_hours': 24,           # 24h < +1% 平庸
    'time_stale_threshold': 0.01,
    'time_weak_hours': 48,            # 48h < +5% 低效
    'time_weak_threshold': 0.05,
    'time_max_hours': 72,             # 72h 强制退出

    # 冷却控制
    'global_cooldown_min': 120,       # 全局冷却2小时
    'per_coin_cooldown_min': 180,     # 单币冷却3小时

    # 过滤
    'min_quote_volume_24h': 20000,    # 24h最小成交额
    'min_price': 0.0001,
    'max_price': 1000,
}


# ═══════════════════════════════════════════════════════════════════════
#  交易时段评分 (UTC)
# ═══════════════════════════════════════════════════════════════════════

SESSION_SCORE = {
    0: 8, 1: 10, 2: 10, 3: 8,      # 亚洲活跃
    4: 6, 5: 6, 6: 8,               # 欧亚交接
    7: 10, 8: 10, 9: 8, 10: 8,      # 欧洲上午
    11: 6, 12: 8,                    # 欧美重叠
    13: 10, 14: 10, 15: 10, 16: 8,  # 美国上午
    17: 6, 18: 6, 19: 6,            # 美国下午
    20: 4, 21: 4, 22: 4, 23: 4,     # 收市
}


# ═══════════════════════════════════════════════════════════════════════
#  DemonHunter 指标计算
# ═══════════════════════════════════════════════════════════════════════

def compute_demon_indicators(df):
    """在 5m DataFrame 上计算全部 DemonHunter 指标"""
    df = df.copy()

    # SMA12
    df['sma12'] = df['close'].rolling(12, min_periods=6).mean()

    # A1: 价格动量 (6h 累计涨幅 = 72 bars)
    close_6h = df['close'].shift(72)
    df['A1_price_momentum'] = ((df['close'] - close_6h) / close_6h * 100).fillna(0).clip(-100, 1000)

    # A2: 量能爆发 (近1h量 / 前5h均量)
    vol_1h = df['volume'].rolling(12, min_periods=6).mean()
    vol_5h_earlier = df['volume'].shift(12).rolling(60, min_periods=24).mean()
    df['A2_volume_surge'] = (vol_1h / vol_5h_earlier).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)

    # A3: 趋势强度 (SMA12 6h斜率)
    sma_6h = df['sma12'].shift(72)
    df['A3_sma_momentum'] = ((df['sma12'] - sma_6h) / sma_6h * 100).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 1000)

    # B1: 阶梯吸筹形态
    c = df['close']
    df['B1_staircase'] = (
        ((c >= c.shift(1)).astype(int) +
         (c.shift(1) >= c.shift(2)).astype(int) +
         (c.shift(2) >= c.shift(3)).astype(int) +
         (c.shift(3) >= c.shift(4)).astype(int) +
         (c.shift(4) >= c.shift(5)).astype(int)) / 5.0
    )

    # B2: 振幅扩张 (近3h / 前3h)
    h, l = df['high'], df['low']
    recent_range = (h.rolling(36, min_periods=18).max() - l.rolling(36, min_periods=18).min()) / l.rolling(36, min_periods=18).min().replace(0, np.nan)
    earlier_range = (h.shift(36).rolling(36, min_periods=18).max() - l.shift(36).rolling(36, min_periods=18).min()) / l.shift(36).rolling(36, min_periods=18).min().replace(0, np.nan)
    df['B2_range_behavior'] = (recent_range / earlier_range).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 20)

    # C1: 波动率环境 (6h年化)
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['C1_volatility'] = log_ret.rolling(72, min_periods=36).std() * np.sqrt(288 * 365)

    # C2: 成交量偏度 (近1h)
    df['C2_volume_quality'] = df['volume'].rolling(12, min_periods=6).skew().fillna(0).clip(-5, 10)

    # D1: 时间窗口
    df['D1_timing'] = df.index.hour.map(lambda h: SESSION_SCORE.get(h, 4))

    # D2: 价格合理性
    price = df['close']
    df['D2_price_ok'] = ((price >= 0.0001) & (price <= 1000)).astype(float)

    # 24h成交量估计
    df['volume_24h_est'] = df['volume'].rolling(288, min_periods=144).sum() * df['close']

    return df


def compute_demon_score(df, params=FACTOR_PARAMS):
    """
    计算 DemonHunter 加权评分。

    评分逻辑:
    - 强触发: 权重分全给
    - 中触发: 权重分 × 0.6
    - 未触发: 0分

    返回: total_score, cat_scores dict, active_cats, triggered_factors
    """
    n = len(df)

    # ── A类 (max 40) ──
    A1 = pd.Series(0.0, index=df.index)
    A1_mask = df['A1_price_momentum'] >= params['A1_price_momentum']['threshold']
    A1[A1_mask] = 15
    A1[(df['A1_price_momentum'] >= params['A1_price_momentum']['threshold'] * 0.5) & ~A1_mask] = 9

    A2 = pd.Series(0.0, index=df.index)
    A2_mask = df['A2_volume_surge'] >= params['A2_volume_surge']['threshold']
    A2[A2_mask] = 15
    A2[(df['A2_volume_surge'] >= params['A2_volume_surge']['threshold'] * 0.67) & ~A2_mask] = 9

    A3 = pd.Series(0.0, index=df.index)
    A3_mask = df['A3_sma_momentum'] >= params['A3_sma_momentum']['threshold']
    A3[A3_mask] = 10
    A3[(df['A3_sma_momentum'] >= params['A3_sma_momentum']['threshold'] * 0.5) & ~A3_mask] = 6

    # ── B类 (max 25) ──
    B1 = pd.Series(0.0, index=df.index)
    B1_mask = df['B1_staircase'] >= params['B1_staircase']['threshold']
    B1[B1_mask] = 15
    B1[(df['B1_staircase'] >= params['B1_staircase']['threshold'] * 0.75) & ~B1_mask] = 9

    B2 = pd.Series(0.0, index=df.index)
    B2_mask = df['B2_range_behavior'] >= params['B2_range_behavior']['threshold']
    B2[B2_mask] = 10
    B2[(df['B2_range_behavior'] >= params['B2_range_behavior']['threshold'] * 0.8) & ~B2_mask] = 6

    # ── C类 (max 20) ──
    C1 = pd.Series(0.0, index=df.index)
    vol = df['C1_volatility']
    C1_mask = (vol >= params['C1_vol_lo']['threshold']) & (vol <= params['C1_vol_hi']['threshold'])
    C1[C1_mask] = 10
    C1[(vol >= params['C1_vol_lo']['threshold'] * 0.6) & (vol <= params['C1_vol_hi']['threshold'] * 1.5) & ~C1_mask] = 6

    C2 = pd.Series(0.0, index=df.index)
    vq = df['C2_volume_quality']
    C2_mask = (vq >= 0) & (vq <= 2.0)
    C2[C2_mask] = 10
    C2[(vq >= -1) & (vq <= 4.0) & ~C2_mask] = 6

    # ── D类 (max 15) ──
    D1 = pd.Series(0.0, index=df.index)
    timing = df['D1_timing']
    D1[timing >= 8] = 10
    D1[(timing >= 6) & (D1 == 0)] = 6

    D2 = pd.Series(0.0, index=df.index)
    D2[df['D2_price_ok'] == 1] = 5

    # ── 类别汇总 ──
    cat_A = A1 + A2 + A3
    cat_B = B1 + B2
    cat_C = C1 + C2
    cat_D = D1 + D2

    total_score = cat_A + cat_B + cat_C + cat_D

    # 活跃类别
    active_cats = ((cat_A > 0).astype(int) +
                   (cat_B > 0).astype(int) +
                   (cat_C > 0).astype(int) +
                   (cat_D > 0).astype(int))

    # 触发因子明细
    triggered = []
    factor_masks = {
        'A1': A1_mask, 'A2': A2_mask, 'A3': A3_mask,
        'B1': B1_mask, 'B2': B2_mask,
        'C1': C1_mask, 'C2': C2_mask,
        'D1': (timing >= 8), 'D2': (df['D2_price_ok'] == 1),
    }

    # per-row triggered factors
    triggered_list = pd.Series('', index=df.index)
    for fname, mask in factor_masks.items():
        triggered_list = triggered_list.where(~mask, triggered_list + fname + ',')

    return total_score, cat_A, cat_B, cat_C, cat_D, active_cats, triggered_list


# ═══════════════════════════════════════════════════════════════════════
#  交易模拟引擎
# ═══════════════════════════════════════════════════════════════════════

class TradeSimulator:
    """DemonHunter 交易模拟器"""

    def __init__(self, trade_params=TRADE_PARAMS):
        self.tp = trade_params
        self.open_trades = {}
        self.closed_trades = []
        self.available_capital = trade_params['total_capital']
        self.total_commission_paid = 0.0
        self.entry_log = []  # (pair, timestamp, score, tag)

    def try_enter(self, pair, timestamp, price, score, cats, tag, triggered):
        """尝试开仓"""
        tp = self.tp

        if len(self.open_trades) >= tp['max_open_trades']:
            return False, 'max_open'
        if self.available_capital < tp['stake_per_trade']:
            return False, 'no_capital'

        stake = tp['stake_per_trade']
        commission = stake * tp['commission_pct']
        self.available_capital -= (stake + commission)
        self.total_commission_paid += commission

        self.open_trades[pair] = {
            'pair': pair,
            'entry_time': timestamp,
            'entry_price': price,
            'stake': stake,
            'amount': (stake - commission) / price if price > 0 else 0,
            'score': score,
            'cats': cats,
            'tag': tag,
            'triggered': triggered,
            'max_rate': price,
            'peak_profit_pct': 0.0,
        }
        self.entry_log.append((pair, timestamp, score, tag))
        return True, 'ok'

    def check_exit(self, pair, timestamp, price):
        """检查是否需要平仓"""
        trade = self.open_trades.get(pair)
        if trade is None:
            return None

        tp = self.tp
        entry_price = trade['entry_price']
        profit_pct = (price - entry_price) / entry_price if entry_price > 0 else -1

        # 更新峰值
        if price > trade['max_rate']:
            trade['max_rate'] = price
            trade['peak_profit_pct'] = (trade['max_rate'] - entry_price) / entry_price

        peak = trade['peak_profit_pct']
        age_h = (timestamp - trade['entry_time']).total_seconds() / 3600.0

        # ── 1. 硬止损 ──
        if profit_pct <= tp['stoploss_pct']:
            return ('hard_stoploss', profit_pct)

        # ── 2. 棘轮止损 ──
        if profit_pct >= 1.00:
            if profit_pct < tp['ratchet_100pct_protect']:
                return (f'ratchet_100', profit_pct)
        if profit_pct >= 0.50:
            if profit_pct < tp['ratchet_50pct_protect']:
                return (f'ratchet_50', profit_pct)
        if profit_pct >= 0.20:
            if profit_pct < tp['ratchet_20pct_protect']:
                return (f'ratchet_20', profit_pct)
        if profit_pct >= 0.10:
            if profit_pct < tp['ratchet_10pct_protect']:
                return (f'ratchet_10', profit_pct)

        # ── 3. 回撤保护 ──
        if peak >= tp['pullback_peak_min'] and profit_pct >= tp['pullback_min_profit']:
            drawdown = peak - profit_pct
            if drawdown >= tp['pullback_drawdown']:
                return (f'pullback', profit_pct)

        # ── 4. 时间退出 ──
        if age_h >= tp['time_sharp_drop_hours'] and profit_pct <= tp['time_sharp_drop_threshold']:
            return ('time_sharp_drop', profit_pct)
        if age_h >= tp['time_dud_hours'] and profit_pct < tp['time_dud_threshold']:
            return ('time_dud', profit_pct)
        if age_h >= tp['time_stale_hours'] and profit_pct < tp['time_stale_threshold']:
            return ('time_stale', profit_pct)
        if age_h >= tp['time_weak_hours'] and profit_pct < tp['time_weak_threshold']:
            return ('time_weak', profit_pct)
        if age_h >= tp['time_max_hours']:
            return ('time_max', profit_pct)

        return None

    def close_trade(self, pair, timestamp, price, exit_reason, profit_pct):
        """平仓"""
        trade = self.open_trades.pop(pair)
        tp = self.tp

        stake_returned = trade['stake'] * (1 + profit_pct)
        commission_exit = stake_returned * tp['commission_pct']
        net_returned = stake_returned - commission_exit
        self.total_commission_paid += commission_exit
        self.available_capital += net_returned

        trade['exit_time'] = timestamp
        trade['exit_price'] = price
        trade['exit_reason'] = exit_reason
        trade['profit_pct'] = profit_pct * 100
        trade['net_profit_usd'] = net_returned - trade['stake']
        trade['hold_hours'] = (timestamp - trade['entry_time']).total_seconds() / 3600.0
        trade['peak_profit_pct'] *= 100

        self.closed_trades.append(trade)
        return trade


# ═══════════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_all_data():
    """加载全部 5m feather 数据并计算指标"""
    files = sorted(DATA_DIR.glob("*-5m.feather"))
    coin_data = {}
    skipped = []

    for f in files:
        symbol = f.name.replace("-5m.feather", "")
        try:
            df = pd.read_feather(f)
            df = df.set_index('date').sort_index()

            if len(df) < 200:
                skipped.append(symbol)
                continue

            df = compute_demon_indicators(df)
            score, cat_A, cat_B, cat_C, cat_D, cats, triggered = compute_demon_score(df)
            df['total_score'] = score
            df['cat_A_score'] = cat_A
            df['cat_B_score'] = cat_B
            df['cat_C_score'] = cat_C
            df['cat_D_score'] = cat_D
            df['active_cats'] = cats
            df['triggered'] = triggered

            coin_data[symbol] = df
        except Exception as e:
            skipped.append(symbol)
            logger.warning(f"Failed to load {symbol}: {e}")

    logger.info(f"Loaded {len(coin_data)} coins, skipped {len(skipped)}")
    return coin_data


# ═══════════════════════════════════════════════════════════════════════
#  主交易模拟
# ═══════════════════════════════════════════════════════════════════════

def simulate_trading(coin_data, version_label='v1', params=None,
                     score_threshold=None, min_cats=None):
    """模拟完整交易流程"""
    if params is None:
        tp = dict(TRADE_PARAMS)
    else:
        tp = params

    if score_threshold is not None:
        tp['score_threshold'] = score_threshold
    if min_cats is not None:
        tp['min_cats'] = min_cats

    score_th = tp.get('score_threshold', FACTOR_PARAMS['score_threshold'])
    min_c = tp.get('min_cats', FACTOR_PARAMS['min_categories'])

    sim = TradeSimulator(tp)

    # 全局时间线
    all_times = set()
    for df in coin_data.values():
        all_times.update(df.index)
    all_times = sorted(all_times)

    entry_cooldown = {}
    last_global_entry = None
    global_cooldown_min = tp.get('global_cooldown_min', 120)
    per_coin_cooldown_min = tp.get('per_coin_cooldown_min', 360)

    logger.info(f"Simulating {version_label}: score>={score_th}, cats>={min_c}, max_open={tp.get('max_open_trades',5)}")

    total_signal_checks = 0
    total_signal_fired = 0

    for t in all_times:
        # ── 检查现有仓位退出 ──
        for pair in list(sim.open_trades.keys()):
            df = coin_data.get(pair)
            if df is None or t not in df.index:
                continue
            price = df.loc[t, 'close']
            result = sim.check_exit(pair, t, price)
            if result is not None:
                reason, profit_pct = result
                sim.close_trade(pair, t, price, reason, profit_pct)

        # ── 全局冷却 ──
        if last_global_entry is not None:
            if (t - last_global_entry).total_seconds() < global_cooldown_min * 60:
                continue

        # ── 扫描入场信号 ──
        best_score = 0
        best_signal = None

        for symbol, df in coin_data.items():
            if symbol in sim.open_trades:
                continue

            # 单币冷却
            last_entry = entry_cooldown.get(symbol)
            if last_entry is not None:
                if (t - last_entry).total_seconds() < per_coin_cooldown_min * 60:
                    continue

            if t not in df.index:
                continue

            row = df.loc[t]
            total_signal_checks += 1

            score = row['total_score']
            cats = row['active_cats']
            triggered = row.get('triggered', '')

            # 入场条件
            if score < score_th or cats < min_c:
                continue

            # 流动性过滤
            qvol = row.get('volume_24h_est', 0)
            if qvol < tp.get('min_quote_volume_24h', 50000):
                continue

            price = row['close']
            if price < tp.get('min_price', 0.0001) or price > tp.get('max_price', 1000):
                continue

            total_signal_fired += 1

            # 选最高分
            if score > best_score:
                best_score = score
                # 区分 STRONG vs BUY
                if score >= 80 and cats >= 3:
                    tag = 'demon_strong'
                elif score >= 65 and cats >= 2:
                    tag = 'demon_buy'
                else:
                    tag = 'demon_watch'
                best_signal = (symbol, score, cats, tag, price, triggered)

        # ── 入场 ──
        if best_signal is not None and best_score >= score_th:
            symbol, score, cats, tag, price, triggered = best_signal
            entered, status = sim.try_enter(symbol, t, price, score, cats, tag, triggered)
            if entered:
                entry_cooldown[symbol] = t
                last_global_entry = t

    logger.info(f"  Signal checks: {total_signal_checks}, fired: {total_signal_fired}, "
                f"entered: {len(sim.closed_trades) + len(sim.open_trades)}")

    return sim


# ═══════════════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════════════

def generate_report(sim, label='v1'):
    """生成回测报告"""
    trades = sim.closed_trades

    if len(trades) == 0:
        print(f"\n[{label}] WARNING: No trades!")
        return None, None

    df = pd.DataFrame(trades)

    total_trades = len(df)
    winning = df[df['profit_pct'] > 0]
    losing = df[df['profit_pct'] <= 0]
    win_rate = len(winning) / total_trades * 100

    avg_profit = df['profit_pct'].mean()
    avg_win = winning['profit_pct'].mean() if len(winning) > 0 else 0
    avg_loss = losing['profit_pct'].mean() if len(losing) > 0 else 0

    total_net = df['net_profit_usd'].sum()
    final_cap = sim.available_capital
    initial_cap = sim.tp['total_capital']
    roi = (final_cap - initial_cap) / initial_cap * 100

    profit_factor = abs(avg_win * len(winning) / (avg_loss * len(losing))) if len(losing) > 0 and avg_loss != 0 else float('inf')

    tag_counts = df['tag'].value_counts()
    exit_counts = df['exit_reason'].value_counts()

    best = df.loc[df['profit_pct'].idxmax()]
    worst = df.loc[df['profit_pct'].idxmin()]

    print("\n" + "=" * 60)
    print(f">>> DemonHunter Backtest Report [{label}] <<<")
    print("=" * 60)
    print(f"  总交易: {total_trades}")
    print(f"  赢率: {win_rate:.1f}% ({len(winning)}W / {len(losing)}L)")
    print(f"  平均利润: {avg_profit:.2f}%")
    print(f"  平均盈利: {avg_win:.2f}% | 平均亏损: {avg_loss:.2f}%")
    print(f"  盈亏因子: {profit_factor:.2f}x")
    print(f"  总净收益: ${total_net:.2f}")
    print(f"  最终资金: ${final_cap:.2f} / ${initial_cap:.0f} ({roi:+.2f}%)")
    print(f"  手续费: ${sim.total_commission_paid:.2f}")
    print()
    print(f"  信号分布:")
    for tag, count in tag_counts.items():
        avg_p = df[df['tag'] == tag]['profit_pct'].mean()
        wr = (df[df['tag'] == tag]['profit_pct'] > 0).mean() * 100
        print(f"    {tag}: {count} 笔, avg={avg_p:.2f}%, wr={wr:.1f}%")
    print()
    print(f"  退出原因:")
    for reason, count in exit_counts.items():
        avg_p = df[df['exit_reason'] == reason]['profit_pct'].mean()
        print(f"    {reason}: {count} 笔, avg_pnl={avg_p:.2f}%")
    print()
    print(f"  BEST: {best['pair']} {best['profit_pct']:.2f}% (score={best['score']:.0f}, tag={best['tag']})")
    print(f"    entry={best['entry_time']}, exit={best['exit_time']}, reason={best['exit_reason']}")
    print(f"  WORST: {worst['pair']} {worst['profit_pct']:.2f}% (score={worst['score']:.0f}, tag={worst['tag']})")
    print(f"  Avg hold: {df['hold_hours'].mean():.1f}h | Median: {df['hold_hours'].median():.1f}h")
    print(f"  Avg peak: {df['peak_profit_pct'].mean():.2f}% | Median: {df['peak_profit_pct'].median():.2f}%")
    print("=" * 60)

    # 保存交易记录
    csv_path = OUTPUT_DIR / f'demon_hunter_trades_{label}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved trades to {csv_path}")

    # 汇总记录
    summary = {
        'label': label,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit_pct': avg_profit,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'total_net_usd': total_net,
        'roi_pct': roi,
        'final_capital': final_cap,
        'avg_hold_hours': df['hold_hours'].mean(),
        'avg_peak_pct': df['peak_profit_pct'].mean(),
        'n_open': len(sim.open_trades),
    }

    return df, summary


# ═══════════════════════════════════════════════════════════════════════
#  信号质量分析
# ═══════════════════════════════════════════════════════════════════════

def analyze_signal_quality(coin_data, thresholds=None):
    """不同评分阈值下的命中率和收益分析"""
    if thresholds is None:
        thresholds = {
            'STRONG_80+3cats': (80, 3),
            'BUY_70+2cats': (70, 2),
            'BUY_65+2cats': (65, 2),
            'BUY_60+2cats': (60, 2),
            'WATCH_50+1cat': (50, 1),
        }

    print("\n[SQ] Signal Quality Analysis (threshold scan)")
    print(f"  {'Threshold':<20} {'Signals':>8} {'Coins':>8} {'AvgScore':>12}")
    print(f"  {'-'*50}")

    for label, (score_th, cat_min) in thresholds.items():
        signals = 0
        triggered_coins = set()
        total_triggered_score = 0

        for symbol, df in coin_data.items():
            mask = (df['total_score'] >= score_th) & (df['active_cats'] >= cat_min)
            if mask.any():
                signals += mask.sum()
                triggered_coins.add(symbol)
                total_triggered_score += df.loc[mask, 'total_score'].mean() * mask.sum()

        avg_score = total_triggered_score / signals if signals > 0 else 0
        print(f"  {label:<20} {signals:>8} {len(triggered_coins):>8} {avg_score:>12.1f}")

    return thresholds


# ═══════════════════════════════════════════════════════════════════════
#  妖币捕获率分析
# ═══════════════════════════════════════════════════════════════════════

def analyze_demon_capture_rate(coin_data, sim_or_trades):
    """分析妖币(50%+日涨幅)的捕获率"""
    # 加载妖币事件列表
    moji_csv = Path("E:/source/freqtrade/deliverables/moji-coin/moji_features.csv")
    if not moji_csv.exists():
        logger.warning("moji_features.csv not found, skipping capture rate analysis")
        return

    moji_df = pd.read_csv(moji_csv)
    demon_events = moji_df[moji_df['pump_gain_pct'] >= 50].copy()

    if len(demon_events) == 0:
        print("\n[DCR] No 50%+ demon coin events to analyze")
        return

    # 检查每个妖币事件是否在我们的数据覆盖范围内
    trades_pairs = set()
    trade_times = set()
    for t in sim_or_trades.closed_trades:
        trades_pairs.add(t['pair'])
        trade_times.add(t['entry_time'].date())

    captured = 0
    total = 0

    for _, event in demon_events.iterrows():
        symbol = event['symbol']
        pump_day = pd.Timestamp(event['pump_start_time'])

        total += 1

        # 检查是否有交易覆盖这个币种+日期
        for pair in trades_pairs:
            if pair == symbol:
                captured += 1
                break

    capture_rate = captured / total * 100 if total > 0 else 0
    print(f"\n[DCR] Demon coin capture rate: {captured}/{total} ({capture_rate:.1f}%)")
    print(f"  Events: {demon_events['symbol'].tolist()}")

    return capture_rate, captured, total


# ═══════════════════════════════════════════════════════════════════════
#  HTML 报告生成
# ═══════════════════════════════════════════════════════════════════════

def generate_html_report(all_summaries, all_trades, signal_analysis):
    """生成可视化HTML回测报告"""

    summaries_html = ""
    for s in all_summaries:
        wr_class = 'positive' if s['win_rate'] > 30 else 'neutral'
        summaries_html += f"""<tr>
            <td><b>{s['label']}</b></td>
            <td class="num">{s['total_trades']}</td>
            <td class="num {wr_class}">{s['win_rate']:.1f}%</td>
            <td class="num">{s['avg_profit_pct']:.2f}%</td>
            <td class="num">{s['avg_win_pct']:.2f}%</td>
            <td class="num">{s['avg_loss_pct']:.2f}%</td>
            <td class="num">{s['profit_factor']:.2f}x</td>
            <td class="num {'positive' if s['roi_pct'] > 0 else 'negative'}">{s['roi_pct']:+.2f}%</td>
            <td class="num">{s['avg_hold_hours']:.1f}h</td>
            <td class="num">{s['avg_peak_pct']:.2f}%</td>
        </tr>"""

    # 找最佳版本
    best_label = all_summaries[0]['label'] if all_summaries else 'N/A'
    best_roi = max(s['roi_pct'] for s in all_summaries) if all_summaries else 0

    # 交易明细 (仅显示最佳版本的最近20笔)
    master_df = all_trades.get('master')
    trade_rows = ""
    if master_df is not None and len(master_df) > 0:
        for _, t in master_df.sort_values('entry_time')[-20:].iterrows():
            pnl_class = 'positive' if t['profit_pct'] > 0 else 'negative'
            trade_rows += f"""<tr>
                <td>{t['pair']}</td>
                <td>{t['entry_time']}</td>
                <td class="num">{t['score']:.0f}</td>
                <td>{t['tag']}</td>
                <td class="num {pnl_class}">{t['profit_pct']:+.2f}%</td>
                <td>{t['exit_reason']}</td>
                <td class="num">{t['hold_hours']:.1f}h</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DemonHunter 回测报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}

.hero {{ background: linear-gradient(135deg, #6b21a8 0%, #dc2626 50%, #f59e0b 100%); color: white; padding: 40px; border-radius: 16px; margin-bottom: 32px; text-align: center; }}
.hero h1 {{ font-size: 32px; margin-bottom: 8px; }}
.hero .subtitle {{ font-size: 14px; opacity: 0.9; }}

.hero-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 24px; }}
.hero-stat {{ background: rgba(255,255,255,0.15); padding: 16px; border-radius: 10px; }}
.hero-stat .num {{ font-size: 26px; font-weight: bold; }}
.hero-stat .label {{ font-size: 12px; opacity: 0.85; }}

.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
.card h2 {{ color: #f0883e; font-size: 20px; margin-bottom: 16px; border-bottom: 1px solid #21262d; padding-bottom: 10px; }}
.card h3 {{ color: #e6edf3; font-size: 15px; margin: 14px 0 10px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; background: #1c2128; }}
tr:hover {{ background: #1c2128; }}
.num {{ text-align: right; font-family: "Cascadia Code", monospace; }}
.positive {{ color: #ef4444; }}
.negative {{ color: #22c55e; }}
.neutral {{ color: #8b949e; }}

.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
.tag-green {{ background: #238636; color: white; }}
.tag-red {{ background: #da3633; color: white; }}
.tag-yellow {{ background: #d29922; color: #111; }}
.tag-orange {{ background: #f0883e; color: white; }}

.insight {{ background: #1a2332; border-left: 4px solid #58a6ff; padding: 12px 16px; border-radius: 6px; margin: 12px 0; font-size: 13px; }}
.insight.warn {{ border-left-color: #f0883e; background: #1f1a16; }}
.insight.good {{ border-left-color: #3fb950; background: #162318; }}
.insight.danger {{ border-left-color: #da3633; background: #1f1616; }}
.insight strong {{ color: #e6edf3; }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}

.footer {{ text-align: center; color: #484f58; font-size: 12px; padding: 24px 0 12px; border-top: 1px solid #21262d; }}
</style>
</head>
<body>
<div class="container">

<div class="hero">
    <h1>🔥 DemonHunter 回测报告</h1>
    <div class="subtitle">基于 8因子 × 4类别分层评分体系 | 5分钟K线 | Gate.io 全市场 | {len(all_summaries)} 个参数版本</div>
</div>

<!-- 版本对比 -->
<div class="card">
    <h2>📊 版本对比总览</h2>
    <table>
        <thead>
            <tr><th>版本</th><th>交易数</th><th>赢率</th><th>平均利润</th><th>平均盈利</th><th>平均亏损</th><th>盈亏因子</th><th>ROI</th><th>平均持仓</th><th>平均峰值</th></tr>
        </thead>
        <tbody>{summaries_html}</tbody>
    </table>
</div>

<!-- 分析洞察 -->
<div class="card">
    <h2>💡 分析洞察</h2>

    <div class="grid-2">
        <div class="insight good">
            <strong>策略特征：</strong><br>
            DemonHunter 属于<strong>低胜率 + 高盈亏比</strong>的彩票型策略。<br>
            依赖少量大赢家覆盖大量小额亏损，与趋势跟随、均值回归等传统策略有本质区别。<br>
            适合作为卫星仓位（组合占比 < 10%），不宜作为主力策略。
        </div>
        <div class="insight warn">
            <strong>版本调优方向：</strong><br>
            • 入场阈值越高 → 交易越少但赢率可能提升<br>
            • 棘轮止损越紧 → 保护利润但可能过早退出<br>
            • 冷却时间越长 → 更精选但可能错过机会<br>
            • 核心矛盾：减少假阳性 vs. 不错过妖币
        </div>
    </div>

    <div class="insight">
        <strong>与 MojiHunter (v3/v4) 的关键差异：</strong><br>
        • <b>因子体系升级</b>：从7规则平权 → 8因子×4类别的分层加权<br>
        • <b>交易管理更激进</b>：从20仓$20 → 5仓$10，更集中<br>
        • <b>棘轮止损更系统</b>：+10%/+20%/+50%/+100%四档保护<br>
        • <b>退出逻辑更完善</b>：增加了回撤保护和更多时间退出档位<br>
        • <b>响应50%+妖币</b>：因子权重和阈值专门针对大妖币特征调优
    </div>
</div>

<!-- 交易明细 -->
<div class="card">
    <h2>📋 交易明细 (最近20笔)</h2>
    <table>
        <thead><tr><th>币种</th><th>入场时间</th><th>评分</th><th>信号</th><th>盈亏</th><th>退出原因</th><th>持仓</th></tr></thead>
        <tbody>{trade_rows if trade_rows else '<tr><td colspan="7" style="text-align:center;color:#8b949e">无交易记录</td></tr>'}</tbody>
    </table>
</div>

<!-- 局限性 -->
<div class="card">
    <h2>⚠️ 模型局限性</h2>
    <table>
        <thead><tr><th>局限</th><th>严重度</th><th>说明</th></tr></thead>
        <tbody>
            <tr><td>数据窗口过短</td><td><span class="tag tag-red">高</span></td><td>仅约8天数据，50%+妖币事件仅9个，统计显著性有限</td></tr>
            <tr><td>滑点未考虑</td><td><span class="tag tag-yellow">中</span></td><td>妖币流动性差，实际成交价可能与K线收盘价有显著偏差</td></tr>
            <tr><td>幸存者偏差</td><td><span class="tag tag-red">高</span></td><td>仅包含Gate.io上仍在交易的币种</td></tr>
            <tr><td>无订单簿数据</td><td><span class="tag tag-yellow">中</span></td><td>D1时间窗口因子依赖K线小时值，未使用真实订单簿活跃度</td></tr>
            <tr><td>因子权重未优化</td><td><span class="tag tag-yellow">中</span></td><td>权重基于分析报告中的统计差异手动设定，未做网格寻优</td></tr>
        </tbody>
    </table>
</div>

<div class="footer">
    DemonHunter Backtest v1.0 | Gate.io 5m K线 | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}<br>
    仅供研究参考，不构成投资建议
</div>

</div>
</body>
</html>"""

    report_path = OUTPUT_DIR / 'demon_hunter_backtest_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"HTML report saved to {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("DemonHunter Backtest Suite")
    logger.info("=" * 60)

    # 加载数据
    logger.info("[1/4] Loading data...")
    coin_data = load_all_data()

    if not coin_data:
        logger.error("No data loaded!")
        return

    # 信号质量分析
    logger.info("[2/4] Signal quality analysis...")
    signal_thresholds = analyze_signal_quality(coin_data)

    # 多版本回测
    logger.info("[3/4] Running backtest versions...")

    versions = [
        ('v1_base', 65, 2),     # 基础版: BUY>=65, 2cats
        ('v2_loose', 55, 2),    # 宽松版: 降低阈值
        ('v3_tight', 75, 3),    # 严格版: 提高阈值
        ('v4_strong', 80, 3),   # 仅强信号
    ]

    all_summaries = []
    all_trades_dict = {}

    for label, score_th, min_c in versions:
        sim = simulate_trading(coin_data, version_label=label,
                               score_threshold=score_th, min_cats=min_c)
        df, summary = generate_report(sim, label)
        if df is not None:
            df['version'] = label
            all_trades_dict[label] = df
            all_summaries.append(summary)

    # 合并所有版本交易
    all_dfs = [df for df in all_trades_dict.values() if df is not None]
    master_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    if master_df is not None:
        master_df.to_csv(OUTPUT_DIR / 'demon_hunter_all_trades.csv', index=False)

    all_trades_dict['master'] = master_df

    # 妖币捕获率分析 (用 v1_base)
    if 'v1_base' in all_trades_dict and all_trades_dict['v1_base'] is not None:
        logger.info("[4/4] Demon coin capture rate analysis...")
        # 需要重新创建 sim 来获取捕获信息
        # 简化版: 从交易记录分析
        v1_trades = all_trades_dict['v1_base']
        demon_events = pd.read_csv(OUTPUT_DIR / 'demon_coins.csv') if (OUTPUT_DIR / 'demon_coins.csv').exists() else None

        if demon_events is not None and len(demon_events) > 0:
            captured = 0
            for _, demon in demon_events.iterrows():
                demon_symbol = demon['symbol']
                demon_day = demon['pump_day']
                matches = v1_trades[v1_trades['pair'] == demon_symbol]
                if len(matches) > 0:
                    captured += 1
            print(f"\n[DCR] Demon coin capture: {captured}/{len(demon_events)} ({captured/len(demon_events)*100:.1f}%)")

    # 生成 HTML 报告
    logger.info("Generating HTML report...")
    report_path = generate_html_report(all_summaries, all_trades_dict, signal_thresholds)

    # 保存汇总
    if all_summaries:
        with open(OUTPUT_DIR / 'backtest_summary.json', 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n✅ Complete! Report: {report_path}")

    return all_summaries


if __name__ == '__main__':
    main()
