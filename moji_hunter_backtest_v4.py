"""
MojiCoinHunter v4 多框架独立回测
=================================
基于多时间框架分析结论:
  - 5m 初筛层 (最强鉴别力, pre_change 绝对差 3.87%)
  - 30m 确认层 (优质过滤, sma_slope 绝对差 2.39%)
  - 1h 安全层 (极低假阳性, sma_slope 绝对差 1.57%)
  - 三层全通过才入场

v1-v3 对比:
  v1 (单层5m, score>=60, 7规则): 198笔, 17.7%赢率, -96.6% ROI
  v3 (单层5m, score>=70, 5规则):  93笔, 21.5%赢率, -30.2% ROI
  v4 (三层漏斗): ?笔, ?%赢率, ? ROI

核心假设: 三层合一把控制组穿透率压到接近 0%
  控制组在各框架的漂移: 5m +0.04%, 30m -0.08%, 1h -0.02%
  → 同时跨三个框架正向蓄势的概率极低
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
#  v4 多框架参数 (来自多TF分析)
# ═══════════════════════════════════════════════════════════════════════

LAYER1_5M = {
    'pre_change_pct': 2.0,     # moji 3.91%, ctrl 0.04%
    'sma_slope_pct': 1.2,      # moji 2.59%, ctrl -0.04%
    'vol_concentration': 0.03,  # moji 0.057, ctrl 0.024
    'volume_ratio': 1.5,       # moji 1.98, ctrl 1.14
    'lookback_bars': 72,        # 6h
}

LAYER2_30M = {
    'pre_change_pct': 1.0,     # moji 2.31%, ctrl -0.08%
    'sma_slope_pct': 1.0,      # moji 2.31%, ctrl -0.08%
    'lookback_bars': 12,        # 6h
}

LAYER3_1H = {
    'pre_change_pct': 0.5,     # moji 1.55%, ctrl -0.02%
    'sma_slope_pct': 0.5,      # moji 1.55%, ctrl -0.02%
    'lookback_bars': 6,         # 6h
}

# 反向过滤: 妖币预泵振幅 < 控制组, 说明紧致蓄势
ANTI_RANGE_FILTER = {
    'max_range_pct_5m': 15.0,   # 振幅超过15%反而不好 (控制组平均12%)
}

TRADE_PARAMS = {
    'stake_per_trade': 1.0,
    'max_open_trades': 20,
    'total_capital': 20.0,
    'stoploss_pct': -0.08,
    'commission_pct': 0.002,
    'max_hold_hours': 72,
    'dud_early_hours': 8,
    'dud_half_day_hours': 12,
    'dud_1day_hours': 24,
    'dud_2day_hours': 48,
    'ratchet_10pct_protect': 0.0,
    'ratchet_20pct_protect': 0.10,
    'ratchet_50pct_protect': 0.25,
    'ratchet_100pct_protect': 0.50,
    'pullback_drawdown': 0.12,
    'pullback_min_profit': 0.08,
    'pullback_peak_min': 0.15,
    'global_cooldown_min': 120,
    'min_quote_volume_24h': 30000,
    'require_all_layers': True,  # v4: 三层全通过
}


# ═══════════════════════════════════════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════════════════════════════════════

def _safe_div(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_indicators(df, lookback_bars):
    """计算标准指标集"""
    df = df.copy()
    lb = lookback_bars

    # SMA
    sma_n = max(3, lb // 6)
    df['sma'] = df['close'].rolling(sma_n, min_periods=max(2, sma_n//2)).mean()

    # pre_change_pct: (close_now - close_N_bars_ago) / close_N_bars_ago * 100
    close_lb = df['close'].shift(lb)
    df['pre_change_pct'] = _safe_div(df['close'] - close_lb, close_lb) * 100

    # sma_slope_pct
    sma_now = df['sma']
    sma_lb = df['sma'].shift(lb)
    df['sma_slope_pct'] = _safe_div(sma_now - sma_lb, sma_lb) * 100

    # volume_ratio: 后半段 / 前半段
    half = max(2, lb // 2)
    vol_recent = df['volume'].rolling(half, min_periods=max(1, half//2)).mean()
    vol_early = df['volume'].shift(half).rolling(half, min_periods=max(1, half//2)).mean()
    df['volume_ratio'] = _safe_div(vol_recent, vol_early)

    # vol_concentration: 最后2根bar占总量的比例
    if lb >= 4:
        vol_last2 = df['volume'] + df['volume'].shift(1)
        vol_total = df['volume'].rolling(lb, min_periods=half).sum()
        df['vol_concentration'] = _safe_div(vol_last2, vol_total)
    else:
        df['vol_concentration'] = 0

    # range_pct (反向过滤器用)
    high_lb = df['high'].rolling(lb, min_periods=half).max()
    low_lb = df['low'].rolling(lb, min_periods=half).min()
    df['range_pct'] = _safe_div(high_lb - low_lb, low_lb) * 100

    return df


# ═══════════════════════════════════════════════════════════════════════
#  三层漏斗信号
# ═══════════════════════════════════════════════════════════════════════

def check_layer1_5m(row):
    """5m 初筛层"""
    return (
        row['pre_change_pct'] >= LAYER1_5M['pre_change_pct'] and
        row['sma_slope_pct'] >= LAYER1_5M['sma_slope_pct'] and
        row['vol_concentration'] >= LAYER1_5M['vol_concentration'] and
        row['volume_ratio'] >= LAYER1_5M['volume_ratio'] and
        row['range_pct'] <= ANTI_RANGE_FILTER['max_range_pct_5m']  # 反向: 振幅不能太大
    )


def check_layer2_30m(row):
    """30m 确认层"""
    return (
        row['pre_change_pct'] >= LAYER2_30M['pre_change_pct'] and
        row['sma_slope_pct'] >= LAYER2_30M['sma_slope_pct']
    )


def check_layer3_1h(row):
    """1h 安全层"""
    return (
        row['pre_change_pct'] >= LAYER3_1H['pre_change_pct'] and
        row['sma_slope_pct'] >= LAYER3_1H['sma_slope_pct']
    )


def evaluate_signal(row_5m, row_30m, row_1h, require_all=True):
    """评估多层信号, 返回 (passed, layer_mask)"""
    l1 = check_layer1_5m(row_5m)
    l2 = check_layer2_30m(row_30m)
    l3 = check_layer3_1h(row_1h)

    mask = (l1, l2, l3)

    if require_all:
        passed = l1 and l2 and l3
    else:
        passed = (l1 and l2) or (l1 and l3)  # at least 5m + 1 confirmation

    # 计算综合得分用于排序
    score = 0
    if l1:
        score += (
            (row_5m['pre_change_pct'] - LAYER1_5M['pre_change_pct']) * 15 +
            (row_5m['sma_slope_pct'] - LAYER1_5M['sma_slope_pct']) * 15 +
            (row_5m['vol_concentration'] - LAYER1_5M['vol_concentration']) * 200 +
            (row_5m['volume_ratio'] - LAYER1_5M['volume_ratio']) * 5
        )
    if l2:
        score += (
            (row_30m['pre_change_pct'] - LAYER2_30M['pre_change_pct']) * 25 +
            (row_30m['sma_slope_pct'] - LAYER2_30M['sma_slope_pct']) * 25
        )
    if l3:
        score += (
            (row_1h['pre_change_pct'] - LAYER3_1H['pre_change_pct']) * 30 +
            (row_1h['sma_slope_pct'] - LAYER3_1H['sma_slope_pct']) * 30
        )

    return passed, mask, score


# ═══════════════════════════════════════════════════════════════════════
#  交易管理 (复用)
# ═══════════════════════════════════════════════════════════════════════

class TradeManager:
    def __init__(self, params=TRADE_PARAMS):
        self.tp = params
        self.open = {}
        self.closed = []
        self.capital = params['total_capital']
        self.commission = 0.0

    def try_enter(self, symbol, ts, price, score, layers):
        if len(self.open) >= self.tp['max_open_trades']:
            return False
        if self.capital < self.tp['stake_per_trade']:
            return False

        stake = self.tp['stake_per_trade']
        comm = stake * self.tp['commission_pct']
        self.capital -= (stake + comm)
        self.commission += comm

        self.open[symbol] = {
            'symbol': symbol, 'entry_time': ts, 'entry_price': price,
            'stake': stake, 'amount': (stake - comm) / price,
            'score': score, 'layers': layers, 'max_rate': price,
            'peak_profit_pct': 0.0, 'signal_tag': f"v4_l{''.join(str(int(l)) for l in layers)}",
        }
        return True

    def check_exit(self, symbol, ts, price):
        t = self.open.get(symbol)
        if t is None:
            return None
        tp = self.tp
        ep = t['entry_price']
        pp = (price - ep) / ep

        if price > t['max_rate']:
            t['max_rate'] = price
            t['peak_profit_pct'] = (t['max_rate'] - ep) / ep

        age_h = (ts - t['entry_time']).total_seconds() / 3600.0

        if pp <= tp['stoploss_pct']:
            return ('stoploss', pp)
        if pp >= 1.00 and pp < tp['ratchet_100pct_protect']:
            return ('ratchet_100', pp)
        if pp >= 0.50 and pp < tp['ratchet_50pct_protect']:
            return ('ratchet_50', pp)
        if pp >= 0.20 and pp < tp['ratchet_20pct_protect']:
            return ('ratchet_20', pp)
        if pp >= 0.10 and pp < tp['ratchet_10pct_protect']:
            return ('ratchet_10', pp)
        if age_h >= tp['dud_early_hours'] and pp < -0.05:
            return ('dud_early', pp)
        if age_h >= tp['dud_half_day_hours'] and pp < 0:
            return ('stale_half_day', pp)
        if age_h >= tp['dud_1day_hours'] and pp < 0.01:
            return ('stale_dud', pp)
        if age_h >= tp['dud_2day_hours'] and pp < 0.05:
            return ('medium_2d', pp)
        if age_h >= tp['max_hold_hours']:
            return ('max_hold_3d', pp)
        if t['peak_profit_pct'] >= tp['pullback_peak_min'] and pp >= tp['pullback_min_profit']:
            if t['peak_profit_pct'] - pp >= tp['pullback_drawdown']:
                return ('pullback_exit', pp)
        return None

    def close_trade(self, symbol, ts, price, reason, profit_pct):
        t = self.open.pop(symbol)
        tp = self.tp
        returned = t['stake'] * (1 + profit_pct)
        exit_comm = returned * tp['commission_pct']
        net = returned - exit_comm
        self.commission += exit_comm
        self.capital += net

        t.update({
            'exit_time': ts, 'exit_price': price, 'exit_reason': reason,
            'profit_pct': profit_pct * 100, 'net_profit_usd': net - t['stake'],
            'hold_hours': (ts - t['entry_time']).total_seconds() / 3600.0,
            'peak_profit_pct': t['peak_profit_pct'] * 100,
        })
        self.closed.append(t)
        return t


# ═══════════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """加载1m数据, 重采样到5m/30m/1h, 计算各TF指标"""
    logger.info("Loading 1m data...")
    files_1m = sorted(DATA_DIR.glob("*-1m.feather"))

    coin_data = {}
    for f in files_1m:
        symbol = f.name.replace("-1m.feather", "")
        df = pd.read_feather(f).set_index('date').sort_index()

        # 重采样到各TF
        tf_data = {}
        for tf_name, rule in [('5m', '5min'), ('30m', '30min'), ('1h', '1h')]:
            resampled = df.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna()

            lb = {'5m': 72, '30m': 12, '1h': 6}[tf_name]
            tf_data[tf_name] = compute_indicators(resampled, lb)

        coin_data[symbol] = tf_data

    logger.info(f"Prepared {len(coin_data)} coins with multi-TF indicators")
    return coin_data


# ═══════════════════════════════════════════════════════════════════════
#  回测模拟
# ═══════════════════════════════════════════════════════════════════════

def simulate_trading(coin_data, require_all=True):
    mgr = TradeManager()
    tp = mgr.tp

    # Build global 5m timeline (primary scan frequency)
    all_5m_times = set()
    for symbol, tf_data in coin_data.items():
        all_5m_times.update(tf_data['5m'].index)
    all_5m_times = sorted(all_5m_times)

    pair_cooldown = {}
    last_global_entry = None

    logger.info(f"Simulating trading on {len(all_5m_times)} 5m candles...")

    for ts in all_5m_times:
        # ── 检查退出 ──
        for pair in list(mgr.open.keys()):
            if pair not in coin_data:
                continue
            tf_data = coin_data[pair]
            if ts not in tf_data['5m'].index:
                continue
            price = tf_data['5m'].loc[ts, 'close']
            result = mgr.check_exit(pair, ts, price)
            if result:
                reason, pp = result
                mgr.close_trade(pair, ts, price, reason, pp)

        # ── 全局冷却 ──
        if last_global_entry and (ts - last_global_entry).total_seconds() < tp['global_cooldown_min'] * 60:
            continue

        # ── 扫描信号 ──
        candidates = []
        for symbol, tf_data in coin_data.items():
            if symbol in mgr.open:
                continue
            # 冷却
            le = pair_cooldown.get(symbol)
            if le and (ts - le).total_seconds() < tp['global_cooldown_min'] * 60:
                continue
            # 需三个TF都有该时刻数据
            if (ts not in tf_data['5m'].index or
                ts not in tf_data['30m'].index or
                ts not in tf_data['1h'].index):
                continue

            r5 = tf_data['5m'].loc[ts]
            r30 = tf_data['30m'].loc[ts]
            r1h = tf_data['1h'].loc[ts]

            passed, mask, score = evaluate_signal(r5, r30, r1h, require_all)
            if not passed:
                continue

            price = r5['close']
            if price < 0.0001 or price > 500:
                continue

            # 流动性
            qvol = r5.get('volume', 0) * price * 288  # 粗估24h
            if qvol < tp['min_quote_volume_24h']:
                continue

            candidates.append((symbol, score, price, mask))

        # 选最高分入场
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            symbol, score, price, mask = candidates[0]
            tag = f"v4_l{''.join(str(int(l)) for l in mask)}"
            entered = mgr.try_enter(symbol, ts, price, score, mask)
            if entered:
                pair_cooldown[symbol] = ts
                last_global_entry = ts
                logger.debug(f"  🚀 {symbol} score={score:.0f} layers={mask} @ {ts}")

    return mgr


# ═══════════════════════════════════════════════════════════════════════
#  报告 & 对比
# ═══════════════════════════════════════════════════════════════════════

def print_report(mgr, label="v4"):
    trades = mgr.closed
    if len(trades) == 0:
        print(f"\n⚠️ {label}: 无交易 —— 过滤器太严格或信号未触发")
        return None

    df = pd.DataFrame(trades)
    n = len(df)
    win = df[df['profit_pct'] > 0]
    lose = df[df['profit_pct'] <= 0]
    wr = len(win) / n * 100
    avg_p = df['profit_pct'].mean()
    avg_w = win['profit_pct'].mean() if len(win) > 0 else 0
    avg_l = lose['profit_pct'].mean() if len(lose) > 0 else 0
    final_cap = mgr.capital
    roi = (final_cap - TRADE_PARAMS['total_capital']) / TRADE_PARAMS['total_capital'] * 100

    best = df.loc[df['profit_pct'].idxmax()]
    worst = df.loc[df['profit_pct'].idxmin()]

    print(f"\n{'='*60}")
    print(f"🔥 MojiCoinHunter {label} 回测结果")
    print(f"{'='*60}")
    print(f"  总交易:  {n}")
    print(f"  赢率:    {wr:.1f}% ({len(win)}W/{len(lose)}L)")
    print(f"  盈亏比:  {abs(avg_w/avg_l):.2f}x (avg_win={avg_w:.2f}% / avg_loss={avg_l:.2f}%)")
    print(f"  平均利润: {avg_p:.2f}%")
    print(f"  总净收益: ${df['net_profit_usd'].sum():.2f}")
    print(f"  最终资金: ${final_cap:.2f} (初始 $20)")
    print(f"  ROI:      {roi:.2f}%")
    print(f"  手续费:   ${mgr.commission:.2f}")
    print(f"  平均持仓: {df['hold_hours'].mean():.1f}h")
    print(f"  平均峰值: {df['peak_profit_pct'].mean():.2f}%")

    # 退出原因分布
    exit_dist = df['exit_reason'].value_counts()
    print(f"\n  退出分布:")
    for reason, count in exit_dist.items():
        sub = df[df['exit_reason'] == reason]
        print(f"    {reason:20s}: {count:3d} ({count/n*100:4.1f}%)  avg={sub['profit_pct'].mean():.2f}%")

    print(f"\n  🏆 最佳: {best['symbol']} +{best['profit_pct']:.2f}%")
    print(f"  💀 最差: {worst['symbol']} {worst['profit_pct']:.2f}%")

    df.to_csv(OUTPUT_DIR / f'moji_hunter_{label}_trades.csv', index=False)
    return df


def analyze_layers(mgr):
    """分析各层过滤效果"""
    if len(mgr.closed) == 0:
        return

    df = pd.DataFrame(mgr.closed)
    # 提取层信息
    layers = [l for l in df['layers']]
    l1 = sum(1 for l in layers if l[0]) / len(layers) * 100
    l2 = sum(1 for l in layers if l[1]) / len(layers) * 100
    l3 = sum(1 for l in layers if l[2]) / len(layers) * 100

    print(f"\n📊 层过滤分析 (实际入场条件)")
    print(f"  Layer1 (5m):  {l1:.0f}% 通过")
    print(f"  Layer2 (30m): {l2:.0f}% 通过")
    print(f"  Layer3 (1h):  {l3:.0f}% 通过")


# ═══════════════════════════════════════════════════════════════════════
#  对比运行
# ═══════════════════════════════════════════════════════════════════════

def main():
    coin_data = load_and_prepare()

    # ── v4: 三层全部通过 ──
    logger.info("=== V4: 三层漏斗 (5m + 30m + 1h) ===")
    mgr_v4 = simulate_trading(coin_data, require_all=True)
    df_v4 = print_report(mgr_v4, "v4")

    # ── v4b: 5m + 1 confirmation ──
    logger.info("\n=== V4b: 5m + 至少一个确认 (30m 或 1h) ===")
    # Reset params for fair comparison
    coin_data2 = load_and_prepare()
    mgr_v4b = simulate_trading(coin_data2, require_all=False)
    df_v4b = print_report(mgr_v4b, "v4b")

    # ── v4c: 仅 5m ──
    logger.info("\n=== V4c: 仅 5m (无确认) ===")
    coin_data3 = load_and_prepare()
    mgr_v4c = simulate_trading(coin_data3, require_all=False)

    # v4c: bypass layer2/3 checks
    # Re-sim with layers force-passed
    # Actually let me just create a v4c simulation
    class MgrV4C(TradeManager):
        pass  # same

    mgr_v4c = TradeManager()
    tp = mgr_v4c.tp
    all_5m = sorted(set().union(*[td['5m'].index for td in coin_data3.values()]))
    pair_cd = {}
    last_ge = None

    for ts in all_5m:
        for pair in list(mgr_v4c.open.keys()):
            if pair not in coin_data3:
                continue
            if ts not in coin_data3[pair]['5m'].index:
                continue
            px = coin_data3[pair]['5m'].loc[ts, 'close']
            result = mgr_v4c.check_exit(pair, ts, px)
            if result:
                mgr_v4c.close_trade(pair, ts, px, *result)

        if last_ge and (ts - last_ge).total_seconds() < tp['global_cooldown_min'] * 60:
            continue

        candidates = []
        for sym, td in coin_data3.items():
            if sym in mgr_v4c.open:
                continue
            le = pair_cd.get(sym)
            if le and (ts - le).total_seconds() < tp['global_cooldown_min'] * 60:
                continue
            if ts not in td['5m'].index:
                continue

            r = td['5m'].loc[ts]
            if not check_layer1_5m(r):
                continue
            px = r['close']
            if px < 0.0001 or px > 500:
                continue
            qv = r.get('volume', 0) * px * 288
            if qv < tp['min_quote_volume_24h']:
                continue

            score = (
                (r['pre_change_pct'] - LAYER1_5M['pre_change_pct']) * 15 +
                (r['sma_slope_pct'] - LAYER1_5M['sma_slope_pct']) * 15 +
                (r['vol_concentration'] - LAYER1_5M['vol_concentration']) * 200 +
                (r['volume_ratio'] - LAYER1_5M['volume_ratio']) * 5
            )
            candidates.append((sym, score, px))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            sym, score, px = candidates[0]
            entered = mgr_v4c.try_enter(sym, ts, px, score, (True, False, False))
            if entered:
                pair_cd[sym] = ts
                last_ge = ts

    df_v4c = print_report(mgr_v4c, "v4c")

    # ── 汇总对比 ──
    print(f"\n\n{'='*60}")
    print(f"📊 版本对比汇总")
    print(f"{'='*60}")
    print(f"  {'版本':<8s} {'交易数':>6s} {'赢率':>6s} {'盈亏比':>6s} {'ROI':>8s} {'最终资金':>10s}")
    print(f"  {'-'*45}")
    for label, df in [('v1', None), ('v3', None), ('v4', df_v4), ('v4b', df_v4b), ('v4c', df_v4c)]:
        if label in ('v1', 'v3'):
            # historical results
            if label == 'v1':
                n, wr, rr, roi_str, cap = 198, "17.7%", "2.59x", "-96.66%", "$0.67"
            else:
                n, wr, rr, roi_str, cap = 93, "21.5%", "2.86x", "-30.20%", "$13.96"
        elif df is not None:
            n = len(df)
            wr = f"{len(df[df['profit_pct']>0])/n*100:.1f}%"
            w_df = df[df['profit_pct']>0]
            l_df = df[df['profit_pct']<=0]
            avg_w = w_df['profit_pct'].mean() if len(w_df)>0 else 0
            avg_l = l_df['profit_pct'].mean() if len(l_df)>0 else 0
            rr = f"{abs(avg_w/avg_l):.2f}x"
            total_net = df['net_profit_usd'].sum()
            cap = f"${20 + total_net:.2f}"
            roi_str = f"{total_net/20*100:.2f}%"
        else:
            continue

        print(f"  {label:<8s} {str(n):>6s} {wr:>6s} {rr:>6s} {roi_str:>8s} {cap:>10s}")

    print(f"\n⚠️ 8天数据太短, 彩票策略需3-6个月回测才能可靠评估")


if __name__ == '__main__':
    main()
