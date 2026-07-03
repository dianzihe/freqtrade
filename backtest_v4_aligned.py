"""
妖币猎手 v4 独立回测 (完全对齐策略文件逻辑)
=============================================
实现: breakout + 量异动 + 安静蓄势 + 评分分级 + ROI阶梯 + 棘轮止损
"""

import logging, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

DATA_DIR = Path("E:/source/freqtrade/user_data/data/gate")
OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/moji-coin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── v4 参数 (对齐 妖币猎手.py 默认值) ──
VP = {
    # 闸门
    'score_threshold': 50, 'min_rules': 2,
    # R1: 量比绝对
    'vol_ratio': 2.0, 'vol_ratio_w': 30,
    # R1b: 量比百分位
    'vol_rank': 0.85, 'vol_rank_w': 25,
    # R2: SMA斜率
    'sma_slope_pct': 1.5, 'sma_slope_w': 20,
    # R3: 累计涨幅
    'cum_change_pct': 3.0, 'cum_change_w': 15,
    # R4: 蓄势压缩 (earlier range < 阈值 → 安静)
    'quiet_range_pct': 5.0, 'quiet_w': 20,
    # R5: 突破
    'breakout_w': 25,
    # RSI / 流动性
    'rsi_min': 38, 'min_quote_volume': 50000,
    # 信号分级
    'strong_score': 90, 'strong_rules': 4,
    'medium_score': 70,
    # 价格
    'min_price': 0.0001, 'max_price': 500.0,
}

TP = {
    'stake': 1.0, 'max_trades': 20, 'total_capital': 20.0,
    'stoploss': -0.08, 'commission': 0.002,
    'ratchet': {1.0: 0.50, 0.50: 0.25, 0.20: 0.10, 0.10: 0.0},
    # ROI 阶梯 (对齐策略 minimal_roi)
    'roi_steps': [(0, 0.50), (30, 0.20), (120, 0.10), (360, 0.05)],
    # 时间止退出
    'dud_early': 6, 'dud_threshold': -0.05,
    'stale_half_day': 12, 'stale_dud': 24, 'medium_2d': 48, 'max_hold': 72,
    'pullback_dd': 0.12, 'pullback_min': 0.08, 'pullback_peak': 0.15,
    # 冷却
    'per_pair_cd': 360, 'global_cd': 15,
    'max_daily': 40,
}


def compute_indicators(df):
    df = df.copy()
    # SMA
    df['sma12'] = df['close'].rolling(12, min_periods=6).mean()

    # R1: vol_ratio
    vr = df['volume'].rolling(12, min_periods=3).mean()
    ve = df['volume'].shift(12).rolling(60, min_periods=24).mean()
    df['vol_ratio'] = (vr / ve.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)

    # R1b: vol_ratio_rank (自参照24h分位)
    df['vol_ratio_rank'] = df['vol_ratio'].rolling(288, min_periods=72).rank(pct=True).fillna(0)

    # R2: sma_slope
    s6 = df['sma12'].shift(72)
    df['sma_slope_pct'] = ((df['sma12'] - s6) / s6.replace(0, np.nan) * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # R3: cum_change
    c6 = df['close'].shift(72)
    df['cum_change_pct'] = ((df['close'] - c6) / c6.replace(0, np.nan) * 100).replace([np.inf, -np.inf], 0).fillna(0)

    # R4: quiet_range (earlier range [-72, -36])
    he = df['high'].shift(36).rolling(36, min_periods=18).max()
    le = df['low'].shift(36).rolling(36, min_periods=18).min()
    df['range_earlier'] = ((he - le) / le.replace(0, np.nan) * 100).replace([np.inf, -np.inf], 999).fillna(999)

    # R5: breakout (close > prior_high[-48, -1])
    ph = df['high'].rolling(48, min_periods=24).max().shift(1)
    df['prior_high'] = ph
    df['breakout'] = (df['close'] > ph).astype(int)

    # 24h quote volume
    df['qv24'] = (df['volume'] * df['close']).rolling(288, min_periods=72).sum()

    # RSI
    d = df['close'].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(l != 0, 100.0)
    rsi = rsi.where(~((l == 0) & (g == 0)), 50.0)
    df['rsi'] = rsi.fillna(50.0)

    # 评分
    h = {}
    h['vol_ratio'] = df['vol_ratio'] >= VP['vol_ratio']
    h['vol_rank'] = df['vol_ratio_rank'] >= VP['vol_rank']
    h['sma_slope'] = df['sma_slope_pct'] >= VP['sma_slope_pct']
    h['cum_change'] = df['cum_change_pct'] >= VP['cum_change_pct']
    h['quiet'] = df['range_earlier'] <= VP['quiet_range_pct']
    h['breakout'] = df['breakout'] == 1
    df['moji_score'] = (
        h['vol_ratio'].astype(float) * VP['vol_ratio_w'] +
        h['vol_rank'].astype(float) * VP['vol_rank_w'] +
        h['sma_slope'].astype(float) * VP['sma_slope_w'] +
        h['cum_change'].astype(float) * VP['cum_change_w'] +
        h['quiet'].astype(float) * VP['quiet_w'] +
        h['breakout'].astype(float) * VP['breakout_w']
    )
    df['moji_rules'] = sum(s.astype(int) for s in h.values())

    # 入场gate
    vol_surge = (df['vol_ratio_rank'] >= VP['vol_rank']) | (df['vol_ratio'] >= VP['vol_ratio'])
    df['gate'] = (
        (df['breakout'] == 1) & vol_surge &
        (df['qv24'] >= VP['min_quote_volume']) &
        (df['close'] >= VP['min_price']) & (df['close'] <= VP['max_price']) &
        (df['rsi'] >= VP['rsi_min']) &
        (df['moji_score'] >= VP['score_threshold']) &
        (df['moji_rules'] >= VP['min_rules'])
    )
    # 分级 (tag 模拟)
    df['tag_strong'] = df['gate'] & (df['moji_score'] >= VP['strong_score']) & (df['moji_rules'] >= VP['strong_rules'])
    df['tag_medium'] = df['gate'] & ~df['tag_strong'] & (df['moji_score'] >= VP['medium_score'])
    df['tag_weak'] = df['gate'] & ~df['tag_strong'] & ~df['tag_medium']

    return df


class BacktestEngine:
    def __init__(self):
        self.open_trades = {}
        self.closed = []
        self.capital = TP['total_capital']
        self.total_comm = 0.0
        self.pair_cd = {}
        self._last_global = None
        self._daily_count = {}
        self._total_tickets = 0

    def try_enter(self, symbol, ts, price, score, rules, tag):
        if len(self.open_trades) >= TP['max_trades']:
            return False, 'max_open'
        if self.capital < TP['stake']:
            return False, 'no_capital'
        if self._total_tickets >= 200:
            return False, 'max_total'
        day = ts.date().isoformat()
        if self._daily_count.get(day, 0) >= TP['max_daily']:
            return False, 'daily_limit'
        lp = self.pair_cd.get(symbol)
        if lp and (ts - lp).total_seconds() < TP['per_pair_cd'] * 60:
            return False, 'pair_cd'
        if self._last_global and (ts - self._last_global).total_seconds() < TP['global_cd'] * 60:
            return False, 'global_cd'

        c = TP['stake'] * TP['commission']
        self.capital -= TP['stake'] + c
        self.total_comm += c
        self.open_trades[symbol] = {
            'symbol': symbol, 'entry_time': ts, 'entry_price': price,
            'stake': TP['stake'], 'amount': (TP['stake'] - c) / price,
            'score': score, 'rules': rules, 'tag': tag,
            'max_rate': price, 'peak_profit': 0.0,
        }
        self.pair_cd[symbol] = ts
        self._last_global = ts
        self._daily_count[day] = self._daily_count.get(day, 0) + 1
        self._total_tickets += 1
        return True, tag

    def should_exit(self, symbol, ts, price):
        t = self.open_trades.get(symbol)
        if t is None: return None
        ep = t['entry_price']; pp = (price - ep) / ep
        if price > t['max_rate']:
            t['max_rate'] = price; t['peak_profit'] = (t['max_rate'] - ep) / ep
        age = (ts - t['entry_time']).total_seconds() / 3600

        # ROI 阶梯 (按持仓分钟数)
        age_min = age * 60
        for min_bar, roi_target in TP['roi_steps']:
            if age_min >= min_bar and pp >= roi_target:
                return f'roi_{min_bar}min', pp

        # 硬止损
        if pp <= TP['stoploss']:
            return 'stoploss', pp

        # 棘轮
        for profit_level, protect in TP['ratchet'].items():
            if pp >= profit_level and pp < protect:
                return f'ratchet_{int(profit_level*100)}', pp

        # 时间退出
        if age >= TP['dud_early'] and pp < TP['dud_threshold']:
            return 'dud_early', pp
        if age >= TP['stale_half_day'] and pp < 0:
            return 'stale_half_day', pp
        if age >= TP['stale_dud'] and pp < 0.01:
            return 'stale_dud', pp
        if age >= TP['medium_2d'] and pp < 0.05:
            return 'medium_2d', pp
        if age >= TP['max_hold']:
            return 'max_hold_3d', pp

        # 回撤
        if t['peak_profit'] >= TP['pullback_peak'] and pp >= TP['pullback_min']:
            if t['peak_profit'] - pp >= TP['pullback_dd']:
                return 'pullback_exit', pp
        return None

    def exit_trade(self, sym, ts, px, reason, pp):
        t = self.open_trades.pop(sym)
        rv = t['stake'] * (1 + pp); ec = rv * TP['commission']
        self.total_comm += ec; self.capital += rv - ec
        t.update({
            'exit_time': ts, 'exit_price': px, 'exit_reason': reason,
            'profit_pct': pp * 100, 'net_profit_usd': (rv - ec) - t['stake'],
            'hold_hours': (ts - t['entry_time']).total_seconds() / 3600,
            'peak_profit_pct': t['peak_profit'] * 100,
        })
        self.closed.append(t)

    def force_close_all(self, coin_data):
        for sym in list(self.open_trades.keys()):
            tn = list(coin_data[sym].index)[-1]
            px = coin_data[sym].loc[tn, 'close']
            pp = (px - self.open_trades[sym]['entry_price']) / self.open_trades[sym]['entry_price']
            self.exit_trade(sym, tn, px, 'force_close', pp)


def load_all_data():
    """加载5m feather + 计算所有指标"""
    import glob
    files = sorted(Path("E:/source/freqtrade/user_data/data/gate").glob("*-5m.feather"))
    coin_data = {}
    for f in files:
        sym = f.name.replace("-5m.feather", "")
        df = pd.read_feather(f).set_index('date').sort_index()
        df = compute_indicators(df)
        coin_data[sym] = df
    logger.info(f"Loaded {len(coin_data)} coins")
    return coin_data


def run():
    coin_data = load_all_data()
    eng = BacktestEngine()

    all_ts = sorted(set().union(*[d.index for d in coin_data.values()]))
    logger.info(f"Running on {len(all_ts)} candles...")

    for ts in all_ts:
        # exits
        for sym in list(eng.open_trades.keys()):
            if sym not in coin_data or ts not in coin_data[sym].index: continue
            r = eng.should_exit(sym, ts, coin_data[sym].loc[ts, 'close'])
            if r: eng.exit_trade(sym, ts, coin_data[sym].loc[ts, 'close'], *r)

        # entries
        for sym, d in coin_data.items():
            if sym in eng.open_trades: continue
            if ts not in d.index: continue
            r = d.loc[ts]
            if not r['gate']: continue
            # 分级标签
            if r['tag_strong']: tag = 'moji_strong'
            elif r['tag_medium']: tag = 'moji_medium'
            else: tag = 'moji_weak'
            ok, reason = eng.try_enter(sym, ts, r['close'], r['moji_score'], int(r['moji_rules']), tag)
            if ok: break  # 每个tick最多入场1个 (模拟confirm中的15min global_cd)

    eng.force_close_all(coin_data)
    return eng


def print_report(eng):
    df = pd.DataFrame(eng.closed)
    if len(df) == 0:
        print("\n⚠️ 无交易")
        return

    n = len(df)
    w = df[df['profit_pct'] > 0]; l = df[df['profit_pct'] <= 0]
    nw = len(w); nl = len(l)
    wr = nw / n * 100
    avg_w = w['profit_pct'].mean() if nw > 0 else 0
    avg_l = l['profit_pct'].mean() if nl > 0 else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else float('inf')
    total_net = df['net_profit_usd'].sum()
    final = 20 + total_net
    roi = total_net / 20 * 100
    best = df.loc[df['profit_pct'].idxmax()]
    worst = df.loc[df['profit_pct'].idxmin()]

    # ── 1. 完整 Summary ──
    print(f"\n{'='*70}")
    print(f"📊 MojiCoinHunter v4 回测报告")
    print(f"{'='*70}")
    print(f"  ═══ BACKTESTING SUMMARY ═══")
    print(f"  Total Trades:          {n}")
    print(f"  Win Rate:              {wr:.1f}%  ({nw}W / {nl}L)")
    print(f"  Profit Ratio (avgW/avgL): {rr:.2f}x")
    print(f"  Avg Profit (all):      {df['profit_pct'].mean():.2f}%")
    print(f"  Avg Win:               {avg_w:+.2f}%")
    print(f"  Avg Loss:              {avg_l:+.2f}%")
    print(f"  Best Trade:            {best['symbol']}  {best['profit_pct']:+.2f}%")
    print(f"  Worst Trade:           {worst['symbol']}  {worst['profit_pct']:+.2f}%")
    print(f"  Total Net Profit:      ${total_net:+.2f}")
    print(f"  Final Capital:         ${final:.2f}  (from $20)")
    print(f"  ROI:                   {roi:+.2f}%")
    print(f"  Avg Hold:              {df['hold_hours'].mean():.1f}h")
    print(f"  Avg Peak Profit:       {df['peak_profit_pct'].mean():.2f}%")
    print(f"  Total Commission:      ${eng.total_comm:.2f}")
    print(f"  Open Trades at End:    {len(eng.open_trades)}")

    # ── 2. Exit Reason 分解 ──
    print(f"\n  ═══ EXIT REASON BREAKDOWN ═══")
    exit_stats = df.groupby('exit_reason').agg(
        count=('profit_pct', 'count'),
        avg_profit=('profit_pct', 'mean'),
        sum_net=('net_profit_usd', 'sum'),
    ).sort_values('count', ascending=False)
    for reason, row in exit_stats.iterrows():
        pct = row['count'] / n * 100
        print(f"  {reason:22s}  {int(row['count']):4d}  ({pct:5.1f}%)  avg={row['avg_profit']:+7.2f}%  net=${row['sum_net']:+7.2f}")

    # ── 3. Enter Tag 分解 ──
    print(f"\n  ═══ ENTER TAG BREAKDOWN ═══")
    tag_stats = df.groupby('tag').agg(
        count=('profit_pct', 'count'),
        avg_profit=('profit_pct', 'mean'),
        win_rate=('profit_pct', lambda x: (x > 0).sum() / len(x) * 100),
        sum_net=('net_profit_usd', 'sum'),
    ).sort_values('count', ascending=False)
    for tag, row in tag_stats.iterrows():
        print(f"  {tag:22s}  {int(row['count']):4d}  wr={row['win_rate']:5.1f}%  avg={row['avg_profit']:+7.2f}%  net=${row['sum_net']:+7.2f}")

    # ── 4. 配置信息 ──
    print(f"\n  ═══ CONFIG ═══")
    print(f"  Strategy:       MojiCoinHunter v4 (妖币猎手)")
    print(f"  Timeframe:      5m")
    print(f"  Capital:        $20  ({TP['max_trades']} x ${TP['stake']})")
    print(f"  Stoploss:       {TP['stoploss']*100:+.0f}%")
    print(f"  Commission:     {TP['commission']*100:.1f}%")
    print(f"  ROI Steps:      {TP['roi_steps']}")
    print(f"  Entry Gate:     breakout + vol_surge + score>={VP['score_threshold']} + rules>={VP['min_rules']}")
    print(f"  Rules:          R1(vol_ratio>={VP['vol_ratio']}) + R1b(rank>={VP['vol_rank']}) + R2(slope>={VP['sma_slope_pct']})")
    print(f"                  + R3(cum>={VP['cum_change_pct']}) + R4(quiet<={VP['quiet_range_pct']}) + R5(breakout)")
    print(f"  Signal Tiers:   strong(score>={VP['strong_score']}+rules>={VP['strong_rules']}) > medium(score>={VP['medium_score']}) > weak")
    print(f"  Pair CD:        {TP['per_pair_cd']}min  |  Global CD: {TP['global_cd']}min  |  Daily Max: {TP['max_daily']}")
    print(f"  Price Range:    ${VP['min_price']} ~ ${VP['max_price']}")
    print(f"  Min 24h QVol:   ${VP['min_quote_volume']:,}")

    # ── 5. 交易笔数量级 ──
    print(f"\n  ═══ TRADE COUNT SUMMARY ═══")
    print(f"  Total Entries:      {n}")
    print(f"  Entries per Day:    {n / 8:.1f}  (8 trading days)")
    # Max concurrent: count trades active at each entry/exit boundary
    events = []
    for t in eng.closed:
        events.append((t['entry_time'], +1))
        events.append((t['exit_time'], -1))
    events.sort()
    cur, mx = 0, 0
    for _, delta in events:
        cur += delta; mx = max(mx, cur)
    print(f"  Max Concurrent:     {mx}")
    daily_dist = df['entry_time'].apply(lambda x: x.date().isoformat()).value_counts().sort_index()
    print(f"  Daily Distribution:")
    for day, cnt in daily_dist.items():
        print(f"    {day}: {cnt} entries")

    print(f"\n  {'='*70}")

    df.to_csv(OUTPUT_DIR / 'moji_hunter_v4_aligned_trades.csv', index=False)
    return df


if __name__ == '__main__':
    eng = run()
    print_report(eng)
