"""
彩票数扩展测试: 20/40/60/80 仓位
总资金 $20 固定, 仓位越多每仓越小, 数据只加载一次
"""
import logging, numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("E:/source/freqtrade/deliverables/moji-coin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 复用 v4 的数据加载和信号逻辑 ──
from moji_hunter_backtest_v4 import (
    load_and_prepare, check_layer1_5m, TRADE_PARAMS,
)

class SweepManager:
    def __init__(self, n):
        self.n = n
        self.stake = 20.0 / n
        self.capital = 20.0
        self.open = {}
        self.closed = []
        self.comm = 0.0

    def enter(self, sym, ts, px, sc, ly):
        if len(self.open) >= self.n or self.capital < self.stake:
            return False
        c = self.stake * 0.002
        self.capital -= self.stake + c; self.comm += c
        self.open[sym] = {
            'symbol': sym, 'entry_time': ts, 'entry_price': px,
            'stake': self.stake, 'amount': (self.stake-c)/px,
            'score': sc, 'layers': ly, 'max_rate': px,
            'peak_profit': 0.0, 'tickets': self.n,
        }
        return True

    def should_exit(self, sym, ts, px):
        t = self.open.get(sym)
        if t is None: return None
        ep = t['entry_price']; pp = (px - ep) / ep
        if px > t['max_rate']:
            t['max_rate'] = px; t['peak_profit'] = (t['max_rate'] - ep) / ep
        age = (ts - t['entry_time']).total_seconds() / 3600
        if pp <= -0.08: return 'stoploss', pp
        if pp >= 1.0 and pp < 0.50: return 'ratchet_100', pp
        if pp >= 0.50 and pp < 0.25: return 'ratchet_50', pp
        if pp >= 0.20 and pp < 0.10: return 'ratchet_20', pp
        if pp >= 0.10 and pp < 0.0: return 'ratchet_10', pp
        if age >= 8 and pp < -0.05: return 'dud_early', pp
        if age >= 12 and pp < 0: return 'stale_half_day', pp
        if age >= 24 and pp < 0.01: return 'stale_dud', pp
        if age >= 48 and pp < 0.05: return 'medium_2d', pp
        if age >= 72: return 'max_hold_3d', pp
        if t['peak_profit'] >= 0.15 and pp >= 0.08:
            if t['peak_profit'] - pp >= 0.12: return 'pullback_exit', pp
        return None

    def exit(self, sym, ts, px, reason, pp):
        t = self.open.pop(sym)
        rv = t['stake'] * (1 + pp)
        ec = rv * 0.002; self.comm += ec
        self.capital += rv - ec
        t.update({'exit_time': ts, 'exit_price': px, 'exit_reason': reason,
                  'profit_pct': pp * 100, 'net_profit_usd': (rv - ec) - t['stake'],
                  'hold_hours': (ts - t['entry_time']).total_seconds() / 3600,
                  'peak_profit_pct': t['peak_profit'] * 100})
        self.closed.append(t)

    def force_close_all(self, coin_data):
        for sym in list(self.open.keys()):
            tn = list(coin_data[sym]['5m'].index)[-1]
            px = coin_data[sym]['5m'].loc[tn, 'close']
            pp = (px - self.open[sym]['entry_price']) / self.open[sym]['entry_price']
            self.exit(sym, tn, px, 'force_close', pp)


def run_one(coin_data, n):
    mgr = SweepManager(n)
    tp = TRADE_PARAMS
    all_ts = sorted(set().union(*[d['5m'].index for d in coin_data.values()]))
    cd = {}; lg = None

    for ts in all_ts:
        for sym in list(mgr.open.keys()):
            if sym not in coin_data or ts not in coin_data[sym]['5m'].index: continue
            r = mgr.should_exit(sym, ts, coin_data[sym]['5m'].loc[ts, 'close'])
            if r: mgr.exit(sym, ts, coin_data[sym]['5m'].loc[ts, 'close'], *r)

        if lg and (ts - lg).total_seconds() < 120 * 60: continue

        best = []
        for sym, d in coin_data.items():
            if sym in mgr.open: continue
            if cd.get(sym) and (ts - cd[sym]).total_seconds() < 120 * 60: continue
            if ts not in d['5m'].index: continue
            r = d['5m'].loc[ts]
            if not check_layer1_5m(r): continue
            px = r['close']
            if px < 0.0001 or px > 500: continue
            if r.get('volume', 0) * px * 288 < tp['min_quote_volume_24h']: continue
            sc = ((r['pre_change_pct'] - 2.0) * 15 + (r['sma_slope_pct'] - 1.2) * 15 +
                  (r['vol_concentration'] - 0.03) * 200 + (r['volume_ratio'] - 1.5) * 5)
            best.append((sym, sc, px))
        if best:
            best.sort(key=lambda x: x[1], reverse=True)
            sym, sc, px = best[0]
            if mgr.enter(sym, ts, px, sc, (True, False, False)):
                cd[sym] = ts; lg = ts

    mgr.force_close_all(coin_data)
    return mgr


def summary(mgr):
    df = pd.DataFrame(mgr.closed)
    if len(df) == 0: return dict(tickets=mgr.n, trades=0)
    w = df[df['profit_pct'] > 0]; l = df[df['profit_pct'] <= 0]
    return {
        'tickets': mgr.n, 'stake': mgr.stake, 'trades': len(df),
        'wins': len(w), 'losses': len(l),
        'win_rate': len(w)/len(df)*100,
        'avg_win': w['profit_pct'].mean() if len(w)>0 else 0,
        'avg_loss': l['profit_pct'].mean() if len(l)>0 else 0,
        'profit_ratio': abs(w['profit_pct'].mean()/l['profit_pct'].mean()) if len(l)>0 else float('inf'),
        'total_net': df['net_profit_usd'].sum(),
        'roi': df['net_profit_usd'].sum()/20*100,
        'final_cap': mgr.capital, 'commission': mgr.comm,
        'unique_coins': df['symbol'].nunique(),
        'max_win': df['profit_pct'].max(),
        'max_loss': df['profit_pct'].min(),
        'hits_gt_10pct': len(df[df['profit_pct'] > 10]),
        'hits_gt_20pct': len(df[df['profit_pct'] > 20]),
    }


def main():
    print("Loading data (once)...")
    coin_data = load_and_prepare()

    results = []
    for n in [20, 40, 60, 80]:
        print(f"  Sweeping {n} tickets...")
        mgr = run_one(coin_data, n)
        r = summary(mgr)
        results.append(r)
        print(f"    {r['trades']} trades, {r['win_rate']:.1f}% WR, "
              f"net=${r['total_net']:.2f}, roi={r['roi']:.1f}%")

    # ── 表格 ──
    hdr = f"{'仓位':>5s} {'每仓$':>7s} {'交易':>5s} {'胜/负':>7s} {'赢率':>6s} {'盈亏比':>6s} {'>10%':>5s} {'>20%':>5s} {'净收益':>8s} {'ROI':>7s} {'最终$':>8s} {'币种':>5s}"
    print(f"\n{'='*105}")
    print(f"📊 彩票数扩展测试: $20 → 20/40/60/80 仓位")
    print(f"{'='*105}")
    print(hdr)
    print('-'*105)
    for r in results:
        print(f"{r['tickets']:5d} ${r['stake']:6.4f} {r['trades']:5d} "
              f"{r['wins']:2d}/{r['losses']:2d}  "
              f"{r['win_rate']:5.1f}% {r['profit_ratio']:5.2f}x "
              f"{r['hits_gt_10pct']:4d}  {r['hits_gt_20pct']:4d}  "
              f"${r['total_net']:7.2f} {r['roi']:6.1f}% "
              f"${r['final_cap']:7.2f} {r['unique_coins']:4d}")
    print('-'*105)

    best = max(results, key=lambda x: x['total_net'])
    print(f"\n🏆 最佳: {best['tickets']} 仓位, 每仓 ${best['stake']:.4f}, "
          f"ROI {best['roi']:.1f}%, ${best['total_net']:.2f}")
    print(f"   胜率 {best['win_rate']:.1f}%, 盈亏比 {best['profit_ratio']:.2f}x, "
          f"大赢家(>10%) {best['hits_gt_10pct']}笔, (>20%) {best['hits_gt_20pct']}笔")

    # 趋势分析
    print("\n📈 趋势分析:")
    trades_str = " -> ".join(str(r['trades']) for r in results)
    wr_str = " -> ".join("{:.1f}%".format(r['win_rate']) for r in results)
    net_str = " -> ".join("${:.2f}".format(r['total_net']) for r in results)
    big_str = " -> ".join(str(r['hits_gt_10pct']) for r in results)
    comm_str = " -> ".join("${:.2f}".format(r['commission']) for r in results)
    print(f"   交易数: {trades_str}")
    print(f"   赢率:   {wr_str}")
    print(f"   净收益: {net_str}")
    print(f"   大赢>10%: {big_str}")
    print(f"   手续费: {comm_str}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'ticket_sweep_results.csv', index=False)


if __name__ == '__main__':
    main()
