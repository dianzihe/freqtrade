import pandas as pd, numpy as np, sys
from glob import glob

files = sorted(glob('user_data/data/gate/*.feather'))
print(f'Total files: {len(files)}', flush=True)

ht = 10
wins_a, total_a, rets_a = 0, 0, []
wins_b, total_b, rets_b = 0, 0, []
wins_c, total_c, rets_c = 0, 0, []
wins_d, total_d, rets_d = 0, 0, []

for fi, f in enumerate(files):
    if fi % 20 == 0:
        print(f'  Processing {fi}/{len(files)}...', flush=True)
    try:
        df = pd.read_feather(f)
        df.set_index('date', inplace=True)
        df['pct'] = df['close'].pct_change() * 100
        df['ma20'] = df['close'].rolling(20).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        n = len(df)

        for i in range(20, n):
            # A: MA deviation
            if df.iloc[i]['close'] < df.iloc[i]['ma20'] * 0.9 and i + ht < n:
                pnl = (df.iloc[i+ht]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'] * 100
                total_a += 1
                if pnl > 0: wins_a += 1
                rets_a.append(pnl)

            # B: 2 red candles sum < -5%
            if i >= 2:
                if (df.iloc[i-1]['pct'] < 0 and df.iloc[i]['pct'] < 0 and
                    (df.iloc[i-1]['pct'] + df.iloc[i]['pct']) < -5 and i + ht < n):
                    pnl = (df.iloc[i+ht]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'] * 100
                    total_b += 1
                    if pnl > 0: wins_b += 1
                    rets_b.append(pnl)

            # C: Vol crash
            if df.iloc[i]['pct'] < -2 and df.iloc[i]['vol_ratio'] > 3 and i + ht < n:
                pnl = (df.iloc[i+ht]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'] * 100
                total_c += 1
                if pnl > 0: wins_c += 1
                rets_c.append(pnl)

            # D: Crash -5 to -10
            pct = df.iloc[i]['pct']
            if -10 <= pct < -5 and i + ht < n:
                pnl = (df.iloc[i+ht]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'] * 100
                total_d += 1
                if pnl > 0: wins_d += 1
                rets_d.append(pnl)

    except Exception as e:
        print(f'  Error on {f}: {e}', flush=True)

print(f'\n====== Results (hold {ht}min, all 200 coins) ======')
if total_a > 0:
    print(f'A.MA-Deviation:  {total_a:>5} sig, WR={wins_a/total_a*100:.1f}%, Avg={np.mean(rets_a):+.4f}%, Med={np.median(rets_a):+.4f}%')
if total_b > 0:
    print(f'B.2-Red-Crash:   {total_b:>5} sig, WR={wins_b/total_b*100:.1f}%, Avg={np.mean(rets_b):+.4f}%, Med={np.median(rets_b):+.4f}%')
if total_c > 0:
    print(f'C.Vol-Crash:     {total_c:>5} sig, WR={wins_c/total_c*100:.1f}%, Avg={np.mean(rets_c):+.4f}%, Med={np.median(rets_c):+.4f}%')
if total_d > 0:
    print(f'D.Crash-5-10:    {total_d:>5} sig, WR={wins_d/total_d*100:.1f}%, Avg={np.mean(rets_d):+.4f}%, Med={np.median(rets_d):+.4f}%')
