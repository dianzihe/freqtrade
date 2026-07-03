"""
时间窗口分析脚本 — 基于 Gate 交易所真实 1m 数据
验证博主关于黄金交易时间的说法对加密货币是否适用
输出: time_analysis_report.html
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

DATA = Path('user_data/data/gate')

CATEGORIES = {
    '稳定型': ['BTC', 'ETH', 'SOL', 'XRP', 'LTC', 'HYPE'],
    '中间币': ['XCN', 'IP', 'BAS', 'PEAQ', 'TA'],
    '妖币':   ['H', 'VELVET', 'BEAT', 'COAI', 'ALLO', 'DN', 'STG'],
}

def load_data():
    all_data = {}
    for cat, coins in CATEGORIES.items():
        for coin in coins:
            f = DATA / f'{coin}_USDT-1m.feather'
            if f.exists():
                df = pd.read_feather(f)
                df['coin'] = coin
                df['category'] = cat
                all_data[coin] = df
            else:
                print(f"  WARNING: {coin} MISSING!")
    return pd.concat(all_data.values(), ignore_index=True)

def zone_stats(df, start_h, end_h):
    t = df['bj_hour'] + df['bj_minute'] / 60
    return df[(t >= start_h) & (t < end_h)]

def main():
    combined = load_data()
    combined['bj_hour'] = combined['date'].dt.tz_convert('Asia/Shanghai').dt.hour
    combined['bj_minute'] = combined['date'].dt.tz_convert('Asia/Shanghai').dt.minute
    combined['weekday'] = combined['date'].dt.tz_convert('Asia/Shanghai').dt.dayofweek
    combined['is_weekend'] = combined['weekday'].isin([5, 6])

    # Metrics
    combined['range_pct'] = (combined['high'] - combined['low']) / combined['open'] * 100
    combined['upper_wick'] = (combined['high'] - combined[['open','close']].max(axis=1)) / combined['open'] * 100
    combined['lower_wick'] = (combined[['open','close']].min(axis=1) - combined['low']) / combined['open'] * 100
    combined['wick_ratio'] = np.where(
        combined['range_pct'] > 0,
        (combined['upper_wick']+combined['lower_wick']) / combined['range_pct'], 0
    )
    combined['is_long_wick'] = combined['wick_ratio'] > 0.7
    combined['min_to_round'] = combined['bj_minute'].apply(lambda m: min(m, 60-m))

    workday = combined[~combined['is_weekend']]
    cats = ['稳定型', '中间币', '妖币']

    # ---- 1. Hourly data ----
    hourly_data = {}
    for cat in cats:
        cd = workday[workday['category']==cat]
        hd = {'hours': list(range(24)), 'range': [], 'volume': [], 'wick': []}
        for h in range(24):
            hh = cd[cd['bj_hour']==h]
            hd['range'].append(round(hh['range_pct'].mean(), 5) if len(hh)>0 else 0)
            hd['volume'].append(round(hh['volume'].mean(), 2) if len(hh)>0 else 0)
            hd['wick'].append(round(hh['is_long_wick'].mean()*100, 2) if len(hh)>0 else 0)
        hourly_data[cat] = hd

    # ---- 2. Zone claims ----
    claims = [
        ('A.亚盘活跃(8:30-10:30)', 8.5, 10.5),
        ('C.午餐垃圾(11:00-13:00)', 11, 13),
        ('D.午后启动(13:30-14:30)', 13.5, 14.5),
        ('F.欧盘最佳(15:30-16:30)', 15.5, 16.5),
        ('G.垃圾时间(17:30-18:30)', 17.5, 18.5),
        ('H.美盘活跃(21:30-23:30)', 21.5, 23.5),
        ('I.深夜稳定(0:00-1:00)', 0, 1),
    ]
    zone_data = []
    for zn, sh, eh in claims:
        for cat in cats:
            cd = workday[workday['category']==cat]
            z = zone_stats(cd, sh, eh)
            overall = cd['range_pct'].mean()
            zone_data.append({
                'zone': zn.split('(')[0],
                'cat': cat,
                'range': round(z['range_pct'].mean(), 5),
                'ratio': round(z['range_pct'].mean()/overall, 3) if overall>0 else 0,
                'wick': round(z['is_long_wick'].mean()*100, 2),
                'vol': round(z['volume'].mean(), 2),
            })

    # ---- 3. Round hour ----
    round_data = []
    for cat in cats:
        cd = workday[workday['category']==cat]
        near = cd[cd['min_to_round']<=5]
        far = cd[cd['min_to_round']>=10]
        round_data.append({
            'cat': cat,
            'near_range': round(near['range_pct'].mean(), 5),
            'far_range': round(far['range_pct'].mean(), 5),
            'near_wick': round(near['is_long_wick'].mean()*100, 2),
            'far_wick': round(far['is_long_wick'].mean()*100, 2),
            'near_vol': round(near['volume'].mean(), 2),
            'far_vol': round(far['volume'].mean(), 2),
        })

    # ---- 4. Monday ----
    monday_data = []
    for cat in cats:
        cd = workday[workday['category']==cat]
        mon = cd[cd['weekday']==0]
        tf = cd[(cd['weekday']>=1)&(cd['weekday']<=4)]
        monday_data.append({
            'cat': cat,
            'mon_early_range': round(zone_stats(mon,8.5,9.5)['range_pct'].mean(),5),
            'mon_late_range': round(zone_stats(mon,9.5,10.5)['range_pct'].mean(),5),
            'tf_early_range': round(zone_stats(tf,8.5,9.5)['range_pct'].mean(),5),
            'mon_early_vol': round(zone_stats(mon,8.5,9.5)['volume'].mean(),2),
            'mon_late_vol': round(zone_stats(mon,9.5,10.5)['volume'].mean(),2),
            'tf_early_vol': round(zone_stats(tf,8.5,9.5)['volume'].mean(),2),
        })

    # ---- 5. Midnight ----
    midnight_data = []
    for cat in cats:
        cd = workday[workday['category']==cat]
        mid = zone_stats(cd,0,1)
        day = cd[(cd['bj_hour']>=1)&(cd['bj_hour']<24)]
        threshold = cd['range_pct'].mean()*3
        midnight_data.append({
            'cat': cat,
            'mid_range': round(mid['range_pct'].mean(),5),
            'mid_std': round(mid['range_pct'].std(),5),
            'day_range': round(day['range_pct'].mean(),5),
            'day_std': round(day['range_pct'].std(),5),
            'mid_wick': round(mid['is_long_wick'].mean()*100,2),
            'day_wick': round(day['is_long_wick'].mean()*100,2),
            'mid_spikes': int((mid['range_pct']>threshold).sum()),
            'day_spikes_per_h': round((day['range_pct']>threshold).sum()/day['bj_hour'].nunique(),1),
        })

    # ---- 6. Coin ranking ----
    coin_avg = workday.groupby(['coin','category'])['range_pct'].mean().reset_index()
    coin_avg = coin_avg.sort_values('range_pct', ascending=False)
    coin_rank = [{'coin': r['coin'], 'cat': r['category'], 'range': round(r['range_pct'],5)}
                 for _, r in coin_avg.iterrows()]

    # ---- 7. Day of week ----
    dow_data = []
    days = ['周一','周二','周三','周四','周五']
    for cat in cats:
        cd = workday[workday['category']==cat]
        d = {'cat': cat}
        for i, dn in enumerate(days):
            dd = cd[cd['weekday']==i]
            d[f'd{i}_range'] = round(dd['range_pct'].mean(), 5) if len(dd)>0 else 0
            d[f'd{i}_vol'] = round(dd['volume'].mean(), 2) if len(dd)>0 else 0
        dow_data.append(d)

    # ---- 8. US session risk ----
    us_data = []
    for cat in cats:
        cd = workday[workday['category']==cat]
        us = zone_stats(cd,21.5,23.5)
        other = cd[~cd.index.isin(us.index)]
        t2x = cd['range_pct'].mean()*2
        us_data.append({
            'cat': cat,
            'us_range': round(us['range_pct'].mean(),5),
            'us_max': round(us['range_pct'].max(),5),
            'us_large_pct': round((us['range_pct']>t2x).mean()*100,2),
            'other_range': round(other['range_pct'].mean(),5),
            'other_large_pct': round((other['range_pct']>t2x).mean()*100,2),
        })

    report = {
        'hourly': hourly_data,
        'zones': zone_data,
        'round': round_data,
        'monday': monday_data,
        'midnight': midnight_data,
        'coin_rank': coin_rank,
        'dow': dow_data,
        'us': us_data,
    }

    with open('time_analysis_data.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False)

    print("数据已导出: time_analysis_data.json")
    print(f"总K线数: {len(combined):,} (工作日: {len(workday):,})")

    # Print key findings
    print("\n========== 关键发现 ==========")
    for d in zone_data:
        v = '高波动' if d['ratio']>1.10 else ('低波动' if d['ratio']<0.90 else '≈均值')
        print(f"  {d['zone']:<25} {d['cat']:<8} range={d['range']:.5f}% ratio={d['ratio']:.2f} wick={d['wick']}% | {v}")

    print("\n整点效应:")
    for d in round_data:
        delta = (d['near_range']/d['far_range']-1)*100
        print(f"  {d['cat']}: 整点±5min={d['near_range']:.5f}% vs 远离={d['far_range']:.5f}% ({delta:+.1f}%)")

    print("\n周一效应:")
    for d in monday_data:
        ratio = d['mon_early_range']/d['tf_early_range']*100 if d['tf_early_range']>0 else 0
        print(f"  {d['cat']}: 周一8:30={d['mon_early_range']:.5f}% vs 平时8:30={d['tf_early_range']:.5f}% ({ratio:.0f}%)")

    print("\n凌晨稳定性:")
    for d in midnight_data:
        print(f"  {d['cat']}: 凌晨Std={d['mid_std']:.5f} vs 白天Std={d['day_std']:.5f} ({(d['mid_std']/d['day_std']*100):.0f}%)")

if __name__ == '__main__':
    main()
