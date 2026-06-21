"""
妖币马丁策略可行性分析
对 H, VELVET, DN, BEAT 四个 Gate.io 妖币做量化分析
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone

# ============================================================
# 1. 加载数据
# ============================================================
coins = ["H_USDT-1m", "VELVET_USDT-1m", "DN_USDT-1m", "BEAT_USDT-1m"]
coin_names = ["H", "VELVET", "DN", "BEAT"]
data = {}

for fname, cname in zip(coins, coin_names):
    df = pd.read_feather(f"user_data/data/gate/{fname}.feather")
    df = df.set_index("date").sort_index()
    # 去除极端零值/缺失
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    data[cname] = df
    print(f"{cname}: {len(df)} rows, "
          f"{df.index[0].strftime('%m/%d %H:%M')} → {df.index[-1].strftime('%m/%d %H:%M')}")

# ============================================================
# 2. 基础统计
# ============================================================
print("\n" + "=" * 70)
print("【基础价格统计】")
print("=" * 70)

for cname, df in data.items():
    close = df["close"]
    vol = df["volume"]

    # 价格变化
    pct_change = (close.iloc[-1] / close.iloc[0] - 1) * 100
    max_p = close.max()
    min_p = close.min()
    high_low_range = (max_p / min_p - 1) * 100  # 期间最大振幅

    # 波动率
    returns = close.pct_change().dropna()
    vol_1m = returns.std()  # 1分钟波动率
    vol_1h = close.resample("1h").last().pct_change().dropna().std()  # 小时波动率
    vol_1d = close.resample("1D").last().pct_change().dropna().std()  # 日波动率
    annual_vol = vol_1d * np.sqrt(365) * 100  # 年化波动率

    # 交易量
    avg_vol_usdt = (vol * close).mean()

    # 回撤
    cummax = close.expanding().max()
    drawdown = (close - cummax) / cummax
    max_dd = drawdown.min() * 100

    # 涨跌天数
    daily = close.resample("1D").agg(["first", "last"])
    daily["ret"] = daily["last"] / daily["first"] - 1
    up_days = (daily["ret"] > 0).sum()
    dn_days = (daily["ret"] < 0).sum()

    print(f"\n--- {cname} ---")
    print(f"  期间涨跌: {pct_change:+.1f}%")
    print(f"  价格范围: {min_p:.6f} ~ {max_p:.6f}")
    print(f"  最大振幅: {high_low_range:.1f}%")
    print(f"  1m 波动率(std): {vol_1m*100:.4f}%")
    print(f"  1h 波动率(std): {vol_1h*100:.2f}%")
    print(f"  日波动率(std): {vol_1d*100:.2f}%")
    print(f"  年化波动率: {annual_vol:.0f}%")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  涨/跌天数: {up_days}/{dn_days}")
    print(f"  日均交易额(USDT): {avg_vol_usdt:,.0f}")


# ============================================================
# 3. 马丁策略回测 - 多周期分析
# ============================================================
print("\n" + "=" * 70)
print("【马丁策略回测】")
print("策略: 固定网格 + 加倍加仓")
print("=" * 70)

def martingale_backtest(df, initial_price=None, grid_pct=0.05, max_layers=6,
                        capital=1000, start_capital_ratio=0.5):
    """
    马丁网格策略:
    - 起始基准价用第一根K线的close
    - 每下跌 grid_pct 加一单，仓位翻倍
    - 每上涨 grid_pct 吃一单止盈
    - 持仓成本按加权均价计算
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    if initial_price is None:
        initial_price = close[0]

    # 网格价格层
    grid_prices = [initial_price * (1 + grid_pct * i) for i in range(-max_layers, max_layers + 1)]
    base_size = capital * start_capital_ratio / initial_price  # 首仓数量

    trades = []
    positions = []  # (layer_idx, qty, entry_price)
    entry_layer = 0  # 基准层(价格=initial_price)的grid index
    equity = capital
    cash = capital
    max_equity = capital
    max_dd = 0
    min_equity = capital
    total_fees = 0
    fee_rate = 0.001  # 0.1%

    # 初始建仓在基准价
    # 如果当前价低于基准，先建多层
    pos_cost = 0
    pos_qty = 0

    for i in range(n):
        c = close[i]
        h = high[i]
        l = low[i]

        # --- 检查止盈 ---
        if pos_qty > 0:
            avg_cost = pos_cost / pos_qty if pos_qty > 0 else 0
            # 止盈目标: 当前价 >= 平均成本 * (1 + grid_pct)
            take_profit = avg_cost * (1 + grid_pct)
            if h >= take_profit:
                # 止盈
                sell_val = pos_qty * take_profit * (1 - fee_rate)
                cash += sell_val
                profit = sell_val - pos_cost
                trades.append({
                    "time": df.index[i],
                    "type": "close",
                    "entry_avg": avg_cost,
                    "exit_price": take_profit,
                    "qty": pos_qty,
                    "profit": profit,
                    "profit_pct": profit / pos_cost * 100
                })
                pos_qty = 0
                pos_cost = 0

        # --- 检查加仓 ---
        if pos_qty == 0:
            # 无仓位时，按当前价建立初始仓位
            pos_qty = base_size
            pos_cost = base_size * c * (1 + fee_rate)
            cash -= pos_cost
            trades.append({
                "time": df.index[i],
                "type": "open",
                "price": c,
                "layer": 1,
                "qty": pos_qty,
                "cost": pos_cost
            })
        else:
            avg_cost = pos_cost / pos_qty
            # 价格相对上次加仓价继续下跌 grid_pct 就加仓
            last_entry = trades[-1]["price"] if trades and trades[-1]["type"] == "open" else c
            if len([t for t in trades if t["type"] == "open"]) == 1:
                last_entry = trades[-1]["price"]
            else:
                # 找最近一次open
                for t in reversed(trades):
                    if t["type"] == "open":
                        last_entry = t["price"]
                        break

            # 当前价格 <= 上次入场价 * (1 - grid_pct)，加仓
            threshold = last_entry * (1 - grid_pct)
            if l <= threshold:
                layer_num = len([t for t in trades if t["type"] == "open"]) + 1
                # 马丁加倍
                multiplier = 2 ** (layer_num - 1)
                add_qty = base_size * multiplier
                add_cost = add_qty * threshold * (1 + fee_rate)
                if add_cost <= cash * 0.95:  # 留5%缓冲
                    cash -= add_cost
                    pos_qty += add_qty
                    pos_cost += add_cost
                    trades.append({
                        "time": df.index[i],
                        "type": "open",
                        "price": threshold,
                        "layer": layer_num,
                        "qty": add_qty,
                        "cost": add_cost,
                        "multiplier": multiplier
                    })

        # --- 计算权益 ---
        unrealized = pos_qty * c if pos_qty > 0 else 0
        equity = cash + unrealized
        if equity > max_equity:
            max_equity = equity
        if equity < min_equity:
            min_equity = equity
        dd = (equity - max_equity) / max_equity if max_equity > 0 else 0
        if dd < max_dd:
            max_dd = dd

    # 最终清算
    if pos_qty > 0:
        final_val = pos_qty * close[-1] * (1 - fee_rate)
        cash += final_val
        profit = final_val - pos_cost
        trades.append({
            "time": df.index[-1],
            "type": "final_close",
            "entry_avg": pos_cost / pos_qty,
            "exit_price": close[-1],
            "qty": pos_qty,
            "profit": profit,
            "profit_pct": profit / pos_cost * 100
        })
        pos_qty = 0

    equity = cash
    total_return = (equity - capital) / capital * 100
    return {
        "total_return_pct": total_return,
        "final_equity": equity,
        "max_drawdown_pct": max_dd * 100,
        "trades": trades,
        "capital": capital
    }


def martingale_grid_v2(df, grid_pct=0.03, max_layers=8, capital=1000,
                        start_qty_usdt=50, use_trailing=False):
    """
    改进版马丁网格:
    - 总共有 max_layers*2+1 层网格（上下对称）
    - 居中建仓（一半多仓，留一半网格做空/止盈空间）
    - 价格触碰网格线即触发
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    mid_price = close[0]
    base_qty = start_qty_usdt / mid_price

    # 建立网格: [below_layers ..., mid, ... above_layers]
    grid_levels = [mid_price * (1 + grid_pct * i) for i in range(-max_layers, max_layers + 1)]
    filled_levels = {}  # level_idx -> qty
    pnl = 0
    cash = capital
    trades = []
    equity_curve = []
    max_equity = capital
    max_drawdown = 0

    # 初始在mid层建半仓
    mid_idx = max_layers
    filled_levels[mid_idx] = base_qty
    cash -= base_qty * mid_price * 1.001

    last_touch_idx = mid_idx

    for i in range(n):
        c = close[i]
        h = high[i]
        l = low[i]

        # 检查哪些网格线被穿过
        for level_idx, level_price in enumerate(grid_levels):
            if l <= level_price <= h:
                # 网格线被触及
                if level_idx < last_touch_idx:
                    # 价格下行 → 加仓（马丁翻倍）
                    if level_idx not in filled_levels:
                        layer_dist = last_touch_idx - level_idx
                        mult = 2 ** layer_dist
                        qty = base_qty * mult
                        cost = qty * level_price * 1.001
                        if cost <= cash:
                            cash -= cost
                            filled_levels[level_idx] = qty
                            trades.append({
                                "time": df.index[i],
                                "action": "buy",
                                "price": level_price,
                                "level": -max_layers + level_idx,
                                "qty": qty,
                                "cost": cost,
                                "multiplier": mult
                            })
                elif level_idx > last_touch_idx:
                    # 价格上行 → 卖出止盈
                    for sell_idx in sorted(filled_levels.keys()):
                        if sell_idx < level_idx:
                            qty = filled_levels[sell_idx]
                            sell_val = qty * level_price * 0.999
                            profit = sell_val - qty * grid_levels[sell_idx] * 1.001
                            cash += sell_val
                            pnl += profit
                            trades.append({
                                "time": df.index[i],
                                "action": "sell",
                                "price": level_price,
                                "entry_level": -max_layers + sell_idx,
                                "exit_level": -max_layers + level_idx,
                                "qty": qty,
                                "profit": profit
                            })
                            del filled_levels[sell_idx]
                last_touch_idx = level_idx

        # 权益
        pos_val = sum(qty * c for qty in filled_levels.values())
        equity = cash + pos_val
        equity_curve.append(equity)
        if equity > max_equity:
            max_equity = equity
        dd = (equity - max_equity) / max_equity if max_equity > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd

    # 清算
    final_pos_val = sum(qty * close[-1] * 0.999 for qty in filled_levels.values())
    cash += final_pos_val
    total_return = (cash - capital) / capital * 100

    return {
        "total_return_pct": total_return,
        "final_equity": cash,
        "max_drawdown_pct": max_drawdown * 100,
        "trades": trades,
        "equity_curve": equity_curve,
        "filled_positions_at_end": len(filled_levels)
    }


# 运行多个参数组合
print("\n--- 策略1: 标准马丁（单一方向） ---")
configs = [
    {"grid_pct": 0.03, "max_layers": 5},
    {"grid_pct": 0.05, "max_layers": 4},
    {"grid_pct": 0.08, "max_layers": 3},
    {"grid_pct": 0.10, "max_layers": 3},
]

for cname, df in data.items():
    best = None
    for cfg in configs:
        res = martingale_backtest(df, grid_pct=cfg["grid_pct"],
                                  max_layers=cfg["max_layers"],
                                  capital=1000, start_capital_ratio=0.3)
        if best is None or res["total_return_pct"] > best["total_return_pct"]:
            best = res
            best_cfg = cfg
    print(f"\n{cname} (最优参数: grid={best_cfg['grid_pct']*100:.0f}%, layers={best_cfg['max_layers']}):")
    print(f"  总收益: {best['total_return_pct']:+.2f}%")
    print(f"  最大回撤: {best['max_drawdown_pct']:.2f}%")
    print(f"  最终权益: ${best['final_equity']:.2f}")
    trades_open = [t for t in best["trades"] if t["type"] == "open"]
    trades_close = [t for t in best["trades"] if t["type"] in ("close", "final_close")]
    print(f"  开仓次数: {len(trades_open)}, 平仓次数: {len(trades_close)}")
    if trades_close:
        profits = [t["profit"] for t in trades_close]
        print(f"  胜率: {sum(1 for p in profits if p > 0)/len(profits)*100:.0f}%")
        print(f"  平均盈利: ${np.mean([p for p in profits if p > 0]):.2f}" if any(p > 0 for p in profits) else "  无盈利交易")
        print(f"  平均亏损: ${np.mean([p for p in profits if p < 0]):.2f}" if any(p < 0 for p in profits) else "  无亏损交易")


print("\n--- 策略2: 多空网格马丁 ---")
for cname, df in data.items():
    best = None
    for gp in [0.03, 0.05, 0.08]:
        for ml in [5, 8, 10]:
            res = martingale_grid_v2(df, grid_pct=gp, max_layers=ml, capital=1000, start_qty_usdt=30)
            if best is None or res["total_return_pct"] > best["total_return_pct"]:
                best = res
                best_params = (gp, ml)
    print(f"\n{cname} (最优: grid={best_params[0]*100:.0f}%, layers={best_params[1]}):")
    print(f"  总收益: {best['total_return_pct']:+.2f}%")
    print(f"  最大回撤: {best['max_drawdown_pct']:.2f}%")
    print(f"  最终权益: ${best['final_equity']:.2f}")
    print(f"  交易次数: {len(best['trades'])}")
    print(f"  期末持仓层数: {best['filled_positions_at_end']}")


# ============================================================
# 4. 专门分析 H 币（最妖）
# ============================================================
print("\n" + "=" * 70)
print("【H 币深入分析 - 最具代表性妖币】")
print("=" * 70)

h = data["H"]
h_close = h["close"]
h_ret = h_close.pct_change().dropna()

# 极端波动统计
extreme_up = h_ret[h_ret > 0.01]  # 单分钟涨超1%
extreme_dn = h_ret[h_ret < -0.01]  # 单分钟跌超1%
print(f"单分钟波动>1%: 上涨{len(extreme_up)}次, 下跌{len(extreme_dn)}次 (总{len(h_ret)}根K线)")
print(f"单分钟最大涨幅: {h_ret.max()*100:.2f}%")
print(f"单分钟最大跌幅: {h_ret.min()*100:.2f}%")

# 1小时波动
h_1h = h_close.resample("1h").last().pct_change().dropna()
print(f"1小时最大涨幅: {h_1h.max()*100:.2f}%")
print(f"1小时最大跌幅: {h_1h.min()*100:.2f}%")

# 分时段 1h 波动分布
h_1h_abs = h_1h.abs()
print(f"1小时波动分位数: 50%={h_1h_abs.quantile(0.5)*100:.2f}%, "
      f"75%={h_1h_abs.quantile(0.75)*100:.2f}%, "
      f"90%={h_1h_abs.quantile(0.9)*100:.2f}%, "
      f"95%={h_1h_abs.quantile(0.95)*100:.2f}%")

# 检查 V 形反转频率
h_1h_ret = h_1h.values
v_reversals = 0
v_total = len(h_1h_ret) - 2
for i in range(len(h_1h_ret) - 2):
    r1, r2, r3 = h_1h_ret[i], h_1h_ret[i+1], h_1h_ret[i+2]
    if r1 < -0.02 and r2 < 0 and r3 > 0.02:  # 跌→跌→大幅涨
        v_reversals += 1
    if r1 < -0.03 and r3 > 0.03:
        v_reversals += 1
print(f"V形反转信号(1h级别): {v_reversals} 次 (总{v_total}个窗口)")

# 模拟马丁在H币不同时间窗口的表现
print("\n--- H 币分时段马丁回测 ---")
h_df = data["H"]
total_minutes = len(h_df)
segments = [
    ("全程", 0, total_minutes),
    ("前1/3", 0, total_minutes // 3),
    ("中1/3", total_minutes // 3, 2 * total_minutes // 3),
    ("后1/3", 2 * total_minutes // 3, total_minutes),
]

for seg_name, start, end in segments:
    seg = h_df.iloc[start:end]
    if len(seg) < 500:
        continue
    res = martingale_backtest(seg, grid_pct=0.05, max_layers=4, capital=1000, start_capital_ratio=0.3)
    res_g = martingale_grid_v2(seg, grid_pct=0.05, max_layers=6, capital=1000, start_qty_usdt=30)
    print(f"  {seg_name}: 单向马丁={res['total_return_pct']:+.1f}% (DD={res['max_drawdown_pct']:.1f}%), "
          f"多空网格={res_g['total_return_pct']:+.1f}% (DD={res_g['max_drawdown_pct']:.1f}%)")


# ============================================================
# 5. 综合结论与风险评估
# ============================================================
print("\n" + "=" * 70)
print("【综合评估 - 马丁策略在妖币上的可行性】")
print("=" * 70)

# 统计各币种马丁策略在不同参数下的收益
all_results = []
for cname, df in data.items():
    for gp in [0.03, 0.05, 0.08]:
        for ml in [3, 4, 5]:
            r = martingale_backtest(df, grid_pct=gp, max_layers=ml, capital=1000, start_capital_ratio=0.3)
            all_results.append({
                "coin": cname,
                "grid_pct": gp,
                "max_layers": ml,
                "return": r["total_return_pct"],
                "max_dd": r["max_drawdown_pct"],
                "final_eq": r["final_equity"]
            })

results_df = pd.DataFrame(all_results)

print("\n按币种统计马丁策略表现:")
for cname in coin_names:
    sub = results_df[results_df["coin"] == cname]
    print(f"\n{cname}:")
    print(f"  平均收益: {sub['return'].mean():+.2f}% (最好: {sub['return'].max():+.2f}%, 最差: {sub['return'].min():+.2f}%)")
    print(f"  平均最大回撤: {sub['max_dd'].mean():.2f}%")
    win_rate = (sub["return"] > 0).sum() / len(sub) * 100
    print(f"  策略组合胜率: {win_rate:.0f}% ({sum(sub['return'] > 0)}/{len(sub)})")
    # 风险回报比
    profitable = sub[sub["return"] > 0]
    if len(profitable) > 0:
        print(f"  盈利时平均收益: {profitable['return'].mean():+.2f}%")
        print(f"  盈利时平均DD: {profitable['max_dd'].mean():.2f}%")

best_combo = results_df.loc[results_df["return"].idxmax()]
worst_combo = results_df.loc[results_df["return"].idxmin()]
print(f"\n最佳参数组合: {best_combo['coin']} grid={best_combo['grid_pct']*100:.0f}% "
      f"layers={int(best_combo['max_layers'])} → 收益 {best_combo['return']:+.2f}%")
print(f"最差参数组合: {worst_combo['coin']} grid={worst_combo['grid_pct']*100:.0f}% "
      f"layers={int(worst_combo['max_layers'])} → 收益 {worst_combo['return']:+.2f}%")


# ============================================================
# 6. 波动区间分析 - 马丁策略最关键的要素
# ============================================================
print("\n" + "=" * 70)
print("【波动区间分析 - 马丁网格匹配度】")
print("=" * 70)

for cname, df in data.items():
    close = df["close"]
    # 以初始价的百分比看价格运行区间
    init_price = close.iloc[0]
    pct_from_init = close / init_price - 1

    print(f"\n{cname} (初始价={init_price:.6f}):")
    for pct_level in [-0.9, -0.8, -0.7, -0.6, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]:
        hit = (pct_from_init <= pct_level if pct_level < 0 else pct_from_init >= pct_level)
        pct_time = hit.sum() / len(pct_from_init) * 100
        if pct_time > 0.1:
            print(f"  价格触及 {pct_level*100:+.0f}% 的时间占比: {pct_time:.1f}%")

    # 最大持续下跌（马丁最怕这个）
    cumulative_down = pd.Series(0.0, index=close.index)
    peak = close.iloc[0]
    for i in range(len(close)):
        if close.iloc[i] > peak:
            peak = close.iloc[i]
        cumulative_down.iloc[i] = (close.iloc[i] / peak - 1) * 100

    print(f"  最深累计跌幅: {cumulative_down.min():.1f}%")
    print(f"  当前累计跌幅: {cumulative_down.iloc[-1]:.1f}%")

    # 连续下跌小时数（马丁杀手）
    hourly = close.resample("1h").last().dropna()
    streak = 0
    max_streak = 0
    streak_drops = []
    for i in range(1, len(hourly)):
        if hourly.iloc[i] < hourly.iloc[i-1]:
            streak += 1
        else:
            if streak > 0:
                streak_drops.append(streak)
                max_streak = max(max_streak, streak)
            streak = 0
    if streak > 0:
        max_streak = max(max_streak, streak)

    print(f"  最长连续下跌(小时): {max_streak}")
    if max_streak > 0:
        # 连续下跌期间总跌幅
        print(f"  [RISK] 马丁风险: 连续{max_streak}小时下跌意味着可能需要{2**(max_streak//2)}倍资金接盘")


# ============================================================
# 7. 资金安全垫分析
# ============================================================
print("\n" + "=" * 70)
print("【资金安全垫 - 极端行情下所需准备金】")
print("=" * 70)

for cname, df in data.items():
    close = df["close"]
    init = close.iloc[0]
    worst_price = close.min()
    worst_pct = (worst_price / init - 1) * 100

    print(f"\n{cname} (初始={init:.6f}, 最低={worst_price:.6f}, 跌幅={worst_pct:.1f}%):")
    for grid_pct in [0.03, 0.05, 0.08]:
        layers_needed = int(abs(worst_pct) / (grid_pct * 100)) + 1
        # 计算到该层所需总资金（首仓=1）
        total_mult = sum(2 ** i for i in range(layers_needed))
        print(f"  {grid_pct*100:.0f}% 网格 → 需要{layers_needed}层, "
              f"总资金倍数={total_mult}x首仓 "
              f"(即首仓=$100需总资金=${total_mult * 100})")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
