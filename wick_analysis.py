"""
插针事件深度分析 v3
====================
使用 seconds_from_event 字段精确匹配事件，避免索引重复问题。
"""

import pandas as pd
import numpy as np
from pathlib import Path

FORMATTED = Path("user_data/orderbook_data/formatted")
DEPTH_LEVELS = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def analyze_pair(pair_name):
    base = FORMATTED / pair_name
    wicks = pd.read_csv(base / "wick_events.csv")
    wicks["timestamp"] = pd.to_datetime(wicks["timestamp"])
    snaps = pd.read_parquet(base / "wick_depth_snapshots.parquet")

    if len(snaps) == 0 or len(wicks) == 0:
        print(f"\n  {pair_name}: No data")
        return None

    results = []
    skipped = 0

    for _, wick in wicks.iterrows():
        event_ts = wick["timestamp"]
        event_type = wick["event_type"]
        magnitude = wick["magnitude_pct"]

        # 用精确的 magnitude 和 seconds_from_event 匹配事件
        mag_match = abs(snaps["event_magnitude_pct"] - magnitude) < 0.001
        type_match = snaps["event_type"] == event_type

        # 该事件的所有快照行
        event_rows = snaps[mag_match & type_match].copy()

        if len(event_rows) == 0:
            skipped += 1
            continue

        # 按 seconds_from_event 排序
        event_rows = event_rows.sort_values("seconds_from_event")

        # 找出 before (-25 到 -5 秒) 和 after (+5 到 +25 秒)
        before_mask = (event_rows["seconds_from_event"] >= -25) & (event_rows["seconds_from_event"] <= -5)
        after_mask = (event_rows["seconds_from_event"] >= 5) & (event_rows["seconds_from_event"] <= 25)

        before_rows = event_rows[before_mask]
        after_rows = event_rows[after_mask]

        if len(before_rows) < 3 or len(after_rows) < 3:
            skipped += 1
            continue

        # 取稳定状态：before 最后一行，after 第一行
        depth_any_col = "bid_depth_0_1pct"

        # 对于 before，从后往前找有非NA深度的行
        before_valid = before_rows[before_rows[depth_any_col].notna()]
        after_valid = after_rows[after_rows[depth_any_col].notna()]

        if len(before_valid) == 0 or len(after_valid) == 0:
            skipped += 1
            continue

        before_row = before_valid.iloc[-1]
        after_row = after_valid.iloc[0]

        # 判断方向
        if event_type == "up_wick":
            side = "ask"
            side_cn = "ask(卖单)"
        else:
            side = "bid"
            side_cn = "bid(买单)"

        # 计算各档深度消耗
        level_analysis = []
        for pct in DEPTH_LEVELS:
            col_name = f"{side}_depth_{str(pct).replace('.', '_')}pct"
            if col_name not in before_row.index:
                continue

            d_before = before_row[col_name]
            d_after = after_row[col_name]
            if pd.isna(d_before) or pd.isna(d_after) or d_before <= 0:
                continue

            consumed = max(d_before - d_after, 0)
            pct_consumed = consumed / d_before * 100
            breached = pct_consumed > 50

            level_analysis.append({
                "level_pct": pct,
                "depth_before": d_before,
                "depth_after": d_after,
                "consumed": consumed,
                "consumed_pct": pct_consumed,
                "breached": breached,
            })

        if not level_analysis:
            skipped += 1
            continue

        # 统计
        breached_levels = [la["level_pct"] for la in level_analysis if la["breached"]]
        max_breached = max(breached_levels) if breached_levels else 0
        total_consumed = sum(la["consumed"] for la in level_analysis)

        # 价格信息
        close_price = before_row.get("close", np.nan)
        high_price = event_rows["high"].max() if "high" in event_rows else np.nan
        low_price = event_rows["low"].min() if "low" in event_rows else np.nan

        results.append({
            "pair": pair_name,
            "timestamp": event_ts,
            "event_type": event_type,
            "magnitude_pct": magnitude,
            "side": side_cn,
            "close_price": close_price,
            "high_price": high_price,
            "low_price": low_price,
            "max_breached_level_pct": max_breached,
            "l2_breached": len(breached_levels) > 0,
            "n_levels_breached": len(breached_levels),
            "total_consumed_usdt": total_consumed,
            "level_details": level_analysis,
        })

    if not results:
        print(f"\n  {pair_name}: No events with analyzable depth data (skipped {skipped})")
        return None

    results.sort(key=lambda x: (x["max_breached_level_pct"], x["magnitude_pct"]), reverse=True)

    _print_report(pair_name, results, skipped)
    return results


def _print_report(pair_name, results, skipped):
    n = len(results)
    breached = sum(1 for r in results if r["l2_breached"])
    up = sum(1 for r in results if r["event_type"] == "up_wick")
    down = sum(1 for r in results if r["event_type"] == "down_wick")

    print(f"\n  {pair_name}: {n} events analyzed ({skipped} skipped) | "
          f"L2 breached: {breached} | Up: {up} Down: {down}")
    print(f"  {'='*77}")

    # Top events table
    print(f"  {'#':>3} {'Time':<20} {'Type':>9} {'Mag%':>6} {'Price':>10} "
          f"{'DepthBefore':>13} {'Consumed':>11} {'Breach':>10}")
    print(f"  {'-'*77}")

    for i, r in enumerate(results[:20]):
        breach_str = (",".join(f"{la['level_pct']}%" for la in r["level_details"] if la["breached"])
                      if r["l2_breached"] else "-")
        ts_str = str(r["timestamp"]).replace("  ", " ")
        print(f"  {i+1:3d} {ts_str:<20} {r['event_type']:>9} {r['magnitude_pct']:5.3f}% "
              f"{r['close_price']:>9,.2f} {r['level_details'][-1]['depth_before']:>12,.0f} "
              f"{r['total_consumed_usdt']:>10,.0f} {breach_str:>10}")

    # Detailed breach analysis
    breached_events = [r for r in results if r["l2_breached"]]
    if breached_events:
        print(f"\n  {'='*77}")
        print(f"  L2 Breach Details ({len(breached_events)} events)")
        print(f"  {'='*77}")
        for r in breached_events:
            print(f"\n  {r['timestamp']} | {r['event_type']} | {r['side']} "
                  f"| mag={r['magnitude_pct']:.3f}%")
            print(f"  Price: {r['close_price']:,.2f} | "
                  f"H={r['high_price']:,.2f} L={r['low_price']:,.2f}")
            for la in r["level_details"]:
                if la["breached"]:
                    tag = "BREACHED"
                elif la["consumed_pct"] > 20:
                    tag = "heavy"
                elif la["consumed_pct"] > 5:
                    tag = "consumed"
                else:
                    tag = ""
                if tag:
                    print(f"    {tag:>8} {la['level_pct']:5.1f}%: "
                          f"{la['depth_before']:>12,.0f} -> {la['depth_after']:>12,.0f} "
                          f"({la['consumed']:>10,.0f} USDT, {la['consumed_pct']:5.1f}%)")

    # Large consumption without breach
    large = [r for r in results if not r["l2_breached"] and r["total_consumed_usdt"] > 1000]
    if large:
        print(f"\n  {'='*77}")
        print(f"  Large consumption without breach ({len(large)} events)")
        print(f"  {'='*77}")
        for r in large[:10]:
            max_pct = max(la["consumed_pct"] for la in r["level_details"])
            print(f"  {r['timestamp']} {r['event_type']:>9} | "
                  f"consumed {r['total_consumed_usdt']:>10,.0f} USDT | "
                  f"max level {max_pct:.1f}%")

    # Hour distribution
    hours = {}
    for r in results:
        h = r["timestamp"].strftime("%m-%d %H:00")
        hours.setdefault(h, {"n": 0, "b": 0, "max_mag": 0, "max_c": 0})
        hours[h]["n"] += 1
        if r["l2_breached"]:
            hours[h]["b"] += 1
        hours[h]["max_mag"] = max(hours[h]["max_mag"], r["magnitude_pct"])
        hours[h]["max_c"] = max(hours[h]["max_c"], r["total_consumed_usdt"])

    print(f"\n  {'='*77}")
    print(f"  Hour Distribution")
    print(f"  {'='*77}")
    for h in sorted(hours):
        hs = hours[h]
        print(f"  {h} | events:{hs['n']:>3} breached:{hs['b']:>3} | "
              f"max_mag:{hs['max_mag']:.3f}% max_consume:{hs['max_c']:,.0f}")


def main():
    all_results = {}
    for pair in ["BTC_USDT", "ETH_USDT"]:
        r = analyze_pair(pair)
        if r:
            all_results[pair] = r

    # Summary
    print(f"\n{'='*77}")
    print(f"  BTC vs ETH Summary")
    print(f"{'='*77}")
    for pair, results in all_results.items():
        n = len(results)
        breached = sum(1 for r in results if r["l2_breached"])
        avg_mag = np.mean([r["magnitude_pct"] for r in results])
        max_mag = max(r["magnitude_pct"] for r in results)
        total_c = sum(r["total_consumed_usdt"] for r in results)
        avg_c = np.mean([r["total_consumed_usdt"] for r in results])
        max_c = max(r["total_consumed_usdt"] for r in results)

        print(f"\n  {pair}:")
        print(f"    Events: {n} | L2 breached: {breached} ({breached/max(n,1)*100:.0f}%)")
        print(f"    Magnitude: avg={avg_mag:.3f}% max={max_mag:.3f}%")
        print(f"    Depth consumed: avg={avg_c:,.0f} max={max_c:,.0f} total={total_c:,.0f} USDT")
        if breached > 0:
            max_breach = max(r["max_breached_level_pct"] for r in results)
            print(f"    Deepest breach: {max_breach}% level")

    print()


if __name__ == "__main__":
    main()
