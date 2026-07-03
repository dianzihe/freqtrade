from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pandas as pd


BASE = Path("user_data/orderbook_data/gate/spot/BTC_USDT")
OUT = Path("user_data/backtest_results/microstructure_l2_signal_report.csv")
SUMMARY = Path("user_data/backtest_results/microstructure_l2_signal_summary.txt")


def apply_updates(book: dict[float, float], updates: str) -> None:
    for price_s, amount_s in orjson.loads(updates):
        price = float(price_s)
        amount = float(amount_s)
        if amount <= 0:
            book.pop(price, None)
        else:
            book[price] = amount


def top_depths(
    bids: dict[float, float],
    asks: dict[float, float],
    levels: int = 25,
) -> tuple[float, float, float, float]:
    top_bids = sorted(bids.items(), key=lambda item: item[0], reverse=True)[:levels]
    top_asks = sorted(asks.items(), key=lambda item: item[0])[:levels]
    best_bid = top_bids[0][0] if top_bids else np.nan
    best_ask = top_asks[0][0] if top_asks else np.nan
    return (
        sum(amount for _, amount in top_bids),
        sum(amount for _, amount in top_asks),
        best_bid,
        best_ask,
    )


def add_signals(
    df: pd.DataFrame,
    lookback: int = 24,
    vol_window: int = 20,
    percentile: int = 88,
    n_confirm: int = 2,
    min_floor: float = 0.40,
    valid_bars: int = 25,
) -> pd.DataFrame:
    df = df.copy()
    returns = df["close"].pct_change()
    short_w = max(4, vol_window // 4)
    mid_w = max(8, vol_window // 2)
    vol_s = returns.rolling(short_w).std()
    vol_m = returns.rolling(mid_w).std()
    vol_l = returns.rolling(vol_window).std()
    vol_sum = vol_s + vol_m + vol_l
    p_s = vol_s / vol_sum.replace(0, np.nan)
    p_m = vol_m / vol_sum.replace(0, np.nan)
    p_l = vol_l / vol_sum.replace(0, np.nan)
    eps = 1e-12
    entropy = -(
        p_s * np.log(p_s.clip(lower=eps))
        + p_m * np.log(p_m.clip(lower=eps))
        + p_l * np.log(p_l.clip(lower=eps))
    )
    df["ch1_vol_entropy"] = (entropy / np.log(3)).fillna(0.0)

    total_depth = df["bid_depth_25"] + df["ask_depth_25"]
    depth_mean = total_depth.rolling(window=lookback * 2, min_periods=lookback).mean()
    depth_std = total_depth.rolling(window=lookback * 2, min_periods=lookback).std()
    depth_z = ((total_depth - depth_mean) / depth_std.replace(0, 1e-10)).fillna(0.0)
    df["ch2_depth_erosion"] = (-depth_z).clip(lower=0)

    mid = (df["best_bid"] + df["best_ask"]) / 2.0
    spread_ratio = ((df["best_ask"] - df["best_bid"]) / mid.replace(0, np.nan)).fillna(0.0)
    spread_mean = spread_ratio.rolling(window=lookback * 2, min_periods=lookback).mean()
    spread_std = spread_ratio.rolling(window=lookback * 2, min_periods=lookback).std()
    spread_z = ((spread_ratio - spread_mean) / spread_std.replace(0, 1e-10)).fillna(0.0)
    df["ch3_spread_drift"] = spread_z.clip(lower=0)

    # This order-book capture has no trade-side buy/sell volume, so strict Ch4 net flow is neutral.
    df["ch4_order_flow"] = 0.0

    channels = [
        "ch1_vol_entropy",
        "ch2_depth_erosion",
        "ch3_spread_drift",
        "ch4_order_flow",
    ]
    strength = df[channels].copy()
    strength["ch4_order_flow"] = strength["ch4_order_flow"].abs()
    df["composite_raw"] = strength.max(axis=1)
    df["composite_smooth"] = df["composite_raw"].rolling(window=3, min_periods=1).mean()

    rising = df["composite_smooth"] > df["composite_smooth"].shift(1)
    rising_streak = rising.copy()
    for lag in range(2, n_confirm + 1):
        rising_streak = rising_streak & (
            df["composite_smooth"] > df["composite_smooth"].shift(lag)
        )

    long_w = max(lookback * 4, 48)
    effective = (
        df["composite_smooth"]
        .rolling(long_w, min_periods=lookback * 2)
        .quantile(percentile / 100.0)
        .clip(lower=min_floor)
    )
    trigger = (
        (df["composite_smooth"] > effective)
        & rising_streak
        & (df["composite_smooth"] > min_floor)
    ).fillna(False)

    trigger_values = trigger.astype(int).to_numpy()
    last_trigger = -valid_bars - 1
    for pos, value in enumerate(trigger_values):
        if value and pos - last_trigger <= valid_bars:
            trigger_values[pos] = 0
        elif value:
            last_trigger = pos

    df["adaptive_threshold"] = effective
    df["signal_trigger"] = trigger_values
    return df


def build_book_1m() -> pd.DataFrame:
    l2_files = sorted((BASE / "l2").rglob("*.parquet"))
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows: list[dict[str, object]] = []
    current_minute = None
    row_count = 0

    print(f"L2 files: {len(l2_files)}", flush=True)
    for file_path in l2_files:
        data = pd.read_parquet(
            file_path,
            columns=["exchange_time_ms", "bid_updates", "ask_updates"],
        )
        for row in data.itertuples(index=False):
            minute = pd.to_datetime(int(row.exchange_time_ms), unit="ms", utc=True).floor("min")
            if current_minute is None:
                current_minute = minute
            elif minute != current_minute:
                bid_depth, ask_depth, best_bid, best_ask = top_depths(bids, asks)
                rows.append(
                    {
                        "date": current_minute,
                        "bid_depth_25": bid_depth,
                        "ask_depth_25": ask_depth,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                    }
                )
                current_minute = minute

            apply_updates(bids, row.bid_updates)
            apply_updates(asks, row.ask_updates)
            row_count += 1

        print(f"processed {file_path.name}: rows={row_count:,}, minutes={len(rows):,}", flush=True)

    if current_minute is not None:
        bid_depth, ask_depth, best_bid, best_ask = top_depths(bids, asks)
        rows.append(
            {
                "date": current_minute,
                "bid_depth_25": bid_depth,
                "ask_depth_25": ask_depth,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }
        )

    return pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")


def build_price_1m() -> pd.DataFrame:
    l1_files = sorted((BASE / "l1").rglob("*.parquet"))
    l1 = pd.concat(
        [pd.read_parquet(path, columns=["exchange_time_ms", "mid_price"]) for path in l1_files],
        ignore_index=True,
    )
    l1["ts"] = pd.to_datetime(l1["exchange_time_ms"], unit="ms", utc=True)
    l1 = l1.dropna(subset=["mid_price"]).sort_values("ts")
    price_1m = (
        l1.set_index("ts")["mid_price"]
        .resample("1min")
        .ohlc()
        .dropna()
        .reset_index()
        .rename(columns={"ts": "date"})
    )
    price_1m["volume"] = 1.0
    return price_1m


def main() -> None:
    book_1m = build_book_1m()
    print(
        f"book_1m {len(book_1m):,}: {book_1m['date'].min()} -> {book_1m['date'].max()}",
        flush=True,
    )

    price_1m = build_price_1m()
    print(
        f"price_1m {len(price_1m):,}: {price_1m['date'].min()} -> {price_1m['date'].max()}",
        flush=True,
    )

    df = pd.merge(price_1m, book_1m, on="date", how="inner").sort_values("date").reset_index(
        drop=True
    )
    print(f"merged minutes {len(df):,}: {df['date'].min()} -> {df['date'].max()}", flush=True)

    sig = add_signals(df)
    sig["future_close_15m"] = sig["close"].shift(-15)
    sig["ret_15m_pct"] = (sig["future_close_15m"] / sig["close"] - 1.0) * 100.0
    future_high = pd.concat([sig["high"].shift(-offset) for offset in range(1, 16)], axis=1).max(
        axis=1
    )
    future_low = pd.concat([sig["low"].shift(-offset) for offset in range(1, 16)], axis=1).min(
        axis=1
    )
    sig["max_up_15m_pct"] = (future_high / sig["close"] - 1.0) * 100.0
    sig["max_down_15m_pct"] = (future_low / sig["close"] - 1.0) * 100.0

    columns = [
        "date",
        "close",
        "ch1_vol_entropy",
        "ch2_depth_erosion",
        "ch3_spread_drift",
        "ch4_order_flow",
        "composite_smooth",
        "adaptive_threshold",
        "future_close_15m",
        "ret_15m_pct",
        "max_up_15m_pct",
        "max_down_15m_pct",
    ]
    report = sig.loc[sig["signal_trigger"] == 1, columns].dropna(
        subset=["future_close_15m"]
    )
    report = report.copy()
    for column in columns[1:]:
        report[column] = report[column].astype(float).round(6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT, index=False, encoding="utf-8-sig")

    summary_lines = [
        f"output={OUT.resolve()}",
        f"signals={len(report)}",
        "note=Ch4 is 0 because this L2/L1 capture has no signed trade volume.",
    ]
    if len(report):
        summary_lines.append("")
        summary_lines.append(report.to_string(index=False, max_rows=300))
        summary_lines.append("")
        summary_lines.append("Summary:")
        summary_lines.append(
            report[["ret_15m_pct", "max_up_15m_pct", "max_down_15m_pct"]]
            .describe()
            .to_string()
        )
    SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()
