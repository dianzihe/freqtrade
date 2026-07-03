from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from user_data.strategies.demon_fade_short_strategy import DemonFadeShortStrategy


DATA_DIR = PROJECT / "user_data/data/gate"
PAIR_FILE = PROJECT / "user_data/config/pairs-gate-spot-200.json"
OUTPUT_DIR = PROJECT / "deliverables/demon-fade-short"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OpenShort:
    pair: str
    entry_time: pd.Timestamp
    entry_price: float
    stake: float
    amount: float
    min_rate: float
    max_rate: float
    entry_tag: str


@dataclass
class ClosedShort:
    pair: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    profit_pct: float
    profit_usd: float
    hold_hours: float
    max_profit_pct: float
    max_adverse_pct: float
    exit_reason: str
    entry_tag: str


def pair_to_file_stem(pair: str) -> str:
    return pair.replace("/", "_")


def protected_short_profit(peak_profit: float) -> float | None:
    if peak_profit >= 0.30:
        return 0.15
    if peak_profit >= 0.15:
        return 0.06
    if peak_profit >= 0.08:
        return 0.0
    return None


def build_signal_frames(strategy: DemonFadeShortStrategy, pairs: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        path = DATA_DIR / f"{pair_to_file_stem(pair)}-5m.feather"
        if not path.exists():
            continue
        df = pd.read_feather(path).sort_values("date").reset_index(drop=True)
        if len(df) < strategy.startup_candle_count + 2:
            continue
        df = strategy.populate_indicators(df, {"pair": pair})
        df = strategy.populate_entry_trend(df, {"pair": pair})
        frames[pair] = df
    return frames


def run_backtest() -> tuple[list[ClosedShort], dict[str, float]]:
    strategy = DemonFadeShortStrategy(config={})
    pairs = json.loads(PAIR_FILE.read_text())
    frames = build_signal_frames(strategy, pairs)

    timeline = sorted({ts for df in frames.values() for ts in df["date"].iloc[1:]})
    index_by_pair = {pair: df.set_index("date", drop=False) for pair, df in frames.items()}

    open_trades: list[OpenShort] = []
    closed: list[ClosedShort] = []
    pending_entries: list[tuple[pd.Timestamp, str, str]] = []

    stake = 1.0
    fee_rate = 0.002
    max_open = 5
    hard_stop = abs(float(strategy.stoploss))

    for ts in timeline:
        still_open: list[OpenShort] = []
        for trade in open_trades:
            df = index_by_pair[trade.pair]
            if ts not in df.index:
                still_open.append(trade)
                continue

            row = df.loc[ts]
            trade.min_rate = min(trade.min_rate, float(row["low"]))
            trade.max_rate = max(trade.max_rate, float(row["high"]))
            peak_profit = trade.entry_price / trade.min_rate - 1.0
            adverse = trade.max_rate / trade.entry_price - 1.0

            stop_price = trade.entry_price * (1.0 + hard_stop)
            protected = protected_short_profit(peak_profit)
            if protected is not None:
                stop_price = min(stop_price, trade.entry_price / (1.0 + protected))

            exit_reason = ""
            exit_price = float(row["close"])
            if float(row["high"]) >= stop_price:
                exit_reason = "stop_or_ratchet"
                exit_price = stop_price
            else:
                age_h = (ts - trade.entry_time).total_seconds() / 3600.0
                current_profit = trade.entry_price / float(row["close"]) - 1.0
                if age_h >= 4 and current_profit <= -0.035:
                    exit_reason = "failed_fade"
                elif age_h >= 12 and peak_profit < 0.03:
                    exit_reason = "stale_no_followthrough"
                elif age_h >= 24 and current_profit > 0.04:
                    exit_reason = "time_take_profit"
                elif age_h >= 36:
                    exit_reason = "max_hold"

            if exit_reason:
                gross = trade.amount * (2.0 * trade.entry_price - exit_price)
                entry_fee = trade.stake * fee_rate
                exit_notional = trade.amount * exit_price
                exit_fee = exit_notional * fee_rate
                profit_usd = gross - trade.stake - entry_fee - exit_fee
                profit_pct = profit_usd / trade.stake * 100.0
                closed.append(
                    ClosedShort(
                        pair=trade.pair,
                        entry_time=trade.entry_time.isoformat(),
                        exit_time=ts.isoformat(),
                        entry_price=trade.entry_price,
                        exit_price=exit_price,
                        profit_pct=profit_pct,
                        profit_usd=profit_usd,
                        hold_hours=(ts - trade.entry_time).total_seconds() / 3600.0,
                        max_profit_pct=peak_profit * 100.0,
                        max_adverse_pct=adverse * 100.0,
                        exit_reason=exit_reason,
                        entry_tag=trade.entry_tag,
                    )
                )
            else:
                still_open.append(trade)
        open_trades = still_open

        due = [entry for entry in pending_entries if entry[0] == ts]
        pending_entries = [entry for entry in pending_entries if entry[0] != ts]
        for _, pair, tag in due:
            if len(open_trades) >= max_open or any(t.pair == pair for t in open_trades):
                continue
            df = index_by_pair[pair]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            entry_price = float(row["open"])
            open_trades.append(
                OpenShort(
                    pair=pair,
                    entry_time=ts,
                    entry_price=entry_price,
                    stake=stake,
                    amount=stake / entry_price,
                    min_rate=entry_price,
                    max_rate=entry_price,
                    entry_tag=tag,
                )
            )

        for pair, df in index_by_pair.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            if int(row.get("enter_short", 0)) != 1:
                continue
            next_rows = df[df["date"] > ts]
            if next_rows.empty:
                continue
            next_ts = next_rows.iloc[0]["date"]
            pending_entries.append((next_ts, pair, str(row.get("enter_tag", ""))))

    if closed:
        profits = pd.Series([t.profit_pct for t in closed])
        wins = profits[profits > 0]
        losses = profits[profits <= 0]
        profit_factor = (
            wins.sum() / abs(losses.sum())
            if abs(losses.sum()) > 0
            else float("inf")
        )
        summary = {
            "pairs_loaded": float(len(frames)),
            "total_trades": float(len(closed)),
            "win_rate_pct": float((profits > 0).mean() * 100.0),
            "avg_profit_pct": float(profits.mean()),
            "median_profit_pct": float(profits.median()),
            "best_profit_pct": float(profits.max()),
            "worst_profit_pct": float(profits.min()),
            "profit_factor": float(profit_factor),
            "net_profit_usd": float(sum(t.profit_usd for t in closed)),
            "avg_hold_hours": float(pd.Series([t.hold_hours for t in closed]).mean()),
            "open_trades_left": float(len(open_trades)),
        }
    else:
        summary = {
            "pairs_loaded": float(len(frames)),
            "total_trades": 0.0,
            "open_trades_left": float(len(open_trades)),
        }

    return closed, summary


def main() -> None:
    closed, summary = run_backtest()
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([asdict(t) for t in closed]).to_csv(OUTPUT_DIR / "trades.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
