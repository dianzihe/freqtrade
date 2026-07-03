from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.strategies.lottery_ticket_momentum_strategy import LotteryTicketMomentumStrategy


@dataclass
class SimTrade:
    pair: str
    entry_time: str
    exit_time: str
    entry_rate: float
    exit_rate: float
    stake: float
    profit_abs: float
    profit_pct: float
    enter_tag: str
    exit_reason: str


def _load_pairs(path: Path) -> list[str]:
    return [pair for pair in json.loads(path.read_text(encoding="utf-8")) if pair.endswith("/USDT")]


def _prepare_pair(strategy: LotteryTicketMomentumStrategy, data_dir: Path, pair: str) -> pd.DataFrame | None:
    file_name = f"{pair.replace('/', '_')}-5m.feather"
    path = data_dir / file_name
    if not path.exists():
        return None

    df = pd.read_feather(path).sort_values("date").reset_index(drop=True)
    if len(df) <= strategy.startup_candle_count:
        return None

    df = strategy.populate_indicators(df, {"pair": pair})
    df = strategy.populate_entry_trend(df, {"pair": pair})
    df = strategy.populate_exit_trend(df, {"pair": pair})

    # Match Freqtrade's anti-lookahead behavior: act on the previous closed candle.
    for col in ["enter_long", "exit_long", "enter_tag", "exit_tag"]:
        if col not in df.columns:
            df[col] = "" if col.endswith("tag") else 0
        df[col] = df[col].shift(1)

    df = df.iloc[strategy.startup_candle_count + 1 :].copy()
    df["pair"] = pair
    df["signal_score"] = (
        df["ret_24h"].clip(lower=0)
        * df["volume_ratio"].clip(lower=0)
        * (1 + df["range_24h"].clip(lower=0))
        * (1 + df["off_high_24h"].clip(lower=-0.95))
    )
    return df


def _exit_trade(trade: dict, row: pd.Series, fee_rate: float) -> SimTrade | None:
    entry = trade["entry_rate"]
    minutes = int((row["date"] - trade["entry_time"]).total_seconds() // 60)
    current_profit = float(row["open"]) / entry - 1

    stop_rate = entry * 0.82
    if row["low"] <= stop_rate:
        return _finalize(trade, row, stop_rate, fee_rate, "stoploss")

    if minutes >= LotteryTicketMomentumStrategy.max_hold_minutes:
        return _finalize(trade, row, float(row["open"]), fee_rate, "max_4d_hold")
    if (
        minutes >= LotteryTicketMomentumStrategy.weak_profit_minutes
        and current_profit < LotteryTicketMomentumStrategy.weak_profit
    ):
        return _finalize(trade, row, float(row["open"]), fee_rate, "stale_weak_profit")
    if (
        minutes >= LotteryTicketMomentumStrategy.not_started_minutes
        and current_profit < LotteryTicketMomentumStrategy.not_started_profit
    ):
        return _finalize(trade, row, float(row["open"]), fee_rate, "stale_not_started")

    trade["highest_rate"] = max(trade["highest_rate"], float(row["high"]))
    if trade["highest_rate"] >= entry * 1.20:
        trail_rate = trade["highest_rate"] * 0.92
        if row["low"] <= trail_rate:
            return _finalize(trade, row, trail_rate, fee_rate, "trailing_profit")

    roi_targets = [(360, 1.00, "time_breakeven"), (120, 1.12, "roi_12"), (45, 1.25, "roi_25"), (0, 1.60, "roi_60")]
    for min_minutes, multiplier, reason in reversed(roi_targets):
        if minutes >= min_minutes and row["high"] >= entry * multiplier:
            return _finalize(trade, row, entry * multiplier, fee_rate, reason)

    if int(row.get("exit_long") or 0) == 1:
        return _finalize(trade, row, float(row["open"]), fee_rate, str(row.get("exit_tag") or "exit_signal"))
    return None


def _finalize(trade: dict, row: pd.Series, exit_rate: float, fee_rate: float, reason: str) -> SimTrade:
    gross_return = exit_rate / trade["entry_rate"] - 1
    net_return = gross_return - (fee_rate * 2)
    profit_abs = trade["stake"] * net_return
    return SimTrade(
        pair=trade["pair"],
        entry_time=trade["entry_time"].isoformat(),
        exit_time=row["date"].isoformat(),
        entry_rate=trade["entry_rate"],
        exit_rate=float(exit_rate),
        stake=trade["stake"],
        profit_abs=profit_abs,
        profit_pct=net_return * 100,
        enter_tag=trade["enter_tag"],
        exit_reason=reason,
    )


def run_backtest(data_dir: Path, pairs_file: Path, output: Path, fee_rate: float = 0.002) -> dict:
    strategy = LotteryTicketMomentumStrategy(config={})
    pairs = _load_pairs(pairs_file)
    frames = [
        frame
        for pair in pairs
        if (frame := _prepare_pair(strategy, data_dir, pair)) is not None
    ]
    if not frames:
        raise RuntimeError("No usable 5m Gate data found.")

    data = pd.concat(frames, ignore_index=True).sort_values(["date", "signal_score"], ascending=[True, False])
    grouped = {ts: group for ts, group in data.groupby("date", sort=True)}

    open_trades: list[dict] = []
    closed: list[SimTrade] = []
    daily_counts: dict[date, int] = {}
    total_entries = 0

    for current_time, rows in grouped.items():
        still_open = []
        pair_rows = {row["pair"]: row for _, row in rows.iterrows()}
        for trade in open_trades:
            row = pair_rows.get(trade["pair"])
            if row is None:
                still_open.append(trade)
                continue
            exited = _exit_trade(trade, row, fee_rate)
            if exited is None:
                still_open.append(trade)
            else:
                closed.append(exited)
        open_trades = still_open

        if total_entries >= strategy.max_total_tickets:
            continue

        day = current_time.date()
        day_count = daily_counts.get(day, 0)
        if day_count >= strategy.max_daily_tickets:
            continue

        candidates = rows[rows["enter_long"].fillna(0).astype(int) == 1]
        for _, row in candidates.iterrows():
            if total_entries >= strategy.max_total_tickets:
                break
            if day_count >= strategy.max_daily_tickets:
                break
            if len(open_trades) >= strategy.max_open_trades:
                break
            if any(trade["pair"] == row["pair"] for trade in open_trades):
                continue

            open_trades.append(
                {
                    "pair": row["pair"],
                    "entry_time": current_time,
                    "entry_rate": float(row["open"]),
                    "stake": strategy.ticket_stake,
                    "enter_tag": str(row.get("enter_tag") or ""),
                    "highest_rate": float(row["high"]),
                }
            )
            total_entries += 1
            day_count += 1
            daily_counts[day] = day_count

    for trade in open_trades:
        pair_frame = data[data["pair"] == trade["pair"]]
        last_row = pair_frame.iloc[-1]
        closed.append(_finalize(trade, last_row, float(last_row["close"]), fee_rate, "left_open"))

    trades = [asdict(trade) for trade in closed]
    profit_abs = sum(trade.profit_abs for trade in closed)
    wins = sum(1 for trade in closed if trade.profit_abs > 0)
    losses = sum(1 for trade in closed if trade.profit_abs < 0)
    result = {
        "strategy": "LotteryTicketMomentumStrategy",
        "data_dir": str(data_dir),
        "pairs_file": str(pairs_file),
        "pairs_loaded": len(frames),
        "starting_balance": 20.0,
        "ticket_stake": strategy.ticket_stake,
        "max_daily_tickets": strategy.max_daily_tickets,
        "max_total_tickets": strategy.max_total_tickets,
        "fee_rate": fee_rate,
        "trades": trades,
        "summary": {
            "trades": len(closed),
            "wins": wins,
            "losses": losses,
            "winrate_pct": (wins / len(closed) * 100) if closed else 0.0,
            "profit_abs": profit_abs,
            "profit_pct_on_20_usdt": profit_abs / 20.0 * 100,
            "ending_balance": 20.0 + profit_abs,
            "daily_entries": {day.isoformat(): count for day, count in sorted(daily_counts.items())},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/gate"))
    parser.add_argument("--pairs-file", type=Path, default=Path("user_data/config/pairs-gate-spot-200.json"))
    parser.add_argument("--output", type=Path, default=Path("user_data/backtest_results/lottery_ticket_offline.json"))
    args = parser.parse_args()

    result = run_backtest(args.data_dir, args.pairs_file, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
