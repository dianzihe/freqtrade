from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "user_data/strategies/Meme_波动率网格_马丁.py"
DATA_DIR = ROOT / "user_data/data/gate"
OUT_DIR = ROOT / "user_data/backtest_results"
sys.path.insert(0, str(ROOT))

PAIRS = {
    "stable": ["BTC", "ETH", "SOL", "XRP", "LTC", "HYPE"],
    "mid": ["XCN", "IP", "BAS", "PEAQ", "TA"],
    "meme": ["H", "VELVET", "BEAT", "COAI", "ALLO", "DN", "STG"],
}


@dataclass
class Position:
    pair: str
    side: str
    open_time: pd.Timestamp
    entry_price: float
    stake: float
    amount: float
    entries: int = 1
    max_profit: float = 0.0
    min_profit: float = 0.0
    reduced: bool = False
    entry_tag: str = ""


@dataclass
class SimConfig:
    fee: float = 0.001
    slippage: float = 0.0005
    wallet_start: float = 10000.0
    stake_amount: float = 100.0
    max_open_trades: int = 4
    long_only: bool = False
    enable_dca: bool = True
    enable_reduce: bool = True
    enable_trailing: bool = True


class OfflineBacktester:
    def __init__(self, config: SimConfig, data: pd.DataFrame | None = None) -> None:
        self.config = config
        self.strategy = self._load_strategy()
        self.data = data
        self.wallet = config.wallet_start
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.cooldown_until: pd.Timestamp | None = None

    @staticmethod
    def _load_strategy():
        sys.path.insert(0, str(STRATEGY_PATH.parent))
        spec = importlib.util.spec_from_file_location("meme_vol_grid_strategy", STRATEGY_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.MemeVolatilityGridMartingaleStrategy(
            config={"dry_run_wallet": 10000, "stake_amount": 100}
        )

    def load_data(self) -> pd.DataFrame:
        frames = []
        for group, symbols in PAIRS.items():
            for symbol in symbols:
                path = DATA_DIR / f"{symbol}_USDT-1m.feather"
                if not path.exists():
                    continue
                df = pd.read_feather(path)
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df = df[(df["date"] >= "2026-06-24") & (df["date"] < "2026-06-29")].copy()
                df = self.strategy.populate_indicators(df, {"pair": f"{symbol}/USDT"})
                df = self.strategy.populate_entry_trend(df, {"pair": f"{symbol}/USDT"})
                df = self.strategy.populate_exit_trend(df, {"pair": f"{symbol}/USDT"})
                df["pair"] = symbol
                df["group"] = group
                df["next_open"] = df["open"].shift(-1)
                df["next_date"] = df["date"].shift(-1)
                frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        data = data.dropna(subset=["next_open"]).sort_values(["date", "pair"]).reset_index(drop=True)
        return data

    def run(self) -> dict:
        data = self.data if self.data is not None else self.load_data()
        records = data.sort_values(["date", "pair"]).to_dict("records")
        last_rows = {str(row["pair"]): row for row in data.groupby("pair").tail(1).to_dict("records")}

        current_time = None
        rows_for_mark: list[tuple[str, pd.Series, pd.Series]] = []
        for row in records:
            time = row["date"]
            if current_time is not None and time != current_time:
                self._mark_equity(current_time, rows_for_mark)
                rows_for_mark = []
            current_time = time
            pair = str(row["pair"])
            self._manage_position(pair, row, row)
            self._maybe_enter(pair, row, row)
            rows_for_mark.append((pair, row, row))

        if current_time is not None:
            self._mark_equity(current_time, rows_for_mark)

        for pair, pos in list(self.positions.items()):
            row = last_rows[pair]
            self._close(pos, row["date"], float(row["close"]), "force_end")

        return self._summary()

    def _entry_price(self, side: str, raw_price: float) -> float:
        mult = 1 + self.config.slippage if side == "long" else 1 - self.config.slippage
        return raw_price * mult

    def _exit_price(self, side: str, raw_price: float) -> float:
        mult = 1 - self.config.slippage if side == "long" else 1 + self.config.slippage
        return raw_price * mult

    @staticmethod
    def _profit_ratio(pos: Position, price: float) -> float:
        if pos.side == "long":
            return price / pos.entry_price - 1
        return pos.entry_price / price - 1

    def _pnl(self, pos: Position, price: float) -> float:
        if pos.side == "long":
            gross = pos.amount * (price - pos.entry_price)
        else:
            gross = pos.amount * (pos.entry_price - price)
        fees = (pos.stake + pos.amount * price) * self.config.fee
        return gross - fees

    def _manage_position(self, pair: str, row: pd.Series, next_row: pd.Series) -> None:
        pos = self.positions.get(pair)
        if pos is None:
            return
        time = row["date"]
        exit_price = self._exit_price(pos.side, float(row["next_open"]))
        current_price = float(row["close"])
        profit = self._profit_ratio(pos, current_price)
        pos.max_profit = max(pos.max_profit, profit)
        pos.min_profit = min(pos.min_profit, profit)
        bars = int((time - pos.open_time).total_seconds() // 60)

        if profit <= -self.strategy.hard_stoploss_pct:
            self._close(pos, time, exit_price, "hard_stoploss")
            return
        if profit <= self.strategy.interrupt_loss_after_dca and pos.entries > len(self.strategy.dca_thresholds):
            self._close(pos, time, exit_price, "failed_dca_interrupt")
            return
        if bars >= self.strategy.time_stop_candles and profit < self.strategy.time_stop_min_profit:
            self._close(pos, time, exit_price, "time_stop")
            return
        if self.config.enable_trailing and pos.max_profit >= self.strategy.trailing_stop_positive_offset:
            if profit <= self.strategy.trailing_stop_positive:
                self._close(pos, time, exit_price, "trailing_stop")
                return
        if row["extreme_candle"] == 1:
            self._close(pos, time, exit_price, "extreme_risk")
            return
        if pos.side == "long" and row["exit_long"] == 1:
            self._close(pos, time, exit_price, str(row.get("exit_tag", "signal_exit")))
            return
        if pos.side == "short" and row["exit_short"] == 1:
            self._close(pos, time, exit_price, str(row.get("exit_tag", "signal_exit")))
            return

        if self.config.enable_reduce and profit <= self.strategy.floating_reduce_profit and not pos.reduced:
            reduce_stake = pos.stake * self.strategy.floating_reduce_fraction
            frac = reduce_stake / pos.stake
            realized = self._pnl(
                Position(pos.pair, pos.side, pos.open_time, pos.entry_price, reduce_stake, pos.amount * frac),
                exit_price,
            )
            pos.stake -= reduce_stake
            pos.amount *= 1 - frac
            pos.reduced = True
            self.wallet += realized

        if self.config.enable_dca and pos.entries <= len(self.strategy.dca_thresholds):
            threshold = self.strategy.dca_thresholds[pos.entries - 1]
            if profit <= threshold and profit > self.strategy.interrupt_loss_after_dca:
                if bars >= self.strategy.dca_min_candle_gap * pos.entries:
                    add_stake = min(
                        self.config.stake_amount * self.strategy.dca_multipliers[pos.entries - 1],
                        self.strategy.max_trade_notional,
                    )
                    if self._open_notional() + add_stake <= self.strategy.max_total_notional:
                        price = self._entry_price(pos.side, float(row["next_open"]))
                        add_amount = add_stake / price
                        old_value = pos.amount * pos.entry_price
                        pos.entry_price = (old_value + add_amount * price) / (pos.amount + add_amount)
                        pos.amount += add_amount
                        pos.stake += add_stake
                        pos.entries += 1

    def _maybe_enter(self, pair: str, row: pd.Series, next_row: pd.Series) -> None:
        if pair in self.positions or len(self.positions) >= self.config.max_open_trades:
            return
        time = row["date"]
        if self.cooldown_until is not None and time < self.cooldown_until:
            return
        side = None
        tag = ""
        if row["enter_long"] == 1:
            side = "long"
            tag = str(row.get("enter_tag", "long"))
        elif not self.config.long_only and row["enter_short"] == 1:
            side = "short"
            tag = str(row.get("enter_tag", "short"))
        if side is None:
            return

        equity = max(self.wallet, 1)
        risk_budget = equity * self.strategy.single_trade_risk_pct
        risk_stake = risk_budget / self.strategy.hard_stoploss_pct
        reserve = 1 + sum(self.strategy.dca_multipliers)
        stake = min(self.config.stake_amount, risk_stake / reserve, self.strategy.max_trade_notional)
        if self._open_notional() + stake > self.strategy.max_total_notional:
            return
        price = self._entry_price(side, float(row["next_open"]))
        amount = stake / price
        self.positions[pair] = Position(pair, side, time, price, stake, amount, entry_tag=tag)

    def _close(self, pos: Position, time: pd.Timestamp, price: float, reason: str) -> None:
        pnl = self._pnl(pos, price)
        profit_ratio = pnl / pos.stake if pos.stake else 0
        self.wallet += pnl
        self.trades.append(
            {
                "pair": pos.pair,
                "side": pos.side,
                "open_time": str(pos.open_time),
                "close_time": str(time),
                "duration_min": int((time - pos.open_time).total_seconds() // 60),
                "entry_price": pos.entry_price,
                "exit_price": price,
                "stake": pos.stake,
                "entries": pos.entries,
                "pnl": pnl,
                "profit_ratio": profit_ratio,
                "max_profit": pos.max_profit,
                "min_profit": pos.min_profit,
                "exit_reason": reason,
                "entry_tag": pos.entry_tag,
            }
        )
        self.positions.pop(pos.pair, None)

    def _open_notional(self) -> float:
        return sum(p.stake for p in self.positions.values())

    def _mark_equity(self, time: pd.Timestamp, rows_now: list[tuple[str, pd.Series, pd.Series]]) -> None:
        price_map = {pair: float(row["close"]) for pair, row, _ in rows_now}
        unreal = 0.0
        for pair, pos in self.positions.items():
            if pair in price_map:
                unreal += self._pnl(pos, self._exit_price(pos.side, price_map[pair]))
        self.equity_curve.append({"date": str(time), "equity": self.wallet + unreal, "open_trades": len(self.positions)})

    def _summary(self) -> dict:
        trades = pd.DataFrame(self.trades)
        equity = pd.DataFrame(self.equity_curve)
        if trades.empty:
            return {"trades": [], "summary": {"total_trades": 0}}
        returns = trades["pnl"] / self.config.wallet_start
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        eq = equity["equity"].astype(float)
        peak = eq.cummax()
        dd = eq / peak - 1
        daily = equity.assign(date=pd.to_datetime(equity["date"])).set_index("date")["equity"].resample("1D").last().pct_change().dropna()
        profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if abs(losses["pnl"].sum()) > 0 else math.inf
        expectancy = trades["pnl"].mean()
        gross = trades["pnl"].sum()
        max_consec_loss = self._max_consecutive_losses(trades["pnl"].tolist())
        summary = {
            "total_trades": int(len(trades)),
            "total_profit": float(gross),
            "total_profit_pct": float(gross / self.config.wallet_start),
            "max_drawdown_pct": float(dd.min()),
            "winrate": float(len(wins) / len(trades)),
            "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
            "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
            "payoff_ratio": float((wins["pnl"].mean() / abs(losses["pnl"].mean())) if len(wins) and len(losses) else 0.0),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "avg_duration_min": float(trades["duration_min"].mean()),
            "max_consecutive_losses": int(max_consec_loss),
            "sharpe_daily": float(daily.mean() / daily.std() * np.sqrt(365)) if len(daily) > 1 and daily.std() else 0.0,
            "sortino_daily": float(daily.mean() / daily[daily < 0].std() * np.sqrt(365)) if len(daily[daily < 0]) > 1 and daily[daily < 0].std() else 0.0,
            "calmar": float((gross / self.config.wallet_start) / abs(dd.min())) if dd.min() < 0 else 0.0,
            "by_pair": trades.groupby("pair")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum").to_dict("index"),
            "by_side": trades.groupby("side")["pnl"].agg(["count", "sum", "mean"]).to_dict("index"),
            "by_exit": trades.groupby("exit_reason")["pnl"].agg(["count", "sum", "mean"]).sort_values("sum").to_dict("index"),
            "top_winners": trades.sort_values("pnl", ascending=False).head(10).to_dict("records"),
            "top_losers": trades.sort_values("pnl").head(10).to_dict("records"),
        }
        return {"summary": summary, "trades": trades.to_dict("records"), "equity": self.equity_curve}

    @staticmethod
    def _max_consecutive_losses(pnls: list[float]) -> int:
        best = cur = 0
        for pnl in pnls:
            if pnl <= 0:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_data = OfflineBacktester(SimConfig()).load_data()
    scenarios = {
        "base": SimConfig(),
        "long_only": SimConfig(long_only=True),
        "no_dca": SimConfig(enable_dca=False),
        "no_reduce": SimConfig(enable_reduce=False),
        "fee_slip_stress": SimConfig(fee=0.0015, slippage=0.001),
    }
    results = {}
    for name, cfg in scenarios.items():
        result = OfflineBacktester(cfg, prepared_data).run()
        results[name] = result["summary"]
        (OUT_DIR / f"offline_meme_vol_grid_{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (OUT_DIR / "offline_meme_vol_grid_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
