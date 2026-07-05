#!/usr/bin/env python3
"""Serve a live ETH LOB v7 dashboard from cex_tick_recorder parquet data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from user_data.strategies.lob_latent_regime_v7_strategy import LOBLatentRegimeV7Strategy


LOGGER = logging.getLogger("eth_lob_v7_dashboard")
SNAPSHOT_CACHE: dict[str, Any] = {"key": None, "value": None, "created": 0.0}
SNAPSHOT_LOCK = threading.Lock()

EXCHANGES: dict[str, dict[str, Any]] = {
    "gate": {
        "label": "Gate.io",
        "pair": "ETH_USDT",
        "dir": Path("user_data/orderbook_data/gate/spot/ETH_USDT"),
    },
    "binance": {
        "label": "Binance",
        "pair": "ETHUSDT",
        "dir": Path("user_data/orderbook_data/binance/spot/ETHUSDT"),
    },
}

REFRESH_MS = 60_000
DEFAULT_MINUTES = 1
SIGNAL_FREQ = "1s"
SIGNAL_REFRESH_MS = 1_000


def _json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _json_int(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _time_ms(ts: Any) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


def _iso_time(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def recent_parquets(
    base_dir: Path,
    dataset: str,
    minutes: int,
    *,
    live_only: bool = True,
    max_files: int | None = 2,
) -> list[Path]:
    dataset_dir = base_dir / dataset
    if not dataset_dir.exists():
        return []
    if live_only:
        cutoff = time.time() - (minutes + 10) * 60
        files = [p for p in dataset_dir.rglob("*.parquet") if p.stat().st_mtime >= cutoff]
    else:
        files = list(dataset_dir.rglob("*.parquet"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:max_files] if max_files is not None else files


def read_dataset(
    base_dir: Path,
    dataset: str,
    minutes: int,
    *,
    live_only: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    max_files = 2 if live_only else None
    for path in recent_parquets(base_dir, dataset, minutes, live_only=live_only, max_files=max_files):
        try:
            frames.append(pq.read_table(path).to_pandas())
        except Exception as exc:
            LOGGER.debug("Skipping %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    time_source = None
    if "exchange_time_ms" in df.columns and df["exchange_time_ms"].notna().any():
        time_source = "exchange_time_ms"
    elif "local_time_ms" in df.columns and df["local_time_ms"].notna().any():
        time_source = "local_time_ms"
    if time_source is None:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df[time_source], unit="ms", utc=True)
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(minutes=minutes)
    return df[df["time"] >= cutoff].sort_values("time")


def read_trades_resampled(
    base_dir: Path,
    minutes: int,
    *,
    freq: str,
    live_only: bool = True,
) -> pd.DataFrame:
    trades = read_dataset(base_dir, "trades", minutes, live_only=live_only)
    if trades.empty:
        return pd.DataFrame()

    trades = trades.set_index("time")
    ohlc = trades["price"].resample(freq).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = trades["amount"].resample(freq).sum()
    ohlc["trade_count"] = trades["price"].resample(freq).count()
    ohlc["buy_volume"] = trades["amount"].where(trades["side"] == "buy", 0.0).resample(freq).sum()
    ohlc["sell_volume"] = trades["amount"].where(trades["side"] == "sell", 0.0).resample(freq).sum()
    return ohlc.dropna(subset=["close"]).reset_index().rename(columns={"time": "date"})


def read_trades_1m(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    return read_trades_resampled(base_dir, minutes, freq="1min", live_only=live_only)


def read_trades_1s(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    return read_trades_resampled(base_dir, minutes, freq="1s", live_only=live_only)


def read_depth_resampled(
    base_dir: Path,
    minutes: int,
    *,
    freq: str,
    live_only: bool = True,
) -> pd.DataFrame:
    depth = read_dataset(base_dir, "depth", minutes, live_only=live_only)
    if depth.empty:
        return pd.DataFrame()

    wanted = [
        "bid_depth",
        "ask_depth",
        "total_depth",
        "bid_depth_5bps",
        "ask_depth_5bps",
        "total_depth_5bps",
        "bid_depth_10bps",
        "ask_depth_10bps",
        "total_depth_10bps",
        "bid_depth_25bps",
        "ask_depth_25bps",
        "total_depth_25bps",
        "best_bid",
        "best_ask",
        "spread",
        "mid_price",
        "imbalance",
        "bid_levels",
        "ask_levels",
    ]
    for col in wanted:
        if col not in depth.columns:
            depth[col] = np.nan

    agg = depth.set_index("time").resample(freq).agg(
        {
            "bid_depth": "mean",
            "ask_depth": "mean",
            "total_depth": "mean",
            "bid_depth_5bps": "mean",
            "ask_depth_5bps": "mean",
            "total_depth_5bps": "mean",
            "bid_depth_10bps": "mean",
            "ask_depth_10bps": "mean",
            "total_depth_10bps": "mean",
            "bid_depth_25bps": "mean",
            "ask_depth_25bps": "mean",
            "total_depth_25bps": "mean",
            "best_bid": "last",
            "best_ask": "last",
            "spread": "mean",
            "mid_price": "last",
            "imbalance": "mean",
            "bid_levels": "last",
            "ask_levels": "last",
        }
    )
    return agg.reset_index().rename(columns={"time": "date"})


def read_depth_1m(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    return read_depth_resampled(base_dir, minutes, freq="1min", live_only=live_only)


def read_depth_1s(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    return read_depth_resampled(base_dir, minutes, freq="1s", live_only=live_only)


def _entry_price_amount(entry: Any) -> tuple[float, float] | None:
    if isinstance(entry, dict):
        price = entry.get("price")
        amount = entry.get("amount")
    else:
        try:
            price = entry[0]
            amount = entry[1]
        except (TypeError, IndexError):
            return None
    try:
        return float(price), float(amount)
    except (TypeError, ValueError):
        return None


def _apply_l2_side(book_side: dict[float, float], entries: Any) -> None:
    if entries is None:
        return
    for entry in entries:
        parsed = _entry_price_amount(entry)
        if parsed is None:
            continue
        price, amount = parsed
        if amount <= 0:
            book_side.pop(price, None)
        else:
            book_side[price] = amount


def _top10_ladder(book_side: dict[float, float], *, reverse: bool) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    cumulative = 0.0
    for level, (price, amount) in enumerate(
        sorted(book_side.items(), reverse=reverse)[:10],
        start=1,
    ):
        cumulative += amount
        rows.append(
            {
                "level": level,
                "price": float(price),
                "amount": float(amount),
                "cumulative": float(cumulative),
            }
        )
    return rows


def build_l2_top10(l2: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    if l2.empty or "time" not in l2.columns:
        return pd.DataFrame()

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    by_minute: dict[pd.Timestamp, dict[str, Any]] = {}
    for _, row in l2.sort_values("time").iterrows():
        if bool(row.get("is_snapshot")):
            bids.clear()
            asks.clear()
        _apply_l2_side(bids, row.get("bid_updates"))
        _apply_l2_side(asks, row.get("ask_updates"))
        if not bids and not asks:
            continue

        bid_ladder = _top10_ladder(bids, reverse=True)
        ask_ladder = _top10_ladder(asks, reverse=False)
        bid_depth = sum(level["amount"] for level in bid_ladder)
        ask_depth = sum(level["amount"] for level in ask_ladder)
        bucket = pd.Timestamp(row["time"]).floor(freq)
        by_minute[bucket] = {
            "date": bucket,
            "top10_bid_depth": float(bid_depth),
            "top10_ask_depth": float(ask_depth),
            "top10_total_depth": float(bid_depth + ask_depth),
            "top10_levels": {"bids": bid_ladder, "asks": ask_ladder},
        }

    if not by_minute:
        return pd.DataFrame()
    return pd.DataFrame([by_minute[key] for key in sorted(by_minute)])


def build_l2_top10_1m(l2: pd.DataFrame) -> pd.DataFrame:
    return build_l2_top10(l2, freq="1min")


def build_l2_top10_1s(l2: pd.DataFrame) -> pd.DataFrame:
    return build_l2_top10(l2, freq="1s")


def read_l2_top10_1m(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    return build_l2_top10_1m(read_dataset(base_dir, "l2", minutes, live_only=live_only))


def read_l2_top10_1s(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    return build_l2_top10_1s(read_dataset(base_dir, "l2", minutes, live_only=live_only))


def attach_top10(signals: pd.DataFrame, top10: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or top10.empty:
        return signals
    merged = pd.merge_asof(
        signals.sort_values("date"),
        top10.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(seconds=60),
    )
    return merged


def compute_lob_v7_signals(ohlcv: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()

    df = ohlcv.copy()
    if not depth.empty:
        merge_cols = [
            "date",
            "bid_depth",
            "ask_depth",
            "total_depth",
            "bid_depth_5bps",
            "ask_depth_5bps",
            "total_depth_5bps",
            "bid_depth_10bps",
            "ask_depth_10bps",
            "total_depth_10bps",
            "bid_depth_25bps",
            "ask_depth_25bps",
            "total_depth_25bps",
            "best_bid",
            "best_ask",
            "spread",
            "mid_price",
            "imbalance",
            "bid_levels",
            "ask_levels",
        ]
        df = df.merge(depth[[c for c in merge_cols if c in depth.columns]], on="date", how="left")
        for col in [c for c in merge_cols if c != "date" and c in df.columns]:
            df[col] = df[col].ffill(limit=5)

    strategy = LOBLatentRegimeV7Strategy(config={"candle_type_def": "spot"})
    signals = strategy.populate_indicators(df, {"pair": "ETH/USDT"})
    signals = strategy.populate_entry_trend(signals, {"pair": "ETH/USDT"})
    return signals


def build_signal_rows(
    signals: pd.DataFrame,
    *,
    only_triggers: bool = True,
) -> list[dict[str, Any]]:
    if signals.empty:
        return []

    source = signals.copy()
    if only_triggers:
        source = source[source["lob_trigger"] == 1]

    rows: list[dict[str, Any]] = []
    for _, row in source.sort_values("date").iterrows():
        drift_amp = 1.0 + (
            LOBLatentRegimeV7Strategy.lob_gamma_amp
            if (row.get("lob_spread_drift") or 0.0) > 0.30
            else 0.0
        )
        rows.append(
            {
                "timeframe": SIGNAL_FREQ,
                "time_ms": _time_ms(row["date"]),
                "time": _iso_time(row["date"]),
                "price": _json_float(row.get("close")),
                "signal_value": _json_float(row.get("lob_score")),
                "threshold": _json_float(row.get("lob_threshold")),
                "trigger": _json_int(row.get("lob_trigger")),
                "entry_short": _json_int(row.get("enter_short")),
                "score_delta": _json_float(row.get("lob_score_delta")),
                "ch1": _json_float(row.get("lob_entropy")),
                "ch2": _json_float(row.get("lob_depth_erosion")),
                "ch3": _json_float(row.get("lob_spread_drift")),
                "ch4": _json_float(row.get("lob_ofi_momentum")),
                "ch5": _json_float(row.get("lob_prestress_proxy")),
                "prestress": _json_float(row.get("lob_prestress_proxy")),
                "drift_amp": drift_amp,
                "score_formula": "max(CH1..CH5) * drift_amp",
                "orderbook_status": "ok"
                if _json_float(row.get("top10_total_depth")) is not None
                else "missing_l2",
                "l2_bid_depth": _json_float(row.get("bid_depth")),
                "l2_ask_depth": _json_float(row.get("ask_depth")),
                "l2_total_depth": _json_float(row.get("total_depth")),
                "l2_bid_depth_5bps": _json_float(row.get("bid_depth_5bps")),
                "l2_ask_depth_5bps": _json_float(row.get("ask_depth_5bps")),
                "l2_total_depth_5bps": _json_float(row.get("total_depth_5bps")),
                "l2_bid_depth_10bps": _json_float(row.get("bid_depth_10bps")),
                "l2_ask_depth_10bps": _json_float(row.get("ask_depth_10bps")),
                "l2_total_depth_10bps": _json_float(row.get("total_depth_10bps")),
                "l2_bid_depth_25bps": _json_float(row.get("bid_depth_25bps")),
                "l2_ask_depth_25bps": _json_float(row.get("ask_depth_25bps")),
                "l2_total_depth_25bps": _json_float(row.get("total_depth_25bps")),
                "bid_levels": _json_float(row.get("bid_levels")),
                "ask_levels": _json_float(row.get("ask_levels")),
                "best_bid": _json_float(row.get("best_bid")),
                "best_ask": _json_float(row.get("best_ask")),
                "spread": _json_float(row.get("spread")),
                "mid_price": _json_float(row.get("mid_price")),
                "trade_count": _json_int(row.get("trade_count")),
                "buy_volume": _json_float(row.get("buy_volume")),
                "sell_volume": _json_float(row.get("sell_volume")),
                "second_trade_volume": _json_float(row.get("volume")),
                "second_buy_volume": _json_float(row.get("buy_volume")),
                "second_sell_volume": _json_float(row.get("sell_volume")),
                "top10_bid_depth": _json_float(row.get("top10_bid_depth")),
                "top10_ask_depth": _json_float(row.get("top10_ask_depth")),
                "top10_total_depth": _json_float(row.get("top10_total_depth")),
                "top10_levels": row.get("top10_levels")
                if isinstance(row.get("top10_levels"), dict)
                else {"bids": [], "asks": []},
            }
        )
    return rows


def build_exchange_payload(
    exchange: str,
    minutes: int,
    *,
    live_only: bool = True,
) -> dict[str, Any]:
    cfg = EXCHANGES[exchange]
    trades = read_trades_1s(cfg["dir"], minutes, live_only=live_only)
    depth = read_depth_1s(cfg["dir"], minutes, live_only=live_only)
    top10 = read_l2_top10_1s(cfg["dir"], minutes, live_only=live_only)
    signals = compute_lob_v7_signals(trades, depth)
    signals = attach_top10(signals, top10)
    latest_rows = build_signal_rows(signals.tail(1), only_triggers=False)
    trigger_rows = build_signal_rows(signals, only_triggers=True)[-100:]
    series_rows = build_signal_rows(signals.tail(240), only_triggers=False)
    return {
        "exchange": exchange,
        "label": cfg["label"],
        "pair": cfg["pair"],
        "updated": datetime.now(UTC).isoformat(),
        "data_available": not trades.empty,
        "trade_minutes": int(len(trades)),
        "depth_minutes": int(len(depth)),
        "signal_frequency": SIGNAL_FREQ,
        "latest": latest_rows[-1] if latest_rows else None,
        "recent_signals": trigger_rows,
        "series": series_rows,
    }


def build_snapshot(minutes: int, *, live_only: bool = True) -> dict[str, Any]:
    now = time.monotonic()
    cache_key = (minutes, live_only)
    if (
        SNAPSHOT_CACHE["key"] == cache_key
        and SNAPSHOT_CACHE["value"] is not None
        and now - float(SNAPSHOT_CACHE["created"]) < 0.95
    ):
        return SNAPSHOT_CACHE["value"]
    if not SNAPSHOT_LOCK.acquire(blocking=False):
        if SNAPSHOT_CACHE["value"] is not None:
            return SNAPSHOT_CACHE["value"]
        SNAPSHOT_LOCK.acquire()
    try:
        now = time.monotonic()
        if (
            SNAPSHOT_CACHE["key"] == cache_key
            and SNAPSHOT_CACHE["value"] is not None
            and now - float(SNAPSHOT_CACHE["created"]) < 0.95
        ):
            return SNAPSHOT_CACHE["value"]
        snapshot = {
            "updated": datetime.now(UTC).isoformat(),
            "refresh_ms": SIGNAL_REFRESH_MS,
            "strategy": "LOBLatentRegimeV7Strategy",
            "symbol": "ETH",
            "exchanges": {
                name: build_exchange_payload(name, minutes, live_only=live_only)
                for name in EXCHANGES
            },
        }
        SNAPSHOT_CACHE["key"] = cache_key
        SNAPSHOT_CACHE["value"] = snapshot
        SNAPSHOT_CACHE["created"] = now
        return snapshot
    finally:
        SNAPSHOT_LOCK.release()


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETH LOB v7 信号监控</title>
<style>
body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f5f7fa; color: #17202a; }
header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: #111827; color: #fff; }
h1 { margin: 0; font-size: 20px; }
.sub { color: #cbd5e1; font-size: 13px; margin-top: 4px; }
main { padding: 14px; display: grid; gap: 14px; }
.cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.card, .panel { background: #fff; border: 1px solid #dfe5ec; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(15,23,42,.06); }
.card h2, .panel h2 { margin: 0 0 12px; font-size: 16px; }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.metric { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; min-height: 54px; }
.metric b { display: block; font-size: 12px; color: #64748b; margin-bottom: 4px; }
.metric span { font-size: 18px; font-variant-numeric: tabular-nums; }
.depth { grid-template-columns: repeat(5, minmax(0, 1fr)); margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 7px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; white-space: nowrap; }
th { position: sticky; top: 0; background: #f8fafc; color: #475569; }
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) { text-align: left; }
.ok { color: #0f766e; font-weight: 700; }
.warn { color: #b91c1c; font-weight: 700; }
.muted { color: #64748b; }
@media (max-width: 1000px) { .cards { grid-template-columns: 1fr; } .metrics, .depth { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
</head>
<body>
<header>
  <div>
    <h1>ETH LOB v7 信号监控</h1>
    <div class="sub">Gate.io + Binance | LOBLatentRegimeV7Strategy | 每分钟自动刷新</div>
  </div>
  <div id="status">加载中...</div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="panel">
    <h2>10档订单簿计算</h2>
    <div class="cards" id="top10"></div>
  </section>
  <section class="panel">
    <h2>信号触发记录</h2>
    <div style="overflow:auto; max-height:440px">
      <table>
        <thead>
          <tr>
            <th>发出时间</th><th>交易所</th><th>价格</th><th>信号值</th><th>阈值</th>
            <th>CH1 熵</th><th>CH2 深度侵蚀</th><th>CH3 价差漂移</th><th>CH4 OFI</th>
            <th>L2买盘深度</th><th>L2卖盘深度</th><th>L2总体深度</th>
            <th>5bps深度</th><th>10bps深度</th><th>25bps深度</th><th>价差</th>
          </tr>
        </thead>
        <tbody id="signals"><tr><td colspan="16" class="muted">等待数据...</td></tr></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const REFRESH_MS = 1000;
const fmt = (v, n = 3) => v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(n);
const time = ms => ms ? new Date(ms).toLocaleString("zh-CN", {hour12:false}) : "-";

function metric(name, value, digits = 3, cls = "") {
  const rendered = typeof value === "string" ? value : fmt(value, digits);
  return `<div class="metric"><b>${name}</b><span class="${cls}">${rendered}</span></div>`;
}

function renderCard(payload) {
  const latest = payload.latest || {};
  const cls = latest.trigger === 1 ? "warn" : "ok";
  const dataNote = payload.data_available ? `${payload.trade_minutes}m trades / ${payload.depth_minutes}m depth` : "等待 parquet 落盘";
  return `<article class="card">
    <h2>${payload.label} <span class="muted">${payload.pair}</span></h2>
    <div class="muted">最新分钟 ${time(latest.time_ms)} | ${dataNote}</div>
    <div class="metrics">
      ${metric("信号值", latest.signal_value, 3, cls)}
      ${metric("CH1 熵", latest.ch1)}
      ${metric("CH2 深度", latest.ch2)}
      ${metric("CH3 价差", latest.ch3)}
      ${metric("CH4 OFI", latest.ch4)}
      ${metric("CH5 Prestress", latest.ch5)}
      ${metric("阈值", latest.threshold)}
      ${metric("价格", latest.price, 2)}
      ${metric("触发", latest.trigger || 0, 0, cls)}
    </div>
    <div class="metrics depth">
      ${metric("L2买盘深度", latest.l2_bid_depth, 2)}
      ${metric("L2卖盘深度", latest.l2_ask_depth, 2)}
      ${metric("L2总体深度", latest.l2_total_depth, 2)}
      ${metric("5bps总深度", latest.l2_total_depth_5bps, 2)}
      ${metric("10bps总深度", latest.l2_total_depth_10bps, 2)}
    </div>
  </article>`;
}

function renderTop10Book(payload) {
  const latest = payload.latest || {};
  const levels = latest.top10_levels || {bids: [], asks: []};
  const buySell = fmt(latest.second_buy_volume, 4) + " / " + fmt(latest.second_sell_volume, 4);
  const maxRows = Math.max(levels.bids.length, levels.asks.length, 10);
  const body = Array.from({length: maxRows}).map((_, i) => {
    const bid = levels.bids[i] || {};
    const ask = levels.asks[i] || {};
    return `<tr>
      <td>${i + 1}</td>
      <td>${fmt(bid.price, 2)}</td><td>${fmt(bid.amount, 4)}</td><td>${fmt(bid.cumulative, 4)}</td>
      <td>${fmt(ask.price, 2)}</td><td>${fmt(ask.amount, 4)}</td><td>${fmt(ask.cumulative, 4)}</td>
    </tr>`;
  }).join("");
  return `<article class="card">
    <h2>${payload.label} <span class="muted">前10档</span></h2>
    <div class="metrics depth">
      ${metric("前10档买盘", latest.top10_bid_depth, 4)}
      ${metric("前10档卖盘", latest.top10_ask_depth, 4)}
      ${metric("前10档总深度", latest.top10_total_depth, 4)}
      ${metric("CH2深度信号", latest.ch2, 3)}
      ${metric("CH5预压力", latest.ch5, 3)}
      ${metric("综合信号值", latest.signal_value, 3)}
      ${metric("本秒成交量", latest.second_trade_volume, 4)}
      ${metric("本秒买/卖", buySell, 4)}
      ${metric("订单簿状态", latest.orderbook_status || "-", 0)}
    </div>
    <div class="muted" style="margin-top:8px">计算: ${latest.score_formula || "-"}；drift_amp=${fmt(latest.drift_amp, 2)}；使用前10档累计深度展示订单簿承接。</div>
    <div style="overflow:auto; max-height:300px; margin-top:10px">
      <table>
        <thead><tr><th>档</th><th>Bid价</th><th>Bid量</th><th>Bid累计</th><th>Ask价</th><th>Ask量</th><th>Ask累计</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  </article>`;
}

function renderRows(snapshot) {
  const rows = [];
  for (const payload of Object.values(snapshot.exchanges)) {
    for (const row of payload.recent_signals || []) rows.push({...row, exchange: payload.label});
  }
  rows.sort((a, b) => b.time_ms - a.time_ms);
  if (!rows.length) return `<tr><td colspan="16" class="muted">当前窗口暂无触发信号；上方卡片仍显示最新分钟的信号量、CH1-4 和 L2 深度。</td></tr>`;
  return rows.map(r => `<tr>
    <td>${time(r.time_ms)}</td><td>${r.exchange}</td><td>${fmt(r.price, 2)}</td><td class="warn">${fmt(r.signal_value)}</td><td>${fmt(r.threshold)}</td>
    <td>${fmt(r.ch1)}</td><td>${fmt(r.ch2)}</td><td>${fmt(r.ch3)}</td><td>${fmt(r.ch4)}</td>
    <td>${fmt(r.l2_bid_depth, 2)}</td><td>${fmt(r.l2_ask_depth, 2)}</td><td>${fmt(r.l2_total_depth, 2)}</td>
    <td>${fmt(r.l2_total_depth_5bps, 2)}</td><td>${fmt(r.l2_total_depth_10bps, 2)}</td><td>${fmt(r.l2_total_depth_25bps, 2)}</td><td>${fmt(r.spread, 4)}</td>
  </tr>`).join("");
}

async function refresh() {
  const response = await fetch("/api/snapshot", {cache: "no-store"});
  const snapshot = await response.json();
  document.getElementById("cards").innerHTML = Object.values(snapshot.exchanges).map(renderCard).join("");
  document.getElementById("top10").innerHTML = Object.values(snapshot.exchanges).map(renderTop10Book).join("");
  document.getElementById("signals").innerHTML = renderRows(snapshot);
  document.getElementById("status").textContent = `刷新: ${new Date().toLocaleTimeString("zh-CN", {hour12:false})}`;
}

refresh().catch(err => document.getElementById("status").textContent = `加载失败: ${err}`);
setInterval(() => refresh().catch(console.error), REFRESH_MS);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    minutes = DEFAULT_MINUTES

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug("%s - %s", self.client_address[0], fmt % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        minutes = int(params.get("minutes", [self.minutes])[0])
        if parsed.path in {"/", "/index.html"}:
            self._send(HTML_TEMPLATE, "text/html; charset=utf-8")
        elif parsed.path == "/api/snapshot":
            self._send_json(build_snapshot(minutes))
        elif parsed.path == "/api/health":
            self._send_json({"status": "ok", "time": datetime.now(UTC).isoformat()})
        else:
            self._send_json({"error": "not found"}, status=404)

    def _send(self, content: str, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_json(self, data: Any, status: int = 200) -> None:
        self._send(json.dumps(data, ensure_ascii=False, default=str), "application/json; charset=utf-8", status)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETH LOB v7 signal dashboard")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DashboardHandler.minutes = args.minutes
    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    LOGGER.info("Dashboard ready: http://localhost:%s", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
