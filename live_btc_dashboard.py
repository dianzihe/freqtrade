#!/usr/bin/env python3
"""
BTC Live Monitoring Dashboard
==============================
Reads live parquet data from cex_tick_recorder.py output, builds 1-minute K-line,
computes CH1-4 microstructure signals, and serves an auto-refreshing web dashboard.

Usage:
    .venv/Scripts/python.exe live_btc_dashboard.py [--port 8888] [--minutes 120]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGGER = logging.getLogger("btc_dashboard")

# ── Config ──────────────────────────────────────────────────────────

EXCHANGES = {
    "gate": {
        "pair": "BTC_USDT",
        "dir": Path("user_data/orderbook_data/gate/spot/BTC_USDT"),
    },
    "binance": {
        "pair": "BTCUSDT",
        "dir": Path("user_data/orderbook_data/binance/spot/BTCUSDT"),
    },
}

CH1_KEY = "ch1_vol_entropy"
CH2_KEY = "ch2_depth_erosion"
CH3_KEY = "ch3_spread_drift"
CH4_KEY = "ch4_order_flow"
SIGNAL_KEY = "signal_trigger"
COMPOSITE_KEY = "composite_smooth"

# ── Data Reader ─────────────────────────────────────────────────────

def find_latest_parquet(data_dir: Path, dataset: str) -> Path | None:
    """Find the most recent parquet file in Hive-partitioned directory."""
    dataset_dir = data_dir / dataset
    if not dataset_dir.exists():
        return None
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        return None
    # Return newest by modification time
    return max(parquet_files, key=lambda p: p.stat().st_mtime)


def find_recent_parquets(data_dir: Path, dataset: str, max_age_minutes: int = 120) -> list[Path]:
    """Find parquet files from recent hours."""
    dataset_dir = data_dir / dataset
    if not dataset_dir.exists():
        return []
    cutoff = time.time() - max_age_minutes * 60
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    return [p for p in parquet_files if p.stat().st_mtime > cutoff]


def read_trades(exchange_dir: Path, max_age_minutes: int = 120) -> pd.DataFrame:
    """Read recent trades and aggregate to 1-minute OHLCV."""
    files = find_recent_parquets(exchange_dir, "trades", max_age_minutes)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pq.read_table(f).to_pandas()
            dfs.append(df)
        except Exception as e:
            LOGGER.debug("Skip %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()

    trades = pd.concat(dfs, ignore_index=True)
    if trades.empty:
        return trades

    # Convert timestamps
    trades["time"] = pd.to_datetime(trades["exchange_time_ms"], unit="ms", utc=True)
    trades = trades.sort_values("time")

    # Filter recent
    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(minutes=max_age_minutes)
    trades = trades[trades["time"] >= cutoff]

    if trades.empty:
        return trades

    # Build 1-min OHLCV
    trades.set_index("time", inplace=True)
    ohlcv = trades["price"].resample("1min").ohlc()
    ohlcv.columns = ["open", "high", "low", "close"]
    ohlcv["volume"] = trades["amount"].resample("1min").sum()

    # Buy/sell volume
    buy_mask = trades["side"] == "buy"
    sell_mask = trades["side"] == "sell"
    ohlcv["buy_volume"] = trades.loc[buy_mask, "amount"].resample("1min").sum() if buy_mask.any() else 0
    ohlcv["sell_volume"] = trades.loc[sell_mask, "amount"].resample("1min").sum() if sell_mask.any() else 0
    ohlcv["trade_count"] = trades["price"].resample("1min").count()

    # Buy volume ratio
    total_vol = ohlcv["buy_volume"].fillna(0) + ohlcv["sell_volume"].fillna(0)
    ohlcv["buy_volume_ratio"] = (ohlcv["buy_volume"].fillna(0) / total_vol.replace(0, np.nan)).fillna(0.5)

    # Price std (intra-minute volatility)
    ohlcv["price_std"] = trades["price"].resample("1min").std()

    ohlcv.reset_index(inplace=True)
    ohlcv.rename(columns={"time": "date"}, inplace=True)
    return ohlcv


def read_depth(exchange_dir: Path, max_age_minutes: int = 120) -> pd.DataFrame:
    """Read recent depth stats and aggregate to 1-minute."""
    files = find_recent_parquets(exchange_dir, "depth", max_age_minutes)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pq.read_table(f).to_pandas()
            dfs.append(df)
        except Exception as e:
            LOGGER.debug("Skip %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()

    depth = pd.concat(dfs, ignore_index=True)
    if depth.empty:
        return depth

    depth["time"] = pd.to_datetime(depth["exchange_time_ms"], unit="ms", utc=True)
    depth = depth.sort_values("time")

    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(minutes=max_age_minutes)
    depth = depth[depth["time"] >= cutoff]

    if depth.empty:
        return depth

    depth.set_index("time", inplace=True)
    agg = depth.resample("1min").agg({
        "bid_depth": "mean",
        "ask_depth": "mean",
        "total_depth": "mean",
        "spread": "mean",
        "mid_price": "last",
        "best_bid": "last",
        "best_ask": "last",
        "imbalance": "mean",
        "bid_depth_5bps": "mean",
        "ask_depth_5bps": "mean",
        "total_depth_5bps": "mean",
        "bid_depth_10bps": "mean",
        "ask_depth_10bps": "mean",
        "total_depth_10bps": "mean",
    })
    agg.reset_index(inplace=True)
    agg.rename(columns={"time": "date"}, inplace=True)
    return agg


def read_l1(exchange_dir: Path, max_age_minutes: int = 120) -> pd.DataFrame:
    """Read recent L1 data and aggregate to 1-minute."""
    files = find_recent_parquets(exchange_dir, "l1", max_age_minutes)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pq.read_table(f).to_pandas()
            dfs.append(df)
        except Exception as e:
            LOGGER.debug("Skip %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()

    l1 = pd.concat(dfs, ignore_index=True)
    if l1.empty:
        return l1

    l1["time"] = pd.to_datetime(l1["exchange_time_ms"], unit="ms", utc=True)
    l1 = l1.sort_values("time")

    cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(minutes=max_age_minutes)
    l1 = l1[l1["time"] >= cutoff]

    if l1.empty:
        return l1

    l1.set_index("time", inplace=True)
    # Use spread directly from L1 for improved accuracy
    agg = l1.resample("1min").agg({
        "spread": "mean",
        "mid_price": "last",
        "bid_price": "last",
        "ask_price": "last",
    })
    agg.reset_index(inplace=True)
    agg.rename(columns={"time": "date"}, inplace=True)
    return agg


# ── Signal Computer ─────────────────────────────────────────────────

def compute_ch1_volatility_entropy(close: pd.Series, vol_window: int = 20) -> pd.Series:
    """CH1: Multi-scale volatility structure entropy."""
    returns = close.pct_change()
    w = vol_window
    short_w = max(4, w // 4)
    mid_w = max(8, w // 2)

    vol_s = returns.rolling(short_w).std()
    vol_m = returns.rolling(mid_w).std()
    vol_l = returns.rolling(w).std()

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
    return (entropy / np.log(3)).fillna(0.0)


def compute_ch2_depth_erosion(total_depth: pd.Series, window: int = 24) -> pd.Series:
    """CH2: Depth erosion signal. NaN-safe — gaps produce 0 signal."""
    # Drop NaN temporarily for rolling computation to avoid NaN propagation
    valid = total_depth.notna()
    result = pd.Series(0.0, index=total_depth.index)
    if valid.sum() < window:
        return result
    
    td = total_depth.ffill().bfill()
    depth_mean = td.rolling(window=window * 2, min_periods=window).mean()
    depth_std = td.rolling(window=window * 2, min_periods=window).std()
    depth_z = ((td - depth_mean) / depth_std.replace(0, 1e-10))
    result = (-depth_z).clip(lower=0).fillna(0.0)
    return result


def compute_ch3_spread_drift(spread: pd.Series, mid_price: pd.Series, window: int = 24) -> pd.Series:
    """CH3: Spread drift signal."""
    spread_ratio = spread / mid_price.replace(0, np.nan)
    spread_ratio = spread_ratio.fillna(0.0)
    spread_mean = spread_ratio.rolling(window=window * 2, min_periods=window).mean()
    spread_std = spread_ratio.rolling(window=window * 2, min_periods=window).std()
    spread_z = ((spread_ratio - spread_mean) / spread_std.replace(0, 1e-10)).fillna(0.0)
    return spread_z.clip(lower=0)


def compute_ch4_order_flow(buy_vol: pd.Series, sell_vol: pd.Series, window: int = 24) -> pd.Series:
    """CH4: Order flow imbalance."""
    buy_vol = buy_vol.fillna(0.0)
    sell_vol = sell_vol.fillna(0.0)
    net_flow = buy_vol - sell_vol
    flow_mean = net_flow.rolling(window=window * 2, min_periods=window).mean()
    flow_std = net_flow.rolling(window=window * 2, min_periods=window).std()
    flow_z = ((net_flow - flow_mean) / flow_std.replace(0, 1e-10)).fillna(0.0)
    return flow_z


def compute_signals(
    ohlcv: pd.DataFrame,
    depth: pd.DataFrame | None = None,
    l1: pd.DataFrame | None = None,
    lookback: int = 24,
    threshold_percentile: int = 88,
    confirmation_bars: int = 2,
    min_signal_strength: float = 0.40,
) -> pd.DataFrame:
    """Compute CH1-4 signals and trigger on 1-minute OHLCV data."""
    df = ohlcv.copy()
    if df.empty or len(df) < lookback:
        for col in [CH1_KEY, CH2_KEY, CH3_KEY, CH4_KEY, COMPOSITE_KEY, "adaptive_threshold", SIGNAL_KEY]:
            df[col] = np.nan
        return df

    # CH1: from close prices
    df[CH1_KEY] = compute_ch1_volatility_entropy(df["close"], vol_window=20)

    # CH2: from depth data if available
    if depth is not None and not depth.empty and "total_depth" in depth.columns:
        merged_depth = df.merge(depth[["date", "total_depth"]], on="date", how="left")
        # Forward-fill NaN depth values to handle gaps (recorder lag/crash)
        filled_depth = merged_depth["total_depth"].ffill(limit=10)
        df[CH2_KEY] = compute_ch2_depth_erosion(filled_depth, lookback)
    else:
        df[CH2_KEY] = 0.0

    # CH3: from L1 spread if available
    if l1 is not None and not l1.empty and "spread" in l1.columns and "mid_price" in l1.columns:
        merged_l1 = df.merge(l1[["date", "spread", "mid_price"]], on="date", how="left")
        # Fill missing with OHLCV-based proxy
        fill_spread = merged_l1["spread"].fillna((merged_l1["high"] - merged_l1["low"]) * 0.1)
        fill_mid = merged_l1["mid_price"].fillna(merged_l1["close"])
        df[CH3_KEY] = compute_ch3_spread_drift(fill_spread, fill_mid, lookback)
    else:
        # Fallback: use high-low as spread proxy
        proxy_spread = (df["high"] - df["low"]).clip(lower=0)
        df[CH3_KEY] = compute_ch3_spread_drift(proxy_spread, df["close"], lookback)

    # CH4: from trade buy/sell volume
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        df[CH4_KEY] = compute_ch4_order_flow(df["buy_volume"], df["sell_volume"], lookback)
    elif "buy_volume_ratio" in df.columns:
        buy_vol = df["buy_volume_ratio"].fillna(0.5) * df["volume"].fillna(0)
        sell_vol = (1 - df["buy_volume_ratio"].fillna(0.5)) * df["volume"].fillna(0)
        df[CH4_KEY] = compute_ch4_order_flow(buy_vol, sell_vol, lookback)
    else:
        df[CH4_KEY] = 0.0

    # Composite and trigger
    channels = [CH1_KEY, CH2_KEY, CH3_KEY, CH4_KEY]
    composite = df[channels].copy()
    composite[CH4_KEY] = composite[CH4_KEY].abs()
    df["composite_raw"] = composite.max(axis=1)
    df[COMPOSITE_KEY] = df["composite_raw"].rolling(window=3, min_periods=1).mean()

    # Rising streak detection
    rising = df[COMPOSITE_KEY] > df[COMPOSITE_KEY].shift(1)
    rising_streak = rising.copy()
    for lag in range(2, confirmation_bars + 1):
        rising_streak = rising_streak & (df[COMPOSITE_KEY] > df[COMPOSITE_KEY].shift(lag))

    # Adaptive threshold
    long_w = max(lookback * 4, 48)
    adaptive = df[COMPOSITE_KEY].rolling(long_w, min_periods=lookback * 2).quantile(threshold_percentile / 100.0)
    effective = adaptive.clip(lower=min_signal_strength)

    trigger = (
        (df[COMPOSITE_KEY] > effective)
        & rising_streak
        & (df[COMPOSITE_KEY] > min_signal_strength)
    )

    df["adaptive_threshold"] = effective
    df[SIGNAL_KEY] = trigger.astype(int)
    return df


# ── Data Aggregator ─────────────────────────────────────────────────

def get_dashboard_data(exchange_name: str, max_minutes: int = 120) -> dict[str, Any]:
    """Read all data for one exchange and return a JSON-serializable dict."""
    cfg = EXCHANGES[exchange_name]
    data_dir = cfg["dir"]

    ohlcv = read_trades(data_dir, max_minutes)
    depth = read_depth(data_dir, max_minutes)
    l1 = read_l1(data_dir, max_minutes)

    result: dict[str, Any] = {
        "exchange": exchange_name,
        "pair": cfg["pair"],
        "updated": datetime.now(timezone.utc).isoformat(),
        "candles": [],
        "signals": [],
        "depth_stats": [],
        "data_available": False,
    }

    if ohlcv.empty:
        return result

    result["data_available"] = True

    # Compute signals
    sig_df = compute_signals(ohlcv, depth, l1)

    # Build candle data for ECharts
    for _, row in ohlcv.iterrows():
        ts = row["date"]
        if hasattr(ts, "timestamp"):
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)

        result["candles"].append({
            "t": ts_ms,
            "o": float(row["open"]) if not pd.isna(row["open"]) else None,
            "h": float(row["high"]) if not pd.isna(row["high"]) else None,
            "l": float(row["low"]) if not pd.isna(row["low"]) else None,
            "c": float(row["close"]) if not pd.isna(row["close"]) else None,
            "v": float(row["volume"]) if not pd.isna(row["volume"]) else 0,
        })

    # Build signal data
    for _, row in sig_df.iterrows():
        ts = row["date"]
        if hasattr(ts, "timestamp"):
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)

        ch1 = float(row[CH1_KEY]) if not pd.isna(row[CH1_KEY]) else None
        ch2 = float(row[CH2_KEY]) if not pd.isna(row[CH2_KEY]) else None
        ch3 = float(row[CH3_KEY]) if not pd.isna(row[CH3_KEY]) else None
        ch4 = float(row[CH4_KEY]) if not pd.isna(row[CH4_KEY]) else None
        comp = float(row[COMPOSITE_KEY]) if not pd.isna(row[COMPOSITE_KEY]) else None
        trigger = int(row[SIGNAL_KEY]) if not pd.isna(row[SIGNAL_KEY]) else 0

        threshold = float(row["adaptive_threshold"]) if not pd.isna(row["adaptive_threshold"]) else None

        result["signals"].append({
            "t": ts_ms,
            "ch1": ch1,
            "ch2": ch2,
            "ch3": ch3,
            "ch4": ch4,
            "composite": comp,
            "threshold": threshold,
            "trigger": trigger,
        })

    # Depth stats (per second, aggregated per minute for display)
    if not depth.empty:
        for _, row in depth.iterrows():
            ts = row["date"]
            if hasattr(ts, "timestamp"):
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)

            result["depth_stats"].append({
                "t": ts_ms,
                "bid_depth": float(row.get("bid_depth", 0)) if not pd.isna(row.get("bid_depth", 0)) else 0,
                "ask_depth": float(row.get("ask_depth", 0)) if not pd.isna(row.get("ask_depth", 0)) else 0,
                "total_depth": float(row.get("total_depth", 0)) if not pd.isna(row.get("total_depth", 0)) else 0,
                "spread": float(row.get("spread", 0)) if not pd.isna(row.get("spread", 0)) else 0,
                "imbalance": float(row.get("imbalance", 0)) if not pd.isna(row.get("imbalance", 0)) else 0,
                "depth_5bps": float(row.get("total_depth_5bps", 0)) if not pd.isna(row.get("total_depth_5bps", 0)) else 0,
                "depth_10bps": float(row.get("total_depth_10bps", 0)) if not pd.isna(row.get("total_depth_10bps", 0)) else 0,
            })

    return result


def get_trade_depth_stats(exchange_name: str, max_minutes: int = 120) -> dict[str, Any]:
    """Get per-second trade impact on L2 order book depth."""
    cfg = EXCHANGES[exchange_name]
    data_dir = cfg["dir"]

    # Read trades at full resolution
    trade_files = find_recent_parquets(data_dir, "trades", max_minutes)
    depth_files = find_recent_parquets(data_dir, "depth", max_minutes)

    result: dict[str, Any] = {
        "exchange": exchange_name,
        "pair": cfg["pair"],
        "trades_per_second": [],
        "depth_snapshots": [],
        "data_available": False,
    }

    # Read trades
    if trade_files:
        dfs = []
        for f in trade_files:
            try:
                dfs.append(pq.read_table(f).to_pandas())
            except Exception:
                pass
        if dfs:
            trades = pd.concat(dfs, ignore_index=True)
            trades["time"] = pd.to_datetime(trades["exchange_time_ms"], unit="ms", utc=True)
            cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(minutes=max_minutes)
            trades = trades[trades["time"] >= cutoff]

            if trades.empty:
                return result

            # Pre-compute buy/sell amount columns to avoid lambda in resample
            trades["buy_amount"] = trades["amount"].where(trades["side"] == "buy", 0.0)
            trades["sell_amount"] = trades["amount"].where(trades["side"] == "sell", 0.0)
            trades["price_x_amount"] = trades["price"] * trades["amount"]

            # Aggregate trades per second using simple built-in aggregators
            trades.set_index("time", inplace=True)
            per_sec = trades.resample("1s").agg(
                trade_count=("price", "count"),
                total_volume=("amount", "sum"),
                buy_volume=("buy_amount", "sum"),
                sell_volume=("sell_amount", "sum"),
                price_x_amount=("price_x_amount", "sum"),
            )
            # VWAP = sum(price * amount) / sum(amount)
            total_vol = per_sec["total_volume"]
            per_sec["vwap"] = per_sec["price_x_amount"] / total_vol.replace(0, np.nan)
            per_sec.drop(columns=["price_x_amount"], inplace=True)

            per_sec.reset_index(inplace=True)
            per_sec.rename(columns={"time": "t"}, inplace=True)
            for _, row in per_sec.iterrows():
                result["trades_per_second"].append({
                    "t": int(row["t"].timestamp() * 1000),
                    "count": int(row["trade_count"]) if not pd.isna(row["trade_count"]) else 0,
                    "volume": float(row["total_volume"]) if not pd.isna(row["total_volume"]) else 0,
                    "buy_vol": float(row["buy_volume"]) if not pd.isna(row["buy_volume"]) else 0,
                    "sell_vol": float(row["sell_volume"]) if not pd.isna(row["sell_volume"]) else 0,
                    "vwap": float(row["vwap"]) if not pd.isna(row["vwap"]) else None,
                })
            result["data_available"] = True

    # Read depth
    if depth_files:
        dfs = []
        for f in depth_files:
            try:
                dfs.append(pq.read_table(f).to_pandas())
            except Exception:
                pass
        if dfs:
            depths = pd.concat(dfs, ignore_index=True)
            depths["time"] = pd.to_datetime(depths["exchange_time_ms"], unit="ms", utc=True)
            cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(minutes=max_minutes)
            depths = depths[depths["time"] >= cutoff]
            for _, row in depths.iterrows():
                result["depth_snapshots"].append({
                    "t": int(row["time"].timestamp() * 1000),
                    "bid_depth": float(row["bid_depth"]) if not pd.isna(row["bid_depth"]) else 0,
                    "ask_depth": float(row["ask_depth"]) if not pd.isna(row["ask_depth"]) else 0,
                    "total_depth": float(row["total_depth"]) if not pd.isna(row["total_depth"]) else 0,
                    "bid_depth_5bps": float(row.get("bid_depth_5bps", 0)) if not pd.isna(row.get("bid_depth_5bps", 0)) else 0,
                    "ask_depth_5bps": float(row.get("ask_depth_5bps", 0)) if not pd.isna(row.get("ask_depth_5bps", 0)) else 0,
                    "spread": float(row["spread"]) if not pd.isna(row["spread"]) else 0,
                    "mid_price": float(row["mid_price"]) if not pd.isna(row["mid_price"]) else 0,
                    "imbalance": float(row["imbalance"]) if not pd.isna(row["imbalance"]) else 0,
                })

    return result


# ── HTTP Server ─────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC 实时监控面板</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f6fa; color: #333; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 20px; }
.header .info { font-size: 13px; opacity: 0.8; }
.header .status { display: flex; gap: 16px; font-size: 13px; }
.header .status span { padding: 4px 12px; border-radius: 12px; }
.status-live { background: #27ae60; }
.status-waiting { background: #f39c12; }
.status-error { background: #e74c3c; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; height: calc(100vh - 56px); }
.panel { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; display: flex; flex-direction: column; }
.panel-header { padding: 10px 16px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
.panel-header h2 { font-size: 15px; }
.panel-header .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; }
.badge-gate { background: #e8f5e9; color: #27ae60; }
.badge-binance { background: #fff3e0; color: #f39c12; }
.top-row { grid-row: span 2; }
.bottom-row { grid-row: span 1; }
.chart-container { flex: 1; min-height: 0; }
.signal-row { display: flex; gap: 8px; padding: 8px 12px; border-top: 1px solid #eee; flex-wrap: wrap; }
.signal-tag { font-size: 11px; padding: 2px 8px; border-radius: 4px; white-space: nowrap; }
.signal-tag.ch1 { background: #e3f2fd; color: #1565c0; }
.signal-tag.ch2 { background: #fce4ec; color: #c62828; }
.signal-tag.ch3 { background: #fff3e0; color: #e65100; }
.signal-tag.ch4 { background: #e8f5e9; color: #2e7d32; }
.signal-tag.composite { background: #f3e5f5; color: #7b1fa2; }
.signal-tag.trigger { background: #ff5252; color: #fff; font-weight: bold; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
.depth-panel { grid-column: 1 / -1; }
.signal-list-panel { grid-column: 1 / -1; max-height: 350px; overflow-y: auto; }
.signal-list-panel table { width: 100%; border-collapse: collapse; font-size: 12px; }
.signal-list-panel thead th { position: sticky; top: 0; background: #f8f9fa; padding: 8px 10px; text-align: center; border-bottom: 2px solid #ddd; z-index: 1; }
.signal-list-panel tbody td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; }
.signal-list-panel tbody tr:hover { background: #f5f5f5; }
.signal-list-panel .exchange-col { font-weight: bold; }
.signal-list-panel .exchange-gate { color: #27ae60; }
.signal-list-panel .exchange-binance { color: #f39c12; }
.signal-list-panel .no-signals { text-align: center; padding: 20px; color: #999; font-size: 14px; }
.signal-list-panel .val-high { background: #ffebee; border-radius: 3px; padding: 1px 4px; }
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>BTC 实时监控面板</h1>
        <div class="info">数据源: Gate.io + Binance | 1分钟K线 + CH1-4信号 | 自动刷新: 60秒</div>
    </div>
    <div class="status">
        <span id="gate-status" class="status-waiting">Gate: 等待数据...</span>
        <span id="binance-status" class="status-waiting">Binance: 等待数据...</span>
        <span id="clock">--</span>
    </div>
</div>
<div class="grid">
    <!-- Gate K-line -->
    <div class="panel top-row" id="gate-panel">
        <div class="panel-header">
            <h2>Gate.io · BTC/USDT · K线 + 信号</h2>
            <span class="badge badge-gate">Gate</span>
        </div>
        <div class="chart-container" id="gate-chart"></div>
        <div class="signal-row" id="gate-signals"></div>
    </div>
    <!-- Binance K-line -->
    <div class="panel top-row" id="binance-panel">
        <div class="panel-header">
            <h2>Binance · BTC/USDT · K线 + 信号</h2>
            <span class="badge badge-binance">Binance</span>
        </div>
        <div class="chart-container" id="binance-chart"></div>
        <div class="signal-row" id="binance-signals"></div>
    </div>
    <!-- CH1-4 Channels -->
    <div class="panel bottom-row">
        <div class="panel-header"><h2>Gate · CH1-4 信号通道 & 综合信号</h2></div>
        <div class="chart-container" id="gate-ch-chart"></div>
    </div>
    <div class="panel bottom-row">
        <div class="panel-header"><h2>Binance · CH1-4 信号通道 & 综合信号</h2></div>
        <div class="chart-container" id="binance-ch-chart"></div>
    </div>
    <!-- Depth Panel -->
    <div class="panel depth-panel" style="min-height: 220px;">
        <div class="panel-header"><h2>L2 订单簿深度统计 — 总深度 + 5bps近端深度 + 成交量消耗</h2></div>
        <div class="chart-container" id="depth-chart"></div>
        <div class="signal-row" id="depth-summary" style="font-size:11px;color:#666;"></div>
    </div>
    <!-- Signal History List -->
    <div class="panel signal-list-panel">
        <div class="panel-header"><h2>📋 信号量历史列表 (CH1-4 明细)</h2></div>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>交易所</th>
                        <th>价格</th>
                        <th>CH1 波动率熵</th>
                        <th>CH2 深度侵蚀</th>
                        <th>CH3 价差漂移</th>
                        <th>CH4 订单流</th>
                        <th>综合信号</th>
                        <th>自适应阈值</th>
                    </tr>
                </thead>
                <tbody id="signal-list-body">
                    <tr><td colspan="9" class="no-signals">等待信号数据...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
const REFRESH_MS = 60000;
let gateChart, binanceChart, gateChChart, binanceChChart, depthChart;

function initKlineChart(domId) {
    const dom = document.getElementById(domId);
    const chart = echarts.init(dom);
    chart.setOption({
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            formatter: function(params) {
                let html = '';
                for (let p of params) {
                    if (p.seriesType === 'candlestick') {
                        const d = p.data;
                        html += `<b>${new Date(d[0]).toLocaleTimeString('zh-CN')}</b><br/>
                        O: ${d[1]} H: ${d[2]} L: ${d[3]} C: ${d[4]}<br/>Vol: ${d[5]}`;
                    }
                }
                return html;
            }
        },
        grid: { left: 60, right: 20, top: 10, bottom: 30 },
        xAxis: {
            type: 'time',
            axisLabel: { fontSize: 10 },
            splitLine: { show: false },
        },
        yAxis: {
            type: 'value',
            scale: true,
            axisLabel: { fontSize: 10, formatter: v => v.toFixed(0) },
            splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
        },
        series: [{
            type: 'candlestick',
            name: 'K线',
            data: [],
            itemStyle: {
                color: '#ef5350',
                color0: '#26a69a',
                borderColor: '#ef5350',
                borderColor0: '#26a69a',
            },
            markPoint: {
                data: [],
                symbol: 'pin',
                symbolSize: 20,
                itemStyle: { color: '#ff1744' },
                label: { show: true, fontSize: 9, formatter: p => p.name || '⚠' },
            },
        }],
        dataZoom: [{ type: 'inside', start: 50, end: 100 }],
    });
    return chart;
}

function initChChart(domId, exchange) {
    const dom = document.getElementById(domId);
    const chart = echarts.init(dom);
    chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { data: ['CH1波动率熵', 'CH2深度侵蚀', 'CH3价差漂移', 'CH4订单流', '综合信号', '自适应阈值'], top: 0, textStyle: { fontSize: 10 } },
        grid: { left: 50, right: 15, top: 30, bottom: 20 },
        xAxis: { type: 'time', axisLabel: { fontSize: 10 } },
        yAxis: { type: 'value', min: 0, max: 1, axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { color: '#eee' } } },
        series: [
            { name: 'CH1波动率熵', type: 'line', data: [], smooth: true, lineStyle: { color: '#42a5f5', width: 1.5 }, symbol: 'none' },
            { name: 'CH2深度侵蚀', type: 'line', data: [], smooth: true, lineStyle: { color: '#ef5350', width: 1.5 }, symbol: 'none' },
            { name: 'CH3价差漂移', type: 'line', data: [], smooth: true, lineStyle: { color: '#ff9800', width: 1.5 }, symbol: 'none' },
            { name: 'CH4订单流', type: 'line', data: [], smooth: true, lineStyle: { color: '#66bb6a', width: 1.5 }, symbol: 'none' },
            { name: '综合信号', type: 'line', data: [], smooth: true, lineStyle: { color: '#ab47bc', width: 2 }, symbol: 'none', areaStyle: { color: 'rgba(171,71,188,0.08)' } },
            { name: '自适应阈值', type: 'line', data: [], smooth: true, lineStyle: { color: '#999', width: 1, type: 'dashed' }, symbol: 'none' },
        ],
        dataZoom: [{ type: 'inside', start: 50, end: 100 }],
    });
    return chart;
}

function initDepthChart() {
    const dom = document.getElementById('depth-chart');
    const chart = echarts.init(dom);
    chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: {
            data: ['Gate总深度', 'Binance总深度', 'Gate 5bps深度', 'Binance 5bps深度', 'Gate成交量/s', 'Binance成交量/s'],
            top: 0, textStyle: { fontSize: 10 }
        },
        grid: { left: 65, right: 65, top: 30, bottom: 25 },
        xAxis: { type: 'time', axisLabel: { fontSize: 10 } },
        yAxis: [
            { type: 'value', name: '深度 (BTC)', axisLabel: { fontSize: 10, formatter: v => v.toFixed(1) }, splitLine: { lineStyle: { color: '#eee', type: 'dashed' } } },
            { type: 'value', name: '成交量 (BTC/s)', axisLabel: { fontSize: 10, formatter: v => v.toFixed(3) } },
        ],
        series: [
            { name: 'Gate总深度', type: 'line', data: [], smooth: true, yAxisIndex: 0, lineStyle: { color: '#27ae60', width: 2 }, symbol: 'none' },
            { name: 'Binance总深度', type: 'line', data: [], smooth: true, yAxisIndex: 0, lineStyle: { color: '#f39c12', width: 2 }, symbol: 'none' },
            { name: 'Gate 5bps深度', type: 'line', data: [], smooth: true, yAxisIndex: 0, lineStyle: { color: '#27ae60', width: 1, type: 'dashed' }, symbol: 'none' },
            { name: 'Binance 5bps深度', type: 'line', data: [], smooth: true, yAxisIndex: 0, lineStyle: { color: '#f39c12', width: 1, type: 'dashed' }, symbol: 'none' },
            { name: 'Gate成交量/s', type: 'bar', data: [], yAxisIndex: 1, itemStyle: { color: 'rgba(39,174,96,0.35)' }, barMaxWidth: 20 },
            { name: 'Binance成交量/s', type: 'bar', data: [], yAxisIndex: 1, itemStyle: { color: 'rgba(243,156,18,0.35)' }, barMaxWidth: 20 },
        ],
        dataZoom: [{ type: 'inside', start: 50, end: 100 }],
    });
    return chart;
}

async function fetchData(exchange) {
    try {
        const resp = await fetch(`/api/data?exchange=${exchange}`);
        return await resp.json();
    } catch (e) {
        console.error(`Fetch ${exchange} error:`, e);
        return null;
    }
}

async function fetchDepthData() {
    try {
        const [gateResp, binResp] = await Promise.all([
            fetch('/api/trade_depth?exchange=gate'),
            fetch('/api/trade_depth?exchange=binance'),
        ]);
        return {
            gate: await gateResp.json(),
            binance: await binResp.json(),
        };
    } catch (e) {
        console.error('Fetch depth error:', e);
        return null;
    }
}

function updateKlineChart(chart, data, signalDivId) {
    if (!data || !data.data_available) return;
    const candles = data.candles.map(c => [c.t, c.o, c.h, c.l, c.c, c.v]);
    const signals = data.signals || [];
    const markPoints = signals
        .filter(s => s.trigger === 1)
        .map((s, i) => ({
            name: '⚠',
            coord: [s.t, data.candles.find(c => c.t === s.t)?.c || null],
            value: `CH1:${s.ch1?.toFixed(2)} CH2:${s.ch2?.toFixed(2)} CH3:${s.ch3?.toFixed(2)} CH4:${s.ch4?.toFixed(2)}`,
        }))
        .filter(mp => mp.coord[1] !== null);

    chart.setOption({
        series: [{
            data: candles,
            markPoint: { data: markPoints },
        }],
    });

    // Update latest signal values
    const latestSignals = signals.filter(s => s.trigger === 1).slice(-1);
    const div = document.getElementById(signalDivId);
    if (latestSignals.length > 0) {
        const s = latestSignals[0];
        div.innerHTML = `
            <span class="signal-tag trigger">⚠ 触发!</span>
            <span class="signal-tag ch1">CH1: ${s.ch1?.toFixed(3) || '-'}</span>
            <span class="signal-tag ch2">CH2: ${s.ch2?.toFixed(3) || '-'}</span>
            <span class="signal-tag ch3">CH3: ${s.ch3?.toFixed(3) || '-'}</span>
            <span class="signal-tag ch4">CH4: ${s.ch4?.toFixed(3) || '-'}</span>
            <span class="signal-tag composite">综合: ${s.composite?.toFixed(3) || '-'}</span>
        `;
    } else {
        const latest = signals[signals.length - 1];
        div.innerHTML = latest
            ? `最新: CH1:${latest.ch1?.toFixed(3) || '-'} CH2:${latest.ch2?.toFixed(3) || '-'} CH3:${latest.ch3?.toFixed(3) || '-'} CH4:${latest.ch4?.toFixed(3) || '-'} 综合:${latest.composite?.toFixed(3) || '-'}`
            : '等待数据...';
    }
}

function updateChChart(chart, data) {
    if (!data || !data.data_available) return;
    const signals = data.signals || [];
    chart.setOption({
        series: [
            { data: signals.map(s => [s.t, s.ch1]) },
            { data: signals.map(s => [s.t, s.ch2]) },
            { data: signals.map(s => [s.t, s.ch3]) },
            { data: signals.map(s => [s.t, s.ch4]) },
            { data: signals.map(s => [s.t, s.composite]) },
            { data: signals.map(s => [s.t, s.threshold !== undefined ? s.threshold : null]).filter(p => p[1] !== null) },
        ],
    });
}

function updateDepthChart(chart, depthData) {
    if (!depthData) return;
    const gateTrades = (depthData.gate.trades_per_second || []).slice(-300);
    const binTrades = (depthData.binance.trades_per_second || []).slice(-300);
    const gateDepth = (depthData.gate.depth_snapshots || []).slice(-300);
    const binDepth = (depthData.binance.depth_snapshots || []).slice(-300);

    // Gate: total depth & 5bps depth
    const gateTotal = gateDepth.map(d => [d.t, +(d.total_depth || 0).toFixed(1)]);
    // Gate 5bps depth - need to compute from bid+ask 5bps
    const gate5bps = gateDepth.map(d => {
        const b5 = +(d.bid_depth_5bps || d.bid_depth || 0);
        const a5 = +(d.ask_depth_5bps || d.ask_depth || 0);
        return [d.t, +(b5 + a5).toFixed(1)];
    });

    // Binance: total depth & 5bps depth
    const binTotal = binDepth.map(d => [d.t, +(d.total_depth || 0).toFixed(1)]);
    const bin5bps = binDepth.map(d => {
        const b5 = +(d.bid_depth_5bps || d.bid_depth || 0);
        const a5 = +(d.ask_depth_5bps || d.ask_depth || 0);
        return [d.t, +(b5 + a5).toFixed(1)];
    });

    // Trade volumes (BTC/s) - to show depth consumption
    const gateVol = gateTrades.map(d => [d.t, +(d.volume || 0).toFixed(4)]);
    const binVol = binTrades.map(d => [d.t, +(d.volume || 0).toFixed(4)]);

    // Compute latest depth summary for text display
    const latestG = gateDepth.length > 0 ? gateDepth[gateDepth.length - 1] : null;
    const latestB = binDepth.length > 0 ? binDepth[binDepth.length - 1] : null;
    const summary = document.getElementById('depth-summary');
    if (summary) {
        let html = '';
        if (latestG) {
            html += `<span style="color:#27ae60;font-weight:bold;">Gate</span>
                总深度: ${(latestG.total_depth||0).toFixed(1)} BTC
                | 买盘: ${(latestG.bid_depth||0).toFixed(1)}
                | 卖盘: ${(latestG.ask_depth||0).toFixed(1)}
                | 5bps: ${((latestG.bid_depth_5bps||0)+(latestG.ask_depth_5bps||0)).toFixed(1)}
                | 中间价: ${(latestG.mid_price||0).toFixed(2)}`;
        }
        if (latestB) {
            html += `&nbsp;&nbsp;&nbsp;<span style="color:#f39c12;font-weight:bold;">Binance</span>
                总深度: ${(latestB.total_depth||0).toFixed(0)} BTC
                | 买盘: ${(latestB.bid_depth||0).toFixed(0)}
                | 卖盘: ${(latestB.ask_depth||0).toFixed(0)}
                | 5bps: ${((latestB.bid_depth_5bps||0)+(latestB.ask_depth_5bps||0)).toFixed(1)}
                | 中间价: ${(latestB.mid_price||0).toFixed(2)}`;
        }
        if (!latestG && !latestB) html = '等待深度数据...';
        summary.innerHTML = html;
    }

    chart.setOption({
        series: [
            { data: gateTotal },
            { data: binTotal },
            { data: gate5bps },
            { data: bin5bps },
            { data: gateVol },
            { data: binVol },
        ],
    });
}

function updateStatus(exchange, data) {
    const el = document.getElementById(`${exchange}-status`);
    if (!data) {
        el.className = 'status-error';
        el.textContent = `${exchange}: 连接失败`;
        return;
    }
    if (data.data_available) {
        el.className = 'status-live';
        el.textContent = `${exchange}: 活跃 (${data.candles.length}根K线)`;
    } else {
        el.className = 'status-waiting';
        el.textContent = `${exchange}: 等待数据...`;
    }
}

function updateSignalList(gateData, binData) {
    const tbody = document.getElementById('signal-list-body');
    const allTriggers = [];

    function collectSignals(data, exchange) {
        if (!data || !data.data_available) return;
        const signals = data.signals || [];
        let latestPrice = null;
        for (const s of signals) {
            if (s.trigger === 1) {
                // Find corresponding candle close price
                const candle = data.candles.find(c => c.t === s.t);
                const price = candle ? candle.c : null;
                if (price) latestPrice = price;
                allTriggers.push({
                    t: s.t,
                    exchange: exchange,
                    price: price || latestPrice,
                    ch1: s.ch1,
                    ch2: s.ch2,
                    ch3: s.ch3,
                    ch4: s.ch4,
                    composite: s.composite,
                    threshold: s.threshold,
                });
            }
        }
    }

    collectSignals(gateData, 'gate');
    collectSignals(binData, 'binance');

    // Sort by time descending (newest first)
    allTriggers.sort((a, b) => b.t - a.t);

    if (allTriggers.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="no-signals">暂无触发信号</td></tr>';
        return;
    }

    tbody.innerHTML = allTriggers.map(s => {
        const exClass = s.exchange === 'gate' ? 'exchange-gate' : 'exchange-binance';
        const exLabel = s.exchange === 'gate' ? 'Gate' : 'Binance';
        const timeStr = new Date(s.t).toLocaleString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', month: '2-digit', day: '2-digit' });
        const maxCh = Math.max(s.ch1||0, s.ch2||0, s.ch3||0, Math.abs(s.ch4||0));
        const highClass = (val) => (val !== null && val !== undefined && val >= 0.5) ? ' class="val-high"' : '';
        return `<tr>
            <td>${timeStr}</td>
            <td class="exchange-col ${exClass}">${exLabel}</td>
            <td>${s.price ? s.price.toFixed(2) : '-'}</td>
            <td${highClass(s.ch1)}>${s.ch1 != null ? s.ch1.toFixed(3) : '-'}</td>
            <td${highClass(s.ch2)}>${s.ch2 != null ? s.ch2.toFixed(3) : '-'}</td>
            <td${highClass(s.ch3)}>${s.ch3 != null ? s.ch3.toFixed(3) : '-'}</td>
            <td${highClass(Math.abs(s.ch4))}>${s.ch4 != null ? s.ch4.toFixed(3) : '-'}</td>
            <td${highClass(s.composite)}>${s.composite != null ? s.composite.toFixed(3) : '-'}</td>
            <td>${s.threshold != null ? s.threshold.toFixed(3) : '-'}</td>
        </tr>`;
    }).join('');
}

async function refreshAll() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('zh-CN');

    const [gateData, binData, depthData] = await Promise.all([
        fetchData('gate'),
        fetchData('binance'),
        fetchDepthData(),
    ]);

    updateKlineChart(gateChart, gateData, 'gate-signals');
    updateKlineChart(binanceChart, binData, 'binance-signals');
    updateChChart(gateChChart, gateData);
    updateChChart(binanceChChart, binData);
    updateDepthChart(depthChart, depthData);
    updateSignalList(gateData, binData);
    updateStatus('gate', gateData);
    updateStatus('binance', binData);
}

// Init
gateChart = initKlineChart('gate-chart');
binanceChart = initKlineChart('binance-chart');
gateChChart = initChChart('gate-ch-chart');
binanceChChart = initChChart('binance-ch-chart');
depthChart = initDepthChart();

// Handle window resize
window.addEventListener('resize', () => {
    gateChart.resize();
    binanceChart.resize();
    gateChChart.resize();
    binanceChChart.resize();
    depthChart.resize();
});

// Initial load + periodic refresh
refreshAll();
setInterval(refreshAll, REFRESH_MS);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the live dashboard."""

    def log_message(self, format, *args):
        LOGGER.debug("%s - %s", self.client_address[0], format % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/data":
            exchange = params.get("exchange", ["gate"])[0]
            if exchange not in EXCHANGES:
                self._json({"error": f"Unknown exchange: {exchange}"}, 400)
                return
            data = get_dashboard_data(exchange)
            self._json(data)
        elif path == "/api/trade_depth":
            exchange = params.get("exchange", ["gate"])[0]
            if exchange not in EXCHANGES:
                self._json({"error": f"Unknown exchange: {exchange}"}, 400)
                return
            data = get_trade_depth_stats(exchange)
            self._json(data)
        elif path == "/api/health":
            self._json({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})
        else:
            self._json({"error": "Not found"}, 404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="BTC Live Monitoring Dashboard")
    parser.add_argument("--port", type=int, default=8888, help="HTTP server port")
    parser.add_argument("--minutes", type=int, default=120, help="Minutes of historical data to load")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    LOGGER.info("Starting BTC Live Dashboard on http://localhost:%d", args.port)
    LOGGER.info("Reading data from:")
    for name, cfg in EXCHANGES.items():
        LOGGER.info("  %s: %s", name, cfg["dir"])

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    LOGGER.info("Dashboard ready. Open http://localhost:%d in your browser.", args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
