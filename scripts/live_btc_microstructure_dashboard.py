#!/usr/bin/env python3
"""Serve a live BTC microstructure dashboard from cex_tick_recorder parquet data."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - exercised through runtime diagnostics.
    GaussianHMM = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - sklearn is optional for this script.
    StandardScaler = None


LOGGER = logging.getLogger("btc_microstructure_dashboard")

EXCHANGES: dict[str, dict[str, Any]] = {
    "gate": {
        "label": "Gate.io",
        "pair": "BTC_USDT",
        "dir": Path("user_data/orderbook_data/gate/spot/BTC_USDT"),
    },
    "binance": {
        "label": "Binance",
        "pair": "BTCUSDT",
        "dir": Path("user_data/orderbook_data/binance/spot/BTCUSDT"),
    },
}

CH1 = "ch1"
CH2 = "ch2"
CH3 = "ch3"
CH4 = "ch4"
DEFAULT_DUCKDB_PATH = Path("user_data/orderbook_data/btc_microstructure.duckdb")
LEGACY_ALGO_VERSION = "live_v1"
PAPERLIKE_ALGO_VERSION = "live_v2_paperlike"
DEFAULT_ALGO_VERSION = PAPERLIKE_ALGO_VERSION
EDGE_DIFF_STEPS = 3
SIGNAL_QUANTILE = 0.85
DRIFT_RISE_THR = 0.30
GAMMA_AMP = 0.35
MIN_TRIGGER_GAP = 5
MULTI_CHANNEL_FLOOR = 0.05


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
    max_files: int | None = 24,
    live_only: bool = True,
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


def read_dataset(base_dir: Path, dataset: str, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    max_files = 24 if live_only else None
    for path in recent_parquets(base_dir, dataset, minutes, max_files=max_files, live_only=live_only):
        try:
            frames.append(pq.read_table(path).to_pandas())
        except Exception as exc:
            LOGGER.debug("Skipping %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "exchange_time_ms" not in df.columns or df.empty:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["exchange_time_ms"], unit="ms", utc=True)
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(minutes=minutes)
    return df[df["time"] >= cutoff].sort_values("time")


def read_trades_1m(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
    trades = read_dataset(base_dir, "trades", minutes, live_only=live_only)
    if trades.empty:
        return pd.DataFrame()

    trades = trades.set_index("time")
    ohlc = trades["price"].resample("1min").ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = trades["amount"].resample("1min").sum()
    ohlc["trade_count"] = trades["price"].resample("1min").count()
    ohlc["buy_volume"] = trades["amount"].where(trades["side"] == "buy", 0.0).resample("1min").sum()
    ohlc["sell_volume"] = trades["amount"].where(trades["side"] == "sell", 0.0).resample("1min").sum()
    ohlc = ohlc.dropna(subset=["close"]).reset_index().rename(columns={"time": "date"})
    return ohlc


def read_depth_1m(base_dir: Path, minutes: int, *, live_only: bool = True) -> pd.DataFrame:
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
        "spread",
        "mid_price",
        "imbalance",
    ]
    for col in wanted:
        if col not in depth.columns:
            depth[col] = np.nan

    agg = depth.set_index("time").resample("1min").agg(
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
            "spread": "mean",
            "mid_price": "last",
            "imbalance": "mean",
        }
    )
    return agg.reset_index().rename(columns={"time": "date"})


def _norm01(series: pd.Series, window: int = 50) -> pd.Series:
    floor = series.rolling(window, min_periods=max(5, window // 4)).quantile(0.05)
    cap = series.rolling(window, min_periods=max(5, window // 4)).quantile(0.95)
    scaled = (series - floor) / (cap - floor).replace(0, np.nan)
    return scaled.clip(0, 1).fillna(0.0)


def _score_entropy(channels: pd.DataFrame) -> pd.Series:
    positive = channels.clip(lower=0.0)
    total = positive.sum(axis=1).replace(0, np.nan)
    probs = positive.div(total, axis=0).replace(0, np.nan)
    entropy = -(probs * np.log(probs.clip(lower=1e-12))).sum(axis=1)
    return (entropy / np.log(len(channels.columns))).fillna(0.0).clip(0, 1)


def _hmm_regime_channel(features: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return a HMM posterior entropy/pre-stress channel aligned to features."""
    result = pd.Series(0.0, index=features.index)
    clean = features.replace([np.inf, -np.inf], np.nan).dropna()
    if GaussianHMM is None or StandardScaler is None:
        return result, "proxy"
    if len(clean) < 30:
        return result, "hmm_insufficient"

    try:
        scaled = StandardScaler().fit_transform(clean)
        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=100,
            random_state=7,
            min_covar=1e-4,
        )
        model.fit(scaled)
        posterior = pd.DataFrame(model.predict_proba(scaled), index=clean.index)
    except Exception as exc:
        LOGGER.debug("HMM regime channel failed: %s", exc)
        return result, "hmm_failed"

    entropy = -(posterior * np.log(posterior.clip(lower=1e-12))).sum(axis=1) / np.log(3)
    spread_ranked_states = np.argsort(model.means_[:, 0])
    prestress_state = int(spread_ranked_states[1])
    prestress = posterior[prestress_state].rolling(5, min_periods=1).mean()
    result.loc[clean.index] = pd.concat([entropy, prestress], axis=1).max(axis=1).clip(0, 1)
    return result.fillna(0.0), "hmm"


def _deduplicate_triggers(raw_trigger: pd.Series, min_gap: int = MIN_TRIGGER_GAP) -> pd.Series:
    trigger = pd.Series(0, index=raw_trigger.index, dtype=int)
    last_idx: int | None = None
    for idx, is_trigger in raw_trigger.fillna(False).items():
        if not is_trigger:
            continue
        pos = int(raw_trigger.index.get_loc(idx))
        if last_idx is None or pos - last_idx >= min_gap:
            trigger.loc[idx] = 1
            last_idx = pos
    return trigger


def compute_signals_v1(ohlcv: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()

    df = ohlcv.copy()
    returns = df["close"].pct_change()
    vol_s = returns.rolling(5, min_periods=3).std()
    vol_m = returns.rolling(12, min_periods=6).std()
    vol_l = returns.rolling(24, min_periods=12).std()
    total_vol = (vol_s + vol_m + vol_l).replace(0, np.nan)
    probs = [vol_s / total_vol, vol_m / total_vol, vol_l / total_vol]
    entropy = sum(-(p * np.log(p.clip(lower=1e-12))) for p in probs)
    df[CH1] = (entropy / np.log(3)).fillna(0.0).clip(0, 1)

    if not depth.empty:
        merged = df[["date"]].merge(depth[["date", "total_depth", "spread", "mid_price"]], on="date", how="left")
        td = merged["total_depth"].ffill(limit=10)
        mean_depth = td.rolling(48, min_periods=12).mean()
        std_depth = td.rolling(48, min_periods=12).std().replace(0, np.nan)
        df[CH2] = (-(td - mean_depth) / std_depth).clip(lower=0).fillna(0.0).clip(0, 3) / 3.0

        spread_ratio = (merged["spread"] / merged["mid_price"].replace(0, np.nan)).fillna(0.0)
        spread_mean = spread_ratio.rolling(24, min_periods=8).mean()
        spread_std = spread_ratio.rolling(24, min_periods=8).std().replace(0, np.nan)
        df[CH3] = ((spread_ratio - spread_mean) / spread_std).clip(lower=0).fillna(0.0).clip(0, 3) / 3.0
    else:
        df[CH2] = 0.0
        df[CH3] = 0.0

    flow = df["buy_volume"].fillna(0.0) - df["sell_volume"].fillna(0.0)
    flow_total = (df["buy_volume"].fillna(0.0) + df["sell_volume"].fillna(0.0)).replace(0, np.nan)
    df[CH4] = (flow / flow_total).fillna(0.0).clip(-1, 1)

    df["signal_value"] = df[[CH1, CH2, CH3]].join(df[CH4].abs()).max(axis=1)
    rolling_threshold = df["signal_value"].rolling(48, min_periods=12).quantile(0.88)
    df["threshold"] = rolling_threshold.clip(lower=0.40).fillna(0.40)
    df["trigger"] = ((df["signal_value"] >= df["threshold"]) & (df["signal_value"] >= 0.40)).astype(int)
    df["score_delta"] = df["signal_value"].diff().fillna(0.0)
    df["dominant_channel"] = df[[CH1, CH2, CH3]].join(df[CH4].abs()).idxmax(axis=1)
    df["nonzero_channel_count"] = (df[[CH1, CH2, CH3]].join(df[CH4].abs()) > MULTI_CHANNEL_FLOOR).sum(axis=1)
    df["data_quality"] = np.where(depth.empty, "missing_depth", "ok")
    df["regime_model"] = "legacy"
    return df


def compute_signals_paperlike(ohlcv: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()

    df = ohlcv.copy()
    if not depth.empty:
        merge_cols = ["date", "total_depth", "spread", "mid_price"]
        available = [c for c in merge_cols if c in depth.columns]
        merged = df[["date"]].merge(depth[available], on="date", how="left")
    else:
        merged = df[["date"]].copy()
        merged["total_depth"] = np.nan
        merged["spread"] = np.nan
        merged["mid_price"] = np.nan

    total_depth = merged["total_depth"].ffill(limit=3)
    mid_price = merged["mid_price"].fillna(df["close"]).replace(0, np.nan)
    spread_value = merged["spread"].ffill(limit=3)
    spread_ratio = (spread_value / mid_price).replace([np.inf, -np.inf], np.nan)

    returns = df["close"].pct_change().fillna(0.0)
    roll_vol = returns.rolling(10, min_periods=5).std().fillna(0.0)
    flow_total = (df["buy_volume"].fillna(0.0) + df["sell_volume"].fillna(0.0)).replace(0, np.nan)
    imbalance = ((df["buy_volume"].fillna(0.0) - df["sell_volume"].fillna(0.0)) / flow_total).fillna(0.0).clip(-1, 1)
    ofi = (imbalance * df["volume"].fillna(0.0)).fillna(0.0)

    depth_velocity = -total_depth.diff(5)
    depth_below_mean = ((total_depth.rolling(20, min_periods=8).mean() - total_depth) / total_depth.rolling(20, min_periods=8).mean()).clip(lower=0)
    df[CH2] = ((_norm01(depth_velocity.clip(lower=0), 30) + _norm01(depth_below_mean, 30)) / 2).clip(0, 1)

    spread_short = spread_ratio.rolling(5, min_periods=3).mean()
    spread_long = spread_ratio.rolling(20, min_periods=8).mean()
    spread_cross = ((spread_short - spread_long) / spread_long.replace(0, np.nan)).clip(lower=0)
    spread_momentum = spread_ratio.diff(3).clip(lower=0)
    df[CH3] = ((_norm01(spread_cross, 30) + _norm01(spread_momentum, 30)) / 2).clip(0, 1)

    ofi_strength = ofi.abs().rolling(10, min_periods=5).mean()
    imbalance_strength = imbalance.abs().rolling(10, min_periods=5).mean()
    df[CH4] = ((_norm01(ofi_strength, 30) + _norm01(imbalance_strength, 30)) / 2).clip(0, 1)

    entropy_inputs = pd.DataFrame(
        {
            "volatility": _norm01(roll_vol, 30),
            "depth": df[CH2],
            "spread": df[CH3],
            "flow": df[CH4],
        },
        index=df.index,
    )
    hmm_features = pd.DataFrame(
        {
            "spread": spread_ratio,
            "depth": total_depth,
            "imbalance": imbalance,
            "roll_vol": roll_vol,
            "ofi": ofi,
        },
        index=df.index,
    )
    hmm_ch1, regime_model = _hmm_regime_channel(hmm_features)
    if regime_model == "hmm":
        df[CH1] = hmm_ch1
    else:
        df[CH1] = _score_entropy(entropy_inputs)
    df["regime_model"] = regime_model

    channel_values = df[[CH1, CH2, CH3, CH4]].fillna(0.0).clip(0, 1)
    score_raw = channel_values.max(axis=1)
    df["signal_value"] = (score_raw * (1 + GAMMA_AMP * (df[CH3] > DRIFT_RISE_THR).astype(float))).clip(0, 1.35)
    df["score_delta"] = df["signal_value"].diff(EDGE_DIFF_STEPS).fillna(0.0)
    threshold = df["signal_value"].rolling(48, min_periods=20).quantile(SIGNAL_QUANTILE)
    df["threshold"] = threshold.fillna(np.inf)
    df["dominant_channel"] = channel_values.idxmax(axis=1)
    df["nonzero_channel_count"] = (channel_values > MULTI_CHANNEL_FLOOR).sum(axis=1)
    enough_history = df["threshold"].replace(np.inf, np.nan).notna()
    has_depth = total_depth.notna()
    stale_depth = (
        (total_depth.rolling(10, min_periods=10).std().fillna(np.nan) <= 1e-9)
        & (spread_value.rolling(10, min_periods=10).std().fillna(np.nan) <= 1e-12)
        & enough_history
    )
    df["data_quality"] = np.select(
        [~enough_history, ~has_depth, stale_depth],
        ["warmup", "missing_depth", "stale_depth"],
        default="ok",
    )

    above = df["signal_value"] > df["threshold"]
    rising = df["score_delta"] > 0
    confirmed = df["nonzero_channel_count"] >= 2
    raw_trigger = above & rising & confirmed & (df["data_quality"] == "ok")
    df["trigger"] = _deduplicate_triggers(raw_trigger)
    df["threshold"] = df["threshold"].replace(np.inf, np.nan).fillna(0.0)
    return df


def compute_signals(
    ohlcv: pd.DataFrame,
    depth: pd.DataFrame,
    *,
    algo_version: str = DEFAULT_ALGO_VERSION,
) -> pd.DataFrame:
    if algo_version == LEGACY_ALGO_VERSION:
        return compute_signals_v1(ohlcv, depth)
    if algo_version == PAPERLIKE_ALGO_VERSION:
        return compute_signals_paperlike(ohlcv, depth)
    raise ValueError(f"unknown algo_version: {algo_version}")


def build_signal_rows(signals: pd.DataFrame, depth: pd.DataFrame, *, only_triggers: bool = True) -> list[dict[str, Any]]:
    if signals.empty:
        return []

    cols = [
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
        "spread",
        "mid_price",
    ]
    source = signals.copy()
    if not depth.empty:
        available = [c for c in cols if c in depth.columns]
        source = source.merge(depth[available], on="date", how="left")
    else:
        for col in cols[1:]:
            source[col] = np.nan

    if only_triggers:
        source = source[source["trigger"] == 1]

    rows: list[dict[str, Any]] = []
    for _, row in source.sort_values("date").iterrows():
        bid_5 = _json_float(row.get("bid_depth_5bps")) or 0.0
        ask_5 = _json_float(row.get("ask_depth_5bps")) or 0.0
        bid_10 = _json_float(row.get("bid_depth_10bps")) or 0.0
        ask_10 = _json_float(row.get("ask_depth_10bps")) or 0.0
        rows.append(
            {
                "time_ms": _time_ms(row["date"]),
                "time": _iso_time(row["date"]),
                "price": _json_float(row.get("close")),
                "ch1": _json_float(row.get(CH1)),
                "ch2": _json_float(row.get(CH2)),
                "ch3": _json_float(row.get(CH3)),
                "ch4": _json_float(row.get(CH4)),
                "signal_value": _json_float(row.get("signal_value")),
                "threshold": _json_float(row.get("threshold")),
                "trigger": _json_int(row.get("trigger")),
                "score_delta": _json_float(row.get("score_delta")),
                "dominant_channel": row.get("dominant_channel") if pd.notna(row.get("dominant_channel")) else None,
                "nonzero_channel_count": _json_int(row.get("nonzero_channel_count")),
                "data_quality": row.get("data_quality") if pd.notna(row.get("data_quality")) else None,
                "regime_model": row.get("regime_model") if pd.notna(row.get("regime_model")) else None,
                "l2_bid_depth": _json_float(row.get("bid_depth")),
                "l2_ask_depth": _json_float(row.get("ask_depth")),
                "l2_total_depth": _json_float(row.get("total_depth")),
                "l2_total_depth_5bps": _json_float(row.get("total_depth_5bps")) or bid_5 + ask_5,
                "l2_total_depth_10bps": _json_float(row.get("total_depth_10bps")) or bid_10 + ask_10,
                "spread": _json_float(row.get("spread")),
                "mid_price": _json_float(row.get("mid_price")),
            }
        )
    return rows


def build_exchange_payload(
    exchange: str,
    minutes: int,
    *,
    algo_version: str = DEFAULT_ALGO_VERSION,
    live_only: bool = True,
) -> dict[str, Any]:
    cfg = EXCHANGES[exchange]
    trades = read_trades_1m(cfg["dir"], minutes, live_only=live_only)
    depth = read_depth_1m(cfg["dir"], minutes, live_only=live_only)
    signals = compute_signals(trades, depth, algo_version=algo_version)
    latest_rows = build_signal_rows(signals.tail(1), depth, only_triggers=False)
    trigger_rows = build_signal_rows(signals, depth, only_triggers=True)[-50:]
    all_rows = build_signal_rows(signals.tail(180), depth, only_triggers=False)

    latest_depth = build_signal_rows(signals.tail(1), depth, only_triggers=False)
    return {
        "exchange": exchange,
        "label": cfg["label"],
        "pair": cfg["pair"],
        "algo_version": algo_version,
        "updated": datetime.now(UTC).isoformat(),
        "data_available": not trades.empty,
        "latest": latest_rows[-1] if latest_rows else None,
        "latest_depth": latest_depth[-1] if latest_depth else None,
        "recent_signals": trigger_rows,
        "series": all_rows,
    }


def build_snapshot(
    minutes: int,
    *,
    algo_version: str = DEFAULT_ALGO_VERSION,
    live_only: bool = True,
) -> dict[str, Any]:
    return {
        "updated": datetime.now(UTC).isoformat(),
        "refresh_ms": 60000,
        "algo_version": algo_version,
        "exchanges": {
            name: build_exchange_payload(name, minutes, algo_version=algo_version, live_only=live_only)
            for name in EXCHANGES
        },
    }


def _parquet_glob(exchange: str, dataset: str) -> str:
    base = EXCHANGES[exchange]["dir"] / dataset / "date=*" / "hour=*" / "*.parquet"
    return base.as_posix()


def ensure_duckdb_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        create table if not exists btc_signal_features_1m (
            event_time timestamptz not null,
            time_ms bigint not null,
            exchange varchar not null,
            pair varchar not null,
            price double,
            ch1 double,
            ch2 double,
            ch3 double,
            ch4 double,
            signal_value double,
            threshold double,
            trigger integer,
            score_delta double,
            dominant_channel varchar,
            nonzero_channel_count integer,
            data_quality varchar,
            regime_model varchar,
            l2_bid_depth double,
            l2_ask_depth double,
            l2_total_depth double,
            l2_total_depth_5bps double,
            l2_total_depth_10bps double,
            spread double,
            mid_price double,
            algo_version varchar not null,
            created_at timestamptz not null
        )
        """
    )
    for column_sql in (
        "score_delta double",
        "dominant_channel varchar",
        "nonzero_channel_count integer",
        "data_quality varchar",
        "regime_model varchar",
    ):
        con.execute(f"alter table btc_signal_features_1m add column if not exists {column_sql}")
    con.execute(
        """
        create unique index if not exists idx_btc_signal_features_1m_key
        on btc_signal_features_1m (time_ms, exchange, algo_version)
        """
    )
    for exchange in EXCHANGES:
        for dataset in ("l1", "l2", "trades", "depth"):
            view_name = f"{exchange}_{dataset}"
            con.execute(
                f"""
                create or replace view {view_name} as
                select * from read_parquet('{_parquet_glob(exchange, dataset)}', union_by_name=true)
                """
            )


def flatten_signal_features(
    snapshot: dict[str, Any],
    *,
    algo_version: str = DEFAULT_ALGO_VERSION,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    created_at = created_at or datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for exchange, payload in snapshot.get("exchanges", {}).items():
        pair = payload.get("pair")
        for row in payload.get("series") or []:
            rows.append(
                {
                    "event_time": row.get("time"),
                    "time_ms": row.get("time_ms"),
                    "exchange": exchange,
                    "pair": pair,
                    "price": row.get("price"),
                    "ch1": row.get("ch1"),
                    "ch2": row.get("ch2"),
                    "ch3": row.get("ch3"),
                    "ch4": row.get("ch4"),
                    "signal_value": row.get("signal_value"),
                    "threshold": row.get("threshold"),
                    "trigger": row.get("trigger"),
                    "score_delta": row.get("score_delta"),
                    "dominant_channel": row.get("dominant_channel"),
                    "nonzero_channel_count": row.get("nonzero_channel_count"),
                    "data_quality": row.get("data_quality"),
                    "regime_model": row.get("regime_model"),
                    "l2_bid_depth": row.get("l2_bid_depth"),
                    "l2_ask_depth": row.get("l2_ask_depth"),
                    "l2_total_depth": row.get("l2_total_depth"),
                    "l2_total_depth_5bps": row.get("l2_total_depth_5bps"),
                    "l2_total_depth_10bps": row.get("l2_total_depth_10bps"),
                    "spread": row.get("spread"),
                    "mid_price": row.get("mid_price"),
                    "algo_version": algo_version,
                    "created_at": created_at,
                }
            )
    return rows


def persist_signal_feature_rows(rows: list[dict[str, Any]], *, db_path: Path = DEFAULT_DUCKDB_PATH) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        ensure_duckdb_schema(con)
        if not rows:
            return 0
        incoming = pd.DataFrame(rows)
        con.register("incoming_signal_features", incoming)
        con.execute(
            """
            delete from btc_signal_features_1m
            using incoming_signal_features
            where btc_signal_features_1m.time_ms = incoming_signal_features.time_ms
              and btc_signal_features_1m.exchange = incoming_signal_features.exchange
              and btc_signal_features_1m.algo_version = incoming_signal_features.algo_version
            """
        )
        con.execute(
            """
            insert into btc_signal_features_1m (
                event_time, time_ms, exchange, pair, price,
                ch1, ch2, ch3, ch4, signal_value, threshold, trigger,
                score_delta, dominant_channel, nonzero_channel_count, data_quality,
                regime_model,
                l2_bid_depth, l2_ask_depth, l2_total_depth,
                l2_total_depth_5bps, l2_total_depth_10bps,
                spread, mid_price, algo_version, created_at
            )
            select
                cast(event_time as timestamptz), time_ms, exchange, pair, price,
                ch1, ch2, ch3, ch4, signal_value, threshold, trigger,
                score_delta, dominant_channel, nonzero_channel_count, data_quality,
                regime_model,
                l2_bid_depth, l2_ask_depth, l2_total_depth,
                l2_total_depth_5bps, l2_total_depth_10bps,
                spread, mid_price, algo_version, cast(created_at as timestamptz)
            from incoming_signal_features
            """
        )
    return len(rows)


def persist_snapshot_features(
    snapshot: dict[str, Any],
    *,
    db_path: Path = DEFAULT_DUCKDB_PATH,
    algo_version: str = DEFAULT_ALGO_VERSION,
) -> int:
    rows = flatten_signal_features(snapshot, algo_version=algo_version)
    return persist_signal_feature_rows(rows, db_path=db_path)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC 微结构信号监控</title>
<style>
body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f3f5f7; color: #17202a; }
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
    <h1>BTC 微结构信号监控</h1>
    <div class="sub">Gate.io + Binance | 发出时间、信号值、CH1-CH4、L2订单簿深度 | 每分钟自动刷新</div>
  </div>
  <div id="status">加载中...</div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="panel">
    <h2>信号触发记录</h2>
    <div style="overflow:auto; max-height:420px">
      <table>
        <thead>
          <tr>
            <th>发出时间</th><th>交易所</th><th>价格</th><th>信号值</th><th>阈值</th>
            <th>CH1</th><th>CH2</th><th>CH3</th><th>CH4</th>
            <th>L2买盘</th><th>L2卖盘</th><th>L2总深度</th><th>5bps深度</th><th>10bps深度</th><th>价差</th>
          </tr>
        </thead>
        <tbody id="signals"><tr><td colspan="15" class="muted">等待数据...</td></tr></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const REFRESH_MS = 60000;
const fmt = (v, n = 3) => v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(n);
const time = ms => ms ? new Date(ms).toLocaleString("zh-CN", {hour12:false}) : "-";

function metric(name, value, digits = 3, cls = "") {
  return `<div class="metric"><b>${name}</b><span class="${cls}">${fmt(value, digits)}</span></div>`;
}

function renderCard(payload) {
  const latest = payload.latest || {};
  const cls = latest.trigger === 1 ? "warn" : "ok";
  return `<article class="card">
    <h2>${payload.label} <span class="muted">${payload.pair}</span></h2>
    <div class="muted">最近发出时间: ${time(latest.time_ms)}</div>
    <div class="metrics">
      ${metric("信号值", latest.signal_value, 3, cls)}
      ${metric("CH1", latest.ch1)}
      ${metric("CH2", latest.ch2)}
      ${metric("CH3", latest.ch3)}
      ${metric("CH4", latest.ch4)}
      ${metric("阈值", latest.threshold)}
      ${metric("价格", latest.price, 2)}
      ${metric("触发", latest.trigger || 0, 0, cls)}
    </div>
    <div class="metrics depth">
      ${metric("L2买盘深度", latest.l2_bid_depth, 2)}
      ${metric("L2卖盘深度", latest.l2_ask_depth, 2)}
      ${metric("L2总深度", latest.l2_total_depth, 2)}
      ${metric("5bps深度", latest.l2_total_depth_5bps, 2)}
      ${metric("10bps深度", latest.l2_total_depth_10bps, 2)}
    </div>
  </article>`;
}

function renderRows(snapshot) {
  const rows = [];
  for (const [exchange, payload] of Object.entries(snapshot.exchanges)) {
    for (const row of payload.recent_signals || []) rows.push({...row, exchange: payload.label});
  }
  rows.sort((a, b) => b.time_ms - a.time_ms);
  if (!rows.length) return `<tr><td colspan="15" class="muted">当前窗口暂无触发信号，顶部卡片仍显示最新分钟的信号量和L2深度。</td></tr>`;
  return rows.map(r => `<tr>
    <td>${time(r.time_ms)}</td><td>${r.exchange}</td><td>${fmt(r.price, 2)}</td><td class="warn">${fmt(r.signal_value)}</td><td>${fmt(r.threshold)}</td>
    <td>${fmt(r.ch1)}</td><td>${fmt(r.ch2)}</td><td>${fmt(r.ch3)}</td><td>${fmt(r.ch4)}</td>
    <td>${fmt(r.l2_bid_depth, 2)}</td><td>${fmt(r.l2_ask_depth, 2)}</td><td>${fmt(r.l2_total_depth, 2)}</td>
    <td>${fmt(r.l2_total_depth_5bps, 2)}</td><td>${fmt(r.l2_total_depth_10bps, 2)}</td><td>${fmt(r.spread, 4)}</td>
  </tr>`).join("");
}

async function refresh() {
  const response = await fetch("/api/snapshot", {cache: "no-store"});
  const snapshot = await response.json();
  document.getElementById("cards").innerHTML = Object.values(snapshot.exchanges).map(renderCard).join("");
  document.getElementById("signals").innerHTML = renderRows(snapshot);
  document.getElementById("status").textContent = `刷新: ${new Date().toLocaleTimeString("zh-CN", {hour12:false})}`;
}

refresh().catch(err => document.getElementById("status").textContent = `加载失败: ${err}`);
setInterval(() => refresh().catch(console.error), REFRESH_MS);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    minutes = 180
    algo_version = DEFAULT_ALGO_VERSION

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug("%s - %s", self.client_address[0], fmt % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        minutes = int(params.get("minutes", [self.minutes])[0])
        algo_version = params.get("algo_version", [self.algo_version])[0]
        if parsed.path in {"/", "/index.html"}:
            self._send(HTML_TEMPLATE, "text/html; charset=utf-8")
        elif parsed.path == "/api/snapshot":
            snapshot = build_snapshot(minutes, algo_version=algo_version)
            try:
                persist_snapshot_features(snapshot, algo_version=algo_version)
            except Exception as exc:
                LOGGER.warning("failed to persist signal features: %s", exc)
            self._send_json(snapshot)
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
    parser = argparse.ArgumentParser(description="BTC microstructure signal dashboard")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--minutes", type=int, default=180)
    parser.add_argument("--algo-version", default=DEFAULT_ALGO_VERSION)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DashboardHandler.minutes = args.minutes
    DashboardHandler.algo_version = args.algo_version
    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    LOGGER.info("Dashboard ready: http://localhost:%s", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
