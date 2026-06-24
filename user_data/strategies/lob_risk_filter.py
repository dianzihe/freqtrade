"""Lightweight limit-order-book risk filter for Freqtrade strategies.

Use this as a risk gate, not as an entry signal:

    dataframe = self.populate_lob_risk(dataframe, metadata)
    if not self.is_lob_risk_blocked(dataframe):
        dataframe.loc[entry_condition, "enter_long"] = 1

The filter is neutral outside live/dry-run because ordinary OHLCV backtests do
not contain historical order book snapshots.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import fmean
from typing import Deque

import pandas as pd


@dataclass(frozen=True)
class LobRiskConfig:
    orderbook_levels: int = 5
    short_window: int = 5
    long_window: int = 30
    threshold_window: int = 50
    threshold_percentile: float = 0.85
    min_history: int = 30
    min_gap: int = 5
    depth_erosion_scale: float = 0.15
    spread_drift_scale: float = 10.0


@dataclass(frozen=True)
class LobRiskSignal:
    pair: str
    ready: bool
    triggered: bool
    score: float = 0.0
    threshold: float = float("inf")
    spread_bps: float = 0.0
    total_depth: float = 0.0
    depth_erosion: float = 0.0
    spread_drift_bps: float = 0.0
    imbalance: float = 0.0
    reason: str = "ok"


@dataclass
class _PairState:
    spreads_bps: Deque[float]
    depths: Deque[float]
    scores: Deque[float]
    ticks_since_trigger: int = 1_000_000
    risk_ticks_remaining: int = 0
    last_score: float = 0.0


class LobRiskFilter:
    """Stateful per-pair detector for depth erosion and spread drift."""

    def __init__(self, config: LobRiskConfig | None = None) -> None:
        self.config = config or LobRiskConfig()
        maxlen = max(self.config.long_window, self.config.threshold_window, self.config.min_history) + 2
        self._states: defaultdict[str, _PairState] = defaultdict(
            lambda: _PairState(
                spreads_bps=deque(maxlen=maxlen),
                depths=deque(maxlen=maxlen),
                scores=deque(maxlen=maxlen),
            )
        )

    def update_from_orderbook(self, pair: str, orderbook: dict) -> LobRiskSignal:
        metrics = self._extract_metrics(orderbook)
        if metrics is None:
            return LobRiskSignal(pair=pair, ready=False, triggered=False, reason="invalid_orderbook")

        spread_bps, total_depth, imbalance = metrics
        state = self._states[pair]
        state.spreads_bps.append(spread_bps)
        state.depths.append(total_depth)

        if len(state.depths) < self.config.min_history:
            signal = LobRiskSignal(
                pair=pair,
                ready=False,
                triggered=False,
                spread_bps=spread_bps,
                total_depth=total_depth,
                imbalance=imbalance,
                reason="warming_up",
            )
            state.scores.append(signal.score)
            state.last_score = signal.score
            state.ticks_since_trigger += 1
            return signal

        depth_erosion = self._depth_erosion(state.depths)
        spread_drift_bps = self._spread_drift(state.spreads_bps)
        score = max(
            depth_erosion / max(self.config.depth_erosion_scale, 1e-12),
            spread_drift_bps / max(self.config.spread_drift_scale, 1e-12),
            0.0,
        )
        threshold = self._adaptive_threshold(state.scores)
        rising = score > state.last_score
        edge_triggered = (
            score > threshold
            and rising
            and state.ticks_since_trigger >= self.config.min_gap
        )

        if edge_triggered:
            state.ticks_since_trigger = 0
            state.risk_ticks_remaining = self.config.min_gap
        else:
            state.ticks_since_trigger += 1
            state.risk_ticks_remaining = max(state.risk_ticks_remaining - 1, 0)
        triggered = edge_triggered or state.risk_ticks_remaining > 0
        state.scores.append(score)
        state.last_score = score

        return LobRiskSignal(
            pair=pair,
            ready=True,
            triggered=triggered,
            score=score,
            threshold=threshold,
            spread_bps=spread_bps,
            total_depth=total_depth,
            depth_erosion=depth_erosion,
            spread_drift_bps=spread_drift_bps,
            imbalance=imbalance,
        )

    def _extract_metrics(self, orderbook: dict) -> tuple[float, float, float] | None:
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        if not bids or not asks:
            return None

        levels = self.config.orderbook_levels
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            bid_depth = sum(max(float(level[1]), 0.0) for level in bids[:levels])
            ask_depth = sum(max(float(level[1]), 0.0) for level in asks[:levels])
        except (TypeError, ValueError, IndexError):
            return None

        mid = (best_bid + best_ask) / 2.0
        total_depth = bid_depth + ask_depth
        if mid <= 0 or best_ask <= best_bid or total_depth <= 0:
            return None

        spread_bps = ((best_ask - best_bid) / mid) * 10_000.0
        imbalance = (bid_depth - ask_depth) / total_depth
        return spread_bps, total_depth, imbalance

    def _depth_erosion(self, depths: Deque[float]) -> float:
        short = list(depths)[-self.config.short_window :]
        long = list(depths)[-self.config.long_window :]
        long_mean = fmean(long)
        if long_mean <= 0:
            return 0.0
        return max((long_mean - fmean(short)) / long_mean, 0.0)

    def _spread_drift(self, spreads_bps: Deque[float]) -> float:
        short = list(spreads_bps)[-self.config.short_window :]
        long = list(spreads_bps)[-self.config.long_window :]
        return max(fmean(short) - fmean(long), 0.0)

    def _adaptive_threshold(self, scores: Deque[float]) -> float:
        if len(scores) < self.config.threshold_window:
            return float("inf")
        sample = sorted(list(scores)[-self.config.threshold_window :])
        pct = min(max(self.config.threshold_percentile, 0.0), 1.0)
        index = min(int(round((len(sample) - 1) * pct)), len(sample) - 1)
        return sample[index]


def apply_lob_signal_to_dataframe(dataframe: pd.DataFrame, signal: LobRiskSignal) -> pd.DataFrame:
    """Annotate the latest row with a LOB risk signal."""

    if dataframe.empty:
        return dataframe

    defaults = {
        "lob_risk_ready": False,
        "lob_risk_trigger": False,
        "lob_score": 0.0,
        "lob_threshold": float("inf"),
        "lob_spread_bps": 0.0,
        "lob_total_depth": 0.0,
        "lob_depth_erosion": 0.0,
        "lob_spread_drift_bps": 0.0,
        "lob_imbalance": 0.0,
        "lob_risk_reason": "not_updated",
    }
    for column, value in defaults.items():
        if column not in dataframe.columns:
            dataframe[column] = value

    last = dataframe.index[-1]
    dataframe.at[last, "lob_risk_ready"] = bool(signal.ready)
    dataframe.at[last, "lob_risk_trigger"] = bool(signal.triggered)
    dataframe.at[last, "lob_score"] = signal.score
    dataframe.at[last, "lob_threshold"] = signal.threshold
    dataframe.at[last, "lob_spread_bps"] = signal.spread_bps
    dataframe.at[last, "lob_total_depth"] = signal.total_depth
    dataframe.at[last, "lob_depth_erosion"] = signal.depth_erosion
    dataframe.at[last, "lob_spread_drift_bps"] = signal.spread_drift_bps
    dataframe.at[last, "lob_imbalance"] = signal.imbalance
    dataframe.at[last, "lob_risk_reason"] = signal.reason
    dataframe["lob_risk_ready"] = dataframe["lob_risk_ready"].astype(object)
    dataframe["lob_risk_trigger"] = dataframe["lob_risk_trigger"].astype(object)
    return dataframe


class LobRiskFilterMixin:
    """Mixin for Freqtrade strategies that want live/dry-run LOB risk columns."""

    lob_risk_config = LobRiskConfig()

    @property
    def lob_risk_filter(self) -> LobRiskFilter:
        detector = getattr(self, "_lob_risk_filter", None)
        if detector is None:
            detector = LobRiskFilter(self.lob_risk_config)
            self._lob_risk_filter = detector
        return detector

    def populate_lob_risk(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")
        signal = LobRiskSignal(pair=pair, ready=False, triggered=False, reason="inactive_runmode")

        dp = getattr(self, "dp", None)
        runmode = getattr(getattr(dp, "runmode", None), "value", None)
        if dp is not None and runmode in {"live", "dry_run"}:
            try:
                orderbook = dp.orderbook(pair, self.lob_risk_config.orderbook_levels)
                signal = self.lob_risk_filter.update_from_orderbook(pair, orderbook)
            except Exception:
                signal = LobRiskSignal(pair=pair, ready=False, triggered=False, reason="orderbook_error")

        return apply_lob_signal_to_dataframe(dataframe, signal)

    def is_lob_risk_blocked(self, dataframe: pd.DataFrame) -> bool:
        if dataframe.empty or "lob_risk_trigger" not in dataframe.columns:
            return False
        return bool(dataframe["lob_risk_trigger"].iloc[-1])
