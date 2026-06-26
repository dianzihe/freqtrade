from types import SimpleNamespace

import pandas as pd

from user_data.strategies.lob_risk_filter import (
    LobHmmConfig,
    LobHmmRiskFilter,
    LobRiskConfig,
    LobRiskFilter,
    LobRiskFilterMixin,
    apply_lob_signal_to_dataframe,
)


def _orderbook(bid: float = 100.0, ask: float = 100.1, size: float = 10.0) -> dict:
    return {
        "bids": [[bid - idx * 0.01, size] for idx in range(5)],
        "asks": [[ask + idx * 0.01, size] for idx in range(5)],
    }


class _FakeHmm:
    fit_calls = 0

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, features) -> "_FakeHmm":
        type(self).fit_calls += 1
        return self

    def predict_proba(self, features):
        last = features[-1]
        if last[1] < -0.5 or last[3] > 0.5:
            return [[0.20, 0.45, 0.35]]
        return [[0.96, 0.03, 0.01]]


class _AlwaysStableHmm(_FakeHmm):
    def predict_proba(self, features):
        return [[0.97, 0.02, 0.01]]


def test_stable_orderbook_does_not_trigger_after_warmup() -> None:
    config = LobRiskConfig(
        short_window=3,
        long_window=8,
        threshold_window=8,
        min_history=8,
        min_gap=2,
    )
    detector = LobRiskFilter(config)

    signals = [detector.update_from_orderbook("BTC/USDT", _orderbook()) for _ in range(20)]

    assert all(signal.ready for signal in signals[7:])
    assert not any(signal.triggered for signal in signals)
    assert signals[-1].score == 0.0


def test_depth_collapse_with_spread_widening_triggers_on_rising_edge() -> None:
    config = LobRiskConfig(
        short_window=3,
        long_window=8,
        threshold_window=8,
        min_history=8,
        depth_erosion_scale=0.20,
        spread_drift_scale=15.0,
        min_gap=2,
    )
    detector = LobRiskFilter(config)

    for _ in range(12):
        detector.update_from_orderbook("BTC/USDT", _orderbook(size=10.0))

    degraded = [
        detector.update_from_orderbook("BTC/USDT", _orderbook(bid=99.5, ask=100.5, size=3.0))
        for _ in range(4)
    ]

    assert any(signal.triggered for signal in degraded)
    assert degraded[-1].score > 1.0
    assert degraded[-1].depth_erosion > 0
    assert degraded[-1].spread_drift_bps > 0


def test_malformed_orderbook_returns_neutral_signal() -> None:
    detector = LobRiskFilter(LobRiskConfig(min_history=3))

    signal = detector.update_from_orderbook("BTC/USDT", {"bids": [], "asks": []})

    assert signal.pair == "BTC/USDT"
    assert signal.ready is False
    assert signal.triggered is False
    assert signal.reason == "invalid_orderbook"


def test_apply_lob_signal_to_dataframe_updates_latest_row_only() -> None:
    detector = LobRiskFilter(LobRiskConfig(min_history=1))
    dataframe = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    signal = detector.update_from_orderbook("BTC/USDT", _orderbook(size=4.0))

    result = apply_lob_signal_to_dataframe(dataframe, signal)

    assert result["lob_risk_trigger"].iloc[:-1].tolist() == [False, False]
    assert result.loc[result.index[-1], "lob_score"] == signal.score
    assert result.loc[result.index[-1], "lob_spread_bps"] == signal.spread_bps


def test_mixin_populates_live_orderbook_risk_columns_and_blocks_entries() -> None:
    class DummyDataProvider:
        runmode = SimpleNamespace(value="dry_run")

        def __init__(self) -> None:
            self.calls = 0

        def orderbook(self, pair: str, maximum: int) -> dict:
            self.calls += 1
            if self.calls <= 12:
                return _orderbook(size=10.0)
            return _orderbook(bid=99.5, ask=100.5, size=3.0)

    class DummyStrategy(LobRiskFilterMixin):
        lob_risk_config = LobRiskConfig(
            short_window=3,
            long_window=8,
            threshold_window=8,
            min_history=8,
            depth_erosion_scale=0.20,
            spread_drift_scale=15.0,
        )

        def __init__(self) -> None:
            self.dp = DummyDataProvider()

    strategy = DummyStrategy()
    dataframe = pd.DataFrame({"close": [100.0]})

    for _ in range(16):
        dataframe = strategy.populate_lob_risk(dataframe.copy(), {"pair": "BTC/USDT"})

    assert "lob_risk_trigger" in dataframe.columns
    assert dataframe.loc[dataframe.index[-1], "lob_risk_trigger"] is True
    assert strategy.is_lob_risk_blocked(dataframe) is True


def test_hmm_filter_is_safe_when_hmm_dependency_is_unavailable() -> None:
    detector = LobHmmRiskFilter(LobHmmConfig(hmm_min_samples=4), hmm_model_cls=None)

    signals = [detector.update_from_orderbook("BTC/USDT", _orderbook()) for _ in range(6)]

    assert signals[-1].ready is False
    assert signals[-1].triggered is False
    assert signals[-1].reason == "hmm_unavailable"
    assert signals[-1].hmm_ready is False


def test_hmm_filter_emits_entropy_and_triggers_on_latent_stress() -> None:
    config = LobHmmConfig(
        short_window=3,
        long_window=8,
        threshold_window=8,
        min_history=8,
        min_gap=2,
        hmm_min_samples=8,
        hmm_retrain_interval=20,
        entropy_scale=0.25,
        prestress_scale=0.30,
    )
    detector = LobHmmRiskFilter(config, hmm_model_cls=_FakeHmm)

    for _ in range(12):
        signal = detector.update_from_orderbook("BTC/USDT", _orderbook(size=10.0))
    assert signal.hmm_ready is True
    assert signal.hmm_state == 0

    degraded = [
        detector.update_from_orderbook("BTC/USDT", _orderbook(bid=99.5, ask=100.5, size=3.0))
        for _ in range(4)
    ]

    assert any(signal.triggered for signal in degraded)
    assert degraded[-1].hmm_entropy > 0.50
    assert degraded[-1].hmm_prestress_probability == 0.45
    assert degraded[-1].hmm_state == 1


def test_hmm_filter_retrains_on_schedule_not_every_tick() -> None:
    _AlwaysStableHmm.fit_calls = 0
    config = LobHmmConfig(
        short_window=3,
        long_window=8,
        threshold_window=8,
        min_history=8,
        hmm_min_samples=8,
        hmm_retrain_interval=5,
    )
    detector = LobHmmRiskFilter(config, hmm_model_cls=_AlwaysStableHmm)

    for _ in range(18):
        detector.update_from_orderbook("BTC/USDT", _orderbook())

    assert _AlwaysStableHmm.fit_calls == 3


def test_mixin_hmm_mode_populates_hmm_columns() -> None:
    class DummyDataProvider:
        runmode = SimpleNamespace(value="dry_run")

        def __init__(self) -> None:
            self.calls = 0

        def orderbook(self, pair: str, maximum: int) -> dict:
            self.calls += 1
            if self.calls <= 12:
                return _orderbook(size=10.0)
            return _orderbook(bid=99.5, ask=100.5, size=3.0)

    class DummyStrategy(LobRiskFilterMixin):
        lob_filter_mode = "hmm"
        lob_risk_config = LobHmmConfig(
            short_window=3,
            long_window=8,
            threshold_window=8,
            min_history=8,
            hmm_min_samples=8,
            entropy_scale=0.25,
            prestress_scale=0.30,
        )
        lob_hmm_model_cls = _FakeHmm

        def __init__(self) -> None:
            self.dp = DummyDataProvider()

    strategy = DummyStrategy()
    dataframe = pd.DataFrame({"close": [100.0]})

    for _ in range(16):
        dataframe = strategy.populate_lob_risk(dataframe.copy(), {"pair": "BTC/USDT"})

    assert dataframe.loc[dataframe.index[-1], "lob_hmm_ready"] is True
    assert dataframe.loc[dataframe.index[-1], "lob_hmm_entropy"] > 0.50
    assert dataframe.loc[dataframe.index[-1], "lob_hmm_state"] == 1
