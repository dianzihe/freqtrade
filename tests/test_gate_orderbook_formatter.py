import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "user_data" / "scripts" / "format_gate_orderbook_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("format_gate_orderbook_data", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_l2_replayer_applies_snapshot_deltas_deletes_and_sorts_levels() -> None:
    module = load_module()
    replayer = module.L2Replayer(levels=2)

    rows = [
        {
            "exchange_time_ms": 1_000,
            "first_update_id": 10,
            "update_id": 10,
            "bid_updates": '[["100","1"],["99","2"]]',
            "ask_updates": '[["101","1"],["102","2"]]',
        },
        {
            "exchange_time_ms": 1_250,
            "first_update_id": 11,
            "update_id": 12,
            "bid_updates": '[["100","0"],["98","3"]]',
            "ask_updates": '[["100.5","4"]]',
        },
    ]

    features = [replayer.apply(row) for row in rows]

    assert features[0]["best_bid"] == 100.0
    assert features[0]["best_ask"] == 101.0
    assert features[0]["bids"] == [[100.0, 1.0], [99.0, 2.0]]
    assert features[1]["best_bid"] == 99.0
    assert features[1]["best_ask"] == 100.5
    assert features[1]["bids"] == [[99.0, 2.0], [98.0, 3.0]]
    assert features[1]["asks"] == [[100.5, 4.0], [101.0, 1.0]]
    assert features[1]["sequence_gap"] is False


def test_l2_replayer_marks_sequence_gap() -> None:
    module = load_module()
    replayer = module.L2Replayer(levels=2)

    replayer.apply(
        {
            "exchange_time_ms": 1_000,
            "first_update_id": 10,
            "update_id": 10,
            "bid_updates": '[["100","1"]]',
            "ask_updates": '[["101","1"]]',
        }
    )
    feature = replayer.apply(
        {
            "exchange_time_ms": 2_000,
            "first_update_id": 15,
            "update_id": 15,
            "bid_updates": '[["100","2"]]',
            "ask_updates": "[]",
        }
    )

    assert feature["sequence_gap"] is True
    assert replayer.sequence_gap_count == 1


def test_resample_l2_features_uses_last_book_state_and_gap_count() -> None:
    module = load_module()
    features = pd.DataFrame(
        [
            {
                "exchange_time_ms": 1_000,
                "best_bid": 100.0,
                "best_ask": 101.0,
                "mid_price": 100.5,
                "spread": 1.0,
                "spread_bps": 99.5025,
                "bid_depth_5": 1.0,
                "ask_depth_5": 1.0,
                "bid_depth_10": 1.0,
                "ask_depth_10": 1.0,
                "bid_depth_25": 1.0,
                "ask_depth_25": 1.0,
                "imbalance_25": 0.5,
                "sequence_gap": False,
                "bids": [[100.0, 1.0]],
                "asks": [[101.0, 1.0]],
            },
            {
                "exchange_time_ms": 1_900,
                "best_bid": 99.0,
                "best_ask": 100.0,
                "mid_price": 99.5,
                "spread": 1.0,
                "spread_bps": 100.5025,
                "bid_depth_5": 2.0,
                "ask_depth_5": 2.0,
                "bid_depth_10": 2.0,
                "ask_depth_10": 2.0,
                "bid_depth_25": 2.0,
                "ask_depth_25": 2.0,
                "imbalance_25": 0.5,
                "sequence_gap": True,
                "bids": [[99.0, 2.0]],
                "asks": [[100.0, 2.0]],
            },
        ]
    )

    result = module.resample_l2_features(features, "1s")

    assert len(result) == 1
    assert result.iloc[0]["best_bid"] == 99.0
    assert result.iloc[0]["sequence_gap_count"] == 1
