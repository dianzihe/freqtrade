from pathlib import Path

import pandas as pd

from scripts.validate_gate_xinghe_data import validate_feather, validate_inventory


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_feather(path)


def test_validate_feather_reports_duplicate_and_invalid_ohlc(tmp_path: Path) -> None:
    path = tmp_path / "BTC_USDT-1m.feather"
    _write(
        path,
        [
            {
                "date": pd.Timestamp("2026-06-01", tz="UTC"),
                "open": 100.0,
                "high": 99.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1.0,
            },
            {
                "date": pd.Timestamp("2026-06-01", tz="UTC"),
                "open": 0.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            },
        ],
    )

    result = validate_feather(path)

    assert result["duplicate_timestamps"] == 1
    assert result["invalid_ohlc_rows"] == 2
    assert result["ok"] is False


def test_validate_inventory_requires_all_futures_series(tmp_path: Path) -> None:
    row = {
        "date": pd.Timestamp("2026-06-01", tz="UTC"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1.0,
    }
    _write(tmp_path / "BTC_USDT-1m.feather", [row])
    _write(tmp_path / "futures" / "BTC_USDT_USDT-1m-futures.feather", [row])

    result = validate_inventory(tmp_path, assets=("BTC",), timeframes=("1m",))

    assert result["ok"] is False
    assert "futures/BTC_USDT_USDT-1h-mark.feather" in result["missing"]
    assert "futures/BTC_USDT_USDT-1h-funding_rate.feather" in result["missing"]


def test_validate_feather_accepts_negative_funding_rates(tmp_path: Path) -> None:
    path = tmp_path / "BTC_USDT_USDT-1h-funding_rate.feather"
    _write(
        path,
        [
            {
                "date": pd.Timestamp("2026-06-01", tz="UTC"),
                "open": -0.0001,
                "high": -0.0001,
                "low": -0.0001,
                "close": -0.0001,
                "volume": 0.0,
            }
        ],
    )

    result = validate_feather(path)

    assert result["invalid_ohlc_rows"] == 0
    assert result["ok"] is True
