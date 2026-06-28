from pathlib import Path
import importlib.util

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "user_data" / "scripts" / "build_tick_indicators.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_tick_indicators", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def micros(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1_000_000)


def write_aggtrades(path: Path) -> None:
    rows = [
        ("2026-01-01 00:00:00", 1, 100.0, 1.0, "B"),
        ("2026-01-01 00:01:00", 2, 101.0, 2.0, "S"),
        ("2026-01-01 00:04:59", 3, 102.0, 3.0, "B"),
        ("2026-01-01 00:05:00", 4, 103.0, 4.0, "S"),
        ("2026-01-01 00:14:59", 5, 104.0, 5.0, "B"),
        ("2026-01-01 00:15:00", 6, 105.0, 6.0, "B"),
        ("2026-01-01 00:59:59", 7, 106.0, 7.0, "S"),
        ("2026-01-01 01:00:00", 8, 107.0, 8.0, "B"),
    ]
    lines = ["ticker,market,datetime,trade_id,price,volume,BS,first_trade_id,last_trade_id,isBestPriceMatch"]
    for ts, trade_id, price, volume, side in rows:
        lines.append(
            f"BTCUSDT,binance,{micros(ts)},{trade_id},{price},{volume},{side},{trade_id},{trade_id},True"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_tick_indicators_writes_multiple_timeframes(tmp_path: Path) -> None:
    module = load_module()
    csv_path = tmp_path / "BTCUSDT-aggTrades-2026-01.csv"
    write_aggtrades(csv_path)

    outputs = module.build_tick_indicators(
        files=[csv_path],
        output_dir=tmp_path / "out",
        timeframes=["1m", "5m", "15m", "1h"],
        pair="BTC_USDT",
        chunksize=3,
    )

    assert set(outputs) == {"1m", "5m", "15m", "1h"}
    for timeframe, output_path in outputs.items():
        assert output_path == tmp_path / "out" / f"BTC_USDT-{timeframe}.feather"
        assert output_path.exists()

    candles_1m = pd.read_feather(outputs["1m"])
    candles_5m = pd.read_feather(outputs["5m"])
    candles_15m = pd.read_feather(outputs["15m"])
    candles_1h = pd.read_feather(outputs["1h"])

    assert len(candles_1m) == 8
    assert len(candles_5m) == 6
    assert len(candles_15m) == 4
    assert len(candles_1h) == 2

    first_5m = candles_5m.iloc[0]
    assert first_5m["date"] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    assert first_5m["open"] == 100.0
    assert first_5m["high"] == 102.0
    assert first_5m["low"] == 100.0
    assert first_5m["close"] == 102.0
    assert first_5m["volume"] == 6.0
    assert first_5m["trade_count"] == 3
    assert first_5m["trade_intensity"] == pytest.approx(3 / 300)
    assert first_5m["buy_volume_ratio"] == pytest.approx(4 / 6)
    assert first_5m["vwap"] == pytest.approx((100 * 1 + 101 * 2 + 102 * 3) / 6)
    assert first_5m["vwap_deviation"] == pytest.approx(
        (102 - first_5m["vwap"]) / first_5m["vwap"]
    )
