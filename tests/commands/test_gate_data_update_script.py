import subprocess
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "update_gate_1m_data.ps1"
EXCHANGES = ["binance", "okx", "bybit", "kucoin", "mexc"]


def test_gate_data_update_script_dry_run_prints_download_command():
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Days",
            "3",
            "-DryRun",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "download-data" in output
    assert "user_data\\config-gate-data-200.json" in output
    assert "user_data\\pairs-gate-spot-200.json" in output
    assert "--timeframes 1m 5m 15m 1h" in output
    assert "--days 3" in output
    assert "--trading-mode spot" in output


def test_exchange_data_update_scripts_default_to_five_day_dry_run():
    for exchange in EXCHANGES:
        script = ROOT / "scripts" / f"update_{exchange}_1m_data.ps1"
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-DryRun",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        output = result.stdout
        assert "download-data" in output
        assert f"user_data\\config-{exchange}-data-200.json" in output
        assert f"user_data\\pairs-{exchange}-spot-200.json" in output
        assert "--timeframes 1m" in output
        assert "--days 5" in output
        assert "--trading-mode spot" in output


def test_exchange_data_configs_and_pairs_files_are_consistent():
    for exchange in EXCHANGES:
        config = json.loads((ROOT / "user_data" / f"config-{exchange}-data-200.json").read_text())
        pairs = json.loads((ROOT / "user_data" / f"pairs-{exchange}-spot-200.json").read_text())

        assert config["exchange"]["name"] == exchange
        assert config["trading_mode"] == "spot"
        assert config["timeframe"] == "1m"
        assert config["dataformat_ohlcv"] == "feather"
        assert config["pairlists"][0]["number_assets"] == 200
        assert pairs
        assert all(pair.endswith("/USDT") for pair in pairs)
