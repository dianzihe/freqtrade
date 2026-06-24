import json
from pathlib import Path
from zipfile import ZipFile

from scripts.run_gate_xinghe_backtests import build_matrix, parse_archive, render_report


def test_build_matrix_contains_exact_spot_futures_timeframes() -> None:
    matrix = build_matrix(Path("data"), Path("results"))

    assert [(item["market"], item["timeframe"]) for item in matrix] == [
        ("spot", "1m"),
        ("spot", "15m"),
        ("futures", "1m"),
        ("futures", "15m"),
    ]
    assert matrix[0]["timerange"] == "20260618-20260624"
    assert matrix[-1]["timerange"] == "20260425-20260624"


def test_parse_archive_and_render_report(tmp_path: Path) -> None:
    archive = tmp_path / "result.zip"
    payload = {
        "strategy": {
            "GateXingheSpotStrategy": {
                "total_trades": 2,
                "profit_total": -0.01,
                "profit_total_abs": -10.0,
                "profit_factor": 0.8,
                "expectancy": -5.0,
                "max_drawdown_account": 0.03,
                "backtest_start": "2026-06-01 00:00:00",
                "backtest_end": "2026-06-02 00:00:00",
                "trades": [
                    {"pair": "BTC/USDT", "is_short": False, "funding_fees": 0.0},
                    {"pair": "ETH/USDT", "is_short": False, "funding_fees": 0.0},
                ],
            }
        }
    }
    with ZipFile(archive, "w") as output:
        output.writestr("result.json", json.dumps(payload))

    parsed = parse_archive("spot-1m", archive)
    report = render_report([parsed])

    assert parsed["trades"] == 2
    assert parsed["profit_pct"] == -1.0
    assert parsed["long_trades"] == 2
    assert "| spot-1m | 2 | -1.00% | 3.00% | 0.80 |" in report
