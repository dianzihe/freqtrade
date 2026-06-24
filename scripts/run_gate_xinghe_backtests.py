import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


def build_matrix(data_dir: Path, results_dir: Path) -> list[dict]:
    return [
        {
            "label": "spot-1m",
            "market": "spot",
            "timeframe": "1m",
            "timerange": "20260618-20260624",
            "config": Path("user_data/config-gate-xinghe-spot-backtest.json"),
            "strategy": "GateXingheSpotStrategy",
            "data_dir": data_dir,
            "results_dir": results_dir,
        },
        {
            "label": "spot-15m",
            "market": "spot",
            "timeframe": "15m",
            "timerange": "20260425-20260624",
            "config": Path("user_data/config-gate-xinghe-spot-backtest.json"),
            "strategy": "GateXingheSpotStrategy",
            "data_dir": data_dir,
            "results_dir": results_dir,
        },
        {
            "label": "futures-1m",
            "market": "futures",
            "timeframe": "1m",
            "timerange": "20260618-20260624",
            "config": Path("user_data/config-gate-xinghe-futures-backtest.json"),
            "strategy": "GateXingheFuturesStrategy",
            "data_dir": data_dir,
            "results_dir": results_dir,
        },
        {
            "label": "futures-15m",
            "market": "futures",
            "timeframe": "15m",
            "timerange": "20260425-20260624",
            "config": Path("user_data/config-gate-xinghe-futures-backtest.json"),
            "strategy": "GateXingheFuturesStrategy",
            "data_dir": data_dir,
            "results_dir": results_dir,
        },
    ]


def _strategy_payload(archive: Path) -> tuple[str, dict]:
    with ZipFile(archive) as source:
        names = [
            name
            for name in source.namelist()
            if name.endswith(".json")
            and not name.endswith("_config.json")
            and not name.endswith(".meta.json")
        ]
        payload = json.loads(source.read(names[0]))
    strategy_name = next(iter(payload["strategy"]))
    return strategy_name, payload["strategy"][strategy_name]


def parse_archive(label: str, archive: Path) -> dict:
    strategy_name, result = _strategy_payload(archive)
    trades = result.get("trades", [])
    pair_stats: dict[str, dict] = defaultdict(lambda: {"trades": 0, "profit_abs": 0.0})
    exit_reasons: Counter[str] = Counter()
    long_trades = 0
    short_trades = 0
    funding_fees = 0.0
    for trade in trades:
        pair = trade.get("pair", "unknown")
        pair_stats[pair]["trades"] += 1
        pair_stats[pair]["profit_abs"] += float(trade.get("profit_abs", 0.0))
        exit_reasons[trade.get("exit_reason", "unknown")] += 1
        short_trades += int(bool(trade.get("is_short")))
        long_trades += int(not bool(trade.get("is_short")))
        funding_fees += float(trade.get("funding_fees", 0.0) or 0.0)

    return {
        "label": label,
        "archive": str(archive),
        "strategy": strategy_name,
        "start": result.get("backtest_start"),
        "end": result.get("backtest_end"),
        "trades": int(result.get("total_trades", len(trades))),
        "profit_pct": float(result.get("profit_total", 0.0)) * 100,
        "profit_abs": float(result.get("profit_total_abs", 0.0)),
        "max_drawdown_pct": float(result.get("max_drawdown_account", 0.0)) * 100,
        "profit_factor": float(result.get("profit_factor", 0.0) or 0.0),
        "expectancy": float(result.get("expectancy", 0.0) or 0.0),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "funding_fees": funding_fees,
        "pairs": dict(pair_stats),
        "exit_reasons": dict(exit_reasons),
    }


def render_report(results: list[dict]) -> str:
    lines = [
        "# Gate Xinghe-Inspired Strategy Backtests",
        "",
        "Historical research only. Negative or positive results do not establish live tradability.",
        "",
        "| Window | Trades | Profit | Max DD | Profit factor | Long / Short | Funding fees |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['label']} | {result['trades']} | {result['profit_pct']:.2f}% | "
            f"{result['max_drawdown_pct']:.2f}% | {result['profit_factor']:.2f} | "
            f"{result['long_trades']} / {result['short_trades']} | "
            f"{result['funding_fees']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            "- Treat 1m results as a short recent execution sample.",
            "- Treat 15m results as the broader regime sample.",
            "- Negative expectancy or profit factor below 1 blocks dry-run promotion.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_matrix(data_dir: Path, results_dir: Path) -> list[tuple[str, Path]]:
    results_dir.mkdir(parents=True, exist_ok=True)
    archives: list[tuple[str, Path]] = []
    for task in build_matrix(data_dir, results_dir):
        before = set(results_dir.glob("*.zip"))
        command = [
            sys.executable,
            "-m",
            "freqtrade",
            "backtesting",
            "--config",
            str(task["config"]),
            "--strategy",
            task["strategy"],
            "--timeframe",
            task["timeframe"],
            "--data-dir",
            str(task["data_dir"]),
            "--timerange",
            task["timerange"],
            "--cache",
            "none",
            "--export",
            "trades",
            "--backtest-directory",
            str(results_dir),
        ]
        subprocess.run(command, check=True)
        created = sorted(
            set(results_dir.glob("*.zip")) - before,
            key=lambda path: path.stat().st_mtime,
        )
        if not created:
            raise RuntimeError(f"No archive generated for {task['label']}")
        archives.append((task["label"], created[-1]))
    return archives


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("user_data/data/gate"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("user_data/backtest_results/gate_xinghe"),
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Existing result in label=path form.",
    )
    args = parser.parse_args()

    archives = run_matrix(args.data_dir, args.results_dir) if args.run else []
    for item in args.archive:
        label, raw_path = item.split("=", 1)
        archives.append((label, Path(raw_path)))
    if not archives:
        parser.error("Use --run or provide at least one --archive label=path.")

    parsed = [parse_archive(label, path) for label, path in archives]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.results_dir / "report.md").write_text(render_report(parsed), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
