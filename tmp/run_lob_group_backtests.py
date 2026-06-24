from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "user_data" / "backtest_results" / "lob_groups"
LOG_DIR = OUT_DIR / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "LTC/USDT",
    "HYPE/USDT",
    "XCN/USDT",
    "IP/USDT",
    "BAS/USDT",
    "PEAQ/USDT",
    "TA/USDT",
    "H/USDT",
    "VELVET/USDT",
    "BEAT/USDT",
    "COAI/USDT",
    "ALLO/USDT",
    "DN/USDT",
    "STG/USDT",
]

CATEGORIES = {
    "stable": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LTC/USDT", "HYPE/USDT"],
    "mid": ["XCN/USDT", "IP/USDT", "BAS/USDT", "PEAQ/USDT", "TA/USDT"],
    "meme": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT", "ALLO/USDT", "DN/USDT", "STG/USDT"],
}

STRATEGIES = [
    ("ChanlunCenterBreakoutStrategy", "15m"),
    ("ChanlunSecondBuyStrategy", "15m"),
    ("ChanlunSecondBuyMinimal", "15m"),
    ("ChanlunSecondBuyTrailing", "15m"),
    ("ChanlunMacdDivergenceStrategy", "15m"),
    ("ChanlunRsiTimingStrategy", "15m"),
    ("ChanlunVolumeConfirmationStrategy", "15m"),
    ("ChanlunMaRsiSecondBuyStrategy", "15m"),
    ("VolatilityBreakoutMomentumStrategy", "15m"),
    ("VolumeAtrMeanReversionStrategy", "15m"),
    ("MultiFactorBtcSyncStrategy", "15m"),
    ("RiskFirstStrongTrendStrategy", "15m"),
    ("SwingTrendFollowStrategy", "15m"),
    ("GateScalpMomentumStrategy", "5m"),
    ("TrendPyramidAntiMartingaleStrategy", "5m"),
    ("TrendPyramidStrategy", "1h"),
    ("MemeLimitedDcaMartingaleStrategy", "1m"),
    ("MemeVolatilityGridMartingaleStrategy", "1m"),
    ("MemeAntiMartingaleTrendStrategy", "1m"),
    ("MemeHedgeProxyMartingaleStrategy", "1m"),
    ("TopReversalShortStrategy", "1m"),
    ("TrendFilteredGridStrategy", "1m"),
]

PAIR_RE = re.compile(
    r"^[|│]\s*(?P<pair>[A-Z0-9]+/USDT|TOTAL)\s*[|│]"
    r"\s*(?P<trades>\d+)\s*[|│]"
    r"\s*(?P<avg>[-+]?\d+(?:\.\d+)?)\s*[|│]"
    r"\s*(?P<profit_abs>[-+]?\d+(?:\.\d+)?)\s*[|│]"
    r"\s*(?P<profit_pct>[-+]?\d+(?:\.\d+)?)\s*[|│]"
)

METRIC_RE = re.compile(r"^[|│]\s*(?P<metric>[^|│]+?)\s*[|│]\s*(?P<value>[^|│]+?)\s*[|│]")


def write_progress(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (OUT_DIR / "progress.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_log(text: str) -> dict:
    pair_results = {}
    summary = {}

    for line in text.splitlines():
        match = PAIR_RE.match(line)
        if match:
            row = {
                "trades": int(match.group("trades")),
                "avg_profit_pct": float(match.group("avg")),
                "profit_abs": float(match.group("profit_abs")),
                "profit_pct": float(match.group("profit_pct")),
            }
            pair = match.group("pair")
            if pair in PAIRS:
                pair_results[pair] = row

        metric_match = METRIC_RE.match(line)
        if metric_match:
            metric = metric_match.group("metric").strip()
            value = metric_match.group("value").strip()
            summary[metric] = value

    total = {
        "trades": sum(row["trades"] for row in pair_results.values()),
        "avg_profit_pct": 0.0,
        "profit_abs": round(sum(row["profit_abs"] for row in pair_results.values()), 3),
        "profit_pct": round(sum(row["profit_pct"] for row in pair_results.values()), 3),
    }
    total["avg_profit_pct"] = (
        round(
            sum(row["avg_profit_pct"] * row["trades"] for row in pair_results.values())
            / total["trades"],
            3,
        )
        if total["trades"]
        else 0.0
    )

    return {"pair_results": pair_results, "total": total, "summary": summary}


def aggregate_categories(pair_results: dict) -> dict:
    categories = {}
    for name, pairs in CATEGORIES.items():
        rows = {pair: pair_results[pair] for pair in pairs if pair in pair_results}
        profit_abs = sum(row["profit_abs"] for row in rows.values())
        profit_pct = sum(row["profit_pct"] for row in rows.values())
        trades = sum(row["trades"] for row in rows.values())
        categories[name] = {
            "pairs": list(rows),
            "trades": trades,
            "profit_abs": round(profit_abs, 3),
            "profit_pct": round(profit_pct, 3),
        }
    return categories


def run_strategy(strategy: str, timeframe: str, index: int, total: int) -> dict:
    log_path = LOG_DIR / f"{strategy}_{timeframe}.log"
    cmd = [
        sys.executable,
        "-m",
        "freqtrade",
        "backtesting",
        "--config",
        "user_data/backtest_config_1m.json",
        "--strategy",
        strategy,
        "--timeframe",
        timeframe,
        "--timerange",
        "20260608-20260622",
        "--export",
        "none",
        "--cache",
        "none",
    ]
    write_progress(f"[{index}/{total}] START {strategy} {timeframe}")
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    elapsed = round(time.time() - started, 1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        errors = [
            line.strip()
            for line in text.splitlines()
            if "ERROR" in line or "Exception" in line or "ValueError" in line
        ]
        error = errors[-1] if errors else text[-500:]
        write_progress(f"[{index}/{total}] FAIL {strategy} {timeframe} {elapsed}s {error[:140]}")
        return {
            "strategy": strategy,
            "timeframe": timeframe,
            "elapsed_sec": elapsed,
            "error": error,
            "log": str(log_path.relative_to(ROOT)),
        }

    parsed = parse_log(text)
    result = {
        "strategy": strategy,
        "timeframe": timeframe,
        "elapsed_sec": elapsed,
        "error": None,
        "log": str(log_path.relative_to(ROOT)),
        **parsed,
    }
    result["category_results"] = aggregate_categories(parsed["pair_results"])
    total_row = parsed.get("total") or {}
    write_progress(
        f"[{index}/{total}] OK {strategy} {timeframe} {elapsed}s "
        f"trades={total_row.get('trades', 0)} profit={total_row.get('profit_pct', 0)}%"
    )
    return result


def build_markdown(results: list[dict]) -> str:
    ok = [result for result in results if not result.get("error")]
    failed = [result for result in results if result.get("error")]
    lines = [
        "# LOB Group Backtest Summary",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        "- Exchange/data: Gate spot OHLCV",
        "- Timerange: 2026-06-08 to 2026-06-22",
        f"- Successful: {len(ok)}",
        f"- Failed: {len(failed)}",
        "",
        "## Category Results",
        "",
        "| Strategy | TF | Stable % | Stable Trades | Mid % | Mid Trades | Meme % | Meme Trades | Total % | Total Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(ok, key=lambda item: item["total"].get("profit_pct", -999), reverse=True):
        cats = result["category_results"]
        total = result["total"]
        lines.append(
            f"| {result['strategy']} | {result['timeframe']} | "
            f"{cats['stable']['profit_pct']:.2f} | {cats['stable']['trades']} | "
            f"{cats['mid']['profit_pct']:.2f} | {cats['mid']['trades']} | "
            f"{cats['meme']['profit_pct']:.2f} | {cats['meme']['trades']} | "
            f"{total.get('profit_pct', 0):.2f} | {total.get('trades', 0)} |"
        )
    if failed:
        lines.extend(["", "## Failed", ""])
        for result in failed:
            lines.append(f"- {result['strategy']} {result['timeframe']}: {result['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    progress = OUT_DIR / "progress.txt"
    if progress.exists():
        progress.unlink()

    results = []
    total = len(STRATEGIES)
    write_progress(f"RUN start strategies={total}")
    for index, (strategy, timeframe) in enumerate(STRATEGIES, start=1):
        try:
            results.append(run_strategy(strategy, timeframe, index, total))
        except subprocess.TimeoutExpired:
            write_progress(f"[{index}/{total}] TIMEOUT {strategy} {timeframe}")
            results.append({"strategy": strategy, "timeframe": timeframe, "error": "timeout"})

        payload = {"generated_at": datetime.now().isoformat(), "results": results}
        (OUT_DIR / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUT_DIR / "summary.md").write_text(build_markdown(results), encoding="utf-8")

    write_progress("RUN complete")


if __name__ == "__main__":
    main()
