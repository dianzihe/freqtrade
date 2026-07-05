#!/usr/bin/env python3
"""金麒麟回测 - 断点续跑"""
import subprocess, json, os, time, sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
RESULTS_DIR = PROJECT_DIR / "user_data" / "bt_results" / "goldkylin"

STRATEGY_CLASS = "GoldKylinBtcEthBreakoutStrategy"

# 已完成的任务 (check existing logs)
completed = set()
for log_file in RESULTS_DIR.glob("log_*.txt"):
    name = log_file.stem.replace("log_", "")
    with open(log_file, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if "Result for strategy" in content:
        completed.add(name)

print(f"已完成: {len(completed)} 个")
for c in sorted(completed):
    print(f"  {c}")

# 需要运行的任务
ALL_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "XCN/USDT",
    "H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT",
]
TIMEFRAMES = ["1m", "5m", "15m"]

pending = []
for pair in ALL_PAIRS:
    pslug = pair.replace("/", "_")
    for tf in TIMEFRAMES:
        task_id = f"{pslug}_{tf}"
        if task_id not in completed:
            pending.append((pair, tf, task_id))

print(f"\n待运行: {len(pending)} 个")
for _, _, tid in pending:
    print(f"  {tid}")

if not pending:
    print("全部完成!")
    sys.exit(0)

# 导入主脚本函数
sys.path.insert(0, str(PROJECT_DIR))
from backtest_goldkylin import run_backtest, parse_backtest_output, analyze_trades

# 运行待完成的任务
success = 0
fail = 0
for i, (pair, tf, tid) in enumerate(pending, 1):
    label = f"[{i}/{len(pending)}] {pair} {tf}"
    print(f"{label} ...", end=" ", flush=True)
    result = run_backtest(pair, tf)
    if result.get("error"):
        fail += 1
        print(f"FAIL ({result['elapsed']:.1f}s) {result['error'][:80]}")
    else:
        success += 1
        trades = result.get("total_trades", 0)
        pnl = result.get("total_profit_pct", 0) or 0
        print(f"OK ({result['elapsed']:.1f}s) Trades={trades} PnL={pnl}%")

print(f"\n续跑完成: 成功 {success}, 失败 {fail}")
