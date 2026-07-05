#!/usr/bin/env python3
"""
恢复批量回测 —— 跳过已完成的组合，从断点继续
"""

import subprocess
import json
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(r"F:\source\freqtrade")
FREQTRADE_BIN = PROJECT_DIR / ".venv" / "Scripts" / "freqtrade"
CONFIG_PATH = PROJECT_DIR / "user_data" / "config" / "config.json"
CONFIG_GATE_PATH = PROJECT_DIR / "user_data" / "config" / "config_gate_backtest.json"
DATA_DIR = PROJECT_DIR / "user_data" / "data" / "gate"
OUTPUT_DIR = PROJECT_DIR / "deliverables" / "backtest_batch_20260705"
RESULTS_DIR = OUTPUT_DIR / "results"
SUMMARY_DIR = OUTPUT_DIR / "summary"
MASTER_FILE = SUMMARY_DIR / "master_results.json"

TIMERANGE = "20260608-20260627"

TIMEFRAMES = ["1m", "5m", "15m"]

STRATEGIES = [
    ("金麒麟_区间突破_风控版", "GoldKylinBtcEthBreakoutStrategy"),
    ("星河量化", "XingheMajorGridStrategySpot"),
    ("Meme_限制定投_马丁", "MemeLimitedMartingaleSpotStrategy"),
    ("Meme_波动率网格_马丁", "MemeVolatilityGridMartingaleSpotStrategy"),
    ("Meme_对冲代理_马丁", "MemeHedgeProxyMartingaleSpotStrategy"),
    ("Meme_反马丁_趋势", "MemeAntiMartingaleTrendStrategy"),
    ("AggressiveMartingale", "AggressiveMartingale"),
    ("顶部反转_做空", "TopReversalShortSpotStrategy"),
]

COINS = {
    "主流": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中型": ["XCN/USDT"],
    "妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

ALL_PAIRS = []
for cat, pairs in COINS.items():
    ALL_PAIRS.extend(pairs)


def parse_backtest_output(stdout_text):
    result = {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": 0.0, "total_profit_pct": 0.0, "total_profit_abs": 0.0,
        "avg_profit_pct": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_abs": 0.0,
        "drawdown_start": "", "drawdown_end": "", "profit_factor": 0.0,
        "sharpe_ratio": 0.0, "avg_duration": "",
        "best_pair": "", "worst_pair": "", "raw_summary": "",
    }
    if not stdout_text:
        return result
    
    result["raw_summary"] = stdout_text[-3000:] if len(stdout_text) > 3000 else stdout_text
    
    for line in stdout_text.split("\n"):
        line = line.strip()
        m = re.search(r"Total/Daily Avg Trades\s*\|\s*(\d+)\s*/\s*([\d.]+)", line)
        if m: result["total_trades"] = int(m.group(1))
        m = re.search(r"Winning\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", line)
        if m: result["winning_trades"] = int(m.group(1)); result["losing_trades"] = int(m.group(2))
        m = re.search(r"Win\s+%\s*\|\s*([\d.]+)", line)
        if m: result["win_rate"] = float(m.group(1))
        m = re.search(r"Total profit %\s*\|\s*([\d.-]+)", line)
        if m: result["total_profit_pct"] = float(m.group(1))
        m = re.search(r"Total profit\s+\|\s*([\d.-]+)\s*USDT", line)
        if m: result["total_profit_abs"] = float(m.group(1))
        m = re.search(r"Avg\s+profit\s+%\s*\|\s*([\d.-]+)", line)
        if m: result["avg_profit_pct"] = float(m.group(1))
        m = re.search(r"Max\s+drawdown\s*\|\s*([\d.]+)%\s*\(USDT\)", line)
        if m: result["max_drawdown_pct"] = float(m.group(1))
        m = re.search(r"Max\s+drawdown\s*\|\s*([\d.]+)%\s*\|\s*([\d.-]+)\s*USDT", line)
        if m: result["max_drawdown_pct"] = float(m.group(1)); result["max_drawdown_abs"] = float(m.group(2))
        m = re.search(r"Drawdown\s+Start\s*\|\s*([\d\-T:]+)", line)
        if m: result["drawdown_start"] = m.group(1)
        m = re.search(r"Drawdown\s+End\s*\|\s*([\d\-T:]+)", line)
        if m: result["drawdown_end"] = m.group(1)
        m = re.search(r"Profit\s+factor\s*\|\s*([\d.]+)", line)
        if m: result["profit_factor"] = float(m.group(1))
        m = re.search(r"Sharpe\s*\|?\s*([\d.]+)", line)
        if m: result["sharpe_ratio"] = float(m.group(1))
        m = re.search(r"Avg\s+duration\s*\|\s*([\d:,\s\w]+?)(?:\s*\|)", line)
        if m: result["avg_duration"] = m.group(1).strip()
    
    return result


def parse_trade_details(export_file):
    details = {"trades": [], "position_adds": 0, "liquidation_risk": 0, "max_concurrent_positions": 0}
    if not export_file.exists():
        return details
    try:
        with open(export_file, "r") as f:
            data = json.load(f)
        trades = data.get("trades", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        all_trades = []
        max_concurrent = 0
        for t in trades:
            trade_info = {
                "pair": t.get("pair", ""),
                "open_date": t.get("open_date", ""),
                "close_date": t.get("close_date", ""),
                "profit_pct": (t.get("profit_ratio", 0) or 0) * 100,
                "profit_abs": t.get("profit_abs", 0) or 0,
                "open_rate": t.get("open_rate", 0) or 0,
                "close_rate": t.get("close_rate", 0) or 0,
                "is_short": t.get("is_short", False),
                "max_drawdown_pct": 0,
            }
            if t.get("max_rate") and t.get("min_rate") and t.get("open_rate"):
                open_r = t["open_rate"]
                if not t.get("is_short"):
                    dd = (min(t.get("min_rate", open_r), open_r) - open_r) / open_r * 100
                else:
                    dd = (open_r - max(t.get("max_rate", open_r), open_r)) / open_r * 100
                trade_info["max_drawdown_pct"] = round(dd, 2)
            all_trades.append(trade_info)
        details["trades"] = all_trades
        
        sorted_trades = sorted(all_trades, key=lambda x: x.get("open_date", ""))
        position_adds = 0
        open_trades = []
        for t in sorted_trades:
            open_trades = [ot for ot in open_trades if ot.get("close_date", "") > t.get("open_date", "")]
            open_trades.append(t)
            if len(open_trades) > max_concurrent:
                max_concurrent = len(open_trades)
            same_pair_dir = sum(1 for ot in open_trades if ot.get("pair") == t.get("pair") and ot.get("is_short") == t.get("is_short"))
            if same_pair_dir > 1:
                position_adds += 1
        
        details["position_adds"] = position_adds
        details["max_concurrent_positions"] = max_concurrent
        for t in all_trades:
            if abs(t.get("max_drawdown_pct", 0)) > 50:
                details["liquidation_risk"] += 1
    except Exception as e:
        details["parse_error"] = str(e)
    return details


def run_single_backtest(strategy_name, class_name, timeframe, pair):
    safe_pair = pair.replace("/", "_")
    result_file = RESULTS_DIR / f"{class_name}_{safe_pair}_{timeframe}.json"
    
    cmd = [
        str(FREQTRADE_BIN), "backtesting",
        "--config", str(CONFIG_PATH),
        "--config", str(CONFIG_GATE_PATH),
        "--strategy", class_name,
        "--timeframe", timeframe,
        "-p", pair,
        "--datadir", str(DATA_DIR),
        "--timerange", TIMERANGE,
        "--export", "signals",
        "--export-filename", str(result_file.with_suffix("")),
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        success = result.returncode == 0
        metrics = parse_backtest_output(result.stdout)
        trade_details = parse_trade_details(result_file)
        return {
            "strategy_name": strategy_name, "class_name": class_name,
            "timeframe": timeframe, "pair": pair,
            "success": success, "elapsed_sec": round(elapsed, 1),
            "returncode": result.returncode,
            "metrics": metrics, "trade_details": trade_details,
            "stderr": result.stderr[:1000] if not success else "",
            "timestamp": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "strategy_name": strategy_name, "class_name": class_name,
            "timeframe": timeframe, "pair": pair,
            "success": False, "elapsed_sec": round(time.time() - start_time, 1),
            "returncode": -1, "error": "TIMEOUT",
            "metrics": {}, "trade_details": {},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "strategy_name": strategy_name, "class_name": class_name,
            "timeframe": timeframe, "pair": pair,
            "success": False, "elapsed_sec": round(time.time() - start_time, 1),
            "returncode": -2, "error": str(e),
            "metrics": {}, "trade_details": {},
            "timestamp": datetime.now().isoformat(),
        }


def load_existing():
    if not MASTER_FILE.exists():
        return [], set()
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    # Build set of (class_name, timeframe, pair) already SUCCESSFULLY done
    completed = set()
    for r in results:
        if r.get("success"):
            completed.add((r.get("class_name", ""), r.get("timeframe", ""), r.get("pair", "")))
    # Remove failed entries from results (will be re-run and replaced)
    results = [r for r in results if r.get("success")]
    return results, completed


def main():
    print(f"恢复回测: {datetime.now().isoformat()}")
    
    existing_results, completed = load_existing()
    print(f"已有 {len(existing_results)} 条记录")
    
    # Build all combos and filter
    all_combos = []
    for s_name, s_class in STRATEGIES:
        for tf in TIMEFRAMES:
            for pair in ALL_PAIRS:
                key = (s_class, tf, pair)
                if key not in completed:
                    all_combos.append((s_name, s_class, tf, pair))
    
    total_todo = len(all_combos)
    total_all = total_todo + len(completed)
    print(f"待执行: {total_todo}/{total_all}")
    
    if total_todo == 0:
        print("所有回测已完成!")
        return existing_results
    
    # Count what's missing by strategy
    missing_by_strat = Counter()
    for s_name, _, _, _ in all_combos:
        missing_by_strat[s_name] += 1
    print("各策略剩余:")
    for s, c in missing_by_strat.items():
        print(f"  {s}: {c}")
    
    print()
    print("=" * 60)
    
    results = existing_results[:]
    run_count = 0
    
    for s_name, s_class, tf, pair in all_combos:
        idx = len(completed) + run_count + 1
        run_count += 1
        print(f"[{idx}/{total_all}] {s_name} | {tf} | {pair} ... ", end="", flush=True)
        
        result = run_single_backtest(s_name, s_class, tf, pair)
        results.append(result)
        
        if result["success"]:
            m = result.get("metrics", {})
            trades = m.get("total_trades", 0)
            profit = m.get("total_profit_pct", 0)
            wr = m.get("win_rate", 0)
            dd = m.get("max_drawdown_pct", 0)
            adds = result.get("trade_details", {}).get("position_adds", 0)
            liq = result.get("trade_details", {}).get("liquidation_risk", 0)
            print(f"OK ({result['elapsed_sec']:.0f}s) trades={trades} profit={profit:.1f}% DD={dd:.1f}% adds={adds} liq={liq}")
        else:
            err = result.get("error", f"rc={result.get('returncode')}")
            print(f"FAIL ({result['elapsed_sec']:.0f}s) {err}")
        
        # Save every 10 runs
        if run_count % 10 == 0:
            with open(MASTER_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"  [已保存 {len(results)} 条]")
    
    # Final save
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    success_count = sum(1 for r in results if r.get("success"))
    print()
    print(f"=" * 60)
    print(f"回测完成: 成功 {success_count}/{len(results)}")
    print(f"结果已保存: {MASTER_FILE}")
    
    return results


if __name__ == "__main__":
    main()
