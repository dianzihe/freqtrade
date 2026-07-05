#!/usr/bin/env python3
"""
全策略 × 全周期 × 全币种 批量回测脚本
8个策略 × 3个时间周期 × 9个币种 = 216个回测组合
"""

import subprocess
import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

# === 配置 ===
PROJECT_DIR = Path(r"F:\source\freqtrade")
FREQTRADE_BIN = PROJECT_DIR / ".venv" / "Scripts" / "freqtrade"
CONFIG_PATH = PROJECT_DIR / "user_data" / "config" / "config.json"
CONFIG_GATE_PATH = PROJECT_DIR / "user_data" / "config" / "config_gate_backtest.json"
DATA_DIR = PROJECT_DIR / "user_data" / "data" / "gate"
OUTPUT_DIR = PROJECT_DIR / "deliverables" / "backtest_batch_20260705"
RESULTS_DIR = OUTPUT_DIR / "results"
SUMMARY_DIR = OUTPUT_DIR / "summary"

TIMERANGE = "20260608-20260627"  # 统一回测区间（1m/5m 全覆盖，15m 部分更长但截取一致）

TIMEFRAMES = ["1m", "5m", "15m"]

STRATEGIES = [
    ("金麒麟_区间突破_风控版", "GoldKylinBtcEthBreakoutStrategy"),
    ("星河量化", "XingheMajorGridStrategySpot"),       # Spot wrapper (原始需futures)
    ("Meme_限制定投_马丁", "MemeLimitedMartingaleSpotStrategy"),       # Spot wrapper (基类can_short=True)
    ("Meme_波动率网格_马丁", "MemeVolatilityGridMartingaleSpotStrategy"),  # Spot wrapper
    ("Meme_对冲代理_马丁", "MemeHedgeProxyMartingaleSpotStrategy"),         # Spot wrapper
    ("Meme_反马丁_趋势", "MemeAntiMartingaleTrendStrategy"),
    ("AggressiveMartingale", "AggressiveMartingale"),
    ("顶部反转_做空", "TopReversalShortSpotStrategy"),   # Spot wrapper (纯做空→0交易)
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
    """解析 freqtrade backtesting 输出，提取关键指标"""
    result = {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "total_profit_pct": 0.0,
        "total_profit_abs": 0.0,
        "avg_profit_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_abs": 0.0,
        "drawdown_start": "",
        "drawdown_end": "",
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "avg_duration": "",
        "best_pair": "",
        "worst_pair": "",
        "entry_reason_counts": {},
        "exit_reason_counts": {},
        "raw_summary": "",
    }
    
    if not stdout_text:
        return result
    
    result["raw_summary"] = stdout_text[-3000:] if len(stdout_text) > 3000 else stdout_text
    
    # Parse SUMMARY METRICS table
    for line in stdout_text.split("\n"):
        line = line.strip()
        
        # Total/Daily profit
        m = re.search(r"Total/Daily Avg Trades\s*\|\s*(\d+)\s*/\s*([\d.]+)", line)
        if m:
            result["total_trades"] = int(m.group(1))
        
        # Win rate
        m = re.search(r"Winning\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", line)
        if m:
            result["winning_trades"] = int(m.group(1))
            result["losing_trades"] = int(m.group(2))
        m = re.search(r"Win\s+%\s*\|\s*([\d.]+)", line)
        if m:
            result["win_rate"] = float(m.group(1))
        
        # Total profit %
        m = re.search(r"Total profit %\s*\|\s*([\d.-]+)", line)
        if m:
            result["total_profit_pct"] = float(m.group(1))
        
        # Total profit abs
        m = re.search(r"Total profit\s+\|\s*([\d.-]+)\s*USDT", line)
        if m:
            result["total_profit_abs"] = float(m.group(1))
        
        # Avg profit %
        m = re.search(r"Avg\s+profit\s+%\s*\|\s*([\d.-]+)", line)
        if m:
            result["avg_profit_pct"] = float(m.group(1))
        
        # Max drawdown
        m = re.search(r"Max\s+drawdown\s*\|\s*([\d.]+)%\s*\(USDT\)", line)
        if m:
            result["max_drawdown_pct"] = float(m.group(1))
        m = re.search(r"Max\s+drawdown\s*\|\s*([\d.]+)%\s*\|\s*([\d.-]+)\s*USDT", line)
        if m:
            result["max_drawdown_pct"] = float(m.group(1))
            result["max_drawdown_abs"] = float(m.group(2))
        
        # Drawdown start/end
        m = re.search(r"Drawdown\s+Start\s*\|\s*([\d\-T:]+)", line)
        if m:
            result["drawdown_start"] = m.group(1)
        m = re.search(r"Drawdown\s+End\s*\|\s*([\d\-T:]+)", line)
        if m:
            result["drawdown_end"] = m.group(1)
        
        # Profit factor
        m = re.search(r"Profit\s+factor\s*\|\s*([\d.]+)", line)
        if m:
            result["profit_factor"] = float(m.group(1))
        
        # Sharpe ratio
        m = re.search(r"Sharpe\s*\|?\s*([\d.]+)", line)
        if m:
            result["sharpe_ratio"] = float(m.group(1))
        
        # Avg duration
        m = re.search(r"Avg\s+duration\s*\|\s*([\d:,\s\w]+?)(?:\s*\|)", line)
        if m:
            result["avg_duration"] = m.group(1).strip()
        
        # Best pair
        m = re.search(r"Best\s+pair\s*\|\s*([\w/]+)\s+([\d.-]+)%", line)
        if m:
            result["best_pair"] = m.group(1)
        
        # Worst pair
        m = re.search(r"Worst\s+pair\s*\|\s*([\w/]+)\s+([\d.-]+)%", line)
        if m:
            result["worst_pair"] = m.group(1)
    
    return result


def parse_trade_details(export_file):
    """解析导出的交易详情 JSON"""
    details = {
        "trades": [],
        "position_adds": 0,
        "liquidation_risk": 0,
        "max_concurrent_positions": 0,
        "max_position_size_pct": 0.0,
        "entry_signals": 0,
        "exit_signals": 0,
    }
    
    if not export_file.exists():
        return details
    
    try:
        with open(export_file, "r") as f:
            data = json.load(f)
        
        # Parse trades
        trades = data.get("trades", []) if isinstance(data, dict) else []
        if not trades and isinstance(data, list):
            trades = data
        
        all_trades = []
        position_count = 0
        max_concurrent = 0
        total_entry_signals = 0
        total_exit_signals = 0
        
        # Track position sizing for liquidation risk
        max_position_relative = 0.0
        
        for t in trades:
            trade_info = {
                "pair": t.get("pair", ""),
                "open_date": t.get("open_date", ""),
                "close_date": t.get("close_date", ""),
                "profit_pct": t.get("profit_ratio", 0) * 100 if t.get("profit_ratio") else 0,
                "profit_abs": t.get("profit_abs", 0),
                "stake_amount": t.get("stake_amount", 0),
                "open_rate": t.get("open_rate", 0),
                "close_rate": t.get("close_rate", 0),
                "enter_tag": t.get("enter_tag", ""),
                "exit_reason": t.get("exit_reason", ""),
                "is_short": t.get("is_short", False),
                "max_drawdown_pct": 0,
                "duration_min": 0,
            }
            
            # Calculate trade-level max drawdown
            if t.get("max_rate") and t.get("min_rate") and t.get("open_rate"):
                open_r = t["open_rate"]
                if not t.get("is_short"):
                    dd = (min(t.get("min_rate", open_r), open_r) - open_r) / open_r * 100
                else:
                    dd = (open_r - max(t.get("max_rate", open_r), open_r)) / open_r * 100
                trade_info["max_drawdown_pct"] = round(dd, 2)
            
            # Duration
            if t.get("open_date") and t.get("close_date"):
                try:
                    od = datetime.fromisoformat(t["open_date"].replace("Z", "+00:00"))
                    cd = datetime.fromisoformat(t["close_date"].replace("Z", "+00:00"))
                    trade_info["duration_min"] = round((cd - od).total_seconds() / 60, 1)
                except:
                    pass
            
            all_trades.append(trade_info)
            total_entry_signals += 1
        
        # Count position additions (consecutive entries same pair same direction)
        position_adds = 0
        sorted_trades = sorted(all_trades, key=lambda x: x.get("open_date", ""))
        open_trades = []
        for t in sorted_trades:
            open_trades.append(t)
            # Clean closed
            open_trades = [ot for ot in open_trades if ot.get("close_date", "") > t.get("open_date", "")]
            if len(open_trades) > max_concurrent:
                max_concurrent = len(open_trades)
            if len(open_trades) > 1:
                # Check if same pair same direction
                same_pair_dir = sum(1 for ot in open_trades if ot.get("pair") == t.get("pair") and ot.get("is_short") == t.get("is_short"))
                if same_pair_dir > 1:
                    position_adds += 1
        
        details["trades"] = all_trades
        details["position_adds"] = position_adds
        details["max_concurrent_positions"] = max_concurrent
        details["entry_signals"] = total_entry_signals
        details["exit_signals"] = total_exit_signals
        
        # Liquidation risk: if any single trade drawdown > 50% of stake
        for t in all_trades:
            dd = abs(t.get("max_drawdown_pct", 0))
            if dd > 50:
                details["liquidation_risk"] += 1
        
    except Exception as e:
        details["parse_error"] = str(e)
    
    return details


def run_single_backtest(strategy_name, class_name, timeframe, pair, run_id):
    """运行单个回测"""
    safe_pair = pair.replace("/", "_")
    result_file = RESULTS_DIR / f"{class_name}_{safe_pair}_{timeframe}.json"
    
    # Build command
    cmd = [
        str(FREQTRADE_BIN),
        "backtesting",
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
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )
        elapsed = time.time() - start_time
        
        stdout_text = result.stdout
        stderr_text = result.stderr
        
        success = result.returncode == 0
        
        # Parse metrics
        metrics = parse_backtest_output(stdout_text)
        
        # Parse trade details if export succeeded
        trade_details = parse_trade_details(result_file)
        
        return {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "class_name": class_name,
            "timeframe": timeframe,
            "pair": pair,
            "success": success,
            "elapsed_sec": round(elapsed, 1),
            "returncode": result.returncode,
            "metrics": metrics,
            "trade_details": trade_details,
            "stderr": stderr_text[:1000] if not success else "",
            "timestamp": datetime.now().isoformat(),
        }
    
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "class_name": class_name,
            "timeframe": timeframe,
            "pair": pair,
            "success": False,
            "elapsed_sec": round(elapsed, 1),
            "returncode": -1,
            "error": "TIMEOUT (300s)",
            "metrics": {},
            "trade_details": {},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "class_name": class_name,
            "timeframe": timeframe,
            "pair": pair,
            "success": False,
            "elapsed_sec": round(elapsed, 1),
            "returncode": -2,
            "error": str(e),
            "metrics": {},
            "trade_details": {},
            "timestamp": datetime.now().isoformat(),
        }


def run_all_backtests():
    """运行所有回测组合"""
    # Build all combinations
    all_runs = []
    run_id = 0
    for s_name, s_class in STRATEGIES:
        for tf in TIMEFRAMES:
            for pair in ALL_PAIRS:
                run_id += 1
                all_runs.append((run_id, s_name, s_class, tf, pair))
    
    total = len(all_runs)
    print(f"=" * 70)
    print(f"  全策略批量回测")
    print(f"  策略: {len(STRATEGIES)} | 周期: {len(TIMEFRAMES)} | 币种: {len(ALL_PAIRS)}")
    print(f"  总计: {total} 个回测组合")
    print(f"  回测区间: {TIMERANGE}")
    print(f"=" * 70)
    print()
    
    all_results = []
    success_count = 0
    fail_count = 0
    
    for run_id, s_name, s_class, tf, pair in all_runs:
        print(f"[{run_id}/{total}] {s_name} | {tf} | {pair} ... ", end="", flush=True)
        
        result = run_single_backtest(s_name, s_class, tf, pair, run_id)
        all_results.append(result)
        
        if result["success"]:
            m = result.get("metrics", {})
            trades = m.get("total_trades", 0)
            profit = m.get("total_profit_pct", 0)
            wr = m.get("win_rate", 0)
            dd = m.get("max_drawdown_pct", 0)
            adds = result.get("trade_details", {}).get("position_adds", 0)
            liq = result.get("trade_details", {}).get("liquidation_risk", 0)
            print(f"OK ({result['elapsed_sec']:.0f}s) trades={trades} profit={profit:.1f}% WR={wr:.0f}% DD={dd:.1f}% adds={adds} liq_risk={liq}")
            success_count += 1
        else:
            err = result.get("error", f"rc={result.get('returncode')}")
            print(f"FAIL ({result['elapsed_sec']:.0f}s) {err}")
            fail_count += 1
        
        # Save intermediate summary after each run
        if run_id % 10 == 0:
            save_intermediate_summary(all_results)
    
    # Final summary
    print()
    print(f"=" * 70)
    print(f"  回测完成: 成功 {success_count}/{total}, 失败 {fail_count}/{total}")
    print(f"=" * 70)
    
    # Save final results
    master_file = SUMMARY_DIR / "master_results.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存: {master_file}")
    
    return all_results


def save_intermediate_summary(results):
    """保存中间汇总"""
    summary_file = SUMMARY_DIR / "master_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)


def generate_performance_table(results):
    """生成绩效汇总表格"""
    rows = []
    for r in results:
        m = r.get("metrics", {})
        td = r.get("trade_details", {})
        
        # Determine coin category
        cat = "其他"
        for cname, pairs in COINS.items():
            if r["pair"] in pairs:
                cat = cname
                break
        
        rows.append({
            "策略": r["strategy_name"],
            "周期": r["timeframe"],
            "币种": r["pair"],
            "类别": cat,
            "状态": "成功" if r["success"] else "失败",
            "总交易数": m.get("total_trades", 0),
            "胜率%": round(m.get("win_rate", 0), 1),
            "总收益%": round(m.get("total_profit_pct", 0), 2),
            "平均收益%": round(m.get("avg_profit_pct", 0), 2),
            "最大回撤%": round(m.get("max_drawdown_pct", 0), 2),
            "盈亏比": round(m.get("profit_factor", 0), 2),
            "夏普比": round(m.get("sharpe_ratio", 0), 2),
            "加仓次数": td.get("position_adds", 0),
            "爆仓风险": td.get("liquidation_risk", 0),
            "最大并发": td.get("max_concurrent_positions", 0),
            "耗时秒": r.get("elapsed_sec", 0),
        })
    
    return rows


if __name__ == "__main__":
    print(f"回测启动时间: {datetime.now().isoformat()}")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    results = run_all_backtests()
    
    # Generate summary table
    table = generate_performance_table(results)
    
    # Save as CSV
    import csv
    csv_path = SUMMARY_DIR / "performance_summary.csv"
    if table:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
        print(f"CSV汇总已保存: {csv_path}")
    
    print("\n全部完成!")
