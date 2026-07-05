#!/usr/bin/env python3
"""
金麒麟区间突破风控版 — 全币种全周期批量回测

覆盖:
- 时间周期: 1m, 5m, 15m
- 稳定主流币: BTC, ETH, SOL, XRP
- 中端中型币: XCN
- 高波动妖币: H, VELVET, BEAT, COAI

每次回测独立运行，分别输出:
- 绩效指标 (总收益%, 胜率, Sharpe, Sortino, 最大回撤)
- 盈亏曲线 (逐笔交易明细)
- 最大浮亏
- 加仓触发记录 (策略无DCA, 标注"不适用")
- 爆仓风险统计 (止损触发次数/最大单笔亏损/组合清仓触发)
"""

import subprocess
import json
import re
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
DATA_DIR = str((PROJECT_DIR / "user_data" / "data" / "gate").resolve())
RESULTS_DIR = PROJECT_DIR / "user_data" / "bt_results" / "goldkylin"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_CLASS = "GoldKylinBtcEthBreakoutStrategy"
STRATEGY_FILE = "金麒麟_区间突破_风控版.py"

TIMEFRAMES = ["1m", "5m", "15m"]

CATEGORIES = {
    "stable":  ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "mid":     ["XCN/USDT"],
    "highvol": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

ALL_PAIRS = [p for ps in CATEGORIES.values() for p in ps]

# ============================================================
# 基础配置模板
# ============================================================
BASE_CONFIG = {
    "max_open_trades": 2,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "CNY",
    "dry_run": True,
    "dry_run_wallet": 10000,
    "cancel_open_orders_on_exit": False,
    "trading_mode": "spot",
    "margin_mode": "",
    "unfilledtimeout": {"entry": 10, "exit": 10, "unit": "minutes"},
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": True,
        "order_book_top": 1,
        "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
    },
    "exit_pricing": {
        "price_side": "same",
        "use_order_book": True,
        "order_book_top": 1,
    },
    "exchange": {
        "name": "gate",
        "key": "",
        "secret": "",
        "ccxt_config": {
            "proxies": {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            }
        },
        "ccxt_async_config": {
            "aiohttp_proxy": "http://127.0.0.1:7890",
        },
        "enable_ws": False,
        "pair_whitelist": [],
        "pair_blacklist": [],
    },
    "pairlists": [{"method": "StaticPairList"}],
    "telegram": {"enabled": False, "token": "", "chat_id": ""},
    "api_server": {
        "enabled": False,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "x",
        "password": "x",
        "jwt_secret_key": "somethingRandomSomethingRandom123",
    },
    "bot_name": "freqtrade",
    "initial_state": "running",
    "force_entry_enable": False,
    "internals": {"process_throttle_secs": 5},
    "datadir": DATA_DIR,
}


def write_config(pair: str, timeframe: str) -> Path:
    """写入临时配置文件"""
    cfg = json.loads(json.dumps(BASE_CONFIG))
    cfg["exchange"]["pair_whitelist"] = [pair]
    cfg["timeframe"] = timeframe
    path = RESULTS_DIR / f"config_{pair.replace('/', '_')}_{timeframe}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path


def parse_backtest_output(stdout: str) -> dict:
    """解析 freqtrade backtesting 输出文本"""
    result = {
        "pair_results": {},
        "total_trades": 0,
        "total_profit_pct": None,
        "total_profit_abs": None,
        "win_rate": None,
        "max_drawdown": None,
        "max_drawdown_abs": None,
        "avg_duration": None,
        "sharpe": None,
        "sortino": None,
        "profit_factor": None,
        "expectancy": None,
        "avg_profit_pct": None,
        "best_pair": None,
        "worst_pair": None,
        "trades_per_day": None,
    }

    # 提取各交易对结果
    for line in stdout.splitlines():
        m = re.search(
            r"│\s*(\S+/USDT)\s*│\s*(\d+)\s*│\s*([-\d.]+)\s*│\s*([-\d.]+)", line,
        )
        if m:
            pair = m.group(1)
            result["pair_results"][pair] = {
                "trades": int(m.group(2)),
                "profit_pct": float(m.group(3)),
                "profit_abs": float(m.group(4)),
            }

    # TOTAL 行
    m = re.search(r"TOTAL\s*│\s*(\d+)\s*│\s*([-\d.]+)%?\s*│\s*([-\d.]+)", stdout)
    if m:
        result["total_trades"] = int(m.group(1))
        result["total_profit_pct"] = float(m.group(2))
        result["total_profit_abs"] = float(m.group(3))

    # Win Rate
    m = re.search(r"Win\s*/\s*Loss\s*│\s*(\d+)\s*/\s*(\d+)", stdout)
    if m:
        w, l = int(m.group(1)), int(m.group(2))
        result["win_rate"] = round(w / (w + l) * 100, 1) if (w + l) > 0 else None
        result["win_count"] = w
        result["loss_count"] = l

    # Max Drawdown
    for pat in [
        r"Max\s+drawdown.*?([-\d.]+)%",
        r"Drawdown.*?([-\d.]+)%",
    ]:
        m = re.search(pat, stdout)
        if m:
            result["max_drawdown"] = float(m.group(1))
            break

    # Max Drawdown Abs
    m = re.search(r"Drawdown.*?([-\d.]+)\s*(USDT|USD)", stdout)
    if m:
        result["max_drawdown_abs"] = float(m.group(1))

    # Sharpe
    m = re.search(r"Sharpe.*?([-\d.]+)", stdout)
    if m:
        result["sharpe"] = float(m.group(1))

    # Sortino
    m = re.search(r"Sortino.*?([-\d.]+)", stdout)
    if m:
        result["sortino"] = float(m.group(1))

    # Profit Factor
    m = re.search(r"Profit\s*factor.*?([-\d.]+)", stdout)
    if m:
        result["profit_factor"] = float(m.group(1))

    # Expectancy
    m = re.search(r"Expectancy.*?([-\d.]+)", stdout)
    if m:
        result["expectancy"] = float(m.group(1))

    # Avg duration
    m = re.search(r"Avg.*?duration.*?(\d+:\d+)", stdout)
    if m:
        result["avg_duration"] = m.group(1)

    # Trades per day
    m = re.search(r"Trades\s*/\s*day.*?([-\d.]+)", stdout)
    if m:
        result["trades_per_day"] = float(m.group(1))

    # Avg Profit %
    m = re.search(r"Avg\s+profit\s*%?\s*│\s*([-\d.]+)", stdout)
    if m:
        result["avg_profit_pct"] = float(m.group(1))

    return result


def analyze_trades(trades_file: Path) -> dict:
    """分析逐笔交易明细"""
    if not trades_file.exists():
        return {
            "trades": [],
            "liquidation_stats": {},
            "pnl_curve": [],
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "max_drawdown_pct": 0,
        }

    df = pd.read_feather(trades_file)
    trades = []
    pnl_curve = []
    cumulative_pnl = 0.0
    max_cumulative = 0.0
    max_drawdown_pct = 0.0
    max_floating_loss = 0.0

    stoploss_hits = 0
    emergency_exits = 0
    time_stops = 0
    roi_exits = 0
    signal_exits = 0

    for _, row in df.iterrows():
        trade = {
            "pair": str(row.get("pair", "")),
            "open_date": str(row.get("open_date", "")),
            "close_date": str(row.get("close_date", "")),
            "open_rate": float(row.get("open_rate", 0)),
            "close_rate": float(row.get("close_rate", 0)),
            "stake_amount": float(row.get("stake_amount", 0)),
            "amount": float(row.get("amount", 0)),
            "profit_ratio": float(row.get("profit_ratio", 0)),
            "profit_abs": float(row.get("profit_abs", 0)),
            "exit_reason": str(row.get("sell_reason", row.get("exit_reason", ""))),
            "enter_tag": str(row.get("enter_tag", "")),
            "is_short": bool(row.get("is_short", False)),
        }
        trades.append(trade)
        cumulative_pnl += trade["profit_abs"]

        if cumulative_pnl > max_cumulative:
            max_cumulative = cumulative_pnl
        drawdown = (max_cumulative - cumulative_pnl) / 10000.0 * 100
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown

        if trade["profit_ratio"] < 0:
            floating_loss = abs(trade["profit_abs"])
            if floating_loss > max_floating_loss:
                max_floating_loss = floating_loss

        pnl_curve.append({
            "trade_num": len(trades),
            "close_date": trade["close_date"],
            "profit_abs": trade["profit_abs"],
            "profit_ratio": trade["profit_ratio"],
            "cumulative_pnl": cumulative_pnl,
            "drawdown_pct": drawdown,
        })

        # 分类出场原因
        reason = trade["exit_reason"].lower()
        if "stop" in reason or "stoploss" in reason:
            stoploss_hits += 1
        elif "emergency" in reason or "extreme" in reason or "account_float" in reason:
            emergency_exits += 1
        elif "time" in reason:
            time_stops += 1
        elif "roi" in reason:
            roi_exits += 1
        else:
            signal_exits += 1

    total_trades = len(trades)
    win_trades = sum(1 for t in trades if t["profit_ratio"] > 0)
    loss_trades = sum(1 for t in trades if t["profit_ratio"] < 0)

    max_loss = min((t["profit_ratio"] for t in trades), default=0)
    max_loss_abs = max(
        (abs(t["profit_abs"]) for t in trades if t["profit_abs"] < 0), default=0
    )

    consecutive_losses = 0
    max_consecutive_losses = 0
    for t in trades:
        if t["profit_ratio"] < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    liquidation_stats = {
        "stoploss_hits": stoploss_hits,
        "stoploss_hit_pct": (
            round(stoploss_hits / total_trades * 100, 1) if total_trades > 0 else 0
        ),
        "emergency_exits": emergency_exits,
        "time_stops": time_stops,
        "roi_exits": roi_exits,
        "signal_exits": signal_exits,
        "max_single_loss_pct": round(max_loss * 100, 2),
        "max_single_loss_abs": round(max_loss_abs, 2),
        "max_consecutive_losses": max_consecutive_losses,
        "max_floating_loss_abs": round(max_floating_loss, 2),
    }

    return {
        "trades": trades,
        "liquidation_stats": liquidation_stats,
        "pnl_curve": pnl_curve,
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
    }


def run_backtest(pair: str, timeframe: str) -> dict:
    """运行单个回测并返回完整结果"""
    pair_slug = pair.replace("/", "_")
    config_path = write_config(pair, timeframe)
    trades_base = str(RESULTS_DIR / f"trades_{pair_slug}_{timeframe}")
    log_file = RESULTS_DIR / f"log_{pair_slug}_{timeframe}.txt"

    cmd = [
        VENV_PYTHON, "-u", "-m", "freqtrade", "backtesting",
        "--config", str(config_path),
        "--strategy", STRATEGY_CLASS,
        "--timeframe", timeframe,
        "--export", "trades",
        "--export-filename", trades_base,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.time() - t0
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(stdout)
            if stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(stderr)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {"pair": pair, "timeframe": timeframe, "error": "TIMEOUT", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        return {"pair": pair, "timeframe": timeframe, "error": str(e), "elapsed": elapsed}

    if returncode != 0:
        err_lines = [
            l.strip()
            for l in (stdout + stderr).splitlines()
            if any(k in l for k in ["ERROR", "Exception", "Fatal", "ValueError", "KeyError"])
        ]
        return {
            "pair": pair,
            "timeframe": timeframe,
            "error": "\n".join(err_lines[-5:]) or stdout[-500:],
            "elapsed": elapsed,
        }

    # 解析输出
    parsed = parse_backtest_output(stdout)
    parsed["pair"] = pair
    parsed["timeframe"] = timeframe
    parsed["elapsed"] = elapsed

    # 分析交易明细
    trade_export = Path(trades_base + ".feather")
    if trade_export.exists():
        trade_analysis = analyze_trades(trade_export)
        parsed["trade_analysis"] = trade_analysis
    else:
        parsed["trade_analysis"] = {
            "trades": [],
            "pnl_curve": [],
            "liquidation_stats": {},
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "max_drawdown_pct": 0,
        }

    return parsed


def main():
    print("=" * 70)
    print("  金麒麟区间突破风控版 — 全币种全周期批量回测")
    print(f"  策略类名: {STRATEGY_CLASS}")
    print(f"  数据目录: {DATA_DIR}")
    print("=" * 70)

    tasks = [(pair, tf) for tf in TIMEFRAMES for pair in ALL_PAIRS]
    total = len(tasks)

    print(f"\n共 {total} 个回测任务 (9币种 × 3周期)")
    print("-" * 70)

    all_results = []
    success_count = 0
    fail_count = 0
    total_trades_all = 0

    for i, (pair, tf) in enumerate(tasks, 1):
        cat = [c for c, ps in CATEGORIES.items() if pair in ps][0]
        label = f"[{i:2d}/{total}] {pair:>12s} {tf:>3s} [{cat}]"
        print(f"{label} ...", end=" ", flush=True)

        result = run_backtest(pair, tf)
        all_results.append(result)

        if result.get("error"):
            fail_count += 1
            print(f"FAIL ({result['elapsed']:.1f}s)")
        else:
            success_count += 1
            trades = result.get("total_trades", 0)
            total_trades_all += trades
            pnl = result.get("total_profit_pct", 0) or 0
            dd = result.get("max_drawdown", 0) or 0
            wr = result.get("win_rate", 0) or 0
            print(f"OK ({result['elapsed']:.1f}s) Trades={trades} PnL={pnl}% DD={dd}% WR={wr}%")

    # 保存完整结果
    output = {
        "strategy": STRATEGY_CLASS,
        "strategy_file": STRATEGY_FILE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_notes": "全量可用数据(各币种数据起始时间不同, 均使用最大可用范围)",
        "summary": {
            "total_tasks": total,
            "success": success_count,
            "failed": fail_count,
            "total_trades_all_runs": total_trades_all,
        },
        "results": all_results,
    }

    output_json = RESULTS_DIR / "all_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  完成! 成功 {success_count}/{total}, 失败 {fail_count}, 总成交 {total_trades_all} 笔")
    print(f"  结果保存至: {output_json}")
    print("=" * 70)

    print_summary_table(all_results)
    return all_results


def print_summary_table(results: list):
    """打印绩效汇总表格"""
    print("\n" + "=" * 115)
    print("  绩效汇总表")
    print("=" * 115)
    header = (
        f"{'交易对':>12s} {'周期':>4s} {'总收益%':>8s} {'笔数':>5s} "
        f"{'胜率%':>7s} {'最大回撤%':>9s} {'Sharpe':>7s} "
        f"{'最大单笔亏%':>11s} {'止损次数':>8s} {'最大连亏':>8s}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        if r.get("error"):
            print(f"{r['pair']:>12s} {r['timeframe']:>4s}  ERROR: {r['error'][:60]}")
            continue

        ta = r.get("trade_analysis", {})
        ls = ta.get("liquidation_stats", {})
        row = (
            f"{r['pair']:>12s} {r['timeframe']:>4s} "
            f"{str(r.get('total_profit_pct', '-')):>8s} "
            f"{r.get('total_trades', 0):>5d} "
            f"{str(r.get('win_rate', '-')):>7s} "
            f"{str(r.get('max_drawdown', '-')):>9s} "
            f"{str(r.get('sharpe', '-')):>7s} "
            f"{str(ls.get('max_single_loss_pct', '-')):>11s} "
            f"{ls.get('stoploss_hits', 0):>8d} "
            f"{ls.get('max_consecutive_losses', 0):>8d}"
        )
        print(row)

    print("=" * 115)


if __name__ == "__main__":
    main()
