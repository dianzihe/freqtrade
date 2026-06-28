# -*- coding: utf-8 -*-
"""
Gate 全策略 x 全时间框架 x 全币种 批量回测脚本 v2
"""

import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
FREQTRADE_EXE = str(PROJECT_ROOT / ".venv" / "Scripts" / "freqtrade")
OUTPUT_DIR = PROJECT_ROOT / "user_data" / "backtest_results" / "gate_full"
CONFIG_DIR = PROJECT_ROOT / "user_data" / "config"

ALL_COINS = ["BTC","ETH","SOL","XRP","LTC","HYPE","XCN","IP","BAS","PEAQ","TA","H","VELVET","BEAT","COAI","ALLO","DN","STG"]
COINS_FUTURES = [c for c in ALL_COINS if c not in ("PEAQ", "DN")]
TIMEFRAMES = ["1m", "5m", "15m", "1h"]

STRATEGIES = [
    ("限制定投_马丁", "MemeLimitedDcaMartingaleStrategy", "spot", ALL_COINS, 4),
    ("波动率网格_马丁", "MemeVolatilityGridMartingaleStrategy", "spot", ALL_COINS, 4),
    ("反马丁_趋势", "MemeAntiMartingaleTrendStrategy", "spot", ALL_COINS, 4),
    ("对冲代理_马丁", "MemeHedgeProxyMartingaleStrategy", "spot", ALL_COINS, 4),
    ("趋势网格", "TrendFilteredGridStrategy", "spot", ALL_COINS, 4),
    ("趋势金字塔", "TrendPyramidStrategy", "spot", ALL_COINS, 5),
    ("顶部反转做空", "TopReversalShortStrategy", "futures", COINS_FUTURES, 5),
]


def parse_backtest_table(text: str) -> dict:
    """Parse the STRATEGY SUMMARY table from freqtrade output."""
    result = {"trades": "0", "profit_total_pct": "0.00", "winrate": "0.0", "max_drawdown": "0.00", "profit_factor": "N/A"}
    lines = text.split("\n")
    import re

    # Find the STRATEGY SUMMARY table and extract data row
    # Freqtrade uses unicode box chars: │ as separator
    in_summary = False
    for line in lines:
        if "STRATEGY SUMMARY" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if "│" not in line:
            continue

        parts = [p.strip() for p in line.split("│")]
        # Skip empty lines, dividers (────), and headers
        if len(parts) < 8:
            continue
        col1 = parts[1]
        if not col1 or col1 in ("Strategy", "Trades", "Avg Profit %", ""):
            continue
        if "─" in col1 or col1.startswith("-"):
            continue

        # This is a data row
        result["trades"] = parts[2]
        result["profit_total_pct"] = parts[5]
        stat_block = parts[7]
        stat_parts = stat_block.split()
        if len(stat_parts) >= 4:
            result["winrate"] = stat_parts[3]
        result["max_drawdown"] = parts[8] if len(parts) > 8 else "0.00"
        break

    # Fallback: try ASCII pipe format
    if result["trades"] == "0":
        in_table = False
        for line in lines:
            if "STRATEGY SUMMARY" in line:
                in_table = True
                continue
            if not in_table:
                continue
            if "|" not in line:
                continue
            if "---" in line:
                # Divider line, next line has data
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            col1 = parts[1]
            if not col1 or col1 in ("Strategy", "Trades", "Avg Profit %", ""):
                continue
            result["trades"] = parts[2]
            result["profit_total_pct"] = parts[5]
            stat_parts = parts[7].split()
            if len(stat_parts) >= 4:
                result["winrate"] = stat_parts[3]
            result["max_drawdown"] = parts[8] if len(parts) > 8 else "0.00"
            break

    # Find Profit Factor
    for line in lines:
        if "Profit factor" in line and "Profit factor" != line.strip().replace("│", "").strip():
            # The actual value line, not header
            sep = "│" if "│" in line else "|"
            pf_parts = [p.strip() for p in line.split(sep)]
            # Format: │ Profit factor │ value │
            if len(pf_parts) >= 3:
                result["profit_factor"] = pf_parts[2].strip()

    return result


def build_config(name: str, strategy: str, mode: str, coins: list, tf: str, max_open: int) -> dict:
    pair_fmt = f"/USDT:USDT" if mode == "futures" else f"/USDT"
    return {
        "max_open_trades": max_open,
        "stake_currency": "USDT",
        "stake_amount": 50,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "CNY",
        "dry_run": True,
        "dry_run_wallet": 1000,
        "trading_mode": mode,
        "margin_mode": "isolated" if mode == "futures" else "",
        "timeframe": tf,
        "dataformat_ohlcv": "feather",
        "position_adjustment_enable": True,
        "cancel_open_orders_on_exit": False,
        "use_exit_signal": True,
        "exit_profit_only": False,
        "ignore_roi_if_entry_signal": False,
        "unfilledtimeout": {"entry": 10, "exit": 10, "unit": "minutes"},
        "entry_pricing": {"price_side": "same", "use_order_book": True, "order_book_top": 1,
                          "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1}},
        "exit_pricing": {"price_side": "same", "use_order_book": True, "order_book_top": 1},
        "order_types": {"entry": "limit", "exit": "limit", "emergency_exit": "market",
                        "force_exit": "market", "force_entry": "market", "stoploss": "market",
                        "stoploss_on_exchange": False},
        "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
        "exchange": {
            "name": "gate", "key": "", "secret": "",
            "ccxt_config": {"proxies": {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}},
            "ccxt_async_config": {"aiohttp_proxy": "http://127.0.0.1:7890"},
            "enable_ws": False,
            "pair_whitelist": [f"{c}{pair_fmt}" for c in coins],
            "pair_blacklist": [".*3L/USDT", ".*3S/USDT", ".*5L/USDT", ".*5S/USDT"]
                              + ([] if mode == "futures" else [".*/USDT:USDT"]),
        },
        "pairlists": [{"method": "StaticPairList", "allow_inactive": True}],
        "telegram": {"enabled": False, "token": "", "chat_id": ""},
        "api_server": {"enabled": False, "listen_ip_address": "127.0.0.1", "listen_port": 8082,
                       "verbosity": "error", "enable_openapi": False,
                       "jwt_secret_key": "gate-bt-secret-key-minimum-32-chars-long",
                       "CORS_origins": [], "username": "x", "password": "x"},
        "bot_name": f"bt-{name}-{tf}",
        "internals": {"process_throttle_secs": 5},
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(STRATEGIES) * len(TIMEFRAMES)
    summary_path = OUTPUT_DIR / "backtest_summary.json"
    all_records = []

    print(f"{'='*60}")
    print(f"  Gate Full Backtest - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Strategies: {len(STRATEGIES)} | Timeframes: {len(TIMEFRAMES)} | Total: {total}")
    print(f"{'='*60}")
    print("")

    idx = 0
    for s_name, s_class, s_mode, s_coins, s_max_open in STRATEGIES:
        for tf in TIMEFRAMES:
            idx += 1
            t0 = time.time()

            config = build_config(s_name, s_class, s_mode, s_coins, tf, s_max_open)
            config_path = CONFIG_DIR / f"_bt_{s_name}_{tf}.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            export_path = OUTPUT_DIR / f"result_{s_name}_{tf}.json"

            print(f"[{idx}/{total}] {s_name:12s} | {tf:3s} | {s_mode:7s} | {len(s_coins)} coins", end=" ", flush=True)

            try:
                proc = subprocess.run(
                    [FREQTRADE_EXE, "backtesting",
                     "--config", str(config_path),
                     "--strategy", s_class,
                     "--strategy-path", str(PROJECT_ROOT / "user_data" / "strategies"),
                     "--timeframe", tf,
                     "--timerange", "20260608-20260627",
                     "--export", "signals",
                     "--export-filename", str(export_path),
                     ],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    cwd=str(PROJECT_ROOT),
                    timeout=600,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                )
                elapsed = time.time() - t0
                output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")

                if proc.returncode != 0:
                    if "No data found" in output_text:
                        print(f"-> SKIP (no data) in {elapsed:.0f}s")
                        record = {"strategy": s_name, "timeframe": tf, "mode": s_mode,
                                  "coins": len(s_coins), "status": "NO_DATA",
                                  "trades": "0", "profit_total_pct": "0", "winrate": "0",
                                  "max_drawdown": "0", "elapsed_sec": round(elapsed, 1)}
                    else:
                        err_line = [l for l in output_text.split("\n") if "ERROR" in l or "CRITICAL" in l]
                        err_msg = err_line[0][:100] if err_line else "unknown"
                        print(f"-> FAIL: {err_msg}")
                        record = {"strategy": s_name, "timeframe": tf, "mode": s_mode,
                                  "coins": len(s_coins), "status": "FAIL",
                                  "trades": "0", "profit_total_pct": "0", "winrate": "0",
                                  "max_drawdown": "0", "elapsed_sec": round(elapsed, 1)}
                else:
                    parsed = parse_backtest_table(output_text)
                    print(f"-> Trades: {parsed['trades']:>5s} | Profit: {parsed['profit_total_pct']:>7s}% | Win: {parsed['winrate']:>5s}% | DD: {parsed['max_drawdown']} | {elapsed:.0f}s")
                    record = {"strategy": s_name, "timeframe": tf, "mode": s_mode,
                              "coins": len(s_coins), "status": "OK",
                              "trades": parsed["trades"],
                              "profit_total_pct": parsed["profit_total_pct"],
                              "winrate": parsed["winrate"],
                              "max_drawdown": parsed["max_drawdown"],
                              "profit_factor": parsed.get("profit_factor", "N/A"),
                              "elapsed_sec": round(elapsed, 1)}

            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                print(f"-> TIMEOUT ({elapsed:.0f}s)")
                record = {"strategy": s_name, "timeframe": tf, "mode": s_mode,
                          "coins": len(s_coins), "status": "TIMEOUT",
                          "trades": "0", "profit_total_pct": "0", "winrate": "0",
                          "max_drawdown": "0", "elapsed_sec": round(elapsed, 1)}
            except Exception as e:
                elapsed = time.time() - t0
                print(f"-> ERROR: {e}")
                record = {"strategy": s_name, "timeframe": tf, "mode": s_mode,
                          "coins": len(s_coins), "status": "ERROR",
                          "trades": "0", "profit_total_pct": "0", "winrate": "0",
                          "max_drawdown": "0", "elapsed_sec": round(elapsed, 1),
                          "error": str(e)[:200]}

            all_records.append(record)

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # Print final summary
    ok = sum(1 for r in all_records if r["status"] == "OK")
    print(f"\n{'='*60}")
    print(f"  Complete! OK: {ok}/{len(all_records)}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
