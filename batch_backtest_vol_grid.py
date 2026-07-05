# -*- coding: utf-8 -*-
"""
================================================================================
波动率网格马丁策略 - 全币种全周期独立回测 + 综合分析报告
================================================================================
策略: MemeVolatilityGridMartingaleStrategy
数据源: user_data/data/gate
币种: BTC, ETH, SOL, XRP, XCN, H, VELVET, BEAT, COAI
周期: 1m, 5m, 15m
模式: futures (isolated margin) — 策略 can_short=True 要求期货模式
================================================================================
"""

import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
FREQTRADE_EXE = str(PROJECT_ROOT / ".venv" / "Scripts" / "freqtrade")
STRATEGY_PATH = PROJECT_ROOT / "user_data" / "strategies"
DATA_DIR = PROJECT_ROOT / "user_data" / "data" / "gate"
OUTPUT_DIR = PROJECT_ROOT / "user_data" / "backtest_results" / "vol_grid_martingale_v4"

STRATEGY_CLASS = "MemeVolatilityGridMartingaleStrategy"
TIMERANGE = "20260608-20260627"
MAINTENANCE_MARGIN_RATE = 0.005

COINS_CATEGORIES = {
    "稳定主流币": ["BTC", "ETH", "SOL", "XRP"],
    "中端中型币种": ["XCN"],
    "高波动妖币": ["H", "VELVET", "BEAT", "COAI"],
}
TIMEFRAMES = ["1m", "5m", "15m"]


# ============================================================================
# 1. Config Builder
# ============================================================================
def build_config(coin: str, timeframe: str) -> dict:
    pair = f"{coin}/USDT:USDT"
    return {
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "stake_amount": 50,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "CNY",
        "dry_run": True,
        "dry_run_wallet": 1000,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "timeframe": timeframe,
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
        "exchange": {"name": "gate", "key": "", "secret": "",
                     "pair_whitelist": [pair], "pair_blacklist": []},
        "pairlists": [{"method": "StaticPairList", "allow_inactive": True}],
        "telegram": {"enabled": False, "token": "", "chat_id": ""},
        "api_server": {"enabled": False, "listen_ip_address": "127.0.0.1",
                       "listen_port": 8082, "username": "x", "password": "x",
                       "jwt_secret_key": "somethingRandomSomethingRandom123"},
        "bot_name": f"bt-volgrid-{coin}-{timeframe}",
        "internals": {"process_throttle_secs": 5},
    }


# ============================================================================
# 2. Backtest Runner
# ============================================================================
def run_single_backtest(coin: str, timeframe: str) -> dict:
    run_dir = OUTPUT_DIR / f"{coin}_{timeframe}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(coin, timeframe)
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    bt_dir = run_dir / "backtest_output"
    bt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        FREQTRADE_EXE, "backtesting",
        "--config", str(config_path),
        "--strategy", STRATEGY_CLASS,
        "--strategy-path", str(STRATEGY_PATH),
        "--datadir", str(DATA_DIR),
        "--timeframe", timeframe,
        "--timerange", TIMERANGE,
        "--export", "trades",
        "--backtest-directory", str(bt_dir),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT),
            timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        elapsed = time.time() - t0
        output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")

        with open(run_dir / "output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)

        metrics = parse_console_output(output_text)
        metrics["coin"] = coin
        metrics["timeframe"] = timeframe
        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["status"] = "OK" if proc.returncode == 0 else "FAIL"

        # Parse result zip
        zip_files = list(bt_dir.glob("backtest-result-*.zip"))
        if zip_files:
            trades_data, wallet_df = parse_result_zip(zip_files[0])
            metrics["trades"] = trades_data or []
            metrics["wallet"] = wallet_df
            metrics["trade_count"] = len(trades_data) if trades_data else 0
            analysis = analyze_trades(trades_data or [], coin, timeframe)
            metrics["analysis"] = analysis
        else:
            metrics["trades"] = []
            metrics["trade_count"] = 0
            metrics["analysis"] = {}

        return metrics

    except subprocess.TimeoutExpired:
        return {"coin": coin, "timeframe": timeframe, "status": "TIMEOUT",
                "elapsed_sec": round(time.time() - t0, 1), "trades": [], "trade_count": 0,
                "analysis": {}}
    except Exception as e:
        return {"coin": coin, "timeframe": timeframe, "status": "ERROR",
                "elapsed_sec": round(time.time() - t0, 1), "error": str(e)[:300],
                "trades": [], "trade_count": 0, "analysis": {}}


# ============================================================================
# 3. Output Parser
# ============================================================================
def parse_console_output(text: str) -> dict:
    r = {
        "profit_total_pct": 0.0, "profit_total_abs": 0.0, "winrate": 0.0,
        "max_drawdown_pct": 0.0, "max_drawdown_abs": 0.0, "profit_factor": 0.0,
        "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "cagr": 0.0,
        "trades_per_day": 0.0, "total_trades": 0, "long_trades": 0, "short_trades": 0,
        "long_profit_pct": 0.0, "short_profit_pct": 0.0,
        "avg_duration": "", "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
        "max_consec_wins": 0, "max_consec_losses": 0,
        "starting_balance": 1000.0, "final_balance": 1000.0,
        "backtest_start": "", "backtest_end": "", "backtest_days": 0,
        "market_change": 0.0, "total_volume": 0.0, "avg_stake": 0.0,
    }
    patterns = {
        "profit_total_pct": r"Total profit %\s*[│|]\s*([-\d.]+)%",
        "profit_total_abs": r"Absolute profit\s*[│|]\s*([-\d.]+)\s*USDT",
        "winrate": r"Win%\s*[│|]\s*([\d.]+)",
        "max_drawdown_pct": r"Max % of account underwater\s*[│|]\s*([-\d.]+)%",
        "max_drawdown_abs": r"Absolute drawdown\s*[│|]\s*([-\d.]+)\s*USDT",
        "profit_factor": r"Profit factor\s*[│|]\s*([-\d.]+)",
        "sharpe": r"Sharpe \(closed trades\)\s*[│|]\s*([-\d.]+)",
        "sortino": r"Sortino \(closed trades\)\s*[│|]\s*([-\d.]+)",
        "calmar": r"Calmar \(closed trades\)\s*[│|]\s*([-\d.]+)",
        "cagr": r"CAGR %\s*[│|]\s*([-\d.]+)%",
        "trades_per_day": r"Total/Daily Avg Trades\s*[│|]\s*\d+\s*/\s*([\d.]+)",
        "total_trades": r"Total/Daily Avg Trades\s*[│|]\s*(\d+)\s*/",
        "long_trades": r"Long / Short trades\s*[│|]\s*(\d+)\s*/",
        "short_trades": r"Long / Short trades\s*[│|]\s*\d+\s*/\s*(\d+)",
        "long_profit_pct": r"Long / Short profit %\s*[│|]\s*([-\d.]+)%",
        "short_profit_pct": r"Long / Short profit %\s*[│|]\s*[-\d.]+%\s*/\s*([-\d.]+)%",
        "best_trade_pct": r"Best trade\s*[│|]\s*.*?([-\d.]+)%",
        "worst_trade_pct": r"Worst trade\s*[│|]\s*.*?([-\d.]+)%",
        "max_consec_wins": r"Max Consecutive Wins / Loss\s*[│|]\s*(\d+)\s*/",
        "max_consec_losses": r"Max Consecutive Wins / Loss\s*[│|]\s*\d+\s*/\s*(\d+)",
        "starting_balance": r"Starting balance\s*[│|]\s*([\d.]+)\s*USDT",
        "final_balance": r"Final balance\s*[│|]\s*([\d.]+)\s*USDT",
        "backtest_start": r"Backtesting from\s*[│|]\s*(.+?)\s*[│|]",
        "backtest_end": r"Backtesting to\s*[│|]\s*(.+?)\s*[│|]",
        "backtest_days": r"Backtesting with data from.*\((\d+)\s*days\)",
        "market_change": r"Market change\s*[│|]\s*([-\d.]+)%",
        "total_volume": r"Total trade volume\s*[│|]\s*([\d.]+)\s*USDT",
        "avg_stake": r"Avg\. stake amount\s*[│|]\s*([\d.]+)\s*USDT",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1).strip()
            try:
                if key in ("backtest_start", "backtest_end", "avg_duration"):
                    r[key] = val
                else:
                    r[key] = float(val)
            except (ValueError, TypeError):
                pass

    # Avg duration
    m = re.search(r"Avg Duration\s*[│|]\s*(.+?)\s*[│|]", text)
    if m:
        r["avg_duration"] = m.group(1).strip()
    # Exit reason breakdown
    exit_reasons = {}
    for m in re.finditer(r"[│|]\s*(\w[\w_]*)\s*[│|]\s*(\d+)\s*[│|]\s*([-\d.]+)\s*[│|]\s*([-\d.]+)\s*[│|]", text):
        reason = m.group(1).strip()
        if reason in ("exit_signal", "stoploss", "trail_stop", "exit_signal", "roi",
                       "custom_exit", "force_exit", "emergency_exit", "time_stop",
                       "oco_exchange_tp", "oco_exchange_sl", "failed_dca_interrupt"):
            exit_reasons[reason] = {
                "trades": int(m.group(2)),
                "avg_profit_pct": float(m.group(3)),
                "total_profit_usdt": float(m.group(4)),
            }
    r["exit_reasons"] = exit_reasons
    return r


# ============================================================================
# 4. Result Zip Parser
# ============================================================================
def parse_result_zip(zip_path: Path):
    trades_data = []
    wallet_df = None
    try:
        zf = zipfile.ZipFile(zip_path)
        for name in zf.namelist():
            if name.endswith(".json") and "config" not in name:
                content = zf.read(name).decode("utf-8")
                data = json.loads(content)
                strat = data.get("strategy", {})
                for sk, sv in strat.items():
                    trades_data = sv.get("trades", [])
                    break
                break
        for name in zf.namelist():
            if "wallet" in name and name.endswith(".feather"):
                wallet_bytes = zf.read(name)
                wallet_df = pd.read_feather(io.BytesIO(wallet_bytes))
                break
    except Exception as e:
        print(f"  [WARN] Failed to parse zip {zip_path}: {e}")
    return trades_data, wallet_df


# ============================================================================
# 5. Trade Analysis (DCA, Floating Loss, Liquidation Risk)
# ============================================================================
def analyze_trades(trades: list, coin: str, timeframe: str) -> dict:
    if not trades:
        return {}

    analysis = {
        "total_trades": len(trades),
        "dca_trades": 0,
        "max_dca_entries": 0,
        "dca_records": [],
        "max_floating_loss_pct": 0.0,
        "max_floating_loss_trade": None,
        "liquidation_near_misses": 0,
        "liquidation_would_occur": 0,
        "min_liq_distance_pct": 999.0,
        "floating_loss_distribution": {"<5%": 0, "5-10%": 0, "10-20%": 0, "20-30%": 0, ">30%": 0},
        "liq_distance_distribution": {"<5%": 0, "5-10%": 0, "10-20%": 0, "20-30%": 0, ">30%": 0},
        "exit_reason_counts": {},
        "top_winners": [],
        "top_losers": [],
        "long_count": 0,
        "short_count": 0,
    }

    for t in trades:
        is_short = t.get("is_short", False)
        open_rate = t.get("open_rate", 0)
        close_rate = t.get("close_rate", 0)
        min_rate = t.get("min_rate", open_rate)
        max_rate = t.get("max_rate", open_rate)
        leverage = t.get("leverage", 1.0) or 1.0
        profit_ratio = t.get("profit_ratio", 0)
        profit_abs = t.get("profit_abs", 0)
        exit_reason = t.get("exit_reason", "unknown")
        stake = t.get("stake_amount", 0)

        if is_short:
            analysis["short_count"] += 1
        else:
            analysis["long_count"] += 1

        # Exit reason
        analysis["exit_reason_counts"][exit_reason] = analysis["exit_reason_counts"].get(exit_reason, 0) + 1

        # --- DCA analysis ---
        orders = t.get("orders", [])
        entry_orders = [o for o in orders if o.get("ft_is_entry", False)]
        num_entries = len(entry_orders)
        if num_entries > 1:
            analysis["dca_trades"] += 1
            analysis["max_dca_entries"] = max(analysis["max_dca_entries"], num_entries)
            dca_record = {
                "pair": t.get("pair", ""),
                "is_short": is_short,
                "open_date": t.get("open_date", ""),
                "close_date": t.get("close_date", ""),
                "open_rate": open_rate,
                "close_rate": close_rate,
                "leverage": leverage,
                "num_entries": num_entries,
                "entries": [],
                "profit_ratio": profit_ratio,
                "profit_abs": profit_abs,
                "exit_reason": exit_reason,
            }
            for o in entry_orders:
                dca_record["entries"].append({
                    "side": o.get("ft_order_side", ""),
                    "price": o.get("safe_price", 0),
                    "amount": o.get("amount", 0),
                    "cost": o.get("cost", 0),
                    "time": o.get("order_filled_timestamp", 0),
                })
            analysis["dca_records"].append(dca_record)

        # --- Max floating loss ---
        if is_short:
            float_loss_pct = (open_rate - max_rate) / open_rate * leverage if open_rate > 0 else 0
        else:
            float_loss_pct = (min_rate - open_rate) / open_rate * leverage if open_rate > 0 else 0

        float_loss_pct = float_loss_pct * 100  # to percentage

        if float_loss_pct < 0:
            abs_loss = abs(float_loss_pct)
            analysis["floating_loss_distribution"]["<5%"] += 1 if abs_loss < 5 else 0
            analysis["floating_loss_distribution"]["5-10%"] += 1 if 5 <= abs_loss < 10 else 0
            analysis["floating_loss_distribution"]["10-20%"] += 1 if 10 <= abs_loss < 20 else 0
            analysis["floating_loss_distribution"]["20-30%"] += 1 if 20 <= abs_loss < 30 else 0
            analysis["floating_loss_distribution"][">30%"] += 1 if abs_loss >= 30 else 0

        if float_loss_pct < analysis["max_floating_loss_pct"]:
            analysis["max_floating_loss_pct"] = float_loss_pct
            analysis["max_floating_loss_trade"] = {
                "open_date": t.get("open_date", ""),
                "open_rate": open_rate,
                "min_rate": min_rate,
                "max_rate": max_rate,
                "is_short": is_short,
                "leverage": leverage,
                "float_loss_pct": float_loss_pct,
                "exit_reason": exit_reason,
                "profit_ratio": profit_ratio,
            }

        # --- Liquidation risk analysis ---
        # Theoretical liquidation price (isolated margin)
        # Long: liq = entry * (1 - 1/lev + mmr)
        # Short: liq = entry * (1 + 1/lev - mmr)
        if is_short:
            liq_price = open_rate * (1 + 1.0 / leverage - MAINTENANCE_MARGIN_RATE)
            liq_distance_pct = (liq_price - max_rate) / open_rate * 100 if open_rate > 0 else 0
        else:
            liq_price = open_rate * (1 - 1.0 / leverage + MAINTENANCE_MARGIN_RATE)
            liq_distance_pct = (min_rate - liq_price) / open_rate * 100 if open_rate > 0 else 0

        if liq_distance_pct < 0:
            analysis["liquidation_would_occur"] += 1
        elif liq_distance_pct < 5:
            analysis["liquidation_near_misses"] += 1

        if liq_distance_pct < analysis["min_liq_distance_pct"]:
            analysis["min_liq_distance_pct"] = liq_distance_pct

        abs_liq_dist = abs(liq_distance_pct) if liq_distance_pct < 0 else liq_distance_pct
        if abs_liq_dist < 5:
            analysis["liq_distance_distribution"]["<5%"] += 1
        elif abs_liq_dist < 10:
            analysis["liq_distance_distribution"]["5-10%"] += 1
        elif abs_liq_dist < 20:
            analysis["liq_distance_distribution"]["10-20%"] += 1
        elif abs_liq_dist < 30:
            analysis["liq_distance_distribution"]["20-30%"] += 1
        else:
            analysis["liq_distance_distribution"][">30%"] += 1

    # Top winners/losers
    sorted_trades = sorted(trades, key=lambda t: t.get("profit_abs", 0))
    analysis["top_losers"] = [
        {
            "open_date": t.get("open_date", ""),
            "close_date": t.get("close_date", ""),
            "is_short": t.get("is_short", False),
            "open_rate": t.get("open_rate", 0),
            "close_rate": t.get("close_rate", 0),
            "profit_abs": t.get("profit_abs", 0),
            "profit_ratio": t.get("profit_ratio", 0),
            "exit_reason": t.get("exit_reason", ""),
            "leverage": t.get("leverage", 1),
        }
        for t in sorted_trades[:5]
    ]
    analysis["top_winners"] = [
        {
            "open_date": t.get("open_date", ""),
            "close_date": t.get("close_date", ""),
            "is_short": t.get("is_short", False),
            "open_rate": t.get("open_rate", 0),
            "close_rate": t.get("close_rate", 0),
            "profit_abs": t.get("profit_abs", 0),
            "profit_ratio": t.get("profit_ratio", 0),
            "exit_reason": t.get("exit_reason", ""),
            "leverage": t.get("leverage", 1),
        }
        for t in sorted_trades[-5:][::-1]
    ]

    return analysis


# ============================================================================
# 6. HTML Report Generator
# ============================================================================
def generate_html_report(all_results: list) -> Path:
    report_path = OUTPUT_DIR / "vol_grid_backtest_report.html"
    coin_category = {}
    for cat, coins in COINS_CATEGORIES.items():
        for c in coins:
            coin_category[c] = cat

    # Build summary table rows
    summary_rows = []
    for r in all_results:
        coin = r.get("coin", "")
        tf = r.get("timeframe", "")
        status = r.get("status", "UNKNOWN")
        cat = coin_category.get(coin, "")
        a = r.get("analysis", {})

        row = {
            "coin": coin, "timeframe": tf, "category": cat, "status": status,
            "trades": r.get("total_trades", a.get("total_trades", 0)),
            "profit_pct": r.get("profit_total_pct", 0),
            "profit_abs": r.get("profit_total_abs", 0),
            "winrate": r.get("winrate", 0),
            "max_dd_pct": r.get("max_drawdown_pct", 0),
            "max_dd_abs": r.get("max_drawdown_abs", 0),
            "profit_factor": r.get("profit_factor", 0),
            "sharpe": r.get("sharpe", 0),
            "sortino": r.get("sortino", 0),
            "calmar": r.get("calmar", 0),
            "cagr": r.get("cagr", 0),
            "long_trades": r.get("long_trades", 0),
            "short_trades": r.get("short_trades", 0),
            "long_profit": r.get("long_profit_pct", 0),
            "short_profit": r.get("short_profit_pct", 0),
            "best_trade": r.get("best_trade_pct", 0),
            "worst_trade": r.get("worst_trade_pct", 0),
            "max_consec_wins": r.get("max_consec_wins", 0),
            "max_consec_losses": r.get("max_consec_losses", 0),
            "avg_duration": r.get("avg_duration", ""),
            "dca_trades": a.get("dca_trades", 0),
            "max_dca_entries": a.get("max_dca_entries", 0),
            "max_float_loss": a.get("max_floating_loss_pct", 0),
            "liq_near_miss": a.get("liquidation_near_misses", 0),
            "liq_would_occur": a.get("liquidation_would_occur", 0),
            "min_liq_dist": a.get("min_liq_distance_pct", 999),
            "final_balance": r.get("final_balance", 1000),
            "elapsed": r.get("elapsed_sec", 0),
        }
        summary_rows.append(row)

    # Generate per-run detail sections
    detail_sections = []
    for r in all_results:
        if r.get("status") != "OK":
            continue
        detail_sections.append(generate_detail_section(r))

    # Generate equity curve SVGs
    equity_svgs = {}
    for r in all_results:
        if r.get("status") != "OK":
            continue
        wallet_df = r.get("wallet")
        if wallet_df is not None and len(wallet_df) > 0:
            equity_svgs[f"{r['coin']}_{r['timeframe']}"] = generate_equity_svg(wallet_df, r["coin"], r["timeframe"])

    html = build_html(summary_rows, detail_sections, equity_svgs, all_results)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def generate_equity_svg(wallet_df: pd.DataFrame, coin: str, timeframe: str) -> str:
    """Generate inline SVG equity curve chart."""
    dates = wallet_df["date"].tolist()
    balances = wallet_df["balance"].tolist()
    n = len(balances)
    if n < 2:
        return "<p>数据不足</p>"

    # Downsample for performance
    step = max(1, n // 200)
    balances = balances[::step]
    n = len(balances)

    min_b = min(balances)
    max_b = max(balances)
    if max_b == min_b:
        max_b = min_b + 1
    range_b = max_b - min_b

    W, H = 600, 200
    PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 20, 30
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    pts = []
    for i, b in enumerate(balances):
        x = PAD_L + (i / max(n - 1, 1)) * plot_w
        y = PAD_T + (1 - (b - min_b) / range_b) * plot_h
        pts.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(pts)
    start_b = balances[0]
    end_b = balances[-1]
    color = "#e74c3c" if end_b < start_b else "#27ae60"

    # Drawdown area
    peak = balances[0]
    dd_pts = []
    for i, b in enumerate(balances):
        peak = max(peak, b)
        x = PAD_L + (i / max(n - 1, 1)) * plot_w
        y_peak = PAD_T + (1 - (peak - min_b) / range_b) * plot_h
        y_b = PAD_T + (1 - (b - min_b) / range_b) * plot_h
        dd_pts.append(f"{x:.1f},{y_peak:.1f} {x:.1f},{y_b:.1f}")

    return f'''
    <svg viewBox="0 0 {W} {H}" style="width:100%;max-width:600px;">
      <rect x="0" y="0" width="{W}" height="{H}" fill="#fafafa" rx="4"/>
      <text x="{W//2}" y="14" text-anchor="middle" font-size="11" fill="#555">{coin}/USDT {timeframe} - 权益曲线</text>
      <line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+plot_h}" stroke="#ccc" stroke-width="0.5"/>
      <line x1="{PAD_L}" y1="{PAD_T+plot_h}" x2="{PAD_L+plot_w}" y2="{PAD_T+plot_h}" stroke="#ccc" stroke-width="0.5"/>
      <text x="{PAD_L-5}" y="{PAD_T+8}" text-anchor="end" font-size="9" fill="#888">{max_b:.1f}</text>
      <text x="{PAD_L-5}" y="{PAD_T+plot_h}" text-anchor="end" font-size="9" fill="#888">{min_b:.1f}</text>
      <text x="{PAD_L-5}" y="{PAD_T+plot_h//2}" text-anchor="end" font-size="9" fill="#888">{(max_b+min_b)/2:.1f}</text>
      <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>
      <text x="{PAD_L+plot_w-5}" y="{PAD_T+12}" text-anchor="end" font-size="9" fill="{color}">
        起始: {start_b:.2f} -> 终值: {end_b:.2f} ({(end_b/start_b-1)*100:+.2f}%)
      </text>
    </svg>'''


def generate_detail_section(r: dict) -> str:
    coin = r["coin"]
    tf = r["timeframe"]
    a = r.get("analysis", {})
    trades = r.get("trades", [])

    # DCA records table
    dca_html = ""
    dca_records = a.get("dca_records", [])
    if dca_records:
        dca_rows = ""
        for dr in dca_records:
            entries_html = ""
            for e in dr["entries"]:
                entries_html += f"<div>  {e['side']} @ {e['price']:.6f} amt={e['amount']:.6f} cost={e['cost']:.2f}</div>"
            dca_rows += f"""
            <tr>
              <td>{dr['open_date'][:19]}</td>
              <td>{'做空' if dr['is_short'] else '做多'}</td>
              <td>{dr['num_entries']}</td>
              <td>{dr['leverage']:.1f}x</td>
              <td>{dr['open_rate']:.6f}</td>
              <td>{dr['close_rate']:.6f}</td>
              <td style="color:{'#e74c3c' if dr['profit_abs'] < 0 else '#27ae60'}">{dr['profit_abs']:.4f}</td>
              <td>{dr['exit_reason']}</td>
            </tr>"""
        dca_html = f"""
        <h5>加仓触发记录 (DCA Trades: {len(dca_records)})</h5>
        <table class="detail-table">
          <tr><th>开仓时间</th><th>方向</th><th>加仓次数</th><th>杠杆</th><th>开仓价</th><th>平仓价</th><th>盈亏(USDT)</th><th>退出原因</th></tr>
          {dca_rows}
        </table>"""
    else:
        dca_html = "<p>无加仓触发记录</p>"

    # Top winners/losers
    winners_html = ""
    for w in a.get("top_winners", []):
        winners_html += f"<tr><td>{w['open_date'][:19]}</td><td>{'空' if w['is_short'] else '多'}</td><td>{w['leverage']:.1f}x</td><td>{w['open_rate']:.6f}</td><td>{w['close_rate']:.6f}</td><td style='color:#27ae60'>{w['profit_abs']:.4f}</td><td>{w['exit_reason']}</td></tr>"
    losers_html = ""
    for l in a.get("top_losers", []):
        losers_html += f"<tr><td>{l['open_date'][:19]}</td><td>{'空' if l['is_short'] else '多'}</td><td>{l['leverage']:.1f}x</td><td>{l['open_rate']:.6f}</td><td>{l['close_rate']:.6f}</td><td style='color:#e74c3c'>{l['profit_abs']:.4f}</td><td>{l['exit_reason']}</td></tr>"

    # Floating loss & liquidation risk
    fl = a.get("max_floating_loss_pct", 0)
    fl_trade = a.get("max_floating_loss_trade") or {}
    liq_miss = a.get("liquidation_near_misses", 0)
    liq_occur = a.get("liquidation_would_occur", 0)
    min_liq = a.get("min_liq_distance_pct", 999)

    # Pre-compute floating loss trade display values
    fl_open_date = str(fl_trade.get("open_date", ""))[:19] if fl_trade else "N/A"
    fl_open_rate = f"{fl_trade.get('open_rate', 0):.6f}" if fl_trade else "N/A"
    fl_min_rate = f"{fl_trade.get('min_rate', 0):.6f}" if fl_trade else "N/A"
    fl_max_rate = f"{fl_trade.get('max_rate', 0):.6f}" if fl_trade else "N/A"
    fl_direction = "做空" if fl_trade.get("is_short") else ("做多" if fl_trade else "N/A")
    fl_leverage = f"{fl_trade.get('leverage', 0):.1f}x" if fl_trade else "N/A"

    fl_dist = a.get("floating_loss_distribution", {})
    liq_dist = a.get("liq_distance_distribution", {})

    # Exit reasons
    exit_counts = a.get("exit_reason_counts", {})
    exit_html = ""
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        exit_html += f"<tr><td>{reason}</td><td>{count}</td><td>{count/max(len(trades),1)*100:.1f}%</td></tr>"

    # Equity curve SVG
    wallet_df = r.get("wallet")
    equity_svg = ""
    if wallet_df is not None and len(wallet_df) > 0:
        equity_svg = generate_equity_svg(wallet_df, coin, tf)

    return f"""
    <div class="detail-card" id="{coin}_{tf}">
      <h3>{coin}/USDT:USDT - {tf}</h3>
      <div class="detail-grid">
        <div class="detail-left">
          <h5>核心绩效指标</h5>
          <table class="metrics-table">
            <tr><td>总交易数</td><td>{r.get('total_trades', 0)}</td><td>日均交易</td><td>{r.get('trades_per_day', 0):.2f}</td></tr>
            <tr><td>总收益率</td><td style="color:{'#e74c3c' if r.get('profit_total_pct',0)<0 else '#27ae60'}">{r.get('profit_total_pct', 0):+.2f}%</td><td>绝对盈亏</td><td style="color:{'#e74c3c' if r.get('profit_total_abs',0)<0 else '#27ae60'}">{r.get('profit_total_abs', 0):+.4f} USDT</td></tr>
            <tr><td>胜率</td><td>{r.get('winrate', 0):.1f}%</td><td>盈亏比</td><td>{r.get('profit_factor', 0):.2f}</td></tr>
            <tr><td>最大回撤</td><td style="color:#e74c3c">{r.get('max_drawdown_pct', 0):.2f}%</td><td>回撤金额</td><td>{r.get('max_drawdown_abs', 0):.2f} USDT</td></tr>
            <tr><td>Sharpe</td><td>{r.get('sharpe', 0):.2f}</td><td>Sortino</td><td>{r.get('sortino', 0):.2f}</td></tr>
            <tr><td>Calmar</td><td>{r.get('calmar', 0):.2f}</td><td>CAGR</td><td>{r.get('cagr', 0):.2f}%</td></tr>
            <tr><td>多/空交易</td><td>{r.get('long_trades', 0)} / {r.get('short_trades', 0)}</td><td>多/空收益</td><td>{r.get('long_profit_pct', 0):+.2f}% / {r.get('short_profit_pct', 0):+.2f}%</td></tr>
            <tr><td>最佳交易</td><td style="color:#27ae60">{r.get('best_trade_pct', 0):+.2f}%</td><td>最差交易</td><td style="color:#e74c3c">{r.get('worst_trade_pct', 0):+.2f}%</td></tr>
            <tr><td>最大连胜</td><td>{r.get('max_consec_wins', 0)}</td><td>最大连亏</td><td>{r.get('max_consec_losses', 0)}</td></tr>
            <tr><td>平均持仓</td><td>{r.get('avg_duration', '')}</td><td>市场变化</td><td>{r.get('market_change', 0):+.2f}%</td></tr>
            <tr><td>起始余额</td><td>{r.get('starting_balance', 0):.2f} USDT</td><td>终值余额</td><td>{r.get('final_balance', 0):.2f} USDT</td></tr>
          </table>

          <h5>退出原因分布</h5>
          <table class="detail-table">
            <tr><th>退出原因</th><th>次数</th><th>占比</th></tr>
            {exit_html}
          </table>
        </div>

        <div class="detail-right">
          <h5>盈亏曲线</h5>
          {equity_svg}

          <h5>最大浮亏分析</h5>
          <table class="metrics-table">
            <tr><td>最大浮亏(含杠杆)</td><td style="color:#e74c3c">{fl:.2f}%</td></tr>
            <tr><td>浮亏交易-开仓时间</td><td>{fl_open_date}</td></tr>
            <tr><td>浮亏交易-开仓价</td><td>{fl_open_rate}</td></tr>
            <tr><td>浮亏交易-最低价</td><td>{fl_min_rate}</td></tr>
            <tr><td>浮亏交易-最高价</td><td>{fl_max_rate}</td></tr>
            <tr><td>浮亏交易-方向</td><td>{fl_direction}</td></tr>
            <tr><td>浮亏交易-杠杆</td><td>{fl_leverage}</td></tr>
          </table>
          <div class="dist-bar">
            <span>浮亏分布: &lt;5%: {fl_dist.get('<5%',0)} | 5-10%: {fl_dist.get('5-10%',0)} | 10-20%: {fl_dist.get('10-20%',0)} | 20-30%: {fl_dist.get('20-30%',0)} | &gt;30%: {fl_dist.get('>30%',0)}</span>
          </div>

          <h5>爆仓风险统计</h5>
          <table class="metrics-table">
            <tr><td>理论爆仓次数</td><td style="color:{'#e74c3c' if liq_occur > 0 else '#27ae60'}">{liq_occur}</td></tr>
            <tr><td>接近爆仓(&lt;5%)</td><td style="color:{'#e74c3c' if liq_miss > 0 else '#27ae60'}">{liq_miss}</td></tr>
            <tr><td>最小距爆仓距离</td><td>{min_liq:.2f}%</td></tr>
          </table>
          <div class="dist-bar">
            <span>距爆仓分布: &lt;5%: {liq_dist.get('<5%',0)} | 5-10%: {liq_dist.get('5-10%',0)} | 10-20%: {liq_dist.get('10-20%',0)} | 20-30%: {liq_dist.get('20-30%',0)} | &gt;30%: {liq_dist.get('>30%',0)}</span>
          </div>

          <h5>Top 5 盈利交易</h5>
          <table class="detail-table">
            <tr><th>时间</th><th>方向</th><th>杠杆</th><th>开仓价</th><th>平仓价</th><th>盈亏</th><th>退出</th></tr>
            {winners_html}
          </table>

          <h5>Top 5 亏损交易</h5>
          <table class="detail-table">
            <tr><th>时间</th><th>方向</th><th>杠杆</th><th>开仓价</th><th>平仓价</th><th>盈亏</th><th>退出</th></tr>
            {losers_html}
          </table>
        </div>
      </div>
      {dca_html}
    </div>"""


def build_html(summary_rows, detail_sections, equity_svgs, all_results):
    # Summary table
    summary_html = ""
    for row in sorted(summary_rows, key=lambda r: (r["coin"], r["timeframe"])):
        profit_color = "#e74c3c" if row["profit_pct"] < 0 else "#27ae60"
        dd_color = "#e74c3c" if abs(row["max_dd_pct"]) > 5 else "#333"
        status_badge = "ok" if row["status"] == "OK" else "fail"
        summary_html += f"""
        <tr>
          <td>{row['coin']}</td>
          <td>{row['timeframe']}</td>
          <td>{row['category']}</td>
          <td><span class="badge {status_badge}">{row['status']}</span></td>
          <td>{row['trades']}</td>
          <td style="color:{profit_color}">{row['profit_pct']:+.2f}%</td>
          <td style="color:{profit_color}">{row['profit_abs']:+.2f}</td>
          <td>{row['winrate']:.1f}%</td>
          <td style="color:{dd_color}">{row['max_dd_pct']:.2f}%</td>
          <td>{row['profit_factor']:.2f}</td>
          <td>{row['sharpe']:.2f}</td>
          <td>{row['long_trades']}/{row['short_trades']}</td>
          <td>{row['dca_trades']}</td>
          <td style="color:#e74c3c">{row['max_float_loss']:.1f}%</td>
          <td>{row['liq_near_miss']}</td>
          <td style="color:{'#e74c3c' if row['liq_would_occur']>0 else '#27ae60'}">{row['liq_would_occur']}</td>
          <td>{row['final_balance']:.2f}</td>
          <td>{row['elapsed']:.0f}s</td>
        </tr>"""

    details_html = "\n".join(detail_sections)

    # Category summary
    cat_summary = {}
    for row in summary_rows:
        cat = row["category"]
        if cat not in cat_summary:
            cat_summary[cat] = {"count": 0, "total_profit": 0, "total_trades": 0, "total_dd": 0, "wins": 0}
        cat_summary[cat]["count"] += 1
        cat_summary[cat]["total_profit"] += row["profit_pct"]
        cat_summary[cat]["total_trades"] += row["trades"]
        cat_summary[cat]["total_dd"] += abs(row["max_dd_pct"])
        if row["winrate"] > 50:
            cat_summary[cat]["wins"] += 1

    cat_html = ""
    for cat, stats in cat_summary.items():
        avg_profit = stats["total_profit"] / stats["count"] if stats["count"] else 0
        cat_html += f"""
        <div class="cat-card">
          <h4>{cat}</h4>
          <p>组合数: {stats['count']} | 平均收益: {avg_profit:+.2f}% | 总交易: {stats['total_trades']} | 胜率>50%组合: {stats['wins']}</p>
        </div>"""

    ok_count = sum(1 for r in all_results if r.get("status") == "OK")
    total_count = len(all_results)
    profit_count = sum(1 for r in all_results if r.get("profit_total_pct", 0) > 0)
    loss_count = sum(1 for r in all_results if r.get("profit_total_pct", 0) < 0)
    total_dca = sum(r.get("analysis", {}).get("dca_trades", 0) for r in all_results)
    total_liq = sum(r.get("analysis", {}).get("liquidation_would_occur", 0) for r in all_results)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>波动率网格马丁策略 - 全币种全周期回测报告</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ text-align: center; color: #2c3e50; margin: 20px 0; font-size: 24px; }}
  h2 {{ color: #2c3e50; margin: 30px 0 15px; font-size: 20px; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
  h3 {{ color: #34495e; margin: 20px 0 10px; font-size: 18px; }}
  h5 {{ color: #2c3e50; margin: 15px 0 8px; font-size: 14px; font-weight: 600; }}
  .header-info {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .header-info p {{ margin: 4px 0; }}
  .summary-table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 12px; }}
  .summary-table th {{ background: #34495e; color: #fff; padding: 10px 6px; text-align: center; white-space: nowrap; }}
  .summary-table td {{ padding: 6px; text-align: center; border-bottom: 1px solid #eee; white-space: nowrap; }}
  .summary-table tr:hover {{ background: #f0f7ff; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge.ok {{ background: #27ae60; color: #fff; }}
  .badge.fail {{ background: #e74c3c; color: #fff; }}
  .cat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 15px 0; }}
  .cat-card {{ background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3498db; }}
  .detail-card {{ background: #fff; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 1px 5px rgba(0,0,0,0.1); }}
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 900px) {{ .detail-grid {{ grid-template-columns: 1fr; }} }}
  .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0; }}
  .metrics-table td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
  .metrics-table td:first-child {{ color: #7f8c8d; }}
  .metrics-table td:nth-child(3) {{ color: #7f8c8d; }}
  .detail-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin: 8px 0; }}
  .detail-table th {{ background: #ecf0f1; padding: 5px; text-align: left; }}
  .detail-table td {{ padding: 4px 5px; border-bottom: 1px solid #eee; }}
  .dist-bar {{ background: #ecf0f1; padding: 6px 10px; border-radius: 4px; font-size: 11px; margin: 5px 0; color: #555; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0; }}
  .stat-box {{ background: #fff; padding: 12px; border-radius: 6px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
  .stat-box .label {{ font-size: 11px; color: #7f8c8d; }}
  .stat-box .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
  .nav {{ position: sticky; top: 0; background: #fff; padding: 10px 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 100; display: flex; gap: 10px; flex-wrap: wrap; font-size: 12px; }}
  .nav a {{ color: #3498db; text-decoration: none; padding: 2px 8px; border-radius: 3px; }}
  .nav a:hover {{ background: #ebf5fb; }}
</style>
</head>
<body>
<div class="nav">
  <a href="#summary">汇总表</a>
  <a href="#categories">分类对比</a>
  <a href="#details">详细分析</a>
</div>
<div class="container">
  <h1>波动率网格马丁策略 - 全币种全周期独立回测报告</h1>
  <div class="header-info">
    <p><strong>策略:</strong> MemeVolatilityGridMartingaleStrategy (Meme_波动率网格_马丁.py)</p>
    <p><strong>数据源:</strong> user_data/data/gate (Gate.io 永续合约数据)</p>
    <p><strong>回测时间:</strong> {TIMERANGE} (2026-06-08 ~ 2026-06-27)</p>
    <p><strong>交易模式:</strong> Isolated Futures (隔离保证金期货)</p>
    <p><strong>初始资金:</strong> 1000 USDT | <strong>单笔开仓:</strong> 50 USDT | <strong>最大持仓:</strong> 1</p>
    <p><strong>币种分类:</strong> 稳定主流币(BTC,ETH,SOL,XRP) | 中端中型币种(XCN) | 高波动妖币(H,VELVET,BEAT,COAI)</p>
    <p><strong>周期:</strong> 1m, 5m, 15m | <strong>总回测组数:</strong> {total_count} (成功: {ok_count})</p>
    <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-box"><div class="label">回测组数</div><div class="value">{total_count}</div></div>
    <div class="stat-box"><div class="label">成功</div><div class="value" style="color:#27ae60">{ok_count}</div></div>
    <div class="stat-box"><div class="label">失败</div><div class="value" style="color:#e74c3c">{total_count - ok_count}</div></div>
    <div class="stat-box"><div class="label">盈利组合</div><div class="value" style="color:#27ae60">{profit_count}</div></div>
    <div class="stat-box"><div class="label">亏损组合</div><div class="value" style="color:#e74c3c">{loss_count}</div></div>
    <div class="stat-box"><div class="label">DCA触发</div><div class="value">{total_dca}</div></div>
    <div class="stat-box"><div class="label">爆仓风险</div><div class="value" style="color:#e74c3c">{total_liq}</div></div>
  </div>

  <h2 id="summary">一、全量绩效汇总表</h2>
  <table class="summary-table">
    <thead>
      <tr>
        <th>币种</th><th>周期</th><th>分类</th><th>状态</th>
        <th>交易数</th><th>收益率</th><th>盈亏(USDT)</th><th>胜率</th>
        <th>最大回撤</th><th>盈亏比</th><th>Sharpe</th>
        <th>多/空</th><th>DCA交易</th><th>最大浮亏</th>
        <th>接近爆仓</th><th>理论爆仓</th><th>终值</th><th>耗时</th>
      </tr>
    </thead>
    <tbody>
      {summary_html}
    </tbody>
  </table>

  <h2 id="categories">二、分类对比</h2>
  <div class="cat-grid">{cat_html}</div>

  <h2 id="details">三、逐币逐周期详细分析</h2>
  {details_html}
</div>
</body>
</html>"""


# ============================================================================
# 7. Main
# ============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_coins = [(coin, cat) for cat, coins in COINS_CATEGORIES.items() for coin in coins]
    total = len(all_coins) * len(TIMEFRAMES)

    print(f"{'='*70}")
    print(f"  波动率网格马丁策略 - 全币种全周期独立回测")
    print(f"  策略: {STRATEGY_CLASS}")
    print(f"  币种: {len(all_coins)} 个 | 周期: {len(TIMEFRAMES)} 种 | 总计: {total} 组")
    print(f"  时间范围: {TIMERANGE}")
    print(f"  模式: futures (isolated)")
    print(f"{'='*70}")
    print()

    all_results = []
    tasks = []
    for coin, cat in all_coins:
        for tf in TIMEFRAMES:
            tasks.append((coin, tf))

    # Run with 3 parallel workers
    completed = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {executor.submit(run_single_backtest, coin, tf): (coin, tf)
                          for coin, tf in tasks}
        for future in as_completed(future_to_task):
            coin, tf = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                all_results.append(result)
                if result["status"] == "OK":
                    a = result.get("analysis", {})
                    print(f"[{completed}/{total}] {coin:8s} {tf:3s} -> "
                          f"交易: {result.get('total_trades', 0):>3} | "
                          f"收益: {result.get('profit_total_pct', 0):>+7.2f}% | "
                          f"胜率: {result.get('winrate', 0):>5.1f}% | "
                          f"回撤: {result.get('max_drawdown_pct', 0):>6.2f}% | "
                          f"DCA: {a.get('dca_trades', 0)} | "
                          f"浮亏: {a.get('max_floating_loss_pct', 0):>6.1f}% | "
                          f"{result.get('elapsed_sec', 0):.0f}s")
                else:
                    print(f"[{completed}/{total}] {coin:8s} {tf:3s} -> {result['status']} "
                          f"({result.get('elapsed_sec', 0):.0f}s)")
            except Exception as e:
                print(f"[{completed}/{total}] {coin:8s} {tf:3s} -> EXCEPTION: {e}")
                all_results.append({"coin": coin, "timeframe": tf, "status": "EXCEPTION",
                                    "error": str(e)[:300], "trades": [], "analysis": {}})

    # Sort results for consistent ordering
    all_results.sort(key=lambda r: (r.get("coin", ""), r.get("timeframe", "")))

    # Save summary JSON
    summary_path = OUTPUT_DIR / "summary.json"
    serializable_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ("wallet",)}
        if "trades" in sr and isinstance(sr["trades"], list):
            sr["trade_count"] = len(sr["trades"])
        serializable_results.append(sr)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

    # Generate HTML report
    print(f"\n{'='*70}")
    print("  生成综合分析报告...")
    report_path = generate_html_report(all_results)

    ok = sum(1 for r in all_results if r.get("status") == "OK")
    profitable = sum(1 for r in all_results if r.get("profit_total_pct", 0) > 0)
    total_dca = sum(r.get("analysis", {}).get("dca_trades", 0) for r in all_results)
    total_liq = sum(r.get("analysis", {}).get("liquidation_would_occur", 0) for r in all_results)
    print(f"\n{'='*70}")
    print(f"  回测完成!")
    print(f"  成功: {ok}/{total} | 盈利: {profitable}/{ok}")
    print(f"  DCA触发总次数: {total_dca} | 理论爆仓总次数: {total_liq}")
    print(f"  结果目录: {OUTPUT_DIR}")
    print(f"  综合报告: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
