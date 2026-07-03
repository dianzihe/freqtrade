# -*- coding: utf-8 -*-
"""
Meme_对冲代理_马丁 全币种 x 全周期 Hyperopt 参数优化脚本
=========================================================

优化目标: 对 MemeHedgeProxyMartingaleStrategy 进行 buy/sell 空间超参搜索
数据源:   user_data/data/gate (feather 格式)
时间范围: 2026-06-08 → 2026-06-27

币种分组:
  - 稳定主流币: BTC, ETH, SOL, XRP
  - 中端中型币种: XCN
  - 高波动妖币: H, VELVET, BEAT, COAI

K线周期: 1m, 5m, 15m

输出:
  - 每币种每周期独立 hyperopt 结果
  - 最佳参数 + backtest 验证
  - 综合 HTML 报告（含绩效指标、盈亏曲线、最大浮亏、加仓记录、爆仓风险）

用法:
  python hyperopt_meme_hedge.py [--epochs 50] [--only-coin BTC] [--only-tf 5m]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
STRATEGY_PATH = str(PROJECT_ROOT / "user_data" / "strategies")
CONFIG_DIR = PROJECT_ROOT / "user_data" / "config"
DATA_DIR = PROJECT_ROOT / "user_data" / "data" / "gate"
OUTPUT_DIR = PROJECT_ROOT / "user_data" / "hyperopt_results" / "meme_hedge"
REPORT_DIR = OUTPUT_DIR / "reports"
TRADE_EXPORT_DIR = OUTPUT_DIR / "trades"
PARAMS_DIR = OUTPUT_DIR / "params"
LOG_DIR = OUTPUT_DIR / "logs"

# ── Run Configuration ──────────────────────────────────────────────────
COIN_GROUPS = {
    "稳定主流币": ["BTC", "ETH", "SOL", "XRP"],
    "中端中型币种": ["XCN"],
    "高波动妖币": ["H", "VELVET", "BEAT", "COAI"],
}
ALL_COINS = [c for g in COIN_GROUPS.values() for c in g]
TIMEFRAMES = ["1m", "5m", "15m"]
TIMERANGE = "20260608-20260627"
DEFAULT_EPOCHS = 100
HYPEROPT_LOSS = "SharpeHyperOptLoss"
RANDOM_STATE = 42
HYPEROPT_SPACES = "buy sell"
MIN_TRADES = 1
FREQTRADE_MODULE = ["-m", "freqtrade"]


def coin_group(coin: str) -> str:
    for grp, coins in COIN_GROUPS.items():
        if coin in coins:
            return grp
    return "unknown"


def build_config(pair: str, tf: str, proxy_host: str = "127.0.0.1:7890") -> dict:
    """Generate per-run hyperopt config."""
    return {
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "stake_amount": 50,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "CNY",
        "dry_run": True,
        "dry_run_wallet": 1000,
        "trading_mode": "spot",
        "timeframe": tf,
        "dataformat_ohlcv": "feather",
        "position_adjustment_enable": True,
        "cancel_open_orders_on_exit": False,
        "use_exit_signal": True,
        "exit_profit_only": False,
        "ignore_roi_if_entry_signal": False,
        "unfilledtimeout": {"entry": 10, "exit": 10, "unit": "minutes"},
        "entry_pricing": {"price_side": "same", "use_order_book": False},
        "exit_pricing": {"price_side": "same", "use_order_book": False},
        "order_types": {
            "entry": "limit", "exit": "limit",
            "emergency_exit": "market", "force_exit": "market",
            "force_entry": "market", "stoploss": "market",
            "stoploss_on_exchange": False,
        },
        "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
        "exchange": {
            "name": "gate",
            "key": "dummy",
            "secret": "dummy",
            "ccxt_config": {
                "enableRateLimit": False,
                "proxies": {
                    "http": f"http://{proxy_host}",
                    "https": f"http://{proxy_host}",
                },
            },
            "ccxt_async_config": {
                "aiohttp_proxy": f"http://{proxy_host}",
            },
            "enable_ws": False,
            "pair_whitelist": [f"{pair}/USDT"],
            "pair_blacklist": [
                ".*3L/USDT", ".*3S/USDT", ".*5L/USDT", ".*5S/USDT",
                ".*/USDT:USDT",
            ],
        },
        "pairlists": [{"method": "StaticPairList", "allow_inactive": True}],
        "telegram": {"enabled": False, "token": "dummy", "chat_id": "dummy"},
        "api_server": {
            "enabled": False, "listen_ip_address": "127.0.0.1",
            "listen_port": 8082, "username": "x", "password": "x",
            "jwt_secret_key": "hyperopt-secret-key-minimum-32-chars",
        },
        "internals": {"process_throttle_secs": 5},
    }


# ── Output Parsers ────────────────────────────────────────────────────

def parse_hyperopt_output(text: str) -> dict:
    """Parse freqtrade hyperopt output to extract best params and results."""
    result = {
        "best_loss": None,
        "best_params": {},
        "total_profit": None,
        "winrate": None,
        "max_drawdown": None,
        "trades": None,
        "error": None,
    }

    lines = text.split("\n")

    # Extract best result line
    # Format: * Best: {loss} | ...
    for line in lines:
        m = re.search(r"\*\s*Best:\s+([-\d.]+)", line)
        if m:
            result["best_loss"] = float(m.group(1))
            break

    # Extract best parameters block
    in_params = False
    params_text = []
    for line in lines:
        if "Best result" in line or "Buy hyperspace params" in line:
            in_params = True
            continue
        if in_params:
            if line.strip() == "" or "=======" in line or "Sell hyperspace" in line:
                break
            params_text.append(line.strip())

    for line in params_text:
        m = re.match(r"(\w+)\s*=\s*([-\d.]+)", line)
        if m:
            key, val = m.group(1), m.group(2)
            try:
                result["best_params"][key] = int(val) if val.isdigit() or (val.startswith("-") and val[1:].isdigit()) else float(val)
            except ValueError:
                result["best_params"][key] = val

    # Fallback: try JSON-style output (--print-json)
    if not result["best_params"]:
        try:
            json_start = text.index('"params"')
            json_block = text[json_start - 50:]
        except ValueError:
            json_block = None
        if json_block:
            for line in json_block.split("\n"):
                m = re.match(r'\s*"(\w+)":\s*([-\d.]+)', line)
                if m:
                    key, val = m.group(1), m.group(2)
                    try:
                        result["best_params"][key] = int(val) if val.isdigit() else float(val)
                    except ValueError:
                        result["best_params"][key] = val

    # Extract backtest summary from hyperopt output
    in_bt = False
    for line in lines:
        if "BACKTESTING REPORT" in line:
            in_bt = True
            continue
        if in_bt and "│" in line:
            parts = [p.strip() for p in line.split("│")]
            if len(parts) >= 9 and parts[1] not in ("", "Pairs", "───"):
                try:
                    result["trades"] = parts[2]
                    result["total_profit"] = parts[5]
                    result["max_drawdown"] = parts[8] if len(parts) > 8 else "N/A"
                except (IndexError, ValueError):
                    pass
                break

    # Extract winrate
    for line in lines:
        m = re.search(r"Winrate:\s*([\d.]+)%", line)
        if m:
            result["winrate"] = m.group(1)
            break

    return result


def parse_backtest_output(text: str) -> dict:
    """Parse freqtrade backtest output for detailed metrics."""
    result = {
        "trades": "0", "profit_total_pct": "0.00", "winrate": "0.0",
        "max_drawdown": "0.00", "profit_factor": "N/A",
        "avg_profit_pct": "0.00", "sharpe": "N/A", "sortino": "N/A",
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "max_consecutive_losses": 0, "error": None,
    }

    lines = text.split("\n")

    # STRATEGY SUMMARY table
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
        if len(parts) < 9:
            continue
        if parts[1] in ("", "Strategy", "Trades", "───", "Pairs"):
            continue
        # Data row
        result["trades"] = parts[2]
        result["profit_total_pct"] = parts[5]
        result["max_drawdown"] = parts[8] if len(parts) > 8 else "0.00"
        stat_block = parts[7] if len(parts) > 7 else ""
        stat_parts = stat_block.split()
        if len(stat_parts) >= 4:
            result["total_trades"] = int(stat_parts[0]) if stat_parts[0].isdigit() else 0
            result["winning_trades"] = int(stat_parts[1]) if stat_parts[1].isdigit() else 0
            result["losing_trades"] = int(stat_parts[2]) if stat_parts[2].isdigit() else 0
            result["winrate"] = stat_parts[3]
        if len(parts) > 9:
            result["avg_profit_pct"] = parts[9]
        break

    # Profit factor
    for line in lines:
        if "Profit factor" in line and "│" in line:
            pf_parts = [p.strip() for p in line.split("│")]
            if len(pf_parts) >= 3:
                val = pf_parts[2].strip()
                if val and val != "Profit factor":
                    result["profit_factor"] = val

    # Sharpe / Sortino
    for line in lines:
        m = re.search(r"Sharpe.*?ratio.*?([-\d.]+)", line, re.IGNORECASE)
        if m:
            result["sharpe"] = m.group(1)
            break
    for line in lines:
        m = re.search(r"Sortino.*?ratio.*?([-\d.]+)", line, re.IGNORECASE)
        if m:
            result["sortino"] = m.group(1)
            break

    # Max consecutive losses - approximate from trade analysis
    for line in lines:
        m = re.search(r"Max\s+consecutive\s+losses?[:\s]+(\d+)", line, re.IGNORECASE)
        if m:
            result["max_consecutive_losses"] = int(m.group(1))
            break

    return result


def parse_trade_json(trade_file: Path) -> dict:
    """Parse exported trade JSON for DCA analysis."""
    analysis = {
        "total_trades": 0, "dca_trades": 0, "dca_levels": defaultdict(int),
        "dca_max_level": 0, "liquidations": 0, "max_drawdown_trade": 0.0,
        "profit_curve": [], "drawdown_curve": [],
    }
    if not trade_file.exists():
        return analysis

    try:
        with open(trade_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return analysis

    trades = data.get("trades", [])
    analysis["total_trades"] = len(trades)

    cumulative_profit = 0.0
    peak = 0.0
    for i, t in enumerate(trades):
        pnl = t.get("profit_abs", 0) or 0
        cumulative_profit += pnl
        analysis["profit_curve"].append({"x": i, "y": round(cumulative_profit, 4)})

        if cumulative_profit > peak:
            peak = cumulative_profit
        dd = peak - cumulative_profit
        analysis["drawdown_curve"].append({"x": i, "y": round(dd, 4)})
        analysis["max_drawdown_trade"] = max(analysis["max_drawdown_trade"], dd)

        # DCA detection
        enter_tag = t.get("enter_tag", "")
        if enter_tag and "dca" in enter_tag.lower():
            analysis["dca_trades"] += 1
        # Count entries via orders
        orders = t.get("orders", [])
        entry_orders = [o for o in orders if o.get("ft_is_entry")]
        if len(entry_orders) > 1:
            level = len(entry_orders) - 1
            analysis["dca_levels"][level] += 1
            analysis["dca_max_level"] = max(analysis["dca_max_level"], level)

        # Liquidation detection: if exit_reason contains "liquidation"
        exit_reason = t.get("exit_reason", "")
        if "liquidat" in exit_reason.lower():
            analysis["liquidations"] += 1

    return analysis


# ── Command Runners ────────────────────────────────────────────────────

def run_hyperopt(pair: str, tf: str, epochs: int, cwd: str, proxy_host: str) -> dict:
    """Run hyperopt for a single coin × timeframe."""
    config = build_config(pair, tf, proxy_host)
    config_path = CONFIG_DIR / f"_hp_{pair}_{tf}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    pair_str = f"{pair}/USDT"
    cmd = [
        PYTHON_EXE, *FREQTRADE_MODULE, "hyperopt",
        "--config", str(config_path),
        "--strategy", "MemeHedgeProxyMartingaleStrategy",
        "--strategy-path", STRATEGY_PATH,
        "--timeframe", tf,
        "--timerange", TIMERANGE,
        "-p", pair_str,
        "--spaces", *HYPEROPT_SPACES.split(),
        "--epochs", str(epochs),
        "--hyperopt-loss", HYPEROPT_LOSS,
        "--disable-param-export",
        "--random-state", str(RANDOM_STATE),
        "--min-trades", str(MIN_TRADES),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=cwd, timeout=1800,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        elapsed = time.time() - t0
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")

        if proc.returncode != 0:
            return {"status": "FAIL", "error": _extract_error(output),
                    "elapsed_sec": elapsed, "output": output}

        result = parse_hyperopt_output(output)
        result["status"] = "OK"
        result["elapsed_sec"] = elapsed
        result["output"] = output
        return result

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": "1800s timeout",
                "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:300],
                "elapsed_sec": time.time() - t0}


def run_backtest(pair: str, tf: str, params: dict, cwd: str, proxy_host: str) -> dict:
    """Run full backtest with optimized params.

    We use the strategy's auto-loading mechanism: write params as JSON next to
    the strategy file (replacing any existing one), run backtest, then archive.
    """
    config = build_config(pair, tf, proxy_host)
    config_path = CONFIG_DIR / f"_bt_{pair}_{tf}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    pair_str = f"{pair}/USDT"
    trade_file = TRADE_EXPORT_DIR / f"trades_{pair}_{tf}"

    # Build the params.json expected by the strategy loader
    params_json = {
        "strategy_name": "MemeHedgeProxyMartingaleStrategy",
        "params": {
            "buy": {},
            "sell": {},
        },
    }
    # Classify each param into buy or sell space
    buy_params_def = {
        "buy_atr_pct_min", "buy_rng_low", "buy_rng_high",
        "hedge_ratio", "hedge_trigger_level",
        "buy_rsi_max", "dca_max_entries", "dca_step_pct", "dca_cooldown_min",
    }
    sell_params_def = {
        "sell_rsi_min", "take_profit_pct", "time_stop_candles",
    }
    for k, v in params.items():
        if k in buy_params_def:
            params_json["params"]["buy"][k] = v
        elif k in sell_params_def:
            params_json["params"]["sell"][k] = v

    # Write to strategy-name.json (freqtrade auto-loads from here)
    strategy_param_file = PROJECT_ROOT / "user_data" / "strategies" / "Meme_对冲代理_马丁.json"
    backup = None
    if strategy_param_file.exists():
        backup = strategy_param_file.read_text(encoding="utf-8")

    try:
        strategy_param_file.parent.mkdir(parents=True, exist_ok=True)
        strategy_param_file.write_text(json.dumps(params_json, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    except Exception:
        pass  # Continue anyway; params may not auto-load but backtest runs

    cmd = [
        PYTHON_EXE, *FREQTRADE_MODULE, "backtesting",
        "--config", str(config_path),
        "--strategy", "MemeHedgeProxyMartingaleStrategy",
        "--strategy-path", STRATEGY_PATH,
        "--timeframe", tf,
        "--timerange", TIMERANGE,
        "-p", pair_str,
        "--export", "trades",
        "--export-directory", str(TRADE_EXPORT_DIR),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=cwd, timeout=900,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        elapsed = time.time() - t0
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")

        if proc.returncode != 0:
            return {"status": "FAIL", "error": _extract_error(output),
                    "elapsed_sec": elapsed}

        result = parse_backtest_output(output)

        # Find exported trade file
        actual_trade_file = None
        for f in TRADE_EXPORT_DIR.glob(f"*{pair.replace('/', '_')}*{tf}*.json"):
            actual_trade_file = f
            break
        if actual_trade_file:
            trade_analysis = parse_trade_json(actual_trade_file)
            result.update(trade_analysis)

        result["status"] = "OK"
        result["elapsed_sec"] = elapsed
        return result

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": "900s timeout",
                "elapsed_sec": time.time() - t0}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)[:300],
                "elapsed_sec": time.time() - t0}
    finally:
        # Restore or clean up param file
        try:
            if backup is not None:
                strategy_param_file.write_text(backup, encoding="utf-8")
            elif strategy_param_file.exists():
                strategy_param_file.unlink()
        except Exception:
            pass


def _extract_error(text: str) -> str:
    """Extract the most relevant error message."""
    for line in text.split("\n"):
        if "ERROR" in line or "CRITICAL" in line:
            return line[:200]
    return text[-200:]


# ── Report Generation ──────────────────────────────────────────────────

def generate_html_report(all_results: list) -> str:
    """Generate comprehensive HTML report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build summary table rows
    summary_rows = []
    for r in all_results:
        hp = r.get("hyperopt", {})
        bt = r.get("backtest", {})

        summary_rows.append({
            "coin": r["coin"],
            "group": coin_group(r["coin"]),
            "tf": r["tf"],
            "hp_status": hp.get("status", "N/A"),
            "bt_status": bt.get("status", "N/A"),
            "trades": bt.get("trades", "N/A"),
            "profit": bt.get("profit_total_pct", "N/A"),
            "winrate": bt.get("winrate", "N/A"),
            "max_dd": bt.get("max_drawdown", "N/A"),
            "profit_factor": bt.get("profit_factor", "N/A"),
            "sharpe": bt.get("sharpe", "N/A"),
            "total_trades": bt.get("total_trades", 0),
            "dca_max_level": bt.get("dca_max_level", 0),
            "dca_trades": bt.get("dca_trades", 0),
            "liquidations": bt.get("liquidations", 0),
            "max_dd_trade": bt.get("max_drawdown_trade", 0),
            "hp_elapsed": hp.get("elapsed_sec", 0),
            "bt_elapsed": bt.get("elapsed_sec", 0),
            "best_params": hp.get("best_params", {}),
            "profit_curve": bt.get("profit_curve", []),
            "drawdown_curve": bt.get("drawdown_curve", []),
        })

    # Build best params section
    params_rows = []
    for r in all_results:
        hp = r.get("hyperopt", {})
        params = hp.get("best_params", {})
        if not params:
            continue
        row_data = {
            "coin": r["coin"],
            "tf": r["tf"],
            "group": coin_group(r["coin"]),
        }
        row_data.update(params)
        params_rows.append(row_data)

    # Generate HTML
    html = _build_html(now, summary_rows, params_rows, all_results)
    return html


def _build_html(now: str, summary_rows: list, params_rows: list, all_results: list) -> str:
    """Build the full HTML report."""
    # Generate summary table
    summary_tbody = ""
    for r in summary_rows:
        profit_val = r.get("profit", "N/A")
        try:
            pf = float(profit_val)
            profit_cls = "positive" if pf > 0 else ("negative" if pf < 0 else "")
        except (ValueError, TypeError):
            profit_cls = ""
        summary_tbody += f"""
        <tr>
            <td>{r['coin']}</td>
            <td>{r['group']}</td>
            <td>{r['tf']}</td>
            <td><span class="badge badge-{r['hp_status'].lower()}">{r['hp_status']}</span></td>
            <td><span class="badge badge-{r['bt_status'].lower()}">{r['bt_status']}</span></td>
            <td>{r['trades']}</td>
            <td class="{profit_cls}">{r['profit']}%</td>
            <td>{r['winrate']}%</td>
            <td>{r['max_dd']}%</td>
            <td>{r['profit_factor']}</td>
            <td>{r['total_trades']}</td>
            <td>{r['dca_max_level']}</td>
            <td>{r['liquidations']}</td>
            <td>{r['hp_elapsed']:.0f}s + {r['bt_elapsed']:.0f}s</td>
        </tr>"""

    # Generate params table
    params_thead = ""
    params_tbody = ""
    if params_rows:
        param_keys = [k for k in params_rows[0].keys() if k not in ("coin", "tf", "group")]
        params_thead = "".join(f"<th>{k}</th>" for k in param_keys)
        for r in params_rows:
            cells = "".join(f"<td>{r.get(k, '')}</td>" for k in param_keys)
            params_tbody += f"""
            <tr>
                <td>{r['coin']}</td>
                <td>{r['tf']}</td>
                {cells}
            </tr>"""

    # Group summary
    group_summary = ""
    for grp in COIN_GROUPS:
        grp_rows = [r for r in summary_rows if r["group"] == grp]
        group_summary += f"""
        <tr>
            <td><strong>{grp}</strong></td>
            <td>{len(grp_rows)}</td>
            <td>{sum(1 for r in grp_rows if r['hp_status'] == 'OK')}</td>
            <td>{sum(1 for r in grp_rows if r['bt_status'] == 'OK')}</td>
        </tr>"""

    # Profit by timeframe
    profit_tf = {"1m": [], "5m": [], "15m": []}
    for r in summary_rows:
        p = r.get("profit", "N/A")
        try:
            profit_tf[r["tf"]].append(float(p))
        except (ValueError, TypeError):
            pass
    profit_tf_html = ""
    for tf_name, vals in profit_tf.items():
        if vals:
            avg_p = sum(vals) / len(vals)
            profit_tf_html += f"<tr><td>{tf_name}</td><td>{len(vals)}</td><td>{avg_p:.2f}%</td><td>{min(vals):.2f}%</td><td>{max(vals):.2f}%</td></tr>"

    # Liquidation summary
    liq_total = sum(r.get("liquidations", 0) for r in summary_rows)
    liq_coins = [r for r in summary_rows if r.get("liquidations", 0) > 0]

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meme_对冲代理_马丁 Hyperopt 优化报告</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #1a1d23; color: #e0e0e0; padding: 20px; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ color: #00d4aa; margin-bottom: 8px; font-size: 24px; }}
    h2 {{ color: #00d4aa; margin: 24px 0 12px; font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
    .meta {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
    .card {{ background: #252830; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .stat {{ background: #2a2d35; border-radius: 6px; padding: 12px; text-align: center; }}
    .stat .value {{ font-size: 22px; font-weight: bold; color: #00d4aa; }}
    .stat .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 16px; }}
    th {{ background: #2a2d35; padding: 8px 10px; text-align: left; font-weight: 600; color: #aaa; white-space: nowrap; position: sticky; top: 0; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #333; }}
    tr:hover {{ background: #2a2d35; }}
    .positive {{ color: #ef5350; }}
    .negative {{ color: #26a69a; }}
    .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
    .badge-ok {{ background: #1b5e20; color: #66bb6a; }}
    .badge-fail {{ background: #b71c1c; color: #ef9a9a; }}
    .badge-timeout {{ background: #e65100; color: #ffcc80; }}
    .badge-error {{ background: #b71c1c; color: #ef9a9a; }}
    .badge-no_data {{ background: #555; color: #ccc; }}
    .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; }}
    .tab {{ padding: 8px 16px; border-radius: 6px 6px 0 0; cursor: pointer; background: #2a2d35; color: #888; border: none; font-size: 13px; }}
    .tab.active {{ background: #00d4aa; color: #1a1d23; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .warn-box {{ background: #3e2723; border-left: 3px solid #ff7043; padding: 12px; margin: 12px 0; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">

<h1>Meme_对冲代理_马丁 — Hyperopt 参数优化报告</h1>
<p class="meta">
    策略: MemeHedgeProxyMartingaleStrategy |
    数据: Gate Spot (feather) |
    时间: {TIMERANGE} |
    损失函数: {HYPEROPT_LOSS} |
    Epochs: {DEFAULT_EPOCHS} |
    生成时间: {now}
</p>

<div class="stats">
    <div class="stat"><div class="value">{len(all_results)}</div><div class="label">总运行数</div></div>
    <div class="stat"><div class="value">{sum(1 for r in summary_rows if r['hp_status'] == 'OK')}</div><div class="label">Hyperopt 成功</div></div>
    <div class="stat"><div class="value">{sum(1 for r in summary_rows if r['bt_status'] == 'OK')}</div><div class="label">Backtest 成功</div></div>
    <div class="stat"><div class="value">{sum(r.get('total_trades', 0) for r in summary_rows)}</div><div class="label">全部交易</div></div>
    <div class="stat"><div class="value">{liq_total}</div><div class="label">爆仓事件</div></div>
    <div class="stat"><div class="value">{sum(r.get('hp_elapsed', 0) + r.get('bt_elapsed', 0) for r in summary_rows) / 60:.0f} min</div><div class="label">总用时</div></div>
</div>

<div class="card">
    <h2>币种分组统计</h2>
    <table>
        <tr><th>分组</th><th>运行数</th><th>Hyperopt 成功</th><th>Backtest 成功</th></tr>
        {group_summary}
    </table>
</div>

<div class="tabs">
    <button class="tab active" onclick="showTab('summary')">绩效总览</button>
    <button class="tab" onclick="showTab('params')">最佳参数</button>
    <button class="tab" onclick="showTab('timeframe')">周期对比</button>
    <button class="tab" onclick="showTab('risk')">风险分析</button>
    <button class="tab" onclick="showTab('curves')">盈亏曲线</button>
</div>

<div id="tab-summary" class="tab-content active">
    <h2>全币种全周期绩效明细</h2>
    <div style="overflow-x: auto;">
    <table>
        <thead>
            <tr>
                <th>币种</th><th>分组</th><th>周期</th><th>HP状态</th><th>BT状态</th>
                <th>交易数</th><th>收益率</th><th>胜率</th><th>最大回撤</th>
                <th>盈亏比</th><th>总交易</th><th>最大DCA级</th><th>爆仓</th><th>耗时</th>
            </tr>
        </thead>
        <tbody>{summary_tbody}</tbody>
    </table>
    </div>
</div>

<div id="tab-params" class="tab-content">
    <h2>各币种最佳参数</h2>
    <div style="overflow-x: auto;">
    <table>
        <thead>
            <tr>
                <th>币种</th><th>周期</th>
                {params_thead}
            </tr>
        </thead>
        <tbody>{params_tbody}</tbody>
    </table>
    </div>
</div>

<div id="tab-timeframe" class="tab-content">
    <h2>K 线周期绩效对比</h2>
    <table>
        <tr><th>周期</th><th>样本数</th><th>平均收益率</th><th>最低收益率</th><th>最高收益率</th></tr>
        {profit_tf_html}
    </table>
</div>

<div id="tab-risk" class="tab-content">
    <h2>爆仓风险与最大浮亏分析</h2>
    <div class="warn-box">
        <strong>⚠️ 高风险提示：</strong>以下为 Backtest 模拟结果。实盘中市场冲击成本、滑点、极端行情可能导致更严重亏损。
        马丁策略本质是负偏度策略，建议始终保留至少 30% 保证金作为安全垫。
    </div>

    <table>
        <tr><th>币种</th><th>周期</th><th>最大浮亏($)</th><th>最大DCA级数</th><th>DCA触发次数</th><th>爆仓次数</th><th>连续亏损</th></tr>
"""
    for r in summary_rows:
        html += f"""
        <tr>
            <td>{r['coin']}</td>
            <td>{r['tf']}</td>
            <td class="negative">-{r.get('max_dd_trade', 0):.2f}</td>
            <td>{r.get('dca_max_level', 0)}</td>
            <td>{r.get('dca_trades', 0)}</td>
            <td>{r.get('liquidations', 0)}</td>
            <td>{r.get('max_consecutive_losses', 'N/A')}</td>
        </tr>"""

    html += """
    </table>
</div>

<div id="tab-curves" class="tab-content">
    <h2>盈亏曲线 / 最大浮亏曲线</h2>
    <div class="card">
        <canvas id="profitChart" height="400"></canvas>
    </div>
    <div class="card">
        <canvas id="drawdownChart" height="400"></canvas>
    </div>
    <div style="margin-top: 8px; font-size: 12px; color: #888;">
        <strong>图例说明：</strong>盈亏曲线(上)展示累计盈亏变化；浮亏曲线(下)展示最大回撤幅度。
    </div>
</div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
function showTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    var tabBtn = document.querySelector('.tab[onclick*="' + name + '"]');
    if (tabBtn) tabBtn.classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
    if (name === 'curves') { renderCurves(); }
}

// Prepare curve data from results
var curveData = """ + json.dumps([
    {"coin": r["coin"], "tf": r["tf"], "profit": r.get("profit_curve", []), "drawdown": r.get("drawdown_curve", [])}
    for r in summary_rows if r.get("profit_curve") and len(r["profit_curve"]) > 3
], ensure_ascii=False) + """;

var profitColors = ['#00d4aa','#ef5350','#42a5f5','#ff9800','#ab47bc','#66bb6a','#ec407a','#26c6da','#ffee58'];
var randColor = function(i) { return profitColors[i % profitColors.length]; };

function renderCurves() {
    if (window._curvesRendered) return;
    window._curvesRendered = true;

    var profitCtx = document.getElementById('profitChart');
    var ddCtx = document.getElementById('drawdownChart');

    var profitDatasets = [];
    var ddDatasets = [];
    for (var i = 0; i < curveData.length; i++) {
        var d = curveData[i];
        var color = randColor(i);
        var label = d.coin + ' ' + d.tf;
        var profits = d.profit || [];
        var dds = d.drawdown || [];
        profitDatasets.push({
            label: label, data: profits.map(function(p) { return {x: p.x, y: p.y}; }),
            borderColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false
        });
        ddDatasets.push({
            label: label, data: dds.map(function(p) { return {x: p.x, y: p.y}; }),
            borderColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false
        });
    }

    new Chart(profitCtx, {
        type: 'line',
        data: { datasets: profitDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#aaa', boxWidth: 12, font: {size: 10} } },
                title: { display: true, text: '累计盈亏曲线', color: '#00d4aa', font: {size: 14} }
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: '交易序号', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#2a2d35' } },
                y: { title: { display: true, text: '累计盈亏 ($)', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#2a2d35' } }
            }
        }
    });

    new Chart(ddCtx, {
        type: 'line',
        data: { datasets: ddDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#aaa', boxWidth: 12, font: {size: 10} } },
                title: { display: true, text: '最大浮亏曲线', color: '#00d4aa', font: {size: 14} }
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: '交易序号', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#2a2d35' } },
                y: { title: { display: true, text: '浮亏 ($)', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#2a2d35' } }
            }
        }
    });
}
</script>
</body>
</html>"""

    return html


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Meme_对冲代理_马丁 Hyperopt")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Hyperopt epochs per run")
    parser.add_argument("--only-coin", type=str, help="Run only specific coin")
    parser.add_argument("--only-tf", type=str, help="Run only specific timeframe")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest verification")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, don't execute")
    parser.add_argument("--proxy-host", type=str, default="127.0.0.1:7890",
                        help="Proxy host:port for Gate API (required for exchange init)")
    args = parser.parse_args()

    coins = [args.only_coin] if args.only_coin else ALL_COINS
    tfs = [args.only_tf] if args.only_tf else TIMEFRAMES

    # Validate
    for c in coins:
        if c not in ALL_COINS:
            print(f"ERROR: Unknown coin '{c}'. Valid: {ALL_COINS}")
            return
    for t in tfs:
        if t not in TIMEFRAMES:
            print(f"ERROR: Unknown timeframe '{t}'. Valid: {TIMEFRAMES}")
            return

    # Create output directories
    for d in [OUTPUT_DIR, REPORT_DIR, TRADE_EXPORT_DIR, PARAMS_DIR, LOG_DIR, CONFIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight check ──
    if not args.dry_run:
        import socket
        proxy_host, proxy_port = args.proxy_host.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((proxy_host, int(proxy_port)))
        sock.close()
        if result != 0:
            print(f"\n{'='*70}")
            print(f"  ⚠️  WARNING: Proxy {args.proxy_host} is NOT reachable!")
            print(f"  freqtrade hyperopt requires exchange connectivity to initialize.")
            print(f"  Please start your proxy (Clash/V2Ray/etc.) before running this script.")
            print(f"  Or use --dry-run to preview commands without executing.")
            print(f"{'='*70}\n")
            if input("Continue anyway? (y/N): ").strip().lower() != 'y':
                return

    cwd = str(PROJECT_ROOT)
    all_results = []
    total = len(coins) * len(tfs)

    print(f"{'='*70}")
    print(f"  Meme_对冲代理_马丁 Hyperopt — 全币种全周期优化")
    print(f"  币种: {len(coins)} | 周期: {len(tfs)} | 总计: {total} | Epochs: {args.epochs}")
    print(f"  时间范围: {TIMERANGE} | 损失函数: {HYPEROPT_LOSS}")
    print(f"{'='*70}\n")

    idx = 0
    for coin in coins:
        for tf_name in tfs:
            idx += 1
            grp = coin_group(coin)
            print(f"\n[{idx}/{total}] {grp} | {coin} | {tf_name}")
            print(f"{'─'*50}")

            # ── Phase 1: Hyperopt ──
            print(f"  [Hyperopt] Starting ({args.epochs} epochs)...", end=" ", flush=True)

            if args.dry_run:
                print("SKIP (dry-run)")
                hp_result = {"status": "DRY", "best_params": {}, "elapsed_sec": 0}
            else:
                hp_result = run_hyperopt(coin, tf_name, args.epochs, cwd, args.proxy_host)
                if hp_result["status"] == "OK":
                    print(f"DONE ({hp_result['elapsed_sec']:.0f}s)")
                    print(f"    Best loss: {hp_result.get('best_loss', 'N/A')}")
                    print(f"    Params: {hp_result.get('best_params', {})}")
                else:
                    print(f"{hp_result['status']}: {hp_result.get('error', 'unknown')}")

            # Save params
            if hp_result.get("best_params"):
                params_file = PARAMS_DIR / f"params_{coin}_{tf_name}.json"
                with open(params_file, "w", encoding="utf-8") as f:
                    json.dump({"coin": coin, "tf": tf_name, "group": grp,
                               "best_loss": hp_result.get("best_loss"),
                               "best_params": hp_result["best_params"],
                               "elapsed_sec": hp_result.get("elapsed_sec")}, f, indent=2)

            # ── Phase 2: Backtest Verification ──
            bt_result = {"status": "SKIPPED"}
            if not args.skip_backtest and hp_result["status"] == "OK" and hp_result.get("best_params"):
                print(f"  [Backtest] Verifying best params...", end=" ", flush=True)
                bt_result = run_backtest(coin, tf_name, hp_result["best_params"], cwd, args.proxy_host)
                if bt_result["status"] == "OK":
                    print(f"DONE ({bt_result['elapsed_sec']:.0f}s)")
                    print(f"    Trades: {bt_result.get('trades', 'N/A')} | "
                          f"Profit: {bt_result.get('profit_total_pct', 'N/A')}% | "
                          f"DD: {bt_result.get('max_drawdown', 'N/A')}% | "
                          f"DCA: L{bt_result.get('dca_max_level', 0)} "
                          f"({bt_result.get('dca_trades', 0)}x)")
                else:
                    print(f"{bt_result['status']}: {bt_result.get('error', 'unknown')}")
            elif args.skip_backtest:
                print(f"  [Backtest] Skipped")

            all_results.append({
                "coin": coin, "tf": tf_name, "group": grp,
                "hyperopt": hp_result, "backtest": bt_result,
            })

            # Save intermediate results
            summary_file = OUTPUT_DIR / "hyperopt_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # ── Generate Report ──
    print(f"\n{'='*70}")
    print("  Generating HTML report...")

    html = generate_html_report(all_results)
    report_path = REPORT_DIR / "hyperopt_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Print final summary
    hp_ok = sum(1 for r in all_results if r["hyperopt"].get("status") == "OK")
    bt_ok = sum(1 for r in all_results if r["backtest"].get("status") == "OK")
    total_time = sum(r["hyperopt"].get("elapsed_sec", 0) + r["backtest"].get("elapsed_sec", 0) for r in all_results)

    print(f"  Hyperopt OK: {hp_ok}/{len(all_results)}")
    print(f"  Backtest OK: {bt_ok}/{len(all_results)}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
