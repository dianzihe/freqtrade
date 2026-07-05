#!/usr/bin/env python3
"""
Meme_限制定投_马丁 全币种全周期批量回测脚本
=================================================
8 币种 × 3 周期 = 24 个回测
  - 稳定主流币: BTC, ETH, SOL, XRP
  - 中端中型币: XCN
  - 高波动妖币: H, VELVET, BEAT, COAI

输出目录: deliverables/backtest_meme_limited_YYYYMMDD_HHMMSS/
  - results/<class>_<pair>_<tf>.json  — 回测原始导出
  - metrics/<class>_<pair>_<tf>_metrics.json  — 关键指标摘要
  - dca_logs/<class>_<pair>_<tf>_dca.csv  — 加仓触发记录
  - html/index.html  — 汇总 HTML 报告
  - run.log  — 运行日志

用法:
  .venv/Scripts/python backtest_meme_limited_all.py
  # 仅跑某周期: --timeframes 1m 5m
  # 仅跑某币种: --pairs BTC ETH
  # 指定回测区间: --timerange 20260601-20260630
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"F:\source\freqtrade")
PYTHON_BIN = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
FREQTRADE_MODULE = "-m freqtrade"
CONFIG_MAIN = PROJECT_DIR / "user_data" / "config" / "config.json"
CONFIG_GATE = PROJECT_DIR / "user_data" / "config" / "config_gate_backtest.json"
DATA_DIR = PROJECT_DIR / "user_data" / "data" / "gate"
STRATEGY_PATH = PROJECT_DIR / "user_data" / "strategies"

# 若 config_gate_backtest.json 不存在则只用主配置
CONFIG_FILES = [str(CONFIG_MAIN)]
if CONFIG_GATE.exists():
    CONFIG_FILES.append(str(CONFIG_GATE))

# ── 策略配置 ────────────────────────────────────────────────────────────────
STRATEGY_NAME = "Meme_限制定投_马丁"
STRATEGY_CLASS = "MemeLimitedMartingaleSpotStrategy"
STRATEGY_FILE = "Meme_限制定投_马丁_spot.py"

# ── 币种列表 ─────────────────────────────────────────────────────────────────
COINS_ALL = {
    "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中端中型币": ["XCN/USDT"],
    "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

# ── 默认回测区间 ──────────────────────────────────────────────────────────────
DEFAULT_TIMERANGE = "20260608-20260627"  # 覆盖 1m/5m/15m 有足够数据


def log(msg: str, log_file=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def safe_pair_name(pair: str) -> str:
    return pair.replace("/", "_")


def run_single_backtest(
    pair: str,
    timeframe: str,
    timerange: str,
    out_dir: Path,
    log_file: Path,
) -> dict:
    """
    运行单个回测，返回结果字典。
    """
    safe_pair = safe_pair_name(pair)
    base_name = f"{STRATEGY_CLASS}_{safe_pair}_{timeframe}"
    export_base = out_dir / "results" / base_name

    cmd = [
        str(PYTHON_BIN),
        "-
m",
        "freqtrade",
        "backtesting",
        *[item for cf in CONFIG_FILES for item in ("--config", cf)],
        "--strategy", STRATEGY_CLASS,
        "--strategy-path", str(STRATEGY_PATH),
        "--timeframe", timeframe,
        "-p", pair,
        "--datadir", str(DATA_DIR),
        "--timerange", timerange,
        "--export", "signals",
        "--backtest-directory", str(out_dir / "results"),
        "--notes", base_name,
    ]

    log(f"  ▶ 启动: {pair} {timeframe}  timerange={timerange}", log_file)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )
        elapsed = time.time() - start
        success = result.returncode == 0
        log(f"   {'✓ 完成' if success else '✗ 失败'} ({elapsed:.1f}s)  rc={result.returncode}", log_file)

        if not success:
            # 记录错误信息到日志
            err_tail = (result.stderr or "")[-2000:]
            out_tail = (result.stdout or "")[-2000:]
            log(f"   STDERR:\n{err_tail}", log_file)
            log(f"   STDOUT:\n{out_tail}", log_file)

        return {
            "pair": pair,
            "timeframe": timeframe,
            "success": success,
            "elapsed_sec": round(elapsed, 1),
            "returncode": result.returncode,
            "stdout": result.stdout[-5000:] if result.stdout else "",
            "stderr": result.stderr[-3000:] if result.stderr else "",
            "export_base": str(export_base),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log(f"   ✗ 超时 ({elapsed:.1f}s)", log_file)
        return {
            "pair": pair,
            "timeframe": timeframe,
            "success": False,
            "elapsed_sec": round(elapsed, 1),
            "returncode": -1,
            "error": "TIMEOUT",
            "stdout": "",
            "stderr": "TIMEOUT",
            "export_base": str(export_base),
        }
    except Exception as e:
        elapsed = time.time() - start
        log(f"   ✗ 异常: {e}", log_file)
        return {
            "pair": pair,
            "timeframe": timeframe,
            "success": False,
            "elapsed_sec": round(elapsed, 1),
            "returncode": -2,
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "export_base": str(export_base),
        }


def parse_backtest_json(export_base: str) -> Optional[dict]:
    """
    读取 freqtrade 导出的 backtest-result.json。
    使用 --backtest-directory + --notes 时，文件名为 <notes>-backtest-result.json
    保存在 <backtest-directory>/ 下。
    """
    # export_base 此时是 base_name (不含路径)，实际文件在 results 目录
    results_dir = Path(export_base).parent if "/" in export_base or "\\" in export_base else Path(".")
    # 兼容两种传参方式
    for candidate in [
        Path(export_base + "-backtest-result.json"),
        Path(export_base).parent / f"{Path(export_base).name}-backtest-result.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  读取 {candidate} 失败: {e}")
                return None
    return None


def parse_trades_json(export_base: str) -> Optional[list]:
    """读取 freqtrade 导出的 trades.json"""
    trades_path = Path(export_base + "-trades.json")
    if trades_path.exists():
        try:
            with open(trades_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  读取 {trades_path} 失败: {e}")
    return None


def extract_metrics(backtest_result: dict, pair: str, timeframe: str) -> dict:
    """从回测结果中提取关键绩效指标"""
    metrics = {
        "pair": pair,
        "timeframe": timeframe,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "total_profit_pct": 0.0,
        "total_profit_abs": 0.0,
        "avg_profit_pct": 0.0,
        "median_profit_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_abs": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "avg_duration": "",
        "best_trade_pct": 0.0,
        "worst_trade_pct": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "total_open_trades": 0,
        "max_open_trades": 0,
    }

    if not backtest_result:
        return metrics

    # 尝试兼容不同版本的 freqtrade 输出格式
    try:
        # 新版本: results 是 list of dict
        results = backtest_result.get("results", [])
        if not results and "strategy" in backtest_result:
            # 另一种格式
            strat = backtest_result["strategy"]
            if isinstance(strat, dict):
                results = strat.get("results", [])

        if not results:
            return metrics

        total = len(results)
        wins = sum(1 for r in results if r.get("profit_ratio", 0) > 0)
        losses = total - wins

        profits = [r.get("profit_ratio", 0) for r in results]
        profit_abs = [r.get("profit_abs", 0) for r in results]

        win_profits = [p for p in profits if p > 0]
        loss_profits = [p for p in profits if p < 0]

        metrics.update({
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(wins / total * 100, 2) if total else 0,
            "total_profit_pct": round(sum(profits) * 100, 4),
            "avg_profit_pct": round(sum(profits) / total * 100, 4) if total else 0,
            "median_profit_pct": round(float(__import__('statistics').median(profits)) * 100, 4) if profits else 0,
            "best_trade_pct": round(max(profits) * 100, 4) if profits else 0,
            "worst_trade_pct": round(min(profits) * 100, 4) if profits else 0,
        })

        # 最大回撤（从 cumulative 曲线计算）
        cumulative = [0]
        for p in profits:
            cumulative.append(cumulative[-1] + p)
        cumulative = cumulative[1:]  # 去掉初始 0
        peak = 0
        max_dd = 0
        for val in cumulative:
            peak = max(peak, val)
            dd = peak - val
            max_dd = max(max_dd, dd)
        metrics["max_drawdown_pct"] = round(max_dd * 100, 4)

        # 盈亏因子
        avg_win = sum(win_profits) / len(win_profits) if win_profits else 0
        avg_loss = abs(sum(loss_profits) / len(loss_profits)) if loss_profits else 0
        if avg_loss > 0:
            metrics["profit_factor"] = round(avg_win / avg_loss, 4)

    except Exception as e:
        print(f"  提取指标异常: {e}")

    return metrics


def extract_dca_records(trades: list, out_csv: Path, log_file: Path) -> int:
    """
    从 trades 数据中提取加仓触发记录，输出 CSV。
    返回加仓次数。
    """
    dca_rows = []
    for trade in (trades or []):
        pair = trade.get("pair", "")
        open_ts = trade.get("open_timestamp", 0)
        # freqtrade 导出格式中，加仓记录在 events 或 orders 字段
        events = trade.get("events", [])
        orders = trade.get("orders", [])
        # 统计入场订单（第一个是开仓，其余是加仓）
        entry_orders = [o for o in orders if o.get("role") in (None, "entry", "initial")]
        dca_orders = [o for o in orders if o.get("role") == "dca" or o.get("is_dca", False)]
        # 备用：通过 order_id 数量判断
        all_orders = orders if orders else []
        n_entries = len(all_orders)
        if n_entries > 1:
            for i, o in enumerate(all_orders):
                dca_rows.append({
                    "pair": pair,
                    "trade_id": trade.get("trade_id", ""),
                    "order_seq": i,
                    "is_dca": i >= 1,
                    "price": o.get("price", 0),
                    "amount": o.get("amount", 0),
                    "timestamp": o.get("timestamp", open_ts + i * 3600000),
                    "profit_ratio_before": "",  # 需要事件序列才能填
                })
    if dca_rows:
        import csv
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dca_rows[0].keys())
            writer.writeheader()
            writer.writerows(dca_rows)
    return len(dca_rows)


def count_consecutive(numbers: list) -> tuple:
    """统计最大连续盈利/亏损次数"""
    max_win = max_loss = cur_win = cur_loss = 0
    for n in numbers:
        if n > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        elif n < 0:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
        else:
            cur_win = cur_loss = 0
    return max_win, max_loss


def build_html_report(all_metrics: list, out_dir: Path, timerange: str):
    """生成汇总 HTML 报告"""
    html_path = out_dir / "html" / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # 按类别分组
    cats = {
        "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
        "中端中型币": ["XCN/USDT"],
        "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
    }

    rows = []
    for cat, pairs in cats.items():
        for pair in pairs:
            for tf in ["1m", "5m", "15m"]:
                m = next((x for x in all_metrics if x["pair"] == pair and x["timeframe"] == tf), None)
                if m:
                    status = "✓" if m.get("success", True) else "✗"
                    rows.append(f"""
      <tr>
        <td>{cat}</td>
        <td>{pair}</td>
        <td>{tf}</td>
        <td>{status}</td>
        <td>{m['total_trades']}</td>
        <td>{m['win_rate_pct']}%</td>
        <td>{m['total_profit_pct']}%</td>
        <td>{m['avg_profit_pct']}%</td>
        <td>{m['max_drawdown_pct']}%</td>
        <td>{m['profit_factor']}</td>
        <td>{m.get('sharpe_ratio', 0)}</td>
      </tr>""")
                else:
                    rows.append(f"""
      <tr>
        <td>{cat}</td>
        <td>{pair}</td>
        <td>{tf}</td>
        <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
      </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Meme_限制定投_马丁 回测报告</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; font-size: 13px; }}
  th {{ background: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .cat-header {{ background: #34495e; color: white; font-weight: bold; }}
  .positive {{ color: green; }}
  .negative {{ color: red; }}
  .summary {{ margin: 20px 0; padding: 15px; background: white; border-radius: 8px; }}
</style>
</head>
<body>
<h1>📊 Meme_限制定投_马丁 全币种全周期回测报告</h1>
<div class="summary">
  <p>策略: {STRATEGY_NAME} ({STRATEGY_CLASS})</p>
  <p>回测区间: {timerange}</p>
  <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  <p>共计: {len(all_metrics)} / 24 个回测完成</p>
</div>
<table>
  <thead>
    <tr>
      <th>类别</th><th>币种</th><th>周期</th><th>状态</th>
      <th>交易数</th><th>胜率</th><th>总收益%</th><th>均收益%</th>
      <th>最大回撤%</th><th>盈亏因子</th><th>Sharpe</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
<p style="margin-top:20px; color:#888; font-size:12px;">
  详细数据见 results/ 目录，DCA 记录见 dca_logs/ 目录。
</p>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Meme_限制定投_马丁 批量回测")
    parser.add_argument("--timeframes", nargs="*", default=["1m", "5m", "15m"],
                        help="周期列表 (默认: 1m 5m 15m)")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="币种列表 (默认: 全部8个)")
    parser.add_argument("--timerange", default=DEFAULT_TIMERANGE,
                        help=f"回测区间 (默认: {DEFAULT_TIMERANGE})")
    parser.add_argument("--skip-done", action="store_true",
                        help="跳过已有结果的回测")
    args = parser.parse_args()

    # 收集所有币种
    all_pairs = []
    for plist in COINS_ALL.values():
        all_pairs.extend(plist)
    if args.pairs:
        # 支持 BTC 自动补全为 BTC/USDT
        mapped = []
        for p in args.pairs:
            if "/" not in p:
                p = p.upper() + "/USDT"
            mapped.append(p)
        all_pairs = [p for p in all_pairs if p in mapped]

    timeframes = args.timeframes
    timerange = args.timerange

    # 输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_DIR / "deliverables" / f"backtest_meme_limited_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    (out_dir / "dca_logs").mkdir(exist_ok=True)
    (out_dir / "html").mkdir(exist_ok=True)

    log_file = out_dir / "run.log"

    log("=" * 60, log_file)
    log(f"Meme_限制定投_马丁 批量回测", log_file)
    log(f"  策略类: {STRATEGY_CLASS}", log_file)
    log(f"  币种: {all_pairs}", log_file)
    log(f"  周期: {timeframes}", log_file)
    log(f"  区间: {timerange}", log_file)
    log(f"  输出: {out_dir}", log_file)
    log("=" * 60, log_file)

    # 检查 freqtrade 可执行文件
    if not FREQTRADE_BIN.exists():
        log(f"✗ 找不到 freqtrade: {FREQTRADE_BIN}", log_file)
        sys.exit(1)

    all_metrics = []
    total = len(all_pairs) * len(timeframes)
    count = 0

    for pair in all_pairs:
        for tf in timeframes:
            count += 1
            safe_pair = safe_pair_name(pair)
            metrics_file = out_dir / "metrics" / f"{STRATEGY_CLASS}_{safe_pair}_{tf}_metrics.json"

            # 跳过已完成
            if args.skip_done and metrics_file.exists():
                log(f"[{count}/{total}] 跳过已完成: {pair} {tf}", log_file)
                # 读取已有指标
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        all_metrics.append(json.load(f))
                except Exception:
                    pass
                continue

            log(f"[{count}/{total}] 回测中: {pair} {tf}", log_file)

            result = run_single_backtest(pair, tf, timerange, out_dir, log_file)

            if result["success"]:
                # 解析回测结果
                backtest_data = parse_backtest_json(result["export_base"])
                trades_data = parse_trades_json(result["export_base"])

                metrics = extract_metrics(backtest_data, pair, tf)
                metrics["success"] = True
                metrics["elapsed_sec"] = result["elapsed_sec"]

                # 提取 DCA 记录
                if trades_data:
                    dca_csv = out_dir / "dca_logs" / f"{STRATEGY_CLASS}_{safe_pair}_{tf}_dca.csv"
                    n_dca = extract_dca_records(trades_data, dca_csv, log_file)
                    metrics["dca_count"] = n_dca
                else:
                    metrics["dca_count"] = 0

                # 保存指标
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                all_metrics.append(metrics)
                log(f"  指标: 交易={metrics['total_trades']} 胜率={metrics['win_rate_pct']}% "
                    f"收益={metrics['total_profit_pct']}% 回撤={metrics['max_drawdown_pct']}%", log_file)
            else:
                metrics = {
                    "pair": pair,
                    "timeframe": tf,
                    "success": False,
                    "error": result.get("error", f"rc={result['returncode']}"),
                    "total_trades": 0,
                }
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                all_metrics.append(metrics)

            # 每个回测之间稍作休息，避免系统资源耗尽
            time.sleep(2)

    # 生成汇总报告
    log("生成 HTML 汇总报告...", log_file)
    html_path = build_html_report(all_metrics, out_dir, timerange)
    log(f"HTML 报告: {html_path}", log_file)

    # 保存总览 JSON
    summary = {
        "strategy": STRATEGY_NAME,
        "class": STRATEGY_CLASS,
        "timerange": timerange,
        "total_backtests": total,
        "completed": sum(1 for m in all_metrics if m.get("success", False)),
        "metrics": all_metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("=" * 60, log_file)
    log(f"全部完成! 输出目录: {out_dir}", log_file)
    log(f"HTML 报告: {html_path}", log_file)
    log("=" * 60, log_file)
    print(f"\n✅ 完成! HTML 报告: {html_path}")


if __name__ == "__main__":
    main()
