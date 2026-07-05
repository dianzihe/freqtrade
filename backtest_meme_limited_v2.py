#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme_限制定投_马丁 全币种全周期批量回测
用法: .venv/Scripts/python backtest_meme_limited_v2.py
"""
import subprocess, json, time, os, sys, re
from datetime import datetime
from pathlib import Path

PROJECT     = Path(r"F:\source\freqtrade")
PYTHON_BIN  = PROJECT / ".venv" / "Scripts" / "python.exe"
CONFIGS     = [
    str(PROJECT / "user_data" / "config" / "config.json"),
    str(PROJECT / "user_data" / "config" / "config_gate_backtest.json"),
]
DATA_DIR    = str(PROJECT / "user_data" / "data" / "gate")
STRAT_CLASS = "MemeLimitedMartingaleSpotStrategy"
STRAT_PATH  = str(PROJECT / "user_data" / "strategies")

COINS = {
    "稳定主流币": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "中端中型币": ["XCN/USDT"],
    "高波动妖币": ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT"],
}

DEFAULT_TIMERANGE = "20260608-20260627"


def build_cmd(pair, timeframe):
    return [
        str(PYTHON_BIN), "-m", "freqtrade", "backtesting",
        *[x for c in CONFIGS for x in ("--config", c)],
        "--strategy", STRAT_CLASS,
        "--strategy-path", STRAT_PATH,
        "--timeframe", timeframe,
        "-p", pair,
        "--datadir", DATA_DIR,
    ]


def parse_metrics(stdout_text, pair, tf):
    """解析 freqtrade backtesting stdout 表格。"""
    lines = stdout_text.split("\n")
    m = {
        "pair": pair, "timeframe": tf,
        "total_trades": 0, "winning": 0, "losing": 0,
        "win_pct": 0.0, "tot_profit_pct": 0.0, "tot_profit_abs": 0.0,
        "avg_profit_pct": 0.0, "max_dd_pct": 0.0, "max_dd_abs": 0.0,
        "sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0,
        "calmar": 0.0, "sqn": 0.0, "cagr": 0.0,
        "exit_reasons": {}, "enter_tags": {}, "duration": "",
    }
    try:
        # ── BACKTESTING REPORT ─────────────────────────────────────
        in_sec = False
        for ln in lines:
            if "BACKTESTING REPORT" in ln:
                in_sec = True
                continue
            if in_sec and ("┴" in ln or "═══════════════════════════" in ln):
                break
            if in_sec and "│" in ln:
                if "TOTAL" in ln or "Pair" in ln:
                    continue
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 7:
                    m["total_trades"]   = int(cols[1])
                    m["avg_profit_pct"] = float(cols[2])
                    m["tot_profit_abs"] = float(cols[3])
                    m["tot_profit_pct"] = float(cols[4])
                    m["duration"]        = cols[5]
                    wi = cols[6].split()
                    if len(wi) >= 4:
                        m["winning"] = int(wi[0])
                        m["losing"]  = int(wi[2])
                        m["win_pct"]  = float(wi[3])
        # ── EXIT REASON STATS ─────────────────────────────────────
        in_sec = False
        for ln in lines:
            if "EXIT REASON STATS" in ln:
                in_sec = True
                continue
            if in_sec and ("┴" in ln):
                break
            if in_sec and "│" in ln and "Exit Reason" not in ln and "TOTAL" not in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 1:
                    m["exit_reasons"][cols[0]] = int(cols[1])
        # ── ENTER TAG STATS ──────────────────────────────────────
        in_sec = False
        for ln in lines:
            if "ENTER TAG STATS" in ln:
                in_sec = True
                continue
            if in_sec and ("┴" in ln):
                break
            if in_sec and "│" in ln and "Enter Tag" not in ln and "TOTAL" not in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 1:
                    m["enter_tags"][cols[0]] = int(cols[1])
        # ── SUMMARY METRICS ──────────────────────────────────────
        for ln in lines:
            s = ln.strip()
            if "│ Sharpe" in s and "closed" in s:
                m["sharpe"] = float(s.split("│")[-1].strip())
            elif "│ Sortino" in s and "closed" in s:
                m["sortino"] = float(s.split("│")[-1].strip())
            elif "│ Profit factor" in s:
                m["profit_factor"] = float(s.split("│")[-1].strip())
            elif "│ Calmar" in s and "Max open" not in s:
                m["calmar"] = float(s.split("│")[-1].strip())
            elif "│ SQN" in s:
                m["sqn"] = float(s.split("│")[-1].strip())
            elif "│ CAGR" in s:
                v = s.split("│")[-1].strip().rstrip("%")
                m["cagr"] = float(v)
            elif "│ Max drawdown" in s and "│" in s:
                cell = s.split("│")[-1].strip()
                pm = re.search(r"([\d.]+)%", cell)
                am = re.search(r"([\d.]+)\s+USDT", cell)
                if pm:
                    m["max_dd_pct"] = float(pm.group(1))
                if am:
                    m["max_dd_abs"] = float(am.group(1))
    except Exception as e:
        print(f"  解析异常: {e}", flush=True)
    return m


def run_one(pair, timeframe, out_dir, timerange):
    """运行单个回测，返回指标字典。"""
    safe = pair.replace("/", "_")
    name = f"{STRAT_CLASS}_{safe}_{timeframe}"
    cmd  = build_cmd(pair, timeframe)
    cmd += ["--timerange", timerange]

    print(f"  ▶ {pair} {timeframe}", flush=True)
    print(f"  CMD: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, cwd=str(PROJECT),
            capture_output=True, text=True, timeout=600
        )
        sec = round(time.time() - t0, 1)
        ok  = r.returncode == 0
        print(f"  {'✓' if ok else '✗'} ({sec}s) rc={r.returncode}", flush=True)

        # 保存原始输出
        res_dir = out_dir / "results"
        res_dir.mkdir(exist_ok=True)
        with open(res_dir / f"{name}_stdout.txt", "w", encoding="utf-8") as f:
            f.write(r.stdout)
        if r.stderr:
            with open(res_dir / f"{name}_stderr.txt", "w", encoding="utf-8") as f:
                f.write(r.stderr)

        if not ok:
            return {"pair": pair, "timeframe": timeframe,
                    "success": False, "error": (r.stderr or "")[:300]}

        metrics = parse_metrics(r.stdout, pair, timeframe)
        metrics["success"]    = True
        metrics["elapsed_sec"] = sec

        # 保存指标 JSON
        met_dir = out_dir / "metrics"
        met_dir.mkdir(exist_ok=True)
        with open(met_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"   交易={metrics['total_trades']}  "
              f"胜率={metrics['win_pct']}%  "
              f"收益={metrics['tot_profit_pct']}%  "
              f"回撤={metrics['max_dd_pct']}%  "
              f"Sharpe={metrics['sharpe']}", flush=True)
        return metrics

    except subprocess.TimeoutExpired:
        print(f"  ✗ 超时", flush=True)
        return {"pair": pair, "timeframe": timeframe,
                "success": False, "error": "TIMEOUT"}
    except Exception as e:
        print(f"  ✗ 异常: {e}", flush=True)
        return {"pair": pair, "timeframe": timeframe,
                "success": False, "error": str(e)}


def build_html(all_metrics, out_dir, timerange):
    """生成汇总 HTML 报告。"""
    rows = ""
    for cat, pairs in COINS.items():
        for pair in pairs:
            for tf in ["1m", "5m", "15m"]:
                m = next((x for x in all_metrics
                          if x["pair"] == pair and x["timeframe"] == tf), None)
                if m and m.get("success"):
                    rows += (
                        f"<tr><td>{cat}</td><td>{pair}</td><td>{tf}</td>"
                        f"<td style='color:green'>✓</td>"
                        f"<td>{m['total_trades']}</td>"
                        f"<td>{m['win_pct']}%</td>"
                        f"<td>{m['tot_profit_pct']}%</td>"
                        f"<td>{m['avg_profit_pct']}%</td>"
                        f"<td style='color:red'>{m['max_dd_pct']}%</td>"
                        f"<td>{m['profit_factor']}</td>"
                        f"<td>{m['sharpe']}</td></tr>\n"
                    )
                else:
                    err = (m.get("error", "未完成")[:20]) if m else "未完成"
                    rows += (
                        f"<tr><td>{cat}</td><td>{pair}</td><td>{tf}</td>"
                        f"<td style='color:red'>{err}</td>"
                        f"<td>—</td><td>—</td><td>—</td>"
                        f"<td>—</td><td>—</td><td>—</td><td>—</td></tr>\n"
                    )

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>Meme_限制定投_马丁 回测报告</title>
<style>
  body {{ font-family: 'Segoe UI',sans-serif; margin:20px; background:#f5f5f5; }}
  h1 {{ color:#333; }}
  table {{ border-collapse:collapse; width:100%; background:white; }}
  th,td {{ border:1px solid #ddd; padding:8px; text-align:center; font-size:13px; }}
  th {{ background:#2c3e50; color:white; }}
  tr:nth-child(even) {{ background:#f9f9f9; }}
  .summary {{ margin:20px 0; padding:15px; background:white; border-radius:8px; }}
</style></head><body>
<h1>📊 Meme_限制定投_马丁 全币种全周期回测报告</h1>
<div class="summary">
  <p>策略: """ + STRAT_CLASS + """</p>
  <p>回测区间: """ + timerange + """</p>
  <p>生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
  <p>完成: """ + str(sum(1 for m in all_metrics if m.get("success"))) + """ / """ + str(len(all_metrics)) + """ 个回测</p>
</div>
<table>
  <thead><tr>
    <th>类别</th><th>币种</th><th>周期</th><th>状态</th>
    <th>交易数</th><th>胜率</th><th>总收益%</th><th>均收益%</th>
    <th>最大回撤%</th><th>盈亏因子</th><th>Sharpe</th>
  </tr></thead>
  <tbody>""" + rows + """  </tbody>
</table>
<p style="margin-top:20px;color:#888;font-size:12px;">
  详细 stdout 见 results/ 目录；指标 JSON 见 metrics/ 目录。
</p></body></html>"""

    html_dir = out_dir / "html"
    html_dir.mkdir(exist_ok=True)
    p = html_dir / "index.html"
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None,
                   help="币种列表，如 BTC ETH H")
    ap.add_argument("--timeframes", nargs="*",
                   default=["1m", "5m", "15m"],
                   help="周期列表")
    ap.add_argument("--timerange", default=DEFAULT_TIMERANGE,
                   help="回测区间")
    args = ap.parse_args()

    timerange = args.timerange

    # 收集币种
    all_pairs = []
    for pl in COINS.values():
        all_pairs.extend(pl)
    if args.pairs:
        want = []
        for p in args.pairs:
            if "/" not in p:
                want.append(p.upper() + "/USDT")
            else:
                want.append(p.upper())
        all_pairs = [p for p in all_pairs if p in want]
    timeframes = args.timeframes

    # 输出目录
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT / "deliverables" / f"backtest_meme_limited_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Meme_限制定投_马丁 批量回测")
    print(f"  币种: {all_pairs}")
    print(f"  周期: {timeframes}")
    print(f"  区间: {timerange}")
    print(f"  输出: {out}")
    print("=" * 60)

    all_results = []
    total = len(all_pairs) * len(timeframes)
    cnt = 0
    for pair in all_pairs:
        for tf in timeframes:
            cnt += 1
            print(f"\n[{cnt}/{total}]", flush=True)
            r = run_one(pair, tf, out, timerange)
            all_results.append(r)
            time.sleep(1)

    print("\n生成 HTML 报告...", flush=True)
    html_path = build_html(all_results, out, timerange)
    print(f"✅ 完成! 报告: {html_path}")

    # 保存总览
    summary = {
        "strategy": STRAT_CLASS,
        "timerange": timerange,
        "total": total,
        "done": sum(1 for r in all_results if r.get("success")),
        "results": all_results,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
