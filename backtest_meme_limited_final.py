#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme_限制定投_马丁 全币种全周期批量回测
最终可靠版：使用 Popen 逐行读取，避免 Windows 子进程返回码问题。
"""
import subprocess, json, time, sys, re, os
from datetime import datetime
from pathlib import Path

PROJECT    = Path(r"F:\source\freqtrade")
PYTHON     = str(PROJECT / ".venv" / "Scripts" / "python.exe")
STRAT     = "MemeLimitedMartingaleSpotStrategy"
STRATPATH = "user_data/strategies"
DATADIR   = "user_data/data/gate"
CONFIGS   = [
    "user_data/config/config.json",
    "user_data/config/config_gate_backtest.json",
]
TIMERANGE = "20260608-20260627"

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "XCN/USDT",
    "H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT",
]
TFS   = ["1m", "5m", "15m"]


def run_bt(pair, tf, out_file):
    """
    运行单个回测，stdout 逐行写入 out_file。
    返回 (return_code, stdout_text)。
    """
    safe = pair.replace("/", "_")
    cmd = [
        PYTHON, "-u", "-m", "freqtrade", "backtesting",
        *[x for c in CONFIGS for x in ("--config", c)],
        "--strategy", STRAT,
        "--strategy-path", STRATPATH,
        "--timeframe", tf,
        "-p", pair,
        "--datadir", DATADIR,
        "--timerange", TIMERANGE,
    ]
    print(f"  CMD: {' '.join(cmd[:6])}...", flush=True)
    stdout_lines = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with open(out_file, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                stdout_lines.append(line)
        proc.wait()
        rc = proc.returncode
        print(f"  rc={rc}  ({len(stdout_lines)} 行输出)", flush=True)
        return rc, "".join(stdout_lines)
    except Exception as e:
        print(f"  ✗ 异常: {e}", flush=True)
        return -1, ""


def parse_stdout(text):
    """解析 freqtrade backtesting stdout。"""
    m = {
        "total_trades": 0, "win_pct": 0.0, "tot_profit_pct": 0.0,
        "max_dd_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "profit_factor": 0.0, "exit_reasons": {}, "enter_tags": {},
    }
    lines = text.split("\n")
    try:
        # BACKTESTING REPORT
        in_sec = False
        for ln in lines:
            if "BACKTESTING REPORT" in ln:
                in_sec = True; continue
            if in_sec and ("┴" in ln or "══" in ln):
                break
            if in_sec and "│" in ln and "TOTAL" not in ln and "Pair" not in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 7:
                    m["total_trades"] = int(cols[1])
                    m["avg_profit_pct"] = float(cols[2])
                    m["tot_profit_abs"] = float(cols[3])
                    m["tot_profit_pct"] = float(cols[4])
                    wi = cols[6].split()
                    if len(wi) >= 4:
                        m["win_pct"] = float(wi[3])
        # EXIT REASON STATS
        in_sec = False
        for ln in lines:
            if "EXIT REASON STATS" in ln:
                in_sec = True; continue
            if in_sec and "┴" in ln: break
            if in_sec and "│" in ln and "Exit Reason" not in ln and "TOTAL" not in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if cols:
                    m.setdefault("exit_reasons", {})[cols[0]] = int(cols[1])
        # SUMMARY METRICS
        for ln in lines:
            if "│ Sharpe (closed" in ln:
                cols = [c.strip() for c in ln.split("│")]
                if len(cols) >= 3:
                    m["sharpe"] = float(cols[-1])
            elif "│ Sortino (closed" in ln:
                cols = [c.strip() for c in ln.split("│")]
                if len(cols) >= 3:
                    m["sortino"] = float(cols[-1])
            elif "│ Profit factor" in ln:
                cols = [c.strip() for c in ln.split("│")]
                if len(cols) >= 3:
                    m["profit_factor"] = float(cols[-1])
            elif "│ Max drawdown" in ln and "│" in ln:
                cell = ln.split("│")[-1].strip()
                pm = re.search(r"([\d.]+)%", cell)
                if pm:
                    m["max_dd_pct"] = float(pm.group(1))
    except Exception as e:
        print(f"  解析异常: {e}", flush=True)
    return m


def main():
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT / "deliverables" / f"backtest_meme_limited_{ts}"
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Meme_限制定投_马丁 批量回测 (Popen 版)")
    print(f"  共: {len(PAIRS)} 币种 × {len(TFS)} 周期 = {len(PAIRS)*len(TFS)} 个")
    print(f"  输出: {out}")
    print("=" * 60)

    all_results = []
    total = len(PAIRS) * len(TFS)
    cnt = 0

    for pair in PAIRS:
        for tf in TFS:
            cnt += 1
            safe = pair.replace("/", "_")
            name = f"{STRAT}_{safe}_{tf}"
            out_file = str(out / "results" / f"{name}_stdout.txt")
            print(f"\n[{cnt}/{total}] {pair} {tf}", flush=True)

            rc, stdout_text = run_bt(pair, tf, out_file)

            if rc == 0 and stdout_text:
                m = parse_stdout(stdout_text)
                m["pair"]    = pair
                m["timeframe"] = tf
                m["success"]  = True
                print(f"  ✓ 交易={m['total_trades']} 胜率={m['win_pct']}% "
                      f"收益={m['tot_profit_pct']}% 回撤={m['max_dd_pct']}%",
                      flush=True)
            else:
                m = {"pair": pair, "timeframe": tf, "success": False,
                      "error": f"rc={rc}"}
                print(f"  ✗ 失败 rc={rc}", flush=True)

            # 保存指标
            with open(out / "metrics" / f"{name}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)
            all_results.append(m)
            time.sleep(1)

    # 保存汇总
    summary = {
        "strategy": STRAT, "timerange": TIMERANGE,
        "total": total,
        "done": sum(1 for r in all_results if r.get("success")),
        "results": all_results,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成! 成功: {summary['done']}/{total}")
    print(f"  输出: {out}")


if __name__ == "__main__":
    main()
