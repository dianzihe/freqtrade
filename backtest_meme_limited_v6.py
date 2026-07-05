#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme_限制定投_马丁 全币种全周期批量回测
v6: 使用 subprocess.run(capture_output=True) + 正确合并 stderr，避免 Popen 丢输出问题。
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


def run_bt(pair, tf):
    """
    运行单个回测，返回 (return_code, stdout_text, stderr_text)。
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
    print(f"  CMD: {' '.join(cmd[:8])} ...", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + "\n" + stderr
        rc = proc.returncode
        print(f"  rc={rc}  stdout={len(stdout)}ch  stderr={len(stderr)}ch", flush=True)
        if rc != 0 and stderr:
            print(f"  STDERR: {stderr[:500]}", flush=True)
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        print(f"  ✗ 超时 (300s)", flush=True)
        return -2, "", "timeout"
    except Exception as e:
        print(f"  ✗ 异常: {e}", flush=True)
        return -1, "", str(e)


def parse_stdout(text):
    """解析 freqtrade backtesting stdout。"""
    m = {
        "total_trades": 0, "win_pct": 0.0, "tot_profit_pct": 0.0,
        "max_dd_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "profit_factor": 0.0, "cagr": 0.0,
        "exit_reasons": {}, "enter_tags": {},
        "avg_duration": "", "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
    }
    lines = text.split("\n")
    try:
        # BACKTESTING REPORT — 找 TOTAL 行
        in_sec = False
        for i, ln in enumerate(lines):
            if "BACKTESTING REPORT" in ln:
                in_sec = True; continue
            if in_sec and ("┴" in ln or "══" in ln):
                break
            if in_sec and "│" in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 2 and cols[0] == "TOTAL":
                    # TOTAL │ 28 │ 0.94 │ 2.603 │ 0.26 │ 0:33:00 │ 22 0 6 78.6
                    if len(cols) >= 7:
                        m["total_trades"]   = int(cols[1])
                        m["avg_profit_pct"] = float(cols[2])
                        m["tot_profit_usdt"]= float(cols[3])
                        m["tot_profit_pct"] = float(cols[4])
                        m["avg_duration"]   = cols[5]
                        # win_pct from last column
                        wcol = cols[6]
                        nums = re.findall(r"[\d.]+", wcol)
                        if nums:
                            m["win_pct"] = float(nums[-1])
                    break

        # EXIT REASON STATS
        in_sec = False
        for ln in lines:
            if "EXIT REASON STATS" in ln:
                in_sec = True; continue
            if in_sec and ("┴" in ln or "══" in ln):
                break
            if in_sec and "│" in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 2 and cols[0] != "Exit Reason" and cols[0] != "TOTAL":
                    m["exit_reasons"][cols[0]] = int(cols[1])

        # ENTER TAG STATS
        in_sec = False
        for ln in lines:
            if "ENTER TAG STATS" in ln:
                in_sec = True; continue
            if in_sec and ("┴" in ln or "══" in ln):
                break
            if in_sec and "│" in ln:
                cols = [c.strip() for c in ln.split("│") if c.strip()]
                if len(cols) >= 2 and cols[0] != "Enter Tag" and cols[0] != "TOTAL":
                    m["enter_tags"][cols[0]] = int(cols[1])

        # SUMMARY METRICS
        for ln in lines:
            if "│ Sharpe (closed" in ln:
                cols = [c.strip() for c in ln.split("│")]
                val = cols[-1]
                try: m["sharpe"] = float(re.search(r"([\d.]+)", val).group(1))
                except: pass
            elif "│ Sortino (closed" in ln:
                cols = [c.strip() for c in ln.split("│")]
                val = cols[-1]
                try: m["sortino"] = float(re.search(r"([\d.]+)", val).group(1))
                except: pass
            elif "│ Profit factor" in ln:
                cols = [c.strip() for c in ln.split("│")]
                val = cols[-1]
                try: m["profit_factor"] = float(re.search(r"([\d.]+)", val).group(1))
                except: pass
            elif "│ CAGR" in ln:
                cols = [c.strip() for c in ln.split("│")]
                val = cols[-1]
                try: m["cagr"] = float(re.search(r"([\d.]+)", val).group(1))
                except: pass
            elif "│ Max % of account underwater" in ln and "closed" not in ln:
                # 取第一个（闭仓）回撤
                cell = ln.split("│")[-1].strip()
                pm = re.search(r"([\d.]+)%", cell)
                if pm and m["max_dd_pct"] == 0.0:
                    m["max_dd_pct"] = float(pm.group(1))
            elif "│ Max drawdown" in ln:
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
    print(f"Meme_限制定投_马丁 批量回测 (v6 subprocess.run)")
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
            print(f"\n[{cnt}/{total}] {pair} {tf}", flush=True)

            rc, stdout_text, stderr_text = run_bt(pair, tf)

            # 保存原始输出
            with open(out / "results" / f"{name}_stdout.txt", "w", encoding="utf-8") as f:
                f.write(stdout_text)
            if stderr_text:
                with open(out / "results" / f"{name}_stderr.txt", "w", encoding="utf-8") as f:
                    f.write(stderr_text)

            if rc == 0 and stdout_text:
                m = parse_stdout(stdout_text)
                m["pair"]       = pair
                m["timeframe"]  = tf
                m["success"]    = True
                print(f"  ✓ 交易={m['total_trades']} 胜率={m['win_pct']}% "
                      f"收益={m['tot_profit_pct']}% 回撤={m['max_dd_pct']}% "
                      f"Sharpe={m['sharpe']}",
                      flush=True)
            else:
                m = {"pair": pair, "timeframe": tf, "success": False,
                      "error": f"rc={rc}", "stderr": stderr_text[:500]}
                print(f"  ✗ 失败 rc={rc}", flush=True)

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

    # 生成简易对比表
    print("\n--- 结果汇总 ---")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['pair']:12s} {r['timeframe']:4s} | "
                  f"交易={r['total_trades']:3d} 胜率={r['win_pct']:5.1f}% "
                  f"收益={r['tot_profit_pct']:6.2f}% 回撤={r['max_dd_pct']:5.2f}%")
        else:
            print(f"  {r['pair']:12s} {r['timeframe']:4s} | ✗ 失败")


if __name__ == "__main__":
    main()
