#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme_限制定投_马丁 全币种全周期批量回测 — 最简可靠版
每个回测独立运行，stdout 直接保存到文件，然后解析。
"""
import subprocess, json, time, sys, re, os
from datetime import datetime
from pathlib import Path

PROJECT    = Path(r"F:\source\freqtrade")
PYTHON     = str(PROJECT / ".venv" / "Scripts" / "python.exe")
STRAT      = "MemeLimitedMartingaleSpotStrategy"
STRAT_PATH = "user_data/strategies"
DATA_DIR   = "user_data/data/gate"
CONFIGS    = [
    "user_data/config/config.json",
    "user_data/config/config_gate_backtest.json",
]
TIMERANGE  = "20260608-20260627"

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "XCN/USDT",
    "H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT",
]
TFS    = ["1m", "5m", "15m"]


def run_bt(pair, tf, out_dir):
    """运行单个回测，保存 stdout 到文件，返回 rc。"""
    safe = pair.replace("/", "_")
    name = f"{STRAT}_{safe}_{tf}"
    out_file = out_dir / "results" / f"{name}_stdout.txt"

    # 构造命令字符串（用 shell=True 方式，更可靠）
    cmd = (
        f'"{PYTHON}" -u -m freqtrade backtesting '
        f'--config {CONFIGS[0]} '
        f'--config {CONFIGS[1]} '
        f'--strategy {STRAT} '
        f'--strategy-path {STRAT_PATH} '
        f'--timeframe {tf} '
        f'-p "{pair}" '
        f'--datadir {DATA_DIR} '
        f'--timerange {TIMERANGE} '
        f'> "{out_file}" 2>&1'
    )
    print(f"  CMD: {cmd[:120]}...", flush=True)
    rc = os.system(f'cd /d "{PROJECT}" && {cmd}')
    return rc, out_file


def parse_stdout(path):
    """解析 freqtrade backtesting stdout 文件。"""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return {"total_trades": 0, "error": "无法读取输出文件"}

    m = {"total_trades": 0, "win_pct": 0.0, "tot_profit_pct": 0.0,
          "max_dd_pct": 0.0, "sharpe": 0.0, "profit_factor": 0.0,
          "exit_reasons": {}, "enter_tags": {}}

    lines = text.split("\n")
    # ── BACKTESTING REPORT ─────────────────────────────────────
    in_sec = False
    for ln in lines:
        if "BACKTESTING REPORT" in ln:
            in_sec = True; continue
        if in_sec and ("┴" in ln or "═" in ln):
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
    # ── EXIT REASON STATS ─────────────────────────────────────
    in_sec = False
    for ln in lines:
        if "EXIT REASON STATS" in ln:
            in_sec = True; continue
        if in_sec and "┴" in ln: break
        if in_sec and "│" in ln and "Exit Reason" not in ln and "TOTAL" not in ln:
            cols = [c.strip() for c in ln.split("│") if c.strip()]
            if len(cols) >= 1:
                m.setdefault("exit_reasons", {})[cols[0]] = int(cols[1])
    # ── SUMMARY METRICS ──────────────────────────────────────
    for ln in lines:
        if "│ Sharpe (closed" in ln:
            m["sharpe"] = float(ln.split("│")[-1].strip())
        elif "│ Sortino (closed" in ln:
            m["sortino"] = float(ln.split("│")[-1].strip())
        elif "│ Profit factor" in ln:
            m["profit_factor"] = float(ln.split("│")[-1].strip())
        elif "│ Max drawdown" in ln and "│" in ln:
            cell = ln.split("│")[-1].strip()
            pm = re.search(r"([\d.]+)%", cell)
            if pm:
                m["max_dd_pct"] = float(pm.group(1))
    return m


def main():
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = PROJECT / "deliverables" / f"backtest_meme_limited_{ts}"
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Meme_限制定投_马丁 批量回测 (os.system 版)")
    print(f"  共: {len(PAIRS)} 币种 × {len(TFS)} 周期 = {len(PAIRS)*len(TFS)} 个")
    print(f"  输出: {out}")
    print("=" * 60)

    all_results = []
    total = len(PAIRS) * len(TFS)
    cnt = 0

    for pair in PAIRS:
        for tf in TFS:
            cnt += 1
            print(f"\n[{cnt}/{total}] {pair} {tf}", flush=True)
            rc, out_file = run_bt(pair, tf, out)

            if rc == 0 and out_file.exists():
                m = parse_stdout(out_file)
                m["pair"]     = pair
                m["timeframe"] = tf
                m["success"]  = True
                print(f"  ✓ 交易={m['total_trades']} 胜率={m.get('win_pct',0)}% "
                      f"收益={m.get('tot_profit_pct',0)}% 回撤={m.get('max_dd_pct',0)}%",
                      flush=True)
            else:
                m = {"pair": pair, "timeframe": tf, "success": False, "error": f"rc={rc}"}
                print(f"  ✗ 失败 rc={rc}", flush=True)

            # 保存指标
            safe = pair.replace("/", "_")
            name = f"{STRAT}_{safe}_{tf}"
            with open(out / "metrics" / f"{name}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)
            all_results.append(m)
            time.sleep(1)

    # 保存汇总
    summary = {"timerange": TIMERANGE, "total": total,
               "done": sum(1 for r in all_results if r.get("success")),
               "results": all_results}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成! 输出: {out}")
    print(f"  成功: {summary['done']}/{total}")


if __name__ == "__main__":
    main()
