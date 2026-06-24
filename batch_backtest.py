#!/usr/bin/env python3
"""
批量回测 v5 — 串行，纯文件输出，最稳定
每个策略×时间框架跑一次（所有币种），结果保存到 JSON
"""
import subprocess, json, re, os, time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PROJECT_DIR   = Path(__file__).parent
VENV_PYTHON  = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
LOG_DIR      = PROJECT_DIR / "user_data" / "bt_logs"
LOG_DIR.mkdir(exist_ok=True)
OUT_JSON     = PROJECT_DIR / "user_data" / "backtest_results.json"
PROGRESS     = PROJECT_DIR / "user_data" / "bt_progress.txt"

STRATEGIES = [
    "ChanlunCenterBreakoutStrategy",
    "ChanlunSecondBuyStrategy",
    "ChanlunSecondBuyMinimal",
    "ChanlunSecondBuyTrailing",
    "ChanlunMacdDivergenceStrategy",
    "ChanlunRsiTimingStrategy",
    "ChanlunVolumeConfirmationStrategy",
    "ChanlunMaRsiSecondBuyStrategy",
    "MemeLimitedDcaMartingaleStrategy",
    "MemeVolatilityGridMartingaleStrategy",
    "MemeAntiMartingaleTrendStrategy",
    "MemeHedgeProxyMartingaleStrategy",
    "VolatilityBreakoutMomentumStrategy",
    "VolumeAtrMeanReversionStrategy",
    "MultiFactorBtcSyncStrategy",
    "RiskFirstStrongTrendStrategy",
    "TrendPyramidStrategy",
    "SwingTrendFollowStrategy",
    "TrendPyramidAntiMartingaleStrategy",
    "TrendFilteredGridStrategy",
    "TopReversalShortStrategy",
    "GateScalpMomentumStrategy",
    "ManualOnlyStrategy",
]

CATEGORIES = {
    "stable": ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","LTC/USDT","HYPE/USDT"],
    "mid":    ["XCN/USDT","IP/USDT","BAS/USDT","PEAQ/USDT","TA/USDT"],
    "meme":   ["H/USDT","VELVET/USDT","BEAT/USDT","COAI/USDT","ALLO/USDT","DN/USDT","STG/USDT"],
}
TF15M_OK = {"BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT",
             "H/USDT","VELVET/USDT","BEAT/USDT","DN/USDT"}


def run_one(strategy: str, timeframe: str) -> dict:
    # 确定币种
    if timeframe == "1m":
        pairs = list({p for ps in CATEGORIES.values() for p in ps})
    else:
        pairs = [p for p in {p for ps in CATEGORIES.values() for p in ps} if p in TF15M_OK]

    # 写临时配置
    cfg_path = PROJECT_DIR / f"user_data/backtest_config_{timeframe}.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["exchange"]["pair_whitelist"] = pairs
    cfg["timeframe"] = timeframe
    tmp = PROJECT_DIR / "user_data" / f"_tmp_{strategy}_{timeframe}.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    log = LOG_DIR / f"{strategy}_{timeframe}.log"
    cmd = [
        VENV_PYTHON, "-u", "-m", "freqtrade", "backtesting",
        "--config", str(tmp),
        "--strategy", strategy,
        "--timeframe", timeframe,
        "--timerange", "20240101-",
        "--export", "none",
    ]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=str(PROJECT_DIR),
                          stdout=lf, stderr=subprocess.STDOUT,
                          timeout=300,
                          env={**os.environ,"PYTHONIOENCODING":"utf-8"})
    elapsed = time.time() - t0
    tmp.unlink(missing_ok=True)

    with open(log, encoding="utf-8", errors="replace") as f:
        out = f.read()

    if rc.returncode != 0:
        err = "\n".join(
            l.strip() for l in out.splitlines()
            if any(k in l for k in ["ERROR","Exception","Fatal","ValueError"])
        )[-5:]
        return {"error": err or out[-400:], "elapsed": elapsed}

    # 解析
    result = {"pair_results": {}, "error": None, "elapsed": elapsed}
    for line in out.splitlines():
        m = re.search(r"│\s*(\S+/USDT)\s*│\s*(\d+)\s*│\s*([-\d.]+)%?", line)
        if m:
            pair = m.group(1).replace("_","/")
            result["pair_results"][pair] = {
                "trades": int(m.group(2)),
                "profit_pct": float(m.group(3)),
            }
    m = re.search(r"TOTAL\s*│\s*(\d+)\s*│\s*([-\d.]+)%", out)
    if m:
        result["total_trades"]   = int(m.group(1))
        result["total_profit_pct"] = float(m.group(2))
    m = re.search(r"Win\s*/\s*Loss\s*│\s*(\d+)\s*/\s*(\d+)", out)
    if m:
        w,l = int(m.group(1)), int(m.group(2))
        result["win_rate"] = w/(w+l) if w+l>0 else None
    for key,pat in [("max_dd",r"Max\s*drawdown.*?([-\d.]+)%"),
                     ("sharpe",r"Sharpe.*?([-\d.]+)"),
                     ("sortino",r"Sortino.*?([-\d.]+)")]:
        m = re.search(pat, out)
        if m:
            try: result[key] = float(m.group(1))
            except ValueError: pass
    return result


def aggregate(result: dict) -> dict:
    cat_res = {}
    pr = result.get("pair_results", {})
    for cat, pairs in CATEGORIES.items():
        cp = {p: pr[p] for p in pairs if p in pr}
        if not cp: continue
        tot_t = sum(v["trades"] for v in cp.values())
        if tot_t > 0:
            wprofit = sum(v["profit_pct"]*v["trades"] for v in cp.values())
            avg_profit = wprofit / tot_t
        else:
            avg_profit = 0
        cat_res[cat] = {"pairs": list(cp), "trades": tot_t, "profit_pct": avg_profit}
    result["category_results"] = cat_res
    return result


def main():
    t0 = datetime.now()
    tasks = [(s,tf) for s in STRATEGIES for tf in ("1m","15m")]
    n = len(tasks)
    print(f"共 {n} 个任务，开始 {t0.strftime('%H:%M')}")
    print("="*50)

    all_results = []
    with open(PROGRESS, "w", encoding="utf-8") as pf:
        pf.write(f"start {t0}\n")

    for i,(s,tf) in enumerate(tasks,1):
        print(f"[{i}/{n}] {s}/{tf} ...", end=" ", flush=True)
        r = run_one(s, tf)
        r["strategy"] = s; r["timeframe"] = tf
        all_results.append(aggregate(r) if not r.get("error") else r)

        status = "OK" if not r.get("error") else "FAIL"
        info = ""
        if not r.get("error"):
            info = f"profit={r.get('total_profit_pct','?')}%, trades={r.get('total_trades',0)}, {r['elapsed']:.0f}s"
        else:
            info = f"{r['error'][:60]}"
        print(f"{status} {info}")
        with open(PROGRESS,"a",encoding="utf-8") as pf:
            pf.write(f"[{i}/{n}] {s}/{tf} {status} {info}\n")

    with open(OUT_JSON,"w",encoding="utf-8") as f:
        json.dump({"timestamp":t0.isoformat(),"results":all_results},
                  f, ensure_ascii=False, indent=2)

    elapsed = (datetime.now()-t0).total_seconds()
    ok = sum(1 for r in all_results if not r.get("error"))
    print(f"\n完成！成功 {ok}/{n}，耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"结果: {OUT_JSON}")


if __name__=="__main__":
    main()
