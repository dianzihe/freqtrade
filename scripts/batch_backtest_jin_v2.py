"""
金麒麟多周期多币种批量回测 - 正确版
每个(时间周期, 币种)组合独立运行，生成独立报告
"""
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

FREQTRADE = "F:/source/freqtrade/.venv/Scripts/freqtrade.exe"
STRATEGY_DIR = "F:/source/freqtrade/user_data/strategies"
DATA_DIR = "F:/source/freqtrade/user_data/data/gate"
CONFIG = "F:/source/freqtrade/user_data/backtest_config.json"
RESULT_DIR = Path("F:/source/freqtrade/user_data/backtest_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 时间周期
TIME_FRAMES = ["1m", "5m", "15m", "1h"]

# 三类币种（freqtrade pair 格式）
CATEGORIES = {
    "稳定型": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LTC/USDT", "HYPE/USDT"],
    "中间币": ["XCN/USDT", "IP/USDT", "BAS/USDT", "PEAQ/USDT", "TA/USDT"],
    "妖币":   ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT", "ALLO/USDT", "DN/USDT", "STG/USDT"],
}

def run_one(pair: str, tf: str, strategy_class: str) -> dict:
    """对一个币种+周期跑回测，返回结果dict"""
    pair_fn = pair.replace("/", "_")
    out_json = RESULT_DIR / f"{strategy_class}__{pair_fn}.json"
    out_log = RESULT_DIR / f"{strategy_class}__{pair_fn}.log"

    cmd = [
        FREQTRADE,
        "backtesting",
        "-c", CONFIG,
        "--strategy", strategy_class,
        "--strategy-path", STRATEGY_DIR,
        "-p", pair,
        "--datadir", DATA_DIR,
        "--export", "trades",
        "--export-filename", str(out_json),
    ]

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd="F:/source/freqtrade",
        timeout=300,
    )

    # 保存日志
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.write(r.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(r.stderr)

    # 解析摘要
    d = {
        "pair": pair, "timeframe": tf,
        "strategy": strategy_class,
        "returncode": r.returncode,
        "log": str(out_log),
        "json": str(out_json) if out_json.exists() else None,
    }

    # 从 stdout 提取关键行
    for line in r.stdout.splitlines():
        s = line.strip()
        if "Total profit %" in s or "Tot Profit %" in s:
            d["tot_profit_pct"] = s
        if "Trades:" in s and "Avg" not in s:
            d["trades"] = s
        if "Win " in s and "Draw" in s:
            d["win_draw_loss"] = s

    return d


def main():
    all_results = []
    total = sum(len(pairs) for pairs in CATEGORIES.values()) * len(TIME_FRAMES)
    done = 0

    for tf in TIME_FRAMES:
        strategy_class = f"OptimizedGoldStrategy_{tf}"
        print(f"\n{'#'*60}")
        print(f"# 时间周期: {tf}  (策略: {strategy_class})")
        print(f"{'#'*60}")

        for cat, pairs in CATEGORIES.items():
            for pair in pairs:
                done += 1
                print(f"\n[{done}/{total}] {tf} | {cat} | {pair}")

                try:
                    res = run_one(pair, tf, strategy_class)
                    all_results.append(res)
                    pct = res.get("tot_profit_pct", "N/A")
                    print(f"  -> {pct}")
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    all_results.append({
                        "pair": pair, "timeframe": tf,
                        "error": str(e)
                    })

    # 保存汇总
    summary_file = RESULT_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"完成！汇总: {summary_file}")
    print(f"{'='*60}")

    # 打印简要表格
    print(f"\n{'Pair':<12} {'TF':<6} {'Trades':<10} {'Profit%':<12} {'Win%':<8}")
    print("-" * 60)
    for r in all_results:
        if "error" in r:
            print(f"{r['pair']:<12} {r['timeframe']:<6} ERROR: {r['error']}")
        else:
            pct = r.get("tot_profit_pct", "N/A")
            print(f"{r['pair']:<12} {r['timeframe']:<6} {pct}")


if __name__ == "__main__":
    main()
