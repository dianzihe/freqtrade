"""
金麒麟多周期多币种批量回测 V3
直接逐对运行回测，从日志文件解析结果，生成汇总报告
"""
import subprocess, json, os, sys
from pathlib import Path
from datetime import datetime

EXE = "F:/source/freqtrade/.venv/Scripts/freqtrade.exe"
STRAT_DIR = "F:/source/freqtrade/user_data/strategies"
DATA_DIR = "F:/source/freqtrade/user_data/data/gate"
CONFIG = "F:/source/freqtrade/user_data/backtest_config.json"
OUT = Path("F:/source/freqtrade/user_data/backtest_results")
OUT.mkdir(parents=True, exist_ok=True)

TFS = ["1m", "5m", "15m", "1h"]
PAIRS = {
    "稳定型": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LTC/USDT", "HYPE/USDT"],
    "中间币": ["XCN/USDT", "IP/USDT", "BAS/USDT", "PEAQ/USDT", "TA/USDT"],
    "妖币":   ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT", "ALLO/USDT", "DN/USDT", "STG/USDT"],
}

def run(pair, tf):
    strat = f"OptimizedGoldStrategy_{tf}"
    pair_fn = pair.replace("/", "_")
    log_path = OUT / f"{strat}__{pair_fn}.log"
    json_path = OUT / f"{strat}__{pair_fn}.json"

    cmd = [
        EXE, "backtesting",
        "-c", CONFIG,
        "--strategy", strat,
        "--strategy-path", STRAT_DIR,
        "-p", pair,
        "--datadir", DATA_DIR,
        "--export", "trades",
        "--backtest-directory", str(OUT),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                      encoding="utf-8", errors="replace",
                      cwd="F:/source/freqtrade", timeout=300)
    # 保存完整输出
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.write(r.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(r.stderr)
    return log_path


def parse_log(log_path):
    """从日志提取关键指标"""
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    res = {"file": log_path.name, "has_trades": False}
    for line in txt.splitlines():
        # Total profit % | -17.78%
        if "Total profit %" in line:
            try:
                val = line.split("|")[2].strip().replace("%", "")
                res["total_profit_pct"] = float(val)
                res["has_trades"] = True
            except: pass
        # | OptimizedGoldStrategy_1m | 41 | ...
        if "OptimizedGoldStrategy_" in line and "|" in line and "Strategy" not in line:
            parts = [x.strip() for x in line.split("|") if x.strip()]
            if len(parts) >= 7:
                try:
                    res["trades"] = int(parts[1])
                    res["win_pct"] = float(parts[6].replace("%", ""))
                except: pass
    return res


def main():
    all_logs = list(OUT.glob("OptimizedGoldStrategy_*.log"))
    done = set()
    for l in all_logs:
        name = l.stem
        if "__" in name:
            done.add(name)

    print(f"已有 {len(done)} 个回测结果，开始补充剩余...")

    total = sum(len(v) for v in PAIRS.values()) * len(TFS)
    count = 0

    for tf in TFS:
        for cat, pairs in PAIRS.items():
            for pair in pairs:
                count += 1
                name = f"OptimizedGoldStrategy_{tf}__{pair.replace('/', '_')}"
                if name in done:
                    print(f"[{count}/{total}] 跳过（已有） {tf} {pair}")
                    continue
                print(f"[{count}/{total}] 运行中... {tf} {pair}")
                try:
                    run(pair, tf)
                    print(f"  -> 完成")
                except Exception as e:
                    print(f"  -> 错误: {e}")

    # 解析所有日志
    print("\n解析结果...")
    all_res = []
    for log in sorted(OUT.glob("OptimizedGoldStrategy_*.log")):
        r = parse_log(log)
        r["file"] = log.stem
        if "__" in log.stem:
            parts = log.stem.split("__")
            r["strategy"] = parts[0]
            r["pair"] = parts[1].replace("_", "/")
            if "_" in r["strategy"]:
                r["tf"] = r["strategy"].split("_")[-1]
        all_res.append(r)

    # 打印汇总
    print("\n" + "=" * 80)
    print("金麒麟策略 多周期回测汇总")
    print("=" * 80)
    for tf in sorted(set(r.get("tf","") for r in all_res)):
        tf_res = [r for r in all_res if r.get("tf") == tf]
        print(f"\n### {tf} 周期 ###")
        print(f"{'Pair':<16} {'Trades':<10} {'Profit%':<12} {'Win%':<8}")
        print("-" * 60)
        for r in sorted(tf_res, key=lambda x: x.get("pair","")):
            pair = r.get("pair", "N/A")
            if r.get("has_trades"):
                pct = f"{r.get('total_profit_pct',0):.2f}%"
                tr = str(r.get("trades", "-"))
                win = f"{r.get('win_pct',0):.1f}%"
                print(f"{pair:<16} {tr:<10} {pct:<12} {win:<8}")
            else:
                print(f"{pair:<16} {'0':<10} {'N/A':<12} {'N/A':<8}")

    # 保存JSON
    jf = OUT / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {jf}")


if __name__ == "__main__":
    main()
