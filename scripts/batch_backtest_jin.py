"""
金麒麟多周期多币种批量回测脚本
用法: python scripts/batch_backtest_jin.py
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# 回测参数
# ============================================================
STRATEGY_PREFIX = "金麒麟"
TIME_FRAMES = ["1m", "5m", "15m", "1h"]

# 三类币种（注意：文件名格式是 BTC_USDT-1m.feather，回测pair格式是 BTC/USDT）
STABLE_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LTC/USDT", "HYPE/USDT"]
MID_COINS = ["XCN/USDT", "IP/USDT", "BAS/USDT", "PEAQ/USDT", "TA/USDT"]
DEGENE_COINS = ["H/USDT", "VELVET/USDT", "BEAT/USDT", "COAI/USDT", "ALLO/USDT", "DN/USDT", "STG/USDT"]

ALL_CATEGORIES = {
    "稳定型": STABLE_COINS,
    "中间币": MID_COINS,
    "妖币": DEGENE_COINS,
}

# 回测时间范围（根据数据量自动调整，这里用一个较宽的范围）
# 1m 数据大约 5366 根K线 ≈ 3.7天；1h 数据 448根 ≈ 18.7天
# 为了确保有足够数据，用一个统一的起始时间
BACKTEST_TIMERANGE = "--timerange 20240101-20260101"  # 根据实际数据调整

FREQTRADE = Path("..venv/Scripts/freqtrade.exe")
STRATEGY_DIR = Path("user_data/strategies")
DATA_DIR = Path("user_data/data/gate")
RESULT_DIR = Path("user_data/backtest_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_pairs_str(pairs: list) -> str:
    return ",".join(pairs)


def run_backtest(strategy_name: str, tf: str, category: str, pairs: list) -> dict:
    """对单个时间周期+单个币种分类运行回测，返回结果摘要"""
    pairs_str = build_pairs_str(pairs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULT_DIR / f"{strategy_name}_{category}_{timestamp}.json"

    cmd = [
        str(FREQTRADE),
        "backtesting",
        "--strategy", strategy_name,
        "--strategy-path", str(STRATEGY_DIR),
        "-p", pairs_str,
        "--datadir", str(DATA_DIR),
        BACKTEST_TIMERANGE,
        "--export", "json",   # 导出交易记录
        "--export-filename", str(output_file),
    ]

    print(f"\n{'='*60}")
    print(f"[回测] 策略={strategy_name} | 周期={tf} | 分类={category}")
    print(f"[回测] 币种: {pairs_str}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd="F:/source/freqtrade",
    )

    # 保存详细输出
    log_file = RESULT_DIR / f"{strategy_name}_{category}_{timestamp}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"COMMAND: {' '.join(cmd)}\n")
        f.write(f"\n{'='*60} STDOUT {'='*60}\n")
        f.write(result.stdout or "")
        f.write(f"\n{'='*60} STDERR {'='*60}\n")
        f.write(result.stderr or "")
        f.write(f"\nRETURN CODE: {result.returncode}\n")

    # 解析关键指标
    summary = {
        "strategy": strategy_name,
        "timeframe": tf,
        "category": category,
        "pairs": pairs,
        "returncode": result.returncode,
        "log_file": str(log_file),
        "output_file": str(output_file) if output_file.exists() else None,
    }

    # 尝试从 stdout 提取关键指标
    stdout = result.stdout or ""
    for line in stdout.splitlines():
        if "Total profit" in line or "总利润" in line or "profit" in line.lower():
            summary["profit_line"] = line.strip()
        if "Trades:" in line or "交易次数" in line:
            summary["trades_line"] = line.strip()
        if "Win Rate" in line or "胜率" in line:
            summary["winrate_line"] = line.strip()

    print(stdout[-2000:] if len(stdout) > 2000 else stdout)
    if result.stderr:
        print("[STDERR]", result.stderr[-1000:])

    return summary


def main():
    if not FREQTRADE.exists():
        print(f"错误: 找不到 freqtrade: {FREQTRADE}")
        sys.exit(1)

    all_results = []

    for tf in TIME_FRAMES:
        strategy_name = f"OptimizedGoldStrategy_{tf}"
        print(f"\n{'#'*60}")
        print(f"# 开始回测时间周期: {tf} (策略: {strategy_name})")
        print(f"{'#'*60}")

        for cat_name, pairs in ALL_CATEGORIES.items():
            # 过滤掉没有数据的币种
            available_pairs = []
            for p in pairs:
                pair_file_tf = p.replace("/", "_") + f"-{tf}.feather"
                if (DATA_DIR / pair_file_tf).exists():
                    available_pairs.append(p)
                else:
                    print(f"  [跳过] {p} 无 {tf} 数据")

            if not available_pairs:
                print(f"  [跳过] 分类 {cat_name} 在 {tf} 无可用数据")
                continue

            summary = run_backtest(strategy_name, tf, cat_name, available_pairs)
            all_results.append(summary)

    # 保存汇总报告
    summary_file = RESULT_DIR / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"回测完成！汇总报告: {summary_file}")
    print(f"{'='*60}")

    # 打印简要汇总
    print("\n简要结果汇总:")
    for r in all_results:
        status = "✓" if r["returncode"] == 0 else "✗"
        profit = r.get("profit_line", "N/A")
        print(f"  {status} {r['strategy']} | {r['category']} | {profit}")


if __name__ == "__main__":
    main()
