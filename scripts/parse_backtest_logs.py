"""
解析批量回测日志，生成汇总报告
用法: python scripts/parse_backtest_logs.py
"""
import re
import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("F:/source/freqtrade/user_data/backtest_results")
OUTPUT_FILE = LOG_DIR / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
OUTPUT_JSON = LOG_DIR / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# 从日志文件中提取关键指标的正则
PATTERNS = {
    "pair": re.compile(r"Result for strategy \S+"),
    "total_profit_pct": re.compile(r"\| Total profit %\s+\|\s+([-\d.]+)%"),
    "total_profit_usdt": re.compile(r"\| Tot Profit USDT\s+\|\s+([-\d.]+)"),
    "trades": re.compile(r"\| Strategy\s+\|\s+(\d+)\s+\|"),
    "win_rate": re.compile(r"\| Strategy\s+.+\|\s+([\d.]+)\s+$"),
    "drawdown": re.compile(r"\| Strategy\s+.+\|\s+([\d.]+ USDT\s+[\d.]+%)\s+"),
    "backtest_period": re.compile(r"Backtested (.+?) \|"),
}


def parse_log_file(log_path: Path) -> dict:
    """解析单个日志文件，提取回测摘要"""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    result = {
        "log_file": str(log_path.name),
        "pair": "",
        "timeframe": "",
        "strategy": "",
        "total_profit_pct": None,
        "total_profit_usdt": None,
        "trades": None,
        "win_rate_pct": None,
        "drawdown": None,
        "backtest_period": None,
        "has_trades": False,
    }

    # 从文件名提取策略和时间周期
    fname = log_path.stem  # e.g. OptimizedGoldStrategy_1m__BTC_USDT
    if "__" in fname:
        parts = fname.split("__")
        result["strategy"] = parts[0]
        pair_tf = parts[1]  # BTC_USDT
        result["pair"] = pair_tf.replace("_", "/")

    if "_" in result["strategy"]:
        tf_part = result["strategy"].split("_")[-1]  # 1m, 5m, etc.
        result["timeframe"] = tf_part

    # 从日志内容提取指标
    for line in lines:
        # Total profit %
        m = re.search(r"Total profit %\s+\|\s+([-\d.]+)", line)
        if m:
            result["total_profit_pct"] = float(m.group(1))

        # Tot Profit USDT
        m = re.search(r"Tot Profit USDT\s+\|\s+([-\d.]+)", line)
        if m:
            result["total_profit_usdt"] = float(m.group(1))

        # Trades count (from STRATEGY SUMMARY table)
        m = re.search(r"\|\s+OptimizedGoldStrategy_\S+\s+\|\s+(\d+)\s+\|", line)
        if m:
            result["trades"] = int(m.group(1))

        # Win % (last number in the strategy summary line)
        m = re.search(r"\|\s+OptimizedGoldStrategy_\S+.+\|\s+([\d.]+)\s+\|", line)
        if m and result["win_rate_pct"] is None:
            # This is fragile, but let's try
            pass

        # Backtest period
        m = re.search(r"Backtested (.+?) \|", line)
        if m:
            result["backtest_period"] = m.group(1)

    # 判断是否有交易
    if result["trades"] and result["trades"] > 0:
        result["has_trades"] = True

    return result


def main():
    log_files = list(LOG_DIR.glob("OptimizedGoldStrategy_*.log"))
    print(f"找到 {len(log_files)} 个日志文件")

    all_results = []
    for lf in sorted(log_files):
        r = parse_log_file(lf)
        all_results.append(r)

    # 按时间周期和币种分类打印
    # 分组: {tf: {pair: result}}
    grouped = {}
    for r in all_results:
        tf = r["timeframe"] or "unknown"
        if tf not in grouped:
            grouped[tf] = {}
        grouped[tf][r["pair"]] = r

    # 打印报告
    lines_out = []
    lines_out.append("=" * 80)
    lines_out.append("金麒麟策略多周期回测汇总报告")
    lines_out.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines_out.append("=" * 80)

    for tf in sorted(grouped.keys()):
        lines_out.append(f"\n{'#' * 60}")
        lines_out.append(f"# 时间周期: {tf}")
        lines_out.append(f"{'#' * 60}")
        lines_out.append(f"{'Pair':<16} {'Trades':<10} {'Profit%':<12} {'Win%':<8} {'DD':<15}")
        lines_out.append("-" * 80)

        for pair in sorted(grouped[tf].keys()):
            r = grouped[tf][pair]
            if r["has_trades"]:
                profit = f"{r['total_profit_pct']:.2f}%" if r["total_profit_pct"] else "N/A"
                trades = str(r["trades"]) if r["trades"] else "0"
                lines_out.append(f"{pair:<16} {trades:<10} {profit:<12} {'N/A':<8} {'N/A':<15}")
            else:
                lines_out.append(f"{pair:<16} {'0':<10} {'N/A':<12} {'N/A':<8} {'N/A':<15}")

        # 小计
        tf_results = list(grouped[tf].values())
        has_trade_results = [r for r in tf_results if r["has_trades"]]
        if has_trade_results:
            avg_profit = sum(r["total_profit_pct"] for r in has_trade_results if r["total_profit_pct"]) / len(has_trade_results)
            lines_out.append(f"{'':<16} {'':<10} {'':<12}")
            lines_out.append(f"{'平均Profit%':<16} {avg_profit:.2f}%")

    report = "\n".join(lines_out)
    print(report)

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {OUTPUT_FILE}")
    print(f"JSON已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
