"""
正确解析回测日志，生成汇总报告
用法: python scripts/parse_logs_v2.py
"""
import re
import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("F:/source/freqtrade/user_data/backtest_results")

# 日志文件中摘要表的正则
# | Total profit % | -17.78% |
RE_TOT_PCT = re.compile(r"\|\s*Total profit %\s*\|\s*([-\d.]+)%")
# | OptimizedGoldStrategy_1m | 41 | -0.48 | -177.834 | -17.78 | 0:55:00 | 4 0 37 9.8 | ... |
RE_SUMMARY = re.compile(
    r"\|\s*OptimizedGoldStrategy_(\S+)\s*\|"
    r"\s*(\d+)\s*\|"  # trades
    r"\s*([-\d.]+)\s*\|"  # avg profit %
    r"\s*([-\d.]+)\s*\|"  # tot profit USDT
    r"\s*([-\d.]+)\s*\|"  # tot profit %
    r".*?"  # skip duration
    r"\|\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)"  # win draw loss win%
)


def parse_log(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    r = {"file": path.stem, "pair": "", "tf": "", "has_result": False}

    # 从文件名提取
    name = path.stem  # OptimizedGoldStrategy_1m__BTC_USDT
    if "__" in name:
        parts = name.split("__")
        r["strategy"] = parts[0]
        pair_raw = parts[1]
        r["pair"] = pair_raw.replace("_", "/")
        if "_" in r["strategy"]:
            r["tf"] = r["strategy"].split("_")[-1]

    # 从内容提取
    full_text = "\n".join(lines)
    m = RE_TOT_PCT.search(full_text)
    if m:
        r["tot_profit_pct"] = float(m.group(1))
        r["has_result"] = True

    # 解析 STRATEGY SUMMARY 行
    for line in lines:
        m = RE_SUMMARY.match(line.strip())
        if m:
            r["tf"] = m.group(1)
            r["trades"] = int(m.group(2))
            r["avg_profit_pct"] = float(m.group(3))
            r["tot_profit_usdt"] = float(m.group(4))
            r["tot_profit_pct2"] = float(m.group(5))
            r["wins"] = int(m.group(6))
            r["draws"] = int(m.group(7))
            r["losses"] = int(m.group(8))
            r["win_pct"] = float(m.group(9))
            r["has_result"] = True
            break

    return r


def main():
    logs = sorted(LOG_DIR.glob("OptimizedGoldStrategy_*.log"))
    print(f"找到 {len(logs)} 个日志文件")

    results = [parse_log(l) for l in logs]

    # 按 tf 分组
    by_tf = {}
    for r in results:
        tf = r["tf"] or "unknown"
        by_tf.setdefault(tf, []).append(r)

    # 打印报告
    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append("金麒麟策略 多周期回测汇总")
    out_lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out_lines.append("=" * 80)

    for tf in sorted(by_tf.keys()):
        out_lines.append(f"\n{'#' * 60}")
        out_lines.append(f"# {tf} 周期")
        out_lines.append(f"{'#' * 60}")
        out_lines.append(f"{'币种':<16} {'交易数':<8} {'总收益%':<12} {'胜率%':<8} {'盈亏(USDT)':<12}")
        out_lines.append("-" * 70)

        for r in by_tf[tf]:
            if r["has_result"]:
                pct = f"{r.get('tot_profit_pct', 0):.2f}%"
                win = f"{r.get('win_pct', 0):.1f}%"
                profit = f"{r.get('tot_profit_usdt', 0):.2f}"
                out_lines.append(
                    f"{r['pair']:<16} {r.get('trades',0):<8} {pct:<12} {win:<8} {profit:<12}"
                )
            else:
                out_lines.append(f"{r['pair']:<16} 无交易")

        # 小计
        tf_vals = [r for r in by_tf[tf] if r["has_result"]]
        if tf_vals:
            avg_pct = sum(r.get("tot_profit_pct", 0) for r in tf_vals) / len(tf_vals)
            out_lines.append("-" * 70)
            out_lines.append(f"{'平均':<16} {'':<8} {avg_pct:.2f}%")

    report = "\n".join(out_lines)
    print(report)

    # 保存
    txt_path = LOG_DIR / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    json_path = LOG_DIR / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    txt_path.write_text(report, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {txt_path}")
    print(f"JSON已保存: {json_path}")


if __name__ == "__main__":
    main()
