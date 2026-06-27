"""
正确解析回测日志，生成汇总报告
用法: python scripts/gen_final_report.py
"""
import re
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("F:/source/freqtrade/user_data/backtest_results")

# 正确的日志解析
def parse_log(log_path):
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    if "Total profit %" not in txt:
        return None
    
    # 提取币种
    # 文件名: OptimizedGoldStrategy_1m__BTC_USDT.log
    stem = log_path.stem
    pair_raw = stem.split("__")[1]  # BTC_USDT
    pair = pair_raw.replace("_", "/")  # BTC/USDT
    
    # 提取时间周期
    strat_part = stem.split("__")[0]  # OptimizedGoldStrategy_1m
    tf = strat_part.split("_")[-1]  # 1m
    
    result = {"pair": pair, "tf": tf, "log": log_path.name}
    
    # 提取总收益%
    m = re.search(r"Total profit %\s*\|\s*(-?[\d.]+)%", txt)
    if m:
        result["pct"] = float(m.group(1))
    
    # 提取交易数和胜率（从 STRATEGY SUMMARY 表）
    for line in txt.splitlines():
        if "| OptimizedGoldStrategy_" in line and "|" in line:
            # 跳过表头行
            if "Strategy" in line or "Enter Tag" in line or "Exit Reason" in line:
                continue
            parts = [x.strip() for x in line.split("|") if x.strip()]
            if len(parts) >= 8:
                try:
                    result["trades"] = int(parts[1])
                    result["win_pct"] = float(parts[6].replace("%", ""))
                    result["wins"] = int(parts[3])
                    result["losses"] = int(parts[5])
                except:
                    pass
                break
    
    return result


def main():
    logs = sorted(LOG_DIR.glob("OptimizedGoldStrategy_*.log"))
    print(f"找到 {len(logs)} 个日志文件")
    
    all_results = []
    for log in logs:
        r = parse_log(log)
        if r:
            all_results.append(r)
    
    print(f"成功解析 {len(all_results)} 个结果")
    
    # 按时周期分组
    tfs = sorted(set(r["tf"] for r in all_results))
    
    # 打印报告
    lines = []
    lines.append("=" * 100)
    lines.append("金麒麟策略 多周期多币种 回测汇总报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 100)
    lines.append("")
    
    for tf in tfs:
        tf_results = sorted(
            [r for r in all_results if r["tf"] == tf],
            key=lambda x: x["pair"]
        )
        
        lines.append(f"{'#' * 60}")
        lines.append(f"# 时间周期: {tf}")
        lines.append(f"{'#' * 60}")
        lines.append(f"{'币种':<14} {'交易数':>8} {'总收益%':>12} {'胜率%':>8} {'盈亏USDT':>12}")
        lines.append("-" * 80)
        
        for r in tf_results:
            pct = f"{r.get('pct', 0):.2f}%" if "pct" in r else "N/A"
            win = f"{r.get('win_pct', 0):.1f}%" if "win_pct" in r else "N/A"
            trades = str(r.get("trades", "-"))
            lines.append(f"{r['pair']:<14} {trades:>8} {pct:>12} {win:>8} {'N/A':>12}")
        
        # 平均
        pcts = [r["pct"] for r in tf_results if "pct" in r]
        if pcts:
            avg = sum(pcts) / len(pcts)
            lines.append("-" * 80)
            lines.append(f"{'平均':<14} {'':>8} {avg:.2f}%")
        lines.append("")
    
    # 所有周期合计
    all_pcts = [r["pct"] for r in all_results if "pct" in r]
    if all_pcts:
        lines.append("=" * 100)
        lines.append(f"全部 {len(all_pcts)} 个组合 平均收益: {sum(all_pcts)/len(all_pcts):.2f}%")
        lines.append(f"正收益组合数: {sum(1 for p in all_pcts if p > 0)}")
        lines.append(f"负收益组合数: {sum(1 for p in all_pcts if p <= 0)}")
        lines.append("=" * 100)
    
    report = "\n".join(lines)
    print(report)
    
    # 保存
    out = LOG_DIR / f"FINAL_REPORT_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
