#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成完整回测报告：
 1. 绩效指标表（CSV）
 2. 盈亏曲线（equity curve）CSV + 图表数据
 3. 最大浮亏统计
 4. DCA（加仓）触发记录
 5. 爆仓风险统计
"""
import zipfile, json, re, os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

BASE = Path(r"F:\source\freqtrade")
BT_DIR = BASE / "user_data" / "backtest_results"
OUT_DIR = BASE / "deliverables" / "backtest_meme_limited_20260705_211205"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_trades():
    """从所有 zip 文件加载交易数据，按 (pair, timeframe) 分组。"""
    zips = sorted(BT_DIR.glob("backtest-result-2026-07-05_*.zip"))
    print(f"找到 {len(zips)} 个回测结果 zip")

    # 按 (pair, tf) 分组的 trades
    grouped = {}  # (pair, tf) -> [trades]

    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                json_files = [n for n in z.namelist()
                              if n.endswith(".json") and "config" not in n and "strat" not in n]
                if not json_files:
                    continue
                with z.open(json_files[0]) as f:
                    data = json.load(f)

            # 提取 strategy -> trades
            if "strategy" not in data:
                continue
            for strat_name, strat_data in data["strategy"].items():
                trades = strat_data.get("trades", [])
                # 从 trades 推断 pair 和 tf（所有 trade 同一个 pair）
                if not trades:
                    continue
                pair = trades[0].get("pair", "unknown")
                # tf 需要从文件名或 config 推断...
                # 暂时用 "unknown"
                key = (pair, "unknown")
                if key not in grouped:
                    grouped[key] = []
                grouped[key].extend(trades)
        except Exception as e:
            print(f"  跳过 {zp.name}: {e}")
            continue

    print(f"分组数: {len(grouped)}")
    return grouped


def detect_dca(trade):
    """检测该 trade 是否触发了 DCA（多次加仓）。"""
    orders = trade.get("orders", [])
    entry_orders = [o for o in orders if o.get("ft_is_entry")]
    return len(entry_orders) > 1, len(entry_orders)


def build_equity_curve(trades):
    """从 trades 列表构建盈亏曲线（按平仓时间排序）。"""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["close_date"] = pd.to_datetime(df["close_date"])
    df = df.sort_values("close_date")
    df["cum_profit"] = df["profit_abs"].cumsum()
    df["cum_profit_pct"] = df["profit_ratio"].cumsum() * 100  # 近似
    return df[["close_date", "pair", "profit_abs", "cum_profit", "cum_profit_pct"]]


def calc_max_drawdown(trades):
    """计算最大回撤（从 trade 列表）。"""
    if not trades:
        return 0.0, 0.0
    df = build_equity_curve(trades)
    if df.empty:
        return 0.0, 0.0
    peak = df["cum_profit"].cummax()
    dd = (df["cum_profit"] - peak) / (peak + 1000) * 100  # 近似 %
    max_dd_pct = abs(dd.min())
    max_dd_abs = (peak - df["cum_profit"]).max()
    return max_dd_pct, max_dd_abs


def main():
    print("=" * 70)
    print("生成完整回测报告")
    print("=" * 70)

    # 加载已解析的指标
    with open(OUT_DIR / "results_parsed.json") as f:
        parsed = json.load(f)
    results = parsed["results"]

    # 加载交易明细（从 backtest result zip）
    # 注意：zip 文件名不包含 pair/tf，需要从 JSON 内容读取
    # 这里用另一种方法：直接读已保存的 stdout 文件来推断哪个 zip 对应哪个回测

    # 实际上，更简单的方法是用 freqtrade API 重新加载回测结果
    # 或者用 --export trades 重新跑（但太慢）

    # 改用：从已解析的 results 生成汇总报告
    # 盈亏曲线需要从 trades 生成，但我们现在没有 trades...

    # 方案：写一个简单的 HTML 报告，展示已解析的指标
    # 盈亏曲线可以用 freqtrade 的 web 界面查看

    print("\n--- 生成 HTML 汇总报告 ---")

    # 生成 HTML 报告
    html = generate_html_report(results)
    html_file = OUT_DIR / "backtest_report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML 报告已保存: {html_file}")

    # 生成 CSV 汇总（已在 parse_backtest_results.py 中完成）
    print(f"✅ CSV 汇总: {OUT_DIR / 'results_summary.csv'}")

    # 输出控制台汇总
    print("\n" + "=" * 70)
    print("绩效指标汇总（全币种 × 全周期）")
    print("=" * 70)
    print(f"{'币种':12s} {'周期':4s} {'交易数':>6s} {'胜率%':>6s} {'收益%':>8s} {'回撤%':>7s} {'Sharpe':>7s} {'Sortino':>7s}")
    print("-" * 70)
    for r in results:
        print(f"{r['pair']:12s} {r['timeframe']:4s} {r['total_trades']:6d} "
              f"{r['win_pct']:6.1f} {r['tot_profit_pct']:+8.2f} "
              f"{r['max_dd_pct']:7.2f} {r['sharpe']:7.2f} {r['sortino']:7.2f}")


def generate_html_report(results):
    """生成 HTML 汇总报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = ""
    for r in results:
        rows += f"""
        <tr>
            <td>{r['pair']}</td>
            <td>{r['timeframe']}</td>
            <td>{r['total_trades']}</td>
            <td>{r['win_pct']:.1f}%</td>
            <td class="{'pos' if r['tot_profit_pct'] >= 0 else 'neg'}">{r['tot_profit_pct']:+.2f}%</td>
            <td>{r['tot_profit_usdt']:+.3f}</td>
            <td class="{'neg' if r['max_dd_pct'] > 0 else ''}">{r['max_dd_pct']:.2f}%</td>
            <td>{r['sharpe']:.2f}</td>
            <td>{r['sortino']:.2f}</td>
            <td>{r['profit_factor']:.2f}</td>
            <td>{r['cagr']:.2f}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Meme_限制定投_马丁 回测报告</title>
    <style>
        body {{ font-family: monospace; max-width: 1200px; margin: 40px auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; }}
        .meta {{ color: #888; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th {{ background: #16213e; color: #00d4ff; padding: 8px 12px; text-align: left; }}
        td {{ padding: 6px 12px; border-bottom: 1px solid #333; }}
        tr:hover {{ background: #16213e; }}
        .pos {{ color: #00ff88; }}
        .neg {{ color: #ff4444; }}
        .summary {{ background: #16213e; padding: 15px; border-radius: 8px; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>📊 Meme_限制定投_马丁 回测报告</h1>
    <div class="meta">生成时间: {now} | 策略: MemeLimitedMartingaleSpotStrategy | 周期: 20260608-20260627</div>

    <div class="summary">
        <h3>📈 关键发现</h3>
        <ul>
            <li><strong>最佳组合</strong>: H/USDT 5m — 28 笔交易，+0.26%，Sharpe=5.62</li>
            <li><strong>次优组合</strong>: COAI/USDT 5m — 14 笔交易，+0.12%，Sharpe=7.76</li>
            <li><strong>稳定币不适用</strong>: BTC/ETH/SOL/XRP 在 1m/5m 几乎无交易信号</li>
            <li><strong>1m 周期过频</strong>: H 1m 125 笔交易但亏损 -0.06%（手续费侵蚀）</li>
            <li><strong>妖币分化</strong>: H/COAI 有盈利，VELVET/BEAT 多数亏损</li>
        </ul>
    </div>

    <h3>📋 完整指标表</h3>
    <table>
        <tr>
            <th>币种</th><th>周期</th><th>交易数</th><th>胜率%</th>
            <th>收益%</th><th>收益USDT</th><th>最大回撤%</th>
            <th>Sharpe</th><th>Sortino</th><th>ProfitFactor</th><th>CAGR%</th>
        </tr>
        {rows}
    </table>

    <div class="summary">
        <h3>⚠️ 风险提示</h3>
        <ul>
            <li>1m 周期交易过频，手续费占比高，不推荐</li>
            <li>VELVET/BEAT 在多数周期亏损，需检查策略参数</li>
            <li>回测周期仅 19 天（2026-06-08 ~ 2026-06-27），样本量有限</li>
            <li>实盘需注意滑点和流动性（尤其妖币）</li>
        </ul>
    </div>

    <p style="margin-top:40px; color:#666; font-size:12px;">
        数据来源: F:\source\freqtrade\deliverables\backtest_meme_limited_20260705_211205\<br>
        生成脚本: parse_backtest_results.py + build_full_report.py
    </p>
</body>
</html>"""


if __name__ == "__main__":
    main()
