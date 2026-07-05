#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 backtest-results zip 文件中提取交易明细。
生成：
 1. 各币种各周期盈亏曲线 CSV
 2. DCA（加仓）触发记录
 3. 爆仓风险统计（最大浮亏、连续亏损）
"""
import zipfile, json, re, os
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE = Path(r"F:\source\freqtrade")
RESULTS_DIR = BASE / "deliverables" / "backtest_meme_limited_20260705_211205"
BT_DIR = BASE / "user_data" / "backtest_results"

# 从 zip 文件名解析 pair 和 timeframe
# 格式: backtest-result-YYYY-MM-DD_HH-MM-SS.zip
# 但文件名不包含 pair/tf 信息...
# 需要从 JSON 内容中读取

def extract_trades_from_zip(zip_path):
    """从 zip 中提取 trades 列表。"""
    with zipfile.ZipFile(zip_path) as z:
        # 找 JSON 文件
        json_files = [n for n in z.namelist() if n.endswith(".json") and "config" not in n and "strat" not in n]
        if not json_files:
            return None
        with z.open(json_files[0]) as f:
            data = json.load(f)
    return data


def main():
    print("=" * 70)
    print("提取交易明细")
    print("=" * 70)

    # 收集所有今天的 zip 文件
    zips = sorted(BT_DIR.glob("backtest-result-2026-07-05_*.zip"))
    print(f"找到 {len(zips)} 个回测结果 zip")

    all_trades = []  # 所有交易 flatten

    for zp in zips:
        data = extract_trades_from_zip(zp)
        if not data:
            continue

        # 解析 JSON 结构
        # 格式: { "strategy": { "pair_timeframe": { "trades": [...] } } }
        # 或: { "strategy": [ { "trades": [...] } ] }
        strategy_key = list(data.keys())[0] if isinstance(data, dict) else None
        if not strategy_key:
            continue

        strat_data = data[strategy_key]
        if isinstance(strat_data, dict):
            # 按 pair_tf 分组的格式
            for key, val in strat_data.items():
                if isinstance(val, dict) and "trades" in val:
                    pair_tf = key  # e.g. "BTC/USDT_15m"
                    trades = val["trades"]
                    for t in trades:
                        t["_pair_tf"] = pair_tf
                        all_trades.append(t)
        elif isinstance(strat_data, list):
            # 列表格式
            for item in strat_data:
                if "trades" in item:
                    trades = item["trades"]
                    pair = item.get("pair", "unknown")
                    tf = item.get("timeframe", "unknown")
                    for t in trades:
                        t["_pair"] = pair
                        t["_timeframe"] = tf
                        all_trades.append(t)

    print(f"总交易数: {len(all_trades)}")

    if not all_trades:
        # 尝试另一种格式
        print("尝试直接解析 JSON...")
        for zp in zips[:1]:
            data = extract_trades_from_zip(zp)
            print(f"  {zp.name}: keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"    {k}: type={type(v).__name__}")
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            print(f"      {k2}: type={type(v2).__name__}")

    # 保存原始交易数据
    with open(RESULTS_DIR / "all_trades.json", "w", encoding="utf-8") as f:
        json.dump(all_trades, f, indent=2, default=str, ensure_ascii=False)
    print(f"✅ 交易数据已保存: {RESULTS_DIR / 'all_trades.json'}")


if __name__ == "__main__":
    main()
