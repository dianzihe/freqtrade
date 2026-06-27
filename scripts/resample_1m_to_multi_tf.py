"""
将 gate 目录下的 1m feather 数据重采样为 5m, 15m, 1h
用法: python scripts/resample_1m_to_multi_tf.py
"""
import pandas as pd
import os
import glob
from pathlib import Path

DATA_DIR = Path("user_data/data/gate")
TARGET_TFS = ["5m", "15m", "1h"]

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """将 1m OHLCV 重采样到目标周期"""
    # 确保 date 是 datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    rule_map = {"5m": "5min", "15m": "15min", "1h": "1h"}
    rule = rule_map[tf]

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    resampled = df.resample(rule).agg(agg).dropna().reset_index()
    return resampled


def main():
    files = list(DATA_DIR.glob("*_USDT-1m.feather"))
    print(f"找到 {len(files)} 个 1m 数据文件")

    for f in files:
        pair_tf = f.stem  # e.g. BTC_USDT-1m
        pair = pair_tf.replace("-1m", "")  # e.g. BTC_USDT

        # 读取 1m 数据
        df_1m = pd.read_feather(f)
        if df_1m.empty:
            print(f"  {pair} 1m 数据为空，跳过")
            continue

        for tf in TARGET_TFS:
            out_name = f"{pair}-{tf}.feather"
            out_path = DATA_DIR / out_name

            # 如果已存在则跳过
            if out_path.exists():
                print(f"  已存在: {out_name}，跳过")
                continue

            try:
                df_tf = resample_ohlcv(df_1m, tf)
                df_tf.to_feather(out_path)
                print(f"  [OK] {pair} -> {tf} ({len(df_tf)} bars)")
            except Exception as e:
                print(f"  [FAIL] {pair} -> {tf}: {e}")


if __name__ == "__main__":
    main()
