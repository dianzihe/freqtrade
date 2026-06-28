import requests
import pandas as pd
import time
import csv
import os
from datetime import datetime

# ============ 配置区 ============
API_KEY = "509e3486-e8c3-407d-ad1a-c5da78479ff5"   # 替换成你自己的 CoinAPI Key

# CoinAPI 的 symbol_id 格式: 交易所_交易类型_基础币_计价币
SYMBOL_ID = "BINANCE_SPOT_BTC_USDT"

# 时间范围 (UTC，ISO8601 格式)
TIME_START = "2024-06-05T00:00:00"
TIME_END   = "2024-06-06T23:59:59"

OUTPUT_FILE = "binance_btc_trades_0605_0606.csv"

# 每次请求返回的最大条数 (CoinAPI 上限 100000)
LIMIT_PER_REQUEST = 100000
# ================================

BASE_URL = "https://rest.coinapi.io/v1/trades/{symbol_id}/history"

headers = {
    "X-CoinAPI-Key": API_KEY,
    "Accept": "application/json"
}

# CSV 字段(CoinAPI trades 接口返回的标准字段)
CSV_FIELDS = ["symbol_id", "time_exchange", "time_coinapi", "uuid",
              "price", "size", "taker_side"]


def estimate_progress(current_time_str):
    """根据当前已下载到的时间，估算进度百分比"""
    try:
        fmt = "%Y-%m-%dT%H:%M:%S"
        # 去掉可能存在的小数秒和 Z
        cur = current_time_str.split(".")[0].replace("Z", "")
        t_cur = datetime.strptime(cur, fmt)
        t_start = datetime.strptime(TIME_START.split(".")[0], fmt)
        t_end = datetime.strptime(TIME_END.split(".")[0], fmt)
        total = (t_end - t_start).total_seconds()
        done = (t_cur - t_start).total_seconds()
        pct = max(0, min(100, done / total * 100))
        return pct
    except Exception:
        return None


def download_trades():
    url = BASE_URL.format(symbol_id=SYMBOL_ID)
    current_start = TIME_START
    total_count = 0
    request_count = 0
    start_clock = time.time()

    # 打开 CSV 文件，准备边下边写
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        while True:
            params = {
                "time_start": current_start,
                "time_end": TIME_END,
                "limit": LIMIT_PER_REQUEST
            }

            request_count += 1
            print(f"[请求 #{request_count}] 发起请求，起点 time_start={current_start} ...")

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=120)
            except requests.exceptions.RequestException as e:
                print(f"  网络异常: {e}，5 秒后重试...")
                time.sleep(5)
                continue

            # 限速处理
            if resp.status_code == 429:
                print("  触发限速(429)，等待 10 秒后重试...")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                print(f"  请求失败，状态码 {resp.status_code}: {resp.text}")
                break

            data = resp.json()
            if not data:
                print("  返回空数据，下载完成。")
                break

            # 立即写入文件
            writer.writerows(data)
            f.flush()  # 强制刷盘，确保数据落地

            total_count += len(data)

            # 进度与速度统计
            last_time = data[-1]["time_exchange"]
            elapsed = time.time() - start_clock
            speed = total_count / elapsed if elapsed > 0 else 0
            pct = estimate_progress(last_time)
            pct_str = f"{pct:5.1f}%" if pct is not None else "  N/A"

            print(f"  ✓ 本次 {len(data):>6} 条 | "
                  f"累计 {total_count:>9} 条 | "
                  f"进度 {pct_str} | "
                  f"已下载到 {last_time} | "
                  f"速度 {speed:,.0f} 条/秒")

            # 判断是否到末尾
            if len(data) < LIMIT_PER_REQUEST:
                print("  返回数量少于上限，已到达时间范围末尾，下载完成。")
                break

            # 用最后一条的时间作为下次请求起点
            if last_time == current_start:
                print("  时间未推进，停止以避免死循环。")
                break
            current_start = last_time

            # 礼貌性延时
            time.sleep(0.3)

    elapsed = time.time() - start_clock
    print(f"\n========== 下载结束 ==========")
    print(f"总条数: {total_count:,}")
    print(f"总请求次数: {request_count}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"文件保存在: {os.path.abspath(OUTPUT_FILE)}")

    return total_count


def preview_file():
    """下载完后预览前几行"""
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE, nrows=5)
            print("\n数据预览(前 5 行):")
            print(df)
        except Exception as e:
            print(f"预览失败: {e}")


if __name__ == "__main__":
    download_trades()
    preview_file()