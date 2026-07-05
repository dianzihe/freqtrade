"""
测量 Binance WebSocket 深度流的流量大小
对比 @depth20@100ms vs @depth@100ms 的消息频率和大小
"""
import asyncio
import json
import time
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
CHINA_TZ = timezone(timedelta(hours=8))

async def measure_stream(stream_name: str, duration_sec: int = 30):
    """测量单个流的流量"""
    import websockets
    
    symbol = "btcusdt"
    url = f"wss://stream.binance.com:9443/ws/{symbol}@{stream_name}"
    
    print(f"\n{'='*60}")
    print(f"测量流: {stream_name}")
    print(f"URL: {url}")
    print(f"时长: {duration_sec} 秒")
    print(f"{'='*60}")
    
    msg_count = 0
    total_bytes = 0
    msg_sizes = []
    update_ids = []
    start_time = time.monotonic()
    
    try:
        async with websockets.connect(url, proxy="http://127.0.0.1:7890") as ws:
            print(f"  已连接，开始接收...")
            end_time = start_time + duration_sec
            
            while time.monotonic() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    msg_count += 1
                    
                    if isinstance(msg, bytes):
                        sz = len(msg)
                    else:
                        sz = len(msg.encode('utf-8'))
                    
                    total_bytes += sz
                    msg_sizes.append(sz)
                    
                    # 解析 update_id
                    try:
                        data = json.loads(msg)
                        if 'data' in data:
                            update_ids.append(data['data'].get('u', data['data'].get('lastUpdateId', 0)))
                        elif 'u' in data:
                            update_ids.append(data['u'])
                    except:
                        pass
                    
                    if msg_count <= 3:
                        print(f"  样本消息 {msg_count}: {sz} bytes")
                        try:
                            d = json.loads(msg)
                            if 'data' in d:
                                d = d['data']
                            print(f"    keys: {list(d.keys())}")
                            if 'b' in d:
                                print(f"    bid levels: {len(d['b'])}")
                            if 'a' in d:
                                print(f"    ask levels: {len(d['a'])}")
                        except:
                            pass
                            
                except asyncio.TimeoutError:
                    continue
            
            elapsed = time.monotonic() - start_time
            
            print(f"\n  结果:")
            print(f"    消息数:     {msg_count}")
            print(f"    时长:       {elapsed:.1f} 秒")
            print(f"    消息速率:   {msg_count/elapsed:.1f} 条/秒")
            print(f"    总流量:     {total_bytes/1024:.1f} KB")
            print(f"    平均速率:   {total_bytes/elapsed/1024:.1f} KB/s")
            print(f"    平均消息大小: {total_bytes/msg_count:.0f} bytes")
            if msg_sizes:
                print(f"    消息大小:   min={min(msg_sizes)} max={max(msg_sizes)} median={sorted(msg_sizes)[len(msg_sizes)//2]}")
            
            if update_ids:
                update_ids.sort()
                print(f"    Update ID 范围: {update_ids[0]} ~ {update_ids[-1]}")
                print(f"    更新数: {len(update_ids)} (丢失: {'是' if len(update_ids) != update_ids[-1]-update_ids[0]+1 else '否'})")
            
            return {
                "stream": stream_name,
                "msg_count": msg_count,
                "elapsed": elapsed,
                "msg_per_sec": msg_count / elapsed,
                "total_kb": total_bytes / 1024,
                "kb_per_sec": total_bytes / elapsed / 1024,
                "avg_msg_bytes": total_bytes / msg_count,
                "min_msg_bytes": min(msg_sizes) if msg_sizes else 0,
                "max_msg_bytes": max(msg_sizes) if msg_sizes else 0,
            }
            
    except Exception as e:
        print(f"  错误: {e}")
        return None

async def main():
    print("Binance WebSocket 深度流流量测量")
    print("=" * 60)
    print("正在测量不同深度流的流量...")
    print("（每个流测量 30 秒）")
    
    results = []
    
    # 测量 depth20@100ms
    r1 = await measure_stream("depth20@100ms", duration_sec=20)
    if r1:
        results.append(r1)
    
    await asyncio.sleep(2)
    
    # 测量 depth@100ms (diff stream)
    r2 = await measure_stream("depth@100ms", duration_sec=20)
    if r2:
        results.append(r2)
    
    # 对比
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("对比总结")
        print(f"{'='*60}")
        for r in results:
            print(f"\n{r['stream']}:")
            print(f"  消息速率: {r['msg_per_sec']:.1f} 条/秒")
            print(f"  流量速率: {r['kb_per_sec']:.1f} KB/s ({r['kb_per_sec']*8/1024:.2f} Mbps)")
            print(f"  平均消息: {r['avg_msg_bytes']:.0f} bytes")
        
        if results[0]['msg_per_sec'] > 0 and results[1]['msg_per_sec'] > 0:
            ratio = results[1]['kb_per_sec'] / results[0]['kb_per_sec']
            print(f"\n  @depth@100ms 流量是 @depth20@100ms 的 {ratio:.1f}x")

if __name__ == "__main__":
    asyncio.run(main())
