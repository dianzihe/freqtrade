#!/usr/bin/env python3
# 生成 27 个回测命令清单（用于手动补跑）
PAIRS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT",
         "XCN/USDT",
         "H/USDT","VELVET/USDT","BEAT/USDT","COAI/USDT"]
TFS   = ["1m","5m","15m"]
for p in PAIRS:
    for tf in TFS:
        print(f".venv/Scripts/python -m freqtrade backtesting "
              f"--config user_data/config/config.json "
              f"--config user_data/config/config_gate_backtest.json "
              f"--strategy MemeLimitedMartingaleSpotStrategy "
              f"--strategy-path user_data/strategies "
              f"--timeframe {tf} -p \"{p}\" "
              f"--datadir user_data/data/gate "
              f"--timerange 20260608-20260627 "
              f"> deliverables/bt_manual/{p.replace('/','_')}_{tf}.txt 2>&1")
