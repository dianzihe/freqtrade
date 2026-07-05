#!/usr/bin/env python3
import json

cfg = json.load(open("user_data/config/config.json"))
gate = json.load(open("user_data/config/config_gate_backtest.json"))

# 保留 Gate 交易所设置，合并回测必需字段
for k in ["stake_currency","stake_amount","max_open_trades",
         "tradable_balance_ratio","minimal_roi","stoploss",
         "trailing_stop","timeframe","unfilledtimeout",
         "use_exit_signal","exit_profit_only",
         "position_adjustment_enable","max_entry_position_adjustment",
         "order_types","order_time_in_force"]:
    if k in cfg:
        gate[k] = cfg[k]

json.dump(gate, open("user_data/config/config_gate_backtest.json","w"),
            indent=2, ensure_ascii=False)
print("✓ 配置已更新")
