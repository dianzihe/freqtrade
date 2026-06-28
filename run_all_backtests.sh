#!/bin/bash
# Gate 全策略回测脚本
# 4 策略 × 4 时间框架 = 16 次回测
# 3 个做多策略(spot) + 1 个做空策略(futures)

set -e

PROJECT_DIR="/f/source/freqtrade"
FREQTRADE="$PROJECT_DIR/.venv/Scripts/freqtrade"
STRATEGY_DIR="$PROJECT_DIR/user_data/strategies"
CONFIG_DIR="$PROJECT_DIR/user_data/config"
OUTPUT_DIR="$PROJECT_DIR/user_data/backtest_results/gate_full"
SUMMARY_FILE="$OUTPUT_DIR/backtest_summary.json"

mkdir -p "$OUTPUT_DIR"

ALL_COINS="BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,LTC/USDT,HYPE/USDT,XCN/USDT,IP/USDT,BAS/USDT,PEAQ/USDT,TA/USDT,H/USDT,VELVET/USDT,BEAT/USDT,COAI/USDT,ALLO/USDT,DN/USDT,STG/USDT"
FUT_COINS="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,LTC/USDT:USDT,HYPE/USDT:USDT,XCN/USDT:USDT,IP/USDT:USDT,BAS/USDT:USDT,TA/USDT:USDT,H/USDT:USDT,VELVET/USDT:USDT,BEAT/USDT:USDT,COAI/USDT:USDT,ALLO/USDT:USDT,STG/USDT:USDT"

TIMEFRAMES=("1m" "5m" "15m" "1h")

echo '[]' > "$SUMMARY_FILE"

# 辅助函数：创建配置并运行回测
run_bt() {
    local NAME="$1"
    local STRATEGY="$2"
    local MODE="$3"
    local COINS="$4"
    local TF="$5"
    local EXTRA_MAX_OPEN="$6"

    local CONFIG_FILE="$CONFIG_DIR/_bt_${NAME}_${TF}.json"
    local EXPORT_FILE="$OUTPUT_DIR/result_${NAME}_${TF}.json"

    # 构建配置
    python -c "
import json
config = {
    'max_open_trades': $EXTRA_MAX_OPEN,
    'stake_currency': 'USDT',
    'stake_amount': 50,
    'tradable_balance_ratio': 0.99,
    'fiat_display_currency': 'CNY',
    'dry_run': True,
    'dry_run_wallet': 1000,
    'trading_mode': '$MODE',
    'margin_mode': 'isolated' if '$MODE' == 'futures' else '',
    'timeframe': '$TF',
    'dataformat_ohlcv': 'feather',
    'position_adjustment_enable': True,
    'cancel_open_orders_on_exit': False,
    'use_exit_signal': True,
    'exit_profit_only': False,
    'ignore_roi_if_entry_signal': False,
    'unfilledtimeout': {'entry': 10, 'exit': 10, 'unit': 'minutes'},
    'entry_pricing': {'price_side': 'same', 'use_order_book': True, 'order_book_top': 1, 'check_depth_of_market': {'enabled': False, 'bids_to_ask_delta': 1}},
    'exit_pricing': {'price_side': 'same', 'use_order_book': True, 'order_book_top': 1},
    'order_types': {'entry': 'limit', 'exit': 'limit', 'emergency_exit': 'market', 'force_exit': 'market', 'force_entry': 'market', 'stoploss': 'market', 'stoploss_on_exchange': False},
    'order_time_in_force': {'entry': 'GTC', 'exit': 'GTC'},
    'exchange': {
        'name': 'gate',
        'key': '', 'secret': '',
        'ccxt_config': {'proxies': {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}},
        'ccxt_async_config': {'aiohttp_proxy': 'http://127.0.0.1:7890'},
        'enable_ws': False,
        'pair_whitelist': '$COINS'.split(','),
        'pair_blacklist': ['.*3L/USDT', '.*3S/USDT', '.*5L/USDT', '.*5S/USDT'] + ([] if '$MODE' == 'futures' else ['.*/USDT:USDT']),
    },
    'pairlists': [{'method': 'StaticPairList', 'allow_inactive': True}],
    'telegram': {'enabled': False, 'token': '', 'chat_id': ''},
    'api_server': {'enabled': False, 'listen_ip_address': '127.0.0.1', 'listen_port': 8082, 'verbosity': 'error', 'enable_openapi': False, 'jwt_secret_key': 'gate-bt-secret-key-minimum-32-chars-long', 'CORS_origins': [], 'username': 'x', 'password': 'x'},
    'bot_name': 'bt-${NAME}-${TF}',
    'internals': {'process_throttle_secs': 5},
}
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
"

    echo "[$(date +%H:%M:%S)] $NAME | $TF | $MODE ..."

    START_TIME=$(date +%s)
    $FREQTRADE backtesting \
        --config "$CONFIG_FILE" \
        --strategy "$STRATEGY" \
        --strategy-path "$STRATEGY_DIR" \
        --timeframe "$TF" \
        --timerange "20260608-20260627" \
        --export signals \
        --export-filename "$EXPORT_FILE" \
        2>&1 | tee "$OUTPUT_DIR/log_${NAME}_${TF}.txt"
    RC=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # 解析结果
    LOG="$OUTPUT_DIR/log_${NAME}_${TF}.txt"
    TRADES=$(grep -oP 'Trades\s*\|\s*\K\d+' "$LOG" | head -1 || echo "0")
    PROFIT_TOTAL=$(grep -oP 'Tot Profit %\s*\|\s*\K[-\d.]+' "$LOG" | head -1 || echo "N/A")
    WINRATE=$(grep -oP 'Win\s+\d+\s+\d+\s+\d+\s+\K[\d.]+' "$LOG" | tail -1 || echo "N/A")
    DRAWDOWN=$(grep -oP 'Drawdown\s*\|\s*\K[-\d.]+' "$LOG" | head -1 || echo "N/A")
    
    echo "  -> Trades: $TRADES, Profit: ${PROFIT_TOTAL}%, Win: ${WINRATE}%, DD: ${DRAWDOWN}%, Time: ${ELAPSED}s"

    # 追加到汇总
    python -c "
import json
summary = json.load(open('$SUMMARY_FILE'))
summary.append({
    'strategy': '$NAME',
    'timeframe': '$TF',
    'mode': '$MODE',
    'status': 'OK' if $RC == 0 else 'FAIL',
    'trades': '$TRADES',
    'profit_total_pct': '$PROFIT_TOTAL',
    'winrate': '$WINRATE',
    'max_drawdown': '$DRAWDOWN',
    'elapsed_sec': $ELAPSED,
})
json.dump(summary, open('$SUMMARY_FILE', 'w'), indent=2)
"
}

# ====== 运行全部回测 ======
echo "=========================================="
echo "  Gate Full Backtest - $(date)"
echo "  16 backtests total"
echo "=========================================="

# 1. 马丁_DCA (spot, 18 coins)
for TF in "${TIMEFRAMES[@]}"; do
    run_bt "马丁_DCA" "MemeMartingaleBaseStrategy" "spot" "$ALL_COINS" "$TF" 4
done

# 2. 趋势网格 (spot, 18 coins)
for TF in "${TIMEFRAMES[@]}"; do
    run_bt "趋势网格" "TrendFilteredGridStrategy" "spot" "$ALL_COINS" "$TF" 4
done

# 3. 趋势金字塔 (spot, 18 coins)
for TF in "${TIMEFRAMES[@]}"; do
    run_bt "趋势金字塔" "TrendPyramidStrategy" "spot" "$ALL_COINS" "$TF" 5
done

# 4. 顶部反转做空 (futures, 16 coins)
for TF in "${TIMEFRAMES[@]}"; do
    run_bt "顶部反转做空" "TopReversalShortStrategy" "futures" "$FUT_COINS" "$TF" 5
done

echo ""
echo "=========================================="
echo "  All backtests complete!"
echo "  Summary: $SUMMARY_FILE"
echo "=========================================="
