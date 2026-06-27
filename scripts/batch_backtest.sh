#!/bin/bash
# 金麒麟多周期多币种批量回测 Shell 版
# 用法: bash scripts/batch_backtest.sh

set +e
cd "F:/source/freqtrade"

EXE=".venv/Scripts/freqtrade.exe"
CFG="user_data/backtest_config.json"
STRAT_DIR="user_data/strategies"
DATA_DIR="user_data/data/gate"
OUT_DIR="user_data/backtest_results"

mkdir -p "$OUT_DIR"

TFS=("1m" "5m" "15m" "1h")
PAIRS_STABLE=("BTC/USDT" "ETH/USDT" "SOL/USDT" "XRP/USDT" "LTC/USDT" "HYPE/USDT")
PAIRS_MID=("XCN/USDT" "IP/USDT" "BAS/USDT" "PEAQ/USDT" "TA/USDT")
PAIRS_DEGENE=("H/USDT" "VELVET/USDT" "BEAT/USDT" "COAI/USDT" "ALLO/USDT" "DN/USDT" "STG/USDT")

ALL_PAIRS=("${PAIRS_STABLE[@]}" "${PAIRS_MID[@]}" "${PAIRS_DEGENE[@]}")
TOTAL=$(( ${#ALL_PAIRS[@]} * ${#TFS[@]} ))
DONE=0

run_backtest() {
    local pair="$1"
    local tf="$2"
    local strat="OptimizedGoldStrategy_${tf}"
    local pair_fn="${pair//\//_}"
    local log_file="$OUT_DIR/${strat}__${pair_fn}.log"
    local json_file="$OUT_DIR/${strat}__${pair_fn}.json"

    # 跳过已有日志且含结果的
    if grep -q "Total profit %" "$log_file" 2>/dev/null; then
        echo "  [SKIP] 已有结果: $tf $pair"
        return 0
    fi

    DONE=$(( DONE + 1 ))
    echo "[$DONE/$TOTAL] 运行中... $tf $pair"

    $EXE backtesting -c "$CFG" \
        --strategy "$strat" \
        --strategy-path "$STRAT_DIR" \
        -p "$pair" \
        --datadir "$DATA_DIR" \
        --export trades \
        --backtest-directory "$OUT_DIR" \
        > "$log_file" 2>&1

    if grep -q "Total profit %" "$log_file" 2>/dev/null; then
        local pct=$(grep "Total profit %" "$log_file" | head -1 | sed 's/.*| *\([0-9.-]*\)%.*/\1/')
        echo "  -> ${pct}%"
    else
        echo "  -> 无交易或错误"
    fi
}

echo "=== 开始批量回测 (共 $TOTAL 个组合) ==="

for tf in "${TFS[@]}"; do
    echo ""
    echo "### 时间周期: $tf ###"
    for pair in "${ALL_PAIRS[@]}"; do
        run_backtest "$pair" "$tf"
    done
done

echo ""
echo "=== 批量回测完成 ==="
echo "日志目录: $OUT_DIR"
