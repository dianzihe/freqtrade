#!/bin/bash
# 全币种全周期批量回测 — shell 版本（更可靠）
# 用法: bash backtest_meme_limited_v3.sh

PROJECT="F:/source/freqtrade"
VENV="$PROJECT/.venv/Scripts/python.exe"
STRAT="MemeLimitedMartingaleSpotStrategy"
STRAT_PATH="user_data/strategies"
DATA_DIR="user_data/data/gate"
CONFIG1="user_data/config/config.json"
CONFIG2="user_data/config/config_gate_backtest.json"
TIMERANGE="20260608-20260627"
OUTDIR="$PROJECT/deliverables/backtest_meme_limited_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR/results" "$OUTDIR/metrics" "$OUTDIR/html"

PAIRS=(
  "BTC/USDT" "ETH/USDT" "SOL/USDT" "XRP/USDT"
  "XCN/USDT"
  "H/USDT" "VELVET/USDT" "BEAT/USDT" "COAI/USDT"
)
TFS=("1m" "5m" "15m")

total=$(( ${#PAIRS[@]} * ${#TFS[@]} ))
cnt=0
all_metrics="[]"

echo "============================================================"
echo "Meme_限制定投_马丁 批量回测 (shell版)"
echo "  币种: ${#PAIRS[@]}  周期: ${#TFS[@]}  共: $total 个"
echo "  输出: $OUTDIR"
echo "============================================================"

for pair in "${PAIRS[@]}"; do
  for tf in "${TFS[@]}"; do
    cnt=$((cnt+1))
    safe=$(echo "$pair" | sed 's|/|_|g')
    name="${STRAT}_${safe}_${tf}"
    echo ""
    echo "[$cnt/$total] $pair $tf"

    "$VENV" -m freqtrade backtesting \
      --config "$CONFIG1" \
      --config "$CONFIG2" \
      --strategy "$STRAT" \
      --strategy-path "$STRAT_PATH" \
      --timeframe "$tf" \
      -p "$pair" \
      --datadir "$DATA_DIR" \
      --timerange "$TIMERANGE" \
      2>&1 | tee "$OUTDIR/results/${name}_stdout.txt"

    # 解析 stdout 提取指标（用 python 做解析）
    pycode="
import sys, json, re
text = open('$OUTDIR/results/${name}_stdout.txt').read()
m = {'pair':'$pair','timeframe':'$tf','total_trades':0,'win_pct':0,'tot_profit_pct':0,'max_dd_pct':0,'sharpe':0,'profit_factor':0}
lines = text.split(chr(10))
in_report = False
for ln in lines:
    if 'BACKTESTING REPORT' in ln:
        in_report = True; continue
    if in_report and ('┴' in ln or '═' in ln):
        break
    if in_report and '│' in ln and 'TOTAL' not in ln and 'Pair' not in ln:
        cols = [c.strip() for c in ln.split('│') if c.strip()]
        if len(cols)>=7:
            m['total_trades'] = int(cols[1])
            m['avg_profit_pct'] = float(cols[2])
            m['tot_profit_abs'] = float(cols[3])
            m['tot_profit_pct'] = float(cols[4])
            wi = cols[6].split()
            if len(wi)>=4: m['win_pct']=float(wi[3])
in_sum = False
for ln in lines:
    if 'SUMMARY METRICS' in ln: in_sum=True; continue
    if in_sum and '│' in ln:
        s = ln.strip()
        cols = [c.strip() for c in s.split('│')]
        if len(cols)>=3:
            key=cols[-2]; val=cols[-1]
            if 'Sharpe' in key: m['sharpe']=float(val)
            if 'Profit factor' in key: m['profit_factor']=float(val)
            if 'Max drawdown' in key:
                pm=re.search(r'([\\d.]+)%',val)
                if pm: m['max_dd_pct']=float(pm.group(1))
json.dump(m, sys.stdout)
"
    metrics=$("$VENV" -c "$pycode" 2>/dev/null)
    echo "$metrics" > "$OUTDIR/metrics/${name}_metrics.json"
    echo "   ↳ 交易=$(echo $metrics | $VENV -c 'import sys,json; print(json.load(sys.stdin)[\"total_trades\"])')"

    sleep 1
  done
done

echo ""
echo "✅ 全部完成! 输出: $OUTDIR"
echo "HTML 报告: $OUTDIR/html/index.html"
