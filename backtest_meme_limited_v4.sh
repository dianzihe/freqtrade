#!/bin/bash
# 全币种全周期批量回测 — 最简 shell 版本
# 用法: bash backtest_meme_limited_v4.sh

PROJECT="F:/source/freqtrade"
VENV="$PROJECT/.venv/Scripts/python.exe"
STRAT="MemeLimitedMartingaleSpotStrategy"
STRAT_PATH="user_data/strategies"
DATA_DIR="user_data/data/gate"
CONFIG1="user_data/config/config.json"
CONFIG2="user_data/config/config_gate_backtest.json"
TIMERANGE="20260608-20260627"
OUTDIR="$PROJECT/deliverables/backtest_meme_limited_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR/results" "$OUTDIR/metrics"

PAIRS=("BTC/USDT" "ETH/USDT" "SOL/USDT" "XRP/USDT" "XCN/USDT" "H/USDT" "VELVET/USDT" "BEAT/USDT" "COAI/USDT")
TFS=("1m" "5m" "15m")

total=$(( ${#PAIRS[@]} * ${#TFS[@]} ))
cnt=0

echo "=============================================================="
echo "Meme_限制定投_马丁 批量回测 (shell版)"
echo "  共: $total 个回测"
echo "  输出: $OUTDIR"
echo "=============================================================="

for pair in "${PAIRS[@]}"; do
  safe=$(echo "$pair" | sed 's|/|_|g')
  for tf in "${TFS[@]}"; do
    cnt=$((cnt+1))
    name="${STRAT}_${safe}_${tf}"
    outfile="$OUTDIR/results/${name}_stdout.txt"
    
    echo ""
    echo "[$cnt/$total] $pair $tf"
    
    "$VENV" -u -m freqtrade backtesting \
      --config "$CONFIG1" \
      --config "$CONFIG2" \
      --strategy "$STRAT" \
      --strategy-path "$STRAT_PATH" \
      --timeframe "$tf" \
      -p "$pair" \
      --datadir "$DATA_DIR" \
      --timerange "$TIMERANGE" \
      > "$outfile" 2>&1
    
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "  ✓ 完成 (rc=$rc)"
      # 解析指标
      python -c "
import sys, re, json
text = open('$outfile').read()
m = {'pair':'$pair','timeframe':'$tf','total_trades':0,'win_pct':0,'tot_profit_pct':0,'max_dd_pct':0,'sharpe':0,'profit_factor':0}
for ln in text.split(chr(10)):
    if '│' in ln and 'TOTAL' not in ln and 'Pair' not in ln:
        cols=[c.strip() for c in ln.split('│') if c.strip()]
        if len(cols)>=7 and cols[0] not in ['Pair','TOTAL']:
            m['total_trades']=int(cols[1]); m['tot_profit_pct']=float(cols[4])
    if 'Sharpe (closed' in ln:
        m['sharpe']=float(ln.split('│')[-1].strip())
    if 'Profit factor' in ln:
        m['profit_factor']=float(ln.split('│')[-1].strip())
    if 'Max drawdown' in ln:
        cell=ln.split('│')[-1].strip()
        pm=re.search(r'([\\d.]+)%',cell)
        if pm: m['max_dd_pct']=float(pm.group(1))
json.dump(m, open('$OUTDIR/metrics/${name}_metrics.json','w'), indent=2)
print('  交易='+str(m['total_trades'])+' 收益='+str(m['tot_profit_pct'])+'% 回撤='+str(m['max_dd_pct'])+'%')
" 2>/dev/null
    else
      echo "  ✗ 失败 (rc=$rc)"
    fi
    sleep 1
  done
done

echo ""
echo "✅ 全部完成! 输出: $OUTDIR"
echo "用 Python 生成 HTML 报告..."
