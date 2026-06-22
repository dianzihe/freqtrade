# Gate 1m Scalping Backtest Report

Strategy: `GateScalpMomentumStrategy`

Data: local Gate.io spot 1m OHLCV, `user_data/data/gate`

Period: `2026-06-08 01:00:00` to `2026-06-21 13:51:00`

Fee assumption from Freqtrade exchange model: `0.2000%`

## Group Summary

| Group | Pairs | Trades | Total Profit | Profit USDT | Win Rate | Max Drawdown | Avg Duration | Market Change | Profit Factor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stable large caps | BTC, ETH, SOL, XRP | 33 | -1.70% | -17.019 | 12.1% | 1.70% | 0:12:00 | 2.70% | 0.07 |
| High-volatility coins | H, VELVET, BEAT, VVV, DN | 215 | -12.04% | -120.371 | 37.7% | 12.04% | 0:04:00 | 10.98% | 0.30 |

## Stable Large Caps

| Pair | Trades | Total Profit | Win Rate | Avg Duration |
|---|---:|---:|---:|---:|
| BTC/USDT | 4 | -0.19% | 0.0% | 0:14:00 |
| XRP/USDT | 8 | -0.32% | 25.0% | 0:10:00 |
| ETH/USDT | 8 | -0.44% | 12.5% | 0:14:00 |
| SOL/USDT | 13 | -0.75% | 7.7% | 0:11:00 |

Best pair: `BTC/USDT` at `-0.19%`

Worst pair: `SOL/USDT` at `-0.75%`

## High-Volatility Coins

| Pair | Trades | Total Profit | Win Rate | Avg Duration |
|---|---:|---:|---:|---:|
| BEAT/USDT | 59 | -1.57% | 55.9% | 0:02:00 |
| VVV/USDT | 44 | -1.68% | 31.8% | 0:09:00 |
| VELVET/USDT | 33 | -2.03% | 30.3% | 0:02:00 |
| DN/USDT | 43 | -3.02% | 34.9% | 0:02:00 |
| H/USDT | 36 | -3.73% | 25.0% | 0:03:00 |

Best pair: `BEAT/USDT` at `-1.57%`

Worst pair: `H/USDT` at `-3.73%`

## Interpretation

This first scalp implementation is not profitable on the tested Gate.io sample. The stable group loses less because it trades less and avoids the sharpest stop-loss churn. The high-volatility group creates many more entries and a higher win rate, but the average win is too small relative to stop-losses and the 0.2% fee assumption.

The largest issue is exit asymmetry: ROI exits were profitable, but `stop_loss` and `scalp_momentum_fade` dominated losses. Before using this strategy live, the next iteration should either reduce low-quality entries in high-volatility coins or make exits more selective so small rebounds are not repeatedly converted into fee-adjusted losses.
