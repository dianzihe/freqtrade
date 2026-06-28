# Gate Xinghe Futures Grid Backtest

Date: 2026-06-27

Strategy: `GateXingheFuturesGridStrategy`

Data source: `user_data/data/gate`

Mode: Gate isolated futures, local OHLCV data, 0.05% fee, `stake_amount=50`, `dry_run_wallet=1000`, `max_open_trades=4`.

## Requested Baskets

- Stable: BTC, ETH, SOL, XRP, LTC, HYPE
- Middle: XCN, IP, BAS, PEAQ, TA
- Meme: H, VELVET, BEAT, COAI, ALLO, DN, STG

PEAQ and DN were excluded from executable futures backtests because Freqtrade reported no usable leverage tiers for `PEAQ/USDT:USDT` and `DN/USDT:USDT`. Local futures OHLCV files exist, but the current Gate futures metadata is insufficient for backtesting these two pairs.

Several pairs also lack local `mark` and `funding_rate` files. Freqtrade warned about those gaps but completed the OHLCV-based futures backtests for the remaining 16 pairs.

## Summary

| Timeframe | Timerange | Pairs | Trades | Profit USDT | Profit % | Max DD | Winrate | Profit Factor | SQN | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | 2026-06-18 to 2026-06-24 | 16 | 799 | -137.026 | -13.70% | 14.42% | 31.7% | 0.50 | -2.01 | 0.045 |
| 5m | 2026-06-08 20:00 to 2026-06-27 | 16 | 483 | 14.931 | 1.49% | 3.96% | 34.8% | 1.12 | 0.41 | 0.681 |

## Basket Results

| Timeframe | Basket | Trades | Profit USDT | Profit % | Winrate |
|---|---|---:|---:|---:|---:|
| 1m | Stable | 359 | -8.128 | -0.81% | 26.7% |
| 1m | Middle | 191 | -20.898 | -2.09% | 26.7% |
| 1m | Meme | 249 | -108.001 | -10.80% | 42.6% |
| 5m | Stable | 372 | -43.759 | -4.38% | 29.0% |
| 5m | Middle | 88 | 38.436 | 3.84% | 50.0% |
| 5m | Meme | 23 | 20.253 | 2.03% | 69.6% |

## Pair Results

### 1m

| Pair | Trades | Profit USDT | Profit % | Winrate |
|---|---:|---:|---:|---:|
| ALLO/USDT:USDT | 11 | 0.265 | 0.03% | 63.6% |
| BAS/USDT:USDT | 90 | -7.792 | -0.78% | 30.0% |
| BEAT/USDT:USDT | 56 | -24.578 | -2.46% | 46.4% |
| BTC/USDT:USDT | 33 | -0.039 | -0.00% | 36.4% |
| COAI/USDT:USDT | 26 | -10.266 | -1.03% | 26.9% |
| ETH/USDT:USDT | 64 | -0.476 | -0.05% | 28.1% |
| H/USDT:USDT | 65 | -44.448 | -4.44% | 49.2% |
| HYPE/USDT:USDT | 116 | -4.837 | -0.48% | 27.6% |
| IP/USDT:USDT | 42 | -2.142 | -0.21% | 33.3% |
| LTC/USDT:USDT | 79 | -1.605 | -0.16% | 27.8% |
| SOL/USDT:USDT | 0 | 0.000 | 0.00% | 0.0% |
| STG/USDT:USDT | 28 | -6.501 | -0.65% | 35.7% |
| TA/USDT:USDT | 52 | -10.962 | -1.10% | 17.3% |
| VELVET/USDT:USDT | 63 | -22.473 | -2.25% | 38.1% |
| XCN/USDT:USDT | 7 | -0.001 | -0.00% | 14.3% |
| XRP/USDT:USDT | 67 | -1.170 | -0.12% | 17.9% |

### 5m

| Pair | Trades | Profit USDT | Profit % | Winrate |
|---|---:|---:|---:|---:|
| ALLO/USDT:USDT | 3 | 2.983 | 0.30% | 100.0% |
| BAS/USDT:USDT | 12 | 14.694 | 1.47% | 75.0% |
| BEAT/USDT:USDT | 4 | 3.724 | 0.37% | 100.0% |
| BTC/USDT:USDT | 92 | -4.904 | -0.49% | 26.1% |
| COAI/USDT:USDT | 1 | 0.554 | 0.06% | 100.0% |
| ETH/USDT:USDT | 90 | -4.475 | -0.45% | 22.2% |
| H/USDT:USDT | 0 | 0.000 | 0.00% | 0.0% |
| HYPE/USDT:USDT | 28 | -11.952 | -1.20% | 53.6% |
| IP/USDT:USDT | 23 | 32.452 | 3.25% | 52.2% |
| LTC/USDT:USDT | 74 | -13.317 | -1.33% | 29.7% |
| SOL/USDT:USDT | 0 | 0.000 | 0.00% | 0.0% |
| STG/USDT:USDT | 11 | 6.343 | 0.63% | 36.4% |
| TA/USDT:USDT | 24 | -3.316 | -0.33% | 37.5% |
| VELVET/USDT:USDT | 4 | 6.649 | 0.66% | 100.0% |
| XCN/USDT:USDT | 29 | -5.394 | -0.54% | 48.3% |
| XRP/USDT:USDT | 88 | -9.111 | -0.91% | 30.7% |

## Takeaways

- The 1m window is not usable as-is: total return was -13.70%, with the meme basket contributing most of the loss.
- The 5m window was positive at +1.49%, but the p-value is 0.681 and SQN is 0.41, so this is not strong evidence of a durable edge.
- Stable pairs were weak on both windows. The 5m profit came from IP, BAS, and a small number of meme trades, not from broad market robustness.
- Before promotion to dry-run, the strategy needs parameter tightening or basket-specific filters, especially for stable majors and the 1m window.

## Artifacts

- 1m result: `user_data/backtest_results/gate_xinghe_futures_grid_20260627/backtest-result-2026-06-27_23-31-15.zip`
- 5m result: `user_data/backtest_results/gate_xinghe_futures_grid_20260627/backtest-result-2026-06-27_23-32-46.zip`
