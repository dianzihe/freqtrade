# Robust Meme Volatility Risk Strategy

Code: `user_data/strategies/robust_meme_volatility_risk_strategy.py`

## Layer 1: Strategy Core

The strategy is based on the prior `MemeVolatilityGridMartingaleStrategy`, but converts it into a controlled risk system.

Entry:

- Long: close below lower Bollinger band, RSI below the reversion threshold, ATR within tradable range, volume expansion, range position not too low or too high, no extreme candle, no active LOB/HMM risk.
- Short: documented in code as the symmetric upper-band condition, but disabled by default because the provided configs are spot. To trade shorts, run a futures config and set `can_short = True` in a futures-specific subclass.

Position adjustment:

- Uses `adjust_trade_position`.
- Adds are limited by `max_dca_entries`.
- Adds use controlled equal/partial sizing via `dca_stake_multiplier`, not open-ended martingale.
- If floating loss reaches the reduce threshold, the strategy partially exits once.
- If interrupt loss is reached, no more DCA is allowed.

Exit:

- ROI and trailing stop are native Freqtrade exits.
- Signal exit: mean reversion to `bb_mid`, RSI take-profit, reversal warning, LOB/HMM risk, or extreme candle.
- Custom exit: account floating-loss liquidation, failed-DCA interruption, time stop, and LOB profit protection.

Stake sizing:

- `custom_stake_amount` uses account-equity percentage risk.
- Default risk is 1% of equity, capped by hard stop distance and notional cap.

## Layer 2: Risk System

Priority from highest to lowest:

1. Exchange availability and order execution. If the exchange is unreachable, local exits cannot execute.
2. Hard stoploss and emergency market exit.
3. Account floating-loss liquidation via `custom_exit`.
4. Failed-DCA interruption.
5. Floating-loss partial reduction.
6. Spread, funding, open-trade, and notional checks in `confirm_trade_entry`.
7. LOB/HMM risk gate.
8. Extreme candle filter.
9. Signal exits and trailing profit protection.
10. Time stop.

Pseudo-flow:

```text
on new entry:
  if cooldown active: reject
  if open trade count >= cap: reject
  if single/order notional > cap: reject
  if total open notional > cap: reject
  if spread too wide: reject
  if funding rate bad for direction: reject
  if LOB/HMM risk or extreme candle: no signal
  else allow entry

on open trade:
  if account float loss >= hard account threshold: exit
  if DCA exhausted and loss keeps widening: exit
  if loss reaches reduce threshold once: partial reduce
  if DCA threshold reached and no risk block: controlled add
  if time stop reached with no progress: exit
  if trailing/ROI/signal exit fires: exit
```

## Layer 3: Parameters

| Parameter | Meaning | Default | Suggested Range | Impact |
|---|---:|---:|---:|---|
| `max_strategy_leverage` | Strategy leverage cap | 1.0 | 1.0-3.0 | Prevents accidental over-leverage |
| `max_custom_open_trades` | Custom open-trade cap | 4 | 1-10 | Portfolio concentration |
| `max_dca_entries` | Maximum DCA adds | 3 | 0-4 | Inventory growth risk |
| `hard_stoploss_param` | Hard custom stop | 0.18 | 0.08-0.25 | Maximum single-trade loss line |
| `single_trade_risk_pct` | Equity risk per trade | 0.010 | 0.003-0.020 | Position size |
| `single_trade_notional_cap` | Max single notional | 350 | 50-5000 | Order size cap |
| `total_open_notional_cap` | Max total notional | 1400 | 100-15000 | Portfolio exposure |
| `buy_rsi_max` | Oversold entry RSI | 32 | 18-42 | Entry frequency |
| `buy_volume_ratio_min` | Volume confirmation | 1.25 | 0.8-2.5 | Liquidity confirmation |
| `buy_atr_min` | Minimum volatility | 0.006 | 0.002-0.020 | Avoid dead markets |
| `buy_atr_max` | Maximum volatility | 0.055 | 0.020-0.090 | Avoid chaos |
| `buy_range_position_min/max` | Avoid structural lows/highs | 0.10/0.75 | 0.03-0.90 | Reversion quality |
| `buy_bb_std` | Bollinger width | 2.2 | 1.6-3.2 | Signal selectivity |
| `sell_rsi_exit` | RSI profit exit | 60 | 52-78 | Exit speed |
| `sell_mean_reclaim_buffer` | BB mid reclaim buffer | 0.003 | 0-0.020 | Profit taking |
| `time_stop_candles` | Max holding candles | 360 | 60-720 | Time risk |
| `time_stop_min_profit` | Time stop threshold | -0.010 | -0.050-0.020 | Time exit strictness |
| `dca_step_1/2/3` | DCA trigger losses | -0.05/-0.10/-0.15 | -0.02--0.26 | Add timing |
| `dca_stake_multiplier` | Add size multiplier | 0.70 | 0.40-1.20 | Inventory slope |
| `floating_reduce_profit` | Partial-reduce loss | -0.110 | -0.18--0.04 | Loss de-risking |
| `account_float_loss_exit` | Account float loss exit | 0.12 | 0.06-0.25 | Hard portfolio risk |
| `max_spread_bps` | Spread entry filter | 45 bps | 5-150 | Slippage control |
| `max_funding_rate` | Funding avoidance | 0.003 | 0.0005-0.020 | Futures cost filter |
| `extreme_candle_pct` | Spike candle block | 0.090 | 0.030-0.250 | Event risk |

Structural parameters are leverage, notional caps, DCA count, hard stop, and LOB/HMM mode. Market-sensitive parameters are RSI, ATR, Bollinger width, DCA steps, exit thresholds, spread/funding limits, and time stop.

## Layer 4: Interruption Logic

- Failed DCA interruption: if DCA is exhausted and loss keeps widening past `interrupt_loss_after_dca`, `custom_exit` returns `interrupt_after_failed_dca`.
- Consecutive loss cooldown: native `StoplossGuard` plus a defensive `bot_loop_start` cooldown where available.
- Daily drawdown: native `MaxDrawdown` protection.
- Extreme volatility: entry blocked when a candle exceeds either absolute range or ATR multiple.
- Exchange/data anomalies: entry rejects when live/dry-run orderbook read fails; local stops cannot execute during exchange outage.

## Native, Config, And External Pieces

Freqtrade native:

- `stoploss`
- `custom_stoploss`
- `trailing_stop`
- `minimal_roi`
- `adjust_trade_position`
- `custom_stake_amount`
- `custom_exit`
- `confirm_trade_entry`
- `leverage`
- `protections`

Config required:

- `position_adjustment_enable` must not be disabled in config.
- For real stop protection, prefer `stoploss_on_exchange = true` where exchange support is reliable.
- Futures funding checks need futures mode and a DataProvider/exchange that exposes funding rates.
- Shorting requires futures config and a futures-specific subclass with `can_short = True`.

External or live-only:

- LOB/HMM risk filter needs live/dry-run orderbook snapshots. Ordinary OHLCV backtests do not validate it.
- Exchange outage handling needs monitoring outside the strategy.
- Order retry policy is mostly Freqtrade/exchange configuration, not strategy code.

## Safety Failure Modes

- Local `custom_stoploss` and `custom_exit` cannot execute if the bot, network, or exchange API is down.
- Market stop orders can slip heavily during flash crashes.
- `stoploss_on_exchange = false` means hard stop is bot-managed, not exchange-resident.
- Funding checks are skipped if the exchange/DataProvider does not expose funding data.
- Spread checks are skipped in backtests and only active in live/dry-run orderbook mode.
- LOB/HMM detector can be stale or unavailable during data interruptions; the strategy treats orderbook read failures as entry-blocking.
- Partial reduce through `adjust_trade_position` depends on exchange support and Freqtrade position adjustment being enabled.

