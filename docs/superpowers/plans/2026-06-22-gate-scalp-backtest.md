# Gate Scalp Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 1m spot scalp strategy and compare its Gate.io backtest performance on stable large caps versus high-volatility coins.

**Architecture:** Add one focused Freqtrade strategy under `user_data/strategies`, two Gate.io backtest configs with fixed pairlists, and one small pytest file that validates generated indicators and signal columns. Use existing local Gate.io 1m Feather data for reproducible backtests.

**Tech Stack:** Freqtrade strategy API v3, pandas, pytest, Gate.io 1m OHLCV Feather data.

---

### Task 1: Strategy Behavior Tests

**Files:**
- Create: `tests/strategy/test_gate_scalp_momentum_strategy.py`

- [ ] Write tests that instantiate the strategy, feed synthetic 1m OHLCV data, and assert indicator columns plus entry/exit tags are produced.
- [ ] Run `python -m pytest tests/strategy/test_gate_scalp_momentum_strategy.py -q` and confirm the test fails before the strategy exists.

### Task 2: Strategy Implementation

**Files:**
- Create: `user_data/strategies/GateScalpMomentumStrategy.py`

- [ ] Implement `GateScalpMomentumStrategy` with 1m timeframe, long-only spot behavior, tight ROI/stoploss, EMA/RSI/volume/ATR style filters, and momentum/timeout exits.
- [ ] Run the strategy test and confirm it passes.

### Task 3: Backtest Configs

**Files:**
- Create: `user_data/config-gate-scalp-stable-backtest.json`
- Create: `user_data/config-gate-scalp-meme-backtest.json`

- [ ] Add fixed Gate.io pairlists for `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `XRP/USDT`.
- [ ] Add fixed Gate.io pairlists for `H/USDT`, `VELVET/USDT`, `BEAT/USDT`, `VVV/USDT`, `DN/USDT`.
- [ ] Keep stake, timeframe, spot mode, and Feather data format aligned with existing Gate configs.

### Task 4: Verification And Backtests

**Files:**
- Generated output: `user_data/backtest_results/*`

- [ ] Run `python -m pytest tests/strategy/test_gate_scalp_momentum_strategy.py -q`.
- [ ] Run `python -m freqtrade backtesting --config user_data/config-gate-scalp-stable-backtest.json --strategy GateScalpMomentumStrategy --timerange 20260601-20260622`.
- [ ] Run `python -m freqtrade backtesting --config user_data/config-gate-scalp-meme-backtest.json --strategy GateScalpMomentumStrategy --timerange 20260601-20260622`.
- [ ] Summarize trades, total profit, win rate, drawdown, and best/worst pairs from the fresh outputs.
