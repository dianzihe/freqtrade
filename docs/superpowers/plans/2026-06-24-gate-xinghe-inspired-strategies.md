# Gate Xinghe-Inspired Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate conservative Gate spot and isolated USDT-perpetual strategies inspired by Xinghe's trend-filtered adaptive grid and basket-exit concepts.

**Architecture:** Two independent Freqtrade strategy classes share only small stateless indicator helpers. A dedicated Gate data configuration downloads native spot/futures candles plus futures mark and funding-rate series, while a focused runner executes the 1m/15m matrix and writes reproducible JSON/Markdown reports.

**Tech Stack:** Python 3.11+, Freqtrade strategy interface v3, pandas/numpy, pytest, PowerShell, Gate via CCXT.

---

### Task 1: Download and validate Gate market data

**Files:**
- Create: `user_data/config-gate-futures-data.json`
- Create: `scripts/download_gate_xinghe_data.ps1`
- Create: `scripts/validate_gate_xinghe_data.py`
- Test: `tests/scripts/test_validate_gate_xinghe_data.py`

- [ ] Write failing tests for filename discovery, duplicate timestamps, invalid OHLC values, and missing futures/mark/funding-rate series.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/scripts/test_validate_gate_xinghe_data.py -q` and verify failure because the validator module is absent.
- [ ] Implement pure validation helpers and a CLI that checks the eight spot and perpetual pairs at 1m/15m.
- [ ] Add conservative spot/futures download commands for BTC, ETH, SOL, XRP, BNB, DOGE, ADA, and AVAX.
- [ ] Run validator tests and confirm they pass.
- [ ] Download native spot candles and futures candles/mark/funding-rate data without erasing existing archives.
- [ ] Run the validator and save its JSON inventory.

### Task 2: Implement the conservative spot grid strategy

**Files:**
- Create: `user_data/strategies/gate_xinghe_spot_strategy.py`
- Test: `tests/strategy/test_gate_xinghe_spot_strategy.py`

- [ ] Write failing tests for indicator output, long-only entry, risk-fuse blocking, three-layer limit, widening DCA thresholds, decreasing stake sizes, and hard protections.
- [ ] Run the strategy test and verify failure because the class is absent.
- [ ] Implement EMA/ATR/RSI/volume indicators and trend-filtered pullback entry.
- [ ] Implement reserved-capital stake sizing and at most three decreasing DCA additions.
- [ ] Implement basket-profit, trend-failure, timeout, hard-stop, and protections behavior.
- [ ] Run the focused test file until green.

### Task 3: Implement the isolated futures bidirectional strategy

**Files:**
- Create: `user_data/strategies/gate_xinghe_futures_strategy.py`
- Test: `tests/strategy/test_gate_xinghe_futures_strategy.py`

- [ ] Write failing tests for long/short signals, leverage capped at 2x, funding-aware entry filtering, two-layer limit, decreasing DCA, trend-preserving additions, and risk exits.
- [ ] Run the strategy test and verify failure because the class is absent.
- [ ] Implement symmetric EMA/ATR/RSI/volume regime logic for long and short entries.
- [ ] Implement isolated-compatible 2x leverage and two decreasing adverse-price additions.
- [ ] Implement funding-cost, trend-reversal, timeout, basket-profit, hard-stop, and protections behavior.
- [ ] Run the focused test file until green.

### Task 4: Add reproducible backtest configurations and runner

**Files:**
- Create: `user_data/config-gate-xinghe-spot-backtest.json`
- Create: `user_data/config-gate-xinghe-futures-backtest.json`
- Create: `scripts/run_gate_xinghe_backtests.py`
- Test: `tests/scripts/test_run_gate_xinghe_backtests.py`

- [ ] Write failing tests for the exact four-run matrix, pair symbol conversion, chronological sample splits, result parsing, and Markdown summary generation.
- [ ] Verify the tests fail because the runner module is absent.
- [ ] Implement a subprocess runner for A/B at 1m and 15m with exported Freqtrade results.
- [ ] Add chronological full/early/middle/recent windows based on common data coverage.
- [ ] Record trades, profit, profit factor, expectancy, drawdown, duration, pair detail, side detail, exit reasons, and funding fees when present.
- [ ] Run runner tests until green.

### Task 5: Execute backtests and bias checks

**Files:**
- Generate: `user_data/backtest_results/gate_xinghe/summary.json`
- Generate: `user_data/backtest_results/gate_xinghe/report.md`
- Generate: Freqtrade backtest archives and analysis logs under `user_data/backtest_results/gate_xinghe/`

- [ ] Run the 1m/15m spot and futures full-sample backtests.
- [ ] Run early/middle/recent chronological windows where sample length supports them.
- [ ] Run `lookahead-analysis` for both strategies and both timeframes.
- [ ] Run `recursive-analysis` for both strategies and both timeframes.
- [ ] If a command fails, capture the exact command and error in the report rather than treating it as passing.

### Task 6: Final verification

**Files:**
- Modify only files created by Tasks 1-5 if verification exposes defects.

- [ ] Run focused new tests.
- [ ] Run related existing strategy tests.
- [ ] Run `ruff check` on new Python files.
- [ ] Run `git diff --check`.
- [ ] Re-run data validation after all downloads.
- [ ] Compare every design completion criterion against generated evidence.
