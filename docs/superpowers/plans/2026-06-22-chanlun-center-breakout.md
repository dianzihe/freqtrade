# Chanlun Center Breakout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a first-pass, reproducible Chanlun center-breakout strategy for BTC/USDT, ETH/USDT, SOL/USDT, and XRP/USDT in Freqtrade.

**Architecture:** Put Chanlun signal construction in a pure pandas helper module under `user_data/strategies` so it can be unit-tested without starting Freqtrade. Keep the Freqtrade strategy as a thin adapter that adds the helper's columns and maps signals to entries/exits.

**Tech Stack:** Python, pandas, pytest, Freqtrade strategy interface, local Binance feather OHLCV data.

---

### Task 1: Chanlun Signal Core

**Files:**
- Create: `user_data/strategies/chanlun_core.py`
- Create: `tests/strategy/test_chanlun_core.py`

- [ ] **Step 1: Write failing tests for pivot, stroke, center, and signal behavior**
- [ ] **Step 2: Run `python -m pytest tests/strategy/test_chanlun_core.py -q` and confirm missing module failure**
- [ ] **Step 3: Implement minimal deterministic Chanlun helper functions**
- [ ] **Step 4: Re-run the focused tests and confirm they pass**

### Task 2: Freqtrade Strategy Adapter

**Files:**
- Create: `user_data/strategies/ChanlunCenterBreakoutStrategy.py`
- Create: `user_data/config-chanlun-stable-backtest.json`

- [ ] **Step 1: Add a strategy class with 1m timeframe and the Chanlun helper columns**
- [ ] **Step 2: Add a static-pair backtest config for BTC, ETH, SOL, and XRP**
- [ ] **Step 3: Run the focused helper tests again**
- [ ] **Step 4: Run Freqtrade environment/backtest verification; install missing dependencies only if needed**

### Task 3: Results Handoff

**Files:**
- Use generated backtest output under `user_data/backtest_results`

- [ ] **Step 1: Run a backtest over the local data window**
- [ ] **Step 2: Report pair-level results, caveats, and the exact command to reproduce**
