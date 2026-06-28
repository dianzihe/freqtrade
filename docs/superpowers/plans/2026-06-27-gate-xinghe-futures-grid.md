# Gate Xinghe Futures Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gate futures Freqtrade strategy inspired by `星河量化.mq5`.

**Architecture:** Create one strategy file with self-contained indicator helpers and one focused unit test file. The implementation translates Xinghe's mixed trend signal, ATR dynamic grid, and bounded DCA into Freqtrade's `IStrategy` hooks.

**Tech Stack:** Python, Freqtrade `IStrategy`, pandas, pytest.

---

### Task 1: Strategy Tests

**Files:**
- Create: `tests/strategy/test_gate_xinghe_futures_grid_strategy.py`

- [ ] Write failing tests for indicators, long/short entries, grid threshold, and DCA risk gates.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/strategy/test_gate_xinghe_futures_grid_strategy.py -q` and confirm failure because the strategy does not exist yet.

### Task 2: Strategy Implementation

**Files:**
- Create: `user_data/strategies/gate_xinghe_futures_grid_strategy.py`

- [ ] Implement `GateXingheFuturesGridStrategy`.
- [ ] Run the focused pytest command and confirm all new tests pass.
- [ ] Run `.\.venv\Scripts\python.exe -m py_compile user_data/strategies/gate_xinghe_futures_grid_strategy.py`.
