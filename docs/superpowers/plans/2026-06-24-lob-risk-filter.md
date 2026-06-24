# LOB Risk Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight LOB instability risk filter for Freqtrade strategies.

**Architecture:** Add a pure helper module in `user_data/strategies` with rolling per-pair state, a safe signal object, and a Freqtrade mixin. Keep it independent from the upstream research script and use it only as a risk gate.

**Tech Stack:** Python, pandas, pytest, Freqtrade strategy DataProvider orderbook access.

---

### Task 1: Risk Filter Unit Tests

**Files:**
- Create: `tests/strategy/test_lob_risk_filter.py`
- Create: `user_data/strategies/lob_risk_filter.py`

- [ ] Write tests for stable books, degraded books, malformed books, dataframe annotation, and live/dry-run mixin behavior.
- [ ] Run `pytest tests/strategy/test_lob_risk_filter.py -q` and verify it fails because the module does not exist.
- [ ] Implement `LobRiskConfig`, `LobRiskSignal`, `LobRiskFilter`, `apply_lob_signal_to_dataframe`, and `LobRiskFilterMixin`.
- [ ] Run `pytest tests/strategy/test_lob_risk_filter.py -q` and verify it passes.

### Task 2: Integration Notes

**Files:**
- Modify: `user_data/strategies/lob_risk_filter.py`

- [ ] Add concise module docs showing how an existing strategy calls `populate_lob_risk()` and uses `is_lob_risk_blocked()`.
- [ ] Run the focused tests again.

### Task 3: Verification

**Files:**
- Test: `tests/strategy/test_lob_risk_filter.py`

- [ ] Run `pytest tests/strategy/test_lob_risk_filter.py -q`.
- [ ] Run a syntax/import check for `user_data/strategies/lob_risk_filter.py`.
- [ ] Report that this is live/dry-run ready as a risk gate, while OHLCV backtests remain neutral unless historical order books are supplied.

