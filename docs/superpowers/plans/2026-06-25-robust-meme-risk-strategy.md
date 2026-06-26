# Robust Meme Risk Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete engineering-grade Freqtrade strategy based on the prior meme volatility grid winner, with layered risk controls and interruption logic.

**Architecture:** Create a new strategy file rather than modifying existing strategies. The strategy combines Bollinger/ATR mean-reversion entries, controlled DCA, LOB/HMM risk gating, custom stake sizing, custom exits, custom stoploss, leverage checks, spread/funding filters, drawdown cooldown, and emergency interruption logic.

**Tech Stack:** Freqtrade `IStrategy`, Hyperopt parameters, pandas/numpy indicators, `LobRiskFilterMixin`, pytest.

---

### Task 1: Tests First

**Files:**
- Create: `tests/strategy/test_robust_meme_volatility_risk_strategy.py`

- [ ] Test entry generation, spread/funding filters, stake cap, DCA interruption, time stop, emergency drawdown exit, and leverage clamp.
- [ ] Run `pytest tests/strategy/test_robust_meme_volatility_risk_strategy.py -q` and verify import failure before implementation.

### Task 2: Strategy Implementation

**Files:**
- Create: `user_data/strategies/robust_meme_volatility_risk_strategy.py`

- [ ] Implement full strategy code with all requested layers.
- [ ] Run focused tests until passing.

### Task 3: Verification

**Files:**
- Test: `tests/strategy/test_robust_meme_volatility_risk_strategy.py`
- Compile: `user_data/strategies/robust_meme_volatility_risk_strategy.py`

- [ ] Run focused pytest.
- [ ] Run py_compile.
- [ ] Summarize native vs config/external safeguards and known failure modes.

