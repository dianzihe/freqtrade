# LOB Latent Regime v7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Freqtrade strategy adapted from the v7 LOB latent-regime detector.

**Architecture:** The strategy file owns its own lightweight v7-style feature extraction and signal trigger logic. It does not import existing Chinese-named strategy modules, which keeps strategy loading and backtesting simpler.

**Tech Stack:** Python, pandas, numpy, Freqtrade `IStrategy`, pytest.

---

### Task 1: Focused Tests

**Files:**
- Create: `tests/strategy/test_lob_latent_regime_v7_strategy.py`
- Create later: `user_data/strategies/lob_latent_regime_v7_strategy.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `LOBLatentRegimeV7Strategy`, build synthetic OHLCV/L2 data, and assert:
- indicator columns exist,
- L2 depth/spread deterioration produces high channel values,
- cooldown keeps triggers separated,
- final stressed bearish candle enters short.

- [ ] **Step 2: Run focused tests to verify RED**

Run:

```powershell
$env:TEMP=(Resolve-Path .tmp).Path; $env:TMP=$env:TEMP; .\.venv\Scripts\pytest.exe tests\strategy\test_lob_latent_regime_v7_strategy.py -q -p no:cacheprovider
```

Expected: import failure because `user_data.strategies.lob_latent_regime_v7_strategy` does not exist.

### Task 2: Strategy Implementation

**Files:**
- Create: `user_data/strategies/lob_latent_regime_v7_strategy.py`
- Test: `tests/strategy/test_lob_latent_regime_v7_strategy.py`

- [ ] **Step 1: Implement `LOBLatentRegimeV7Strategy`**

Add an `IStrategy` class with:
- `INTERFACE_VERSION = 3`
- `can_short = True`
- `timeframe = "5m"`
- conservative ROI, stoploss, and trailing stop settings
- helper methods for L2 parsing, normalization, cooldown filtering, and channel calculation

- [ ] **Step 2: Run focused tests to verify GREEN**

Run:

```powershell
$env:TEMP=(Resolve-Path .tmp).Path; $env:TMP=$env:TEMP; .\.venv\Scripts\pytest.exe tests\strategy\test_lob_latent_regime_v7_strategy.py -q -p no:cacheprovider
```

Expected: all tests pass.

### Task 3: Loading Verification

**Files:**
- Existing: `tests/strategy/test_strategy_loading.py`
- Created: `user_data/strategies/lob_latent_regime_v7_strategy.py`

- [ ] **Step 1: Run strategy loading tests**

Run:

```powershell
$env:TEMP=(Resolve-Path .tmp).Path; $env:TMP=$env:TEMP; .\.venv\Scripts\pytest.exe tests\strategy\test_strategy_loading.py -q -p no:cacheprovider
```

Expected: loading tests pass or expose unrelated pre-existing dirty-worktree failures.

- [ ] **Step 2: Summarize results**

Report created files and exact verification outcomes.
