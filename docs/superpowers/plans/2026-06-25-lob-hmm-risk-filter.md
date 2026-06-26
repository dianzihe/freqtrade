# LOB HMM Risk Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-style HMM posterior entropy detector to the existing Freqtrade LOB risk filter.

**Architecture:** Extend the current pure Python `lob_risk_filter.py` module with a separate HMM detector class and lazy `hmmlearn` dependency. Keep simple mode as the default so existing strategies and tests are unchanged.

**Tech Stack:** Python, pandas, numpy, optional hmmlearn GaussianHMM, pytest.

---

### Task 1: HMM Detector Tests

**Files:**
- Modify: `tests/strategy/test_lob_risk_filter.py`

- [ ] Add tests for unavailable HMM dependency, controlled HMM posterior output, retrain cadence, and mixin HMM mode.
- [ ] Run `pytest tests/strategy/test_lob_risk_filter.py -q` and verify the new tests fail because HMM classes do not exist.

### Task 2: HMM Detector Implementation

**Files:**
- Modify: `user_data/strategies/lob_risk_filter.py`

- [ ] Add `LobHmmConfig`, HMM signal fields, `_HmmPairState`, and `LobHmmRiskFilter`.
- [ ] Implement lazy import of `hmmlearn.hmm.GaussianHMM` with safe fallback.
- [ ] Implement rolling feature construction, z-score scaling, scheduled retraining, posterior entropy, and pre-stress posterior scoring.
- [ ] Extend `apply_lob_signal_to_dataframe` with HMM columns.
- [ ] Extend `LobRiskFilterMixin` with `lob_filter_mode = "simple" | "hmm"`.
- [ ] Run `pytest tests/strategy/test_lob_risk_filter.py -q` and verify all tests pass.

### Task 3: Dependency Verification

**Files:**
- No source file changes required unless the environment lacks `hmmlearn`.

- [ ] Check `.venv` for `hmmlearn`.
- [ ] If missing, install `hmmlearn` into `.venv`.
- [ ] Verify `from hmmlearn.hmm import GaussianHMM` works in `.venv`.

### Task 4: Completion Verification

**Files:**
- Test: `tests/strategy/test_lob_risk_filter.py`
- Compile: `user_data/strategies/lob_risk_filter.py`

- [ ] Run `pytest tests/strategy/test_lob_risk_filter.py -q`.
- [ ] Run `.venv/Scripts/python.exe -m py_compile user_data/strategies/lob_risk_filter.py`.
- [ ] Report HMM mode limitations: live/dry-run only without historical LOB snapshots, model stability still needs dry-run validation.

