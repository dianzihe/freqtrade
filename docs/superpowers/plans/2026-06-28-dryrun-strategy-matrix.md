# Dryrun Strategy Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run five selected Gate strategies in dry-run mode across 1m and 5m timeframes and expose a one-minute refreshing profitability dashboard.

**Architecture:** Keep the existing sqlite-backed static dashboard pattern, but drive it from a 10-item strategy matrix. Each bot gets an isolated config, DB, API port, and log file while the dashboard aggregates closed/open trades from sqlite.

**Tech Stack:** PowerShell launcher scripts, Python sqlite/html dashboard generator, Freqtrade JSON configs, pytest.

---

### Task 1: Dashboard and Config Tests

**Files:**
- Modify: `tests/commands/test_meme_dryrun_dashboard.py`

- [ ] Add tests that require 10 configured bots, unique ports/DBs, dry-run enabled, 100U wallets, 1m/5m coverage, and 60 second HTML refresh.
- [ ] Run `pytest tests/commands/test_meme_dryrun_dashboard.py -q` and confirm the new expectations fail before implementation.

### Task 2: Dashboard Matrix

**Files:**
- Modify: `scripts/meme_dryrun_dashboard.py`

- [ ] Replace the four-bot registry with a 10-bot strategy matrix.
- [ ] Summarize sqlite trades into equity, profit, profit rate, closed/open trades, wins/losses, win rate, and recent trades.
- [ ] Render a Chinese dashboard grouped by timeframe and refreshing every 60 seconds.

### Task 3: Dry-Run Configs and Launch Scripts

**Files:**
- Modify/Create: `user_data/config/config-dryrun-*.json`
- Modify: `scripts/start_meme_dryrun.ps1`
- Modify: `scripts/watch_meme_dryrun_dashboard.ps1`

- [ ] Create isolated configs for five strategies across 1m and 5m.
- [ ] Update the launcher to start every config in the matrix and write per-bot logs.
- [ ] Update the watcher to regenerate the dashboard every 60 seconds.

### Task 4: Verification

**Files:**
- Test: `tests/commands/test_meme_dryrun_dashboard.py`

- [ ] Run the focused pytest file with workspace-local temp variables.
- [ ] Compile the dashboard script.
- [ ] Generate the dashboard HTML once and inspect that it contains 10 cards and 60 second refresh text.
