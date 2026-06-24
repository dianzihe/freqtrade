# Trend Pyramid Anti-Martingale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and backtest a Freqtrade strategy that follows trends and adds only to profitable positions using anti-martingale pyramiding, with both 1m and 15m Gate data coverage.

**Architecture:** Maintain one standalone user strategy that trades Gate spot data on the configured candle size, filters for strong trends, enters on breakouts, blocks entries during excessive volatility or sharp selloffs, and uses `adjust_trade_position` for winner-only pyramiding. Add focused pytest coverage for indicator/signal columns, circuit-breaker entry blocking, and position-adjustment behavior, plus separate Gate backtest configs for 1m and 15m windows.

**Tech Stack:** Python, pandas, Freqtrade `IStrategy`, pytest, Gate feather OHLCV data under `user_data/data/gate`.

---

### Task 1: Strategy Behavior Tests

**Files:**
- Create: `tests/strategy/test_trend_pyramid_anti_martingale_strategy.py`
- Create later: `user_data/strategies/trend_pyramid_anti_martingale_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.trend_pyramid_anti_martingale_strategy import (
    TrendPyramidAntiMartingaleStrategy,
)


def _trend_frame(rows: int = 260) -> pd.DataFrame:
    base = pd.Series(range(rows), dtype="float64")
    close = 100 + base * 0.08
    return pd.DataFrame(
        {
    "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1000 + base * 3,
        }
    )


def test_populate_indicators_adds_trend_pyramid_columns() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})

    dataframe = strategy.populate_indicators(_trend_frame(), {"pair": "BTC/USDT"})

    expected_columns = {
        "ema_fast",
        "ema_slow",
        "ema_trend",
        "atr_pct",
        "breakout_high",
        "trend_strength",
        "volume_ratio",
    }
    assert expected_columns.issubset(dataframe.columns)
    assert dataframe["atr_pct"].fillna(0).ge(0).all()


def test_entry_signal_requires_breakout_volume_and_trend() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})
    dataframe = _trend_frame()
    dataframe["ema_fast"] = [100.0] * 259 + [125.0]
    dataframe["ema_slow"] = [99.0] * 259 + [120.0]
    dataframe["ema_trend"] = [98.0] * 259 + [116.0]
    dataframe["trend_strength"] = [0.01] * 259 + [0.04]
    dataframe["breakout_high"] = [130.0] * 259 + [124.0]
    dataframe["volume_ratio"] = [1.0] * 259 + [1.6]
    dataframe["atr_pct"] = [0.006] * 260

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[259, "enter_long"] == 1
    assert result.loc[259, "enter_tag"] == "trend_breakout"


def test_pyramid_adds_only_to_profitable_trades() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})
    trade = SimpleNamespace(nr_of_successful_entries=1, stake_amount=100.0)

    losing = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=98.0,
        current_profit=-0.02,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=98.0,
        current_exit_rate=98.0,
        current_entry_profit=-0.02,
        current_exit_profit=-0.02,
    )
    winning = strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        current_rate=106.0,
        current_profit=0.065,
        min_stake=10.0,
        max_stake=1000.0,
        current_entry_rate=106.0,
        current_exit_rate=106.0,
        current_entry_profit=0.065,
        current_exit_profit=0.065,
    )

    assert losing is None
    assert winning == (45.0, "pyramid_add_1")


def test_circuit_breaker_blocks_entry_after_sharp_drop() -> None:
    strategy = TrendPyramidAntiMartingaleStrategy(config={})
    dataframe = _trend_frame()
    dataframe["ema_fast"] = [100.0] * 259 + [119.0]
    dataframe["ema_slow"] = [99.0] * 259 + [114.0]
    dataframe["ema_trend"] = [98.0] * 259 + [110.0]
    dataframe["trend_strength"] = [0.01] * 259 + [0.04]
    dataframe["breakout_high"] = [130.0] * 259 + [120.0]
    dataframe["volume_ratio"] = [1.0] * 259 + [1.6]
    dataframe["atr_pct"] = [0.006] * 260
    dataframe["momentum_3"] = [0.0] * 259 + [0.02]
    dataframe["rsi"] = [55.0] * 260
    dataframe["market_circuit_breaker"] = [0] * 259 + [1]

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[259, "enter_long"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests\strategy\test_trend_pyramid_anti_martingale_strategy.py -q`
Expected: FAIL because `market_circuit_breaker` is not yet part of the entry logic.

### Task 2: Implement Strategy

**Files:**
- Create: `user_data/strategies/trend_pyramid_anti_martingale_strategy.py`

- [ ] **Step 1: Implement indicators and trend breakout entry**

Use EMA 20/60/120, ATR percentage, 72-candle breakout high, volume ratio, and trend-strength filter.

- [ ] **Step 2: Implement circuit-breaker filters**

Create `market_circuit_breaker` from excessive ATR percentage, a sharp 6-candle drawdown, or very weak RSI with negative momentum. Require it to be off before any new long entry.

- [ ] **Step 3: Implement winner-only pyramiding**

Use `position_adjustment_enable = True`, `max_entry_position_adjustment = 2`, profit thresholds `[0.055, 0.105]`, and add sizes `[0.45, 0.30]` times the original stake amount.

- [ ] **Step 4: Implement exits**

Exit when price loses EMA slow, momentum fades below RSI 45, or price reaches an exhaustion candle. Keep a protective static stoploss and enable trailing stop to protect late trend profits.

- [ ] **Step 5: Run tests**

Run: `.venv\Scripts\python -m pytest tests\strategy\test_trend_pyramid_anti_martingale_strategy.py -q`
Expected: PASS.

### Task 3: Backtest Configuration

**Files:**
- Modify: `user_data/config-trend-pyramid-backtest.json`
- Create: `user_data/config-trend-pyramid-1m-backtest.json`
- Create: `user_data/config-trend-pyramid-15m-backtest.json`

- [ ] **Step 1: Create 1m config**

Use exchange `gate`, strategy `TrendPyramidAntiMartingaleStrategy`, timeframe `1m`, feather data, static whitelist of the Gate pairs that also have 15m data: `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `XRP/USDT`, `BEAT/USDT`, `DN/USDT`, `H/USDT`, `VELVET/USDT`, `VVV/USDT`.

- [ ] **Step 2: Create 15m config**

Use the same exchange, strategy, stake settings, pair list, and data directory with timeframe `15m`.

- [ ] **Step 3: Run initial 1m backtest**

Run: `.venv\Scripts\python -m freqtrade backtesting --config user_data\config-trend-pyramid-1m-backtest.json --datadir user_data\data\gate --strategy TrendPyramidAntiMartingaleStrategy --timerange 20260601-`

- [ ] **Step 4: Run initial 15m backtest**

Run: `.venv\Scripts\python -m freqtrade backtesting --config user_data\config-trend-pyramid-15m-backtest.json --datadir user_data\data\gate --strategy TrendPyramidAntiMartingaleStrategy --timerange 20260601-`

- [ ] **Step 5: Tune if results are weak**

If no trades or poor profitability, inspect the backtest summaries. Prefer conservative changes first: lower breakout lookback from 72 to 48, lower volume ratio from 1.35 to 1.15, relax trend strength from 0.008 to 0.004 for 15m, or raise profit-add thresholds if drawdown expands.

### Task 4: Final Verification

**Files:**
- Inspect: `user_data/backtest_results/.last_result.json`
- Inspect: generated backtest result JSON.

- [ ] **Step 1: Run targeted tests**

Run: `.venv\Scripts\python -m pytest tests\strategy\test_trend_pyramid_anti_martingale_strategy.py -q`
Expected: all tests pass.

- [ ] **Step 2: Run final 1m backtest**

Run: `.venv\Scripts\python -m freqtrade backtesting --config user_data\config-trend-pyramid-1m-backtest.json --datadir user_data\data\gate --strategy TrendPyramidAntiMartingaleStrategy --timerange 20260601-`
Expected: command exits successfully and prints a strategy summary.

- [ ] **Step 3: Run final 15m backtest**

Run: `.venv\Scripts\python -m freqtrade backtesting --config user_data\config-trend-pyramid-15m-backtest.json --datadir user_data\data\gate --strategy TrendPyramidAntiMartingaleStrategy --timerange 20260601-`
Expected: command exits successfully and prints a strategy summary.

- [ ] **Step 4: Report evidence**

Summarize trade count, total profit, win rate, drawdown, and whether tuning was applied for both 1m and 15m.
