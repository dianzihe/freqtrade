# LOB Risk Filter Design

## Goal

Integrate the LOB-Latent-Regimes idea into Freqtrade as a live/dry-run risk filter, not as an entry signal or standalone trading strategy.

## Scope

The first version implements a lightweight detector using top-N order book snapshots:

- spread in basis points
- top-N bid and ask depth
- total depth
- depth erosion
- spread drift
- adaptive rolling threshold
- rising-edge trigger

The first version deliberately does not include HMM entropy, model retraining, or historical order book replay. Those need a separate validation pass.

## Architecture

Create a pure Python helper under `user_data/strategies/lob_risk_filter.py`. The helper owns rolling per-pair state and exposes one main API:

```python
signal = filter.update_from_orderbook(pair, orderbook)
```

The returned signal can be consumed by Freqtrade strategies to block new entries, reduce stake, reduce leverage, or trigger protective exits.

A small mixin in the same module provides Freqtrade-friendly methods:

```python
dataframe = self.populate_lob_risk(dataframe, metadata)
blocked = self.is_lob_risk_blocked(dataframe)
```

## Data Flow

1. Strategy calls `self.dp.orderbook(pair, levels)` in live/dry-run only.
2. The risk filter aggregates top-N depth and spread.
3. The filter compares short-window and long-window depth/spread behavior.
4. The filter emits a trigger only when score exceeds an adaptive threshold and is rising.
5. The strategy stores the latest signal on the latest dataframe row.

Backtesting with ordinary OHLCV data remains neutral because historical LOB snapshots are unavailable.

## Validation

Unit tests cover:

- stable order book snapshots do not trigger
- depth collapse with spread widening triggers after enough history
- malformed or empty books return a safe neutral signal
- the Freqtrade mixin adds risk columns and blocks entries only in live/dry-run style operation

