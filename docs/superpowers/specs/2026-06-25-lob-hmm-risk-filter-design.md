# LOB HMM Risk Filter Design

## Goal

Add the paper-style HMM detector to the existing LOB risk filter while keeping the current lightweight detector intact.

## Scope

The HMM version will run only in live/dry-run paths that have order book snapshots. It will not make OHLCV backtests claim historical order book coverage. Ordinary backtests remain neutral unless an external historical order book feed is added later.

## Architecture

Extend `user_data/strategies/lob_risk_filter.py` with:

- `LobHmmConfig`: extends the existing rolling detector configuration with HMM training controls.
- `LobHmmRiskFilter`: computes the same LOB features as the simple filter, trains a 3-state Gaussian HMM on a rolling window, and emits posterior entropy and pre-stress posterior channels.
- `LobRiskFilterMixin` mode selection: `lob_filter_mode = "simple"` uses the existing detector; `lob_filter_mode = "hmm"` uses the HMM detector.

`hmmlearn` is imported lazily. If it is unavailable, the HMM detector emits a safe non-triggering signal with reason `hmm_unavailable` instead of crashing a strategy.

## Data Flow

1. Strategy calls `populate_lob_risk(dataframe, metadata)`.
2. The mixin reads `self.dp.orderbook(pair, levels)` in live/dry-run mode.
3. `LobHmmRiskFilter` extracts spread, depth, imbalance, depth erosion, and spread drift.
4. After `hmm_min_samples` snapshots, the filter z-scores the rolling feature matrix and fits a Gaussian HMM.
5. Each new snapshot is transformed with the latest scaler and passed through `predict_proba`.
6. The HMM entropy and pre-stress posterior become extra channels in the existing MAX/rising-edge/adaptive-threshold detector.

## State And Retraining

Each pair has independent rolling state:

- raw spread/depth history
- feature matrix window
- score history
- trained HMM model
- scaler mean and standard deviation
- model version and retrain countdown

The HMM retrains every `hmm_retrain_interval` updates after the first fit. This keeps live operation bounded and avoids fitting on every tick.

## Output Columns

The existing LOB columns remain. HMM mode adds:

- `lob_hmm_ready`
- `lob_hmm_entropy`
- `lob_hmm_state`
- `lob_hmm_prestress_probability`
- `lob_hmm_model_version`

## Validation

Focused tests cover:

- HMM mode is safe when `hmmlearn` is unavailable.
- A controlled HMM model produces posterior entropy and state columns.
- The HMM detector retrains on schedule, not every update.
- Mixin mode selection uses HMM mode when configured.
- Existing simple-mode tests keep passing.

