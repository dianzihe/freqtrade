# LOB Latent Regime v7 Freqtrade Design

## Goal

Create a standalone Freqtrade strategy adapted from `Experiments/v7.py` that can be loaded and backtested without depending on the existing Chinese-named microstructure modules.

## Scope

The strategy keeps the deployable parts of v7: channel scoring, MAX aggregation, rising-edge detection, adaptive thresholding, multi-bar confirmation, and cooldown deduplication. It intentionally excludes synthetic LOB generation, HMM training, publication plots, threshold sweeps, and statistical evaluation code because those belong to offline research, not Freqtrade runtime.

## Strategy File

Create `user_data/strategies/lob_latent_regime_v7_strategy.py` with class `LOBLatentRegimeV7Strategy`.

The file will:
- Use standard OHLCV columns.
- Prefer optional L2 and flow columns when present: `bids`, `asks`, `bid_depth_10`, `ask_depth_10`, `best_bid`, `best_ask`, `buy_volume`, and `sell_volume`.
- Add indicator columns prefixed with `lob_`.
- Support shorts with `can_short = True`.

## Signal Design

The strategy computes five v7-style channels:
- `lob_entropy`: volatility-structure entropy proxy.
- `lob_prestress_proxy`: pre-stress proxy from volatility and spread pressure.
- `lob_spread_drift`: widening spread pressure.
- `lob_depth_erosion`: declining book depth pressure.
- `lob_ofi_momentum`: absolute order-flow imbalance momentum.

It then creates:
- `lob_score`: smoothed MAX channel score.
- `lob_score_delta`: rising edge over a small lag.
- `lob_threshold`: trailing rolling percentile threshold.
- `lob_trigger`: confirmed and cooldown-filtered warning trigger.

## Trading Logic

`lob_trigger` is a warning signal, not a complete trading edge. Short entries require:
- `lob_trigger == 1`.
- Short-term downside momentum.
- Price below a slow EMA.
- Nonzero volume.

Long entries remain disabled by default because the v7 signal detects stress, not bullish direction.

Exits use signal fade, price recovery above EMA, or standard ROI/stoploss/trailing stop.

## Testing

Add `tests/strategy/test_lob_latent_regime_v7_strategy.py`.

Tests cover:
- Strategy indicator columns are created.
- L2 book snapshots drive depth erosion and spread drift.
- Multi-trigger confirmation plus cooldown deduplicates triggers.
- A stressed, bearish final candle produces a short entry.

## Acceptance

The strategy is acceptable when:
- Focused tests pass.
- Strategy loading tests pass for the new class.
- No existing user changes are reverted.
