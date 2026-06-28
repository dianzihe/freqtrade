# Gate Xinghe Futures Grid Design

## Goal

Build a Freqtrade futures strategy inspired by `星河量化.mq5`, adapted for Gate perpetual markets.

## Scope

The first version implements a directional long/short futures strategy with ATR-spaced DCA. It does not copy MT5 hedge-mode behavior or the complex "rescue" system. Freqtrade will manage one trade direction per trade, so the strategy exits on trend reversal or risk conditions instead of opening separate rescue hedges.

## Strategy Model

- Platform target: Gate futures.
- Freqtrade class: `GateXingheFuturesGridStrategy`.
- Trading direction: `can_short = True`.
- Timeframe: `1m`, with enough startup candles for trend and volatility indicators.
- Entry: wait for a mixed trend signal. A bullish signal opens long; a bearish signal opens short.
- Trend signal: majority vote from EMA trend, ATR expansion candle direction, and rolling volatility breakout.
- DCA: use `adjust_trade_position` for losing trades only. Additions require loss beyond an ATR-derived grid threshold and a maximum number of successful entries.
- Stake sizing: reserve capital on first entry; DCA stake uses a configured multiplier ladder and is capped by `max_stake`.
- Exits: average-profit target, trend reversal, volatility risk fuse, and hard stoploss.

## Risk Controls

- Maximum DCA levels are limited.
- DCA is blocked after deep loss beyond a fuse threshold.
- Entry is blocked when ATR percent is outside configured bounds.
- The strategy avoids assuming `Order.stake_amount`; when order-derived sizing is needed, it uses `safe_cost`.

## Verification

Add focused unit tests for:

- indicator columns and bounded signals;
- long and short entry behavior;
- ATR grid threshold calculation;
- DCA acceptance after enough adverse movement;
- DCA rejection under risk-fuse conditions.
