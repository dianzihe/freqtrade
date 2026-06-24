import pandas as pd

from user_data.strategies.market_regime_strategies import (
    MultiFactorBtcSyncStrategy,
    RiskFirstStrongTrendStrategy,
    VolatilityBreakoutMomentumStrategy,
    VolumeAtrMeanReversionStrategy,
)


def _frame(rows: int = 80) -> pd.DataFrame:
    close = pd.Series([100.0 + i * 0.1 for i in range(rows)])
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=rows, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1000.0] * rows,
        }
    )


def test_breakout_momentum_requires_breakout_volatility_and_volume() -> None:
    dataframe = _frame()
    dataframe["close"] = [100.0] * 79 + [105.0]
    dataframe["breakout_high"] = [103.0] * 80
    dataframe["atr_pct"] = [0.01] * 80
    dataframe["volume_ratio"] = [1.0] * 79 + [1.8]
    dataframe["trend_strength"] = [0.03] * 80
    strategy = VolatilityBreakoutMomentumStrategy(config={})

    result = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})

    assert result.loc[79, "enter_long"] == 1
    assert result.loc[79, "enter_tag"] == "vol_breakout_momentum"
    assert strategy.stoploss == -0.025


def test_mean_reversion_requires_lower_band_atr_and_volume_extreme() -> None:
    dataframe = _frame()
    dataframe["close"] = [100.0] * 79 + [92.0]
    dataframe["bb_lower"] = [95.0] * 80
    dataframe["bb_mid"] = [100.0] * 80
    dataframe["atr_pct"] = [0.012] * 80
    dataframe["volume_ratio"] = [1.0] * 79 + [2.5]
    dataframe["rsi"] = [50.0] * 79 + [27.0]
    strategy = VolumeAtrMeanReversionStrategy(config={})

    result = strategy.populate_entry_trend(dataframe, {"pair": "ETH/USDT"})
    exit_result = strategy.populate_exit_trend(dataframe, {"pair": "ETH/USDT"})

    assert result.loc[79, "enter_long"] == 1
    assert result.loc[79, "enter_tag"] == "volume_atr_mean_reversion"
    assert exit_result.loc[79, "exit_long"] == 0

    dataframe.loc[79, "close"] = 101.0
    exit_result = strategy.populate_exit_trend(dataframe, {"pair": "ETH/USDT"})
    assert exit_result.loc[79, "exit_long"] == 1
    assert exit_result.loc[79, "exit_tag"] == "mean_reversion_mid_exit"


def test_multifactor_strategy_requires_positive_score_and_btc_sync() -> None:
    dataframe = _frame()
    dataframe["trend_strength"] = [0.04] * 80
    dataframe["atr_pct"] = [0.01] * 80
    dataframe["volume_ratio"] = [1.3] * 80
    dataframe["btc_corr"] = [0.55] * 80
    dataframe["close"] = [100.0] * 79 + [104.0]
    dataframe["breakout_high"] = [103.0] * 80
    strategy = MultiFactorBtcSyncStrategy(config={})

    result = strategy.populate_entry_trend(dataframe, {"pair": "SOL/USDT"})

    assert result.loc[79, "factor_score"] >= 4
    assert result.loc[79, "enter_long"] == 1
    assert result.loc[79, "enter_tag"] == "multi_factor_btc_sync"


def test_risk_first_strategy_trades_only_strong_trends_with_tight_stop() -> None:
    dataframe = _frame()
    dataframe["close"] = [100.0] * 79 + [106.0]
    dataframe["ema_fast"] = [103.0] * 80
    dataframe["ema_slow"] = [101.0] * 80
    dataframe["ema_trend"] = [99.0] * 80
    dataframe["trend_strength"] = [0.055] * 80
    dataframe["atr_pct"] = [0.009] * 80
    dataframe["volume_ratio"] = [1.25] * 80
    dataframe["breakout_high"] = [105.0] * 80
    strategy = RiskFirstStrongTrendStrategy(config={})

    result = strategy.populate_entry_trend(dataframe, {"pair": "XRP/USDT"})

    assert result.loc[79, "enter_long"] == 1
    assert result.loc[79, "enter_tag"] == "risk_first_strong_trend"
    assert strategy.max_open_trades == 2
    assert strategy.stoploss == -0.015
