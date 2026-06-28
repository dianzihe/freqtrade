# -*- coding: utf-8 -*-
"""
微结构潜在状态检测策略 — 示例（演示如何使用信号模块）
=========================================================

⚠️ 这不是一个完整的可盈利策略，仅展示如何将
    微结构_信号模块.py 的预警信号融入交易逻辑。

    如果要实盘，应该把这个信号作为你自己策略中的一个
    辅助确认条件，而不是唯一的入场依据。

数据要求: 必须使用预处理的 tick 数据（含扩展列）
"""

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy

# 导入信号模块：一行代码获得四个通道 + 综合触发信号
try:
    from user_data.strategies.微结构_信号模块 import add_lob_regime_signals
except ImportError:
    from 微结构_信号模块 import add_lob_regime_signals


class LOBLatentRegimeStrategy(IStrategy):
    """
    示例策略：在信号触发时做空。

    问题（已知）：
    - 信号只告诉你"有压力"，不告诉你方向
    - 仅靠信号做交易 ≈ 仅靠 MACD 金叉做交易 —— 太单一
    - 需要结合趋势、波动率、市场状态等额外条件

    建议改写：
    把 signal_trigger 作为你主力策略的"增强入场条件"，
    而非独立策略。
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    startup_candle_count = 200
    process_only_new_candles = True

    minimal_roi = {"0": 0.08, "15": 0.05, "30": 0.03, "60": 0.01, "120": 0.0}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit", "exit": "limit",
        "stoploss": "market", "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "subplots": {
            "通道信号": {
                "ch1_vol_entropy":       {"color": "#2196F3"},
                "ch2_depth_erosion":     {"color": "#FF5722"},
                "ch3_spread_drift":      {"color": "#4CAF50"},
                "ch4_order_flow":        {"color": "#9C27B0"},
            },
            "聚合信号": {
                "composite_smooth":    {"color": "#E91E63"},
                "adaptive_threshold":  {"color": "#607D8B"},
                "signal_trigger":      {"color": "#FF0000", "type": "bar"},
            },
        },
    }

    def informative_pairs(self):
        return []

    # ---- 指标计算：一行调用信号模块 ----

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        调用信号模块，获得 signal_trigger 列。

        信号模块会新增以下列到 dataframe：
          ch1_vol_entropy, ch2_depth_erosion, ch3_spread_drift,
          ch4_order_flow, composite_smooth, adaptive_threshold,
          signal_trigger
        """
        dataframe = add_lob_regime_signals(
            dataframe,
            lookback_period=24,
            threshold_percentile=88,
            confirmation_bars=2,
        )
        return dataframe

    # ---- 入场：信号触发 + 方向确认 ----

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_short"] = 0
        dataframe["enter_long"] = 0

        trigger = dataframe["signal_trigger"] == 1
        downtrend = dataframe["close"].pct_change(5) < 0
        volume_ok = dataframe["volume"] > dataframe["volume"].rolling(20).mean()
        rsi_ok = ta.RSI(dataframe, timeperiod=14) > 28

        dataframe.loc[
            trigger & downtrend & volume_ok & rsi_ok & (dataframe["volume"] > 0),
            "enter_short"
        ] = 1

        return dataframe

    # ---- 出场 ----

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_short"] = 0
        dataframe["exit_long"] = 0

        # 连续阴线后阳线 → 压力释放
        consec_down = (dataframe["close"] < dataframe["open"]).rolling(3, min_periods=3).sum() == 3
        released = consec_down.shift(1) & (dataframe["close"] > dataframe["open"])

        # 信号消退
        faded = (
            (dataframe["composite_smooth"] < dataframe["adaptive_threshold"] * 0.5)
            & (dataframe["signal_trigger"] == 0)
        )

        # 趋势反转
        reversed_ = dataframe["close"] > ta.SMA(dataframe, timeperiod=15)

        exit_mask = (released | faded | reversed_) & (dataframe["volume"] > 0)
        dataframe.loc[exit_mask, "exit_short"] = 1

        return dataframe
