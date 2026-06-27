# -*- coding: utf-8 -*-
"""
Gate 剥头皮动量策略

基于回踩低点的动态止损 + 分批减仓（R 倍数）的动量剥头皮策略，5 分钟周期，追踪止损让利润奔跑。
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
from freqtrade.exchange import timeframe_to_prev_date


class GateScalpMomentumStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False

    timeframe = "5m"
    startup_candle_count = 100
    process_only_new_candles = True

    # 让利润奔跑：ROI 只做一个较高的总上限，主要靠 R 倍数减仓 + 追踪止损离场
    minimal_roi = {"0": 0.06}

    # 这是"最大允许亏损"安全网，必须比 custom_stoploss 计算出的回踩低点止损更宽，
    # 否则会提前被截断。真正的止损由 custom_stoploss 按回踩低点动态决定。
    stoploss = -0.05

    # 剩余底仓（2R 之后）交给追踪止损去吃趋势
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.020
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True            # 启用按回踩低点的动态止损
    position_adjustment_enable = True     # 启用分批减仓（1R / 1.5R）

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ---------- 工具函数 ----------
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    @staticmethod
    def _atr(dataframe: DataFrame, period: int = 14) -> pd.Series:
        high_low = dataframe["high"] - dataframe["low"]
        high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
        low_close = (dataframe["low"] - dataframe["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    # ---------- 15m 趋势过滤（多周期）----------
    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=21, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        # 15m 趋势向上标志
        dataframe["trend_up"] = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["close"] > dataframe["ema_slow"])
        ).astype(int)
        return dataframe

    # ---------- 基础周期 5m 指标 ----------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema50"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20, min_periods=5).mean()
        dataframe["atr"] = self._atr(dataframe, 14)
        dataframe["atr_pct"] = (dataframe["atr"] / dataframe["close"]).replace([np.inf, -np.inf], 0)

        # 日内锚定 VWAP（每个交易日重置）
        tp = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        tpv = tp * dataframe["volume"]
        day = dataframe["date"].dt.date
        cum_tpv = tpv.groupby(day).cumsum()
        cum_vol = dataframe["volume"].groupby(day).cumsum().replace(0, np.nan)
        dataframe["vwap"] = (cum_tpv / cum_vol).bfill()

        # 箱体（不含当前根，避免前视）
        dataframe["box_high"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["box_low"] = dataframe["low"].rolling(20).min().shift(1)
        dataframe["box_range_pct"] = (
            (dataframe["box_high"] - dataframe["box_low"]) / dataframe["box_low"]
        ).replace([np.inf, -np.inf], np.nan)

        # 止损参考位：近 5 根最低点下方一点点（回踩低点 / 插针低点）
        dataframe["swing_low"] = dataframe["low"].rolling(5, min_periods=2).min()
        dataframe["stop_level"] = dataframe["swing_low"] * 0.998

        # 当根 K 线结构特征
        dataframe["bull_candle"] = (dataframe["close"] > dataframe["open"]).astype(int)
        body_low = dataframe[["open", "close"]].min(axis=1)
        rng = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        dataframe["lower_wick_ratio"] = ((body_low - dataframe["low"]) / rng).fillna(0)
        dataframe["range_pct"] = (rng / dataframe["close"]).fillna(0)
        return dataframe

    # ---------- 入场：三套模板 ----------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        # 通用波动率过滤
        volatility_ok = dataframe["atr_pct"].between(0.0008, 0.04)

        # ============ 模板1：VWAP / EMA20 回踩继续 ============
        # 5m+15m 趋势向上；价格回踩 VWAP 或 EMA20（下影触及），收回站上；出现止跌小阳线
        trend_up = (
            (dataframe["trend_up_15m"] == 1)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["ema20"] > dataframe["ema50"])
        )
        touched_support = (
            (dataframe["low"] <= dataframe["ema20"] * 1.002)
            | (dataframe["low"] <= dataframe["vwap"] * 1.002)
        )
        held_above = (
            (dataframe["close"] >= dataframe["ema20"] * 0.999)
            | (dataframe["close"] >= dataframe["vwap"] * 0.999)
        )
        stop_falling = (  # 止跌小结构：收阳 + 不破前低 + RSI 拐头
            (dataframe["bull_candle"] == 1)
            & (dataframe["low"] >= dataframe["low"].shift(1) * 0.999)
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))
        )
        vol_ok_1 = dataframe["volume"] > dataframe["volume_mean"]

        tmpl1 = trend_up & touched_support & held_above & stop_falling & vol_ok_1 & volatility_ok
        dataframe.loc[tmpl1, ["enter_long", "enter_tag"]] = (1, "t1_vwap_pullback")

        # ============ 模板2：箱体突破回踩 ============
        # 前期窄幅箱体；放量突破上沿；不追第一根大阳，等回踩上沿不破、重新放量
        tight_box = dataframe["box_range_pct"] < 0.03
        broke_recently = (
            (dataframe["close"].shift(1) > dataframe["box_high"])
            | (dataframe["high"].rolling(3).max().shift(1) > dataframe["box_high"])
        )
        retest_hold = (  # 回踩到上沿附近但收盘仍站在上沿之上
            (dataframe["low"] <= dataframe["box_high"] * 1.004)
            & (dataframe["close"] > dataframe["box_high"])
        )
        revolume = dataframe["volume"] > dataframe["volume_mean"] * 1.1

        tmpl2 = tight_box & broke_recently & retest_hold & revolume & volatility_ok
        dataframe.loc[tmpl2, ["enter_long", "enter_tag"]] = (1, "t2_box_retest")

        # ============ 模板3：插针反转（小仓试多）============
        # 急跌出现长下影；收回关键价位；放量；只吃反弹
        had_drop = dataframe["close"].shift(1) < dataframe["close"].shift(4)
        long_lower_wick = dataframe["lower_wick_ratio"] > 0.5
        recovered = (
            (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > (dataframe["high"] + dataframe["low"]) / 2)
        )
        vol_spike = dataframe["volume"] > dataframe["volume_mean"] * 1.5
        big_range = dataframe["range_pct"] > 0.006  # 确实是一根有波动的针

        tmpl3 = had_drop & long_lower_wick & recovered & vol_spike & big_range
        dataframe.loc[tmpl3, ["enter_long", "enter_tag"]] = (1, "t3_wick_reversal")

        return dataframe

    # ---------- 出场信号（兜底，主力靠 R 减仓 + 追踪止损）----------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        # 爆量见顶 / 急速冲高回落，主动了结
        exhaustion = (
            (dataframe["rsi"] > 80)
            & (dataframe["volume"] > dataframe["volume_mean"] * 2.0)
        )
        dataframe.loc[exhaustion, ["exit_long", "exit_tag"]] = (1, "exhaustion")
        return dataframe

    # ---------- 仓位：插针试多用小仓 ----------
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        if entry_tag == "t3_wick_reversal":
            stake = proposed_stake * 0.5  # 插针反转风险高，只用一半仓位试多
            if min_stake is not None:
                stake = max(stake, min_stake)
            return stake
        return proposed_stake

    # ---------- 动态止损：放在回踩低点 / 插针低点下方 ----------
    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, **kwargs):
        stop_price = trade.get_custom_data("stop_price")

        if stop_price is None:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return None
            entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            candle = df.loc[df["date"] == entry_date]
            if candle.empty:
                return None

            stop_price = float(candle["stop_level"].iloc[0])
            # 安全校验：止损必须低于开仓价，且距离不要超过硬止损上限
            min_allowed = trade.open_rate * (1 + self.stoploss)  # 例如 open*0.95
            if not np.isfinite(stop_price) or stop_price >= trade.open_rate:
                stop_price = trade.open_rate * 0.988
            stop_price = max(stop_price, min_allowed)

            trade.set_custom_data("stop_price", stop_price)
            r_pct = (trade.open_rate - stop_price) / trade.open_rate
            trade.set_custom_data("r_pct", float(r_pct))

        # 1R 减仓后把止损抬到保本，锁住风险
        if trade.get_custom_data("tp1_done"):
            stop_price = max(stop_price, trade.open_rate)

        return stoploss_from_absolute(
            stop_price, current_rate, is_short=False, leverage=trade.leverage
        )

    # ---------- R 倍数分批减仓：1R 减仓，1.5R 再减仓 ----------
    def adjust_trade_position(self, trade, current_time, current_rate,
                            current_profit, min_stake, max_stake, **kwargs):
        r = trade.get_custom_data("r_pct")
        if not r or r <= 0:
            return None

        # 到 1R：减约 40% 仓位
        if not trade.get_custom_data("tp1_done") and current_profit >= 1.0 * r:
            trade.set_custom_data("tp1_done", True)
            return -(trade.stake_amount * 0.4)

        # 到 1.5R：再减约 50% 剩余仓位（剩下底仓让追踪止损去跑）
        if (
            trade.get_custom_data("tp1_done")
            and not trade.get_custom_data("tp2_done")
            and current_profit >= 1.5 * r
        ):
            trade.set_custom_data("tp2_done", True)
            return -(trade.stake_amount * 0.5)

        return None