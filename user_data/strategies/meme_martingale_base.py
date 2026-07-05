# -*- coding: utf-8 -*-
"""
Meme 币马丁格尔期货策略基类。
子策略只需实现 populate_entry_trend / populate_exit_trend 及信号指标。
风险结构参数(加仓次数/步长)默认冻结不优化，防过拟合。
"""
import logging
from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter,
                                stoploss_from_open)
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class MemeMartingaleBaseStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    # 马丁需要动态加仓
    position_adjustment_enable = True

    # 由 custom_stoploss 全权接管，这里给一个宽松硬底
    stoploss = -0.99
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 100

    # ---- 风险结构参数：默认冻结，不进 Hyperopt ----
    dca_max_entries = IntParameter(1, 4, default=3, space="buy",
                                   optimize=False, load=True)
    dca_step_pct = DecimalParameter(0.04, 0.12, default=0.06, space="buy",
                                    decimals=3, optimize=False, load=True)
    dca_size_mult = DecimalParameter(1.2, 2.2, default=1.6, space="buy",
                                     decimals=2, optimize=False, load=True)

    # ---- 信号参数：可优化 ----
    take_profit_pct = DecimalParameter(0.03, 0.15, default=0.06, space="sell",
                                       decimals=3, optimize=True, load=True)

    # 中断线在"满仓总回撤"之上再加的余量
    DCA_INTERRUPT_EXTRA_LOSS = 0.05
    # ATR 加仓保护：波动过大时不补仓
    ATR_GUARD_MULT = 3.0

    leverage_num = 3.0

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,  # 子策略启用 OCO 时会覆盖为 False
    }

    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 布林带：均值回归核心。下轨=超卖参考，中轨=回归目标。
        bb_period, bb_std = 20, 2.0
        ma = dataframe["close"].rolling(bb_period).mean()
        sd = dataframe["close"].rolling(bb_period).std()
        dataframe["bb_mid"] = ma
        dataframe["bb_lower"] = ma - bb_std * sd
        dataframe["bb_upper"] = ma + bb_std * sd

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        # ATR 相对自身均值的倍数，用于识别"波动率突然爆炸"
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["atr"].rolling(50).mean()

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)
        # EMA 斜率：用于趋势过滤
        dataframe["ema_slope"] = (dataframe["ema_fast"] - dataframe["ema_fast"].shift(5)) / dataframe["ema_fast"].shift(5)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # 区间位置：价格在近期高低点中的相对位置 (0=底部, 1=顶部)
        high_50 = dataframe["high"].rolling(50).max()
        low_50 = dataframe["low"].rolling(50).min()
        range_span = high_50 - low_50
        dataframe["range_position"] = ((dataframe["close"] - low_50) / range_span).fillna(0.5)

        return dataframe

    # 子类实现 ↓
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ------------------------------------------------------------------
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return min(self.leverage_num, max_leverage)

    # ------------------------------------------------------------------
    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        """首仓只用总额的一部分，为后续马丁加仓预留资金。"""
        reserve_ratio = 1.0
        mult = self.dca_size_mult.value
        for i in range(1, self.dca_max_entries.value + 1):
            reserve_ratio += mult ** i
        first = proposed_stake / reserve_ratio
        if min_stake:
            first = max(first, min_stake)
        return first

    # ------------------------------------------------------------------
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """马丁加仓核心：价格每跌 dca_step_pct 补一仓，仓位按 dca_size_mult 放大。"""
        filled = trade.nr_of_successful_entries

        # 保护(1)：已达最大加仓次数
        if filled > self.dca_max_entries.value:
            return None

        # 价格层面收益（考虑方向 + 杠杆）
        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
        if trade.open_rate:
            raw_move = (current_rate - trade.open_rate) / trade.open_rate
            price_profit = -raw_move if trade.is_short else raw_move
        else:
            price_profit = current_profit / lev

        # 保护(2)：未跌到下一档加仓线
        trigger = -(self.dca_step_pct.value * filled)
        if price_profit > trigger:
            return None

        # 保护(3)：ATR 过大（异常波动/瀑布），暂停补仓
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is not None and len(df) > 0:
            atr_pct = df["atr_pct"].iloc[-1]
            if atr_pct and atr_pct > self.dca_step_pct.value * self.ATR_GUARD_MULT:
                logger.warning("[%s] ATR 过大(%.3f) 暂停加仓", trade.pair, atr_pct)
                return None

        # 保护(4)：逼近强平价则停止加仓（依赖 RiskExtMixin）
        if hasattr(self, "_liq_price_move") and hasattr(self, "_mark_price"):
            mark = self._mark_price(trade.pair, current_rate)
            if trade.open_rate:
                mm = (mark - trade.open_rate) / trade.open_rate
                mark_profit = -mm if trade.is_short else mm
                liq_move = -abs(self._liq_price_move(trade))
                if mark_profit < liq_move * getattr(self, "LIQ_SAFETY_RATIO", 0.75):
                    logger.warning("[%s] 逼近强平安全线，停止加仓 mark_profit=%.3f",
                                   trade.pair, mark_profit)
                    return None

        # 计算本次加仓金额
        try:
            first_stake = trade.orders[0].cost
        except Exception:
            first_stake = trade.stake_amount
        add_stake = first_stake * (self.dca_size_mult.value ** filled)
        if min_stake:
            add_stake = max(add_stake, min_stake)
        if add_stake > max_stake:
            return None

        logger.info("[%s] 马丁加仓#%d stake=%.4f price_profit=%.3f",
                    trade.pair, filled, add_stake, price_profit)
        return add_stake

    # ------------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:
        """中断线用 mark price 计算并被强平安全线收紧；盈利后渐进移动止盈。"""
        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)

        mark = self._mark_price(pair, current_rate) if hasattr(self, "_mark_price") else current_rate
        if trade.open_rate:
            raw_move = (mark - trade.open_rate) / trade.open_rate
            price_profit = -raw_move if trade.is_short else raw_move
        else:
            price_profit = current_profit / lev

        filled = trade.nr_of_successful_entries

        # 马丁中断：加仓耗尽 + 深跌 → 市价砍仓保命
        if filled > self.dca_max_entries.value:
            wanted = -(self.dca_step_pct.value * self.dca_max_entries.value
                       + self.DCA_INTERRUPT_EXTRA_LOSS)
            interrupt = (self._safe_interrupt_level(trade, wanted)
                         if hasattr(self, "_safe_interrupt_level") else wanted)
            if price_profit < interrupt:
                logger.warning("[%s] 马丁中断平仓 price_profit(mark)=%.3f level=%.3f",
                               pair, price_profit, interrupt)
                return 0.0001  # 贴现价 → 立即离场

        # 渐进移动止盈：盈利超阈值后锁定一半利润
        if price_profit > self.take_profit_pct.value:
            new_sl = stoploss_from_open(current_profit * 0.5, current_profit,
                                        is_short=trade.is_short, leverage=lev)
            return new_sl if (new_sl and new_sl != 0) else None

        return None