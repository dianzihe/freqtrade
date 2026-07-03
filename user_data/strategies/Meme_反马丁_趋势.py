# -*- coding: utf-8 -*-
"""
Meme 反马丁趋势策略 (Anti-Martingale Trend)
===========================================
设计哲学 (与传统马丁完全相反):
  - 传统马丁: 亏损加仓摊成本, 赌反转 → 趋势行情会爆仓。
  - 反马丁:   盈利加仓让利润奔跑, 亏损不加 → 靠移动止损快速认错离场。
              单次大趋势的盈利覆盖前面 N 次小止损, 适合 meme 的"低频暴涨"分布。

三大机制:
  1. 入场: 多头排列 + ADX 趋势强度 + 唐奇安突破 + 放量确认 (四重过滤, 宁缺毋滥)。
  2. 反马丁加仓: 浮盈达阈值才加, 加仓量按金字塔递减(ratio<1), 越往上加得越少,
     避免在趋势末端顶部重仓。每次加仓前重新确认趋势仍有效。
  3. 吊灯移动止损 (Chandelier Exit): stop = 开仓以来最高价 - ATR*mult, 只收紧不放松,
     既是初始止损也是 trailing, 趋势走多远止损跟多远。

⚠️ 上实盘前务必:
  - 依赖 TA-Lib; 先 `freqtrade backtesting` 用历史数据验证参数(尤其 adx_threshold / atr_mult)。
  - 反马丁在震荡市会持续小亏(连环止损), 这是设计内的代价, 不要因连亏就关掉它。
  - 加仓会放大单笔风险敞口, max_stake / 钱包余额要留足。
"""

import logging
from datetime import datetime
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_absolute
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class MemeAntiMartingaleTrendStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False                 # meme 顺势做多为主; 做空可后续扩展

    # ---- 仓位调整(反马丁加仓的开关) ----
    position_adjustment_enable = True
    max_entry_position_adjustment = 3  # 底仓之外最多再加 3 次

    # ---- 止损: 硬止损兜底, 实际由 custom_stoploss(吊灯)主导 ----
    use_custom_stoploss = True
    stoploss = -0.18                  # 极端兜底, 正常触不到

    # ---- 关闭固定 ROI, 完全靠 trailing 退出, 让利润奔跑 ----
    minimal_roi = {"0": 100}

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 220

    # ================== 可调参数 ==================
    # 趋势识别
    ema_fast = 21
    ema_slow = 55
    ema_trend = 200
    adx_threshold = 25                # 趋势强度门槛, 越高越严
    donchian_period = 20              # 突破前 N 周期高点入场
    vol_ma_period = 20
    vol_mult = 1.3                    # 入场要求放量倍数
    atr_period = 14

    # 反马丁加仓: 浮盈达到这些阈值时, 依次触发第1/2/3次加仓
    am_profit_steps = [0.04, 0.09, 0.16]
    am_stake_ratio = 0.6              # 金字塔递减: 每次加仓量 = 底仓 * ratio^n
    am_trend_recheck = True          # 加仓前重新确认趋势

    # 吊灯移动止损
    chandelier_atr_mult = 3.0

    # ----------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.ema_trend)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        # 唐奇安上轨(用 shift(1) 防止用到当前未收盘bar)
        dataframe["donchian_up"] = (
            dataframe["high"].rolling(self.donchian_period).max().shift(1)
        )
        dataframe["donchian_dn"] = (
            dataframe["low"].rolling(self.donchian_period).min().shift(1)
        )
        dataframe["vol_ma"] = dataframe["volume"].rolling(self.vol_ma_period).mean()
        return dataframe

    # ----------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["ema_fast"] > dataframe["ema_slow"],     # 多头排列
            dataframe["ema_slow"] > dataframe["ema_trend"],    # 大趋势向上
            dataframe["adx"] > self.adx_threshold,             # 趋势够强
            dataframe["close"] > dataframe["donchian_up"],     # 突破前高
            dataframe["volume"] > dataframe["vol_ma"] * self.vol_mult,  # 放量
        ]
        dataframe.loc[
            reduce(lambda a, b: a & b, conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_breakout")
        return dataframe

    # ----------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 主退出靠吊灯止损; 这里只做趋势硬反转的兜底退出
        conditions = [
            (dataframe["ema_fast"] < dataframe["ema_slow"])    # 快慢线死叉
            | (dataframe["close"] < dataframe["donchian_dn"])  # 跌破前低
        ]
        dataframe.loc[
            reduce(lambda a, b: a & b, conditions),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_reversal")
        return dataframe

    # ----------------------------------------------------------
    def _first_stake(self, trade: Trade) -> float:
        """取底仓(第一笔成交entry)的名义价值, 作为金字塔基准。"""
        entries = [
            o for o in trade.orders
            if o.ft_is_entry and o.status == "closed" and (o.filled or 0) > 0
        ]
        if entries:
            o = entries[0]
            return abs((o.average or o.price or 0) * (o.filled or 0))
        return trade.stake_amount

    def _trend_ok(self, pair: str) -> bool:
        """加仓前重新确认趋势(读已分析的最新K线)。"""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return False
        last = df.iloc[-1]
        return bool(
            last["ema_fast"] > last["ema_slow"]
            and last["ema_slow"] > last["ema_trend"]
            and last["adx"] > self.adx_threshold
        )

    # ----------------------------------------------------------
    def adjust_trade_position(
        self, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, min_stake, max_stake: float,
        current_entry_rate: float, current_exit_rate: float,
        current_entry_profit: float, current_exit_profit: float, **kwargs,
    ):
        """反马丁: 仅在浮盈达阈值时加仓, 量按金字塔递减。亏损时返回 None(绝不加)。"""
        count = trade.nr_of_successful_entries  # 含底仓, 底仓后第一次加仓时 count==1
        if count == 0 or count > len(self.am_profit_steps):
            return None

        threshold = self.am_profit_steps[count - 1]
        if current_profit < threshold:
            return None                      # 浮盈未达本级阈值

        if self.am_trend_recheck and not self._trend_ok(trade.pair):
            return None                      # 趋势已转弱, 不在末端追加

        base = self._first_stake(trade)
        add_stake = base * (self.am_stake_ratio ** count)

        if min_stake is not None and add_stake < min_stake:
            add_stake = min_stake
        if max_stake is not None and add_stake > max_stake:
            add_stake = max_stake
        if max_stake is not None and max_stake <= 0:
            return None

        logger.info(
            "[AntiM] %s 第%d次金字塔加仓 profit=%.3f stake=%.4g",
            trade.pair, count, current_profit, add_stake,
        )
        return add_stake

    # ----------------------------------------------------------
    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ) -> float:
        """吊灯移动止损: stop = 开仓以来最高价 - ATR*mult。只收紧不放松(框架保证)。"""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        atr = df.iloc[-1]["atr"]
        if atr is None or atr <= 0:
            return None

        # trade.max_rate: freqtrade 记录的开仓以来最高价
        peak = trade.max_rate or current_rate
        stop_price = peak - atr * self.chandelier_atr_mult

        return stoploss_from_absolute(
            stop_price, current_rate,
            is_short=trade.is_short, leverage=trade.leverage,
        )