# Optimized_Gold_Strategy.py
# freqtrade IStrategy —— 由 MQ5 "谷歌黄金" 思路移植
# 核心：真·SuperTrend + 真·QQE + ADX趋势过滤 + 时间过滤 + ATR动态止损止盈 + 风险百分比仓位

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy, DecimalParameter, IntParameter, BooleanParameter,
    stoploss_from_absolute
)
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date

import talib.abstract as ta


# ============================================================
# 辅助函数：真正的 SuperTrend
# 为什么单独写：原MQ5用EMA冒充ST，丢失了ST"突破才翻转、否则锁住通道"的核心特性。
# ============================================================
def supertrend(df: DataFrame, period: int, multiplier: float):
    """返回 (direction, line)。direction: 1=多头趋势, -1=空头趋势。"""
    atr = ta.ATR(df, timeperiod=period)
    hl2 = (df['high'] + df['low']) / 2.0
    basic_ub = (hl2 + multiplier * atr).values
    basic_lb = (hl2 - multiplier * atr).values
    close = df['close'].values
    n = len(df)

    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    direction = np.ones(n)

    for i in range(1, n):
        # 上轨只能向下收紧；价格上穿才允许重置（ST的"棘轮"特性，避免来回抖动）
        if np.isnan(basic_ub[i]):
            final_ub[i] = final_ub[i - 1]
        elif (basic_ub[i] < final_ub[i - 1]) or (close[i - 1] > final_ub[i - 1]):
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if np.isnan(basic_lb[i]):
            final_lb[i] = final_lb[i - 1]
        elif (basic_lb[i] > final_lb[i - 1]) or (close[i - 1] < final_lb[i - 1]):
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # 只有收盘价真正突破对侧轨道才翻转方向，否则延续——这才是趋势跟踪的意义
        if close[i] > final_ub[i - 1]:
            direction[i] = 1
        elif close[i] < final_lb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    line = np.where(direction == 1, final_lb, final_ub)
    return pd.Series(direction, index=df.index), pd.Series(line, index=df.index)


# ============================================================
# 辅助函数：真正的 QQE (Qualitative Quantitative Estimation)
# 为什么单独写：原MQ5用 RSI>50 冒充。真QQE是"平滑RSI + 自适应ATR通道"的趋势翻转系统，
# 它的滞后比裸RSI更可控，假信号更少。
# ============================================================
def qqe(df: DataFrame, rsi_period: int = 14, sf: int = 5, factor: float = 4.236):
    """返回 (rsi_ma, qqe_trend)。qqe_trend: 1=多, -1=空。"""
    rsi = ta.RSI(df, timeperiod=rsi_period)
    rsi_ma = ta.EMA(rsi, timeperiod=sf)
    # 转为 Series 以支持 .shift() / .abs()；后续循环仍用 .values
    rsi_ma_s = pd.Series(rsi_ma, index=df.index)
    wilders = rsi_period * 2 - 1
    atr_rsi = (rsi_ma_s - rsi_ma_s.shift(1)).abs()
    smooth_atr = ta.EMA(atr_rsi, timeperiod=wilders)
    dar = ta.EMA(smooth_atr, timeperiod=wilders) * factor

    r = rsi_ma_s.values
    d = pd.Series(dar, index=df.index).values
    n = len(df)
    longband = np.zeros(n)
    shortband = np.zeros(n)
    trend = np.ones(n)

    for i in range(1, n):
        if np.isnan(r[i]) or np.isnan(d[i]):
            longband[i] = longband[i - 1]
            shortband[i] = shortband[i - 1]
            trend[i] = trend[i - 1]
            continue

        new_long = r[i] - d[i]
        new_short = r[i] + d[i]

        # 多头通道只能上移（锁利），跌破才重置；空头通道反之
        if r[i - 1] > longband[i - 1] and r[i] > longband[i - 1]:
            longband[i] = max(longband[i - 1], new_long)
        else:
            longband[i] = new_long

        if r[i - 1] < shortband[i - 1] and r[i] < shortband[i - 1]:
            shortband[i] = min(shortband[i - 1], new_short)
        else:
            shortband[i] = new_short

        # 平滑RSI上穿空头通道→转多；下穿多头通道→转空；否则延续
        if r[i - 1] <= shortband[i - 1] and r[i] > shortband[i - 1]:
            trend[i] = 1
        elif r[i - 1] >= longband[i - 1] and r[i] < longband[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    return rsi_ma, pd.Series(trend, index=df.index)


class OptimizedGoldStrategy_15m(IStrategy):
    """
    移植自 MQ5 "谷歌黄金" 趋势策略。
    设计为 15m 趋势跟随（多周期回测版）；现货可用，永续合约可双向（需 trading_mode=futures）。
    """

    INTERFACE_VERSION = 3

    # 原策略是黄金H4单边趋势思路，时间框架改为15m（多周期回测版）
    timeframe = '15m'

    can_short = False              # 回测spot模式：做空信号自动忽略
    process_only_new_candles = True   # 关键：只在新K线收盘后计算，避免用未完成K线触发信号
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True

    # minimal_roi 设很大几乎不触发——真正的止盈交给 ATR 动态目标(custom_exit)。
    # 这里保留一个"保险性"的极高止盈，防止极端行情下逻辑漏接。
    minimal_roi = {"0": 100.0}

    # stoploss 这里只是 freqtrade 要求的"硬底线"兜底值，真正止损由 custom_stoploss 用ATR动态算。
    stoploss = -0.30

    # 关掉内置 trailing，自己用 ATR 实现（原MQ5是1.5×ATR且保本后才移动）
    trailing_stop = False

    startup_candle_count = 200    # EMA52 + QQE Wilder(27) 需要足够预热，否则前段指标失真

    # ---- 可优化参数（hyperopt 用），默认值对齐原MQ5 ----
    st_period = IntParameter(7, 20, default=10, space='buy')
    st_mult = DecimalParameter(2.0, 4.0, default=3.0, space='buy')
    rsi_period = IntParameter(8, 21, default=14, space='buy')
    adx_period = IntParameter(10, 20, default=14, space='buy')
    adx_threshold = DecimalParameter(15.0, 35.0, default=25.0, space='buy')  # 原版20对黄金H4偏低，默认提到25
    atr_period = IntParameter(10, 20, default=14, space='sell')
    stop_atr = DecimalParameter(1.5, 3.5, default=2.0, space='sell')
    take_atr = DecimalParameter(2.0, 6.0, default=4.0, space='sell')
    trail_atr = DecimalParameter(1.0, 2.5, default=1.5, space='sell')

    use_adx = True
    use_time_filter = True
    start_hour = 8        # 注意：freqtrade 内部为 UTC，需按你的交易所时区重新标定！
    end_hour = 20

    # 风险管理
    use_money_mgmt = True
    risk_percent = 2.0    # 单笔风险占总权益的百分比

    # ============================================================
    # 指标计算
    # ============================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ATR：止损/止盈/仓位都依赖它，必须最先算
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=int(self.atr_period.value))

        # ADX：趋势强度过滤，弱趋势(震荡)时直接不开仓，这是原策略避免来回打脸的核心
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=int(self.adx_period.value))

        # EMA52：原MQ5的"趋势基准线"，作为大方向确认
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=52)

        # 真·SuperTrend：方向过滤
        st_dir, st_line = supertrend(dataframe, int(self.st_period.value), float(self.st_mult.value))
        dataframe['st_dir'] = st_dir
        dataframe['st_line'] = st_line

        # 真·QQE：动量趋势确认
        rsi_ma, qqe_trend = qqe(dataframe, rsi_period=int(self.rsi_period.value))
        dataframe['rsi_ma'] = rsi_ma
        dataframe['qqe_trend'] = qqe_trend

        # 小时（UTC），用于时间过滤
        dataframe['hour'] = dataframe['date'].dt.hour

        return dataframe

    # ============================================================
    # 入场逻辑
    # 为什么这样组合：SuperTrend定方向 + QQE确认动量 + EMA52确认大势 + ADX确认趋势强度。
    # 但要清醒——这几个条件相关性较高，与其说"四重过滤"，不如说"宁可少做也不在震荡里硬刚"。
    # ============================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 趋势强度门槛：ADX低于阈值=震荡，直接不交易（关掉则恒为True）
        trend_strong = (dataframe['adx'] > self.adx_threshold.value) if self.use_adx else True

        # 时间窗口：只在指定时段开仓。注意UTC时区！
        if self.use_time_filter:
            in_session = (dataframe['hour'] >= self.start_hour) & (dataframe['hour'] < self.end_hour)
        else:
            in_session = True

        # --- 做多 ---
        dataframe.loc[
            (dataframe['st_dir'] == 1) &            # SuperTrend 多头
            (dataframe['qqe_trend'] == 1) &         # QQE 多头
            (dataframe['close'] > dataframe['ema_trend']) &  # 价格在EMA52上方
            trend_strong &
            in_session &
            (dataframe['volume'] > 0),              # 防止用无成交量的脏数据触发
            ['enter_long', 'enter_tag']
        ] = (1, 'st_qqe_long')

        # --- 做空（仅永续）---
        dataframe.loc[
            (dataframe['st_dir'] == -1) &
            (dataframe['qqe_trend'] == -1) &
            (dataframe['close'] < dataframe['ema_trend']) &
            trend_strong &
            in_session &
            (dataframe['volume'] > 0),
            ['enter_short', 'enter_tag']
        ] = (1, 'st_qqe_short')

        return dataframe

    # ============================================================
    # 出场逻辑（信号层面）
    # 为什么：趋势策略的核心退出是"趋势翻转就走"，不要傻等止损。
    # 止盈/止损的价格层面退出在 custom_exit / custom_stoploss 里用ATR做。
    # ============================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SuperTrend 翻空 → 平多
        dataframe.loc[
            (dataframe['st_dir'] == -1) & (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'st_flip_exit_long')

        # SuperTrend 翻多 → 平空
        dataframe.loc[
            (dataframe['st_dir'] == 1) & (dataframe['volume'] > 0),
            ['exit_short', 'exit_tag']
        ] = (1, 'st_flip_exit_short')

        return dataframe

    # ============================================================
    # ATR 动态止盈（对应原MQ5的 TakeATR = 4×ATR）
    # 为什么用 custom_exit 而非 minimal_roi：minimal_roi 是固定百分比，
    # 无法随波动率自适应；黄金/币的波动差异巨大，必须用ATR。
    # ============================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        entry_atr = self._entry_atr(pair, trade)
        if entry_atr is None:
            return None

        tp_dist = entry_atr * float(self.take_atr.value)
        if trade.is_short:
            if current_rate <= trade.open_rate - tp_dist:
                return 'atr_take_profit'
        else:
            if current_rate >= trade.open_rate + tp_dist:
                return 'atr_take_profit'
        return None

    # ============================================================
    # ATR 动态止损 + 移动止损
    # 对应原MQ5：初始 2×ATR；盈利后用 1.5×ATR 跟踪，且"保本后才开始移动"。
    # 为什么 current_profit<=0 时返回初始固定止损：复刻原版"未盈利时不动止损"的逻辑，
    # 避免在浮亏时把止损越拉越近、被正常回踩扫出。
    # ============================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> Optional[float]:
        entry_atr = self._entry_atr(pair, trade)
        if entry_atr is None:
            return None

        # 初始止损：开仓价 ± 2×ATR
        init_dist = entry_atr * float(self.stop_atr.value)
        if trade.is_short:
            init_sl = trade.open_rate + init_dist
        else:
            init_sl = trade.open_rate - init_dist

        # 未盈利：维持初始止损，不移动（复刻"保本后才移"的设计意图）
        if current_profit <= 0:
            return stoploss_from_absolute(init_sl, current_rate,
                                          is_short=trade.is_short, leverage=trade.leverage)

        # 已盈利：用当前价 ± 1.5×ATR 跟踪。freqtrade 只会接受比现有止损更紧的值，故不会反向放松。
        trail_dist = entry_atr * float(self.trail_atr.value)
        if trade.is_short:
            trail_sl = current_rate + trail_dist
        else:
            trail_sl = current_rate - trail_dist

        return stoploss_from_absolute(trail_sl, current_rate,
                                      is_short=trade.is_short, leverage=trade.leverage)

    # ============================================================
    # 风险百分比仓位（对应原MQ5 CalculateLots）
    # 为什么这样算：让"开仓价到止损价"这段亏损 ≈ 账户权益×risk%。
    # CEX 与 MT5 不同——这里用 名义价值=stake×leverage，亏损比例≈ATR止损距离/价格 来反推。
    # ============================================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        if not self.use_money_mgmt:
            return proposed_stake

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return proposed_stake

        atr = df['atr'].iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            return proposed_stake

        # 单笔可承受亏损金额
        try:
            balance = self.wallets.get_total(self.config['stake_currency'])
        except Exception:
            return proposed_stake
        risk_money = balance * self.risk_percent / 100.0

        # 止损距离的相对比例（开仓价的百分比）
        sl_dist = atr * float(self.stop_atr.value)
        sl_pct = sl_dist / current_rate
        if sl_pct <= 0:
            return proposed_stake

        # 名义价值 × sl_pct = risk_money  →  名义价值 = risk_money / sl_pct
        # stake(保证金) = 名义价值 / leverage
        notional = risk_money / sl_pct
        stake = notional / max(leverage, 1.0)

        # 夹在交易所允许范围内
        if min_stake is not None:
            stake = max(stake, min_stake)
        stake = min(stake, max_stake)
        return stake

    # ============================================================
    # 工具：取"开仓那根K线"的ATR
    # 为什么用开仓K线而非当前K线：止损/止盈基准要固定，否则止损会随后续波动率漂移，
    # 出现"止损越走越远"的诡异行为。这也是杜绝隐性 look-ahead 的关键。
    # ============================================================
    def _entry_atr(self, pair: str, trade: Trade) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        row = df.loc[df['date'] == entry_date]
        if row.empty:
            atr = df['atr'].iloc[-1]   # 兜底：找不到就用最新值
        else:
            atr = row['atr'].iloc[0]
        return float(atr) if np.isfinite(atr) else None

    # ============================================================
    # 杠杆（永续）——下一阶段风控接口预留
    # TODO: 接入资金费率(funding rate)规避、最大杠杆动态调整、爆仓距离校验
    # ============================================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        # 暂用保守固定杠杆，真正的杠杆/爆仓风控留待下一阶段
        return min(3.0, max_leverage)