# pragma pylint: disable=missing-docstring, invalid-name
import logging
from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy, IntParameter, DecimalParameter, informative
)
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class TrendPyramidStrategy(IStrategy):
    """
    趋势跟踪 + 金字塔加仓（反马丁格尔）策略

    核心逻辑：
      1. 用 EMA 多头排列 + ADX + MACD 识别上升趋势并入场
      2. 浮盈达到阈值后金字塔式追加仓位（每次加仓金额递减）
      3. ATR 动态止损 + 移动止盈，亏损单绝不加仓
    """

    INTERFACE_VERSION = 3

    # ===================== 基本配置 =====================
    timeframe = '1h'                 # 主周期，趋势策略建议 1h 或 4h
    can_short = False                # 现货默认只做多；合约可改为 True

    # 启用仓位调整（金字塔加仓的前提）
    position_adjustment_enable = True

    # 最大开放交易对数，配合 stake_amount 控制总仓位
    max_open_trades = 5

    # 启动所需的最少 K 线数
    startup_candle_count: int = 200

    # ===================== 止盈止损 =====================
    # 趋势策略让利润奔跑，minimal_roi 设得很宽松，主要靠 trailing 和信号离场
    minimal_roi = {
        "0": 0.50,      # 50% 直接走（极端行情兜底）
        "720": 0.20,    # 12小时后 20%
        "1440": 0.10,   # 24小时后 10%
        "2880": 0.04    # 48小时后 4%
    }

    # 初始固定止损（后面会被 custom_stoploss 的 ATR 止损覆盖）
    stoploss = -0.15

    # 移动止盈
    trailing_stop = True
    trailing_stop_positive = 0.03          # 盈利后回撤 3% 离场
    trailing_stop_positive_offset = 0.06   # 盈利达 6% 后才激活移动止盈
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True

    # ===================== 可优化参数 =====================
    # 趋势判断
    buy_adx = IntParameter(20, 40, default=25, space='buy', optimize=True)
    ema_fast = IntParameter(8, 30, default=20, space='buy', optimize=True)
    ema_slow = IntParameter(40, 100, default=50, space='buy', optimize=True)
    ema_trend = IntParameter(100, 200, default=200, space='buy', optimize=True)

    # ATR 止损倍数
    atr_stop_mult = DecimalParameter(2.0, 5.0, default=3.0, space='sell', optimize=True)

    # 金字塔加仓参数
    # 每次加仓触发的浮盈阈值（每达到一档加一次）
    pyramid_step_profit = DecimalParameter(0.03, 0.10, default=0.05, space='buy', optimize=True)
    # 最大加仓次数
    max_pyramid_entries = IntParameter(2, 5, default=3, space='buy', optimize=True)

    # ===================== 指标计算 =====================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # EMA 趋势均线
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=int(self.ema_fast.value))
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=int(self.ema_slow.value))
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=int(self.ema_trend.value))

        # ADX 趋势强度
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # MACD 动能
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # ATR 用于动态止损
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # RSI 辅助过滤（避免在严重超买时入场）
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    # ===================== 入场信号 =====================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 多头排列：快线 > 慢线 > 长期趋势线
                (dataframe['ema_fast'] > dataframe['ema_slow']) &
                (dataframe['ema_slow'] > dataframe['ema_trend']) &
                # 价格在趋势线上方
                (dataframe['close'] > dataframe['ema_trend']) &
                # 趋势足够强
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['plus_di'] > dataframe['minus_di']) &
                # MACD 多头动能
                (dataframe['macd'] > dataframe['macdsignal']) &
                # 不在极端超买区追高
                (dataframe['rsi'] < 75) &
                # 成交量确认
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    # ===================== 离场信号 =====================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 趋势反转：快线跌破慢线
                (dataframe['ema_fast'] < dataframe['ema_slow']) &
                # 或 MACD 死叉
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe

    # ===================== ATR 动态止损 =====================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle['atr']

        if atr == 0 or current_rate == 0:
            return None

        # 止损价 = 当前价 - ATR * 倍数，转换为相对开仓价的比例
        atr_stop_distance = (atr * self.atr_stop_mult.value) / current_rate

        # 返回负值表示止损比例
        return -atr_stop_distance

    # ===================== 金字塔加仓核心 =====================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """
        反马丁格尔金字塔加仓：
          - 仅在浮盈达到阶梯阈值时加仓
          - 每次加仓金额递减（金字塔结构）
          - 亏损时绝不加仓
        """

        # 亏损中绝不加仓（反马丁的铁律）
        if current_profit <= 0:
            return None

        # 已发生的入场次数（含首仓）
        filled_entries = trade.nr_of_successful_entries
        count_of_entries = trade.nr_of_successful_entries

        # 已达最大加仓次数则停止
        if count_of_entries > self.max_pyramid_entries.value:
            return None

        # 计算当前应该触发到第几档加仓
        # 例如 step=5%，浮盈 12% 时应已加到第 2 档
        target_level = int(current_profit / self.pyramid_step_profit.value)

        # 只有当浮盈跨过新的台阶，且台阶数 >= 已加仓次数时才加仓
        if target_level < count_of_entries:
            return None

        # 确认趋势仍然健康，否则不加仓
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle['ema_fast'] < last_candle['ema_slow']:
            return None

        # ----- 金字塔递减加仓金额 -----
        try:
            # 首仓金额
            first_stake = trade.orders[0].stake_amount
        except (IndexError, AttributeError):
            first_stake = trade.stake_amount

        # 递减系数：第 1 次加仓为首仓的 60%，第 2 次为 40%，第 3 次为 25%...
        decay_factors = [0.6, 0.4, 0.25, 0.15, 0.10]
        idx = count_of_entries - 1  # 当前是第几次加仓（0-based）
        if idx >= len(decay_factors):
            return None

        stake_amount = first_stake * decay_factors[idx]

        # 边界检查
        if min_stake is not None and stake_amount < min_stake:
            stake_amount = min_stake
        if stake_amount > max_stake:
            stake_amount = max_stake

        logger.info(
            f"[金字塔加仓] {trade.pair} 第 {count_of_entries} 次加仓 | "
            f"浮盈 {current_profit:.2%} | 加仓金额 {stake_amount:.2f}"
        )

        return stake_amount