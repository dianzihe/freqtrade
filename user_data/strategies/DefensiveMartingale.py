from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional
import talib.abstract as ta
import pandas as pd


class DefensiveMartingale(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    startup_candle_count = 210

    # 分级止盈，越拖时间越保守
    minimal_roi = {
        "0": 0.012,
        "180": 0.008,
        "360": 0.004
    }
    stoploss = -0.18          # 硬性总止损（相对首仓保证金）
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

    # ---- 马丁核心参数 ----
    position_adjustment_enable = True
    max_entry_position_adjustment = 4   # 首仓 + 最多4次补仓
    max_dca_multiplier = 5              # 预留资金系数
    leverage_level = 2

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 只在大趋势多头背景下的回调超卖点启动马丁，避免逆势硬扛下跌趋势
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema200']) &
                (dataframe['rsi'] < 35) &
                (dataframe['volume'] > dataframe['volume_mean'] * 0.8)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[(dataframe['rsi'] > 70), 'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return self.leverage_level

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                             proposed_stake: float, min_stake: Optional[float], max_stake: float,
                             leverage: float, entry_tag: Optional[str], side: str,
                             **kwargs) -> float:
        # 首仓只用总额度的1/5，为后续4次补仓预留80%资金
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                               current_profit: float, min_stake: Optional[float], max_stake: float,
                               current_entry_rate: float, current_exit_rate: float,
                               current_entry_profit: float, current_exit_profit: float,
                               **kwargs) -> Optional[float]:

        count_of_entries = trade.nr_of_successful_entries
        if count_of_entries >= self.max_entry_position_adjustment + 1:
            return None  # 达到补仓上限，交由止损/止盈处理，不再加仓

        # 补仓间距逐级放大（1.3倍），越补越稀疏，降低连续插针风险
        base_gap = -0.03
        gap_multiplier = 1.3
        trigger = base_gap * (gap_multiplier ** (count_of_entries - 1))

        if current_profit > trigger:
            return None  # 还没跌到触发点，不加仓

        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            last_stake = filled_entries[-1].cost
            dca_multiplier = 1.4  # 防御型倍投系数较低
            next_stake = last_stake * dca_multiplier

            if next_stake > max_stake:
                return None  # 超出账户可用资金上限，强制停止加仓
            return next_stake
        except Exception:
            return None