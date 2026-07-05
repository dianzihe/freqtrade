from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional
import talib.abstract as ta
import pandas as pd


class AggressiveMartingale(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'
    startup_candle_count = 60

    minimal_roi = {
        "0": 0.008,
        "60": 0.005,
        "180": 0.002
    }
    stoploss = -0.28          # 容忍更深回撤，换取更多补仓次数
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

    # ---- 马丁核心参数 ----
    position_adjustment_enable = True
    max_entry_position_adjustment = 7   # 首仓 + 最多7次补仓
    max_dca_multiplier = 10
    leverage_level = 5

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 更宽松的入场条件，交易频率更高，主动捕捉高波动机会
        dataframe.loc[(dataframe['rsi'] < 45), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[(dataframe['rsi'] > 65), 'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return self.leverage_level

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                             proposed_stake: float, min_stake: Optional[float], max_stake: float,
                             leverage: float, entry_tag: Optional[str], side: str,
                             **kwargs) -> float:
        # 首仓只用总额度的1/10，为7次补仓预留90%资金
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                               current_profit: float, min_stake: Optional[float], max_stake: float,
                               current_entry_rate: float, current_exit_rate: float,
                               current_entry_profit: float, current_exit_profit: float,
                               **kwargs) -> Optional[float]:

        count_of_entries = trade.nr_of_successful_entries
        if count_of_entries >= self.max_entry_position_adjustment + 1:
            return None

        # 补仓间距扩大较慢（1.1倍），触发更频繁更密集
        base_gap = -0.015
        gap_multiplier = 1.1
        trigger = base_gap * (gap_multiplier ** (count_of_entries - 1))

        if current_profit > trigger:
            return None

        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            last_stake = filled_entries[-1].cost
            dca_multiplier = 1.8  # 进攻型倍投系数更高，摊低成本更快
            next_stake = last_stake * dca_multiplier

            if next_stake > max_stake:
                return None
            return next_stake
        except Exception:
            return None