# -*- coding: utf-8 -*-
"""
Meme 波动率网格马丁策略（工程化重构版）

核心逻辑：基于布林带下轨 + ATR 宽度判断"可交易的超卖区间"，
做均值回归——而不是假设每次暴跌都会恢复。所有风控、加仓、熔断在基类。
=============================================================
交易标的：妖币（Meme）永续合约 / 主流币
适合行情：宽幅震荡 / 区间盘整（有波动但无单边趋势）
失效行情：单边瀑布、流动性枯竭、趋势性下跌（此时均值回归假设失效，靠基类的
         "马丁中断平仓 + ATR爆炸禁加仓 + 趋势过滤"来止血）
方向：多空双向（永续），现货自动降级为仅做多
持仓风格：波段（时间止损默认 48h）
方向偏好：以做多超卖反弹为主

参数名                 | 含义               | 默认值   | 建议调优范围    | 影响
buy_atr_pct_min        | 入场最低波动率     | 0.006    | 0.003-0.020     | 太低进死盘无利润；太高只在剧烈波动入场，信号少
buy_rsi_max            | 入场 RSI 上限      | 35       | 20-45           | 越低越严格，信号越少但质量越高
dca_max_entries        | 最大加仓次数       | 3        | 1-4             | 风险核心：越大摊薄能力越强，但单边时亏损放大越快
dca_step_pct           | 加仓阶梯跌幅       | 0.06     | 0.04-0.12       | 越小加仓越频繁（弹药消耗快）；越大加仓更稀疏更稳
dca_cooldown_min       | 加仓冷却分钟       | 45       | 15-120          | 防瀑布连续踩刀，越大越保守
sell_rsi_min           | 止盈 RSI 阈值      | 62       | 55-75           | 越高持仓越久博更高回归，但回吐风险增
take_profit_pct        | 目标止盈           | 0.04     | 0.02-0.10       | 越小胜率高单笔薄；越大单笔厚但常吃不到
time_stop_candles      | 时间止损 K 线数    | 192      | 48-288          | 越小越快释放套牢资金，但易在回归前被止损
base_leverage          | 基础杠杆           | 3        | 1-6（结构性）   | 直接放大盈亏与爆仓概率，已被波动率动态压制
MAX_OPEN_TRADES_HARD   | 最大持仓（硬）     | 20       | 不可调          | 控制总敞口
RESERVE_MARGIN_RATIO   | 预留保证金（硬）   | 0.30     | 不可调          | 安全垫，防爆仓
DAILY_DRAWDOWN_LIMIT   | 单日回撤熔断（硬） | 0.30     | 不可调          | 触发冷静期


"""

from pandas import DataFrame

from gate_oco import GateOCOFailsafeMixin
from Meme_马丁_基类 import MemeMartingaleBaseStrategy


class MemeVolatilityGridMartingaleStrategy(GateOCOFailsafeMixin, MemeMartingaleBaseStrategy):
    """波动率网格马丁：交易区间回归，不赌每次都反弹。"""

    # 子策略可覆盖部分参数（其余继承基类）
    dca_tag_prefix = "vol_grid_dca"

    # 交易所端 OCO 失效保险；具体挂单/清理/对账逻辑集中在 gate_oco.py。
    oco_enabled = True
    oco_tp_pct = 0.06
    oco_sl_buffer = 1.15

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)

        # 条件1：波动率足够（太死的盘没回归空间，也赚不到价差）
        wide_enough = dataframe["atr_pct"] > self.buy_atr_pct_min.value
        # 条件2：跌破布林下轨（超卖）。用收盘价比较，process_only_new_candles 保证无未来函数
        lower_band_touch = dataframe["close"] < dataframe["bb_lower"]
        # 条件3：还在区间内（0.12~0.72），不在最底部——已跌穿区间底说明可能转单边，不接
        range_not_dead = dataframe["range_position"].between(0.12, 0.72)
        # 条件4：RSI 超卖确认（增加信号质量，过滤假触碰）
        rsi_oversold = dataframe["rsi"] < self.buy_rsi_max.value
        # 条件5：非下跌趋势（趋势过滤，马丁绝不在下跌趋势里抄底）
        not_downtrend = dataframe["ema_slope"] > -0.01
        # 条件6：ATR 未爆炸（不在瀑布里接刀）
        no_atr_spike = dataframe["atr_ratio"] < self.ATR_SPIKE_BLOCK

        dataframe.loc[
            wide_enough & lower_band_touch & range_not_dead
            & rsi_oversold & not_downtrend & no_atr_spike,
            ["enter_long", "enter_tag"],
        ] = (1, "vol_grid_lower_band")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)

        # 均值回归止盈：价格回到布林中轨 或 RSI 转强 → 离场
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_mid"])
            | (dataframe["rsi"] > self.sell_rsi_min.value),
            ["exit_long", "exit_tag"],
        ] = (1, "vol_grid_mean_exit")

        return dataframe
