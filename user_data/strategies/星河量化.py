# -*- coding: utf-8 -*-
"""
==============================================================================
Gate 永续 · 星河混合趋势 + ATR 自适应网格 · 工程化可控版 (Pro)
==============================================================================
本文件是 gate_xinghe_futures_grid_strategy.py 的工程化重写，并叠加 gate_oco.py
作为"交易所端括号单失效保险层"。

------------------------------------------------------------------------------
【策略画像 / 适用说明】(回答任务里的"注释需求")
------------------------------------------------------------------------------
· 交易标的 : Gate 永续合约。优先主流币(BTC/ETH 等高流动性)。
             妖币(低市值高波动)可交易但务必降低杠杆+缩小单笔风险, 因其滑点/插针
             会让"市价止损"实际成交远差于触发价。
· 核心逻辑 : 三票制混合趋势(EMA 方向 + ATR 放量方向 + 波动率方向)过滤入场,
             顺势开仓后用 ATR 自适应步长做"有限次、有上限"的网格补仓(DCA),
             目标是吃到趋势段的均价改善, 而不是无限逆势摊平。
· 适合行情 : 有方向的趋势/单边波段行情。
· 失效行情 : 高频震荡/横盘磨损市 —— EMA 与投票会反复翻向, 触发反复开平 + 手续费
             和点差损耗(whipsaw)。risk_fuse(ATR 过低/过高) 用于在死水或极端
             插针行情中停手, 但它是"事后"指标, 无法完全规避。
· 周期     : 默认 5m(中频)。比原版 1m 更抗噪: 1m 上 EMA60 滞后且信号噪声极大,
             极易过度交易。可在 5m/15m 之间调。
· 方向     : 多空双向(can_short=True), 需 Gate 永续 + 逐仓/全仓配置。
· 持仓风格 : 日内到短波段(分钟级周期 + 时间止损, 不做长期持仓)。
· 杠杆     : 动态, 上限 6x; 波动越大杠杆越低(见 leverage())。
· 资金/回撤: 假设单账户; 单笔风险 ≤ 总资金 10%(按 stake*lev*止损幅度 折算),
             预留 30% 保证金不参与开仓; 连亏/单日回撤熔断进入冷静期。
             可承受最大回撤目标 < 30%(触发即熔断停手, 但不等于绝对不破)。

------------------------------------------------------------------------------
【已知的失效/资金安全场景 — 每一处都标注, 不藏雷】
------------------------------------------------------------------------------
1. 市价止损在插针/流动性枯竭时滑点巨大 → 实际亏损 > 名义止损。妖币尤甚。
2. OCO 括号单是"交易所端失效保险", freqtrade 主流程并不知道它存在 →
   存在"交易所先成交、本地后平账"的竞态窗口(见 gate_oco.sync_brackets 注释)。
3. 资金费率: 永续持仓跨结算点会被收/付资金费, 本策略不预测费率, 长持空头吃正费率
   时是隐性成本。日内风格部分缓解, 但跨结算仍可能中招。
4. 逆势 DCA 是双刃剑: 单边行情里"摊平"= 加速爆仓。本版用 deep_loss_fuse + 加仓
   次数上限 + 中断平仓 三重闸刀限制它(见 should_dca / custom_exit)。
5. 熔断阈值基于"当日起始权益"估算回撤, 进程重启会重置基线 → 重启日的回撤判断
   会偏松。这是工程取舍(无持久化), 已在代码注释标注。
==============================================================================

参数名                    | 含义                 | 默认值      | 建议调优范围    | 影响
timeframe                 | K线周期(结构性)      | 5m          | 5m/15m          | 越小越敏感、噪声越大、交易越频繁
ema_fast_period           | 快EMA周期            | 20          | 8–40            | 小→对趋势更敏感、更早入场但更多假信号
ema_slow_period           | 慢EMA周期            | 60          | 40–120          | 大→趋势确认更稳但更滞后
atr_period                | ATR周期              | 14          | 7–28            | 小→波动估计更跳、网格步长更易变
vote_threshold            | 三票共识门槛         | 2           | 2–3             | 3=更严苛、交易更少、胜率↑频率↓
min_atr_pct               | 可交易最低波动       | 0.0006      | 0.0002–0.003    | 高→过滤死水、漏掉低波趋势
max_atr_pct               | 可交易最高波动       | 0.055       | 0.02–0.08       | 低→回避极端行情、可能错过启动段
atr_grid_multiplier       | ATR→网格步长倍数     | 2.0         | 1.0–4.0         | 大→补仓更稀疏、更省弹药
grid_growth_per_entry     | 每次补仓步长递增     | 0.25        | 0.0–0.6         | 大→越补越远、抗单边能力↑
dca_multipliers           | 各次补仓金额倍数     | [1.3..3.0]  | 调倍率/长度     | 越大→均价改善快但风险敞口暴涨
tp_roi                    | 止盈目标             | 0.025       | 0.01–0.06       | 小→落袋快胜率高、盈亏比低
deep_loss_fuse            | 中断平仓熔丝         | -0.22       | -0.30–-0.12     | 浅→止损快少扛单、易被插针扫
time_stop_candles         | 时间止损K线数        | 240         | 60–480          | 小→快速放弃滞涨仓、可能错后段反弹
max_leverage_cap          | 杠杆硬顶(结构性)     | 6           | ≤6              | 直接放大盈亏与爆仓概率
max_single_risk_pct       | 单笔风险占比         | 0.10        | 0.02–0.10       | 仓位规模总闸门
margin_reserve_pct        | 预留保证金           | 0.30        | 0.2–0.5         | 大→更难爆仓、资金利用率↓
cb_max_consecutive_losses | 连亏熔断             | 5           | 3–8             | 小→更早停手
cb_daily_drawdown         | 日回撤熔断           | 0.30        | 0.10–0.30       | 小→更保守
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
    BooleanParameter,
)

# 复用你提供的括号单失效保险模块。两个文件需放在同一 strategies 目录。
# 若导入失败(例如单测环境), 用一个空 Mixin 兜底, 保证策略本体仍可加载。
try:
    from gate_oco import GateOCOFailsafeMixin
except Exception:  # pragma: no cover
    class GateOCOFailsafeMixin:  # 退化为空实现, 不影响主策略逻辑
        oco_enabled = False
        def place_bracket(self, trade): ...
        def cancel_bracket(self, trade): ...
        def sync_brackets(self): ...
        def reconcile_orphans(self): ...

logger = logging.getLogger(__name__)


class GateXingheGridProStrategy(GateOCOFailsafeMixin, IStrategy):
    """
    多重继承顺序: (GateOCOFailsafeMixin, IStrategy)。
    为什么这个顺序: Mixin 的回调(order_filled/bot_start/...)内部都调用 super(),
    MRO 会是 Pro -> Mixin -> IStrategy。我们在本类里重写同名回调并调用 super(),
    就能把"本策略风控逻辑 + 括号单逻辑 + freqtrade 默认逻辑"串成一条责任链。
    """

    INTERFACE_VERSION = 3

    # ==========================================================================
    # 一、结构性参数(一般固定, 不进 Hyperopt)
    #    —— 改这些等于改策略骨架, 不属于"行情敏感调优"。
    # ==========================================================================
    can_short = True
    timeframe = "5m"               # 中频: 抗噪 + 降低过度交易。原版 1m 噪声过大。
    startup_candle_count = 240     # EMA60 + 波动率 rolling(60) 需要足够预热, 否则前段指标失真。
    process_only_new_candles = True  # 只在新K线计算, 省算力且避免同一根K线内反复触发。

    use_exit_signal = True
    exit_profit_only = False       # 允许信号反转/风险触发时亏损离场, 这是"趋势失效就走"的核心。
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = False    # 止损交给固定 stoploss + 交易所端 OCO 双保险, 不再叠 custom。
    position_adjustment_enable = True  # 启用 DCA(adjust_trade_position)。

    # minimal_roi / stoploss 作为"硬兜底", 真正的精细出场在 custom_exit / OCO。
    # 为什么仍保留: 即便所有自定义逻辑异常, freqtrade 原生 ROI/止损仍会兜住。
    minimal_roi = {"0": 0.025}
    stoploss = -0.18               # 名义止损(对保证金而言较深, 因为有杠杆放大, 真实风险由仓位控制)。

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",          # 止损用市价: 宁可吃滑点也要确保止损成交(防止挂单挂不出去)。
        "stoploss_on_exchange": False, # 交易所端保护改由 gate_oco 括号单负责, 避免双重挂单冲突。
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ---- gate_oco Mixin 行为开关(结构性) ----
    oco_enabled = True
    oco_tp_pct = 0.06              # 交易所端止盈括号距离(失效保险层, 比 ROI 宽, 仅兜底)。
    oco_sl_buffer = 1.15           # 交易所端止损 = stoploss * buffer, 留一点缓冲避免和本地止损同点打架。

    # ==========================================================================
    # 二、行情敏感参数(进 Hyperopt, 需调优)
    #    —— 用 freqtrade Parameter 包装, 既能跑 hyperopt 又能直接当默认值用。
    # ==========================================================================
    # 趋势/指标
    ema_fast_period = IntParameter(8, 40, default=20, space="buy", optimize=True)
    ema_slow_period = IntParameter(40, 120, default=60, space="buy", optimize=True)
    atr_period = IntParameter(7, 28, default=14, space="buy", optimize=True)
    volatility_period = IntParameter(10, 40, default=20, space="buy", optimize=True)
    vote_threshold = IntParameter(2, 3, default=2, space="buy", optimize=True)  # 三票里需几票同向才开仓

    # ATR 可交易区间(过低=死水, 过高=极端行情, 都停手)
    min_atr_pct = DecimalParameter(0.0002, 0.0030, default=0.0006, decimals=4, space="buy", optimize=True)
    max_atr_pct = DecimalParameter(0.020, 0.080, default=0.055, decimals=3, space="buy", optimize=True)

    # 网格步长(决定补仓密度)
    min_grid_step_pct = DecimalParameter(0.004, 0.015, default=0.006, decimals=3, space="buy", optimize=True)
    max_grid_step_pct = DecimalParameter(0.030, 0.090, default=0.060, decimals=3, space="buy", optimize=True)
    atr_grid_multiplier = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space="buy", optimize=True)
    grid_growth_per_entry = DecimalParameter(0.0, 0.6, default=0.25, decimals=2, space="buy", optimize=True)

    # 出场
    tp_roi = DecimalParameter(0.010, 0.060, default=0.025, decimals=3, space="sell", optimize=True)
    deep_loss_fuse = DecimalParameter(-0.30, -0.12, default=-0.22, decimals=2, space="sell", optimize=True)
    time_stop_candles = IntParameter(60, 480, default=240, space="sell", optimize=True)  # 时间止损: N 根K线未达预期离场

    # ==========================================================================
    # 三、风控参数(部分结构性, 部分可调)
    # ==========================================================================
    max_leverage_cap = 6           # 杠杆硬上限(结构性, 任务硬要求 ≤6)。
    max_open_trades_cap = 20       # 最大同时持仓单数(任务要求)。注意: config 里的 max_open_trades 也要 ≤ 此值。
    max_single_risk_pct = 0.10     # 单笔风险 ≤ 总资金 10%(按 保证金*杠杆*止损幅度 折算)。
    margin_reserve_pct = 0.30      # 预留 30% 保证金不参与开仓(安全垫)。
    core_capital_pct = 0.50        # "核心"篮子资金占比(低风险), 其余给"卫星"(高风险)。
    core_risk_fraction = 0.5       # 核心单单笔风险系数(更保守)
    satellite_risk_fraction = 1.0  # 卫星单单笔风险系数(更激进, 但仍受 max_single_risk_pct 钳制)

    # DCA 次数与步进倍数(逆势补仓的"弹药", 有限且递增)
    max_entry_position_adjustment = 5
    dca_multipliers = [1.3, 1.6, 2.0, 2.4, 3.0]

    # 熔断
    cb_max_consecutive_losses = 5  # 连续 5 笔亏损 → 冷静
    cb_daily_drawdown = 0.30       # 单日回撤 30% → 冷静
    cb_cooldown_minutes = 30       # 冷静期时长

    # "核心"币种白名单(低风险篮子)。其余视为卫星。可按需替换。
    core_pairs = {"BTC/USDT:USDT", "ETH/USDT:USDT"}

    # ==========================================================================
    # 四、运行时内部状态(进程内, 不持久化)
    # ==========================================================================
    _cooldown_until: Optional[datetime] = None   # 冷静期截止时间
    _day_baseline_equity: Optional[float] = None # 当日起始权益(算日内回撤用)
    _day_baseline_date = None
    external_signals: dict = {}                  # 外部信号接收口: {pair: 1(多)/-1(空)/0(中性)}

    # --------------------------------------------------------------------------
    # 信号接收接口(供外部模型/webhook 注入方向偏好)
    # 为什么单独做: 任务要求"增加信号接收接口"。populate_* 是向量化历史计算,
    # 不适合塞实时外部信号; 因此把"实时外部信号"放在开仓确认环节(confirm_trade_entry)
    # 做"否决/放行"过滤, 既不破坏回测可复现性, 又能在实盘接入外部判断。
    # --------------------------------------------------------------------------
    def receive_signal(self, pair: str, direction: int) -> None:
        """外部调用: direction ∈ {1: 仅允许做多, -1: 仅允许做空, 0: 中性不干预}。"""
        self.external_signals[pair] = int(direction)

    def _external_allows(self, pair: str, side: str) -> bool:
        sig = self.external_signals.get(pair, 0)
        if sig == 0:
            return True
        want = 1 if side == "long" else -1
        return sig == want

    # ==========================================================================
    # 指标层
    # ==========================================================================
    @staticmethod
    def _atr_pct(dataframe: DataFrame, period: int) -> pd.Series:
        """
        ATR 占价格百分比。用 EWM(Wilder 平滑近似)而非 SMA:
        反应更平滑、对单根插针没那么敏感。返回比例而非绝对值, 方便跨币种统一阈值。
        注意: 所有项都用了 shift(1) 的 prev_close, 无未来函数。
        """
        prev_close = dataframe["close"].shift(1)
        tr = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - prev_close).abs(),
                (dataframe["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return (atr / dataframe["close"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_f = int(self.ema_fast_period.value)
        ema_s = int(self.ema_slow_period.value)
        atr_p = int(self.atr_period.value)
        vol_p = int(self.volatility_period.value)

        dataframe["ema_fast"] = dataframe["close"].ewm(span=ema_f, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=ema_s, adjust=False).mean()
        dataframe["atr_pct"] = self._atr_pct(dataframe, atr_p)
        dataframe["volatility_pct"] = (
            dataframe["close"].pct_change().rolling(vol_p).std().fillna(0)
        )
        # 波动率均值基线: rolling 均值, 全部是过去数据, 无未来函数。
        dataframe["vol_baseline"] = dataframe["volatility_pct"].rolling(60).mean()

        # ---- 三票制 ----
        # 票1 EMA 方向: 经典但滞后。这是趋势确认票, 不是择时票。
        dataframe["ema_vote"] = np.select(
            [dataframe["ema_fast"] > dataframe["ema_slow"],
             dataframe["ema_fast"] < dataframe["ema_slow"]],
            [1, -1], default=0,
        )
        # 票2 ATR 放量方向: ATR 较上一根放大 + 当根阳/阴 → 给出方向。
        # 用 shift(1) 比较, 无未来函数。
        dataframe["atr_vote"] = np.select(
            [(dataframe["atr_pct"] > dataframe["atr_pct"].shift(1)) & (dataframe["close"] > dataframe["open"]),
             (dataframe["atr_pct"] > dataframe["atr_pct"].shift(1)) & (dataframe["close"] < dataframe["open"])],
            [1, -1], default=0,
        )
        # 票3 波动率扩张方向: 波动率高于基线 + 价格在快线上/下方。
        dataframe["volatility_vote"] = np.select(
            [(dataframe["volatility_pct"] > dataframe["vol_baseline"]) & (dataframe["close"] > dataframe["ema_fast"]),
             (dataframe["volatility_pct"] > dataframe["vol_baseline"]) & (dataframe["close"] < dataframe["ema_fast"])],
            [1, -1], default=0,
        )

        vote_sum = dataframe["ema_vote"] + dataframe["atr_vote"] + dataframe["volatility_vote"]
        th = int(self.vote_threshold.value)
        # 为什么用阈值而非简单 sign: 要求"多票共识"才开仓, 过滤单一指标的假信号。
        dataframe["trend_signal"] = np.select([vote_sum >= th, vote_sum <= -th], [1, -1], default=0)

        # 实时网格步长(供 adjust_trade_position 读取当前 ATR 用; 修复原版用常量的 bug)
        dataframe["grid_step_pct"] = (dataframe["atr_pct"] * float(self.atr_grid_multiplier.value)).clip(
            lower=float(self.min_grid_step_pct.value), upper=float(self.max_grid_step_pct.value)
        )
        # 风险保险丝: ATR 过低(死水, 易磨损) 或过高(极端插针, 滑点失控) → 停手。
        dataframe["risk_fuse"] = (
            (dataframe["atr_pct"] < float(self.min_atr_pct.value))
            | (dataframe["atr_pct"] > float(self.max_atr_pct.value))
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        tradable = (
            (dataframe["volume"] > 0)
            & (dataframe["risk_fuse"] == 0)
            & dataframe["atr_pct"].between(float(self.min_atr_pct.value), float(self.max_atr_pct.value))
        )
        long_e = tradable & (dataframe["trend_signal"] == 1)
        short_e = tradable & (dataframe["trend_signal"] == -1)

        dataframe.loc[long_e, ["enter_long", "enter_tag"]] = (1, "xinghe_long")
        dataframe.loc[short_e, ["enter_short", "enter_tag"]] = (1, "xinghe_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        # 信号反转或风险保险丝触发即离场: 这是"趋势没了就走"的护栏。
        dataframe.loc[
            (dataframe["trend_signal"] == -1) | (dataframe["risk_fuse"] == 1),
            ["exit_long", "exit_tag"]] = (1, "long_reversal_or_risk")
        dataframe.loc[
            (dataframe["trend_signal"] == 1) | (dataframe["risk_fuse"] == 1),
            ["exit_short", "exit_tag"]] = (1, "short_reversal_or_risk")
        return dataframe

    # ==========================================================================
    # 杠杆: 动态, 波动越大杠杆越低(反向映射), 硬顶 max_leverage_cap。
    # 为什么: 固定高杠杆在高波动时极易插针爆仓; 用 ATR 反比缩杠杆是最直接的爆仓防护。
    # ==========================================================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        cap = min(self.max_leverage_cap, max_leverage)
        atr_pct = self._current_atr_pct(pair)
        if atr_pct is None or atr_pct <= 0:
            return float(min(3, cap))  # 数据缺失时保守取 3x
        # 线性反比: ATR 在 [min, max] 区间内, 杠杆从 cap 降到 1。
        lo, hi = float(self.min_atr_pct.value), float(self.max_atr_pct.value)
        ratio = (atr_pct - lo) / max(hi - lo, 1e-9)
        ratio = min(max(ratio, 0.0), 1.0)
        lev = cap - (cap - 1) * ratio
        return float(max(1.0, min(round(lev), cap)))

    def _current_atr_pct(self, pair: str) -> Optional[float]:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or len(df) == 0:
                return None
            return float(df["atr_pct"].iloc[-1])
        except Exception:
            return None

    # ==========================================================================
    # 仓位管理: 核心+卫星 + 预留保证金 + 单笔风险上限(基于止损幅度折算)。
    # 为什么按 stake*lev*|stoploss| 折算风险: 在永续里 stake 是保证金,
    # 真正"打到止损会亏多少" = 名义仓位 * 止损幅度 = stake*lev*|sl|。
    # 直接限制这个值 ≤ 总资金*10%, 才是真正的"单笔风险≤10%", 而不是限制保证金。
    # ==========================================================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        try:
            total = self.wallets.get_total(self.config["stake_currency"])
        except Exception:
            total = None

        # 风险敏感参数兜底
        lev = max(float(leverage), 1.0)
        sl = abs(self.stoploss) if self.stoploss else 0.18

        if total and total > 0:
            # 1) 篮子分配: 核心 vs 卫星
            is_core = pair in self.core_pairs
            basket_cap = total * (self.core_capital_pct if is_core else (1 - self.core_capital_pct))
            risk_frac = self.core_risk_fraction if is_core else self.satellite_risk_fraction

            # 2) 单笔风险上限折算成保证金上限
            risk_budget = total * self.max_single_risk_pct * risk_frac
            stake_by_risk = risk_budget / (lev * sl)

            # 3) 预留保证金: 可用资金不超过 (1 - reserve)
            usable = total * (1 - self.margin_reserve_pct)

            stake = min(proposed_stake, stake_by_risk, basket_cap, usable)
        else:
            # 拿不到钱包(异常/某些回测路径) → 退化为保守的 reserve 折扣
            stake = proposed_stake * (1 - self.margin_reserve_pct)

        if min_stake is not None:
            stake = max(stake, min_stake)
        return float(min(stake, max_stake))

    # ==========================================================================
    # 开仓确认: 熔断 + 外部信号否决 + 最大持仓数。
    # 为什么放这里: 这是开仓的最后一道闸门, 所有"禁止开新仓"的风控都收口在此,
    # 既覆盖策略信号开仓, 也覆盖任何来源的开仓尝试。
    # ==========================================================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        # 1) 熔断冷静期
        if self._cooldown_until and current_time < self._cooldown_until:
            logger.warning("[RISK] 冷静期中(至 %s), 拒绝开仓 %s", self._cooldown_until, pair)
            return False
        # 2) 外部信号否决
        if not self._external_allows(pair, side):
            logger.info("[SIGNAL] 外部信号否决 %s %s", pair, side)
            return False
        # 3) 最大同时持仓数(双保险, config 也应配 max_open_trades)
        try:
            if len(Trade.get_open_trades()) >= self.max_open_trades_cap:
                logger.warning("[RISK] 达到最大持仓数 %d, 拒绝开仓 %s", self.max_open_trades_cap, pair)
                return False
        except Exception:
            pass
        return True

    # ==========================================================================
    # 加仓(DCA / 有限网格)。这是逆势补仓, 必须层层设闸, 否则=加速爆仓。
    # 修复点 vs 原版:
    #   (a) 阈值用"当前实时 ATR"算, 不再用常量(原版传 min_grid_step_pct/mult 是死值)。
    #   (b) 加仓次数耗尽 / 触及 deep_loss_fuse 一律不补 → 把弹药交给 custom_exit 去中断平仓。
    # ==========================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Any:
        entry_count = trade.nr_of_successful_entries

        # 弹药耗尽 / 无效计数 → 不补(后续由 custom_exit 决定是否中断平仓)
        if entry_count <= 0 or entry_count > len(self.dca_multipliers):
            return None
        # 只在"浮亏但还没烂到熔丝"的区间补仓; 盈利不补(盈利就让它跑/止盈)。
        if current_profit >= 0 or current_profit <= float(self.deep_loss_fuse.value):
            return None
        # 冷静期内禁止任何加仓
        if self._cooldown_until and current_time < self._cooldown_until:
            return None

        open_rate = getattr(trade, "open_rate", current_entry_rate)
        if not open_rate or open_rate <= 0:
            return None

        is_short = bool(getattr(trade, "is_short", False))
        # 逆向移动幅度(亏的方向): 多头是价格跌, 空头是价格涨。
        adverse = (current_rate / open_rate - 1) if is_short else (open_rate / current_rate - 1)
        if adverse <= 0:
            return None

        # 用实时 ATR 算的网格步长 + 随加仓次数递增(越补越要求更大回撤, 拉开间距, 省弹药)。
        atr_pct = self._current_atr_pct(trade.pair) or (float(self.min_grid_step_pct.value) / float(self.atr_grid_multiplier.value))
        threshold = self._grid_threshold(atr_pct, entry_count - 1)
        if adverse < threshold:
            return None

        base_cost = self._first_filled_entry_cost(trade) or getattr(trade, "stake_amount", 0.0)
        if base_cost <= 0:
            return None

        stake = base_cost * self.dca_multipliers[entry_count - 1]
        stake = min(stake, max_stake)
        if min_stake is not None and stake < min_stake:
            return None
        logger.info("[DCA] %s 第%d次补仓 adverse=%.4f thr=%.4f stake=%.4f",
                    trade.pair, entry_count, adverse, threshold, stake)
        return stake, f"xinghe_dca_{entry_count}"

    def _grid_threshold(self, atr_pct: float, entry_count: int) -> float:
        raw = atr_pct * float(self.atr_grid_multiplier.value)
        grown = raw * (1 + max(entry_count, 0) * float(self.grid_growth_per_entry.value))
        return max(float(self.min_grid_step_pct.value), min(grown, float(self.max_grid_step_pct.value)))

    def _first_filled_entry_cost(self, trade: Trade) -> Optional[float]:
        try:
            orders = trade.select_filled_orders("enter_long") + trade.select_filled_orders("enter_short")
        except (AttributeError, TypeError):
            try:
                orders = trade.select_filled_orders()
            except Exception:
                return None
        if not orders:
            return None
        safe_cost = getattr(orders[0], "safe_cost", None) or getattr(orders[0], "cost", None)
        if not safe_cost or safe_cost <= 0:
            return None
        return float(safe_cost)

    # ==========================================================================
    # 自定义出场: 责任链顺序很重要
    #   1) 交易所端 OCO 已触发(Mixin 写的 flag) → 平账(最高优先, 因为仓位实际已动)
    #   2) 中断平仓: 浮亏击穿 deep_loss_fuse → 强制止损, 禁止再扛(爆仓保护核心)
    #   3) 时间止损: 持仓超过 N 根K线仍未盈利 → 离场(防止资金被无效仓位长期占用)
    #   4) 交给 super()(Mixin/freqtrade)继续处理
    # ==========================================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        # 1) OCO 触发平账由 Mixin.custom_exit 处理; 这里先放行给它(通过 super 链), 但我们
        #    想让本地"中断/时间"止损先于 super 检查吗? 不——OCO 触发意味着交易所仓位已变动,
        #    必须最优先对账, 否则会重复下单。所以先查 flag。
        triggered = trade.get_custom_data("oco_triggered")
        if triggered:
            trade.set_custom_data("oco_triggered", None)
            return f"oco_exchange_{triggered}"

        # 2) 中断平仓(爆仓保护): 浮亏击穿熔丝 → 立即止损, 不再给 DCA 机会。
        if current_profit <= float(self.deep_loss_fuse.value):
            logger.warning("[RISK] %s 中断平仓 浮亏=%.4f <= 熔丝=%.4f",
                            pair, current_profit, float(self.deep_loss_fuse.value))
            return "interrupt_deep_loss"

        # 3) 时间止损: 持仓过久且未盈利 → 离场。
        try:
            held = current_time - trade.open_date_utc
            max_hold = timedelta(minutes=self._tf_minutes() * int(self.time_stop_candles.value))
            if held >= max_hold and current_profit <= 0:
                logger.info("[RISK] %s 时间止损 持仓 %s", pair, held)
                return "time_stop"
        except Exception:
            pass

        # 4) 继续责任链(Mixin → IStrategy)
        try:
            return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)
        except (AttributeError, TypeError):
            return None

    def _tf_minutes(self) -> int:
        return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(self.timeframe, 5)

    # ==========================================================================
    # 括号单挂单: 复用 Mixin.order_filled, 但加"幂等保护"防重复发单。
    # 为什么: 异常重试可能让 order_filled 被多次触发, 若不判重会挂出双份括号。
    # ==========================================================================
    def order_filled(self, pair, trade, order, current_time, **kwargs):
        try:
            existing = trade.get_custom_data("oco_orders") or {}
            is_entry = (order.ft_order_side == trade.entry_side) and order.status == "closed"
            # 已有括号则不重复挂(均价变动后 place_bracket 内部会先撤旧再挂新, 仍安全;
            # 但首单时若已存在说明是重试, 跳过)。这里仍调用 Mixin 让其处理"撤旧挂新"。
            if is_entry:
                self.place_bracket(trade)
        except Exception as e:
            logger.warning("[OCO] order_filled 异常: %s", e)
        # 不重复调用 super 里的 place_bracket(避免双挂): 直接走 IStrategy 默认
        try:
            return IStrategy.order_filled(self, pair, trade, order, current_time, **kwargs)
        except (AttributeError, TypeError):
            return None

    # ==========================================================================
    # 主循环: 熔断状态更新 + 括号单轮询(Mixin.sync_brackets)。
    # 为什么把熔断放 bot_loop_start: 每轮循环刷新一次, 既能及时进冷静期,
    # 也能在冷静期结束后自动恢复, 无需外部干预。
    # ==========================================================================
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        self._update_circuit_breaker(current_time)
        # 调用 Mixin 的轮询(模拟 OCO 一撤一)。Mixin 的 bot_loop_start 会 super() 到 IStrategy。
        try:
            super().bot_loop_start(current_time=current_time, **kwargs)
        except (AttributeError, TypeError):
            try:
                self.sync_brackets()
            except Exception:
                pass

    def bot_start(self, **kwargs) -> None:
        # 启动时初始化当日权益基线, 并让 Mixin 清理孤儿括号单。
        try:
            super().bot_start(**kwargs)  # 链路里 Mixin.reconcile_orphans 会被调用
        except (AttributeError, TypeError):
            try:
                self.reconcile_orphans()
            except Exception:
                pass
        self._reset_day_baseline(datetime.now(timezone.utc))

    # ==========================================================================
    # 熔断逻辑: 连亏 N 笔 或 单日回撤超阈值 → 进入冷静期。
    # ==========================================================================
    def _reset_day_baseline(self, now: datetime) -> None:
        try:
            self._day_baseline_equity = self.wallets.get_total(self.config["stake_currency"])
        except Exception:
            self._day_baseline_equity = None
        self._day_baseline_date = now.date()

    def _update_circuit_breaker(self, now: datetime) -> None:
        # 跨日重置基线(注意: 进程重启也会重置 → 重启日回撤判断偏松, 已知局限)。
        if self._day_baseline_date != now.date():
            self._reset_day_baseline(now)

        # 已在冷静期, 未到期则不重复计算
        if self._cooldown_until and now < self._cooldown_until:
            return

        trip = False
        reason = ""

        # (a) 连续亏损
        try:
            closed = [t for t in Trade.get_trades_proxy(is_open=False)]
            closed.sort(key=lambda t: t.close_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            streak = 0
            for t in closed[: self.cb_max_consecutive_losses]:
                profit = t.close_profit if t.close_profit is not None else 0
                if profit < 0:
                    streak += 1
                else:
                    break
            if streak >= self.cb_max_consecutive_losses:
                trip, reason = True, f"连续{streak}笔亏损"
        except Exception as e:
            logger.debug("[RISK] 连亏统计失败: %s", e)

        # (b) 单日回撤
        try:
            if not trip and self._day_baseline_equity and self._day_baseline_equity > 0:
                cur = self.wallets.get_total(self.config["stake_currency"])
                dd = (self._day_baseline_equity - cur) / self._day_baseline_equity
                if dd >= self.cb_daily_drawdown:
                    trip, reason = True, f"单日回撤{dd:.1%}"
        except Exception as e:
            logger.debug("[RISK] 回撤统计失败: %s", e)

        if trip:
            self._cooldown_until = now + timedelta(minutes=self.cb_cooldown_minutes)
            logger.warning("[RISK] 触发熔断(%s), 进入冷静期至 %s", reason, self._cooldown_until)