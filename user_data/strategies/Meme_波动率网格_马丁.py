# -*- coding: utf-8 -*-
"""
================================================================================
策略名称：波动率网格马丁 (Meme Volatility Grid Martingale)
================================================================================

【交易标的】
    Gate.io 永续合约，主要面向"妖币/Meme 币"（DOGE、PEPE、WIF 这类高波动、
    无基本面锚、情绪驱动的合约）。也可跑主流币，但主流币波动小、区间回归弱，
    这套逻辑的边际收益不高，真正的设计目标就是高波动品种的区间震荡。

【核心逻辑 —— 为什么这样写】
    本质是"区间回归 + 马丁摊薄"的组合，而不是趋势跟随：
      1. 入场：用 RSI 极值 + 偏离 EMA50 判断价格到了区间的上/下沿，
         做逆势单（超卖做多、超买做空），赌它回归中轨。
      2. 加仓：价格继续朝不利方向走时，按固定步长(dca_step_pct)分档补仓，
         每档仓位按 dca_size_mult 放大，摊低均价——这就是马丁格尔。
         目的：只要价格在区间内最终回归，即使入场点不完美也能靠摊薄回本止盈。
      3. 止盈：不贪。盈利到 take_profit_pct 就用移动止损锁定一半利润，
         或 RSI 回到中性区(populate_exit_trend)主动离场。区间策略吃的是
         "反复的小波动"，不是"一波大行情"，所以止盈必须早、必须落袋。

    为什么用逆势而不用顺势？
        Meme 币在没有消息面时，绝大多数时间是无序高频震荡，顺势追单极易被
        插针打脸。区间回归在"震荡市"胜率显著更高。代价是——一旦转单边趋势，
        逆势 + 马丁会同时放大亏损（见文末风险节，这是本策略的致命面）。

【适合行情】
    ✅ 高波动的横盘震荡 / 宽幅区间          → 主战场，马丁摊薄优势最大
    ⚠️ 低波动窄幅震荡                       → 能盈利但效率低，手续费吃利润
    ❌ 单边趋势（尤其是瀑布式下跌 / 逼空）  → 策略失效区，会连续加仓直到中断线砍仓
    ❌ 流动性枯竭 / 跳空                     → 滑点与强平风险剧增

【周期 (timeframe)】
    15m。为什么不是 1m/5m？
        - 1m/5m 上 RSI 极值信号噪声太大，假信号多，马丁会被频繁触发加仓，
          资金利用率和手续费都难看。
        - 15m 在"信号质量"与"响应速度"之间平衡，一根 K 线 15 分钟，
          区间回归有足够时间发生，又不至于像 1h 那样反应迟钝错过离场。
        - 注意：中断/强平的即时性不依赖 15m 主周期，而是靠交易所端 OCO
          条件单实时触发（见 GateOCOFailsafeMixin），主周期慢不影响保命。

【方向】
    多空双向 (can_short=True)。震荡区间上下沿都是机会，只做多会浪费上沿的
    做空机会，而且单边只做多在下跌趋势里死得更惨。

【持仓风格】
    日内到短波段。区间回归通常在数根到数十根 15m K 线内完成（几小时到 1~2 天）。
    如果一笔持仓超过 2~3 天还没回归，往往意味着"区间假设已经破了"——
    这正是最危险的状态，此时马丁已加满仓，只能等中断线砍仓。

【止盈止损想法 —— 为什么这样设计】
    止盈：分层。移动止损锁半利 + RSI 回中轨主动离场。区间策略的利润来自
         高频小胜，不能等大行情，早止盈是刻意的。
    止损：不用固定百分比止损，而是"马丁中断线"。
         逻辑是：加仓期间不认赔（这是马丁的前提），但加仓次数一旦耗尽
         (超过 dca_max_entries) 且价格继续深跌到中断线，就市价砍仓认赔。
         中断线 = 满仓总回撤 + 余量，并被"强平安全线"强制收紧
         (见 RiskExtMixin._safe_interrupt_level)，确保永远先于交易所强平触发。
    双保险：交易所端挂 OCO 括号单(TP+SL)，即使 bot 进程崩溃，止损仍在交易所侧生效。

================================================================================
                        ⚠️  资金安全与失效场景（务必读完）  ⚠️
================================================================================
    见文件末尾 __doc__ 之外的详细风险注释块。核心一句话：
    马丁格尔的收益曲线是"长期缓慢爬升 + 偶发断崖式归零"。它不是"稳"，
    是"把亏损延后并集中到一次爆发"。用它必须接受"某天可能单笔吃掉数月利润"。
================================================================================
"""

from datetime import datetime
from typing import Optional

from pandas import DataFrame
from freqtrade.persistence import Trade

from gate_oco import GateOCOFailsafeMixin
from risk_ext import RiskExtMixin
# 使用中文文件名导入完整基类（含 BB/ATR_SPIKE_BLOCK 等完整参数）
try:
    from Meme_马丁_基类 import MemeMartingaleBaseStrategy
except ImportError:
    from meme_martingale_base import MemeMartingaleBaseStrategy


class MemeVolatilityGridMartingaleStrategy(
        GateOCOFailsafeMixin, RiskExtMixin, MemeMartingaleBaseStrategy):
    """
    组合顺序 (MRO)：OCO故障保护 → 风控扩展 → 马丁基类 → IStrategy
    为什么是这个顺序：
        OCO 钩子要最先介入订单生命周期；风控扩展(取价/强平校验)次之，
        供基类的加仓/止损逻辑调用；基类兜底核心马丁逻辑。三层的 __init__
        都调 super()，链路完整。
    """

    dca_tag_prefix = "vol_grid_dca"

    # ================== OCO 交易所端故障保护参数 ==================
    oco_enabled = True
    # 止盈距离(占均价)。调优：6%→8%，妖币区间振幅大，6%太早止盈截断利润。
    oco_tp_pct = 0.08
    # SL 相对"中断线"的缓冲倍数。为什么 >1：交易所端 SL 是最后防线，
    # 要略宽于 bot 内部中断线，避免两者同时抢触发导致重复/冲突平仓。
    oco_sl_buffer = 1.15

    # ================== 回测风控参数 ==================
    # 悲观滑点。为什么 0.5%：Meme 币盘口薄，市价加仓/中断砍仓的实际成交
    # 常比理论价差 0.3~1%。回测不建模滑点会严重高估收益。做敏感性测试时
    # 把它调到 1% 再跑一遍，若策略在 1% 滑点下就亏，实盘几乎必亏。
    SLIPPAGE_PCT = 0.005
    # 维持保证金率。gate Meme 档位保守取 0.5%，用于回测估算强平价。
    # 实盘会被真实 liquidationPrice 覆盖，此值仅回测兜底。
    MAINTENANCE_MARGIN_RATE = 0.005

    # ★★★ 关键修复：启用 OCO 时必须关掉交易所原生 stoploss ★★★
    # 为什么：若同时开 stoploss_on_exchange=True 和 OCO 的 SL 条件单，
    # 交易所端会存在两条止损单，一条触发后另一条变成"裸露的反向单"，
    # 可能被误当成开仓单执行，造成反向持仓。二者只能留一个，这里选 OCO。
    order_types = {
        **MemeMartingaleBaseStrategy.order_types,
        "stoploss_on_exchange": False,
    }

    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # --- 调优v2 核心改进 ---
        # 1) ATR 过滤改用 atr_ratio（相对均值倍数）替代绝对值：
        #    原版 atr_pct < dca_step*2=0.12 把高波动行情全过滤了——
        #    但妖币正常波动时 atr_pct 就有 0.08~0.15，策略设计目标恰恰是高波动。
        #    改用 atr_ratio < 2.5 区分"正常高波动"和"异常爆炸"（瀑布/逼空）
        # 2) 增加趋势过滤：EMA 斜率方向化，不在强趋势中逆势入场
        #    原版只在 DCA 加仓时做趋势过滤，入场时没有，导致在单边趋势中频繁开仓
        # 3) RSI 保持 30/70 不变：妖币 RSI 确实少到 30 以下，但放宽到 35 会引入
        #    更多趋势中的假信号，得不偿失。改用 ATR ratio 放宽即可。
        long_cond = (
            (dataframe["rsi"] < 30)
            & (dataframe["close"] < dataframe["ema_slow"])
            & (dataframe["atr_ratio"] < 2.5)
            & (dataframe["ema_slope"] > -0.01)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[long_cond, "enter_long"] = 1

        # --- 做空：区间上沿超买（镜像逻辑）---
        short_cond = (
            (dataframe["rsi"] > 70)
            & (dataframe["close"] > dataframe["ema_slow"])
            & (dataframe["atr_ratio"] < 2.5)
            & (dataframe["ema_slope"] < 0.01)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[short_cond, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        # 调优v3：RSI 回中轨阈值 60/40→57/43，在"让利润多跑"和"落袋为安"之间取平衡。
        # 60/40 对低波动币种（XCN/BEAT）太宽，小赢被回吐变亏损；
        # 55/45 对高波动币种（VELVET/H）太紧，大利润被截断。
        # 57/43 是折中：比原版稍宽让利润跑，但不至于回吐太多。
        dataframe.loc[dataframe["rsi"] > 57, "exit_long"] = 1
        dataframe.loc[dataframe["rsi"] < 43, "exit_short"] = 1
        return dataframe

    # ==================================================================
    # OCO 生命周期挂钩 —— 为什么每个钩子都要有
    # ==================================================================
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        # 每轮轮询对账：交易所端某一腿(TP/SL)触发后，主动撤掉另一腿。
        # 为什么必须做：gate 的两条独立条件单不是原生 OCO，一腿成交后
        # 另一腿不会自动取消，不撤就会变成裸单。用 try 包裹是因为对账失败
        # 不应中断主循环（下一轮会重试）。
        try:
            self.sync_brackets()
        except Exception:
            pass

    def bot_start(self, **kwargs) -> None:
        # 启动时对账：清理上次进程崩溃残留的孤儿条件单。
        # 为什么关键：马丁场景下 bot 崩溃重启很常见，若不清理，旧的 SL/TP
        # 单会和新挂的单叠加，重复平仓或方向错乱。
        try:
            self.reconcile_orphans()
        except Exception:
            pass

    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime,
                     **kwargs) -> None:
        # 每次入场/加仓成交后重挂括号单。
        # 为什么：马丁加仓会改变持仓均价和数量，旧的 TP/SL 价位和数量都失效，
        # 必须"撤旧→按新均价新数量重挂"，否则 SL 覆盖不全仓、TP 价位错位。
        if not self.oco_enabled:
            return
        try:
            if order.ft_order_side == trade.entry_side:
                self.cancel_bracket(trade)
                self.place_bracket(trade)
        except Exception:
            pass

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str,
                           amount: float, rate: float, time_in_force: str,
                           exit_reason: str, current_time: datetime,
                           **kwargs) -> bool:
        # 主动离场前先撤交易所端括号单。
        # 为什么：bot 主动平仓后，若不撤 OCO 单，它们会成为无持仓对应的裸单，
        # 下次价格触及时凭空开出一个反向仓位。
        try:
            self.cancel_bracket(trade)
        except Exception:
            pass
        return True