# -*- coding: utf-8 -*-
"""
Meme 有限 DCA 马丁 子策略 (MemeLimitedMartingaleStrategy)
==================================================================
【继承顺序极其重要 — MRO】
    class MemeLimitedMartingaleStrategy(GateOCOFailsafeMixin, MemeMartingaleBaseStrategy)
    Mixin 必须放在 Base 左边。否则 super() 链断裂：
      - GateOCOFailsafeMixin.bot_loop_start / order_filled / custom_exit
        内部都靠 super() 回调到 Base 的熔断、订单簿、时间止损逻辑；
      - 顺序写反 → OCO 钩子覆盖掉 Base 逻辑，熔断与时间止损静默失效。

标的：    gate.io 永续 Meme/妖币(PEPE/WIF/BONK…) 与主流币永续。
方向：    主做多(暴跌抄底/均值回归)；可选做空(allow_short，默认关，原因见文末)。
持仓风格：波段(15m，单笔最长由 time_stop≈48h 约束)。
适合行情：高波动【超跌反弹 / 区间震荡】。
失效行情：单边瀑布、趋势性阴跌(不回归) —— 这是马丁的天敌，见第四节。
核心想法：跌破布林下轨 + RSI 超卖时分批抄底，价格回归中轨/RSI 转强止盈；
          一旦识别为单边(ATR 爆炸 / EMA 趋势向下)，停止加仓并中断平仓保命。

============================= 结构性参数（一般固定） =============================
参数                     含义                     默认值         影响
-----------------------------------------------------------------------------------
timeframe                周期                     15m            信号质量/反应速度折中；改 1m 易被插针扫损
stoploss                 灾难级硬止损             -0.25→建议-0.40 见 Bug#3，过浅会让中断逻辑失效
MAX_OPEN_TRADES_HARD     最大同时持仓             20             总风险敞口上限
RESERVE_MARGIN_RATIO     预留保证金               0.30           安全垫，永不动用
CORE/SATELLITE_RATIO     核心/卫星资金占比        0.5/0.5        高低风险分配
SINGLE_TRADE_RISK_CAP    单笔风险上限             0.10           防单笔吃满
CONSECUTIVE_LOSS_LIMIT   连亏熔断                 5              防策略集体失效继续放血
DAILY_DRAWDOWN_LIMIT     单日回撤熔断             0.30           当日止血
ATR_SPIKE_BLOCK          ATR 爆炸禁加倍数         3.0            单边瀑布识别
DCA_INTERRUPT_EXTRA_LOSS 加仓耗尽后再亏N中断      0.08           中断平仓触发点

========================= 行情敏感参数（Hyperopt调优） ==========================
参数                   含义                   默认      建议范围    影响
--------------------------------------------------------------------------------
buy_atr_pct_min        入场最低波动率         0.006     0.003-0.020 过高=很少开仓；过低=死盘也进
buy_rsi_max            入场RSI上限            35        20-45       越低越严格，信号越少越优质
buy_range_pos_min      区间底过滤             0.12      0.05-0.35   越高越能避开"跌穿区间"的飞刀
dca_max_entries        最大加仓次数           3         1-4         马丁重灾区：越大越能摊低、爆亏也越大
dca_step_pct           加仓步长               0.06      0.04-0.12   越小加仓越密(更危险)，越大越稀疏
dca_cooldown_min       加仓冷却               45        15-120      防瀑布途中连续踩刀
take_profit_pct        止盈阈值               0.04      0.02-0.10   负偏度策略不宜太贪
sell_rsi_min           止盈RSI                62        55-75       均值回归离场点
time_stop_candles      时间止损(根)           192(48h)  48-288      防死单长期锁死保证金
base_leverage          基础杠杆               3         1-6         与中断阈值强耦合(见Bug#3)，不建议Hyperopt
signal_ttl_min         外部信号有效期         15        5-60        防断线后用陈旧信号开仓 

"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    stoploss_from_open,
)

# 这两个文件即附件；按你的工程实际模块路径调整 import
from gate_oco import GateOCOFailsafeMixin
# from Meme_马丁_基类 import MemeMartingaleBaseStrategy   # 文件名含中文时建议改成英文模块名
from Meme_martingale_base import MemeMartingaleBaseStrategy

logger = logging.getLogger(__name__)


class MemeLimitedMartingaleStrategy(GateOCOFailsafeMixin, MemeMartingaleBaseStrategy):

    # === OCO 与硬止损的冲突修正（关键，见第二节 Bug#4）=====================
    # 基类 order_types 里 stoploss_on_exchange=True，而 OCO Mixin 又会在
    # 交易所端额外挂一条 reduceOnly 止损单 → 同一笔仓位出现【两条交易所止损单】。
    # 这会导致：一条触发后另一条变成"反向裸单/被拒"，甚至重复平仓。
    # 解决：启用 OCO 时，把 freqtrade 自带的 stoploss_on_exchange 关掉，
    #       止损完全交给 OCO 的 SL 腿 + 本地 custom_stoploss 兜底。
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,   # ← 由 True 改为 False，避免与 OCO 重复
        "stoploss_on_exchange_interval": 60,
    }

    # =====================================================================
    # 子策略专属可调参数
    # =====================================================================
    allow_short = BooleanParameter(default=False, space="buy", optimize=False, load=True)
    # range_position 下限：过滤"已经跌穿区间底部"的死亡下跌（接飞刀过滤）
    buy_range_pos_min = DecimalParameter(0.05, 0.35, default=0.12, space="buy",
                                         decimals=2, optimize=True, load=True)
    # 是否启用外部信号通道
    use_external_signal = BooleanParameter(default=True, space="buy",
                                           optimize=False, load=True)
    # 外部信号有效期（分钟）：过期信号不执行，防止断线后用陈旧信号开仓
    signal_ttl_min = IntParameter(5, 60, default=15, space="buy",
                                  optimize=False, load=True)

    # =====================================================================
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 外部信号寄存器：{pair: {"dir": "long"/"short", "expire": datetime, "tag": str}}
        # 为什么用内存 dict + 锁？webhook/外部进程在另一线程写入，
        # freqtrade 策略回调在主循环线程读取，需加锁防竞态。
        # 为什么不持久化？信号是实时短时效数据，重启后陈旧信号本就该作废。
        self._external_signals: dict = {}
        self._signal_lock = threading.Lock()

    # =====================================================================
    # 信号接收接口（供外部 webhook / 量化平台 / 人工调用）
    # =====================================================================
    def push_signal(self, pair: str, direction: str, tag: str = "ext") -> None:
        """
        外部信号入口。示例（在你的 Flask/Webhook 线程里）：
            strategy.push_signal("PEPE/USDT:USDT", "long", tag="tv_alert")

        注意：因 process_only_new_candles=True，信号会在【下一根 15m K 线收盘】时
        被 populate_entry_trend 读到并执行，不是 tick 级即时成交。
        这是有意为之——15m 收盘确认能滤掉插针误触；若要 tick 级即时，
        需把 process_only_new_candles 设为 False，但会引入盘中信号反复跳动的风险（权衡）。
        """
        if direction not in ("long", "short"):
            logger.warning("[SIGNAL] 非法方向 %s，忽略", direction)
            return
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.signal_ttl_min.value)
        with self._signal_lock:
            self._external_signals[pair] = {"dir": direction, "expire": expire, "tag": tag}
        logger.info("[SIGNAL] 收到外部信号 %s %s tag=%s 有效至 %s",
                    pair, direction, tag, expire)

    def _fresh_signal(self, pair: str) -> Optional[dict]:
        """读取并校验信号时效；过期则丢弃。"""
        if not self.use_external_signal.value:
            return None
        with self._signal_lock:
            sig = self._external_signals.get(pair)
            if not sig:
                return None
            if datetime.now(timezone.utc) > sig["expire"]:
                self._external_signals.pop(pair, None)  # 过期清理
                return None
            return sig

    # =====================================================================
    # 入场：均值回归抄底（多）+ 可选做空 + 外部信号注入
    # =====================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_entry_trend(dataframe, metadata)  # 初始化 enter_long/enter_tag
        df["enter_short"] = 0  # 基类未初始化 short 列，这里补上

        pair = metadata["pair"]

        # --- 做多：超跌 + 超卖 + 有波动但非瀑布 + 非陡降趋势 ---
        long_cond = (
            (df["close"] < df["bb_lower"]) &                       # 跌破布林下轨=超卖
            (df["rsi"] < self.buy_rsi_max.value) &                 # RSI 超卖
            (df["atr_pct"] > self.buy_atr_pct_min.value) &         # 死盘(无波动)不做
            (df["atr_ratio"] < self.ATR_SPIKE_BLOCK) &             # ATR 未爆炸(非瀑布)
            (df["range_position"] > self.buy_range_pos_min.value) &# 没跌穿区间底(防接飞刀)
            (df["ema_slope"] > -0.01) &                            # 趋势过滤:非陡峭下跌
            (df["volume"] > 0)
        )
        df.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "bb_dip_long")

        # --- 做空（镜像，默认关；开启前请重读基类 DCA 的趋势过滤问题，见 Bug#6）---
        if self.allow_short.value:
            short_cond = (
                (df["close"] > df["bb_upper"]) &
                (df["rsi"] > self.sell_rsi_min.value) &
                (df["atr_pct"] > self.buy_atr_pct_min.value) &
                (df["atr_ratio"] < self.ATR_SPIKE_BLOCK) &
                (df["range_position"] < (1 - self.buy_range_pos_min.value)) &
                (df["ema_slope"] < 0.01) &
                (df["volume"] > 0)
            )
            df.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "bb_fade_short")

        # --- 外部信号注入：只作用于【最后一根 K 线】 ---
        # 为什么只改最后一行？回测时 _external_signals 为空(不会注入)，
        # 实盘只在最新已收盘 K 线注入 → 从根本上杜绝"历史信号"造成的未来函数。
        sig = self._fresh_signal(pair)
        if sig is not None and len(df) > 0:
            idx = df.index[-1]
            if sig["dir"] == "long":
                df.loc[idx, ["enter_long", "enter_tag"]] = (1, f"ext_{sig['tag']}")
            elif sig["dir"] == "short" and self.allow_short.value:
                df.loc[idx, ["enter_short", "enter_tag"]] = (1, f"ext_{sig['tag']}")

        return df

    # =====================================================================
    # 出场：回归中轨 / RSI 转强 → 止盈信号（custom_exit/止损另有兜底）
    # =====================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_exit_trend(dataframe, metadata)
        df["exit_short"] = 0

        # 多头止盈：价格回到布林中轨(均值回归目标) 或 RSI 超买
        exit_long = (
            (df["close"] >= df["bb_mid"]) | (df["rsi"] > self.sell_rsi_min.value)
        )
        df.loc[exit_long, ["exit_long", "exit_tag"]] = (1, "mean_revert_tp")

        if self.allow_short.value:
            exit_short = (
                (df["close"] <= df["bb_mid"]) | (df["rsi"] < self.buy_rsi_max.value)
            )
            df.loc[exit_short, ["exit_short", "exit_tag"]] = (1, "mean_revert_tp_s")

        return df

    # =====================================================================
    # 动态止损修正版（覆盖基类 Bug#2、Bug#3）
    # =====================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:
        """
        修正要点：
        (A) freqtrade 传入的 current_profit 是【含杠杆】的收益率(price_move * leverage)，
            而 stoploss/阈值常按"价格变动"思考。这里统一先还原成价格层面 price_profit，
            让中断阈值与杠杆解耦——否则阈值的真实触发点会随杠杆漂移(见第四节)。
        (B) 基类用 `return current_profit * 0.5` 做移动止盈是错误的：
            custom_stoploss 的返回值是【相对现价的止损比例】，传正数会把止损设到现价
            上方→对多头瞬间触发；必须用 stoploss_from_open() 正确换算。
        """
        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
        price_profit = current_profit / lev  # 还原为价格层面收益率
        filled = trade.nr_of_successful_entries

        # --- 马丁中断平仓：加仓耗尽 + 价格继续深跌 → 市价砍仓保命 ---
        # 这是全策略最关键的一行风控：阻断"越跌越买直到爆仓"。
        if filled > self.dca_max_entries.value:
            interrupt_price = -(self.dca_step_pct.value * self.dca_max_entries.value
                                + self.DCA_INTERRUPT_EXTRA_LOSS)  # 价格层面阈值
            if price_profit < interrupt_price:
                logger.warning("[%s] 马丁中断平仓 price_profit=%.3f (lev=%.1f)",
                               pair, price_profit, lev)
                return 0.0001  # 贴近现价 → stoploss=market 立即离场(保命优先)

        # --- 移动止盈锁利：盈利达标后，把止损上移到"已实现一半利润"处 ---
        if price_profit > self.take_profit_pct.value:
            new_sl = stoploss_from_open(
                current_profit * 0.5,   # 锁定当前(含杠杆)利润的一半
                current_profit,
                is_short=trade.is_short,
                leverage=lev,
            )
            # stoploss_from_open 在极端情况下可能返回 0；返回 None 退回硬止损更安全
            return new_sl if new_sl and new_sl != 0 else None

        return None  # 其余维持硬止损(基类 stoploss，建议改深，见 Bug#3)

    # =====================================================================
    # 开仓闸门修正版（覆盖基类 Bug#1：避免误杀 DCA 加仓）
    # =====================================================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime,
                            entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        版本无关的健壮写法：
        - 不同 freqtrade 版本中，confirm_trade_entry 是否对【加仓(pos_adjust)】回调
          并不一致。若回调，则基类原版的 "已达 MAX_OPEN_TRADES / 已有持仓则拒绝"
          会把马丁加仓也一并拦死（持仓数已含本单、同 pair 已有单）。
        - 这里显式区分：已有同 pair 持仓 ⇒ 视为加仓/异常重发，
          除"熔断期停手"外一律放行；MAX/重复 检查只对【全新开仓】生效。
        """
        # 1) 熔断冷静期：对新仓与加仓都拦截
        #    为什么连加仓也停？熔断往往发生在单边行情，此时继续马丁加仓 = 加速爆仓。
        if self._cooldown_until and current_time < self._cooldown_until:
            logger.info("[%s] 熔断冷静期，拒绝开/加仓", pair)
            return False

        # 2) 已有同 pair 持仓 → 这是加仓或重连重发，放行（freqtrade 不会重复新开同 pair）
        if Trade.get_trades_proxy(pair=pair, is_open=True):
            return True

        # 3) 以下仅对【新开仓】生效
        if Trade.get_open_trade_count() >= self.MAX_OPEN_TRADES_HARD:
            logger.info("[%s] 已达最大持仓 %d，拒绝新开仓", pair, self.MAX_OPEN_TRADES_HARD)
            return False

        return True