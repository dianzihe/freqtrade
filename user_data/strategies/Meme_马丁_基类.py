# -*- coding: utf-8 -*-
"""
Meme 马丁策略 工程化基类 (MemeMartingaleBaseStrategy) —— 修正版 v2
=============================================================
本版相对原始基类的改动(均标注 [FIX#n])：
  [FIX#1] confirm_trade_entry 区分"新开仓 / 加仓"，避免满仓后误杀马丁加仓。
  [FIX#2] custom_stoploss 移动止盈改用 stoploss_from_open()，
          原版 `return current_profit*0.5` 传正数会把多头止损设到现价上方 → 秒平。
  [FIX#3] 硬 stoploss 由 -0.25 改深为 -0.40，确保任何杠杆下 custom_stoploss
          都先于硬止损生效；否则 1x 时中断逻辑变死代码。
  [FIX#4] order_types 增加 stoploss_on_exchange 开关说明：与 OCO Mixin 同用时，
          子策略需把它关掉，避免交易所端双重止损单。
  [FIX#5] adjust_trade_position 用 trade.orders[0].cost 取首仓金额，
          原版 .stake_amount 属性在 Order 模型上不存在 → 每次抛异常走回退分支。
  [FIX#6] DCA 趋势过滤按 trade.is_short 方向化，避免做空时被反向逻辑禁加。
  [FIX#7] 统一"价格层面"口径：current_profit 含杠杆，DCA/中断阈值按价格思考，
          先除以 leverage 还原，使触发点不随杠杆漂移。

核心设计哲学：
  马丁格尔本身是负偏度策略(小赚多次、爆亏一次)。本基类不试图消除这个特性，
  而是用多层熔断把"爆亏一次"的尾部风险砍掉。所有风控逻辑优先级都高于交易逻辑——
  宁可错过回归，也不在单边行情里把本金喂进去。
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    stoploss_from_open,   # [FIX#2] 正确换算"相对现价止损比例"
)

logger = logging.getLogger(__name__)


class MemeMartingaleBaseStrategy(IStrategy):
    """
    所有 Meme 马丁子策略的公共底座：
    指标计算、加仓、止盈止损、杠杆、熔断、仓位管理、异常处理 全部收敛到这里，
    子策略只负责"什么时候开第一仓"和"什么时候正常止盈"。
    """

    # === 结构性参数（一般固定，不参与 Hyperopt）===========================
    timeframe = "15m"            # 中频。1m 噪音太大易被插针扫损；1h 对 Meme 反应太慢。
    can_short = True             # 永续合约支持双向；现货模式下 freqtrade 会自动忽略做空。
    process_only_new_candles = True   # 只在新 K 线收盘后计算 → 防未来函数+省算力。
                                      # ⚠️ 切勿改 False：否则 leverage/adjust 里 iloc[-1]
                                      # 会取到未收盘 K 线，ATR/RSI 随盘抖动 = 引入未来函数。
    use_custom_stoploss = True
    position_adjustment_enable = True
    startup_candle_count = 200   # 指标预热：BB(20)+ATR(14)+EMA(100)，留足 200 根防冷启动 NaN

    # [FIX#3] 硬止损 -0.40。保持不变。
    # 中断阈值/止盈都交给 custom_stoploss 精细控制；硬止损只是"兜底双保险"。
    stoploss = -0.40

    # 默认 ROI（子类可覆盖）。调优v3：0.10→0.12，适度让利润多跑。
    minimal_roi = {"0": 0.12}

    # [FIX#4] 滑点/挂单 + 止损上交易所端。
    # ⚠️ 与 GateOCOFailsafeMixin 同时使用时，OCO 会在交易所端另挂一条 SL 触发单，
    #    若此处 stoploss_on_exchange 仍为 True → 同一仓位出现两条交易所止损单，
    #    一条成交后另一条可能变裸单/被拒/重复平仓。
    #    因此【启用 OCO 的子策略必须覆盖本字典，把 stoploss_on_exchange 设为 False】。
    #    这里默认 True 是为了"不挂 OCO 的纯基类用法"仍有交易所端兜底。
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",          # 止损必须市价，保命优先于价格
        "stoploss_on_exchange": True,  # 见上方警告：用 OCO 时子策略改 False
        "stoploss_on_exchange_interval": 60,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # =====================================================================
    # 行情敏感参数（参与 Hyperopt 调优）
    # =====================================================================
    # --- 入场 ---
    buy_atr_pct_min = DecimalParameter(0.003, 0.020, default=0.006, space="buy",
                                       decimals=4, optimize=True, load=True)
    buy_rsi_max = IntParameter(20, 45, default=35, space="buy", optimize=True, load=True)

    # --- 加仓（马丁核心，调优重灾区）---
    # dca_step_pct 保持 0.06：调优v2 曾试 0.08 但导致 DCA 触发太晚，摊薄不足。
    dca_max_entries = IntParameter(1, 4, default=3, space="buy", optimize=True, load=True)
    dca_step_pct = DecimalParameter(0.04, 0.12, default=0.06, space="buy",
                                    decimals=3, optimize=True, load=True)
    dca_cooldown_min = IntParameter(15, 120, default=45, space="buy", optimize=True, load=True)

    # --- 止盈止损 ---
    # 调优v2：take_profit_pct 0.04→0.06，让移动止盈在更高位置启动。
    #   原版 0.04 时 3x 杠杆下价格刚动 1.3% 就触发移动止盈，截断利润。
    sell_rsi_min = IntParameter(55, 75, default=62, space="sell", optimize=True, load=True)
    take_profit_pct = DecimalParameter(0.02, 0.10, default=0.06, space="sell",
                                       decimals=3, optimize=True, load=True)
    time_stop_candles = IntParameter(48, 288, default=192, space="sell",
                                     optimize=True, load=True)  # 192*15m = 48h

    # --- 杠杆 / 仓位 ---
    # [FIX#3/#7] 杠杆与中断阈值强耦合，不建议 Hyperopt，固定后再上实盘。
    base_leverage = IntParameter(1, 6, default=3, space="buy", optimize=False, load=True)

    # =====================================================================
    # 不可调的"硬"风控常量（写死，避免被 Hyperopt 调成危险值）
    # =====================================================================
    MAX_OPEN_TRADES_HARD = 20          # 最大同时持仓
    RESERVE_MARGIN_RATIO = 0.30        # 预留 30% 保证金作安全垫，永不动用
    CORE_RATIO = 0.50                  # 核心仓资金占比（低风险）
    SATELLITE_RATIO = 0.50             # 卫星仓资金占比（高风险马丁）
    SINGLE_TRADE_RISK_CAP = 0.10       # 单笔最大风险占总资金 10%
    CONSECUTIVE_LOSS_LIMIT = 5         # 连续亏损熔断阈值
    DAILY_DRAWDOWN_LIMIT = 0.30        # 单日回撤熔断阈值 30%
    COOLDOWN_MINUTES = 30              # 熔断冷静期
    DCA_INTERRUPT_EXTRA_LOSS = 0.05    # 调优v2：0.08→0.05，加仓耗尽后更早认赔止损
    ATR_SPIKE_BLOCK = 3.0              # ATR 突放大到均值 3 倍 → 禁止加仓（防单边）
    # 调优v2 关键修复：马丁中断线改用含杠杆利润直接判断。
    # 原版用 price_profit（除以杠杆）比较中断阈值，在 3x 杠杆下：
    #   中断线(价格) = -(0.06*3+0.08) = -0.26
    #   硬止损(价格) = -0.40/3 = -0.133
    # 硬止损永远先触发 → 中断线是死代码 → 亏损单全部到 -40%。
    # 改用 current_profit（含杠杆）直接判断，-0.20 < -0.40，中断线始终先触发。
    MAX_LEVERAGED_LOSS = 0.20          # 加仓耗尽后，含杠杆亏损达 20% → 中断平仓
    EMA_SLOPE_BLOCK = 0.01             # [FIX#6] 趋势过滤斜率阈值（抽成常量便于复用）

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 本地"熔断状态机"：记录冷静期截止时间。
        # 放内存而非 DB：冷静期是短时状态，重启后重新评估更安全（避免脏状态卡死）。
        self._cooldown_until: Optional[datetime] = None
        # 本地订单簿：模拟 Bracket Order 的子单跟踪。
        self._local_order_book: dict = {}

    # =====================================================================
    # 工具：把含杠杆收益率还原成"价格层面"收益率   [FIX#7]
    # =====================================================================
    def _price_profit(self, trade: Trade, current_profit: float) -> float:
        """
        freqtrade 的 current_profit 是【含杠杆】收益率(price_move * leverage)。
        而 DCA 步长、中断阈值都按"价格变动"设计。若直接拿 current_profit 比阈值，
        触发点会随杠杆漂移（3x 时同一阈值对应的价格变动只有 1/3）。
        统一先除以 leverage 还原到价格口径，使阈值与杠杆解耦。
        """
        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
        return current_profit / lev

    # =====================================================================
    # 1. 指标计算（无未来函数）
    # =====================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        所有指标只使用截至当前 K 线及之前的数据；不使用 bfill / shift(-n)。
        process_only_new_candles=True 保证盘中不会用未收盘数据计算 → 杜绝未来函数。
        """
        # 布林带：均值回归核心。下轨=超卖参考，中轨=回归目标。
        bb_period, bb_std = 20, 2.0
        ma = dataframe["close"].rolling(bb_period).mean()
        sd = dataframe["close"].rolling(bb_period).std()
        dataframe["bb_mid"] = ma
        dataframe["bb_lower"] = ma - bb_std * sd
        dataframe["bb_upper"] = ma + bb_std * sd

        # ATR 百分比：衡量波动率。太低=死盘；太高=单边瀑布要规避。
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        # ATR 相对自身均值的倍数，用于识别"波动率突然爆炸"（瀑布行情）
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["atr"].rolling(50).mean()

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # range_position：当前价在近 N 根高低区间的相对位置 [0,1]。
        # 0=区间底部，1=区间顶部。用于过滤"已经跌穿区间"的死亡下跌。
        n = 48
        roll_low = dataframe["low"].rolling(n).min()
        roll_high = dataframe["high"].rolling(n).max()
        dataframe["range_position"] = (
            (dataframe["close"] - roll_low) / (roll_high - roll_low + 1e-9)
        )

        # 趋势过滤：长周期 EMA 斜率。马丁最怕"持续下跌趋势"，用它禁止下跌趋势中做多。
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_slope"] = dataframe["ema_slow"].diff(5) / dataframe["ema_slow"]

        return dataframe

    # =====================================================================
    # 2. 入场（子类调用 super() 后追加自己的信号）
    # =====================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe

    # =====================================================================
    # 3. 杠杆动态管理：波动率越高，杠杆越低
    # =====================================================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        为什么动态降杠杆？固定杠杆在高波动期等于放大爆仓概率。
        ATR 越高说明插针/瀑布风险越大，自动把杠杆往下压，逆周期控险。
        注意：因 process_only_new_candles=True，iloc[-1] 取的是【已收盘】K 线，安全。
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        lev = float(self.base_leverage.value)
        if df is not None and len(df) > 0:
            atr_pct = df["atr_pct"].iloc[-1]
            if atr_pct > 0.015:
                lev = 1.0          # 极端波动只用 1 倍
            elif atr_pct > 0.010:
                lev = min(lev, 2.0)
            elif atr_pct > 0.007:
                lev = min(lev, 3.0)
        # 硬上限 6 倍，且不超过交易所允许的 max_leverage
        return float(min(lev, 6.0, max_leverage))

    # =====================================================================
    # 4. 仓位管理：核心+卫星 + 预留保证金 + 单笔风险上限
    # =====================================================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
        计算首仓金额。注意：这是"首仓"，后续马丁加仓在 adjust_trade_position 里。
        关键：必须为后续马丁加仓预留空间，否则首仓吃满 → 没钱加仓 → 马丁逻辑失效。
        """
        total = self.wallets.get_total_stake_amount()
        available = total * (1.0 - self.RESERVE_MARGIN_RATIO)  # 砍掉 30% 安全垫

        # 卫星仓（高风险马丁）只能用一半资金，再按最大加仓次数预留份额。
        satellite_budget = available * self.SATELLITE_RATIO
        # 用倍数序列估算总弹药需求，反推首仓
        # 调优v2：与 adjust_trade_position 的 1+0.5*i 倍数序列保持一致
        mult_seq = [1.0] + [1.0 + 0.5 * i for i in range(1, self.dca_max_entries.value + 1)]
        first_stake = satellite_budget / (sum(mult_seq) * max(self.MAX_OPEN_TRADES_HARD // 4, 1))

        # 单笔风险上限：首仓名义敞口不得超过总资金的 10%
        risk_cap = total * self.SINGLE_TRADE_RISK_CAP / max(leverage, 1)
        stake = min(first_stake, risk_cap, max_stake)

        if min_stake:
            stake = max(stake, min_stake)
        return float(stake)

    # =====================================================================
    # 5. 加仓（马丁核心）+ 爆仓保护中断
    # =====================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        马丁加仓逻辑，带三重保护：
          (1) 加仓次数上限     —— 防止无限加仓
          (2) 加仓冷却时间     —— 防止瀑布途中连续加仓踩刀
          (3) ATR 急速放大禁加 —— 单边行情识别，直接停手
        中断平仓在 custom_stoploss 处理（这里只能加/减仓）。

        调优v2 关键修复：DCA 触发改用 current_profit（含杠杆）而非 price_profit。
        原版用 price_profit（除以杠杆），在 3x 杠杆下：
          DCA#1 需价格跌 6% → current_profit=-18% → 接近硬止损
          DCA#2 需价格跌 12% → current_profit=-36% → 已超硬止损，永远不触发
          DCA#3 需价格跌 18% → 不可能触发
        改用 current_profit 后，DCA 在含杠杆亏损 6%/12%/18% 时触发，
        与 MAX_LEVERAGED_LOSS=20% 中断线和 -40% 硬止损形成阶梯：
          DCA#1 @ -6% → DCA#2 @ -12% → DCA#3 @ -18% → 中断 @ -20% → 硬止损 @ -40%
        """
        filled_entries = trade.nr_of_successful_entries  # 已成功加仓次数（含首仓）

        # 已达上限 → 不再加仓
        if filled_entries > self.dca_max_entries.value:
            return None

        # 调优v2：DCA 触发改用 current_profit（含杠杆）
        # 这样 DCA 阈值不随杠杆变化产生不可达的问题
        # dca_step_pct=0.06 意味着含杠杆亏损 6% 触发第一次加仓

        # 浮盈时不加仓（马丁只在浮亏摊低成本时加）
        if current_profit > -self.dca_step_pct.value:
            return None

        # 检查是否触达下一档加仓阈值（阶梯式：-step, -2*step ...，含杠杆层面）
        next_trigger = -self.dca_step_pct.value * filled_entries
        if current_profit > next_trigger:
            return None

        # --- 保护(2)：加仓冷却 ---
        last_order_time = trade.date_last_filled_utc
        if last_order_time and (current_time - last_order_time) < timedelta(
            minutes=self.dca_cooldown_min.value
        ):
            return None

        # --- 保护(3)：ATR 爆炸 = 单边瀑布，禁止逆势加仓 ---
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df is not None and len(df) > 0:
            if df["atr_ratio"].iloc[-1] > self.ATR_SPIKE_BLOCK:
                logger.warning(f"[{trade.pair}] ATR 爆炸，禁止加仓（疑似单边行情）")
                return None
            # [FIX#6] 趋势过滤方向化：只在"逆着趋势加仓"时禁止。
            # 做多怕下跌趋势(slope<0)，做空怕上涨趋势(slope>0)。
            slope = df["ema_slope"].iloc[-1]
            if (not trade.is_short and slope < -self.EMA_SLOPE_BLOCK) or \
               (trade.is_short and slope > self.EMA_SLOPE_BLOCK):
                logger.info(f"[{trade.pair}] 逆势趋势中，禁止加仓 slope={slope:.4f}")
                return None

        # [FIX#5] 取首仓金额：Order 模型用 .cost（成交额），原版 .stake_amount 不存在。
        try:
            first_order = trade.orders[0]
            first_stake = first_order.cost or (
                (first_order.safe_filled or 0) * (first_order.safe_price or 0)
            )
            if not first_stake:  # 仍取不到则回退
                raise ValueError("first_stake empty")
        except Exception:
            first_stake = trade.stake_amount / max(filled_entries, 1)

        # 计算本次加仓金额：递增倍数
        # 调优v2：1+0.2*i → 1+0.5*i，增强摊薄效果。
        # 原版倍数太温和（1.0/1.2/1.4/1.6），3 级加仓后均价只降低 ~3%，
        # 对妖币 10%+ 的反向波动几乎无济于事。
        # 1+0.5*i（1.0/1.5/2.0/2.5）能让 3 级加仓后均价降低 ~6-8%，更有效。
        multiplier = 1.0 + 0.5 * filled_entries
        add_stake = first_stake * multiplier
        add_stake = min(add_stake, max_stake)

        if min_stake and add_stake < min_stake:
            return None

        logger.info(f"[{trade.pair}] 马丁加仓 #{filled_entries} "
                    f"price_profit={price_profit:.3f} stake={add_stake:.2f}")
        return add_stake

    # =====================================================================
    # 6. 动态止损：马丁中断平仓 + 渐进式移动止盈
    # =====================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:
        """
        返回值是"相对当前价的止损比例"。
        调优v2 关键修复：中断判断改用 current_profit（含杠杆），
        不再用 price_profit（除以杠杆），确保中断线在任何杠杆下都先于硬止损触发。
        """
        lev = max(getattr(trade, "leverage", 1.0) or 1.0, 1.0)
        price_profit = current_profit / lev          # 价格层面收益率（用于止盈判断）
        filled_entries = trade.nr_of_successful_entries

        # --- 马丁中断平仓：加仓耗尽 + 含杠杆亏损达阈值 → 市价砍仓保命 ---
        # 全策略最重要的风控：阻断"越跌越买直到爆仓"。
        # 调优v2：原版用 price_profit 比较，在杠杆>1 时硬止损先触发=死代码。
        #   改用 current_profit（含杠杆）直接判断，-0.20 始终先于 -0.40 硬止损。
        if filled_entries > self.dca_max_entries.value:
            if current_profit < -self.MAX_LEVERAGED_LOSS:
                logger.warning(f"[{pair}] 马丁中断平仓触发 "
                               f"current_profit={current_profit:.3f} (lev={lev:.1f}) "
                               f"filled={filled_entries}/{self.dca_max_entries.value}")
                return 0.0001  # 贴近现价 → stoploss=market 立即离场

        # --- 渐进式移动止盈：盈利达标后，把止损上移到"已实现一半利润"处 ---
        # [FIX#2] 原版 `return current_profit*0.5` 是错的：返回正数会把多头止损
        #         设到现价上方导致秒平。必须用 stoploss_from_open 换算成"相对现价比例"。
        if price_profit > self.take_profit_pct.value:
            new_sl = stoploss_from_open(
                current_profit * 0.5,   # 目标：锁定当前(含杠杆)利润的一半
                current_profit,
                is_short=trade.is_short,
                leverage=lev,
            )
            # 极端情况下可能返回 0；返回 None 退回硬止损更安全
            return new_sl if (new_sl and new_sl != 0) else None

        # 其余情况维持硬止损（返回 None = 用 stoploss = -0.40）
        return None

    # =====================================================================
    # 7. 自定义离场：时间止损
    # =====================================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        时间止损：持仓超过 N 根 K 线仍未达预期 → 离场。
        为什么需要它？马丁最怕"长期套牢占用保证金"，时间止损强制释放资金，
        让卫星仓的弹药不被一个死单长期锁死。
        注意：这里用 price 层面比较，与止盈口径一致。
        """
        held = current_time - trade.open_date_utc
        max_hold = timedelta(minutes=self.timeframe_to_minutes() * self.time_stop_candles.value)
        price_profit = self._price_profit(trade, current_profit)   # [FIX#7]
        if held > max_hold and price_profit < self.take_profit_pct.value:
            return "time_stop"
        return None

    def timeframe_to_minutes(self) -> int:
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        return mapping.get(self.timeframe, 15)

    # =====================================================================
    # 8. 熔断体系：连续亏损 + 单日回撤 → 冷静期
    # =====================================================================
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        每个 bot 循环开头评估熔断状态。集中、可观测，且只在主循环更新一次（避免重复计算）。
        """
        try:
            closed = Trade.get_trades_proxy(is_open=False)
        except Exception as e:
            logger.error(f"读取历史交易失败（DB异常）：{e}")
            return

        if not closed:
            return

        # 按平仓时间排序，取最近交易
        closed = sorted(
            closed,
            key=lambda t: t.close_date_utc or datetime.min.replace(tzinfo=timezone.utc)
        )

        # 连续亏损计数（从最近往前数）
        consec = 0
        for t in reversed(closed):
            if t.close_profit is not None and t.close_profit < 0:
                consec += 1
            else:
                break

        # 单日盈亏
        today = current_time.date()
        day_profit = sum(
            t.close_profit_abs or 0 for t in closed
            if t.close_date_utc and t.close_date_utc.date() == today
        )
        total = self.wallets.get_total_stake_amount() if self.wallets else 1
        day_dd = day_profit / total if total else 0

        if consec >= self.CONSECUTIVE_LOSS_LIMIT or day_dd < -self.DAILY_DRAWDOWN_LIMIT:
            self._cooldown_until = current_time + timedelta(minutes=self.COOLDOWN_MINUTES)
            logger.warning(
                f"熔断触发！连亏={consec} 单日回撤={day_dd:.2%} "
                f"冷静期至 {self._cooldown_until}"
            )

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        开仓最后一道闸门。
        [FIX#1] 区分"新开仓 / 加仓"：部分 freqtrade 版本会对加仓(pos_adjust)也回调本函数，
                原版的 "已达 MAX_OPEN_TRADES / 已有同 pair 持仓则拒绝" 会把马丁加仓一并拦死
                （加仓时持仓数已含本单、同 pair 也已有单）。
                修正：已有同 pair 持仓视为加仓/重连重发，除熔断外一律放行；
                      MAX/重复检查只对【全新开仓】生效。
        """
        # 1) 熔断冷静期：对新仓与加仓都拦截。
        #    为什么连加仓也停？熔断多发生在单边行情，此时继续马丁加仓 = 加速爆仓。
        if self._cooldown_until and current_time < self._cooldown_until:
            logger.info(f"[{pair}] 处于熔断冷静期，拒绝开/加仓")
            return False

        # 2) 已有同 pair 持仓 → 加仓或重连重发，放行
        #    （freqtrade 不会对同 pair 重复"新开"，故此处放行不会误开新仓）
        if Trade.get_trades_proxy(pair=pair, is_open=True):
            return True

        # 3) 以下仅对【新开仓】生效
        open_trades = Trade.get_open_trade_count()
        if open_trades >= self.MAX_OPEN_TRADES_HARD:
            logger.info(f"[{pair}] 已达最大持仓 {self.MAX_OPEN_TRADES_HARD}，拒绝新开仓")
            return False

        return True

    # =====================================================================
    # 9. 本地订单簿 / Bracket 模拟
    # =====================================================================
    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime,
                     **kwargs) -> None:
        """
        订单成交回调：在本地订单簿记录主单与子单关系，便于状态同步与对账。
        freqtrade 不原生支持交易所端 OCO/括号单，这里维护本地映射模拟"风险保护罩"。
        注意：[FIX#5] 用 order.safe_amount/safe_price，避免不存在的属性。
        """
        self._local_order_book[order.order_id] = {
            "pair": pair,
            "trade_id": trade.id,
            "side": order.ft_order_side,
            "price": order.safe_price,
            "amount": order.safe_amount,
            "status": order.status,
            "ts": current_time.isoformat(),
        }
        logger.info(f"订单簿更新: {order.order_id} {order.ft_order_side} @ {order.safe_price}")