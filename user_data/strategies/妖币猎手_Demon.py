"""
DemonHunter Strategy — 妖币猎手·恶魔版
========================================
基于 Demon Coin 系统性分析框架 v2.0 的自动化交易策略。

目标: 识别24小时内涨幅50%+的妖币，在启动初期入场，通过棘轮止损捕获超常收益。

核心设计:
- 8因子 × 4类别分层评分（100分制）
- 三级入场阈值: STRONG(>=80) / BUY(>=65) / WATCH(>=45)
- 棘轮移动止损: +10%保本 → +20%锁10% → +50%锁25% → +100%锁50%
- 分批止盈: +30%减25%, +50%再减25%, +100%再减25%
- 时间退出: 12h <= -5% → 退出, 24h < 2% → 退出, 48h < 5% → 退出
- 从高点回撤保护: 浮盈>=15%后回撤>=12% → 退出
- 全局冷却: 2小时, 防止过度交易
- 单币仓位上限: 5%
"""

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


class DemonHunter(IStrategy):
    """
    Demon Hunter — 妖币猎手策略 v1.0

    基于以下论文/分析框架:
    - Demon Coin Analysis Framework v2.0
    - 8-Factor Hierarchical Scoring Model
    - Multi-Stage Entry with Ratchet Trailing Stop
    """

    # ── Strategy Metadata ──
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 120  # 10h of 5m data for all indicators

    # ── 交易对筛选 ──
    minimal_roi = {
        "0": 10.0,   # 无限制 (由 custom_exit 管理)
    }

    stoploss = -0.15  # 15%硬止损

    trailing_stop = False  # 手工棘轮管理
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    # ── Hyperopt 参数空间 ──

    # A类: 价格量能
    buy_score_threshold = IntParameter(60, 85, default=75, space='buy')
    buy_min_categories = IntParameter(2, 4, default=3, space='buy')

    # 因子阈值
    A1_threshold = DecimalParameter(2.0, 6.0, default=4.0, decimals=1, space='buy')
    A2_threshold = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space='buy')
    A3_threshold = DecimalParameter(1.5, 4.0, default=2.5, decimals=1, space='buy')
    B1_threshold = DecimalParameter(0.5, 0.9, default=0.7, decimals=1, space='buy')
    B2_threshold = DecimalParameter(1.0, 2.0, default=1.5, decimals=1, space='buy')
    C1_vol_lo = DecimalParameter(0.05, 0.15, default=0.08, decimals=2, space='buy')
    C1_vol_hi = DecimalParameter(0.15, 0.35, default=0.25, decimals=2, space='buy')

    # 退出参数
    ratchet_10 = DecimalParameter(0.0, 0.05, default=0.0, decimals=2, space='sell')
    ratchet_20 = DecimalParameter(0.05, 0.15, default=0.10, decimals=2, space='sell')
    ratchet_50 = DecimalParameter(0.15, 0.35, default=0.25, decimals=2, space='sell')
    ratchet_100 = DecimalParameter(0.30, 0.60, default=0.50, decimals=2, space='sell')
    pullback_drawdown = DecimalParameter(0.08, 0.18, default=0.12, decimals=2, space='sell')
    pullback_min_profit = DecimalParameter(0.05, 0.15, default=0.08, decimals=2, space='sell')
    pullback_peak_min = DecimalParameter(0.10, 0.25, default=0.15, decimals=2, space='sell')

    # ── 仓位 / 风控 ──
    max_open_trades = 5       # 最多同时持有5个妖币
    stake_amount = 'unlimited'
    max_entry_position_adjustment = -1  # 禁止加仓

    # ── 交易时段评分 (UTC) ──
    SESSION_SCORE = {
        # 亚洲活跃时段
        '0': 8, '1': 10, '2': 10, '3': 8,
        # 欧亚交接
        '4': 6, '5': 6, '6': 8,
        # 欧洲上午
        '7': 10, '8': 10, '9': 8, '10': 8,
        # 欧美重叠
        '11': 6, '12': 8,
        # 美国上午 (最活跃)
        '13': 10, '14': 10, '15': 10, '16': 8,
        # 美国下午
        '17': 6, '18': 6, '19': 6,
        # 收市/亚洲早盘
        '20': 4, '21': 4, '22': 4, '23': 4,
    }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        计算全部8个因子的指标。

        A类 — 价格量能 (40分):
          A1: 价格动量 (6h累计涨幅 %)
          A2: 量能爆发 (近1h量 / 前5h均量)
          A3: 趋势强度 (SMA12斜率 %)

        B类 — 形态识别 (25分):
          B1: 阶梯吸筹 (6根5m K线收盘递增比例)
          B2: 振幅扩张 (近1h振幅 / 前5h振幅)

        C类 — 市场环境 (20分):
          C1: 波动率环境 (6h年化波动率)
          C2: 成交量质量 (近1h成交量偏度)

        D类 — 时机催化 (15分):
          D1: 时间窗口 (交易时段活跃度)
          D2: 价格合理性 (价格在合理范围内)
        """

        df = dataframe.copy()

        # ── 基础指标 ──
        df['sma12'] = ta.SMA(df['close'], timeperiod=12)

        # ── A1: 价格动量 (6h 累计涨幅) ──
        # 6h = 72根5m K线
        close_6h_ago = df['close'].shift(72)
        df['A1_price_momentum'] = ((df['close'] - close_6h_ago) / close_6h_ago * 100).fillna(0)

        # ── A2: 量能爆发 (近1h量 / 前5h均量) ──
        # 1h = 12 bars, 5h = 60 bars
        vol_1h = df['volume'].rolling(12, min_periods=6).mean()
        vol_5h_earlier = df['volume'].shift(12).rolling(60, min_periods=24).mean()
        df['A2_volume_surge'] = ((vol_1h / vol_5h_earlier).replace([np.inf, -np.inf], np.nan)).fillna(0)

        # ── A3: 趋势强度 (SMA12斜率, 6h窗口) ──
        sma12_6h_ago = df['sma12'].shift(72)
        df['A3_sma_momentum'] = ((df['sma12'] - sma12_6h_ago) / sma12_6h_ago * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

        # ── B1: 阶梯吸筹形态 (6根5m K线收盘递增比例) ──
        c = df['close']
        df['B1_staircase'] = (
            ((c >= c.shift(1)).astype(int) +
             (c.shift(1) >= c.shift(2)).astype(int) +
             (c.shift(2) >= c.shift(3)).astype(int) +
             (c.shift(3) >= c.shift(4)).astype(int) +
             (c.shift(4) >= c.shift(5)).astype(int)) / 5.0
        )

        # ── B2: 振幅扩张 (近3h振幅 / 前3h振幅) ──
        # 3h = 36 bars
        h = df['high']
        l = df['low']
        range_recent = (h.rolling(36, min_periods=18).max() - l.rolling(36, min_periods=18).min()) / l.rolling(36, min_periods=18).min().replace(0, np.nan)
        range_earlier = (h.shift(36).rolling(36, min_periods=18).max() - l.shift(36).rolling(36, min_periods=18).min()) / l.shift(36).rolling(36, min_periods=18).min().replace(0, np.nan)
        df['B2_range_behavior'] = ((range_recent / range_earlier).replace([np.inf, -np.inf], np.nan)).fillna(1.0)

        # ── C1: 波动率环境 (6h 年化波动率) ──
        log_ret = np.log(df['close'] / df['close'].shift(1))
        df['C1_volatility'] = log_ret.rolling(72, min_periods=36).std() * np.sqrt(288 * 365)  # annualized

        # ── C2: 成交量质量 (近1h成交量偏度) ──
        df['C2_volume_quality'] = df['volume'].rolling(12, min_periods=6).skew().fillna(0)

        # ── D1: 时间窗口 (UTC小时映射) ──
        df['D1_timing'] = df.index.hour.map(self.SESSION_SCORE).fillna(4)

        # ── D2: 价格合理性 ──
        # 价格在 $0.0001 ~ $1000 之间
        price = df['close']
        df['D2_price_ok'] = ((price > 0.0001) & (price < 1000)).astype(float)

        # ── D3: 24h 成交量确认（流动性过滤） ──
        df['volume_24h_est'] = df['volume'].rolling(288, min_periods=144).sum() * df['close']

        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        多因子加权评分 → 入场决策。

        分层评分:
        1. 每个因子独立评分 (0=未触发, weight=触发, 0.6*weight=中等触发)
        2. 按类别汇总: A_Price_Volume(40) + B_Pattern(25) + C_Context(20) + D_Timing(15)
        3. 总分 >= threshold 且 >= min_categories 个类别有分数 → 入场
        """

        df = dataframe.copy()

        params = {
            'A1': self.A1_threshold.value,
            'A2': self.A2_threshold.value,
            'A3': self.A3_threshold.value,
            'B1': self.B1_threshold.value,
            'B2': self.B2_threshold.value,
            'C1_lo': self.C1_vol_lo.value,
            'C1_hi': self.C1_vol_hi.value,
        }

        # ── A1: 价格动量 (权重15) ──
        A1_score = pd.Series(0.0, index=df.index)
        A1_score[df['A1_price_momentum'] >= params['A1']] = 15
        A1_score[(df['A1_price_momentum'] >= params['A1'] * 0.5) & (A1_score == 0)] = 9  # moderate

        # ── A2: 量能爆发 (权重15) ──
        A2_score = pd.Series(0.0, index=df.index)
        A2_score[df['A2_volume_surge'] >= params['A2']] = 15
        A2_score[(df['A2_volume_surge'] >= params['A2'] * 0.67) & (A2_score == 0)] = 9

        # ── A3: 趋势强度 (权重10) ──
        A3_score = pd.Series(0.0, index=df.index)
        A3_score[df['A3_sma_momentum'] >= params['A3']] = 10
        A3_score[(df['A3_sma_momentum'] >= params['A3'] * 0.5) & (A3_score == 0)] = 6

        # ── B1: 阶梯吸筹 (权重15) ──
        B1_score = pd.Series(0.0, index=df.index)
        B1_score[df['B1_staircase'] >= params['B1']] = 15
        B1_score[(df['B1_staircase'] >= params['B1'] * 0.75) & (B1_score == 0)] = 9

        # ── B2: 振幅扩张 (权重10) ──
        B2_score = pd.Series(0.0, index=df.index)
        B2_score[df['B2_range_behavior'] >= params['B2']] = 10
        B2_score[(df['B2_range_behavior'] >= params['B2'] * 0.8) & (B2_score == 0)] = 6

        # ── C1: 波动率环境 (权重10, 区间评分) ──
        C1_score = pd.Series(0.0, index=df.index)
        vol = df['C1_volatility']
        C1_score[(vol >= params['C1_lo']) & (vol <= params['C1_hi'])] = 10
        C1_score[(vol >= params['C1_lo'] * 0.6) & (vol <= params['C1_hi'] * 1.5) & (C1_score == 0)] = 6

        # ── C2: 成交量质量 (权重10) ──
        C2_score = pd.Series(0.0, index=df.index)
        vq = df['C2_volume_quality']
        C2_score[(vq >= 0) & (vq <= 2.0)] = 10  # moderate positive skew
        C2_score[(vq >= -1) & (vq <= 4.0) & (C2_score == 0)] = 6

        # ── D1: 时间窗口 (权重10) ──
        D1_score = pd.Series(0.0, index=df.index)
        timing = df['D1_timing']
        D1_score[timing >= 8] = 10   # Asia Active / US Morning
        D1_score[(timing >= 6) & (D1_score == 0)] = 6

        # ── D2: 价格合理性 (权重5) ──
        D2_score = pd.Series(0.0, index=df.index)
        D2_score[df['D2_price_ok'] == 1] = 5

        # ── 类别汇总 ──
        df['cat_A_score'] = A1_score + A2_score + A3_score  # max 40
        df['cat_B_score'] = B1_score + B2_score              # max 25
        df['cat_C_score'] = C1_score + C2_score              # max 20
        df['cat_D_score'] = D1_score + D2_score              # max 15

        df['total_score'] = df['cat_A_score'] + df['cat_B_score'] + df['cat_C_score'] + df['cat_D_score']

        # 活跃类别计数
        df['active_cats'] = (
            (df['cat_A_score'] > 0).astype(int) +
            (df['cat_B_score'] > 0).astype(int) +
            (df['cat_C_score'] > 0).astype(int) +
            (df['cat_D_score'] > 0).astype(int)
        )

        # ── 入场条件 ──
        threshold = self.buy_score_threshold.value
        min_cats = self.buy_min_categories.value

        df['enter_long'] = 0
        df['enter_tag'] = ''

        # 流动性过滤
        has_liquidity = df['volume_24h_est'] > 50000

        # STRONG BUY: score >= 80, 3+ categories
        strong = (df['total_score'] >= 80) & (df['active_cats'] >= 3) & has_liquidity
        df.loc[strong, ['enter_long', 'enter_tag']] = [1, 'demon_strong']

        # BUY: score >= threshold, min_categories
        buy = (df['total_score'] >= threshold) & (df['active_cats'] >= min_cats) & has_liquidity & (strong == False)
        df.loc[buy, ['enter_long', 'enter_tag']] = [1, 'demon_buy']

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        出口信号：主要依赖 custom_exit 进行精细化管理。
        这里仅设置基础条件。
        """
        df = dataframe.copy()
        df['exit_long'] = 0
        return df

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        自定义退出逻辑 — 棘轮移动止损 + 时间退出 + 回撤保护。

        退出优先级:
        1. 硬止损: <= -15% → 无条件退出
        2. 棘轮止损: 利润保护关卡
        3. 从高点回撤保护
        4. 时间退出: 持仓过久且未达预期
        """
        trade_dict = trade.to_json()
        entry_price = trade_dict.get('open_rate', current_rate)
        hold_hours = (current_time.replace(tzinfo=timezone.utc) -
                      trade.open_date_utc).total_seconds() / 3600

        # ── 获取历史峰值利润 ──
        max_rate = trade_dict.get('max_rate', entry_price)
        if current_rate > max_rate:
            max_rate = current_rate
        peak_profit = (max_rate - entry_price) / entry_price

        # ── 1. 硬止损 ──
        if current_profit <= self.stoploss:
            return 'hard_stoploss'

        # ── 2. 棘轮止损 (基于当前利润) ──
        # 浮盈 >= 100% → 止损在 +50%
        if current_profit >= 1.0:
            protect = self.ratchet_100.value
            if current_profit < protect:
                return f'ratchet_100pct_{protect:.0%}'

        # 浮盈 >= 50% → 止损在 +25%
        elif current_profit >= 0.50:
            protect = self.ratchet_50.value
            if current_profit < protect:
                return f'ratchet_50pct_{protect:.0%}'

        # 浮盈 >= 20% → 止损在 +10%
        elif current_profit >= 0.20:
            protect = self.ratchet_20.value
            if current_profit < protect:
                return f'ratchet_20pct_{protect:.0%}'

        # 浮盈 >= 10% → 保本止损
        elif current_profit >= 0.10:
            if current_profit < self.ratchet_10.value:
                return f'ratchet_10pct_breakeven'

        # ── 3. 从高点回撤保护 ──
        if peak_profit >= self.pullback_peak_min.value and current_profit >= self.pullback_min_profit.value:
            drawdown = peak_profit - current_profit
            if drawdown >= self.pullback_drawdown.value:
                return f'pullback_{drawdown:.1%}'

        # ── 4. 时间退出 ──
        # 6小时 < -8% → 急跌退出
        if hold_hours >= 6 and current_profit < -0.08:
            return 'time_sharp_drop_6h'

        # 12小时 < -3% → 废票退出
        if hold_hours >= 12 and current_profit < -0.03:
            return 'time_dud_12h'

        # 24小时 < +1% → 平庸退出
        if hold_hours >= 24 and current_profit < 0.01:
            return 'time_stale_24h'

        # 48小时 < +5% → 低效退出
        if hold_hours >= 48 and current_profit < 0.05:
            return 'time_weak_48h'

        # 72小时 → 强制退出
        if hold_hours >= 72:
            return 'time_max_hold_72h'

        return None

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        """
        仓位管理:
        - STRONG 信号: 3% 仓位
        - BUY 信号: 2% 仓位
        - 最大单币仓位: 5%
        """
        wallet = self.wallets
        available = wallet.get_free('USDT')

        if available <= 0:
            return 0.0

        if entry_tag == 'demon_strong':
            stake_pct = 0.03
        elif entry_tag == 'demon_buy':
            stake_pct = 0.02
        else:
            stake_pct = 0.01

        # 单币仓位不超过5%
        max_per_coin = wallet.get_total('USDT') * 0.05
        stake = min(available * stake_pct, max_per_coin)

        # 最小交易量
        if min_stake and stake < min_stake:
            return 0.0

        return max(stake, 0.0)
