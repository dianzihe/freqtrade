# -*- coding: utf-8 -*-
"""
微结构潜在状态检测策略 (LOBLatentRegimeStrategy) — Tick 数据版
================================================================

基于: arXiv:2604.20949 "Early Detection of Latent Microstructure Regimes
in Limit Order Books" (Hiremath & Hiremath, 2026.04)

数据要求
--------
本策略依赖从 Binance aggTrades tick 数据预计算得到的 5 分钟 OHLCV +
微结构扩展列。使用前需先运行预处理脚本：

  python user_data/scripts/build_tick_indicators.py

输出文件: user_data/data/binance/BTC_USDT-5m.feather

扩展列（由预处理脚本产出）:
  trade_count        — 每 5 分钟成交笔数
  buy_volume_ratio   — 买方成交量占比（B/S 标记 → 通道 4，直接订单流）
  price_std          — 成交价格的标准差（→ 通道 3，价差代理）
  trade_intensity    — 每秒成交笔数（→ 通道 2，深度代理）
  large_trade_ratio  — 大单占比（→ 通道 2 辅助，鲸鱼活动）
  vwap               — 成交量加权均价
  vwap_deviation     — 收盘价相对 VWAP 的偏离（方向性信号）

论文四通道映射（从"OHLCV 粗糙代理"升级为"tick 级精确指标"）:
  原论文               OHLCV 代理版          Tick 数据版（本版）
  ──────────          ────────────          ─────────────────
  HMM 状态熵  →  波动率结构熵         →  波动率结构熵（不变，需多 bar）
  深度侵蚀    →  Amihud ILLIQ 趋势    →  trade_intensity 下降 + large_trade_ratio 下降
  价差漂移    →  high-low/close Z     →  price_std 的 Z-score（真实价内离散度）
  订单流动量  →  K 线位置+卖压占比    →  buy_volume_ratio（直接 B/S 标记，无代理误差）

核心机制（继承论文，从 OHLCV 版沿用）:
  - MAX 聚合：四个通道取最大值（论文发现单强通道足够，无需等所有确认）
  - 上升沿条件：要求信号处于上升趋势（捕捉恶化起始，非恶化峰值）
  - 自适应阈值：滚动分位数（适应不同波动率时段）
"""

from pandas import DataFrame
import numpy as np
import pandas as pd

from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter)
import talib.abstract as ta

# ═══════════════════════════════════════════════════════════════
#  Monkey-patch: 保留 feather 文件中的扩展列
# ═══════════════════════════════════════════════════════════════
# freqtrade 在两个地方丢弃扩展列：
#   1. _ohlcv_load(): pairdata.columns = self._columns
#      → 当 feather 有 13 列但 self._columns 只有 6 个名时，ValueError
#   2. clean_ohlcv_dataframe(): groupby.agg() 只聚合 5 个标准列
#      → 扩展列被静默丢弃
#
# 修复策略（两层补丁）：
#   Patch A (_ohlcv_load): 动态扩展 self._columns 以匹配 feather 列数
#   Patch B (ohlcv_load):  在 clean_ohlcv_dataframe 剥离后，按 date merge 回来
import pandas as _pd
from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler

# --- Patch A: 修复 _ohlcv_load 的列名不匹配 ---
_ORIG_OHLCV_LOAD = FeatherDataHandler._ohlcv_load


def _patched_ohlcv_load(self, pair, timeframe, timerange, candle_type):
    filename = self._pair_data_filename(
        self._datadir, pair, timeframe, candle_type=candle_type
    )
    if not filename.exists():
        filename = self._pair_data_filename(
            self._datadir, pair, timeframe, candle_type=candle_type,
            no_timeframe_modify=True,
        )
        if not filename.exists():
            return _pd.DataFrame(columns=self._columns)
    try:
        pairdata = _pd.read_feather(filename)
        feather_cols = list(pairdata.columns)
        # 动态扩展列名：前 6 列用标准名，后面保持 feather 原名
        if len(feather_cols) > len(self._columns):
            extended = list(self._columns)
            for i in range(len(self._columns), len(feather_cols)):
                extended.append(feather_cols[i])
            pairdata.columns = extended
        else:
            pairdata.columns = self._columns[:len(feather_cols)]
        # 类型转换
        for col in ["open", "high", "low", "close", "volume"]:
            if col in pairdata.columns:
                pairdata[col] = pairdata[col].astype("float")
        if "date" in pairdata.columns:
            pairdata["date"] = _pd.to_datetime(
                pairdata["date"], utc=True
            ).dt.as_unit("ms")
        return pairdata
    except Exception:
        return _pd.DataFrame(columns=self._columns)


FeatherDataHandler._ohlcv_load = _patched_ohlcv_load

# --- Patch B: 修复 clean_ohlcv_dataframe 的列剥离 ---
_ORIG_OHLCV_LOAD_PUBLIC = FeatherDataHandler.ohlcv_load


def _patched_ohlcv_load_public(self, pair, timeframe, candle_type, *,
                                timerange=None, fill_missing=True,
                                drop_incomplete=False, startup_candles=0,
                                warn_no_data=True):
    """
    在 freqtrade 原版加载流程后，把被 clean_ohlcv_dataframe 剥离的扩展列 merge 回来。
    """
    result = _ORIG_OHLCV_LOAD_PUBLIC(
        self, pair, timeframe, candle_type,
        timerange=timerange, fill_missing=fill_missing,
        drop_incomplete=drop_incomplete, startup_candles=startup_candles,
        warn_no_data=warn_no_data,
    )

    if len(result) == 0 or len(result.columns) > 6:
        return result

    # 从 feather 文件补回扩展列
    filename = self._pair_data_filename(
        self._datadir, pair, timeframe, candle_type=candle_type
    )
    if not filename.exists():
        filename = self._pair_data_filename(
            self._datadir, pair, timeframe, candle_type=candle_type,
            no_timeframe_modify=True,
        )
    if not filename.exists():
        return result

    try:
        full = _pd.read_feather(filename)
    except Exception:
        return result

    extra_cols = [c for c in full.columns if c not in result.columns]
    if not extra_cols:
        return result

    extra_df = full[["date"] + extra_cols].copy()
    extra_df["date"] = _pd.to_datetime(extra_df["date"], utc=True).dt.as_unit("ms")
    result = result.merge(extra_df, on="date", how="left")

    for c in extra_cols:
        if c in result.columns:
            result[c] = result[c].fillna(0.0)

    return result


FeatherDataHandler.ohlcv_load = _patched_ohlcv_load_public
# ═══════════════════════════════════════════════════════════════


class LOBLatentRegimeStrategy(IStrategy):
    """
    基于 tick 级微结构指标的潜在状态检测策略。

    仅在检测到潜在积累信号时做空——论文发现"压力"在加密货币中
    绝大多数表现为急跌（多单连环爆仓的链式反应）。
    """

    # ====== 基础配置 ======
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    startup_candle_count = 200
    process_only_new_candles = True

    # ====== ROI / 止损 ======
    # 论文中积累→压力约 20-30 个时间步（100-150 分钟）
    minimal_roi = {
        "0":   0.08,
        "15":  0.05,
        "30":  0.03,
        "60":  0.01,
        "120": 0.0,
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    # ====== 出场信号 ======
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ====== 订单配置 ======
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ====== 超参数（hyperopt 可优化） ======
    lookback_period = IntParameter(12, 48, default=24, space="buy")
    volatility_window = IntParameter(10, 40, default=20, space="buy")
    threshold_percentile = IntParameter(75, 95, default=88, space="buy")
    confirmation_bars = IntParameter(1, 4, default=2, space="buy")
    min_signal_strength = DecimalParameter(0.25, 0.65, default=0.40, space="buy")
    trend_confirm_bars = IntParameter(3, 10, default=5, space="buy")
    rsi_lower_bound = IntParameter(20, 40, default=28, space="buy")

    # ====== 绘图配置 ======
    plot_config = {
        "main_plot": {},
        "subplots": {
            "通道信号": {
                "ch1_vol_entropy":        {"color": "#2196F3"},
                "ch2_depth_erosion":      {"color": "#FF5722"},
                "ch3_spread_drift":       {"color": "#4CAF50"},
                "ch4_order_flow":         {"color": "#9C27B0"},
            },
            "聚合信号": {
                "composite_smooth":     {"color": "#E91E63"},
                "adaptive_threshold":   {"color": "#607D8B"},
                "signal_trigger":       {"color": "#FF0000", "type": "bar"},
            },
        },
    }

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    # ================================================================
    #  指标计算
    # ================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        四通道信号计算（使用 tick 级指标） + MAX 聚合触发。

        前置条件：dataframe 中必须包含预处理脚本产出的扩展列：
          trade_count, buy_volume_ratio, price_std, trade_intensity,
          large_trade_ratio, vwap, vwap_deviation

        如果缺少这些列（使用普通 OHLCV 数据），此处会输出警告但不崩溃。
        """
        # --- 通道 1：波动率结构熵 ---
        # 论文原始：3 状态 HMM 的滤波概率熵。
        # 为什么不改用 tick 级替代：HMM 需要拟合整个时间序列的状态转移概率，
        #   单窗口内的信息不足。波动率结构熵虽然是代理，但依然有用——
        #   它捕捉"波动率在不同时间尺度上开始分歧"这一过渡期特征。
        dataframe = self._ch1_volatility_entropy(dataframe)

        # --- 通道 2：深度侵蚀（tick 升级版）---
        # 论文原始：LOB 深度相对滚动基线的单调下降。
        # 论文发现这是最强通道（55% 首发触发）。
        #
        # Tick 升级：使用两个直接可观测的活跃度指标替代 Amihud 代理。
        #   trade_intensity：成交笔数/秒。下降 = 市场参与者减少 = 深度侵蚀。
        #   large_trade_ratio：大单量占比。下降 = 鲸鱼/机构撤退 = 深度侵蚀先兆。
        #   两个指标同时下降 → 高置信度的深度恶化信号。
        dataframe = self._ch2_depth_erosion_tick(dataframe)

        # --- 通道 3：价差漂移（tick 升级版）---
        # 论文原始：bid-ask spread 的标准化趋势。
        #
        # Tick 升级：使用 price_std（5 分钟内所有成交价格的标准差）。
        #   这比 high-low 精确得多——
        #   high-low 被极端价格主导（一笔 outlier 就拉宽），
        #   而 price_std 考虑所有成交价格，更能反映真实的价内离散度。
        #   价格离散度扩大 = spread 扩大 = 做市商撤单。
        dataframe = self._ch3_spread_drift_tick(dataframe)

        # --- 通道 4：订单流（tick 升级版——最大提升）---
        # 论文原始：方向性订单不平衡。
        #
        # Tick 升级：buy_volume_ratio 直接从 B/S 标记计算，
        #   0.50 = 买卖平衡，< 0.45 = 卖压主导。
        #   这是从"粗糙代理"到"精确指标"的最大跃升——
        #   B/S 标记来自交易所原始 taker 方向，误差为零。
        dataframe = self._ch4_order_flow_tick(dataframe)

        # --- MAX 聚合 + 上升沿 + 自适应阈值（逻辑不变） ---
        dataframe = self._compute_composite_trigger(dataframe)

        return dataframe

    # ================================================================
    #  通道 1：波动率结构熵（与 OHLCV 版相同）
    # ================================================================

    def _ch1_volatility_entropy(self, dataframe: DataFrame) -> DataFrame:
        """
        通道 1：波动率结构熵。

        为什么不用 tick 级替代：
        - HMM 需要整段时间序列来估计转移概率，单窗口信息不足。
        - 但波动率结构熵仍然捕捉了关键信息：当短期波动和长期波动
          产生分歧，说明市场内部结构正在变化（状态过渡特征）。

        三尺度分歧 → 熵升高 → 可能处于积累阶段。
        """
        w = self.volatility_window.value
        returns = dataframe["close"].pct_change()

        short_w = max(4, w // 4)
        mid_w   = max(8, w // 2)

        vol_s = returns.rolling(short_w).std()
        vol_m = returns.rolling(mid_w).std()
        vol_l = returns.rolling(w).std()

        vol_sum = vol_s + vol_m + vol_l
        p_s = vol_s / vol_sum.replace(0, np.nan)
        p_m = vol_m / vol_sum.replace(0, np.nan)
        p_l = vol_l / vol_sum.replace(0, np.nan)

        eps = 1e-12
        entropy = -(
            p_s * np.log(p_s.clip(lower=eps)) +
            p_m * np.log(p_m.clip(lower=eps)) +
            p_l * np.log(p_l.clip(lower=eps))
        )

        dataframe["ch1_vol_entropy"] = (entropy / np.log(3)).fillna(0.0)
        return dataframe

    # ================================================================
    #  通道 2：深度侵蚀 — tick 真正升级
    # ================================================================

    def _ch2_depth_erosion_tick(self, dataframe: DataFrame) -> DataFrame:
        """
        通道 2：深度侵蚀（tick 直接度量版）。

        替代 OHLCV 版的 Amihud ILLIQ 代理。
        直接使用两个 tick 级活跃度指标：

        1. trade_intensity（成交笔数/秒）
           为什么有效：深度充足时做市商积极报价，taker 频繁成交。
           深度下降时做市商撤单 → 可成交的机会减少 → 成交频率下降。
           这是"深度侵蚀"的**直接外在表现**。

        2. large_trade_ratio（大单量占比）
           为什么有效：鲸鱼/机构通常最先感知到风险，最早撤退。
           大单占比下降 = 聪明的钱在离场 = 深度即将急剧恶化。

        合成逻辑：
        - 分别对两个指标计算 Z-score 的负向（下降 = 信号）
        - 取两者的加权平均（trade_intensity 权重 0.6，因为更直接）
        - 归一化到 [0, 1]

        注意：trade_intensity 和 large_trade_ratio 天然高度相关
        （机构撤退 → 交易量减少 → 成交频率下降），但这是好事——
        两个独立途径验证同一结论，降低误报率。
        """
        window = self.lookback_period.value

        # ---- 信号 1：成交强度下降 ----
        ti = dataframe.get("trade_intensity", pd.Series(0, index=dataframe.index))
        ti_mean = ti.rolling(window=window * 2, min_periods=window).mean()
        ti_std  = ti.rolling(window=window * 2, min_periods=window).std()
        # Z-score 取反：下降 = 正值（信号）
        ti_z = ((ti_mean - ti) / ti_std.replace(0, 1e-10)).fillna(0.0)
        ti_signal = ti_z.clip(lower=0)

        # ---- 信号 2：大单占比下降 ----
        lr = dataframe.get("large_trade_ratio", pd.Series(0, index=dataframe.index))
        lr_mean = lr.rolling(window=window * 2, min_periods=window).mean()
        lr_std  = lr.rolling(window=window * 2, min_periods=window).std()
        lr_z = ((lr_mean - lr) / lr_std.replace(0, 1e-10)).fillna(0.0)
        lr_signal = lr_z.clip(lower=0)

        # ---- 加权合成 ----
        # trade_intensity 权重大（更直接反映市场参与度），large_trade_ratio 辅助验证
        raw = 0.6 * ti_signal + 0.4 * lr_signal

        # ---- 上升趋势确认 ----
        raw_smooth = raw.rolling(window=max(3, window // 3), min_periods=1).mean()
        half_w = max(2, window // 2)
        trend = raw_smooth - raw_smooth.shift(half_w)
        signal = raw_smooth * (trend > 0).astype(float)

        # 滚动百分位归一化
        rank_pct = signal.rolling(window * 4, min_periods=window).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )
        dataframe["ch2_depth_erosion"] = rank_pct.fillna(0.0).clip(0, 1)
        return dataframe

    # ================================================================
    #  通道 3：价差漂移 — tick 真正升级
    # ================================================================

    def _ch3_spread_drift_tick(self, dataframe: DataFrame) -> DataFrame:
        """
        通道 3：价差漂移（tick 直接度量版）。

        替代 OHLCV 版的 high-low/close Z-score 代理。
        直接使用 price_std（5 分钟内所有成交价格的标准差）。

        为什么 price_std >> high-low：
        - high-low 被单笔异常成交主导。比如一笔 0.001 BTC 的极端价格
          就能把 high-low 拉到不合理范围。
        - price_std 考虑所有成交价格的分布，异常值的影响被稀释。
        - 在流动性正常时，大量成交聚集在窄区间 → price_std 小。
        - 流动性恶化时，成交分散到更宽区间 → price_std 大。

        上升趋势 + Z-score 正值 = spread 在扩大。
        """
        window = self.lookback_period.value

        ps = dataframe.get("price_std", pd.Series(0, index=dataframe.index))

        # Z-score（相对历史的标准化）
        ps_mean = ps.rolling(window=window * 2, min_periods=window).mean()
        ps_std  = ps.rolling(window=window * 2, min_periods=window).std()
        ps_z = ((ps - ps_mean) / ps_std.replace(0, 1e-10)).fillna(0.0)

        # 正值 + 上升趋势
        ps_pos = ps_z.clip(lower=0)
        ps_smooth = ps_pos.rolling(window=max(3, window // 3), min_periods=1).mean()
        half_w = max(2, window // 2)
        ps_trend = ps_smooth - ps_smooth.shift(half_w)

        raw = ps_smooth * (ps_trend > 0).astype(float)

        rank_pct = raw.rolling(window * 4, min_periods=window).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )
        dataframe["ch3_spread_drift"] = rank_pct.fillna(0.0).clip(0, 1)
        return dataframe

    # ================================================================
    #  通道 4：订单流 — tick 最大升级（B/S 标记，零代理误差）
    # ================================================================

    def _ch4_order_flow_tick(self, dataframe: DataFrame) -> DataFrame:
        """
        通道 4：订单流（tick 直接度量版，最大升级）。

        替代 OHLCV 版的"K 线收盘位置 × 卖压占比"代理。
        直接使用 buy_volume_ratio（买方成交量占比）。

        数据来源：Binance aggTrades 的 B/S 标记。
        B = taker 方向为买入（主动吃 ask），S = taker 方向为卖出（主动吃 bid）。
        这是交易所原始数据，无误差、无代理、无估计。

        逻辑：
        - buy_volume_ratio 持续低于 0.50 → 卖压在积累。
        - 下降趋势 → 卖压还在加强（而非已经释放完毕）。
        - buy_volume_ratio < 0.45 且加速下降 → 最高信号。

        这就是论文"订单流动量"通道的**原汁原味实现**——
        aggTrades 的 B/S 标记等价于按 taker 方向归类的订单流。
        """
        window = self.lookback_period.value

        bvr = dataframe.get("buy_volume_ratio", pd.Series(0.5, index=dataframe.index))

        # 反向指标：买盘比例越低 = 卖压越重
        sell_pressure = 1.0 - bvr  # [0, 1]，越高越危险

        # 与自身历史对比（不是绝对 0.5，而是相对近期均值）
        sp_mean = sell_pressure.rolling(window=window * 2, min_periods=window).mean()
        sp_std  = sell_pressure.rolling(window=window * 2, min_periods=window).std()
        sp_z = ((sell_pressure - sp_mean) / sp_std.replace(0, 1e-10)).fillna(0.0)

        # 正值（卖压高于历史）+ 上升趋势（卖压还在加强）
        sp_pos = sp_z.clip(lower=0)
        sp_smooth = sp_pos.rolling(window=max(3, window // 3), min_periods=1).mean()
        half_w = max(2, window // 2)
        sp_trend = sp_smooth - sp_smooth.shift(half_w)

        raw = sp_smooth * (sp_trend > 0).astype(float)

        rank_pct = raw.rolling(window * 4, min_periods=window).apply(
            lambda x: (x.iloc[-1] > x).mean(), raw=False
        )
        dataframe["ch4_order_flow"] = rank_pct.fillna(0.0).clip(0, 1)
        return dataframe

    # ================================================================
    #  MAX 聚合 + 上升沿检测 + 自适应阈值
    # ================================================================

    def _compute_composite_trigger(self, dataframe: DataFrame) -> DataFrame:
        """
        MAX 聚合 + 上升沿 + 自适应阈值（与 OHLCV 版逻辑一致）。

        变更：通道列名从 ch{1,2,3,4}_xxx 改为新的 tick 版通道名。
        """
        window = self.lookback_period.value
        percentile = self.threshold_percentile.value

        channels = [
            "ch1_vol_entropy",
            "ch2_depth_erosion",
            "ch3_spread_drift",
            "ch4_order_flow",
        ]

        # MAX 聚合（论文发现单强通道足矣，无需等全部确认）
        dataframe["composite_raw"] = dataframe[channels].max(axis=1)

        # 3 根平滑
        dataframe["composite_smooth"] = (
            dataframe["composite_raw"].rolling(window=3, min_periods=1).mean()
        )

        # 上升沿检测（捕捉恶化起始，非峰值）
        n_confirm = self.confirmation_bars.value
        rising = dataframe["composite_smooth"] > dataframe["composite_smooth"].shift(1)
        rising_streak = rising.copy()
        for lag in range(2, n_confirm + 1):
            rising_streak = rising_streak & (
                dataframe["composite_smooth"] > dataframe["composite_smooth"].shift(lag)
            )

        # 自适应阈值（滚动分位数）
        long_w = max(window * 4, 48)
        adaptive_threshold = (
            dataframe["composite_smooth"]
            .rolling(long_w, min_periods=window * 2)
            .quantile(percentile / 100.0)
        )

        min_floor = self.min_signal_strength.value
        effective_threshold = adaptive_threshold.clip(lower=min_floor)

        trigger = (
            (dataframe["composite_smooth"] > effective_threshold)
            & rising_streak
            & (dataframe["composite_smooth"] > min_floor)
        )

        dataframe["adaptive_threshold"] = effective_threshold
        dataframe["signal_trigger"] = trigger.astype(int)
        return dataframe

    # ================================================================
    #  入场逻辑
    # ================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场做空条件（与 OHLCV 版一致）：
        1. signal_trigger 触发
        2. 短期趋势向下
        3. 成交量放大
        4. RSI 不过低
        """
        dataframe["enter_short"] = 0
        dataframe["enter_long"] = 0
        dataframe.loc[:, "enter_tag"] = ""

        n_trend = self.trend_confirm_bars.value
        rsi_low = self.rsi_lower_bound.value

        trigger = dataframe["signal_trigger"] == 1
        short_downtrend = dataframe["close"].pct_change(n_trend) < 0
        volume_expanding = (
            dataframe["volume"] > dataframe["volume"].rolling(20, min_periods=1).mean()
        )
        rsi = ta.RSI(dataframe, timeperiod=14)
        not_oversold = rsi > rsi_low

        entry_condition = (
            trigger & short_downtrend & volume_expanding & not_oversold
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[entry_condition, "enter_short"] = 1
        dataframe.loc[entry_condition, "enter_tag"] = "latent_build_up"
        return dataframe

    # ================================================================
    #  出场逻辑
    # ================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场条件（与 OHLCV 版一致）：
        1. 压力释放完成（连续阴线后阳线）
        2. 信号消退（假信号）
        3. 趋势反转
        """
        dataframe["exit_short"] = 0
        dataframe["exit_long"] = 0
        dataframe.loc[:, "exit_tag"] = ""

        consecutive_down_3 = (
            (dataframe["close"] < dataframe["open"]).rolling(3, min_periods=3).sum() == 3
        )
        green_candle = dataframe["close"] > dataframe["open"]
        pressure_released = consecutive_down_3.shift(1) & green_candle

        signal_faded = (
            (dataframe["composite_smooth"] < dataframe["adaptive_threshold"] * 0.5)
            & (dataframe["signal_trigger"] == 0)
        )

        sma_short = ta.SMA(dataframe, timeperiod=15)
        trend_reversed = dataframe["close"] > sma_short

        exit_condition = (
            (pressure_released | signal_faded | trend_reversed)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[exit_condition & pressure_released, "exit_short"] = 1
        dataframe.loc[exit_condition & pressure_released, "exit_tag"] = "pressure_released"

        dataframe.loc[exit_condition & signal_faded & ~pressure_released, "exit_short"] = 1
        dataframe.loc[exit_condition & signal_faded & ~pressure_released, "exit_tag"] = "signal_faded"

        dataframe.loc[
            exit_condition & trend_reversed & ~pressure_released & ~signal_faded,
            "exit_short"
        ] = 1
        dataframe.loc[
            exit_condition & trend_reversed & ~pressure_released & ~signal_faded,
            "exit_tag"
        ] = "trend_reversed"

        return dataframe

    # ================================================================
    #  TODO: 风控预留接口
    # ================================================================
    # [ ] 仓位管理：基于信号强度的动态仓位（custom_stake_amount）
    # [ ] 实盘安全开关：前 N 笔小仓验证
    # [ ] 数据质量检查：tick 指标列缺失时自动降级或报警
    # [ ] 时段过滤：过滤亚盘低流动性时段
