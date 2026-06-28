# -*- coding: utf-8 -*-
"""
微结构潜在状态检测 — 信号模块（纯指标，非交易策略）
====================================================

基于: arXiv:2604.20949 "Early Detection of Latent Microstructure Regimes
in Limit Order Books" (Hiremath & Hiremath, 2026.04)

定位
----
这是一个 **预警信号生成器**，不是一个完整的交易策略。
它的职责：检测限价订单簿微结构从"稳定"向"压力"过渡的潜在积累阶段。
它的不负责：方向判断、仓位管理、止损止盈、何时进出场。

类比：就像 MACD 告诉你"趋势在变"，这个模块告诉你"微观结构在恶化"。
具体恶化之后怎么做，由调用它的策略决定。

用法
----
在任何 freqtrade 策略的 populate_indicators() 中调用：

    from user_data.strategies.微结构_信号模块 import add_lob_regime_signals

    def populate_indicators(self, dataframe, metadata):
        dataframe = add_lob_regime_signals(
            dataframe,
            lookback_period=24,
            threshold_percentile=88,
            confirmation_bars=2,
        )
        return dataframe

然后可以访问以下列：
    signal_trigger  — 1=检测到潜在积累（论文的"预警触发"）
    composite_smooth — 综合信号强度 [0,1]
    ch1~ch4          — 四个分解通道的信号强度

数据要求
--------
必须使用预处理的 tick 数据（含扩展列），参见 user_data/scripts/build_tick_indicators.py
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  Monkey-patch: 保留 feather 文件中的扩展列
# ═══════════════════════════════════════════════════════════════
# 与微结构_潜在状态检测.py 共享同一套 patch 逻辑。
# 导入本模块即生效，无需重复引入。
try:
    from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler

    if not hasattr(FeatherDataHandler, "_lob_patched"):
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
                    return pd.DataFrame(columns=self._columns)
            try:
                pairdata = pd.read_feather(filename)
                feather_cols = list(pairdata.columns)
                if len(feather_cols) > len(self._columns):
                    extended = list(self._columns)
                    for i in range(len(self._columns), len(feather_cols)):
                        extended.append(feather_cols[i])
                    pairdata.columns = extended
                else:
                    pairdata.columns = self._columns[:len(feather_cols)]
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in pairdata.columns:
                        pairdata[col] = pairdata[col].astype("float")
                if "date" in pairdata.columns:
                    pairdata["date"] = pd.to_datetime(pairdata["date"], utc=True).dt.as_unit("ms")
                return pairdata
            except Exception:
                return pd.DataFrame(columns=self._columns)

        FeatherDataHandler._ohlcv_load = _patched_ohlcv_load

        _ORIG_OHLCV_LOAD_PUBLIC = FeatherDataHandler.ohlcv_load

        def _patched_ohlcv_load_public(
            self, pair, timeframe, candle_type, *,
            timerange=None, fill_missing=True, drop_incomplete=False,
            startup_candles=0, warn_no_data=True,
        ):
            result = _ORIG_OHLCV_LOAD_PUBLIC(
                self, pair, timeframe, candle_type,
                timerange=timerange, fill_missing=fill_missing,
                drop_incomplete=drop_incomplete, startup_candles=startup_candles,
                warn_no_data=warn_no_data,
            )
            if len(result) == 0 or len(result.columns) > 6:
                return result
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
                full = pd.read_feather(filename)
            except Exception:
                return result
            extra_cols = [c for c in full.columns if c not in result.columns]
            if not extra_cols:
                return result
            extra_df = full[["date"] + extra_cols].copy()
            extra_df["date"] = pd.to_datetime(extra_df["date"], utc=True).dt.as_unit("ms")
            result = result.merge(extra_df, on="date", how="left")
            for c in extra_cols:
                if c in result.columns:
                    result[c] = result[c].fillna(0.0)
            return result

        FeatherDataHandler.ohlcv_load = _patched_ohlcv_load_public
        FeatherDataHandler._lob_patched = True  # 标记，避免重复 patch
except ImportError:
    pass  # 非 freqtrade 环境（如测试脚本），跳过 patch
# ═══════════════════════════════════════════════════════════════


# ---- 公开 API ----

def add_lob_regime_signals(
    dataframe: pd.DataFrame,
    lookback_period: int = 24,
    volatility_window: int = 20,
    threshold_percentile: int = 88,
    confirmation_bars: int = 2,
    min_signal_strength: float = 0.40,
) -> pd.DataFrame:
    """
    为 DataFrame 添加 LOB 微结构状态检测信号。

    参数
    ----
    lookback_period : int
        信号通道的滚动窗口大小（K 线数）。越大越平滑但越滞后。
        论文中与积累阶段期望持续时间相关，建议 12-48。
    volatility_window : int
        波动率结构熵的时间尺度基数。
    threshold_percentile : int
        自适应阈值的分位数（75-98）。越高 = 越保守 = 漏检多但虚警少。
    confirmation_bars : int
        要求连续上升的 K 线数（1-4）。越高 = 确认越严格。
    min_signal_strength : float
        触发的最小信号强度（0.0-1.0）。低于此值不触发，过滤纯噪声。

    返回
    ----
    DataFrame（原地修改 + 返回），新增以下列：

    ====================  ==========================================
    列名                   含义
    ====================  ==========================================
    ch1_vol_entropy       波动率结构熵 [0,1]（高 = 状态在过渡）
    ch2_depth_erosion     深度侵蚀信号 [0,1]（高 = 流动性在恶化）
    ch3_spread_drift      价差漂移信号 [0,1]（高 = 价差在扩大）
    ch4_order_flow        订单流信号 [0,1]（高 = 卖压在积累）
    composite_smooth      综合信号强度 [0,1]（四个通道的 MAX）
    adaptive_threshold    自适应阈值 [0,1]（动态分位数）
    signal_trigger        预警触发 [0/1]（综合信号 > 阈值且在上��）
    ====================  ==========================================

    使用建议
    --------
    - signal_trigger 作为**辅助确认**，不是主信号
    - 结合趋势方向判断：触发 + 下跌 = 做空；触发 + 上涨 = 等或做多
    - 触发后的"有效期"约 10-30 根 K 线（论文发现）
    - 在低流动性时段（亚盘凌晨）信号噪声大，建议加时段过滤
    """
    df = dataframe.copy()

    df = _ch1_volatility_entropy(df, volatility_window)
    df = _ch2_depth_erosion(df, lookback_period)
    df = _ch3_spread_drift(df, lookback_period)
    df = _ch4_order_flow(df, lookback_period)
    df = _composite_trigger(
        df, lookback_period, threshold_percentile,
        confirmation_bars, min_signal_strength,
    )

    return df


# ---- 私有：四个信号通道 ----

def _ch1_volatility_entropy(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    通道 1：波动率结构熵。

    替代论文的 HMM 状态熵。三尺度波动率分歧 → 熵升高 → 市场内部结构在变化。
    """
    w = vol_window
    returns = df["close"].pct_change()

    short_w = max(4, w // 4)
    mid_w = max(8, w // 2)

    vol_s = returns.rolling(short_w).std()
    vol_m = returns.rolling(mid_w).std()
    vol_l = returns.rolling(w).std()

    vol_sum = vol_s + vol_m + vol_l
    p_s = vol_s / vol_sum.replace(0, np.nan)
    p_m = vol_m / vol_sum.replace(0, np.nan)
    p_l = vol_l / vol_sum.replace(0, np.nan)

    eps = 1e-12
    entropy = -(
        p_s * np.log(p_s.clip(lower=eps))
        + p_m * np.log(p_m.clip(lower=eps))
        + p_l * np.log(p_l.clip(lower=eps))
    )

    df["ch1_vol_entropy"] = (entropy / np.log(3)).fillna(0.0)
    return df


def _ch2_depth_erosion(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    通道 2：深度侵蚀（tick 直接度量版）。

    使用 trade_intensity（成交活跃度下降 = 做市商撤单）
    和 large_trade_ratio（大单占比下降 = 鲸鱼离开）。
    两个指标同时恶化 → 高置信度深度侵蚀信号。
    """
    ti = df.get("trade_intensity", pd.Series(0, index=df.index))
    lr = df.get("large_trade_ratio", pd.Series(0, index=df.index))

    # 成交强度下降
    ti_mean = ti.rolling(window=window * 2, min_periods=window).mean()
    ti_std = ti.rolling(window=window * 2, min_periods=window).std()
    ti_signal = ((ti_mean - ti) / ti_std.replace(0, 1e-10)).fillna(0.0).clip(lower=0)

    # 大单占比下降
    lr_mean = lr.rolling(window=window * 2, min_periods=window).mean()
    lr_std = lr.rolling(window=window * 2, min_periods=window).std()
    lr_signal = ((lr_mean - lr) / lr_std.replace(0, 1e-10)).fillna(0.0).clip(lower=0)

    raw = 0.6 * ti_signal + 0.4 * lr_signal
    raw_smooth = raw.rolling(window=max(3, window // 3), min_periods=1).mean()
    half_w = max(2, window // 2)
    trend = raw_smooth - raw_smooth.shift(half_w)
    signal = raw_smooth * (trend > 0).astype(float)

    rank_pct = _rolling_percentile(signal, window * 4, window)
    df["ch2_depth_erosion"] = rank_pct.fillna(0.0).clip(0, 1)
    return df


def _ch3_spread_drift(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    通道 3：价差漂移（tick 直接度量版）。

    使用 price_std（5 分钟内成交价格的标准差）替代 high-low 代理。
    price_std 扩大 → 做市商撤单 → spread 变宽 → 压力前兆。
    """
    ps = df.get("price_std", pd.Series(0, index=df.index))
    ps_mean = ps.rolling(window=window * 2, min_periods=window).mean()
    ps_std = ps.rolling(window=window * 2, min_periods=window).std()
    ps_z = ((ps - ps_mean) / ps_std.replace(0, 1e-10)).fillna(0.0)

    ps_pos = ps_z.clip(lower=0)
    ps_smooth = ps_pos.rolling(window=max(3, window // 3), min_periods=1).mean()
    half_w = max(2, window // 2)
    ps_trend = ps_smooth - ps_smooth.shift(half_w)
    raw = ps_smooth * (ps_trend > 0).astype(float)

    rank_pct = _rolling_percentile(raw, window * 4, window)
    df["ch3_spread_drift"] = rank_pct.fillna(0.0).clip(0, 1)
    return df


def _ch4_order_flow(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    通道 4：订单流（tick 直接度量版，无代理误差）。

    使用 buy_volume_ratio（交易所 B/S 标记直接计算）。
    buy_ratio 持续下降 → 卖压在积累 → 订单流偏向卖方。
    """
    bvr = df.get("buy_volume_ratio", pd.Series(0.5, index=df.index))
    sell_pressure = 1.0 - bvr

    sp_mean = sell_pressure.rolling(window=window * 2, min_periods=window).mean()
    sp_std = sell_pressure.rolling(window=window * 2, min_periods=window).std()
    sp_z = ((sell_pressure - sp_mean) / sp_std.replace(0, 1e-10)).fillna(0.0)

    sp_pos = sp_z.clip(lower=0)
    sp_smooth = sp_pos.rolling(window=max(3, window // 3), min_periods=1).mean()
    half_w = max(2, window // 2)
    sp_trend = sp_smooth - sp_smooth.shift(half_w)
    raw = sp_smooth * (sp_trend > 0).astype(float)

    rank_pct = _rolling_percentile(raw, window * 4, window)
    df["ch4_order_flow"] = rank_pct.fillna(0.0).clip(0, 1)
    return df


def _composite_trigger(
    df: pd.DataFrame,
    window: int,
    percentile: int,
    n_confirm: int,
    min_floor: float,
) -> pd.DataFrame:
    """
    MAX 聚合 + 上升沿检测 + 自适应阈值 → 预警触发信号。
    """
    channels = [
        "ch1_vol_entropy",
        "ch2_depth_erosion",
        "ch3_spread_drift",
        "ch4_order_flow",
    ]

    # MAX 聚合（论文发现单强通道足矣）
    df["composite_raw"] = df[channels].max(axis=1)
    df["composite_smooth"] = df["composite_raw"].rolling(window=3, min_periods=1).mean()

    # 上升沿检测（连续 N 根上升）
    rising = df["composite_smooth"] > df["composite_smooth"].shift(1)
    rising_streak = rising.copy()
    for lag in range(2, n_confirm + 1):
        rising_streak = rising_streak & (
            df["composite_smooth"] > df["composite_smooth"].shift(lag)
        )

    # 自适应阈值（滚动分位数）
    long_w = max(window * 4, 48)
    adaptive_threshold = (
        df["composite_smooth"]
        .rolling(long_w, min_periods=window * 2)
        .quantile(percentile / 100.0)
    )
    effective = adaptive_threshold.clip(lower=min_floor)

    trigger = (
        (df["composite_smooth"] > effective)
        & rising_streak
        & (df["composite_smooth"] > min_floor)
    )

    df["adaptive_threshold"] = effective
    df["signal_trigger"] = trigger.astype(int)
    return df


# ---- 工具函数 ----

def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    滚动百分位排名：每一点在其滚动窗口内的分位数 → [0, 1]。
    比 min-max 归一化更稳健，不受极端离群值影响。
    """
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
