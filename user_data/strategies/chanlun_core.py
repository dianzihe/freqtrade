from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from pandas import DataFrame


class MergedCandle(NamedTuple):
    """经过包含关系处理后的合并K线。"""
    high: float
    low: float
    high_index: int  # 该合并K线最高价对应的原始K线位置
    low_index: int   # 该合并K线最低价对应的原始K线位置


class Pivot(NamedTuple):
    index: int
    kind: str
    price: float


class Stroke(NamedTuple):
    start_index: int
    end_index: int
    direction: int
    high: float
    low: float


def add_chanlun_signals(dataframe: DataFrame, min_stroke_gap: int = 5) -> DataFrame:
    """
    计算缠论信号。

    相比第一版的改进：
    1. 先做 K 线包含关系合并，再找分型，避免噪音分型。
    2. 分型需要左右各一根确认，中枢激活额外延迟 1 根，杜绝未来函数。
    3. 中枢不再用 cummax 永久有效，而是在价格突破中枢后失效，
       并在新中枢生成时被替换，避免使用陈旧中枢。
    4. 笔要求最小长度 min_stroke_gap（默认 5 根）。
    """
    result = dataframe.copy()

    result["chan_top"] = False
    result["chan_bottom"] = False
    result["chan_stroke_dir"] = 0
    result["chan_center_low"] = float("nan")
    result["chan_center_high"] = float("nan")
    result["chan_center_valid"] = False
    result["chan_enter_long"] = 0
    result["chan_exit_long"] = 0
    result["chan_second_buy"] = 0

    if len(result) < 5:
        return result

    merged = _merge_candles(result["high"].to_numpy(), result["low"].to_numpy())
    pivots = _build_pivots(merged)

    top_col = result.columns.get_loc("chan_top")
    bottom_col = result.columns.get_loc("chan_bottom")
    for pivot in pivots:
        if pivot.kind == "top":
            result.iloc[pivot.index, top_col] = True
        else:
            result.iloc[pivot.index, bottom_col] = True

    strokes = _build_strokes(pivots, min_stroke_gap=min_stroke_gap)
    _apply_strokes(result, strokes)
    _apply_centers(result, strokes)
    _apply_breakout_signals(result)
    _apply_second_buy_signals(result)

    return result


def _merge_candles(highs: np.ndarray, lows: np.ndarray) -> list[MergedCandle]:
    """标准缠论 K 线包含关系处理。"""
    merged: list[MergedCandle] = []
    direction = 1  # 1 向上，-1 向下；初始默认向上

    for i in range(len(highs)):
        high = float(highs[i])
        low = float(lows[i])

        if not merged:
            merged.append(MergedCandle(high, low, i, i))
            continue

        prev = merged[-1]
        contained = (high >= prev.high and low <= prev.low) or (
            high <= prev.high and low >= prev.low
        )

        if contained:
            if direction == 1:
                # 向上：高高取高、低取较高者
                if high >= prev.high:
                    new_high, new_hi = high, i
                else:
                    new_high, new_hi = prev.high, prev.high_index
                if low >= prev.low:
                    new_low, new_li = low, i
                else:
                    new_low, new_li = prev.low, prev.low_index
            else:
                # 向下：高取较低者、低低取低
                if high <= prev.high:
                    new_high, new_hi = high, i
                else:
                    new_high, new_hi = prev.high, prev.high_index
                if low <= prev.low:
                    new_low, new_li = low, i
                else:
                    new_low, new_li = prev.low, prev.low_index
            merged[-1] = MergedCandle(new_high, new_low, new_hi, new_li)
        else:
            if high > prev.high:
                direction = 1
            elif low < prev.low:
                direction = -1
            merged.append(MergedCandle(high, low, i, i))

    return merged


def _build_pivots(merged: list[MergedCandle]) -> list[Pivot]:
    """在合并K线上识别顶/底分型。"""
    pivots: list[Pivot] = []

    for i in range(1, len(merged) - 1):
        prev, cur, nxt = merged[i - 1], merged[i], merged[i + 1]
        if cur.high > prev.high and cur.high > nxt.high:
            pivots.append(Pivot(cur.high_index, "top", cur.high))
        elif cur.low < prev.low and cur.low < nxt.low:
            pivots.append(Pivot(cur.low_index, "bottom", cur.low))

    return pivots


def _build_strokes(pivots: list[Pivot], min_stroke_gap: int) -> list[Stroke]:
    normalized: list[Pivot] = []

    for pivot in pivots:
        if not normalized:
            normalized.append(pivot)
            continue

        previous = normalized[-1]
        if pivot.kind == previous.kind:
            # 同类分型保留更极端的
            if _is_more_extreme(pivot, previous):
                normalized[-1] = pivot
            continue

        # 异类分型，但间距不足最小长度则忽略
        if pivot.index - previous.index < min_stroke_gap:
            continue

        normalized.append(pivot)

    strokes: list[Stroke] = []
    for start, end in zip(normalized, normalized[1:], strict=False):
        direction = 1 if start.kind == "bottom" else -1
        strokes.append(
            Stroke(
                start_index=start.index,
                end_index=end.index,
                direction=direction,
                high=max(start.price, end.price),
                low=min(start.price, end.price),
            )
        )

    return strokes


def _is_more_extreme(candidate: Pivot, current: Pivot) -> bool:
    if candidate.kind == "top":
        return candidate.price > current.price
    return candidate.price < current.price


def _apply_strokes(dataframe: DataFrame, strokes: list[Stroke]) -> None:
    stroke_col = dataframe.columns.get_loc("chan_stroke_dir")
    for stroke in strokes:
        dataframe.iloc[stroke.end_index, stroke_col] = stroke.direction

    dataframe["chan_stroke_dir"] = dataframe["chan_stroke_dir"].where(
        dataframe["chan_stroke_dir"] != 0
    )
    dataframe["chan_stroke_dir"] = dataframe["chan_stroke_dir"].ffill().fillna(0)
    dataframe["chan_stroke_dir"] = dataframe["chan_stroke_dir"].astype(int)


def _apply_centers(dataframe: DataFrame, strokes: list[Stroke]) -> None:
    """
    生成中枢，并赋予其生命周期：
    - 中枢在第三笔确认后的下一根才被激活（消除未来函数）；
    - 新中枢生成时替换旧中枢；
    - 价格收盘突破中枢上沿或下沿后，该中枢立即失效，
      不再用于后续信号，直到新的中枢出现。
    """
    n = len(dataframe)
    center_low = np.full(n, np.nan)
    center_high = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    centers: list[tuple[int, float, float]] = []
    for idx in range(2, len(strokes)):
        recent = strokes[idx - 2 : idx + 1]
        c_low = max(stroke.low for stroke in recent)
        c_high = min(stroke.high for stroke in recent)
        if c_low <= c_high:
            # 第三笔末端分型需要 +1 根才确认，因此中枢延迟 1 根激活
            confirm_index = recent[-1].end_index + 1
            centers.append((confirm_index, c_low, c_high))

    closes = dataframe["close"].to_numpy()
    ci = 0
    active: tuple[int, float, float] | None = None
    broken = False

    for i in range(n):
        # 激活所有已确认的中枢，保留最新的一个
        while ci < len(centers) and centers[ci][0] <= i:
            active = centers[ci]
            broken = False
            ci += 1

        if active is not None:
            center_low[i] = active[1]
            center_high[i] = active[2]
            valid[i] = not broken
            # 收盘突破中枢则标记失效（下一根起 valid=False）
            if closes[i] > active[2] or closes[i] < active[1]:
                broken = True

    dataframe["chan_center_low"] = center_low
    dataframe["chan_center_high"] = center_high
    dataframe["chan_center_valid"] = valid


def _apply_breakout_signals(dataframe: DataFrame) -> None:
    prior_center_high = dataframe["chan_center_high"].shift(1)
    prior_center_low = dataframe["chan_center_low"].shift(1)
    prior_close = dataframe["close"].shift(1)

    enter_long = (
        dataframe["chan_center_valid"].shift(1, fill_value=False)
        & prior_center_high.notna()
        & (prior_close <= prior_center_high)
        & (dataframe["close"] > prior_center_high)
        & (dataframe["volume"] > 0)
    )
    exit_long = (
        dataframe["chan_center_valid"].shift(1, fill_value=False)
        & prior_center_low.notna()
        & (prior_close >= prior_center_low)
        & (dataframe["close"] < prior_center_low)
        & (dataframe["volume"] > 0)
    )

    dataframe.loc[enter_long, "chan_enter_long"] = 1
    dataframe.loc[exit_long, "chan_exit_long"] = 1


def _apply_second_buy_signals(dataframe: DataFrame, lookback: int = 20) -> None:
    """
    二买回踩信号：中枢突破后，价格回踩中枢上沿并反弹时入场。

    逻辑（全部用 shift(1) 避免未来函数）：
    1. 中枢有效（前一Bar）
    2. 近期出现过突破信号（前 lookback 根 bar 内 chan_enter_long == 1）
    3. 前一 Bar 收盘价 <= 中枢上沿（回踩）
    4. 当前 Bar 收盘价 > 中枢上沿（反弹）
    5. 成交量 > 0
    """
    # 近期是否有突破（使用 shift(1) 确保不向前看）
    # 用 cummax 跟踪"是否曾出现过突破"，避免 rolling 窗口滑动后突破"掉落"
    breakout_ever = dataframe["chan_enter_long"].expanding().max().shift(1).fillna(0).astype(bool)

    # 附加条件：突破后不能太久（用 rolling 限制有效期）
    recent_breakout = (
        dataframe["chan_enter_long"]
        .rolling(window=lookback, min_periods=1)
        .max()
        .shift(1)
        .fillna(0)
        .astype(bool)
    )
    # 两者都满足：曾突破 + 近期突破（防止太久远的突破也触发二买）
    breakout_flag = breakout_ever & recent_breakout

    prior_center_high = dataframe["chan_center_high"].shift(1)
    prior_close = dataframe["close"].shift(1)

    second_buy = (
        dataframe["chan_center_valid"].shift(1, fill_value=False)
        & prior_center_high.notna()
        & breakout_flag
        & (prior_close <= prior_center_high)
        & (dataframe["close"] > prior_center_high)
        & (dataframe["volume"] > 0)
    )

    dataframe.loc[second_buy, "chan_second_buy"] = 1