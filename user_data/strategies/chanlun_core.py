from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from pandas import DataFrame


class Pivot(NamedTuple):
    index: int
    kind: str
    price: float


class Stroke(NamedTuple):
    end_index: int
    direction: int
    high: float
    low: float


def add_chanlun_signals(dataframe: DataFrame, min_stroke_gap: int = 3) -> DataFrame:
    result = dataframe.copy()

    result["chan_top"] = False
    result["chan_bottom"] = False
    result["chan_stroke_dir"] = 0
    result["chan_center_low"] = float("nan")
    result["chan_center_high"] = float("nan")
    result["chan_center_valid"] = False
    result["chan_enter_long"] = 0
    result["chan_exit_long"] = 0

    if len(result) < 3:
        return result

    top_mask = (result["high"] > result["high"].shift(1)) & (
        result["high"] > result["high"].shift(-1)
    )
    bottom_mask = (result["low"] < result["low"].shift(1)) & (
        result["low"] < result["low"].shift(-1)
    )

    result.loc[top_mask.fillna(False), "chan_top"] = True
    result.loc[bottom_mask.fillna(False), "chan_bottom"] = True

    pivots = _build_pivots(result)
    strokes = _build_strokes(pivots, min_stroke_gap=min_stroke_gap)
    _apply_strokes(result, strokes)
    _apply_centers(result, strokes)
    _apply_breakout_signals(result)

    return result


def _build_pivots(dataframe: DataFrame) -> list[Pivot]:
    pivots: list[Pivot] = []

    for index, row in dataframe.iterrows():
        positional_index = dataframe.index.get_loc(index)
        if row["chan_top"]:
            pivots.append(Pivot(positional_index, "top", float(row["high"])))
        if row["chan_bottom"]:
            pivots.append(Pivot(positional_index, "bottom", float(row["low"])))

    return pivots


def _build_strokes(pivots: list[Pivot], min_stroke_gap: int) -> list[Stroke]:
    normalized: list[Pivot] = []

    for pivot in pivots:
        if not normalized:
            normalized.append(pivot)
            continue

        previous = normalized[-1]
        if pivot.kind == previous.kind:
            if _is_more_extreme(pivot, previous):
                normalized[-1] = pivot
            continue

        if pivot.index - previous.index < min_stroke_gap:
            continue

        normalized.append(pivot)

    strokes: list[Stroke] = []
    for start, end in zip(normalized, normalized[1:], strict=False):
        direction = 1 if start.kind == "bottom" and end.kind == "top" else -1
        strokes.append(
            Stroke(
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
    for stroke in strokes:
        dataframe.iloc[stroke.end_index, dataframe.columns.get_loc("chan_stroke_dir")] = (
            stroke.direction
        )

    dataframe["chan_stroke_dir"] = dataframe["chan_stroke_dir"].where(
        dataframe["chan_stroke_dir"] != 0
    )
    dataframe["chan_stroke_dir"] = dataframe["chan_stroke_dir"].ffill().fillna(0)
    dataframe["chan_stroke_dir"] = dataframe["chan_stroke_dir"].astype(int)


def _apply_centers(dataframe: DataFrame, strokes: list[Stroke]) -> None:
    for index in range(2, len(strokes)):
        recent = strokes[index - 2 : index + 1]
        center_low = max(stroke.low for stroke in recent)
        center_high = min(stroke.high for stroke in recent)
        end_index = recent[-1].end_index

        if center_low <= center_high:
            dataframe.iloc[end_index, dataframe.columns.get_loc("chan_center_low")] = center_low
            dataframe.iloc[end_index, dataframe.columns.get_loc("chan_center_high")] = center_high
            dataframe.iloc[end_index, dataframe.columns.get_loc("chan_center_valid")] = True

    dataframe["chan_center_low"] = dataframe["chan_center_low"].ffill()
    dataframe["chan_center_high"] = dataframe["chan_center_high"].ffill()
    dataframe["chan_center_valid"] = dataframe["chan_center_valid"].cummax().astype(bool)


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
