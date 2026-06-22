"""
Analyze whether fixed intraday time windows have predictive power for the rest
of the trading day, using downloaded Freqtrade 1-minute OHLCV data.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DATA_DIR = Path("user_data/data")
OUT_DIR = Path("user_data/intraday_time_window_report")
CHART_DIR = OUT_DIR / "charts"

MIN_DAY_MINUTES = 1200
MIN_HOUR_MINUTES = 45
MIN_REMAINING_MINUTES = 60

FONT_FAMILY = ["Microsoft YaHei", "SimHei", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "neutral": {"light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}


@dataclass(frozen=True)
class WindowSpec:
    label: str
    start_hour: int
    end_hour: int
    kind: str

    @property
    def beijing_label(self) -> str:
        start = (self.start_hour + 8) % 24
        end = (self.end_hour + 8) % 24
        return f"{start:02d}:00-{end:02d}:00"

    @property
    def utc_label(self) -> str:
        return f"{self.start_hour:02d}:00-{self.end_hour:02d}:00"


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "axes.unicode_minus": False,
        },
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}%"


def bps(value_pct: float, digits: int = 1) -> str:
    if pd.isna(value_pct):
        return "n/a"
    return f"{value_pct * 100:.{digits}f} bp"


def read_ohlcv() -> pd.DataFrame:
    files = sorted(DATA_DIR.rglob("*.feather"))
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for fp in files:
        try:
            frame = pd.read_feather(fp)
            needed = {"date", "open", "high", "low", "close", "volume"}
            if not needed.issubset(frame.columns):
                skipped.append(str(fp))
                continue
            frame = frame.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            frame["exchange"] = fp.parent.name
            frame["pair"] = fp.stem.replace("-1m", "")
            frames.append(frame)
        except Exception:
            skipped.append(str(fp))

    if not frames:
        raise RuntimeError(f"No readable feather OHLCV files found under {DATA_DIR}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values(["exchange", "pair", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.floor("D")
    df["hour"] = df["date"].dt.hour.astype("int16")
    df["exchange"] = df["exchange"].astype("category")
    df["pair"] = df["pair"].astype("category")
    return df


def complete_day_keys(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["exchange", "pair", "day"], observed=True)
        .agg(
            minute_count=("date", "count"),
            day_open=("open", "first"),
            day_close=("close", "last"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    daily["day_return_pct"] = (daily["day_close"] / daily["day_open"] - 1.0) * 100
    daily["day_range_pct"] = (daily["day_high"] / daily["day_low"] - 1.0) * 100
    daily["is_complete"] = daily["minute_count"] >= MIN_DAY_MINUTES
    return daily


def hourly_profile(df: pd.DataFrame, complete_daily: pd.DataFrame) -> pd.DataFrame:
    valid_keys = complete_daily.loc[complete_daily["is_complete"], ["exchange", "pair", "day"]]
    scoped = df.merge(valid_keys, on=["exchange", "pair", "day"], how="inner")
    hourly = (
        scoped.groupby(["exchange", "pair", "day", "hour"], observed=True)
        .agg(
            minute_count=("date", "count"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    hourly = hourly.loc[hourly["minute_count"] >= MIN_HOUR_MINUTES].copy()
    hourly["return_pct"] = (hourly["close"] / hourly["open"] - 1.0) * 100
    hourly["range_pct"] = (hourly["high"] / hourly["low"] - 1.0) * 100
    return hourly


def make_windows() -> list[WindowSpec]:
    hourly = [WindowSpec(f"{h:02d}:00-{h + 1:02d}:00 UTC", h, h + 1, "one_hour") for h in range(23)]
    four_hour = [
        WindowSpec("Asia early", 0, 4, "four_hour"),
        WindowSpec("Asia late", 4, 8, "four_hour"),
        WindowSpec("Europe morning", 8, 12, "four_hour"),
        WindowSpec("Europe-US overlap", 12, 16, "four_hour"),
        WindowSpec("US afternoon", 16, 20, "four_hour"),
    ]
    early = [WindowSpec(f"First {n}h", 0, n, "early_cumulative") for n in range(1, 13)]
    return hourly + four_hour + early


def calc_return(part: pd.DataFrame) -> float:
    return (part["close"].iloc[-1] / part["open"].iloc[0] - 1.0) * 100


def build_window_observations(hourly: pd.DataFrame, complete_daily: pd.DataFrame, windows: list[WindowSpec]) -> pd.DataFrame:
    valid_keys = complete_daily.loc[complete_daily["is_complete"], ["exchange", "pair", "day"]]
    scoped = hourly.merge(valid_keys, on=["exchange", "pair", "day"], how="inner")

    idx = ["exchange", "pair", "day"]
    wide_open = scoped.set_index(idx + ["hour"])["open"].unstack("hour")
    wide_close = scoped.set_index(idx + ["hour"])["close"].unstack("hour")
    wide_count = scoped.set_index(idx + ["hour"])["minute_count"].unstack("hour").fillna(0)
    daily_idx = complete_daily.loc[complete_daily["is_complete"]].set_index(idx)

    frames: list[pd.DataFrame] = []
    for window in windows:
        segment_hours = list(range(window.start_hour, window.end_hour))
        remaining_hours = list(range(window.end_hour, 24))
        if not segment_hours or not remaining_hours:
            continue

        required_segment = wide_count.reindex(columns=segment_hours, fill_value=0).ge(MIN_HOUR_MINUTES).all(axis=1)
        remaining_minutes = wide_count.reindex(columns=remaining_hours, fill_value=0).sum(axis=1)
        segment_minutes = wide_count.reindex(columns=segment_hours, fill_value=0).sum(axis=1)

        segment_open = wide_open.get(window.start_hour)
        segment_close = wide_close.get(window.end_hour - 1)
        remaining_open = wide_open.get(window.end_hour)
        day_close = daily_idx.loc[wide_open.index, "day_close"]
        day_open = daily_idx.loc[wide_open.index, "day_open"]

        valid = (
            required_segment
            & remaining_minutes.ge(MIN_REMAINING_MINUTES)
            & segment_open.notna()
            & segment_close.notna()
            & remaining_open.notna()
            & day_close.notna()
        )
        if not valid.any():
            continue

        part = pd.DataFrame(index=wide_open.index[valid])
        part["window"] = window.label
        part["kind"] = window.kind
        part["start_hour"] = window.start_hour
        part["end_hour"] = window.end_hour
        part["utc_time"] = window.utc_label
        part["beijing_time"] = window.beijing_label
        part["segment_return_pct"] = (segment_close.loc[valid] / segment_open.loc[valid] - 1.0) * 100
        part["remaining_return_pct"] = (day_close.loc[valid] / remaining_open.loc[valid] - 1.0) * 100
        part["day_return_pct"] = (day_close.loc[valid] / day_open.loc[valid] - 1.0) * 100
        part["segment_minutes"] = segment_minutes.loc[valid].astype(int)
        part["remaining_minutes"] = remaining_minutes.loc[valid].astype(int)
        frames.append(part.reset_index())

    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if obs.empty:
        raise RuntimeError("No valid fixed-window observations were available after completeness filtering.")
    obs["exchange"] = obs["exchange"].astype(str)
    obs["pair"] = obs["pair"].astype(str)
    obs["day"] = pd.to_datetime(obs["day"]).dt.date.astype(str)
    obs["segment_up"] = obs["segment_return_pct"] > 0
    obs["remaining_up"] = obs["remaining_return_pct"] > 0
    obs["same_direction"] = obs["segment_up"] == obs["remaining_up"]
    return obs


def summarize_window_observations(obs: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for (kind, window, start_hour, end_hour, utc_time, beijing_time), part in obs.groupby(
        ["kind", "window", "start_hour", "end_hour", "utc_time", "beijing_time"], sort=False
    ):
        up = part.loc[part["segment_up"]]
        down = part.loc[~part["segment_up"]]
        if len(up) >= 2 and len(down) >= 2:
            t_result = stats.ttest_ind(up["remaining_return_pct"], down["remaining_return_pct"], equal_var=False, nan_policy="omit")
            p_value = float(t_result.pvalue) if not math.isnan(float(t_result.pvalue)) else np.nan
        else:
            p_value = np.nan

        corr = part["segment_return_pct"].corr(part["remaining_return_pct"])
        base_up_rate = part["remaining_up"].mean() * 100
        baseline_accuracy = max(base_up_rate, 100 - base_up_rate)
        same_direction = part["same_direction"].mean() * 100
        summaries.append(
            {
                "kind": kind,
                "window": window,
                "start_hour": int(start_hour),
                "end_hour": int(end_hour),
                "utc_time": utc_time,
                "beijing_time": beijing_time,
                "sample_count": int(len(part)),
                "unique_days": int(part["day"].nunique()),
                "unique_pairs": int(part["pair"].nunique()),
                "unique_exchanges": int(part["exchange"].nunique()),
                "segment_mean_pct": part["segment_return_pct"].mean(),
                "remaining_mean_pct": part["remaining_return_pct"].mean(),
                "segment_remaining_corr": corr,
                "same_direction_accuracy_pct": same_direction,
                "remaining_up_base_rate_pct": base_up_rate,
                "majority_baseline_accuracy_pct": baseline_accuracy,
                "same_direction_edge_vs_baseline_pp": same_direction - baseline_accuracy,
                "remaining_up_if_segment_up_pct": up["remaining_up"].mean() * 100 if len(up) else np.nan,
                "remaining_up_if_segment_down_pct": down["remaining_up"].mean() * 100 if len(down) else np.nan,
                "remaining_mean_if_segment_up_pct": up["remaining_return_pct"].mean() if len(up) else np.nan,
                "remaining_mean_if_segment_down_pct": down["remaining_return_pct"].mean() if len(down) else np.nan,
                "conditional_mean_gap_pct": (up["remaining_return_pct"].mean() - down["remaining_return_pct"].mean())
                if len(up) and len(down)
                else np.nan,
                "welch_p_value": p_value,
            }
        )
    return pd.DataFrame(summaries)


def summarize_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    summary = (
        hourly.groupby("hour")
        .agg(
            sample_count=("return_pct", "count"),
            mean_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            win_rate_pct=("return_pct", lambda s: (s > 0).mean() * 100),
            mean_range_pct=("range_pct", "mean"),
            mean_volume=("volume", "mean"),
        )
        .reset_index()
    )
    summary["beijing_hour"] = (summary["hour"] + 8) % 24
    return summary


def market_day_summary(obs: pd.DataFrame) -> pd.DataFrame:
    market = (
        obs.groupby(["kind", "window", "start_hour", "end_hour", "utc_time", "beijing_time", "day"])
        .agg(
            segment_return_pct=("segment_return_pct", "mean"),
            remaining_return_pct=("remaining_return_pct", "mean"),
            same_direction=("same_direction", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for keys, part in market.groupby(["kind", "window", "start_hour", "end_hour", "utc_time", "beijing_time"], sort=False):
        corr = part["segment_return_pct"].corr(part["remaining_return_pct"])
        rows.append(
            {
                "kind": keys[0],
                "window": keys[1],
                "start_hour": int(keys[2]),
                "end_hour": int(keys[3]),
                "utc_time": keys[4],
                "beijing_time": keys[5],
                "market_day_count": int(len(part)),
                "market_day_corr": corr,
                "market_day_same_direction_pct": (part["segment_return_pct"].gt(0) == part["remaining_return_pct"].gt(0)).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def plot_hourly_return(hourly_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    colors = np.where(hourly_summary["mean_return_pct"] >= 0, COLORS["olive"]["base"], COLORS["orange"]["base"])
    edge_colors = np.where(hourly_summary["mean_return_pct"] >= 0, COLORS["olive"]["dark"], COLORS["orange"]["dark"])
    bars = ax.bar(hourly_summary["hour"], hourly_summary["mean_return_pct"] * 100, color=colors, edgecolor=edge_colors, linewidth=1)
    ax.axhline(0, color=TOKENS["ink"], linewidth=1)
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Mean 1h return (bp)")
    ax.set_xticks(range(24))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for bar, value in zip(bars, hourly_summary["mean_return_pct"] * 100):
        if abs(value) >= 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    add_chart_header(
        fig,
        ax,
        "Hourly return profile",
        "Average 1-hour return by UTC hour across complete instrument-days; positive bars are upward drift.",
    )
    path = CHART_DIR / "hourly_return_profile.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hourly_range(hourly_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.barplot(data=hourly_summary, x="hour", y=hourly_summary["mean_range_pct"] * 100, ax=ax, color=COLORS["blue"]["base"], edgecolor=COLORS["blue"]["dark"], linewidth=1)
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Mean high-low range (bp)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    add_chart_header(
        fig,
        ax,
        "Hourly volatility profile",
        "Average intrahour high-low range by UTC hour; higher bars mark more active trading windows.",
    )
    path = CHART_DIR / "hourly_range_profile.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_early_predictive(early_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    plot_df = early_summary.sort_values("end_hour")
    sns.lineplot(data=plot_df, x="end_hour", y="same_direction_accuracy_pct", marker="o", ax=ax, color=COLORS["blue"]["mid"], label="Same-direction accuracy")
    sns.lineplot(data=plot_df, x="end_hour", y="majority_baseline_accuracy_pct", marker="o", ax=ax, color=COLORS["neutral"]["mid"], linestyle="--", label="Majority baseline")
    ax.set_xlabel("First N hours after 00:00 UTC")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(1, 13))
    ax.set_ylim(max(0, min(plot_df["same_direction_accuracy_pct"].min(), plot_df["majority_baseline_accuracy_pct"].min()) - 4), 100)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "First-hours direction versus the remaining day",
        "Strict test: first N-hour return is compared with the post-window return, excluding the window itself.",
    )
    path = CHART_DIR / "early_predictive_power.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_four_hour_windows(four_hour_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    plot_df = four_hour_summary.sort_values("conditional_mean_gap_pct")
    colors = np.where(plot_df["conditional_mean_gap_pct"] >= 0, COLORS["olive"]["base"], COLORS["orange"]["base"])
    edge_colors = np.where(plot_df["conditional_mean_gap_pct"] >= 0, COLORS["olive"]["dark"], COLORS["orange"]["dark"])
    bars = ax.barh(plot_df["utc_time"], plot_df["conditional_mean_gap_pct"] * 100, color=colors, edgecolor=edge_colors, linewidth=1)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1)
    ax.set_xlabel("Remaining-day mean gap after segment-up vs segment-down (bp)")
    ax.set_ylabel("UTC window")
    for bar, value in zip(bars, plot_df["conditional_mean_gap_pct"] * 100):
        ax.text(value + (0.8 if value >= 0 else -0.8), bar.get_y() + bar.get_height() / 2, f"{value:+.1f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    add_chart_header(
        fig,
        ax,
        "Which 4-hour windows separate the rest of the day",
        "Positive values mean the remaining day was stronger after that window rose than after it fell.",
    )
    path = CHART_DIR / "four_hour_window_gap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_btc_vs_all(all_early: pd.DataFrame, btc_early: pd.DataFrame) -> Path | None:
    if btc_early.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5.4))
    all_plot = all_early.sort_values("end_hour").assign(series="All pairs")
    btc_plot = btc_early.sort_values("end_hour").assign(series="BTC/USDT")
    plot_df = pd.concat([all_plot, btc_plot], ignore_index=True)
    palette = {"All pairs": COLORS["blue"]["mid"], "BTC/USDT": COLORS["orange"]["mid"]}
    sns.lineplot(data=plot_df, x="end_hour", y="same_direction_accuracy_pct", hue="series", marker="o", palette=palette, ax=ax)
    ax.set_xlabel("First N hours after 00:00 UTC")
    ax.set_ylabel("Same-direction accuracy (%)")
    ax.set_xticks(range(1, 13))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "BTC/USDT versus the broad downloaded universe",
        "Direction accuracy uses post-window remaining-day return, so it avoids overlap with the signal window.",
    )
    path = CHART_DIR / "btc_vs_all_early_accuracy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def image_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def table_rows(df: pd.DataFrame, columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    display = df if max_rows is None else df.head(max_rows)
    rows = []
    for _, row in display.iterrows():
        cells = []
        for key, kind in columns:
            value = row[key]
            if kind == "pct":
                text = pct(value, 1)
            elif kind == "bp":
                text = bps(value, 1)
            elif kind == "num":
                text = f"{value:,.0f}"
            elif kind == "float":
                text = "n/a" if pd.isna(value) else f"{value:.3f}"
            elif kind == "p":
                text = "n/a" if pd.isna(value) else ("<0.001" if value < 0.001 else f"{value:.3f}")
            else:
                text = str(value)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def render_html(
    stats_summary: dict[str, object],
    hourly_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    market_summary: pd.DataFrame,
    btc_window_summary: pd.DataFrame,
    chart_paths: dict[str, Path | None],
) -> Path:
    hourly_best = hourly_summary.sort_values("mean_return_pct", ascending=False).iloc[0]
    hourly_worst = hourly_summary.sort_values("mean_return_pct", ascending=True).iloc[0]
    early_summary = window_summary.loc[window_summary["kind"] == "early_cumulative"].sort_values("end_hour")
    four_hour = window_summary.loc[window_summary["kind"] == "four_hour"].sort_values("conditional_mean_gap_pct", ascending=False)
    best_gap = four_hour.iloc[0]
    strongest_reversal = four_hour.sort_values("conditional_mean_gap_pct", ascending=True).iloc[0]
    best_accuracy = early_summary.sort_values("same_direction_edge_vs_baseline_pp", ascending=False).iloc[0]
    market_early = market_summary.loc[market_summary["kind"] == "early_cumulative"].sort_values("end_hour")
    btc_early = btc_window_summary.loc[btc_window_summary["kind"] == "early_cumulative"].sort_values("end_hour") if not btc_window_summary.empty else pd.DataFrame()

    chart_imgs = {
        name: image_data_uri(path)
        for name, path in chart_paths.items()
        if path is not None
    }

    rows_four_hour = table_rows(
        four_hour,
        [
            ("window", "text"),
            ("utc_time", "text"),
            ("beijing_time", "text"),
            ("sample_count", "num"),
            ("same_direction_accuracy_pct", "pct"),
            ("same_direction_edge_vs_baseline_pp", "pct"),
            ("conditional_mean_gap_pct", "bp"),
            ("segment_remaining_corr", "float"),
            ("welch_p_value", "p"),
        ],
    )
    rows_early = table_rows(
        early_summary,
        [
            ("window", "text"),
            ("sample_count", "num"),
            ("same_direction_accuracy_pct", "pct"),
            ("majority_baseline_accuracy_pct", "pct"),
            ("same_direction_edge_vs_baseline_pp", "pct"),
            ("conditional_mean_gap_pct", "bp"),
            ("segment_remaining_corr", "float"),
        ],
    )
    market_rows = table_rows(
        market_early,
        [
            ("window", "text"),
            ("market_day_count", "num"),
            ("market_day_same_direction_pct", "pct"),
            ("market_day_corr", "float"),
        ],
    )
    btc_rows = ""
    if not btc_early.empty:
        btc_rows = table_rows(
            btc_early,
            [
                ("window", "text"),
                ("sample_count", "num"),
                ("same_direction_accuracy_pct", "pct"),
                ("same_direction_edge_vs_baseline_pp", "pct"),
                ("conditional_mean_gap_pct", "bp"),
                ("segment_remaining_corr", "float"),
            ],
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>固定时间段与当天行情走势分析</title>
  <style>
    :root {{
      --ink: #1f2430;
      --muted: #667085;
      --line: #e5e7eb;
      --panel: #ffffff;
      --bg: #f6f7fb;
      --blue: #2e4780;
      --orange: #804126;
      --olive: #386411;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; line-height: 1.62; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 36px 24px 64px; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; font-size: 21px; }}
    h3 {{ margin: 18px 0 10px; font-size: 16px; }}
    p {{ margin: 8px 0 12px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
    .section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 22px 24px; margin: 18px 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 20px 0; }}
    .metric {{ background: #fcfcfd; border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    ul {{ padding-left: 20px; }}
    img {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; background: white; margin: 10px 0 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0 4px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fafafa; }}
    .note {{ color: var(--muted); font-size: 13px; }}
    .callout {{ border-left: 4px solid var(--blue); padding: 10px 14px; background: #f4f7ff; border-radius: 6px; }}
    @media (max-width: 760px) {{
      main {{ padding: 24px 14px 48px; }}
      .summary {{ grid-template-columns: 1fr; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>固定时间段与当天行情走势分析</h1>
  <p class="subtitle">基于本地 <code>user_data/data</code> 下载的 1 分钟 OHLCV；时间按 UTC 统计，并同时给出北京时间参考。</p>

  <section class="section">
    <h2>Executive Summary</h2>
    <ul>
      <li><strong>有分时段差异，但“固定时段方向决定剩余全天方向”的证据不成立。</strong>完整日样本有 {stats_summary["complete_days"]} 个 UTC 日期；横截面样本多但高度相关，适合筛选假设，不适合直接下长期结论。</li>
      <li><strong>严格口径下，早段顺势预测没有跑赢简单基线。</strong>表现最好的 {best_accuracy["window"]} 同向准确率为 {pct(best_accuracy["same_direction_accuracy_pct"], 1)}，仍比“永远猜多数方向”的基线低 {pct(abs(best_accuracy["same_direction_edge_vs_baseline_pp"]), 1)}。</li>
      <li><strong>更有价值的是条件均值差，而不是方向命中率。</strong>{best_gap["utc_time"]} UTC（北京时间 {best_gap["beijing_time"]}）上涨后，剩余当天平均表现比该窗口下跌后高 {bps(best_gap["conditional_mean_gap_pct"], 1)}；但 {strongest_reversal["utc_time"]} UTC（北京时间 {strongest_reversal["beijing_time"]}）更像反转窗口，条件均值差为 {bps(strongest_reversal["conditional_mean_gap_pct"], 1)}。</li>
      <li><strong>小时级收益有明显噪声。</strong>平均 1 小时收益最强为 {int(hourly_best["hour"]):02d}:00 UTC（北京时间 {int(hourly_best["beijing_hour"]):02d}:00，{bps(hourly_best["mean_return_pct"], 1)}），最弱为 {int(hourly_worst["hour"]):02d}:00 UTC（北京时间 {int(hourly_worst["beijing_hour"]):02d}:00，{bps(hourly_worst["mean_return_pct"], 1)}），但这些数值需要跨更长历史复验。</li>
    </ul>
  </section>

  <section class="section">
    <h2>数据范围和检验口径</h2>
    <div class="summary">
      <div class="metric"><strong>{stats_summary["rows"]:,}</strong><span>1 分钟 K 线行数</span></div>
      <div class="metric"><strong>{stats_summary["files"]:,}</strong><span>feather 文件</span></div>
      <div class="metric"><strong>{stats_summary["complete_observations"]:,}</strong><span>完整 instrument-day 样本</span></div>
    </div>
    <p>数据覆盖 {stats_summary["date_min"]} 到 {stats_summary["date_max"]} UTC，交易所 {stats_summary["exchanges"]} 个，交易对 {stats_summary["pairs"]} 个。为了避免半天数据污染，只有分钟数不少于 {MIN_DAY_MINUTES} 的交易对-交易所-日期进入“全天/剩余当天”检验。</p>
    <p class="callout">核心检验不是把某个时段和“整天收益”直接相关，因为那会把该时段本身计入全天，容易高估。这里用更严格的方式：固定时段收益只预测该时段结束后到当天结束的剩余收益。</p>
  </section>

  <section class="section">
    <h2>一天内的收益和波动确实有时间分布</h2>
    <p><strong>分时段本身存在差异。</strong>平均收益、波动和成交量在 UTC 小时上并不均匀，这说明“什么时候交易”会影响短线策略的噪声环境和机会密度。</p>
    <img src="{chart_imgs["hourly_return"]}" alt="Hourly return profile">
    <img src="{chart_imgs["hourly_range"]}" alt="Hourly volatility profile">
    <p class="note">图中收益单位为基点，1 bp = 0.01%。这些是样本内平均值，不代表每一天都会复现。</p>
  </section>

  <section class="section">
    <h2>早段顺势没有跑赢简单基线</h2>
    <p><strong>从 00:00 UTC 开始累计 N 小时后，方向延续率没有跑赢简单多数基线。</strong>也就是说，在这批数据里，“早上涨，后面也更可能涨”不是稳健规则。若要放进策略，应把早段走势当作风险过滤或反转候选，而不是单独顺势开仓信号。</p>
    <img src="{chart_imgs["early_predictive"]}" alt="Early predictive power">
    <table>
      <thead><tr><th>窗口</th><th>样本</th><th>同向准确率</th><th>多数基线</th><th>超额准确率</th><th>条件均值差</th><th>相关</th></tr></thead>
      <tbody>{rows_early}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>4 小时窗口体现条件差异，而不是简单方向命中</h2>
    <p><strong>不同固定窗口的预测力不一样，但多数不是简单同向。</strong>下表的“条件均值差”表示：该窗口上涨后的剩余当天平均收益，减去该窗口下跌后的剩余当天平均收益。正数偏顺势，负数偏反转或均值回归。</p>
    <img src="{chart_imgs["four_hour"]}" alt="Four-hour window gap">
    <table>
      <thead><tr><th>窗口</th><th>UTC</th><th>北京时间</th><th>样本</th><th>同向准确率</th><th>超额准确率</th><th>条件均值差</th><th>相关</th><th>p 值</th></tr></thead>
      <tbody>{rows_four_hour}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>BTC 单独看时，信号更需要谨慎</h2>
    <p><strong>BTC/USDT 的样本比全币种横截面小很多。</strong>全币种结果可能混合了山寨币 beta、交易所差异和重复报价；BTC 结果更接近市场主轴，但样本数也更少。</p>
    {f'<img src="{chart_imgs["btc_vs_all"]}" alt="BTC versus all pairs">' if "btc_vs_all" in chart_imgs else '<p class="note">BTC/USDT 可用样本不足，未生成对比图。</p>'}
    {f'<table><thead><tr><th>窗口</th><th>样本</th><th>同向准确率</th><th>超额准确率</th><th>条件均值差</th><th>相关</th></tr></thead><tbody>{btc_rows}</tbody></table>' if btc_rows else ''}
  </section>

  <section class="section">
    <h2>按自然日聚合后，统计置信度明显下降</h2>
    <p><strong>横截面样本不能当作完全独立样本。</strong>当把同一天的所有币种和交易所先平均成一个市场日后，只有 {stats_summary["complete_days"]} 个日期；这个口径更保守，也提醒我们当前数据窗口太短。</p>
    <table>
      <thead><tr><th>窗口</th><th>市场日数</th><th>市场日同向率</th><th>市场日相关</th></tr></thead>
      <tbody>{market_rows}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>交易上的用法建议</h2>
    <ol>
      <li><strong>不要直接用“某小时平均涨”开仓。</strong>小时平均收益很小，容易被手续费、滑点和样本期风格吞掉。</li>
      <li><strong>更适合做过滤器。</strong>例如在顺势区分度较高的窗口后提高顺势信号权重，在反转区分度较高的窗口后避免追涨杀跌。</li>
      <li><strong>回测时必须做 walk-forward。</strong>至少用更长历史，把不同月份、牛熊阶段、周末/工作日分开验证，并且只用窗口结束后的信号做交易。</li>
      <li><strong>优先验证 BTC 主轴和全币种 beta。</strong>如果 BTC 与全市场方向一致，分时段信号更可信；如果背离，山寨币横截面结果可能只是短期轮动。</li>
    </ol>
  </section>

  <section class="section">
    <h2>主要 caveats</h2>
    <ul>
      <li>当前下载数据只有约数天完整 UTC 日，无法确认长期季节性。</li>
      <li>多个交易所和交易对强相关，横截面样本数会放大显著性，不能等同独立实验。</li>
      <li>报告只分析行情统计关系，没有扣除手续费、滑点、盘口深度，也没有模拟具体开平仓规则。</li>
      <li>UTC 日切分未必等同你的策略交易日；若策略按北京时间或交易所 session 管理风险，应重新按该时区切分。</li>
    </ul>
  </section>
</main>
</body>
</html>"""

    path = OUT_DIR / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    df = read_ohlcv()
    daily = complete_day_keys(df)
    complete_daily = daily.loc[daily["is_complete"]].copy()
    hourly = hourly_profile(df, daily)
    hourly_summary = summarize_hourly(hourly)

    windows = make_windows()
    obs = build_window_observations(hourly, daily, windows)
    window_summary = summarize_window_observations(obs)
    market_summary = market_day_summary(obs)

    btc_obs = obs.loc[obs["pair"].astype(str).eq("BTC_USDT")].copy()
    btc_window_summary = summarize_window_observations(btc_obs) if not btc_obs.empty else pd.DataFrame()

    chart_paths = {
        "hourly_return": plot_hourly_return(hourly_summary),
        "hourly_range": plot_hourly_range(hourly_summary),
        "early_predictive": plot_early_predictive(window_summary.loc[window_summary["kind"] == "early_cumulative"]),
        "four_hour": plot_four_hour_windows(window_summary.loc[window_summary["kind"] == "four_hour"]),
        "btc_vs_all": plot_btc_vs_all(
            window_summary.loc[window_summary["kind"] == "early_cumulative"],
            btc_window_summary.loc[btc_window_summary["kind"] == "early_cumulative"] if not btc_window_summary.empty else pd.DataFrame(),
        ),
    }

    stats_summary = {
        "rows": int(len(df)),
        "files": int(len(list(DATA_DIR.rglob("*.feather")))),
        "exchanges": int(df["exchange"].nunique()),
        "pairs": int(df["pair"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "complete_days": int(complete_daily["day"].nunique()),
        "complete_observations": int(len(complete_daily)),
        "complete_pairs": int(complete_daily["pair"].nunique()),
        "complete_exchanges": int(complete_daily["exchange"].nunique()),
    }

    report_path = render_html(stats_summary, hourly_summary, window_summary, market_summary, btc_window_summary, chart_paths)

    hourly_summary.to_csv(OUT_DIR / "hourly_summary.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(OUT_DIR / "window_summary.csv", index=False, encoding="utf-8-sig")
    market_summary.to_csv(OUT_DIR / "market_day_summary.csv", index=False, encoding="utf-8-sig")
    if not btc_window_summary.empty:
        btc_window_summary.to_csv(OUT_DIR / "btc_window_summary.csv", index=False, encoding="utf-8-sig")
    obs.to_parquet(OUT_DIR / "window_observations.parquet", index=False)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "stats": stats_summary,
                "report": str(report_path),
                "charts": {name: str(path) for name, path in chart_paths.items() if path is not None},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"report": str(report_path), "stats": stats_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
