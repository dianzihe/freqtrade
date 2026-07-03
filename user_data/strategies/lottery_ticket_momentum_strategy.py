from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
    stoploss_from_open,
)


class LotteryTicketMomentumV3(IStrategy):
    """
    V3.1 Lottery-Ticket Momentum

    ============================================================================
    相比 V3 的唯一改动: 入场阈值回到 V1 附近 (V3 撞上低波动窗口 → 0 笔交易).
    架构 (1h/BTC 过滤, 棘轮止损, 票数限制, bug 修复) 全部保留.
    ============================================================================
    调整逻辑:
      - V3 的阈值 (12%涨幅/18%波幅/3.5x量比) 只有极端妖币行情满足, 撞上
        2026-06-25~29 这段低波动期直接空仓.
      - V3.1 下调到 V1 附近并全部参数化, 保留 sniper > breakout > pullback
        的分层门槛, 让 JUP 那类 (8%涨幅/12%波幅/3.8x量比) 能进 breakout 层.

    ⚠️ 提醒: 5 天窗口无法评估彩票策略 (右尾赢家出现概率极低).
       请用 3~6 个月、含妖币行情的窗口回测, 否则会被小样本误导.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    inf_tf = "1h"
    startup_candle_count = 400
    process_only_new_candles = True

    # ── 头寸管理 ──
    max_open_trades = 5
    position_adjustment_enable = False
    use_custom_stoploss = True

    ticket_stake_sniper = 1.0
    ticket_stake_breakout = 1.0
    ticket_stake_pullback = 1.0
    max_daily_tickets = 6
    max_total_tickets = 200

    global_cooldown_min = 120

    time_window_quotas = [
        (0, 6, 2),
        (6, 12, 2),
        (12, 18, 2),
        (18, 24, 2),
    ]

    max_sniper_per_day = 2
    max_breakout_per_day = 3
    max_pullback_per_day = 2

    min_price = 0.001
    max_price = 1000.0

    # ── 退出参数 (未改动) ──
    minimal_roi = {"0": 5.0}
    stoploss = -0.06
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ── 可调超参数 (阈值全部下调 + 新增, 方便 hyperopt) ──
    # 涨幅门槛
    buy_ret_sniper = DecimalParameter(0.08, 0.25, default=0.10, decimals=2, space="buy")
    buy_ret_breakout = DecimalParameter(0.04, 0.15, default=0.06, decimals=2, space="buy")
    buy_ret_pullback = DecimalParameter(0.08, 0.25, default=0.10, decimals=2, space="buy")
    # 波幅门槛
    buy_range_sniper = DecimalParameter(0.10, 0.30, default=0.15, decimals=2, space="buy")
    buy_range_breakout = DecimalParameter(0.06, 0.20, default=0.10, decimals=2, space="buy")
    buy_range_pullback = DecimalParameter(0.10, 0.30, default=0.15, decimals=2, space="buy")
    # 量比门槛
    buy_vol_ratio_a = DecimalParameter(2.2, 5.0, default=2.8, decimals=1, space="buy")
    buy_vol_ratio_b = DecimalParameter(2.0, 4.0, default=2.2, decimals=1, space="buy")
    buy_vol_ratio_c = DecimalParameter(1.5, 3.0, default=1.8, decimals=1, space="buy")
    # RSI
    buy_rsi_low = IntParameter(52, 68, default=55, space="buy")
    buy_rsi_high = IntParameter(72, 86, default=82, space="buy")
    # 流动性下限 (下调)
    buy_min_qvol_a = IntParameter(20000, 80000, default=30000, space="buy")
    buy_min_qvol_b = IntParameter(15000, 60000, default=20000, space="buy")

    # ═══════════════════════════════════════════════════════════════════════
    #  信息对
    # ═══════════════════════════════════════════════════════════════════════

    def informative_pairs(self) -> list[tuple[str, str]]:
        pairs = self.dp.current_whitelist()
        informative = [(pair, self.inf_tf) for pair in pairs]
        if "BTC/USDT" not in pairs:
            informative.append(("BTC/USDT", self.inf_tf))
        return informative

    # ═══════════════════════════════════════════════════════════════════════
    #  工具函数
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return (
            numerator / denominator.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _ema(data: pd.Series, span: int) -> pd.Series:
        return data.ewm(span=span, adjust=False).mean()

    @classmethod
    def _rsi(cls, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = cls._safe_ratio(gain, loss)
        return 100.0 - (100.0 / (1.0 + rs))

    def bot_start(self, **kwargs: Any) -> None:
        self._reset_ticket_counters()

    def _reset_ticket_counters(self) -> None:
        self._ticket_counts_by_day: dict[str, int] = {}
        self._ticket_counts_by_window: dict[str, dict[int, int]] = {}
        self._signal_counts: dict[str, int] = {}
        self._last_any_entry_time: Optional[datetime] = None
        self._total_tickets_bought = 0

    # ═══════════════════════════════════════════════════════════════════════
    #  1h / BTC 指标
    # ═══════════════════════════════════════════════════════════════════════

    def _populate_1h(self, inf: DataFrame) -> DataFrame:
        inf = inf.copy()
        inf["ema20"] = self._ema(inf["close"], 20)
        inf["ema50"] = self._ema(inf["close"], 50)
        inf["ema200"] = self._ema(inf["close"], 200)
        inf["rsi"] = self._rsi(inf["close"], 14)
        return inf

    def _populate_btc(self, btc: DataFrame) -> DataFrame:
        btc = btc.copy()
        out = pd.DataFrame()
        out["date"] = btc["date"]
        out["btc_close"] = btc["close"]
        out["btc_ema20"] = self._ema(btc["close"], 20)
        out["btc_ema50"] = self._ema(btc["close"], 50)
        return out

    # ═══════════════════════════════════════════════════════════════════════
    #  指标计算 (未改动)
    # ═══════════════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = self._ema(dataframe["close"], 20)
        dataframe["ema50"] = self._ema(dataframe["close"], 50)

        NB_24H = 288
        high_24h = dataframe["high"].rolling(NB_24H, min_periods=72).max()
        low_24h = dataframe["low"].rolling(NB_24H, min_periods=72).min()
        close_24h_ago = dataframe["close"].shift(NB_24H)

        dataframe["ret_24h"] = self._safe_ratio(dataframe["close"], close_24h_ago) - 1.0
        dataframe["range_24h"] = self._safe_ratio(high_24h, low_24h) - 1.0
        dataframe["off_high_24h"] = self._safe_ratio(dataframe["close"], high_24h) - 1.0
        dataframe["from_low_24h"] = self._safe_ratio(dataframe["close"], low_24h) - 1.0

        recent_vol = dataframe["volume"].rolling(12, min_periods=3).mean()
        baseline_vol = dataframe["volume"].shift(12).rolling(72, min_periods=24).mean()
        dataframe["volume_ratio"] = self._safe_ratio(recent_vol, baseline_vol)
        dataframe["vol_accel"] = (
            (dataframe["volume_ratio"] > dataframe["volume_ratio"].shift(1))
            & (dataframe["volume_ratio"].shift(1) > dataframe["volume_ratio"].shift(2))
        ).astype(int)

        dataframe["quote_volume_24h"] = (
            dataframe["volume"] * dataframe["close"]
        ).rolling(NB_24H, min_periods=72).sum()

        dataframe["rsi"] = self._rsi(dataframe["close"], 14)

        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        top_wick = dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        dataframe["wick_top_ratio"] = (top_wick / candle_range).fillna(0.0).clip(0.0, 1.0)
        dataframe["green_close"] = (dataframe["close"] > dataframe["open"]).astype(int)

        dataframe["vwap_24h"] = (
            (dataframe["volume"] * dataframe["close"]).rolling(NB_24H, min_periods=72).sum()
            / dataframe["volume"].rolling(NB_24H, min_periods=72).sum().replace(0, np.nan)
        )
        dataframe["above_vwap"] = (dataframe["close"] > dataframe["vwap_24h"]).astype(int)

        if self.dp is not None:
            try:
                inf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_tf)
                if inf is not None and len(inf) > 0:
                    inf = self._populate_1h(inf)
                    dataframe = merge_informative_pair(
                        dataframe, inf, self.timeframe, self.inf_tf, ffill=True
                    )
            except Exception:
                pass

            try:
                btc = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe=self.inf_tf)
                if btc is not None and len(btc) > 0:
                    btc = self._populate_btc(btc)
                    btc["date_merge"] = btc["date"] + pd.Timedelta(hours=1)
                    dataframe = pd.merge_asof(
                        dataframe.sort_values("date"),
                        btc[["date_merge", "btc_close", "btc_ema20", "btc_ema50"]]
                        .sort_values("date_merge"),
                        left_on="date",
                        right_on="date_merge",
                        direction="backward",
                    )
                    dataframe = dataframe.drop(columns=["date_merge"], errors="ignore")
            except Exception:
                pass

        return dataframe

    # ═══════════════════════════════════════════════════════════════════════
    #  入场信号 (阈值下调, 分层保留)
    # ═══════════════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        idx = dataframe.index

        # ── 1h 趋势 (缺数据默认放行) ──
        if "close_1h" in dataframe.columns and "ema50_1h" in dataframe.columns:
            above_ema50_1h = (dataframe["close_1h"] > dataframe["ema50_1h"]).fillna(True)
            above_ema20_1h = (dataframe["close_1h"] > dataframe["ema20_1h"]).fillna(True)
        else:
            above_ema50_1h = pd.Series(True, index=idx)
            above_ema20_1h = pd.Series(True, index=idx)

        # ── BTC 情绪 (三档) ──
        if "btc_close" in dataframe.columns:
            btc_bull = (dataframe["btc_close"] > dataframe["btc_ema50"]).fillna(True)
            btc_neutral = (
                (dataframe["btc_close"] > dataframe["btc_ema20"])
                & (dataframe["btc_close"] <= dataframe["btc_ema50"])
            ).fillna(False)
        else:
            btc_bull = pd.Series(True, index=idx)
            btc_neutral = pd.Series(False, index=idx)
        btc_not_bear = btc_bull | btc_neutral

        green_ok = dataframe["green_close"] == 1
        wicked_ok = dataframe["wick_top_ratio"] <= 0.45

        # ── Sniper A: 强动量突破 (震荡市也允许) ──
        sniper_a = (
            green_ok
            & wicked_ok
            & btc_not_bear
            & above_ema50_1h
            & dataframe["ret_24h"].between(self.buy_ret_sniper.value, 5.0)
            & (dataframe["range_24h"] >= self.buy_range_sniper.value)
            & (dataframe["off_high_24h"] >= -0.18)
            & (dataframe["volume_ratio"] >= self.buy_vol_ratio_a.value)
            & (dataframe["vol_accel"] == 1)
            & (dataframe["quote_volume_24h"] >= float(self.buy_min_qvol_a.value))
            & (dataframe["close"] > dataframe["ema20"])
            & (dataframe["ema20"] >= dataframe["ema50"] * 0.995)
            & dataframe["rsi"].between(self.buy_rsi_low.value, self.buy_rsi_high.value)
            & (dataframe["above_vwap"] == 1)
        )
        dataframe.loc[sniper_a, ["enter_long", "enter_tag"]] = (1, "lottery_sniper_a")

        # ── Breakout B: 中等突破 (仅牛市) ──
        breakout_b = (
            green_ok
            & wicked_ok
            & (dataframe["enter_long"] == 0)
            & btc_bull
            & above_ema20_1h
            & dataframe["ret_24h"].between(self.buy_ret_breakout.value, 1.5)
            & (dataframe["range_24h"] >= self.buy_range_breakout.value)
            & (dataframe["off_high_24h"] >= -0.14)
            & (dataframe["volume_ratio"] >= self.buy_vol_ratio_b.value)
            & (dataframe["vol_accel"] == 1)
            & (dataframe["quote_volume_24h"] >= float(self.buy_min_qvol_b.value))
            & (dataframe["close"] > dataframe["ema20"])
            & dataframe["rsi"].between(self.buy_rsi_low.value, 80)
            & (dataframe["above_vwap"] == 1)
        )
        dataframe.loc[breakout_b, ["enter_long", "enter_tag"]] = (1, "lottery_breakout_b")

        # ── Pullback C: 回调入场 (震荡市也允许) ──
        pullback_c = (
            green_ok
            & wicked_ok
            & (dataframe["enter_long"] == 0)
            & btc_not_bear
            & above_ema50_1h
            & dataframe["ret_24h"].between(self.buy_ret_pullback.value, 4.0)
            & (dataframe["range_24h"] >= self.buy_range_pullback.value)
            & (dataframe["off_high_24h"] >= -0.30)
            & (dataframe["off_high_24h"] <= -0.06)
            & (dataframe["from_low_24h"] >= 0.10)
            & (dataframe["volume_ratio"] >= self.buy_vol_ratio_c.value)
            & (dataframe["close"] > dataframe["ema20"])
            & (dataframe["close"] > dataframe["vwap_24h"])
            & dataframe["rsi"].between(50, 72)
        )
        dataframe.loc[pullback_c, ["enter_long", "enter_tag"]] = (1, "lottery_pullback_c")

        return dataframe

    # ═══════════════════════════════════════════════════════════════════════
    #  退出信号 (关闭)
    # ═══════════════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe

    # ═══════════════════════════════════════════════════════════════════════
    #  阶梯棘轮止损 (未改动)
    # ═══════════════════════════════════════════════════════════════════════

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> float:
        if current_profit >= 1.00:
            return stoploss_from_open(current_profit - 0.20, current_profit)
        if current_profit >= 0.50:
            return stoploss_from_open(current_profit - 0.20, current_profit)
        if current_profit >= 0.25:
            return stoploss_from_open(current_profit - 0.15, current_profit)
        if current_profit >= 0.10:
            return stoploss_from_open(0.0, current_profit)
        return 1

    # ═══════════════════════════════════════════════════════════════════════
    #  自定义退出 (时间止损, 未改动)
    # ═══════════════════════════════════════════════════════════════════════

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        opened = getattr(trade, "open_date_utc", current_time)
        age_h = max((current_time - opened).total_seconds() / 3600.0, 0.0)

        peak_profit = None
        max_rate = getattr(trade, "max_rate", None)
        if max_rate and trade.open_rate:
            peak_profit = (max_rate - trade.open_rate) / trade.open_rate

        if age_h >= 12.0 and (peak_profit is None or peak_profit < 0.03):
            return "stale_dud"
        if age_h >= 36.0:
            return "max_hold"
        return None

    # ═══════════════════════════════════════════════════════════════════════
    #  资金管理 (未改动)
    # ═══════════════════════════════════════════════════════════════════════

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        stake_map = {
            "lottery_sniper_a": self.ticket_stake_sniper,
            "lottery_breakout_b": self.ticket_stake_breakout,
            "lottery_pullback_c": self.ticket_stake_pullback,
        }
        stake = stake_map.get(entry_tag or "", self.ticket_stake_breakout)
        return min(float(stake), float(max_stake))

    # ═══════════════════════════════════════════════════════════════════════
    #  入场确认 (未改动)
    # ═══════════════════════════════════════════════════════════════════════

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None = None,
        side: str = "long",
        **kwargs: Any,
    ) -> bool:
        if not hasattr(self, "_ticket_counts_by_day"):
            self._reset_ticket_counters()

        if rate < self.min_price or rate > self.max_price:
            return False
        if self._total_tickets_bought >= self.max_total_tickets:
            return False

        day_key = current_time.date().isoformat()
        day_count = self._ticket_counts_by_day.get(day_key, 0)
        if day_count >= self.max_daily_tickets:
            return False

        if self._last_any_entry_time is not None:
            gap = (current_time - self._last_any_entry_time).total_seconds() / 60.0
            if gap < self.global_cooldown_min:
                return False

        tag = entry_tag or "unknown"
        max_signal_map = {
            "lottery_sniper_a": self.max_sniper_per_day,
            "lottery_breakout_b": self.max_breakout_per_day,
            "lottery_pullback_c": self.max_pullback_per_day,
        }
        max_for_signal = max_signal_map.get(tag, 99)
        signal_day_key = f"{day_key}:{tag}"
        signal_count = self._signal_counts.get(signal_day_key, 0)
        if signal_count >= max_for_signal:
            return False

        hour = current_time.hour
        window_ok = False
        window_id = -1
        for w_idx, (start, end, quota) in enumerate(self.time_window_quotas):
            if start <= hour < end:
                window_ok = True
                window_id = w_idx
                window_count = self._ticket_counts_by_window.get(day_key, {}).get(w_idx, 0)
                if window_count >= quota:
                    return False
                break
        if not window_ok:
            return False

        expected_stake = {
            "lottery_sniper_a": self.ticket_stake_sniper,
            "lottery_breakout_b": self.ticket_stake_breakout,
            "lottery_pullback_c": self.ticket_stake_pullback,
        }.get(tag, self.ticket_stake_breakout)
        notional = float(amount) * float(rate)
        if notional > expected_stake * 1.08:
            return False

        self._ticket_counts_by_day[day_key] = day_count + 1
        self._total_tickets_bought += 1
        self._last_any_entry_time = current_time
        self._signal_counts[signal_day_key] = signal_count + 1
        if day_key not in self._ticket_counts_by_window:
            self._ticket_counts_by_window[day_key] = {}
        self._ticket_counts_by_window[day_key][window_id] = (
            self._ticket_counts_by_window[day_key].get(window_id, 0) + 1
        )
        return True