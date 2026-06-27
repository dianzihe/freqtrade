# -*- coding: utf-8 -*-
"""
鲁棒 Meme 消融实验策略

继承鲁棒 Meme 策略但关闭 DCA，仅保留原始入场和离场信号。用于回测对比，诊断 DCA 机制的边际贡献。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.persistence import Trade

from user_data.strategies.鲁棒Meme_波动率风控 import RobustMemeVolatilityRiskStrategy


class RobustMemeNoDcaAblationStrategy(RobustMemeVolatilityRiskStrategy):
    """Ablation strategy for backtest diagnostics: same entries/exits, no DCA."""

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | None | tuple[float | None, str | None]:
        return None
