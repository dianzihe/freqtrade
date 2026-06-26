from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.persistence import Trade

from user_data.strategies.robust_meme_volatility_risk_strategy import RobustMemeVolatilityRiskStrategy


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
