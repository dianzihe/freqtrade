# -*- coding: utf-8 -*-
"""
GoldKylin 风控优先 HyperOpt Loss

核心思想:优化器不该只盯总收益,而要在以下之间取平衡:
  1. 风险调整后收益 (Calmar 风格: 年化收益 / 最大回撤)
  2. 交易频次下限 (样本太少的参数直接重罚,防过拟合)
  3. 单笔期望 (要能覆盖费+滑点,expectancy 必须为正)
  4. 回撤硬惩罚 (超过阈值 → 指数级放大 loss)

返回值越小越好。
"""

from __future__ import annotations

import numpy as np
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


class GoldKylinLoss(IHyperOptLoss):

    # ---- 可调硬约束 ----
    MIN_TRADES = 40                 # 回测期内至少 40 笔,否则重罚
    TARGET_TRADES = 150             # 达到此量级不再因频次加分
    MAX_ACCEPTABLE_DD = 0.15        # 回撤软上限;超过开始指数惩罚
    HARD_DD_CUTOFF = 0.30           # 回撤硬红线;触及基本判死刑
    MIN_WIN_EXPECTANCY = 0.0        # 单笔期望必须 > 0

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date,
        max_date,
        config: dict,
        processed: dict,
        backtest_stats: dict,
        **kwargs,
    ) -> float:

        # 1) 交易数不足:直接返回大 loss,并按缺口线性加重
        if trade_count < GoldKylinLoss.MIN_TRADES:
            deficit = (GoldKylinLoss.MIN_TRADES - trade_count) / GoldKylinLoss.MIN_TRADES
            return 10.0 + deficit * 20.0

        total_profit = results["profit_ratio"].sum()          # 累积收益(以 stake 计)
        profit_abs = backtest_stats.get("profit_total", total_profit)

        # 2) 最大回撤(优先用引擎统计,取不到则从 profit 曲线自算)
        max_dd = backtest_stats.get("max_drawdown_account")
        if max_dd is None:
            max_dd = backtest_stats.get("max_drawdown", None)
        if max_dd is None:
            cum = results["profit_ratio"].cumsum()
            peak = cum.cummax()
            max_dd = float((peak - cum).max()) if len(cum) else 0.0
        max_dd = abs(float(max_dd))

        # 回撤触及硬红线:直接判死
        if max_dd >= GoldKylinLoss.HARD_DD_CUTOFF:
            return 30.0 + max_dd * 10.0

        # 3) 单笔期望(expectancy):要覆盖成本
        wins = results.loc[results["profit_ratio"] > 0, "profit_ratio"]
        losses = results.loc[results["profit_ratio"] <= 0, "profit_ratio"]
        win_rate = len(wins) / trade_count
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = abs(losses.mean()) if len(losses) else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        if expectancy <= GoldKylinLoss.MIN_WIN_EXPECTANCY:
            return 8.0 - expectancy * 50.0   # 期望为负 → loss 抬升

        # 4) 年化(用于 Calmar);按回测天数折算
        days = max((max_date - min_date).days, 1)
        annualized = total_profit * (365.0 / days)

        # Calmar 风格核心分:年化收益 / (回撤 + 小 epsilon)
        calmar = annualized / (max_dd + 0.02)

        # 5) 回撤软惩罚:超过软上限后指数放大
        if max_dd > GoldKylinLoss.MAX_ACCEPTABLE_DD:
            over = (max_dd - GoldKylinLoss.MAX_ACCEPTABLE_DD) / GoldKylinLoss.MAX_ACCEPTABLE_DD
            calmar *= np.exp(-2.0 * over)

        # 6) 频次奖励:交易数越接近 TARGET 越稳(缓和过拟合)
        trade_factor = min(1.0, trade_count / GoldKylinLoss.TARGET_TRADES)
        freq_bonus = 0.7 + 0.3 * trade_factor

        # 7) 收益为负则再压一层
        profit_guard = 1.0 if profit_abs > 0 else 0.3

        score = calmar * freq_bonus * profit_guard * (0.5 + expectancy * 20.0)

        # IHyperOptLoss 约定:loss 越小越好,故取负
        return -score