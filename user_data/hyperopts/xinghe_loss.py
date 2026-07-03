# -*- coding: utf-8 -*-
"""
星河网格策略专用 Hyperopt 损失函数。

设计哲学(为什么这样写):
  本策略最大的死法是"逆势 DCA 在单边行情爆仓"。普通的 SharpeHyperOptLoss
  只优化收益/波动比, 不会重罚"深度回撤"和"连续亏损", 容易优化出一个回测曲线
  漂亮、但实盘一遇单边就归零的参数组。

  因此本损失函数 = 风险调整收益, 并对以下三项加惩罚:
    1. 最大回撤(超过目标即指数级惩罚) —— 对应策略的 30% 熔断红线;
    2. 盈利因子过低(profit_factor < 1.1) —— 过滤"靠运气"的组合;
    3. 平均持仓时间过长 —— 资金效率, 也间接惩罚"长期扛单"的逆势摊平。

  loss 越小越好, 所以收益取负、惩罚取正相加。
"""

from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


# 目标最大回撤红线(与策略 cb_daily_drawdown 呼应)。超过则重罚。
TARGET_MAX_DRAWDOWN = 0.30
# 最小可接受盈利因子。
MIN_PROFIT_FACTOR = 1.1


class XingheGridHyperOptLoss(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: dict,
        processed: dict,
        backtest_stats: dict[str, Any],
        *args,
        **kwargs,
    ) -> float:

        # 交易太少 → 统计不可信, 直接给大 loss 劝退。
        if trade_count < 30:
            return 1000.0

        # --- 1) 基础收益(总收益率) ---
        total_profit = results["profit_ratio"].sum()

        # --- 2) 风险调整: 用收益/收益标准差(类 Sharpe) ---
        profit_std = results["profit_ratio"].std()
        if profit_std == 0 or profit_std != profit_std:  # NaN 防护
            risk_adjusted = total_profit
        else:
            expected = results["profit_ratio"].mean()
            risk_adjusted = (expected / profit_std) * (trade_count ** 0.5)

        # --- 3) 最大回撤惩罚 ---
        # 优先取 backtest_stats 里算好的 max_drawdown_account, 没有就退化估算。
        max_dd = 0.0
        try:
            max_dd = abs(backtest_stats.get("max_drawdown_account", 0.0))
        except Exception:
            pass
        if max_dd == 0.0:
            # 退化: 用累计收益曲线的回撤近似
            cum = results["profit_ratio"].cumsum()
            running_max = cum.cummax()
            max_dd = abs((cum - running_max).min())

        # 超过红线指数惩罚: 在红线内温和, 越界后急剧拉大 loss。
        if max_dd <= TARGET_MAX_DRAWDOWN:
            dd_penalty = max_dd * 2.0
        else:
            dd_penalty = (max_dd / TARGET_MAX_DRAWDOWN) ** 3  # 立方惩罚, 严厉劝退爆仓组合

        # --- 4) 盈利因子惩罚 ---
        wins = results.loc[results["profit_ratio"] > 0, "profit_abs"].sum()
        losses = abs(results.loc[results["profit_ratio"] < 0, "profit_abs"].sum())
        profit_factor = wins / losses if losses > 0 else (wins if wins > 0 else 0.0)
        pf_penalty = 0.0 if profit_factor >= MIN_PROFIT_FACTOR else (MIN_PROFIT_FACTOR - profit_factor)

        # --- 5) 持仓时长惩罚(资金效率 + 抑制长期扛单) ---
        try:
            avg_dur_min = results["trade_duration"].mean()  # 分钟
            # 超过 24 小时(1440min)开始温和惩罚
            dur_penalty = max(0.0, (avg_dur_min - 1440) / 1440) * 0.5
        except Exception:
            dur_penalty = 0.0

        # --- 汇总: 收益取负(越赚 loss 越小) + 各惩罚相加 ---
        loss = -risk_adjusted - total_profit + (dd_penalty * 5.0) + (pf_penalty * 3.0) + dur_penalty
        return loss