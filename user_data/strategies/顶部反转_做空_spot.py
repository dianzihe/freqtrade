"""
顶部反转做空 Spot模式包装器
原始策略为纯做空策略，spot 模式下无 short 信号可用，预期 0 交易
"""
from 顶部反转_做空 import TopReversalShortStrategy


class TopReversalShortSpotStrategy(TopReversalShortStrategy):
    can_short = False
