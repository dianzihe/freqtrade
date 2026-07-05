"""
Meme限制定投 Spot模式包装器
原始策略继承自 can_short=True 的基类，spot 模式回测时忽略 short 信号
"""
from Meme_限制定投_马丁 import MemeLimitedMartingaleStrategy


class MemeLimitedMartingaleSpotStrategy(MemeLimitedMartingaleStrategy):
    can_short = False
