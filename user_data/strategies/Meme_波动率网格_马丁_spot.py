"""
Meme波动率网格 Spot模式包装器
原始策略需要 futures 模式（含做空），无资金费率/标记价数据时用 spot 模式回测
注：short 信号在 spot 模式下会被忽略，仅保留 long 信号
"""
from Meme_波动率网格_马丁 import MemeVolatilityGridMartingaleStrategy


class MemeVolatilityGridMartingaleSpotStrategy(MemeVolatilityGridMartingaleStrategy):
    can_short = False
