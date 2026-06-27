"""
生成金麒麟策略的多周期版本（1m, 5m, 15m, 1h）
用法: python scripts/gen_multi_tf_strategies.py
"""
import re
import sys
from pathlib import Path

STRATEGY_SRC = Path("user_data/strategies/金麒麟.py")
OUTPUT_DIR = Path("user_data/strategies")

TIME_FRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
}

def generate_strategy(tf_name: str, tf_value: str):
    """为指定时间周期生成策略文件"""
    content = STRATEGY_SRC.read_text(encoding="utf-8")

    # 替换类名
    content = re.sub(
        r"class\s+OptimizedGoldStrategy",
        f"class OptimizedGoldStrategy_{tf_name}",
        content
    )

    # 替换 timeframe
    content = re.sub(
        r"timeframe\s*=\s*'[^']*'",
        f"timeframe = '{tf_value}'",
        content
    )

    # 更新文档字符串
    content = content.replace(
        "设计为 H4 趋势跟随",
        f"设计为 {tf_value} 趋势跟随（多周期回测版）"
    )
    content = content.replace(
        "原策略是黄金H4单边趋势思路，时间框架保持4h",
        f"原策略是黄金H4单边趋势思路，时间框架改为{tf_value}（多周期回测版）"
    )

    out_path = OUTPUT_DIR / f"金麒麟_{tf_name}.py"
    out_path.write_text(content, encoding="utf-8")
    print(f"[OK] 生成: {out_path.name}")
    return out_path.name


def main():
    if not STRATEGY_SRC.exists():
        print(f"错误: 找不到策略文件 {STRATEGY_SRC}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for tf_name, tf_value in TIME_FRAMES.items():
        name = generate_strategy(tf_name, tf_value)
        generated.append(name)

    print(f"\n共生成 {len(generated)} 个策略文件:")
    for n in generated:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
