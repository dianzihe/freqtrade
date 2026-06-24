# user_data 工作说明

这个目录用于本地下载行情数据、开发 Freqtrade 策略、保存回测/干跑结果和临时分析产物。后续 AI 在这里工作时，优先保持目录可复现、命名可检索、策略逻辑可回滚。

## 目录用途

- `data/`: 交易所行情数据。不要手工改动下载后的行情文件，重新下载或补齐数据时记录交易所、交易对、周期和时间范围。
- `strategies/`: 策略代码和策略辅助模块。策略类名用于 Freqtrade 配置和命令行参数，文件名只用于组织代码。
- `backtest_results/`, `hyperopt_results/`: 回测和参数优化输出。新结果应保留能识别策略、市场、周期和日期的文件名。
- `freqaimodels/`, `hyperopts/`, `notebooks/`, `plot/`, `plots/`: 研究、模型、图表和 notebook 产物。不要把一次性探索代码混进策略文件。
- `logs/`: 运行日志。排查问题时优先引用最新日志中的具体错误和命令。
- 根目录下的 `config-*.json`, `pairs-*.json`: 运行配置和交易对清单。修改前先确认目标交易所、运行模式和策略类名。

## AI 工作原则

1. 先看当前文件和配置，再改动。不要假设策略、交易所或周期仍是上一次任务的状态。
2. 改策略时只改和目标相关的策略文件或共享辅助模块，避免顺手重构无关策略。
3. 保留策略类名，除非明确要迁移配置中的 `"strategy"` 字段。
4. 回测、干跑、实盘配置要分开命名，避免同一配置同时承担多个用途。
5. 不提交 API key、secret、token、真实账号密码或私有代理凭据。
6. 生成的图表、HTML、JSON 报告要放在能看出用途的位置，并在最终回复里说明产物路径。
7. 如果发现用户已有未提交改动，先确认这些改动是否和当前任务有关，不要覆盖策略逻辑。

## strategies 命名规范

策略文件统一使用小写 `snake_case.py`：

- 单策略文件：`<domain>_<idea>_strategy.py`
- 多策略集合：`<domain>_<family>_strategies.py`
- 版本实验：`<domain>_<idea>_v03.py`, `v04`, ..., `v11`
- 辅助模块：`<domain>_core.py` 或 `<domain>_<purpose>.py`

策略类名继续使用 Freqtrade 习惯的 `PascalCase`，并保持和配置里的 `"strategy"` 字段一致。例如文件可以叫 `gate_scalp_momentum_strategy.py`，类名仍是 `GateScalpMomentumStrategy`。

当前策略文件分组：

- `chanlun_core.py`: 缠论信号和结构识别辅助模块。
- `chanlun_center_breakout_strategy.py`: 缠论中枢突破策略。
- `chanlun_filtered_strategies.py`: 缠论过滤型策略集合。
- `chanlun_second_buy_strategy.py`, `chanlun_second_buy_minimal.py`, `chanlun_second_buy_trailing.py`, `chanlun_second_buy_v03.py` 到 `chanlun_second_buy_v11.py`: 缠论二买策略实验线。
- `gate_scalp_momentum_strategy.py`: Gate 1m 剥头皮动量策略。
- `meme_martingale_strategies.py`: Meme 币马丁/反马丁策略集合。
- `manual_only_strategy.py`: 只用于手动/数据下载配置的空策略。
- `sample_strategy.py`: Freqtrade 示例策略。

## 常用操作检查

- 下载数据前：确认 `config-*-data-*.json`、`pairs-*.json`、交易所和周期。
- 新增策略前：先选定文件名、策略类名、配置文件名和回测结果命名。
- 修改策略后：至少运行对应的策略单元测试或一次小范围回测。
- 调整配置后：检查 `"strategy"`、`"exchange"`、`"pair_whitelist"`、`"timeframe"`、`"dry_run"` 和 `db_url`。
- 整理结果前：不要删除 sqlite、回测结果或日志，除非用户明确要求清理。

## 交付时说明

最终回复应说明：

- 改了哪些策略文件或配置文件。
- 策略类名是否变化。
- 跑了哪些测试、回测或静态检查。
- 如果没有运行验证，说明原因和建议的下一步命令。
