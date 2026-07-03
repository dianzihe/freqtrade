# Freqtrade 项目记忆

## 环境信息

- 项目路径：`E:\source\freqtrade`（已从 F 盘迁移）
- Python 版本：3.11.9（managed）
- 虚拟环境：`.venv/`（Windows 路径：`.venv\Scripts\activate`）
- 安装版本：freqtrade 2026.6-dev-2e730c2f7（开发分支）

## 安装状态

- 安装范围：核心依赖 + Hyperopt + FreqAI（LightGBM/XGBoost）+ 开发依赖（pytest/mypy/ruff/pre-commit）+ PyTorch + stable-baselines3
- 安装方式：editable 模式（`pip install -e .`）

## 常用命令

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 验证安装
.venv\Scripts\freqtrade --version

# 启动 dry-run（模拟交易）
.venv/Scripts/freqtrade trade --config user_data/config.json --strategy SampleStrategy --logfile user_data/logs/freqtrade.log

# 查看 API 状态
curl -s -u freqtrader:freqtrader123 http://127.0.0.1:8080/api/v1/status
```

## 网络注意事项（Windows 国内环境）

- 需要通过代理 `http://127.0.0.1:7890` 访问 Binance API
- 已卸载 `aiodns` + `pycares`（Windows 下 c-ares 无法联系 DNS 服务器）
- 配置中设置 `"enable_ws": false` 禁用 WebSocket（代理不支持）
- 代理配置在 `user_data/config.json` 的 `ccxt_config.proxies` 和 `ccxt_async_config.aiohttp_proxy`

## API Server

- 地址：`http://127.0.0.1:8080`
- 账号：freqtrader / freqtrader123
- 日志：`user_data/logs/freqtrade.log`

## 策略文件命名

- 策略文件已全部使用中文名（如 `缠论_中枢突破.py`、`趋势金字塔.py`），方便查看
- 类名保持英文（如 `ChanlunCenterBreakoutStrategy`），配置文件和脚本通过类名引用策略
- 3 个工具库文件也使用中文名：`缠论_核心模块.py`、`盘口_风险过滤器.py`、`Meme_马丁_基类.py`
- 缠论系列策略使用 try/except 双路径导入核心模块（全路径 + 短路径）
- Meme 系列策略使用短路径导入基类（`from Meme_马丁_基类 import ...`）
- 鲁棒Meme系列使用全路径导入（`from user_data.strategies.xxx import ...`）

## 微结构信号模块回测结论 (2026-06-29)

- **信号性质**: 微结构信号模块（arXiv:2604.20949）检测的是订单簿微观结构恶化状态，而非方向性信号
- **覆盖率**: 10分钟窗口 >=0.5% 波动的行情事件中，72.7% 被信号覆盖（60分钟回溯窗）
- **方向准确率**: 5分钟 48.1%，10分钟 51.9% — 接近随机，不预测涨跌方向
- **强度-收益相关性**: 极弱（r≈0.16-0.18），CH4 买压/卖压也无法区分后续走向
- **使用建议**: 信号适合作为波动率预警 / 风险过滤器，不推荐直接作为做多做空入场信号

## Gate L1/L2 订单簿数据质量评估 (2026-06-30)

- **数据路径**: `user_data/orderbook_data/gate/spot/{BTC_USDT,ETH_USDT}/{l1,l2}/`
- **存储格式**: Parquet + Hive 分区（date=YYYY-MM-DD/hour=HH/）
- **L1 Schema**: exchange_time_ms, local_time_ms, update_id, pair, bid_price, bid_amount, ask_price, ask_amount, spread, mid_price
- **L2 Schema**: exchange_time_ms, local_time_ms, first_update_id, update_id, pair, bid_updates(JSON), ask_updates(JSON), bid_update_count, ask_update_count
- **L2 格式**: 增量 diff（非全量快照），amount=0 表示价位撤销
- **覆盖时长**: 67 小时（2026-06-26 13:00 → 06-29 08:00 UTC），仅 BTC+ETH
- **采集质量**: 延迟中位数 2ms，无缺失值/交叉盘/异常价格，JSON 零错误
- **已知问题**: 末尾文件损坏（4个）、L2 仅 1 个快照（需全量回放重建）、BTC L1 有 1,535 个 ID 间隙
- **总评**: 3.9/5.0 — 采集质量优秀但覆盖时长不足、L2 快照缺失
- **报告文件**: `gate_orderbook_quality_report.md`

## Gate 订单簿数据格式化 (2026-07-01)

- **脚本**: `orderbook_formatter.py` — 将原始 L1+L2 Parquet 数据规整为统一格式
- **核心逻辑**: L2 快照+增量重建完整订单簿 → 轻量状态(每 tick) + 深度状态(每 200ms) → 1秒 OHLCV → 插针检测
- **输出**: `user_data/orderbook_data/formatted/{pair}/`
  - `orderbook_states.parquet` — 订单簿状态（含深度）
  - `ohlcv_1s.parquet` — 1秒 K线 + 深度特征
  - `wick_events.csv` — 插针/爆仓事件目录
  - `wick_depth_snapshots.parquet` — 插针前后深度快照
- **执行命令**: `.venv/Scripts/python -u orderbook_formatter.py --pair {BTC_USDT|ETH_USDT}`
