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
