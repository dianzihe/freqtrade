from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from scripts.meme_dryrun_dashboard import DASHBOARD_STRATEGIES, PROJECT_ROOT, resolve_path


def _import_strategy_class(class_name: str) -> type | None:
    """Dynamically import a strategy class by name from user_data/strategies/."""
    strategies_dir = PROJECT_ROOT_PATH / "user_data" / "strategies"
    for py_file in sorted(strategies_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_dryrun_cfg_{py_file.stem}", str(py_file)
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except Exception:
            continue
    return None


def _merge_order_types(strategy_class: type | None) -> dict[str, Any]:
    """Return merged order_types: strategy defaults + safe config overrides."""
    defaults: dict[str, Any] = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    if strategy_class is not None and hasattr(strategy_class, "order_types"):
        strategy_orders = strategy_class.order_types
        # Strategy-defined keys take precedence over defaults
        defaults.update(strategy_orders)
    return defaults


def build_config(item: dict[str, Any]) -> dict[str, Any]:
    is_futures = item["mode"] == "futures"
    strategy_cls = _import_strategy_class(item["strategy"])

    config: dict[str, Any] = {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "strategy": item["strategy"],
        "db_url": f"sqlite:///{item['db']}",
        "max_open_trades": item["max_open_trades"],
        "stake_currency": "USDT",
        "stake_amount": 10,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "CNY",
        "dry_run": True,
        "dry_run_wallet": 100,
        "trading_mode": "futures" if is_futures else "spot",
        "timeframe": item["timeframe"],
        "dataformat_ohlcv": "feather",
        "position_adjustment_enable": True,
        "cancel_open_orders_on_exit": False,
        "use_exit_signal": True,
        "exit_profit_only": False,
        "ignore_roi_if_entry_signal": False,
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "unit": "minutes",
        },
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1,
            },
        },
        "exit_pricing": {
            "price_side": "other" if _merge_order_types(strategy_cls).get("exit") == "market" else "same",
            "use_order_book": False,
            "order_book_top": 1,
        },
        "order_types": _merge_order_types(strategy_cls),
        "order_time_in_force": {
            "entry": "GTC",
            "exit": "GTC",
        },
        "exchange": {
            "name": "gate",
            "key": "",
            "secret": "",
            "ccxt_config": {
                "proxies": {
                    "http": "http://127.0.0.1:7890",
                    "https": "http://127.0.0.1:7890",
                },
            },
            "ccxt_async_config": {
                "aiohttp_proxy": "http://127.0.0.1:7890",
            },
            "enable_ws": False,
            "pair_whitelist": deepcopy(item["pairs"]),
            "pair_blacklist": [
                ".*3L/USDT",
                ".*3S/USDT",
                ".*5L/USDT",
                ".*5S/USDT",
            ],
        },
        "pairlists": [
            {
                "method": "StaticPairList",
                "allow_inactive": True,
            },
            {
                "method": "GateMemeVolatilityPairList",
                "core_pairs": deepcopy(item.get("pairlist_core", [])),
                "number_assets": item.get("pairlist_count", 5),
                "refresh_period": 3600,
            },
        ],
        "telegram": {
            "enabled": False,
            "token": "",
            "chat_id": "",
        },
        "api_server": {
            "enabled": True,
            "listen_ip_address": "127.0.0.1",
            "listen_port": item["port"],
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": f"{item['slug']}-dryrun-secret-20260628",
            "CORS_origins": [],
            "username": item["slug"],
            "password": f"{item['slug']}-dryrun",
        },
        "bot_name": f"dryrun-{item['slug']}",
        "initial_state": "running",
        "force_entry_enable": False,
        "internals": {
            "process_throttle_secs": 5,
        },
    }
    if is_futures:
        config["margin_mode"] = "isolated"
    else:
        config["exchange"]["pair_blacklist"].append(".*/USDT:USDT")
    return config


def write_configs() -> list[Path]:
    output_paths = []
    (PROJECT_ROOT / "user_data" / "config").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "user_data" / "sqlite").mkdir(parents=True, exist_ok=True)
    for item in DASHBOARD_STRATEGIES:
        path = resolve_path(item["config"])
        path.write_text(
            json.dumps(build_config(item), ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        output_paths.append(path)
    return output_paths


def main() -> None:
    for path in write_configs():
        print(path)


if __name__ == "__main__":
    main()
