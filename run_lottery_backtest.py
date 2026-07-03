"""
Custom backtest runner that skips exchange market loading (which requires network access).
This is needed for offline backtesting when the exchange API is unreachable.
"""
import sys
import json
import logging
from pathlib import Path
from unittest.mock import patch

# Patch Exchange to skip market loading BEFORE importing freqtrade
import freqtrade.exchange.exchange as exchange_mod


def patched_reload_markets(self, reload=False, load_leverage_tiers=False):
    """Skip market loading for offline backtesting. Populate dummy markets from config."""
    logging.getLogger('freqtrade').info('Skipping exchange market loading (offline backtest mode)')
    # Build dummy markets dict from config whitelist
    pairs = self._config.get('exchange', {}).get('pair_whitelist', [])
    markets = {}
    for pair in pairs:
        parts = pair.split('/')
        base = parts[0] if len(parts) > 0 else ''
        quote = parts[1] if len(parts) > 1 else ''
        markets[pair] = {
            'symbol': pair,
            'base': base,
            'quote': quote,
            'spot': True,
            'swap': False,
            'future': False,
            'active': True,
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'amount': {'min': 0.0001, 'max': 1000000},
                'price': {'min': 0.00000001, 'max': 1000000},
                'cost': {'min': None, 'max': None},
            },
            'contractSize': None,
            'contract': False,
            'linear': None,
            'inverse': None,
        }
    self._markets = markets

def patched_validate_config(self, config):
    """Skip market-dependent validation, only validate basic config."""
    self.validate_timeframes(config.get("timeframe"))
    # Skip validate_stakecurrency (needs markets)
    self.validate_ordertypes(config.get("order_types", {}))
    self.validate_order_time_in_force(config.get("order_time_in_force", {}))
    self.validate_trading_mode_and_margin_mode(self.trading_mode, self.margin_mode)
    self.validate_pricing(config["exit_pricing"])
    self.validate_pricing(config["entry_pricing"])
    self.validate_orderflow(config["exchange"])
    self.validate_demo_trading(config["exchange"])
    self.validate_freqai(config)
    self._set_startup_candle_count(config)

exchange_mod.Exchange.reload_markets = patched_reload_markets
exchange_mod.Exchange.validate_config = patched_validate_config

# Also patch markets property to return a dummy value
exchange_mod.Exchange._markets = {}

# Now run freqtrade backtesting
from freqtrade.main import main

if __name__ == '__main__':
    args = [
        'backtesting',
        '--config', 'user_data/config_lottery_gate.json',
        '--strategy', 'LotteryTicketMomentumV3',
        '--strategy-path', 'user_data/strategies',
        '--timerange', '20260624-20260702',
        '--export', 'trades',
    ]
    sys.argv = ['freqtrade'] + args
    raise SystemExit(main(sys.argv[1:]))
