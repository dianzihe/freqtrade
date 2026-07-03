import sys
from datetime import datetime, timezone
from pathlib import Path

STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(STRATEGIES_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGIES_DIR))

from gate_oco import GateOCOFailsafeMixin  # noqa: E402


class FakeExchange:
    def __init__(self, fail_on_call: int | None = None):
        self.created = []
        self.canceled = []
        self.fail_on_call = fail_on_call

    def create_order(self, symbol, order_type, side, amount, price, params):
        if self.fail_on_call == len(self.created) + 1:
            raise RuntimeError("create failed")
        order_id = f"order-{len(self.created) + 1}"
        self.created.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )
        return {"id": order_id}

    def cancel_order(self, order_id, symbol, params=None):
        self.canceled.append((order_id, symbol, params or {}))


class FakeTrade:
    id = 42
    pair = "DOGE/USDT:USDT"
    open_rate = 100.0
    amount = 2.5
    is_short = False
    entry_side = "buy"
    open_date_utc = datetime(2026, 6, 1, tzinfo=timezone.utc)

    def __init__(self):
        self._custom_data = {}

    def set_custom_data(self, key, value):
        self._custom_data[key] = value

    def get_custom_data(self, key):
        return self._custom_data.get(key)


class BaseStrategy:
    stoploss = -0.2

    def custom_exit(self, *args, **kwargs):
        return "base_exit"


class OcoStrategy(GateOCOFailsafeMixin, BaseStrategy):
    def __init__(self, exchange):
        self.exchange = exchange

    def _oco_exchange(self):
        return self.exchange


def test_place_bracket_creates_take_profit_and_stop_loss_orders() -> None:
    exchange = FakeExchange()
    strategy = OcoStrategy(exchange)
    trade = FakeTrade()

    strategy.place_bracket(trade)

    assert trade.get_custom_data("oco_orders") == {"tp": "order-1", "sl": "order-2"}
    assert exchange.created[0]["params"] == {
        "reduceOnly": True,
        "takeProfitPrice": 106.0,
        "clientOrderId": "t-oco_42_tp",
    }
    assert exchange.created[1]["params"] == {
        "reduceOnly": True,
        "stopLossPrice": 77.0,
        "clientOrderId": "t-oco_42_sl",
    }


def test_place_bracket_cancels_take_profit_when_stop_loss_creation_fails() -> None:
    exchange = FakeExchange(fail_on_call=2)
    strategy = OcoStrategy(exchange)
    trade = FakeTrade()

    strategy.place_bracket(trade)

    assert exchange.canceled == [("order-1", "DOGE/USDT:USDT", {"stop": True})]
    assert trade.get_custom_data("oco_orders") == {}


def test_custom_exit_consumes_oco_trigger_before_delegating_to_base_exit() -> None:
    strategy = OcoStrategy(FakeExchange())
    trade = FakeTrade()
    trade.set_custom_data("oco_triggered", "sl")

    result = strategy.custom_exit(
        "DOGE/USDT:USDT",
        trade,
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        80.0,
        -0.2,
    )

    assert result == "oco_exchange_sl"
    assert trade.get_custom_data("oco_triggered") is None


def test_volatility_grid_strategy_uses_gate_oco_mixin() -> None:
    from user_data.strategies.Meme_波动率网格_马丁 import (
        MemeVolatilityGridMartingaleStrategy,
    )

    assert issubclass(MemeVolatilityGridMartingaleStrategy, GateOCOFailsafeMixin)
