import json
import sqlite3
from pathlib import Path

from scripts.meme_dryrun_dashboard import (
    DASHBOARD_STRATEGIES,
    build_dashboard_model,
    render_dashboard_html,
)


def test_meme_dryrun_configs_are_isolated_and_funded() -> None:
    seen_dbs = set()
    seen_ports = set()

    for item in DASHBOARD_STRATEGIES:
        config_path = Path(item["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))

        assert config["strategy"] == item["strategy"]
        assert config["dry_run"] is True
        assert config["dry_run_wallet"] == 100
        assert config["db_url"].startswith("sqlite:///")
        assert config["db_url"] not in seen_dbs
        assert config["api_server"]["listen_port"] not in seen_ports
        assert config["exchange"]["pair_whitelist"] == [
            "H/USDT",
            "VELVET/USDT",
            "DN/USDT",
            "BEAT/USDT",
        ]

        seen_dbs.add(config["db_url"])
        seen_ports.add(config["api_server"]["listen_port"])


def test_dashboard_renders_four_strategy_cards(tmp_path: Path) -> None:
    db_path = tmp_path / "limited.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table trades (
                id integer primary key,
                pair text,
                is_open integer,
                open_date text,
                close_date text,
                stake_amount real,
                amount real,
                open_rate real,
                close_rate real,
                close_profit real,
                close_profit_abs real,
                enter_tag text,
                exit_reason text
            )
            """
        )
        conn.execute(
            """
            insert into trades (
                pair, is_open, open_date, close_date, stake_amount, amount,
                open_rate, close_rate, close_profit, close_profit_abs, enter_tag, exit_reason
            )
            values ('H/USDT', 0, '2026-06-21 00:00:00', '2026-06-21 00:30:00',
                    20, 2, 10, 10.5, 0.05, 1.0, 'entry', 'roi')
            """
        )

    strategies = [
        {
            "name": "Limited DCA",
            "strategy": "MemeLimitedDcaMartingaleStrategy",
            "config": "unused.json",
            "db": str(db_path),
            "port": 8091,
        },
        {
            "name": "Volatility Grid",
            "strategy": "MemeVolatilityGridMartingaleStrategy",
            "config": "unused.json",
            "db": str(tmp_path / "missing-grid.sqlite"),
            "port": 8092,
        },
        {
            "name": "Anti Martingale",
            "strategy": "MemeAntiMartingaleTrendStrategy",
            "config": "unused.json",
            "db": str(tmp_path / "missing-anti.sqlite"),
            "port": 8093,
        },
        {
            "name": "Hedge Proxy",
            "strategy": "MemeHedgeProxyMartingaleStrategy",
            "config": "unused.json",
            "db": str(tmp_path / "missing-hedge.sqlite"),
            "port": 8094,
        },
    ]

    model = build_dashboard_model(strategies=strategies, wallet_size=100)
    html = render_dashboard_html(model)

    assert len(model["strategies"]) == 4
    assert model["strategies"][0]["profit_abs"] == 1.0
    assert model["strategies"][0]["equity"] == 101.0
    assert html.count('class="strategy-card"') == 4
    assert "Limited DCA" in html
    assert "Volatility Grid" in html
