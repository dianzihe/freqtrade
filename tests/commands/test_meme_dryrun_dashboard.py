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
    seen_keys = set()

    for item in DASHBOARD_STRATEGIES:
        config_path = Path(item["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))

        assert config["strategy"] == item["strategy"]
        assert config["timeframe"] == item["timeframe"]
        assert config["dry_run"] is True
        assert config["dry_run_wallet"] == 100
        assert config["db_url"].startswith("sqlite:///user_data/sqlite/")
        assert config["db_url"] not in seen_dbs
        assert config["api_server"]["listen_port"] not in seen_ports
        assert config["api_server"]["enabled"] is True
        assert config["exchange"]["name"] == "gate"
        assert config["exchange"]["pair_whitelist"]

        seen_keys.add((item["strategy"], item["timeframe"]))
        seen_dbs.add(config["db_url"])
        seen_ports.add(config["api_server"]["listen_port"])

    assert len(DASHBOARD_STRATEGIES) == 10
    assert {item["timeframe"] for item in DASHBOARD_STRATEGIES} == {"1m", "5m"}
    assert len(seen_keys) == 10


def test_dashboard_renders_ten_strategy_cards_grouped_by_timeframe(tmp_path: Path) -> None:
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
            "name": "Meme限制定投_马丁",
            "strategy": "MemeLimitedDcaMartingaleStrategy",
            "timeframe": "1m",
            "config": "unused.json",
            "db": str(db_path),
            "port": 8091,
        },
        {
            "name": "Meme限制定投_马丁",
            "strategy": "MemeLimitedDcaMartingaleStrategy",
            "timeframe": "5m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-limited-5m.sqlite"),
            "port": 8092,
        },
        {
            "name": "Meme_波动率网格_马丁",
            "strategy": "MemeVolatilityGridMartingaleStrategy",
            "timeframe": "1m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-grid.sqlite"),
            "port": 8093,
        },
        {
            "name": "Meme_波动率网格_马丁",
            "strategy": "MemeVolatilityGridMartingaleStrategy",
            "timeframe": "5m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-grid-5m.sqlite"),
            "port": 8094,
        },
        {
            "name": "gate_xinghe_futures_grid_strategy",
            "strategy": "GateXingheFuturesGridStrategy",
            "timeframe": "1m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-xinghe-1m.sqlite"),
            "port": 8095,
        },
        {
            "name": "gate_xinghe_futures_grid_strategy",
            "strategy": "GateXingheFuturesGridStrategy",
            "timeframe": "5m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-xinghe-5m.sqlite"),
            "port": 8096,
        },
        {
            "name": "顶部反转做空策略",
            "strategy": "TopReversalShortStrategy",
            "timeframe": "1m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-top-1m.sqlite"),
            "port": 8097,
        },
        {
            "name": "顶部反转做空策略",
            "strategy": "TopReversalShortStrategy",
            "timeframe": "5m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-top-5m.sqlite"),
            "port": 8098,
        },
        {
            "name": "趋势金字塔_反马丁",
            "strategy": "TrendPyramidAntiMartingaleStrategy",
            "timeframe": "1m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-trend-1m.sqlite"),
            "port": 8099,
        },
        {
            "name": "趋势金字塔_反马丁",
            "strategy": "TrendPyramidAntiMartingaleStrategy",
            "timeframe": "5m",
            "config": "unused.json",
            "db": str(tmp_path / "missing-trend-5m.sqlite"),
            "port": 8100,
        },
    ]

    model = build_dashboard_model(strategies=strategies, wallet_size=100)
    html = render_dashboard_html(model)

    assert len(model["strategies"]) == 10
    assert model["strategies"][0]["profit_abs"] == 1.0
    assert model["strategies"][0]["equity"] == 101.0
    assert html.count('class="strategy-card"') == 10
    assert 'http-equiv="refresh" content="60"' in html
    assert "1m 周期" in html
    assert "5m 周期" in html
    assert "Meme限制定投_马丁" in html
    assert "顶部反转做空策略" in html
