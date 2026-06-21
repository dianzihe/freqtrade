from __future__ import annotations

import argparse
import html
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "user_data" / "meme_dryrun_dashboard.html"
WALLET_SIZE = 100

DASHBOARD_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "Limited DCA",
        "strategy": "MemeLimitedDcaMartingaleStrategy",
        "config": "user_data/config-meme-limited-dca-dryrun.json",
        "db": "user_data/tradesv3.meme_limited_dca.sqlite",
        "port": 8091,
    },
    {
        "name": "Volatility Grid",
        "strategy": "MemeVolatilityGridMartingaleStrategy",
        "config": "user_data/config-meme-volatility-grid-dryrun.json",
        "db": "user_data/tradesv3.meme_volatility_grid.sqlite",
        "port": 8092,
    },
    {
        "name": "Anti Martingale",
        "strategy": "MemeAntiMartingaleTrendStrategy",
        "config": "user_data/config-meme-anti-martingale-dryrun.json",
        "db": "user_data/tradesv3.meme_anti_martingale.sqlite",
        "port": 8093,
    },
    {
        "name": "Hedge Proxy",
        "strategy": "MemeHedgeProxyMartingaleStrategy",
        "config": "user_data/config-meme-hedge-proxy-dryrun.json",
        "db": "user_data/tradesv3.meme_hedge_proxy.sqlite",
        "port": 8094,
    },
]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info({table})")}


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select name from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def trade_rows(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not table_exists(conn, "trades"):
            return []
        columns = table_columns(conn, "trades")
        wanted = [
            "id",
            "pair",
            "is_open",
            "open_date",
            "close_date",
            "stake_amount",
            "amount",
            "open_rate",
            "close_rate",
            "close_profit",
            "close_profit_abs",
            "enter_tag",
            "exit_reason",
        ]
        selected = [column for column in wanted if column in columns]
        order_column = "open_date" if "open_date" in columns else "id"
        rows = conn.execute(
            f"select {', '.join(selected)} from trades order by {order_column} desc"
        ).fetchall()
        return [dict(row) for row in rows]


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_strategy(item: dict[str, Any], wallet_size: float) -> dict[str, Any]:
    db_path = resolve_path(item["db"])
    rows = trade_rows(db_path)
    closed = [row for row in rows if not bool(row.get("is_open"))]
    open_rows = [row for row in rows if bool(row.get("is_open"))]
    profit_abs = sum(as_float(row.get("close_profit_abs")) for row in closed)
    wins = sum(1 for row in closed if as_float(row.get("close_profit_abs")) > 0)
    losses = sum(1 for row in closed if as_float(row.get("close_profit_abs")) < 0)
    winrate = wins / len(closed) * 100 if closed else 0.0
    equity = wallet_size + profit_abs

    return {
        **item,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "equity": equity,
        "profit_abs": profit_abs,
        "profit_pct": profit_abs / wallet_size * 100 if wallet_size else 0.0,
        "closed_trades": len(closed),
        "open_trades": len(open_rows),
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "recent_trades": rows[:8],
    }


def build_dashboard_model(
    strategies: list[dict[str, Any]] | None = None,
    wallet_size: float = WALLET_SIZE,
) -> dict[str, Any]:
    strategy_items = strategies if strategies is not None else DASHBOARD_STRATEGIES
    summaries = [summarize_strategy(item, wallet_size) for item in strategy_items]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wallet_size": wallet_size,
        "strategies": summaries,
    }


def fmt_usdt(value: float) -> str:
    return f"{value:,.2f} U"


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def css_class_for_profit(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "muted"


def render_recent_trades(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">暂无交易记录，等待模拟盘产生信号。</div>'
    rendered = []
    for row in rows:
        pair = html.escape(str(row.get("pair", "-")))
        state = "持仓中" if bool(row.get("is_open")) else "已平仓"
        profit_abs = as_float(row.get("close_profit_abs"))
        profit_pct = as_float(row.get("close_profit")) * 100
        close_info = fmt_usdt(profit_abs) if not bool(row.get("is_open")) else "浮动中"
        rendered.append(
            "<tr>"
            f"<td>{pair}</td>"
            f"<td>{state}</td>"
            f"<td>{html.escape(str(row.get('open_date', '-')))}</td>"
            f"<td class=\"{css_class_for_profit(profit_abs)}\">{close_info}</td>"
            f"<td class=\"{css_class_for_profit(profit_abs)}\">{fmt_pct(profit_pct)}</td>"
            f"<td>{html.escape(str(row.get('exit_reason') or row.get('enter_tag') or '-'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>交易对</th><th>状态</th><th>开仓时间</th>"
        "<th>收益</th><th>收益率</th><th>标签</th></tr></thead><tbody>"
        + "".join(rendered)
        + "</tbody></table>"
    )


def render_dashboard_html(model: dict[str, Any]) -> str:
    payload = json.dumps(model, ensure_ascii=False)
    cards = []
    for item in model["strategies"]:
        profit_class = css_class_for_profit(item["profit_abs"])
        status = "运行数据已连接" if item["db_exists"] else "等待 dry-run 启动"
        cards.append(
            f"""
            <section class="strategy-card">
                <div class="card-head">
                    <div>
                        <h2>{html.escape(item["name"])}</h2>
                        <p>{html.escape(item["strategy"])}</p>
                    </div>
                    <span class="status">{status}</span>
                </div>
                <div class="metric-grid">
                    <div><span>权益</span><strong>{fmt_usdt(item["equity"])}</strong></div>
                    <div><span>收益</span><strong class="{profit_class}">{fmt_usdt(item["profit_abs"])}</strong></div>
                    <div><span>收益率</span><strong class="{profit_class}">{fmt_pct(item["profit_pct"])}</strong></div>
                    <div><span>已平仓</span><strong>{item["closed_trades"]}</strong></div>
                    <div><span>持仓中</span><strong>{item["open_trades"]}</strong></div>
                    <div><span>胜率</span><strong>{item["winrate"]:.1f}%</strong></div>
                </div>
                <div class="mini-meta">
                    <span>API: 127.0.0.1:{item["port"]}</span>
                    <span>DB: {html.escape(Path(item["db"]).name)}</span>
                </div>
                {render_recent_trades(item["recent_trades"])}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <title>妖币马丁模拟盘看板</title>
    <style>
        :root {{
            --bg: #0f1216;
            --panel: #181d23;
            --panel-2: #202730;
            --line: #303944;
            --text: #edf2f7;
            --muted: #9aa8b6;
            --green: #45d483;
            --red: #ff6b6b;
            --gold: #f4c95d;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: "Segoe UI", Arial, sans-serif;
        }}
        main {{
            width: min(1440px, calc(100% - 32px));
            margin: 0 auto;
            padding: 24px 0 32px;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 16px;
            margin-bottom: 18px;
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 0;
        }}
        header p {{
            margin: 7px 0 0;
            color: var(--muted);
            font-size: 14px;
        }}
        .summary-strip {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 14px;
        }}
        .summary-strip div {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
        }}
        .summary-strip span, .metric-grid span {{
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 6px;
        }}
        .summary-strip strong, .metric-grid strong {{
            font-size: 20px;
        }}
        .strategy-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}
        .strategy-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            min-width: 0;
        }}
        .card-head {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: start;
            margin-bottom: 14px;
        }}
        .card-head h2 {{
            margin: 0;
            font-size: 18px;
            line-height: 1.25;
        }}
        .card-head p, .mini-meta {{
            color: var(--muted);
            font-size: 12px;
        }}
        .card-head p {{ margin: 5px 0 0; overflow-wrap: anywhere; }}
        .status {{
            flex: 0 0 auto;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 5px 9px;
            color: var(--gold);
            font-size: 12px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 12px;
        }}
        .metric-grid div {{
            background: var(--panel-2);
            border-radius: 8px;
            padding: 10px;
            min-width: 0;
        }}
        .positive {{ color: var(--green); }}
        .negative {{ color: var(--red); }}
        .muted {{ color: var(--muted); }}
        .mini-meta {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
            overflow-wrap: anywhere;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th, td {{
            border-top: 1px solid var(--line);
            padding: 8px 6px;
            text-align: left;
            vertical-align: top;
        }}
        th {{ color: var(--muted); font-weight: 600; }}
        .empty {{
            border-top: 1px solid var(--line);
            color: var(--muted);
            padding: 14px 2px 2px;
            font-size: 13px;
        }}
        @media (max-width: 900px) {{
            header {{ align-items: start; flex-direction: column; }}
            .summary-strip, .strategy-grid {{ grid-template-columns: 1fr; }}
            .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
    </style>
</head>
<body>
    <main>
        <header>
            <div>
                <h1>妖币马丁模拟盘看板</h1>
                <p>4 个策略各 100U dry-run，页面每 30 秒自动刷新。</p>
            </div>
            <p>生成时间：{html.escape(model["generated_at"])}</p>
        </header>
        <section class="summary-strip">
            <div><span>策略数量</span><strong>{len(model["strategies"])}</strong></div>
            <div><span>单策略资金</span><strong>{fmt_usdt(model["wallet_size"])}</strong></div>
            <div><span>合计权益</span><strong>{fmt_usdt(sum(i["equity"] for i in model["strategies"]))}</strong></div>
            <div><span>合计收益</span><strong class="{css_class_for_profit(sum(i["profit_abs"] for i in model["strategies"]))}">{fmt_usdt(sum(i["profit_abs"] for i in model["strategies"]))}</strong></div>
        </section>
        <section class="strategy-grid">
            {''.join(cards)}
        </section>
    </main>
    <script type="application/json" id="dashboard-data">{html.escape(payload)}</script>
</body>
</html>
"""


def write_dashboard(output: Path = DEFAULT_OUTPUT) -> Path:
    model = build_dashboard_model()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard_html(model), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = write_dashboard(Path(args.output))
    print(output)


if __name__ == "__main__":
    main()
