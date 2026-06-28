from __future__ import annotations

import argparse
import base64
import html
import json
import sqlite3
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "user_data" / "meme_dryrun_dashboard.html"
WALLET_SIZE = 100
REFRESH_SECONDS = 60
API_TIMEOUT = 5  # seconds

# 核心币种 — 永远固定保留
CORE_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

# 妖币候选池 — 每小时按 24h 涨跌幅绝对值排名，取前 N 个
MEME_CANDIDATES = [
    "H/USDT",
    "VELVET/USDT",
    "DN/USDT",
    "BEAT/USDT",
    "COAI/USDT",
    "ALLO/USDT",
    "STG/USDT",
    "XCN/USDT",
    "IP/USDT",
    "BAS/USDT",
    "PEAQ/USDT",
    "TA/USDT",
    "LTC/USDT",
    "HYPE/USDT",
]

# 完整白名单（用于 StaticPairList）— 核心 + 全部妖币候选
SPOT_WHITELIST = CORE_PAIRS + [p for p in MEME_CANDIDATES if p not in CORE_PAIRS]

# 合约后缀
FUTURES_WHITELIST = [p.replace("/USDT", "/USDT:USDT") for p in SPOT_WHITELIST]
CORE_PAIRS_FUTURES = [p.replace("/USDT", "/USDT:USDT") for p in CORE_PAIRS]

DASHBOARD_STRATEGIES: list[dict[str, Any]] = []

_MATRIX = [
    {
        "name": "Meme限制定投_马丁",
        "strategy": "MemeLimitedDcaMartingaleStrategy",
        "slug": "meme_limited_dca",
        "mode": "spot",
        "pairs": SPOT_WHITELIST,
        "pairlist_core": CORE_PAIRS,
        "pairlist_count": 5,
        "max_open_trades": 4,
    },
    {
        "name": "Meme_波动率网格_马丁",
        "strategy": "MemeVolatilityGridMartingaleStrategy",
        "slug": "meme_volatility_grid",
        "mode": "spot",
        "pairs": SPOT_WHITELIST,
        "pairlist_core": CORE_PAIRS,
        "pairlist_count": 5,
        "max_open_trades": 4,
    },
    {
        "name": "gate_xinghe_futures_grid_strategy",
        "strategy": "GateXingheFuturesGridStrategy",
        "slug": "gate_xinghe_futures_grid",
        "mode": "futures",
        "pairs": FUTURES_WHITELIST,
        "pairlist_core": CORE_PAIRS_FUTURES,
        "pairlist_count": 5,
        "max_open_trades": 4,
    },
    {
        "name": "顶部反转做空策略",
        "strategy": "TopReversalShortStrategy",
        "slug": "top_reversal_short",
        "mode": "futures",
        "pairs": FUTURES_WHITELIST,
        "pairlist_core": CORE_PAIRS_FUTURES,
        "pairlist_count": 5,
        "max_open_trades": 5,
    },
    {
        "name": "趋势金字塔_反马丁",
        "strategy": "TrendPyramidAntiMartingaleStrategy",
        "slug": "trend_pyramid_anti_martingale",
        "mode": "spot",
        "pairs": SPOT_WHITELIST,
        "pairlist_core": CORE_PAIRS,
        "pairlist_count": 5,
        "max_open_trades": 5,
    },
]

for base_index, item in enumerate(_MATRIX):
    for tf_index, timeframe in enumerate(("1m", "5m")):
        port = 8091 + base_index * 2 + tf_index
        slug = f"{item['slug']}_{timeframe}"
        DASHBOARD_STRATEGIES.append(
            {
                **item,
                "timeframe": timeframe,
                "slug": slug,
                "config": f"user_data/config/config-dryrun-{slug}.json",
                "db": f"user_data/sqlite/tradesv3.dryrun_{slug}.sqlite",
                "port": port,
            }
        )


# ── helpers ──────────────────────────────────────────────────────────

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
        if not selected:
            return []
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


# ── API helpers ───────────────────────────────────────────────────────

def _api_auth(slug: str) -> str:
    """Build Basic auth header value for a strategy's API server."""
    creds = base64.b64encode(f"{slug}:{slug}-dryrun".encode()).decode()
    return f"Basic {creds}"


def _api_get(port: int, slug: str, endpoint: str) -> dict[str, Any] | None:
    """Call a strategy's REST API and return parsed JSON, or None on failure."""
    try:
        url = f"http://127.0.0.1:{port}/api/v1/{endpoint}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", _api_auth(slug))
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read())  # type: ignore[no-any-return]
    except Exception:
        return None


def fetch_active_pairs(port: int, slug: str) -> list[str]:
    """Return the current whitelist (active trading pairs) from the API."""
    data = _api_get(port, slug, "whitelist")
    if data and "whitelist" in data:
        return list(data["whitelist"])
    return []


def fetch_total_profit(port: int, slug: str) -> float | None:
    """Return profit_all_coin (realized + unrealized) from the API."""
    data = _api_get(port, slug, "profit")
    if data and "profit_all_coin" in data:
        return as_float(data["profit_all_coin"])
    return None


# ── strategy summary ──────────────────────────────────────────────────

def summarize_strategy(item: dict[str, Any], wallet_size: float) -> dict[str, Any]:
    db_path = resolve_path(item["db"])
    rows = trade_rows(db_path)
    closed = [row for row in rows if not bool(row.get("is_open"))]
    open_rows = [row for row in rows if bool(row.get("is_open"))]

    # Realized P&L from DB
    realized = sum(as_float(row.get("close_profit_abs")) for row in closed)
    wins = sum(1 for row in closed if as_float(row.get("close_profit_abs")) > 0)
    losses = sum(1 for row in closed if as_float(row.get("close_profit_abs")) < 0)
    winrate = wins / len(closed) * 100 if closed else 0.0

    # Unrealized P&L from API (real-time)
    port = item.get("port", 0)
    slug = item.get("slug", "")
    api_total = fetch_total_profit(port, slug)
    unrealized = 0.0
    api_available = False
    if api_total is not None:
        unrealized = api_total - realized
        api_available = True

    total_profit = realized + unrealized
    equity = wallet_size + total_profit

    # Active pairs from API
    active_pairs = fetch_active_pairs(port, slug) if db_path.exists() else []

    return {
        **item,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "equity": equity,
        "profit_abs": realized,
        "unrealized": unrealized,
        "total_profit": total_profit,
        "api_available": api_available,
        "profit_pct": total_profit / wallet_size * 100 if wallet_size else 0.0,
        "closed_trades": len(closed),
        "open_trades": len(open_rows),
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "recent_trades": rows[:8],
        "active_pairs": active_pairs,
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
        "refresh_seconds": REFRESH_SECONDS,
        "strategies": summaries,
    }


# ── formatting ────────────────────────────────────────────────────────

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


# ── HTML renderers ────────────────────────────────────────────────────

def render_recent_trades(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">暂无交易记录，等待 dry-run 产生信号。</div>'
    rendered = []
    for row in rows:
        pair = html.escape(str(row.get("pair", "-")))
        is_open = bool(row.get("is_open"))
        state = "持仓中" if is_open else "已平仓"
        profit_abs = as_float(row.get("close_profit_abs"))
        profit_pct = as_float(row.get("close_profit")) * 100
        close_info = "浮动中" if is_open else fmt_usdt(profit_abs)
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
        '<div class="trade-table-wrap">'
        "<table><thead><tr><th>交易对</th><th>状态</th><th>开仓时间</th>"
        "<th>收益</th><th>收益率</th><th>标签</th></tr></thead><tbody>"
        + "".join(rendered)
        + "</tbody></table>"
        "</div>"
    )


def _render_pair_tags(pairs: list[str], core_pairs: list[str]) -> str:
    """Render a row of colored pair tags: core=blue, meme=purple."""
    if not pairs:
        return ""
    tags = []
    core_set = set(core_pairs)
    for p in pairs:
        cls = "pair-core" if p in core_set else "pair-meme"
        tags.append(f'<span class="{cls}">{html.escape(p)}</span>')
    return '<div class="pair-tags">' + "".join(tags) + "</div>"


def render_strategy_card(item: dict[str, Any]) -> str:
    total_class = css_class_for_profit(item.get("total_profit", 0))
    realized_class = css_class_for_profit(item["profit_abs"])
    unrealized = item.get("unrealized", 0.0)
    unrealized_class = css_class_for_profit(unrealized)
    api_available = item.get("api_available", False)

    # Status — based on actual process health, not just DB file existence
    if api_available:
        status = "运行中"
        status_cls = "status-online"
    elif item["db_exists"]:
        status = "进程离线"
        status_cls = "status-offline"
    else:
        status = "等待启动"
        status_cls = ""

    mode = item.get("mode", "dry-run")

    # Unrealized PnL row (only if API is available and not zero)
    unrealized_row = ""
    if api_available and abs(unrealized) > 0.001:
        unrealized_row = (
            f'<div><span>浮动盈亏</span>'
            f'<strong class="{unrealized_class}">{fmt_usdt(unrealized)}</strong></div>'
        )

    # Active pairs display
    pair_tags_html = _render_pair_tags(
        item.get("active_pairs", []),
        item.get("pairlist_core", []),
    )

    return f"""
    <section class="strategy-card">
        <div class="card-head">
            <div>
                <h3>{html.escape(item["name"])}</h3>
                <p>{html.escape(item["strategy"])} · {html.escape(mode)} · {html.escape(item["timeframe"])}</p>
            </div>
            <span class="status {status_cls}">{status}</span>
        </div>
        <div class="metric-grid">
            <div><span>权益</span><strong class="{total_class}">{fmt_usdt(item["equity"])}</strong></div>
            <div><span>盈亏合计</span><strong class="{total_class}">{fmt_usdt(item.get("total_profit", 0))}</strong></div>
            <div><span>已实现</span><strong class="{realized_class}">{fmt_usdt(item["profit_abs"])}</strong></div>
            {unrealized_row}
            <div><span>已平仓</span><strong>{item["closed_trades"]}</strong></div>
            <div><span>持仓中</span><strong>{item["open_trades"]}</strong></div>
            <div><span>胜率</span><strong>{item["winrate"]:.1f}%</strong></div>
        </div>
        <div class="mini-meta">
            <span>API: 127.0.0.1:{item["port"]}</span>
            <span>DB: {html.escape(Path(item["db"]).name)}</span>
        </div>
        {pair_tags_html}
        {render_recent_trades(item["recent_trades"])}
    </section>
    """


def render_dashboard_html(model: dict[str, Any]) -> str:
    payload = json.dumps(model, ensure_ascii=False)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in model["strategies"]:
        grouped[item["timeframe"]].append(item)

    groups = []
    for timeframe in ("1m", "5m"):
        cards = "".join(render_strategy_card(item) for item in grouped.get(timeframe, []))
        groups.append(
            f"""
            <section class="timeframe-section">
                <h2>{timeframe} 周期</h2>
                <div class="strategy-grid">{cards}</div>
            </section>
            """
        )

    total_profit = sum(i.get("total_profit", i["profit_abs"]) for i in model["strategies"])
    total_equity = sum(i["equity"] for i in model["strategies"])

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
    <title>Gate dry-run 策略集中看板</title>
    <style>
        :root {{
            --bg: #101214;
            --panel: #181c20;
            --panel-2: #22272d;
            --line: #343c45;
            --text: #edf2f7;
            --muted: #9aa8b6;
            --green: #45d483;
            --red: #ff6b6b;
            --gold: #f4c95d;
            --blue: #5b9bd5;
            --purple: #b388ff;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
        }}
        main {{
            width: min(1500px, calc(100% - 32px));
            margin: 0 auto;
            padding: 24px 0 34px;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 16px;
            margin-bottom: 18px;
        }}
        h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
        h1 {{ font-size: 28px; }}
        h2 {{ font-size: 20px; margin: 22px 0 12px; }}
        h3 {{ font-size: 17px; line-height: 1.25; }}
        header p {{
            margin: 7px 0 0;
            color: var(--muted);
            font-size: 14px;
        }}
        .summary-strip {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }}
        .summary-strip div, .strategy-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
        }}
        .summary-strip div {{ padding: 12px; }}
        .summary-strip span, .metric-grid span {{
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 6px;
        }}
        .summary-strip strong, .metric-grid strong {{ font-size: 20px; }}
        .strategy-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}
        .strategy-card {{
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
        .status-online {{
            color: var(--green);
            border-color: var(--green);
        }}
        .status-offline {{
            color: var(--red);
            border-color: var(--red);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 10px;
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
        .pair-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 12px;
        }}
        .pair-core, .pair-meme {{
            font-size: 11px;
            padding: 2px 7px;
            border-radius: 4px;
            white-space: nowrap;
        }}
        .pair-core {{
            background: #1a3a5c;
            color: var(--blue);
            border: 1px solid #1e4d7a;
        }}
        .pair-meme {{
            background: #2a1a3c;
            color: var(--purple);
            border: 1px solid #3e2a5a;
        }}
        .mini-meta {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
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
        .trade-table-wrap {{
            max-height: 320px;
            overflow-y: auto;
        }}
        .trade-table-wrap table {{
            margin-bottom: 0;
        }}
        .empty {{
            border-top: 1px solid var(--line);
            color: var(--muted);
            padding: 14px 2px 2px;
            font-size: 13px;
        }}
        @media (max-width: 980px) {{
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
                <h1>Gate dry-run 策略集中看板</h1>
                <p>5 策略 x 2 周期 = 10 个独立模拟盘 · 每小时波动率动态选币（4 核心 + 5 妖币）· {REFRESH_SECONDS} 秒自动刷新</p>
            </div>
            <p>生成时间：{html.escape(model["generated_at"])}</p>
        </header>
        <section class="summary-strip">
            <div><span>运行单元</span><strong>{len(model["strategies"])}</strong></div>
            <div><span>单元资金</span><strong>{fmt_usdt(model["wallet_size"])}</strong></div>
            <div><span>合计权益</span><strong class="{css_class_for_profit(total_equity - len(model['strategies']) * model['wallet_size'])}">{fmt_usdt(total_equity)}</strong></div>
            <div><span>合计盈利</span><strong class="{css_class_for_profit(total_profit)}">{fmt_usdt(total_profit)}</strong></div>
            <div><span>在线进程</span><strong>{sum(1 for i in model['strategies'] if i.get('api_available', False))}</strong></div>
        </section>
        {''.join(groups)}
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
