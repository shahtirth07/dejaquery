"""
Local stand-in for Snowflake.

For the hackathon demo this runs against SQLite so the whole thing works
offline with zero setup. At the venue, swap `run_query` for a real
snowflake-connector-python call — the SQL this app generates is plain
ANSI-ish SQL and should work against Snowflake with minor dialect tweaks
(SQLite uses `strftime`, Snowflake uses `DATE_TRUNC` / date functions —
see the note at the bottom of this file).
"""

from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "dejaquery_demo.db"

SCHEMA_DESCRIPTION = """
Tables:

regions(region_id INTEGER PRIMARY KEY, region_name TEXT)
  -- region_name in: West, East, Central, North, South

products(product_id INTEGER PRIMARY KEY, product_name TEXT, category TEXT)
  -- category in: Home, Outdoor, Electronics, Office

sales(
    sale_id INTEGER PRIMARY KEY,
    region_id INTEGER REFERENCES regions(region_id),
    product_id INTEGER REFERENCES products(product_id),
    sale_date TEXT,   -- 'YYYY-MM-DD'
    channel TEXT,     -- 'Online' | 'Retail' | 'Partner'
    revenue REAL,
    units INTEGER
)

Notes:
- sale_date spans the last 6 months from today.
- Measurable metrics: revenue, units.
- Join sales -> regions on region_id, sales -> products on product_id when a
  question asks about a region or product name rather than an id.
- Filter by p.category, s.channel, or r.region_name when the question names them.
- Supported question styles (examples, not an exhaustive list):
  * rankings: top / bottom N by revenue or units
  * filters: "in Electronics", "Online channel only", "West region"
  * trends: revenue by month, units by week
  * totals: total revenue this quarter, average order size (revenue/units)
  * breakdowns: revenue by category, by channel, by region
""".strip()


def _seed(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name TEXT);
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT, category TEXT);
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            region_id INTEGER,
            product_id INTEGER,
            sale_date TEXT,
            channel TEXT,
            revenue REAL,
            units INTEGER
        );
        CREATE TABLE agent_cost_log (
            log_id INTEGER PRIMARY KEY,
            question TEXT,
            shape TEXT,
            path TEXT,          -- 'cold' | 'warm' | 'unmatched'
            input_tokens INTEGER,
            output_tokens INTEGER,
            estimated_usd REAL,
            logged_at TEXT DEFAULT (datetime('now'))
        );
        """
    )

    regions = ["West", "East", "Central", "North", "South"]
    products = [
        ("Aria Desk Lamp", "Home"),
        ("Kepler Backpack", "Outdoor"),
        ("Nimbus Headphones", "Electronics"),
        ("Solace Water Bottle", "Outdoor"),
        ("Vantage Monitor", "Electronics"),
        ("Wren Notebook Set", "Office"),
    ]
    channels = ["Online", "Retail", "Partner"]

    cur.executemany(
        "INSERT INTO regions (region_name) VALUES (?)", [(r,) for r in regions]
    )
    cur.executemany(
        "INSERT INTO products (product_name, category) VALUES (?, ?)", products
    )

    random.seed(7)  # deterministic seed data across runs
    today = dt.date.today()
    start = today - dt.timedelta(days=180)
    rows = []
    d = start
    while d <= today:
        # a handful of sales per day, skewed so some region/product combos
        # are obviously "top" — makes demo answers look sensible
        for _ in range(random.randint(2, 6)):
            region_id = random.choices(range(1, 6), weights=[3, 2, 2, 1, 1])[0]
            product_id = random.choices(range(1, 7), weights=[2, 1, 3, 1, 3, 1])[0]
            channel = random.choices(channels, weights=[3, 2, 1])[0]
            units = random.randint(1, 40)
            unit_price = random.uniform(15, 220)
            revenue = round(units * unit_price, 2)
            rows.append((region_id, product_id, d.isoformat(), channel, revenue, units))
        d += dt.timedelta(days=1)

    cur.executemany(
        "INSERT INTO sales (region_id, product_id, sale_date, channel, revenue, units) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(sales)").fetchall()
    }
    return "channel" in cols


def get_connection() -> sqlite3.Connection:
    fresh = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if fresh:
        _seed(conn)
    elif not _schema_is_current(conn):
        conn.close()
        DB_PATH.unlink()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        _seed(conn)
    return conn


def run_query(sql: str) -> tuple[list[str], list[tuple]]:
    """Execute SQL against the demo DB. Returns (column_names, rows).

    Swap this function's body for a real Snowflake call at the venue:

        import snowflake.connector
        conn = snowflake.connector.connect(...)
        cur = conn.cursor()
        cur.execute(sql)
        return [c[0] for c in cur.description], cur.fetchall()

    Everything upstream (intent parsing, skill matching, SQL generation)
    is unaware of where the query actually runs.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, [tuple(r) for r in rows]
    finally:
        conn.close()


def log_cost_entry(
    question: str,
    shape: str | None,
    path: str,
    input_tokens: int,
    output_tokens: int,
    estimated_usd: float,
) -> None:
    """Write one agent cost event as a real row, not a Python object.

    This is the piece that actually ties the token-economy story to
    Snowflake: at the venue, this INSERT runs against a real Snowflake
    table, and `cost_summary_by_shape()` below becomes a real Snowflake
    query, giving this agent exactly the per-pattern cost breakdown
    Snowflake's own native Cortex Agent dashboards don't provide.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO agent_cost_log "
            "(question, shape, path, input_tokens, output_tokens, estimated_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, shape, path, input_tokens, output_tokens, estimated_usd),
        )
        conn.commit()
    finally:
        conn.close()


COST_SUMMARY_SQL = """\
SELECT
    COALESCE(shape, '(unmatched)') AS question_shape,
    path,
    COUNT(*) AS times_asked,
    SUM(input_tokens + output_tokens) AS total_tokens,
    ROUND(SUM(estimated_usd), 5) AS total_usd,
    ROUND(AVG(estimated_usd), 5) AS avg_usd_per_question
FROM agent_cost_log
GROUP BY question_shape, path
ORDER BY question_shape, path"""


def cost_summary_by_shape() -> tuple[list[str], list[tuple]]:
    """The dashboard's cost-breakdown panel — a real aggregate query,
    not a client-side reduce over a JS array. Swap the SQL dialect
    (STRFTIME -> DATE_TRUNC etc.) if you move this to real Snowflake.
    """
    return run_query(COST_SUMMARY_SQL)


def reset_demo_db() -> None:
    """Delete and re-seed the demo database (handy between rehearsals)."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    get_connection().close()
