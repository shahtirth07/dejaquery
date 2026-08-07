"""
The expensive path: turn a natural-language question into SQL by
actually reasoning about the schema. This only runs on the *first*
question of a new shape — see memory_store.py for what happens after.

The cold path is load-bearing: the model receives the raw question and
SCHEMA_DESCRIPTION and must produce the full SQL. No Intent / regex
pre-parse is passed in; slot values (top N, time window, metric,
dimension) are whatever the model reads from the question text.

Set OPENAI_API_KEY in the environment to use the ChatGPT API. Without
it, this falls back to a deterministic offline mock so you can build
and rehearse the demo before you have a key. The mock also takes only
the raw question string (not a pre-parsed Intent). The mock is clearly
labeled — don't present its token counts as real API usage.

Model id is configurable via OPENAI_MODEL (default gpt-4o-mini).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from app.db import SCHEMA_DESCRIPTION

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = f"""You are a SQL generator for a demo analytics database (SQLite dialect).

{SCHEMA_DESCRIPTION}

Rules:
- Return ONLY the SQL query, no explanation, no markdown fences.
- Use SQLite date functions (date('now', ...)), not Snowflake syntax.
- Include an explicit LIMIT when the question asks for a top/bottom N.
- Infer metric, dimensions, filters, ranking direction, and time window
  from the question alone.
- For "by month" / trend questions, group with strftime('%Y-%m', sale_date).
- For "by week", group with strftime('%Y-%W', sale_date).
- For averages like revenue per unit, use SUM(revenue) * 1.0 / NULLIF(SUM(units), 0).
- Prefer clear column aliases (total_revenue, total_units, month, channel, …).
"""


@dataclass
class SqlResult:
    sql: str
    input_tokens: int
    output_tokens: int
    source: str  # "api" | "mock"


def _strip_fences(sql_text: str) -> str:
    text = sql_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _mock_generate(question: str) -> SqlResult:
    """Offline stand-in so the app runs with zero setup.

    Takes the raw natural-language question only — same contract as the
    live API path. Handles the broadened demo shapes (rankings, filters,
    trends, totals) well enough to rehearse without a key.
    """
    q = question.lower()

    metric_col = "units" if "unit" in q else "revenue"
    metric_expr = f"s.{metric_col}"
    ascending = bool(re.search(r"\b(bottom|worst|lowest|least)\b", q))
    order = "ASC" if ascending else "DESC"

    # Time window
    if "last month" in q:
        time_filter = (
            "sale_date >= date('now', 'start of month', '-1 month') "
            "AND sale_date < date('now', 'start of month')"
        )
    elif "this month" in q:
        time_filter = "sale_date >= date('now', 'start of month')"
    elif "last quarter" in q:
        time_filter = (
            "sale_date >= date('now', '-180 day') "
            "AND sale_date < date('now', '-90 day')"
        )
    elif "this quarter" in q:
        time_filter = "sale_date >= date('now', '-90 day')"
    elif "last week" in q:
        time_filter = (
            "sale_date >= date('now', '-14 day') "
            "AND sale_date < date('now', '-7 day')"
        )
    elif "this week" in q:
        time_filter = "sale_date >= date('now', '-7 day')"
    else:
        last_n = re.search(r"\blast\s+(\d+)\s+days?\b", q)
        if last_n:
            time_filter = f"sale_date >= date('now', '-{last_n.group(1)} day')"
        else:
            time_filter = "1=1"

    extra = []
    for cat in ("electronics", "outdoor", "home", "office"):
        if cat in q:
            extra.append(f"p.category = '{cat.title()}'")
            break
    for ch in ("online", "retail", "partner"):
        if ch in q:
            extra.append(f"s.channel = '{ch.title()}'")
            break
    for region in ("west", "east", "central", "north", "south"):
        if re.search(rf"\b{region}\b", q):
            extra.append(f"r.region_name = '{region.title()}'")
            break

    where = time_filter
    if extra:
        where = time_filter + " AND " + " AND ".join(extra)

    # Trends
    if "by month" in q or "per month" in q or "monthly" in q:
        sql = (
            f"SELECT strftime('%Y-%m', s.sale_date) AS month, "
            f"SUM({metric_expr}) AS total_{metric_col}\n"
            f"FROM sales s\n"
            f"WHERE {where}\n"
            f"GROUP BY month\n"
            f"ORDER BY month"
        )
    elif "by channel" in q or "per channel" in q:
        sql = (
            f"SELECT s.channel, SUM({metric_expr}) AS total_{metric_col}\n"
            f"FROM sales s\n"
            f"WHERE {where}\n"
            f"GROUP BY s.channel\n"
            f"ORDER BY total_{metric_col} {order}"
        )
    elif "by category" in q or "per category" in q:
        sql = (
            f"SELECT p.category, SUM({metric_expr}) AS total_{metric_col}\n"
            f"FROM sales s JOIN products p ON s.product_id = p.product_id\n"
            f"WHERE {where}\n"
            f"GROUP BY p.category\n"
            f"ORDER BY total_{metric_col} {order}"
        )
    elif re.search(r"\b(total|sum|overall)\b", q) and "top" not in q and "bottom" not in q:
        needs_product = "p.category" in where
        needs_region = "r.region_name" in where
        joins = []
        if needs_product:
            joins.append("JOIN products p ON s.product_id = p.product_id")
        if needs_region:
            joins.append("JOIN regions r ON s.region_id = r.region_id")
        join_sql = (" " + " ".join(joins)) if joins else ""
        sql = (
            f"SELECT SUM({metric_expr}) AS total_{metric_col}\n"
            f"FROM sales s{join_sql}\n"
            f"WHERE {where}"
        )
    else:
        # Rankings / defaults
        if "product" in q or "item" in q or "p.category" in where:
            join_clause = "products p ON s.product_id = p.product_id"
            group_col = "p.product_name"
            if "r.region_name" in where:
                join_clause = (
                    "products p ON s.product_id = p.product_id "
                    "JOIN regions r ON s.region_id = r.region_id"
                )
        elif "channel" in q and "by channel" not in q:
            join_clause = "regions r ON s.region_id = r.region_id"
            group_col = "r.region_name"
        else:
            join_clause = "regions r ON s.region_id = r.region_id"
            group_col = "r.region_name"
            if "p.category" in where:
                join_clause = (
                    "regions r ON s.region_id = r.region_id "
                    "JOIN products p ON s.product_id = p.product_id"
                )

        top_n_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", q)
        top_n = int(top_n_match.group(1)) if top_n_match else 5
        sql = (
            f"SELECT {group_col}, SUM({metric_expr}) AS total_{metric_col}\n"
            f"FROM sales s JOIN {join_clause}\n"
            f"WHERE {where}\n"
            f"GROUP BY {group_col}\n"
            f"ORDER BY total_{metric_col} {order}\n"
            f"LIMIT {top_n}"
        )

    approx_input = len(SYSTEM_PROMPT.split()) + len(question.split())
    approx_output = len(sql.split())
    return SqlResult(
        sql=sql,
        input_tokens=int(approx_input * 1.3),
        output_tokens=int(approx_output * 1.3),
        source="mock",
    )


def _api_generate(question: str) -> SqlResult:
    """Call ChatGPT with raw question + schema; return SQL and usage tokens."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    sql_text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return SqlResult(
        sql=_strip_fences(sql_text),
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        source="api",
    )


def generate_sql(question: str) -> SqlResult:
    """Produce full SQL from a raw NL question. No Intent / regex input."""
    if os.environ.get("OPENAI_API_KEY"):
        return _api_generate(question)
    return _mock_generate(question)
