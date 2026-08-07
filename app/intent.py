"""
Cheap, deterministic extraction of LIMIT / time-window *slots*.

Used only when the warm path has already decided (via semantic similarity)
to reuse a skill — filling in a new N or time window is regex work, not
reasoning. Cold-path SQL is never built from this Intent. See
sql_agent.generate_sql and memory_store.find_skill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

METRICS = {
    "revenue": ["revenue", "sales", "$", "dollars", "income"],
    "units": ["units", "unit", "quantity", "volume", "items sold"],
}

DIMENSIONS = {
    "region": ["region", "regions"],
    "product": ["product", "products", "item", "items"],
}

TIME_PATTERNS = [
    (r"\bthis quarter\b", "this_quarter"),
    (r"\blast quarter\b", "last_quarter"),
    (r"\bthis month\b", "this_month"),
    (r"\blast month\b", "last_month"),
    (r"\bthis week\b", "this_week"),
    (r"\blast week\b", "last_week"),
    (r"\blast (\d+) days\b", "last_n_days"),
    (r"\ball time\b", "all_time"),
]


@dataclass
class Intent:
    metric: str | None
    dimension: str | None
    top_n: int | None
    time_key: str | None
    time_n: int | None  # only set when time_key == "last_n_days"
    raw_question: str

    @property
    def shape_key(self) -> str | None:
        """The (metric, dimension) pair that identifies a reusable skill."""
        if self.metric and self.dimension:
            return f"{self.metric}::{self.dimension}"
        return None

    @property
    def is_fully_understood(self) -> bool:
        return bool(self.metric and self.dimension and self.time_key)


def parse(question: str) -> Intent:
    q = question.lower()

    metric = next(
        (m for m, kws in METRICS.items() if any(k in q for k in kws)), None
    )
    dimension = next(
        (d for d, kws in DIMENSIONS.items() if any(k in q for k in kws)), None
    )

    top_n_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", q)
    top_n = int(top_n_match.group(1)) if top_n_match else None

    time_key = None
    time_n = None
    for pattern, key in TIME_PATTERNS:
        m = re.search(pattern, q)
        if m:
            time_key = key
            if key == "last_n_days":
                time_n = int(m.group(1))
            break

    return Intent(
        metric=metric,
        dimension=dimension,
        top_n=top_n,
        time_key=time_key,
        time_n=time_n,
        raw_question=question,
    )
