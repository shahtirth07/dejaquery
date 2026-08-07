"""
The memory layer: recognizes repeated question *shapes* and reuses a
learned SQL template instead of asking the LLM to re-derive the query.

Two implementations, same interface (swap one line in main.py):

- LocalMemoryStore   — local sentence embeddings + cosine similarity.
  Offline fallback; no external services once the embedding model is cached.

- EverOSMemoryStore  — talks to a locally running EverOS HTTP server
  (POST /api/v1/memory/add, /flush, /search). Same method names so
  main.py swaps with one line. Base URL from EVEROS_URL.

Matching is deliberately conservative: a wrong warm reuse is worse than
an extra cold LLM call, so borderline scores fall through to cold.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from app.intent import Intent, parse

# Cosine similarity floor for LocalMemoryStore warm reuse.
SIMILARITY_THRESHOLD = 0.86

# EverOS search score floor. Prefer cold when borderline — wrong reuse
# is worse than an extra LLM call. Override via EVEROS_CONFIDENCE_THRESHOLD.
EVEROS_CONFIDENCE_THRESHOLD = float(os.environ.get("EVEROS_CONFIDENCE_THRESHOLD", "0.70"))

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None

_TIME_PHRASES = (
    "this quarter",
    "last quarter",
    "this month",
    "last month",
    "this week",
    "last week",
    "all time",
)


def _get_embedder():
    """Lazy-load so import / reset stay cheap until the first ask."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


def _shape_text(question: str) -> str:
    """Normalize slot literals so embeddings compare *shape*, not N/time wording.

    Raw questions are still what we store and display; this only affects the
    vector used for cosine matching. Category / channel / region words stay
    so "Electronics by region" does not warm-match "Outdoor by region".
    """
    q = question.lower()
    # Keep top vs bottom distinct — they differ in ORDER BY, not just LIMIT.
    q = re.sub(r"\btop\s+\d+\b", "top N", q)
    q = re.sub(r"\bbottom\s+\d+\b", "bottom N", q)
    q = re.sub(r"\blast\s+\d+\s+days?\b", "TIME_WINDOW", q)
    for phrase in _TIME_PHRASES:
        q = q.replace(phrase, "TIME_WINDOW")
    return q


def _embed(question: str) -> np.ndarray:
    vec = _get_embedder().encode(_shape_text(question), normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are L2-normalized at encode time, so cosine == dot product.
    return float(np.dot(a, b))


def _ranking_polarity(q: str) -> str | None:
    if re.search(r"\b(bottom|worst|lowest|least)\b", q):
        return "asc"
    if re.search(r"\b(top|best|highest|most)\b", q):
        return "desc"
    return None


def _named_filters(q: str) -> set[str]:
    found: set[str] = set()
    for cat in ("electronics", "outdoor", "home", "office"):
        if cat in q:
            found.add(f"cat:{cat}")
    for ch in ("online", "retail", "partner"):
        if ch in q:
            found.add(f"ch:{ch}")
    for region in ("west", "east", "central", "north", "south"):
        if re.search(rf"\b{region}\b", q):
            found.add(f"reg:{region}")
    return found


def _metric_hint(q: str) -> str | None:
    if "unit" in q:
        return "units"
    if any(w in q for w in ("revenue", "sales", "dollar")):
        return "revenue"
    return None


def _warm_compatible(skill_question: str, new_question: str) -> bool:
    """Lexical guard so high cosine scores don't reuse the wrong SQL skeleton.

    Embeddings blur top↔bottom and can blur revenue↔units; filters baked into
    a skill template must also appear on the new question.
    """
    a = skill_question.lower()
    b = new_question.lower()

    pa, pb = _ranking_polarity(a), _ranking_polarity(b)
    if pa and pb and pa != pb:
        return False

    ma, mb = _metric_hint(a), _metric_hint(b)
    if ma and mb and ma != mb:
        return False

    # Filters baked into the skill SQL are part of the shape — must match.
    if _named_filters(a) != _named_filters(b):
        return False

    return True


def _sql_to_template(sql: str) -> str:
    """Turn cold-path LLM SQL into a reusable skill template.

    Warm path re-slots top_n and time_filter from the *new* question.
    Category / channel / region filters written by the LLM stay put —
    those are part of the skill's shape, not free slots.
    """
    template = re.sub(r"(?i)\bLIMIT\s+\d+", "LIMIT {top_n}", sql)

    date_patterns = [
        # last month
        r"sale_date\s*>=\s*date\('now',\s*'start of month',\s*'-1 month'\)\s*"
        r"AND\s*sale_date\s*<\s*date\('now',\s*'start of month'\)",
        # this month
        r"sale_date\s*>=\s*date\('now',\s*'start of month'\)",
        # relative day windows (this/last quarter, week, last N days)
        r"sale_date\s*>=\s*date\('now',\s*'-\d+\s*day'\)(?:\s*AND\s*sale_date\s*<\s*date\('now',\s*'-\d+\s*day'\))?",
        r"\b1\s*=\s*1\b",
    ]
    replaced = False
    for pattern in date_patterns:
        new_template, n = re.subn(
            pattern, "{time_filter}", template, count=1, flags=re.IGNORECASE
        )
        if n:
            template = new_template
            replaced = True
            break

    if not replaced:
        # Fall back: replace the whole WHERE clause (older behavior).
        template = re.sub(
            r"(?is)(WHERE\s+).*?(?=\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT\b|$)",
            r"\1{time_filter}\n",
            template,
            count=1,
        )
        if "{time_filter}" not in template:
            template = re.sub(
                r"(?i)(\n)(?=GROUP\s+BY|ORDER\s+BY|LIMIT\b)",
                r"\nWHERE {time_filter}\n",
                template,
                count=1,
            )
    return template.strip()


def time_filter_sql(time_key: str | None, time_n: int | None) -> str:
    if time_key == "last_n_days" and time_n:
        return f"sale_date >= date('now', '-{time_n} day')"
    mapping = {
        "this_week": "sale_date >= date('now', '-7 day')",
        "last_week": "sale_date >= date('now', '-14 day') AND sale_date < date('now', '-7 day')",
        "this_month": "sale_date >= date('now', 'start of month')",
        "last_month": "sale_date >= date('now', 'start of month', '-1 month') "
        "AND sale_date < date('now', 'start of month')",
        "this_quarter": "sale_date >= date('now', '-90 day')",
        "last_quarter": "sale_date >= date('now', '-180 day') AND sale_date < date('now', '-90 day')",
        "all_time": "1=1",
    }
    return mapping.get(time_key, "1=1")


@dataclass
class Skill:
    """A cold-path solve that can be reused on similar questions."""

    question: str
    sql_template: str
    embedding: np.ndarray | None
    learned_at: float
    times_reused: int = 0
    # Populated by find_skill for the winning match (UI / debugging).
    match_similarity: float | None = None


@dataclass
class Case:
    question: str
    path: str  # "cold" | "warm" | "unmatched"
    sql: str
    input_tokens: int
    output_tokens: int
    timestamp: float = field(default_factory=time.time)


class MemoryStore(Protocol):
    """Shared interface for LocalMemoryStore and EverOSMemoryStore."""

    def find_skill(self, question: str) -> Skill | None: ...

    def learn_skill(self, question: str, sql: str) -> Skill | None: ...

    def build_sql_from_skill(self, skill: Skill, question: str) -> str: ...

    def record_case(
        self, question: str, path: str, sql: str, input_tokens: int, output_tokens: int
    ) -> None: ...

    def list_skills(self) -> list[Skill]: ...

    def list_cases(self) -> list[Case]: ...


class LocalMemoryStore:
    """In-process semantic memory: sentence embeddings + cosine similarity."""

    def __init__(self) -> None:
        self._skills: list[Skill] = []
        self._cases: list[Case] = []

    def find_skill(self, question: str) -> Skill | None:
        """Return the closest stored skill if similarity clears the threshold.

        Scores below SIMILARITY_THRESHOLD (including borderline) → None
        so the caller takes the cold path.
        """
        if not self._skills:
            return None

        query_vec = _embed(question)
        best: Skill | None = None
        best_score = -1.0
        for skill in self._skills:
            if not _warm_compatible(skill.question, question):
                continue
            vec = skill.embedding if skill.embedding is not None else _embed(skill.question)
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best = skill

        if best is None or best_score < SIMILARITY_THRESHOLD:
            return None

        best.match_similarity = best_score
        return best

    def learn_skill(self, question: str, sql: str) -> Skill | None:
        """Store a cold-path (question, SQL) pair for future semantic reuse."""
        if not question.strip() or not sql.strip():
            return None
        skill = Skill(
            question=question.strip(),
            sql_template=_sql_to_template(sql),
            embedding=_embed(question),
            learned_at=time.time(),
        )
        self._skills.append(skill)
        return skill

    def build_sql_from_skill(self, skill: Skill, question: str) -> str:
        """Slot-fill the learned template for the new question's parameters.

        Intent parsing here is only for LIMIT / time-window slots — the
        decision to reuse already came from semantic similarity.
        """
        intent: Intent = parse(question)
        skill.times_reused += 1
        return skill.sql_template.format(
            time_filter=time_filter_sql(intent.time_key, intent.time_n),
            top_n=intent.top_n or 5,
        )

    def record_case(
        self, question: str, path: str, sql: str, input_tokens: int, output_tokens: int
    ) -> None:
        self._cases.append(
            Case(
                question=question,
                path=path,
                sql=sql,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    def list_skills(self) -> list[Skill]:
        return list(self._skills)

    def list_cases(self) -> list[Case]:
        return list(self._cases)


class EverOSMemoryStore:
    """EverOS-backed memory over the local HTTP API — same interface as Local.

    Talks to a running EverOS FastAPI server (not an in-process import)::

        POST {EVEROS_URL}/api/v1/memory/add
        POST {EVEROS_URL}/api/v1/memory/flush
        POST {EVEROS_URL}/api/v1/memory/search

    ``EVEROS_URL`` is required (port varies by EverOS version), e.g.
    ``http://127.0.0.1:8000``. Swap in ``app/main.py``::

        from app.memory_store import EverOSMemoryStore as MemoryStore

    LocalMemoryStore remains the offline fallback when EverOS isn't up.
    """

    APP_ID = "default"
    PROJECT_ID = "default"
    USER_ID = os.environ.get("EVEROS_USER_ID", "dejaquery")

    def __init__(self, *, base_url: str | None = None, timeout: float = 30.0) -> None:
        raw = (base_url or os.environ.get("EVEROS_URL") or "").strip().rstrip("/")
        if not raw:
            host = (os.environ.get("EVEROS_HOST") or "127.0.0.1").strip()
            port = (os.environ.get("EVEROS_PORT") or "").strip()
            if port:
                raw = f"http://{host}:{port}"
        if not raw:
            raise ValueError(
                "EverOSMemoryStore requires EVEROS_URL or EVEROS_HOST+EVEROS_PORT "
                "in the environment / .env. Use LocalMemoryStore (MEMORY_BACKEND=local) "
                "when the EverOS server isn't running."
            )
        self.base_url = raw
        self.timeout = timeout
        # Session-local shadow so list_skills() works and we can recover a
        # SQL template if search returns a hit whose body was rewritten.
        self._skills: list[Skill] = []
        self._cases: list[Case] = []

    # ── Public interface (mirrors LocalMemoryStore) ──────────────────────

    def find_skill(self, question: str) -> Skill | None:
        """Search EverOS first; fall back to the session shadow if search misses.

        EverOS hybrid/vector search needs embeddings configured. We request
        ``method=keyword`` so a missing embed provider doesn't 500 the
        warm path. If EverOS still returns nothing useful (cascade lag,
        empty index), match against skills learned earlier in this process
        with the same conservative cosine threshold as LocalMemoryStore.
        """
        hit = self._find_via_everos(question)
        if hit is not None:
            return hit
        return self._find_via_shadow(question)

    def _find_via_everos(self, question: str) -> Skill | None:
        try:
            payload = self._post(
                "/api/v1/memory/search",
                {
                    "user_id": self.USER_ID,
                    "app_id": self.APP_ID,
                    "project_id": self.PROJECT_ID,
                    "query": question,
                    "top_k": 5,
                    # keyword works without an embedding provider; hybrid/vector do not
                    "method": "keyword",
                },
            )
        except Exception:
            return None

        best: Skill | None = None
        best_score = -1.0
        for score, text, question_hint in _iter_search_hits(payload):
            if score < EVEROS_CONFIDENCE_THRESHOLD:
                continue
            hint = (question_hint or _question_from_memory_text(text) or "").strip()
            if hint and not _warm_compatible(hint, question):
                continue
            template = _extract_sql_template(text)
            if not template:
                template = self._shadow_template_for(question_hint or text)
            if not template:
                continue
            if score > best_score:
                best_score = score
                best = Skill(
                    question=(hint or question).strip(),
                    sql_template=_sql_to_template(template)
                    if "{top_n}" not in template
                    else template,
                    embedding=None,
                    learned_at=time.time(),
                    match_similarity=score,
                )
        return best

    def _find_via_shadow(self, question: str) -> Skill | None:
        """Same-session warm path when EverOS search has nothing usable yet."""
        if not self._skills:
            return None
        query_vec = _embed(question)
        best: Skill | None = None
        best_score = -1.0
        for skill in self._skills:
            if not _warm_compatible(skill.question, question):
                continue
            vec = skill.embedding if skill.embedding is not None else _embed(skill.question)
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best = skill
        if best is None or best_score < SIMILARITY_THRESHOLD:
            return None
        best.match_similarity = best_score
        return best

    def learn_skill(self, question: str, sql: str) -> Skill | None:
        """Cold-path: add the solve to EverOS, flush extraction, keep a shadow copy."""
        if not question.strip() or not sql.strip():
            return None

        template = _sql_to_template(sql)
        skill = Skill(
            question=question.strip(),
            sql_template=template,
            embedding=_embed(question),
            learned_at=time.time(),
        )
        self._skills.append(skill)

        session_id = (
            "dejaquery-"
            + hashlib.sha256(question.strip().encode()).hexdigest()[:12]
        )
        content = _pack_memory_content(question, template)
        now_ms = int(time.time() * 1000)

        try:
            self._post(
                "/api/v1/memory/add",
                {
                    "session_id": session_id,
                    "app_id": self.APP_ID,
                    "project_id": self.PROJECT_ID,
                    "messages": [
                        {
                            "sender_id": "agent",
                            "role": "user",
                            "timestamp": now_ms,
                            "content": content,
                        }
                    ],
                },
            )
            self._post(
                "/api/v1/memory/flush",
                {
                    "session_id": session_id,
                    "app_id": self.APP_ID,
                    "project_id": self.PROJECT_ID,
                },
            )
        except Exception:
            # Shadow still holds the skill for this process; EverOS write failed.
            pass

        return skill

    def build_sql_from_skill(self, skill: Skill, question: str) -> str:
        intent: Intent = parse(question)
        skill.times_reused += 1
        return skill.sql_template.format(
            time_filter=time_filter_sql(intent.time_key, intent.time_n),
            top_n=intent.top_n or 5,
        )

    def record_case(
        self, question: str, path: str, sql: str, input_tokens: int, output_tokens: int
    ) -> None:
        self._cases.append(
            Case(
                question=question,
                path=path,
                sql=sql,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    def list_skills(self) -> list[Skill]:
        return list(self._skills)

    def list_cases(self) -> list[Case]:
        return list(self._cases)

    # ── HTTP helpers ─────────────────────────────────────────────────────

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            if not resp.content:
                return {}
            return resp.json()

    def _shadow_template_for(self, text: str) -> str | None:
        """Fall back to a session-local template if EverOS text isn't parseable."""
        hint = (_question_from_memory_text(text) or text or "").strip().lower()
        if not hint or not self._skills:
            return None
        for skill in self._skills:
            if skill.question.lower() in hint or hint in skill.question.lower():
                return skill.sql_template
        return self._skills[-1].sql_template if self._skills else None


_SQL_TEMPLATE_FENCE_OPEN = "```sql-template"
_SQL_TEMPLATE_FENCE_CLOSE = "```"


def _pack_memory_content(question: str, sql_template: str) -> str:
    """Serialize question + SQL for the EverOS /memory/add message body."""
    return (
        f"Question: {question.strip()}\n\n"
        f"{_SQL_TEMPLATE_FENCE_OPEN}\n{sql_template}\n{_SQL_TEMPLATE_FENCE_CLOSE}\n"
    )


def _question_from_memory_text(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"(?im)^Question:\s*(.+)$", text)
    return m.group(1).strip() if m else None


def _extract_sql_template(text: str | None) -> str | None:
    """Pull a sql-template (or plain sql) fenced block out of stored text."""
    if not text:
        return None
    m = re.search(
        r"```(?:sql-template|sql)\s*\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    if re.search(r"(?i)\bSELECT\b", text) and re.search(r"(?i)\bFROM\b", text):
        # Strip a leading "Question: …" line if present.
        cleaned = re.sub(r"(?im)^Question:.*\n+", "", text).strip()
        return cleaned or text.strip()
    return None


def _iter_search_hits(payload: Any) -> list[tuple[float, str, str | None]]:
    """Normalize EverOS search JSON into (score, text, question_hint) triples.

    Handles the v1/v2 envelope (``data.episodes`` / ``agent_skills`` /
    ``agent_cases``) and a few flatter shapes so port/version drift
    doesn't silently yield zero warm hits.
    """
    hits: list[tuple[float, str, str | None]] = []
    if not isinstance(payload, dict):
        return hits

    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return hits

    for ep in data.get("episodes") or []:
        if not isinstance(ep, dict):
            continue
        score = float(ep.get("score") or 0.0)
        parts = [
            ep.get("subject") or "",
            ep.get("summary") or "",
            ep.get("episode") or "",
        ]
        for fact in ep.get("atomic_facts") or []:
            if isinstance(fact, dict) and fact.get("content"):
                parts.append(str(fact["content"]))
                if fact.get("score") is not None:
                    score = max(score, float(fact["score"]))
        text = "\n".join(p for p in parts if p)
        if text.strip():
            hits.append((score, text, _question_from_memory_text(text)))

    for sk in data.get("agent_skills") or []:
        if not isinstance(sk, dict):
            continue
        text = "\n".join(
            p
            for p in (sk.get("description") or "", sk.get("content") or "")
            if p
        )
        if text.strip():
            hits.append(
                (
                    float(sk.get("score") or sk.get("confidence") or 0.0),
                    text,
                    sk.get("description"),
                )
            )

    for case in data.get("agent_cases") or []:
        if not isinstance(case, dict):
            continue
        text = "\n".join(
            p
            for p in (case.get("task_intent") or "", case.get("approach") or "")
            if p
        )
        if text.strip():
            hits.append(
                (
                    float(case.get("score") or case.get("quality_score") or 0.0),
                    text,
                    case.get("task_intent"),
                )
            )

    # Flat list fallbacks some older stubs used.
    for key in ("results", "memories", "items"):
        for item in data.get(key) or payload.get(key) or []:
            if not isinstance(item, dict):
                continue
            text = str(
                item.get("content")
                or item.get("text")
                or item.get("memory")
                or ""
            )
            if not text.strip():
                continue
            hits.append(
                (
                    float(item.get("score") or item.get("confidence") or 0.0),
                    text,
                    _question_from_memory_text(text),
                )
            )

    hits.sort(key=lambda h: h[0], reverse=True)
    return hits
