"""
Déjà Query — the demo backend.

Run with:
    ./run_local.sh

Credentials and ports live in a repo-root ``.env`` (see ``.env.example``).
``MEMORY_BACKEND=everos`` starts EverOS + the app; ``local`` is offline-only.

Then open the APP_HOST:APP_PORT URL from your .env (default http://127.0.0.1:8000).

Flow per question:
  1. semantically search memory for a similar solved question (local embeddings)
  2. if similarity >= threshold -> reuse that skill's SQL template, slot-fill
     LIMIT / time window -> WARM (no LLM call)
  3. if not -> call ChatGPT with raw question + schema -> COLD, then store
     (question, SQL, embedding) so the next similar ask is warm
  4. run the SQL against the local demo DB (swap for Snowflake at the venue)
  5. log tokens + estimated cost, return everything to the frontend
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db, sql_agent
from app.cost import CostLog
from app.memory_store import EverOSMemoryStore, LocalMemoryStore


def _make_memory():
    """Pick memory backend from MEMORY_BACKEND in .env (no hardcoded swap)."""
    backend = (os.environ.get("MEMORY_BACKEND") or "local").strip().lower()
    if backend == "everos":
        return EverOSMemoryStore()
    if backend in ("local", "fallback", ""):
        return LocalMemoryStore()
    raise ValueError(
        f"Unknown MEMORY_BACKEND={backend!r}. Use 'everos' or 'local'."
    )


app = FastAPI(title="Déjà Query")

memory = _make_memory()
cost_log = CostLog()


def _memory_backend() -> dict:
    """Honest label for which memory implementation is live."""
    name = type(memory).__name__
    if name == "EverOSMemoryStore":
        return {
            "id": "everos",
            "label": "EverOS",
            "detail": f"HTTP API at {os.environ.get('EVEROS_URL', '(EVEROS_URL unset)')}",
        }
    return {
        "id": "local",
        "label": "local fallback",
        "detail": "in-process sentence embeddings (not EverOS)",
    }


def _llm_status(source: str | None) -> dict:
    """source is SqlResult.source ('api'|'mock') or None when LLM was skipped."""
    if source is None:
        return {
            "called": False,
            "source": "none",
            "label": "LLM skipped",
            "detail": "semantic memory reuse — no model call",
        }
    if source == "api":
        return {
            "called": True,
            "source": "api",
            "label": "live API",
            "detail": "OpenAI Chat Completions (OPENAI_API_KEY set)",
        }
    return {
        "called": True,
        "source": "mock",
        "label": "offline mock",
        "detail": "no OPENAI_API_KEY — tokens are estimates, not real usage",
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()

    skill = memory.find_skill(question)

    if skill is not None:
        # WARM — semantic match; LLM was not called. Tokens are genuinely zero.
        sql = memory.build_sql_from_skill(skill, question)
        input_tokens, output_tokens = 0, 0
        path = "warm"
        shape_label = skill.question
        llm = _llm_status(None)
        match_similarity = (
            round(skill.match_similarity, 3)
            if skill.match_similarity is not None
            else None
        )
    else:
        # COLD / unmatched — LLM generated SQL (live API or offline mock).
        result = sql_agent.generate_sql(question)
        sql = result.sql
        input_tokens, output_tokens = result.input_tokens, result.output_tokens
        llm = _llm_status(result.source)
        match_similarity = None
        learned = memory.learn_skill(question, sql)
        if learned is not None:
            path = "cold"
            shape_label = question
        else:
            path = "unmatched"
            shape_label = None

    try:
        cols, rows = db.run_query(sql)
        error = None
    except Exception as exc:  # noqa: BLE001 — surface to the demo UI, don't crash
        cols, rows = [], []
        error = str(exc)

    memory.record_case(question, path, sql, input_tokens, output_tokens)
    entry = cost_log.add(question, path, input_tokens, output_tokens)

    # The line that actually connects this to Snowflake's token-economy
    # story: the cost event is a real row, queryable with real SQL, not
    # just a Python object living in this process's memory.
    db.log_cost_entry(
        question=question,
        shape=shape_label,
        path=path,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_usd=entry.estimated_usd,
    )

    backend = _memory_backend()
    return {
        "question": question,
        "path": path,
        "sql": sql,
        "columns": cols,
        "rows": rows[:10],
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": round(entry.estimated_usd, 5),
        "running_total_usd": round(cost_log.total_usd(), 5),
        "savings_pct": cost_log.savings_pct(),
        "skills_learned": len(memory.list_skills()),
        "match_similarity": match_similarity,
        "llm_called": llm["called"],
        "llm_source": llm["source"],
        "llm_label": llm["label"],
        "llm_detail": llm["detail"],
        "memory_store": backend["id"],
        "memory_store_label": backend["label"],
        "memory_store_detail": backend["detail"],
        # Tokens are only "real" when the live API ran; mock estimates
        # and warm zeros must not be presented as billed API usage.
        "tokens_are_real": llm["source"] == "api",
    }


@app.get("/state")
def state():
    backend = _memory_backend()
    llm_cfg = _llm_status("api" if os.environ.get("OPENAI_API_KEY") else "mock")
    return {
        "cost_log": cost_log.as_dicts(),
        "running_total_usd": round(cost_log.total_usd(), 5),
        "savings_pct": cost_log.savings_pct(),
        "skills": [
            {
                "question": s.question,
                "times_reused": s.times_reused,
            }
            for s in memory.list_skills()
        ],
        "memory_store": backend["id"],
        "memory_store_label": backend["label"],
        "memory_store_detail": backend["detail"],
        "llm_config_label": llm_cfg["label"],
        "llm_config_detail": llm_cfg["detail"],
    }


@app.get("/cost-breakdown")
def cost_breakdown():
    """The judge-facing proof: this is a live aggregate query against
    the same database everything else runs on, not a client-side
    calculation. This is the per-pattern cost attribution Snowflake's
    own native agent dashboards don't give you today.
    """
    cols, rows = db.cost_summary_by_shape()
    return {
        "sql": db.COST_SUMMARY_SQL,
        "columns": cols,
        "rows": rows,
    }


@app.post("/reset")
def reset():
    global memory, cost_log
    memory = _make_memory()
    cost_log = CostLog()
    db.reset_demo_db()
    return {"status": "reset"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/favicon.ico")
def favicon():
    # Browsers request this automatically; avoid noisy 404s in the log.
    from fastapi.responses import Response

    return Response(status_code=204)
