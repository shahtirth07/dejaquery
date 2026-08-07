# Déjà Query

**An NL→SQL agent that gets cheaper every time it sees a familiar question shape.**

First ask of a new pattern → **cold path** (LLM writes SQL, real tokens).  
Same shape again (different N, time window, same filters) → **warm path** (reuse the learned template, **0 LLM tokens**).

Built for the EverOS × Snowflake hackathon track: EverOS remembers solved reasoning; Snowflake (SQLite locally) holds both the business data and the agent’s own token-economy log.

---

## Why it exists

Most agents pay full price on every question — even when they’ve already figured out how to answer that *class* of question. Déjà Query separates **reasoning** (once) from **slot-filling** (cheap forever after):

| Path | What happens | Tokens |
|------|----------------|--------|
| **Cold** | OpenAI sees the raw question + schema → full SQL → store as a skill | Real API usage |
| **Warm** | Semantic memory finds a similar past question → slot-fill `LIMIT` / time window | **0** |

The UI proves it live: path badges (`cold · LLM ran` / `warm · LLM skipped`), token counts, savings meter, and a cost breakdown from a real SQL `GROUP BY` — not client-side math.

---

## Quick start

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY for live cold path
./run_local.sh                # EverOS (:8100) + app (:8000) when MEMORY_BACKEND=everos
```

Open **http://127.0.0.1:8000**.

Without an OpenAI key, the cold path uses a labeled **mock** so you can still rehearse the warm/cold story offline.

Set `MEMORY_BACKEND=local` in `.env` to skip EverOS and use in-process embeddings only.

---

## 90-second demo

1. **Reset** the session.
2. Ask: `top 5 Electronics products by revenue this quarter` → **cold**.
3. Ask: `top 3 Electronics products by revenue last month` → **warm**, 0 tokens.
4. Ask a *new* shape (`revenue by channel this quarter`) → **cold** again.
5. Repeat that shape with a different time window → **warm**.
6. Open the **cost breakdown** panel — live SQL over `agent_cost_log`, grouped by pattern and path.

Pitch slides: [`docs/pitch.html`](./docs/pitch.html)

---

## Architecture

```
Question
   │
   ├─► EverOS / local memory ── similar skill? ──► WARM: fill {top_n}, {time_filter}
   │                                              └─► run SQL → log cost
   │
   └─► OpenAI (cold) ── SQL + usage tokens ──► learn skill ──► run SQL → log cost
```

| Layer | Role |
|-------|------|
| **OpenAI** | Cold-path NL→SQL only |
| **EverOS** (or local MiniLM) | Semantic memory of solved questions → reusable SQL skills |
| **SQLite** (`dejaquery_demo.db`) | Demo stand-in for Snowflake: sales data + `agent_cost_log` |
| **FastAPI + static UI** | Ask flow, badges, skills list, cost dashboard |

Snowflake pitch (say this out loud): we don’t only *query* the warehouse — we **store and analyze the agent’s token economy** with SQL (`GROUP BY question_shape, path`), the cost-visibility gap native Cortex Agent dashboards don’t fill.

---

## What’s real vs demo scaffolding

| Piece | Now | At the venue |
|-------|-----|--------------|
| Cold SQL | OpenAI if `OPENAI_API_KEY` set; else labeled mock | Same |
| Memory | `MEMORY_BACKEND=everos` → EverOS HTTP + session shadow; or `local` embeddings | Same |
| Business data | Seeded SQLite (regions, products, channel, ~6 months sales) | Point `run_query()` at Snowflake |
| Cost log | Real table `agent_cost_log` + `/cost-breakdown` | Same table in Snowflake |
| Warm tokens | Exactly **0** (regex slot-fill, no second LLM) | Same |

Wrong warm reuse is worse than an extra cold call: similarity is conservative, with a lexical guard so top≠bottom and filters (category / channel / region) must match.

---

## Project layout

```
app/
  main.py           FastAPI routes, MEMORY_BACKEND wiring
  sql_agent.py      Cold-path OpenAI (+ offline mock)
  memory_store.py   LocalMemoryStore + EverOSMemoryStore
  intent.py         Warm-path LIMIT / time slots only
  db.py             Schema, seed, run_query, cost log
  cost.py           Token → $ estimate + savings
static/index.html   Live dashboard
docs/pitch.html     Submission slide deck (open in browser)
run_local.sh        Starts EverOS + app from .env
```

---

## Environment

See [`.env.example`](./.env.example). Important knobs:

- `OPENAI_API_KEY` / `OPENAI_MODEL` — cold path
- `MEMORY_BACKEND` — `everos` | `local`
- `APP_PORT` / `EVEROS_PORT` — default `8000` / `8100`
- `EVEROS_CONFIDENCE_THRESHOLD` — EverOS search floor (default `0.70`)

Never commit `.env`.

---

## Swap to real Snowflake

In `app/db.py`, replace the body of `run_query()` with `snowflake-connector-python`. Prefer teaching the cold-path prompt Snowflake date dialect (`DATEADD` / `CURRENT_DATE`) and regenerating skills, or update `time_filter_sql()` to emit Snowflake SQL.
