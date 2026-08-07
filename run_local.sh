#!/usr/bin/env bash
# Start EverOS (if configured) + Déjà Query locally. All ports/URLs come from .env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing .env — copy .env.example to .env and fill in values." >&2
  exit 1
fi

# Load .env into this process (and children). Skip blank/comment lines.
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

VENV_BIN="${VENV_BIN:-$ROOT/venv/bin}"
if [[ ! -x "$VENV_BIN/uvicorn" ]]; then
  echo "venv not ready. Run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

MEMORY_BACKEND="${MEMORY_BACKEND:-everos}"
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
EVEROS_HOST="${EVEROS_HOST:-127.0.0.1}"
EVEROS_PORT="${EVEROS_PORT:-8100}"
EVEROS_ROOT="${EVEROS_ROOT:-$ROOT/.everos}"

# Build EverOS URL from host/port unless the user already set EVEROS_URL.
if [[ -z "${EVEROS_URL:-}" ]]; then
  EVEROS_URL="http://${EVEROS_HOST}:${EVEROS_PORT}"
fi
export MEMORY_BACKEND APP_HOST APP_PORT EVEROS_HOST EVEROS_PORT EVEROS_ROOT EVEROS_URL

# Point EverOS's own LLM / embedding clients at the same OpenAI key when present
# (EverOS reads EVEROS_*__* — no secrets hardcoded here).
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  export EVEROS_LLM__API_KEY="${EVEROS_LLM__API_KEY:-$OPENAI_API_KEY}"
  export EVEROS_LLM__BASE_URL="${EVEROS_LLM__BASE_URL:-https://api.openai.com/v1}"
  export EVEROS_LLM__MODEL="${EVEROS_LLM__MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}"
  export EVEROS_EMBEDDING__API_KEY="${EVEROS_EMBEDDING__API_KEY:-$OPENAI_API_KEY}"
  export EVEROS_EMBEDDING__BASE_URL="${EVEROS_EMBEDDING__BASE_URL:-https://api.openai.com/v1}"
  export EVEROS_EMBEDDING__MODEL="${EVEROS_EMBEDDING__MODEL:-text-embedding-3-small}"
fi

EVEROS_PID=""
APP_PID=""

cleanup() {
  echo ""
  echo "Shutting down…"
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "$EVEROS_PID" ]] && kill -0 "$EVEROS_PID" 2>/dev/null; then
    kill "$EVEROS_PID" 2>/dev/null || true
    wait "$EVEROS_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_http() {
  local url="$1"
  local name="$2"
  local tries="${3:-60}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "$name is up ($url)"
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for $name at $url" >&2
  return 1
}

if [[ "$MEMORY_BACKEND" == "everos" ]]; then
  if [[ ! -x "$VENV_BIN/everos" ]]; then
    echo "everos CLI missing. Run: ./venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi

  mkdir -p "$EVEROS_ROOT"
  if [[ ! -f "$EVEROS_ROOT/everos.toml" ]]; then
    echo "Initializing EverOS memory root at $EVEROS_ROOT"
    "$VENV_BIN/everos" init --root "$EVEROS_ROOT" || true
  fi

  if [[ "$EVEROS_PORT" == "$APP_PORT" ]]; then
    echo "EVEROS_PORT ($EVEROS_PORT) must differ from APP_PORT ($APP_PORT)." >&2
    exit 1
  fi

  echo "Starting EverOS on ${EVEROS_HOST}:${EVEROS_PORT} (root=$EVEROS_ROOT)"
  "$VENV_BIN/everos" server start \
    --host "$EVEROS_HOST" \
    --port "$EVEROS_PORT" \
    --root "$EVEROS_ROOT" \
    --log-level "${EVEROS_LOG_LEVEL:-WARNING}" &
  EVEROS_PID=$!

  wait_for_http "${EVEROS_URL}/health" "EverOS" 90
else
  echo "MEMORY_BACKEND=$MEMORY_BACKEND — skipping EverOS (using local fallback)"
fi

echo "Starting Déjà Query on ${APP_HOST}:${APP_PORT} (memory=$MEMORY_BACKEND)"
"$VENV_BIN/uvicorn" app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload &
APP_PID=$!

wait_for_http "http://${APP_HOST}:${APP_PORT}/" "Déjà Query" 60

echo ""
echo "Ready → http://${APP_HOST}:${APP_PORT}"
if [[ "$MEMORY_BACKEND" == "everos" ]]; then
  echo "EverOS → ${EVEROS_URL}"
fi
echo "Ctrl+C to stop both."
echo ""

wait "$APP_PID"
