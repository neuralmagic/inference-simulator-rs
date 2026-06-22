#!/usr/bin/env bash
#
# End-to-end check of vLLM's token-in/token-out endpoint, POST /inference/v1/generate, against
# our mock engine. This is the render-bypass path: the caller supplies token_ids directly (no
# chat template, no tokenizer) and gets token_ids back (no detokenization), which is exactly
# how benchmarks drive a server without paying tokenizer cost. The frontend lowers it to the
# same EngineCoreRequest (prompt_token_ids + sampling_params) our engine already serves.
#
# Override FRONTEND_BIN to point at a built vllm-rs (see scripts/e2e.sh).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29550}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-8}"
FRONTEND_BIN="${FRONTEND_BIN:-$HOME/git/vllm-main/rust/target/debug/vllm-rs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_BIN="$REPO_ROOT/target/debug/vllm-vcr"
BASE_URL="http://${HTTP_HOST}:${HTTP_PORT}"
LOG_DIR="$(mktemp -d)"

frontend_pid=""
engine_pid=""

cleanup() {
    [[ -n "$engine_pid" ]] && kill "$engine_pid" 2>/dev/null || true
    [[ -n "$frontend_pid" ]] && kill "$frontend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    echo "--- frontend log ($LOG_DIR/frontend.log) ---" >&2
    tail -30 "$LOG_DIR/frontend.log" >&2 || true
    echo "--- engine log ($LOG_DIR/engine.log) ---" >&2
    tail -30 "$LOG_DIR/engine.log" >&2 || true
    exit 1
}

[[ -x "$FRONTEND_BIN" ]] || fail "frontend binary not found at $FRONTEND_BIN (build it: cargo build --bin vllm-rs)"
[[ -x "$ENGINE_BIN" ]] || { echo "building engine..."; (cd "$REPO_ROOT" && cargo build); }

echo "logs: $LOG_DIR"

echo "starting frontend ($MODEL) ..."
"$FRONTEND_BIN" serve "$MODEL" \
    --data-parallel-size 1 \
    --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    >"$LOG_DIR/frontend.log" 2>&1 &
frontend_pid=$!

echo "starting mock engine ..."
"$ENGINE_BIN" play \
    --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --log-requests \
    >"$LOG_DIR/engine.log" 2>&1 &
engine_pid=$!

echo "waiting for $BASE_URL/health ..."
for i in $(seq 1 120); do
    kill -0 "$frontend_pid" 2>/dev/null || fail "frontend exited during startup"
    kill -0 "$engine_pid" 2>/dev/null || fail "engine exited during startup"
    curl -fsS "$BASE_URL/health" >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
    sleep 1
    [[ "$i" == "120" ]] && fail "server did not become healthy within 120s"
done

# Token-in/token-out: hand the engine raw prompt token ids, expect generated token ids back.
echo "--- POST /inference/v1/generate (token-in/token-out) ---"
RESP=$(curl -fsS "$BASE_URL/inference/v1/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"token_ids\":[11,22,33,44],\"sampling_params\":{\"max_tokens\":$MAX_TOKENS}}") \
    || fail "generate request failed"
echo "$RESP"

# The frontend returns token_ids out (no detokenization), so assert on the token array, not text.
echo "$RESP" | grep -q '"token_ids"' || fail "no token_ids in generate response"
echo "$RESP" | grep -q '"finish_reason"' || fail "no finish_reason in generate response"
N=$(echo "$RESP" | grep -o '"token_ids":\[[^]]*\]' | tail -1 | grep -o ',' | wc -l | tr -d ' ')
# tail -1 picks the choice's output token_ids; comma count + 1 = token count.
GEN=$((N + 1))
[[ "$GEN" -eq "$MAX_TOKENS" ]] || fail "expected $MAX_TOKENS generated tokens, counted $GEN"

echo ""
echo "PASS: /inference/v1/generate returned $GEN output token_ids, finish_reason present (render-bypass path verified)."
