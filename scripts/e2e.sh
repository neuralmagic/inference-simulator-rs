#!/usr/bin/env bash
#
# End-to-end smoke test: real vLLM Rust frontend  <-->  our mock engine.
#
# Proves bird one: a streaming and non-streaming OpenAI completion flow all the
# way through vLLM's real frontend (tokenizer, chat template, SSE) against our
# mock engine-core backend. No GPU, no NIXL, no model weights, only the tokenizer
# is fetched from HF on first run.
#
# Layout:
#   vllm-rs serve --data-parallel-size-local 0   (binds handshake, waits for engine)
#        │  ZMQ + msgpack engine-core protocol
#   vllm-vcr play --handshake-address ...      (our backend, fakes the model)
#
# Override any of these via env:
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29550}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8000}"
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

# 1. Frontend first: binds the handshake ROUTER and waits for an external engine.
echo "starting frontend ($MODEL) ..."
"$FRONTEND_BIN" serve "$MODEL" \
    --data-parallel-size 1 \
    --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    >"$LOG_DIR/frontend.log" 2>&1 &
frontend_pid=$!

# 2. Our engine: connects as the headless DP engine and completes the handshake.
echo "starting mock engine ..."
"$ENGINE_BIN" play \
    --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --log-requests \
    >"$LOG_DIR/engine.log" 2>&1 &
engine_pid=$!

# 3. Wait for the server to come up (first run downloads the tokenizer from HF).
echo "waiting for $BASE_URL/health ..."
for i in $(seq 1 120); do
    kill -0 "$frontend_pid" 2>/dev/null || fail "frontend exited during startup"
    kill -0 "$engine_pid" 2>/dev/null || fail "engine exited during startup"
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
        echo "server is up after ${i}s"
        break
    fi
    sleep 1
    [[ "$i" == "120" ]] && fail "server did not become healthy within 120s"
done

req() {
    curl -fsS "$BASE_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$1"
}

# 4. Non-streaming completion: expect a finish_reason of length at 16 tokens.
echo "--- non-streaming ---"
NONSTREAM=$(req "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":16}") \
    || fail "non-streaming request failed"
echo "$NONSTREAM"
echo "$NONSTREAM" | grep -q '"finish_reason"' || fail "no finish_reason in non-streaming response"
echo "$NONSTREAM" | grep -q '"role":"assistant"' || fail "no assistant message in non-streaming response"

# 5. Streaming completion: expect multiple SSE chunks and a terminal [DONE].
echo "--- streaming ---"
STREAM=$(curl -fsS -N "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":16,\"stream\":true}") \
    || fail "streaming request failed"
echo "$STREAM" | tail -5
CHUNKS=$(echo "$STREAM" | grep -c '^data: ' || true)
[[ "$CHUNKS" -ge 2 ]] || fail "expected >=2 streaming chunks, got $CHUNKS"
echo "$STREAM" | grep -q '^data: \[DONE\]' || fail "streaming did not terminate with [DONE]"

echo ""
echo "PASS: $CHUNKS streaming chunks, finish_reason present, handshake + protocol verified end-to-end."
