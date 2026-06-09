#!/usr/bin/env bash
#
# End-to-end LoRA metric check: real vLLM Rust frontend <--> our mock engine, exercising the
# LoRA layer all the way to the vllm:lora_requests_info Prometheus gauge.
#
# This needs a vllm-rs built from the LoRA-gauge fork (the upstream Rust frontend at ba94a3b
# does NOT emit vllm:lora_requests_info; see the divergence note). Build it and point
# FRONTEND_BIN at it:
#
#   git clone https://github.com/wseaton/vllm && cd vllm && git checkout lora-info-gauge
#   cd rust && cargo build --bin vllm-rs
#   FRONTEND_BIN=$PWD/target/debug/vllm-rs ./scripts/e2e_lora.sh
#
# Flow:
#   1. load a (fake) LoRA adapter via POST /v1/load_lora_adapter  -> engine add_lora
#   2. send a request targeting that adapter; a slow inter-token latency keeps it decoding
#      (the engine emits scheduler_stats on each decode step, so running_lora_adapters stays
#      populated the whole time, unlike a prefill park which is silent until the first token)
#   3. scrape /metrics mid-flight and assert running_lora_adapters names our adapter
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
ADAPTER="${ADAPTER:-test-adapter}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29550}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8000}"
# Slow decode so the request stays running across the scrape window. ITL_MS per token, over
# DECODE_TOKENS tokens, so it runs ~ITL_MS*DECODE_TOKENS ms; keep that well over the scrape loop.
ITL_MS="${ITL_MS:-2000}"
DECODE_TOKENS="${DECODE_TOKENS:-12}"
FRONTEND_BIN="${FRONTEND_BIN:-$HOME/git/vllm-fork/rust/target/debug/vllm-rs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_BIN="$REPO_ROOT/target/debug/inference-sim"
BASE_URL="http://${HTTP_HOST}:${HTTP_PORT}"
LOG_DIR="$(mktemp -d)"

frontend_pid=""
engine_pid=""
req_pid=""

cleanup() {
    [[ -n "$req_pid" ]] && kill "$req_pid" 2>/dev/null || true
    [[ -n "$engine_pid" ]] && kill "$engine_pid" 2>/dev/null || true
    [[ -n "$frontend_pid" ]] && kill "$frontend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    echo "--- frontend log ($LOG_DIR/frontend.log) ---" >&2
    tail -40 "$LOG_DIR/frontend.log" >&2 || true
    echo "--- engine log ($LOG_DIR/engine.log) ---" >&2
    tail -40 "$LOG_DIR/engine.log" >&2 || true
    exit 1
}

[[ -x "$FRONTEND_BIN" ]] || fail "frontend binary not found at $FRONTEND_BIN
  build the LoRA-gauge fork: git checkout lora-info-gauge && (cd rust && cargo build --bin vllm-rs)
  then re-run with FRONTEND_BIN=/path/to/rust/target/debug/vllm-rs"
[[ -x "$ENGINE_BIN" ]] || { echo "building engine..."; (cd "$REPO_ROOT" && cargo build); }

echo "logs: $LOG_DIR"

# 1. Frontend: binds the handshake and waits for our external engine. The
#    /v1/load_lora_adapter route is only mounted when runtime LoRA updating is enabled (it's
#    an off-by-default admin endpoint), so opt in for this process.
echo "starting frontend ($MODEL) ..."
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "$FRONTEND_BIN" serve "$MODEL" \
    --data-parallel-size 1 \
    --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    >"$LOG_DIR/frontend.log" 2>&1 &
frontend_pid=$!

# 2. Engine: slow inter-token latency so a request stays in decode (emitting scheduler_stats
#    every step) across the scrape window.
echo "starting mock engine (itl=${ITL_MS}ms) ..."
"$ENGINE_BIN" \
    --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --inter-token-latency "$ITL_MS" \
    --log-requests \
    >"$LOG_DIR/engine.log" 2>&1 &
engine_pid=$!

# 3. Wait for health (first run downloads the tokenizer from HF).
echo "waiting for $BASE_URL/health ..."
for i in $(seq 1 120); do
    kill -0 "$frontend_pid" 2>/dev/null || fail "frontend exited during startup"
    kill -0 "$engine_pid" 2>/dev/null || fail "engine exited during startup"
    curl -fsS "$BASE_URL/health" >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
    sleep 1
    [[ "$i" == "120" ]] && fail "server did not become healthy within 120s"
done

# 4. Load a fake adapter. A non-local, non-existent path skips the local-path prefix check,
#    so the frontend just relays add_lora to the engine and exposes the adapter as a model id.
echo "--- load_lora_adapter ($ADAPTER) ---"
LOAD=$(curl -fsS "$BASE_URL/v1/load_lora_adapter" \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$ADAPTER\",\"lora_path\":\"$ADAPTER\"}") \
    || fail "load_lora_adapter request failed (engine add_lora rejected?)"
echo "$LOAD"

# 5. Fire a request against the adapter; the slow ITL keeps it decoding for the scrape window.
echo "--- request against adapter (backgrounded, ${DECODE_TOKENS} tokens @ ${ITL_MS}ms) ---"
curl -fsS "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ADAPTER\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$DECODE_TOKENS}" \
    >"$LOG_DIR/req.log" 2>&1 &
req_pid=$!

# 6. Scrape /metrics mid-flight and assert the gauge names our adapter.
echo "--- scraping /metrics for vllm:lora_requests_info ---"
found=""
for _ in $(seq 1 30); do
    sleep 1
    METRICS=$(curl -fsS "$BASE_URL/metrics" 2>/dev/null) || continue
    LINE=$(echo "$METRICS" | grep 'vllm:lora_requests_info' | grep "$ADAPTER" || true)
    if [[ -n "$LINE" ]]; then found="$LINE"; break; fi
done

[[ -n "$found" ]] || {
    echo "--- vllm:lora_requests_info lines seen ---" >&2
    curl -fsS "$BASE_URL/metrics" 2>/dev/null | grep 'lora' >&2 || echo "(no lora metric lines; is FRONTEND_BIN the patched fork build?)" >&2
    fail "vllm:lora_requests_info never named adapter '$ADAPTER' (gauge missing or engine not reporting it)"
}

echo ""
echo "$found"
echo "PASS: vllm:lora_requests_info reported running adapter '$ADAPTER' end-to-end."
