#!/usr/bin/env bash
#
# Control-plane golden path: emulate the llm-d routing sidecar's NIXL V2 two-step
# (prefill then decode) against our engine, asserting the real vLLM kv_transfer_params
# schema. Runs on the Mac with no NIXL (Noop data plane moves no bytes; this validates
# the wire contract the sidecar relays, not the transfer).
#
#   step 1 (prefill): kv_transfer_params{do_remote_decode:true}, max_tokens=1
#       -> response kv_transfer_params{do_remote_prefill, remote_engine_id, remote_host,
#                                      remote_port, remote_block_ids, remote_request_id}
#   step 2 (decode):  kv_transfer_params = step1's remote_* -> normal completion
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29552}"
HTTP_PORT="${HTTP_PORT:-8002}"
ENGINE_ID="${ENGINE_ID:-mock-prefill-0}"
FRONTEND_BIN="${FRONTEND_BIN:-$HOME/git/vllm-main/rust/target/debug/vllm-rs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_BIN="$REPO_ROOT/target/debug/mock-engine-nixl"
BASE_URL="http://127.0.0.1:${HTTP_PORT}"
LOG_DIR="$(mktemp -d)"

fpid=""; epid=""
cleanup() { [[ -n "$epid" ]] && kill "$epid" 2>/dev/null || true; [[ -n "$fpid" ]] && kill "$fpid" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; echo "--- engine log ---" >&2; tail -40 "$LOG_DIR/engine.log" >&2 || true; exit 1; }

[[ -x "$FRONTEND_BIN" ]] || fail "frontend binary not found at $FRONTEND_BIN"
[[ -x "$ENGINE_BIN" ]] || { echo "building engine..."; (cd "$REPO_ROOT" && cargo build); }
echo "logs: $LOG_DIR"

"$FRONTEND_BIN" serve "$MODEL" --data-parallel-size 1 --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" --port "$HTTP_PORT" >"$LOG_DIR/frontend.log" 2>&1 &
fpid=$!
"$ENGINE_BIN" --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --engine-id "$ENGINE_ID" --side-channel-host 127.0.0.1 --side-channel-port 5600 \
    --log-requests >"$LOG_DIR/engine.log" 2>&1 &
epid=$!

echo "waiting for $BASE_URL/health ..."
for i in $(seq 1 120); do
    kill -0 "$fpid" 2>/dev/null || fail "frontend exited"; kill -0 "$epid" 2>/dev/null || fail "engine exited"
    curl -fsS "$BASE_URL/health" >/dev/null 2>&1 && break
    sleep 1; [[ "$i" == "120" ]] && fail "server not healthy"
done

# Step 1: prefill.
echo "--- step 1: prefill (do_remote_decode) ---"
P=$(curl -fsS "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],
    \"max_tokens\":1,\"stream\":false,\"kv_transfer_params\":{\"do_remote_decode\":true}}") || fail "prefill request failed"
echo "$P" | jq -c '.kv_transfer_params'
KV=$(echo "$P" | jq -c '.kv_transfer_params')
[[ "$KV" != "null" ]] || fail "prefill response missing kv_transfer_params"
[[ "$(echo "$KV" | jq -r '.do_remote_prefill')" == "true" ]] || fail "expected do_remote_prefill:true"
[[ "$(echo "$KV" | jq -r '.remote_engine_id')" == "$ENGINE_ID" ]] || fail "remote_engine_id != $ENGINE_ID"
[[ "$(echo "$KV" | jq -r '.remote_host')" == "127.0.0.1" ]] || fail "remote_host wrong"
[[ "$(echo "$KV" | jq -r '.remote_port')" == "5600" ]] || fail "remote_port wrong"
[[ "$(echo "$KV" | jq -r '.remote_block_ids | length')" -ge 1 ]] || fail "remote_block_ids empty"
[[ "$(echo "$KV" | jq -r '.remote_request_id')" != "null" ]] || fail "remote_request_id missing"

# Step 2: decode, feeding step 1's remote_* (as the sidecar does).
echo "--- step 2: decode (do_remote_prefill) ---"
D=$(curl -fsS "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],
    \"max_tokens\":8,\"stream\":false,\"kv_transfer_params\":$KV}") || fail "decode request failed"
echo "$D" | jq -c '{finish_reason: .choices[0].finish_reason, content: .choices[0].message.content}'
[[ "$(echo "$D" | jq -r '.choices[0].finish_reason')" == "length" ]] || fail "decode did not complete"
grep -q "pulled remote KV before decode" "$LOG_DIR/engine.log" || fail "decode-pull hook never fired"

echo ""
echo "PASS: prefill emits the real kv_transfer_params schema; decode consumes remote_* and pulls. Sidecar-compatible."
