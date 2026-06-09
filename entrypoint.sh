#!/usr/bin/env bash
#
# Launch the two processes that make up a mock model-server pod: the real vLLM Rust
# frontend (vllm-rs) and our mock engine-core backend (inference-simulator-rs), wired over
# the in-pod engine-core handshake. Role, identity, and ports come from the env so the
# same image serves as a prefill or a decode instance in the llm-d P/D path.
set -euo pipefail

MODEL="${MODEL:?MODEL env required (HF model id, used for the tokenizer)}"
ROLE="${MOCK_PD_ROLE:-both}"                 # prefill | decode | both
ENGINE_ID="${POD_NAME:-mock-engine-0}"       # advertised as remote_engine_id
SIDE_CHANNEL_HOST="${POD_IP:-0.0.0.0}"       # advertised as remote_host
SIDE_CHANNEL_PORT="${MOCK_SIDE_CHANNEL_PORT:-5600}"
HANDSHAKE_PORT="${MOCK_HANDSHAKE_PORT:-29550}"

# Latency model knobs (milliseconds; 0 = instant, the binary's default). The frontend
# measures TTFT/ITL from when the engine emits tokens, so these drive the vllm:* timing
# metrics. Left unset -> instant, identical to the pre-latency behavior. Tune per-pod from
# the deployment env (prefill vs decode use different paths, see the manifests).
LATENCY_ARGS=(
    --time-to-first-token "${MOCK_TTFT_MS:-0}"
    --time-to-first-token-std-dev "${MOCK_TTFT_STDDEV_MS:-0}"
    --inter-token-latency "${MOCK_ITL_MS:-0}"
    --inter-token-latency-std-dev "${MOCK_ITL_STDDEV_MS:-0}"
    --prefill-overhead "${MOCK_PREFILL_OVERHEAD_MS:-0}"
    --prefill-time-per-token "${MOCK_PREFILL_TIME_PER_TOKEN_MS:-0}"
    --prefill-time-std-dev "${MOCK_PREFILL_TIME_STDDEV_MS:-0}"
    --kv-cache-transfer-latency "${MOCK_KV_TRANSFER_LATENCY_MS:-0}"
    --kv-cache-transfer-latency-std-dev "${MOCK_KV_TRANSFER_LATENCY_STDDEV_MS:-0}"
    --kv-cache-transfer-time-per-token "${MOCK_KV_TRANSFER_TIME_PER_TOKEN_MS:-0}"
    --kv-cache-transfer-time-std-dev "${MOCK_KV_TRANSFER_TIME_STDDEV_MS:-0}"
    --time-factor-under-load "${MOCK_TIME_FACTOR_UNDER_LOAD:-1.0}"
)

# Scheduler knobs, mirroring vLLM's flags/defaults. max-num-seqs caps the running batch;
# max-num-batched-tokens is the per-step token budget; scheduling-policy is fcfs|priority.
# The waiting queue is unbounded (vLLM never sheds load on queue length).
SCHEDULER_ARGS=(
    --max-num-seqs "${MOCK_MAX_NUM_SEQS:-128}"
    --max-num-batched-tokens "${MOCK_MAX_NUM_BATCHED_TOKENS:-2048}"
    --long-prefill-token-threshold "${MOCK_LONG_PREFILL_TOKEN_THRESHOLD:-0}"
    --scheduling-policy "${MOCK_SCHEDULING_POLICY:-fcfs}"
    --kv-cache-size "${MOCK_KV_CACHE_SIZE:-1024}"
    --tokens-per-block "${MOCK_TOKENS_PER_BLOCK:-16}"
)

# KV-cache events for the cache-aware router (Phase 4). Off unless MOCK_ENABLE_KV_EVENTS=1.
# The precise-prefix-cache-routing guide binds tcp://*:5556 and expects the topic
# kv@<pod-id>@<model>; the deployment sets MOCK_KV_EVENTS_TOPIC=kv@$(POD_IP):8000@$MODEL.
# An empty topic auto-builds kv@<engine_id>@<model>.
EVENT_ARGS=()
if [ "${MOCK_ENABLE_KV_EVENTS:-0}" = "1" ]; then
    EVENT_ARGS=(
        --enable-kv-cache-events
        --kv-events-endpoint "${MOCK_KV_EVENTS_ENDPOINT:-tcp://*:5556}"
        --kv-events-topic "${MOCK_KV_EVENTS_TOPIC:-}"
    )
fi

# Failure injection + context-length limit (Phase 5). All default to off.
FAILURE_ARGS=(
    --max-model-len "${MOCK_MAX_MODEL_LEN:-0}"
    --failure-injection-rate "${MOCK_FAILURE_INJECTION_RATE:-0.0}"
)
if [ -n "${MOCK_FAILURE_TYPES:-}" ]; then
    FAILURE_ARGS+=(--failure-types "${MOCK_FAILURE_TYPES}")
fi

# Decode sits behind the routing sidecar (sidecar :8000 -> vllm-rs :8200); prefill is
# hit directly on :8000. Override with VLLM_PORT if needed.
case "$ROLE" in
    decode) HTTP_PORT="${VLLM_PORT:-8200}" ;;
    *)      HTTP_PORT="${VLLM_PORT:-8000}" ;;
esac

echo "[entrypoint] role=$ROLE model=$MODEL engine_id=$ENGINE_ID http=:$HTTP_PORT nixl=$SIDE_CHANNEL_HOST:$SIDE_CHANNEL_PORT"

# Frontend: binds the engine-core handshake and waits for our engine to join.
vllm-rs serve "$MODEL" \
    --data-parallel-size 1 --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" \
    --host 0.0.0.0 --port "$HTTP_PORT" &
FRONTEND_PID=$!

# Engine: connects as the headless DP engine, plays the NixlConnector role.
inference-sim \
    --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --pd-role "$ROLE" \
    --engine-id "$ENGINE_ID" \
    --side-channel-host "$SIDE_CHANNEL_HOST" \
    --side-channel-port "$SIDE_CHANNEL_PORT" \
    "${LATENCY_ARGS[@]}" \
    "${SCHEDULER_ARGS[@]}" \
    "${EVENT_ARGS[@]}" \
    "${FAILURE_ARGS[@]}" \
    --log-requests &
ENGINE_PID=$!

terminate() { kill "$ENGINE_PID" "$FRONTEND_PID" 2>/dev/null || true; }
trap terminate TERM INT

# If either process exits, tear the pod down so k8s restarts it.
wait -n "$FRONTEND_PID" "$ENGINE_PID"
echo "[entrypoint] a process exited; shutting down"
terminate
wait || true
