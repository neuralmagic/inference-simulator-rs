#!/usr/bin/env bash
#
# Drive a tap-recorded latency capture against the trace-capture-h200 pod and pull
# both recordings: the tap trace (server-side: full per-token ITL arrays, arrival
# schedule, per-gap batch context) and the loadgen's client-side measurements from
# the SAME run, so the two can be compared request-for-request.
#
# Prereqs: deployment scaled to 1 and READY (just capture-up && just capture-status),
# kubecontext pointed at the right cluster, uv installed.
#
# The load generator is deploy/trace-capture/loadgen.py. guidellm 0.6.0 deadlocks
# pre-flight against the Rust frontend regardless of flags (see the memory notes in
# guidellm-concurrent-stall-workaround), so it is not used here.
set -euo pipefail

NS="${NS:-${NAMESPACE:-inference-sim}}"
DEPLOY="${DEPLOY:-trace-capture-h200}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-/tmp/trace-capture-h200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR"

echo "==> port-forwarding $DEPLOY :8000"
kubectl -n "$NS" port-forward "deploy/$DEPLOY" 8000:8000 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

curl -sf http://127.0.0.1:8000/v1/models >/dev/null || {
    echo "frontend not responding on :8000" >&2
    exit 1
}

run_bench() {
    local conc=$1 secs=$2
    echo "==> loadgen concurrency=$conc (${secs}s)"
    uv run --with httpx "$SCRIPT_DIR/loadgen.py" \
        --url http://127.0.0.1:8000 \
        --model "$MODEL" \
        --concurrency "$conc" \
        --duration "$secs" \
        --prompt-tokens 512 \
        --output-tokens 128 \
        --out "$OUT_DIR/c$conc-loadgen.json" \
        --trace-out "$OUT_DIR/client-trace.jsonl"
}

run_bench 1 60
run_bench 16 120

echo "==> fetching tap trace"
kubectl -n "$NS" exec "deploy/$DEPLOY" -c tap -- cat /trace/trace.jsonl \
    > "$OUT_DIR/tap-trace.jsonl"

echo "==> fetching step-stats sidecar (per-step SchedulerStats incl. spec_decoding_stats)"
kubectl -n "$NS" exec "deploy/$DEPLOY" -c tap -- cat /trace/step-stats.jsonl \
    > "$OUT_DIR/step-stats.jsonl" 2>/dev/null || \
    echo "  (no step-stats.jsonl; tap built without --step-stats-out support?)"

echo "==> done"
echo "  tap trace:    $OUT_DIR/tap-trace.jsonl ($(wc -l < "$OUT_DIR/tap-trace.jsonl" | tr -d ' ') lines)"
echo "  step stats:   $OUT_DIR/step-stats.jsonl"
echo "  client trace: $OUT_DIR/client-trace.jsonl"
echo "  client JSON:  $OUT_DIR/c1-loadgen.json $OUT_DIR/c16-loadgen.json"
echo
echo "Remember to scale back down: just capture-down"
