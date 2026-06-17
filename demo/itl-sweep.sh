#!/usr/bin/env bash
# Sweep replay pacing (tokens/sec) to find where the python frontend's
# streaming consumer stops keeping up: past that rate the frontend's
# RequestOutputCollector merges deltas and the qwen3_coder parser
# corrupts/drops tool calls, so the agent loop stalls.
#
# Mechanism: per point, rewrite the captured trace's itl_ms to a constant
# 1000/tps and reload it; the replay rig's existing --latency-trace pacing
# does the rest. The probe is the real agent: 8/8 turns + transcript
# identical to the GPU act-1 means the frontend kept up at that rate.
#
# Usage: demo/itl-sweep.sh [trace] [act1-transcript]
set -uo pipefail
NS="${NS:-${NAMESPACE:-inference-sim}}"
TRACE=${1:-/tmp/demo-trace.jsonl}
ACT1=${2:-/tmp/act1-transcript.txt}
RESULTS=/tmp/itl-sweep-results.txt

: > "$RESULTS"

live_pod() {
  kubectl -n $NS get pod -l llm-d.ai/guide=offline-replay -o json \
    | python3 -c "import sys,json; pods=[p['metadata']['name'] for p in json.load(sys.stdin)['items'] if not p['metadata'].get('deletionTimestamp')]; print(pods[0] if pods else '')"
}

make_variant() { # tps -> writes /tmp/sweep-trace.jsonl with constant per-token gaps
  python3 - "$1" "$TRACE" <<'EOF'
import json, sys
tps = float(sys.argv[1])
itl = 1000.0 / tps
out = []
for line in open(sys.argv[2]):
    d = json.loads(line)
    if "meta" in d:
        out.append(line.strip())
        continue
    n = len(d.get("itl_ms") or [])
    d["itl_ms"] = [itl] * n
    d["ttft_ms"] = 200.0
    out.append(json.dumps(d))
open("/tmp/sweep-trace.jsonl", "w").write("\n".join(out) + "\n")
EOF
}

run_point() {
  local tps=$1
  make_variant "$tps"
  kubectl -n $NS delete pod -l llm-d.ai/guide=offline-replay --wait=false >/dev/null 2>&1
  sleep 5
  local pod=""
  for _ in $(seq 1 60); do
    pod=$(live_pod)
    [ -n "$pod" ] && kubectl -n $NS get pod "$pod" -o jsonpath='{.status.containerStatuses[?(@.name=="sim")].state.running}' 2>/dev/null | grep -q startedAt && break
    sleep 3
  done
  kubectl -n $NS cp /tmp/sweep-trace.jsonl "$pod:/trace/trace.jsonl" -c sim >/dev/null 2>&1
  for _ in $(seq 1 60); do
    kubectl -n $NS get pod "$pod" -o jsonpath='{.status.containerStatuses[?(@.name=="frontend")].ready}' 2>/dev/null | grep -q true && break
    sleep 3
  done
  pkill -f "port-forward deploy/offline-replay" 2>/dev/null
  kubectl -n $NS port-forward deploy/offline-replay 8001:8000 >/dev/null 2>&1 &
  for _ in $(seq 1 30); do curl -s http://localhost:8001/v1/models >/dev/null 2>&1 && break; sleep 1; done

  local t0 t1 turns verdict="DIVERGED"
  t0=$(date +%s)
  bash demo/run-agent.sh http://localhost:8001 /tmp/sweep-ws > /tmp/sweep-transcript.txt 2>&1 || true
  t1=$(date +%s)
  turns=$(grep -oE "done:.* [0-9]+ turns" /tmp/sweep-transcript.txt | grep -oE "[0-9]+" | tail -1)
  if diff -q "$ACT1" /tmp/sweep-transcript.txt >/dev/null 2>&1; then verdict="IDENTICAL"; fi
  printf "%8s tok/s   turns=%-3s transcript=%-10s wall=%ss\n" "$tps" "${turns:-0}" "$verdict" "$((t1 - t0))" | tee -a "$RESULTS"
}

# Descend from comfortable to absurd; the H200 capture ran at ~10 tok/s.
for tps in 25 100 250 500 1000 2500 5000 10000 50000; do
  run_point "$tps"
done

echo; echo "=== sweep complete (probe: 8-turn agent replay) ==="; cat "$RESULTS"
