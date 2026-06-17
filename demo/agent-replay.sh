#!/usr/bin/env bash
# Agent replay, self-driving: live agent capture -> GPU scaled to zero ->
# byte-identical offline replay. Record with:
#   asciinema rec -c "bash demo/agent-replay.sh" demo/agent-replay.cast
#
# Pre-flight (off camera): capture rig 4/4 (just agentic-capture-up), replay
# rig deployed (just replay-up; sim waiting for a trace), port-forward :8000
# to the capture rig.
set -euo pipefail
NS="${NS:-${NAMESPACE:-inference-sim}}"
CYAN=$'\033[1;36m'; GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; RESET=$'\033[0m'

say()  { echo; echo "${CYAN}# $*${RESET}"; sleep 1.5; }
run()  { echo "${YELLOW}\$ $*${RESET}"; "$@"; }
# The pod for a selector, excluding terminating pods (recycles race .items[0]).
live_pod() {
  kubectl -n $NS get pod -l "$1" -o json \
    | python3 -c "import sys,json; pods=[p['metadata']['name'] for p in json.load(sys.stdin)['items'] if not p['metadata'].get('deletionTimestamp')]; print(pods[0] if pods else '')"
}

say "ACT 1: Claude Code -> vLLM /v1/messages -> recording tap -> H200 GPU"
run kubectl -n $NS get pods -l llm-d.ai/guide=trace-capture-agentic
# The tap trace accumulates for the pod's lifetime; checkpoint it so act 2
# replays exactly this take's capture (identical prompts from an earlier run
# would otherwise win the consume-once match in arrival order).
PRE_RECORDS=$(kubectl -n $NS exec deploy/trace-capture-h200-agentic -c tap -- sh -c 'cat /trace/trace.jsonl 2>/dev/null | wc -l' | tr -d ' ')
[ "$PRE_RECORDS" -lt 1 ] && PRE_RECORDS=1 # line 1 is always the tap's meta line
say "A real agent, a real model (Qwen3-Coder-30B), a real GPU. Watch the turns:"
echo "${YELLOW}\$ demo/run-agent.sh http://localhost:8000 /tmp/demo-live${RESET}"
bash demo/run-agent.sh http://localhost:8000 /tmp/demo-live | tee /tmp/act1-transcript.txt
say "The work product:"
run ls /tmp/demo-live
run python3 /tmp/demo-live/test_calculator.py
rm -rf /tmp/act1-ws && cp -r /tmp/demo-live /tmp/act1-ws

say "Every token of every turn was recorded on the wire by the tap:"
run just agentic-capture-fetch /tmp/demo-trace-full.jsonl
# Slice to this take: the meta line plus everything after the checkpoint.
{ head -1 /tmp/demo-trace-full.jsonl; tail -n +"$((PRE_RECORDS + 1))" /tmp/demo-trace-full.jsonl; } > /tmp/demo-trace.jsonl
python3 - <<'EOF'
import json
recs = [json.loads(l) for l in open('/tmp/demo-trace.jsonl') if not l.startswith('{"meta"')]
print(f"  {len(recs)} agent turns captured; per turn:")
for r in recs:
    print(f"    prompt={r['prompt_tokens']:>6} tokens  output={r['output_tokens']:>4} tokens (ids recorded)  finish={r['finish_reason']}")
EOF

say "Now the GPU goes away. Entirely."
run just agentic-capture-down
sleep 8
run kubectl -n $NS get pods -l llm-d.ai/guide=trace-capture-agentic
say "Load the cassette into the simulator (CPU frontend + inference-sim, no GPU):"
kubectl -n $NS delete pod -l llm-d.ai/guide=offline-replay --wait=false >/dev/null 2>&1 || true
sleep 3
until pod=$(live_pod llm-d.ai/guide=offline-replay) && [ -n "$pod" ] \
  && kubectl -n $NS get pod "$pod" -o jsonpath='{.status.containerStatuses[?(@.name=="sim")].state.running}' 2>/dev/null | grep -q startedAt; do
  sleep 3
done
run just replay-load-trace /tmp/demo-trace.jsonl
echo "  waiting for the replay rig to come up..."
until kubectl -n $NS get pod "$pod" -o jsonpath='{.status.containerStatuses[?(@.name=="frontend")].ready}' 2>/dev/null | grep -q true; do sleep 3; done
pkill -f "port-forward deploy/offline-replay" 2>/dev/null || true
kubectl -n $NS port-forward deploy/offline-replay 8001:8000 >/dev/null 2>&1 &
until curl -s http://localhost:8001/v1/models >/dev/null 2>&1; do sleep 1; done
run kubectl -n $NS get pods -l llm-d.ai/guide=offline-replay

say "ACT 2: the SAME agent command, against the simulator. Zero GPUs. Same turns:"
echo "${YELLOW}\$ demo/run-agent.sh http://localhost:8001 /tmp/demo-live${RESET}"
bash demo/run-agent.sh http://localhost:8001 /tmp/demo-live | tee /tmp/act2-transcript.txt

say "The verdict: the workspace AND the agent's words, byte for byte."
# __pycache__ bytecode embeds source mtimes; it always differs and proves nothing.
echo "${YELLOW}\$ diff -r --exclude=__pycache__ /tmp/act1-ws /tmp/demo-live${RESET}"
diff -r --exclude=__pycache__ /tmp/act1-ws /tmp/demo-live
echo "${YELLOW}\$ diff /tmp/act1-transcript.txt /tmp/act2-transcript.txt${RESET}"
if diff /tmp/act1-transcript.txt /tmp/act2-transcript.txt; then
  echo "${GREEN}IDENTICAL: every turn, every token, replayed byte-for-byte. Zero GPU, zero API spend.${RESET}"
else
  echo "DIVERGED (see sim logs for unmatched prefixes)"; exit 1
fi
sleep 3
