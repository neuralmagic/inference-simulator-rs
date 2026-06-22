#!/usr/bin/env bash
#
# Rebuild every README/deck figure from local trace files listed in
# traces/README.md. Run it as `just figures`.
#
# The arrival replays run in REAL TIME (each one spins the actual simulator
# and replays a captured schedule wall-clock), so the full set takes ~30
# minutes. The only figure NOT built here is sim-comparison.png, which needs
# two live serving stacks plus a load generator; its commands are in the
# README's "Comparison with llm-d-inference-sim" section.
#
# Replay legs must mirror each capture's engine config and latency fit
# exactly, or the comparison measures config drift instead of model fidelity.
# Those pairings live here and in traces/README.md; change them together.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${1:-docs/images}"
H8=traces/h200-qwen3-8b
LS=traces/local-sim
WORK="$(mktemp -d)"
FIT="$WORK/fit.jsonl"
PLOT=(uv run scripts/plot_calibration.py)

cargo build --release -q
BIN=target/release/vllm-vcr

# The H200 capture rig's scheduler/cache config, mirrored by the step-model
# replays (the local-sim captures ran engine defaults instead).
H200_SIM=(--sim-arg=--max-num-seqs --sim-arg=1024
          --sim-arg=--max-num-batched-tokens --sim-arg=8192
          --sim-arg=--max-model-len --sim-arg=16384
          --sim-arg=--kv-cache-size --sim-arg=65536)

# Canonical fitting trace (one meta line): the concurrency sweep, a warm
# multiturn capture, and a cold multiturn capture (cold donors). All three
# matter: dropping the warm capture breaks the warm cohorts, dropping the
# cold one softens cold chunk costs and the cache-off what-if with them.
{ cat "$H8/h200-sweep-full.jsonl"
  grep -v '"meta"' "$H8/h200-multiturn-mtfit2.jsonl"
  grep -v '"meta"' "$H8/h200-multiturn-nocache4.jsonl"; } > "$FIT"

# replay NAME TAP [extra calibrate-e2e flags...] -> $WORK/NAME.jsonl
replay() {
    local name=$1 tap=$2
    shift 2
    echo "==> replay $name ($tap)"
    # The verdict gates at 10% on worst-quantile cells; small-n tails can
    # exceed that while medians and totals agree, so don't abort the build.
    "$BIN" inspect calibrate-e2e "$tap" --replay-arrivals --latency-trace "$FIT" \
        --dump-trace "$WORK/$name.jsonl" "$@" || true
}

echo "==> in-sample ITL fidelity + per-token-vs-mean (model level)"
"$BIN" inspect calibrate "$H8/h200-qwen3-tap-trace.jsonl" --dump-samples "$WORK/samples.json" || true
"${PLOT[@]}" --samples "$WORK/samples.json" \
    --trace "$H8/h200-qwen3-tap-trace.jsonl" --out-dir "$OUT"

# --- step-model gates: real H200 multiturn schedules ------------------------
replay factual "$H8/h200-multiturn-mtfit3.jsonl" --replay-sessions "${H200_SIM[@]}"
replay counterfactual "$H8/h200-multiturn-nocache.jsonl" --replay-sessions --cold-prompts "${H200_SIM[@]}"
replay lowrate "$H8/h200-multiturn-nocachelo.jsonl" --replay-sessions --cold-prompts "${H200_SIM[@]}"

"${PLOT[@]}" --compare "captured (real H200, warm multiturn)=$H8/h200-multiturn-mtfit3.jsonl" \
    --compare "modeled replay=$WORK/factual.jsonl" \
    --compare-out step-model-factual.png --out-dir "$OUT"
"${PLOT[@]}" --compare "captured (real H200, cold multiturn)=$H8/h200-multiturn-nocache.jsonl" \
    --compare "modeled replay=$WORK/counterfactual.jsonl" \
    --compare-out step-model-counterfactual.png --out-dir "$OUT"
"${PLOT[@]}" --compare "captured (real H200, low-rate cold)=$H8/h200-multiturn-nocachelo.jsonl" \
    --compare "modeled replay=$WORK/lowrate.jsonl" \
    --compare-out step-model-lowrate.png --out-dir "$OUT"

# --- workload scenarios: local-sim stack captures, engine defaults ----------
replay poisson "$LS/local-sim-poisson-tap.jsonl"
replay burst "$LS/local-sim-burst-tap.jsonl"
replay multiturn "$LS/local-sim-multiturn-tap.jsonl" --replay-sessions \
    --sim-arg=--kv-cache-size --sim-arg=65536
replay multiturn-cold "$LS/local-sim-multiturn-tap.jsonl" --replay-sessions --cold-prompts \
    --sim-arg=--kv-cache-size --sim-arg=65536

"${PLOT[@]}" --compare "captured (live stack run)=$LS/local-sim-poisson-tap.jsonl" \
    --compare "modeled (schedule replay)=$WORK/poisson.jsonl" \
    --compare-out replay-arrivals-poisson.png --out-dir "$OUT"
"${PLOT[@]}" --compare "captured (live stack run)=$LS/local-sim-burst-tap.jsonl" \
    --compare "modeled (schedule replay)=$WORK/burst.jsonl" \
    --compare-out replay-arrivals-burst.png --out-dir "$OUT"
"${PLOT[@]}" --cache-effect "real=$LS/local-sim-multiturn-tap.jsonl" \
    --cache-effect "replay=$WORK/multiturn.jsonl" \
    --cache-effect "nocache=$WORK/multiturn-cold.jsonl" \
    --out-dir "$OUT"

echo "wrote figures to $OUT (replay dumps kept in $WORK)"
echo "not built: sim-comparison.png (needs live stacks; see README)"
