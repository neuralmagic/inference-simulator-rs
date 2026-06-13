# inference-simulator-rs task runner.
#
# The capture workflow end to end:
#   just image-build && just image-push     # tap/frontend image (linux/amd64)
#   just capture-up                         # schedule the rig on the cluster
#   just capture-status                     # wait for "forwarding frames"
#   just capture-run                        # drive load, fetch trace + reports
#   just capture-down                       # release the GPU
#   just calibrate /tmp/trace-capture-h200/tap-trace.jsonl
#   just plots /tmp/trace-capture-h200/tap-trace.jsonl docs/images

image := "quay.io/wseaton/mock-engine-nixl:trace-capture-v8"
namespace := "weaton-dev"
deploy := "trace-capture-h200"

# List available recipes.
default:
    @just --list

# --- repo gates -------------------------------------------------------------------

# Format, lint, and run the full test suite.
check:
    cargo fmt --all
    cargo clippy --workspace --benches --tests --examples --all-features
    cargo test --workspace

# --- capture image ----------------------------------------------------------------

# Build the tap + vllm-rs image for the cluster (linux/amd64, slow under emulation).
image-build:
    podman build --platform linux/amd64 -t {{image}} .

# Push the capture image.
image-push:
    podman push {{image}}

# --- cluster capture rig ----------------------------------------------------------

# Apply the manifest and scale the capture pod up (1x GPU; queues if none free).
capture-up:
    kubectl apply -f deploy/trace-capture/h200-capture.yaml
    kubectl -n {{namespace}} scale deploy {{deploy}} --replicas=1

# Scale the capture pod down (always do this when finished).
capture-down:
    kubectl -n {{namespace}} scale deploy {{deploy}} --replicas=0

# Pod, container, and tap status at a glance.
capture-status:
    kubectl -n {{namespace}} get pods -l llm-d.ai/guide=trace-capture
    -kubectl -n {{namespace}} logs deploy/{{deploy}} -c tap --tail=3
    -kubectl -n {{namespace}} logs deploy/{{deploy}} -c engine --tail=3

# Drive the benchmark load and fetch the tap trace + client-side reports.
capture-run:
    bash deploy/trace-capture/run-capture.sh

# Fetch the tap trace without running load (e.g. mid-capture peek).
capture-fetch out="/tmp/tap-trace.jsonl":
    kubectl -n {{namespace}} exec deploy/{{deploy}} -c tap -- cat /trace/trace.jsonl > {{out}}
    @wc -l {{out}}

# --- agentic capture + offline replay (docs/agentic-offline-replay.md) -------------

# Agentic capture rig: python frontend (/v1/messages) + tap + GPU engine.
agentic-capture-up:
    kubectl apply -f deploy/trace-capture/h200-capture-agentic.yaml
    kubectl -n {{namespace}} scale deploy trace-capture-h200-agentic --replicas=1

agentic-capture-down:
    kubectl -n {{namespace}} scale deploy trace-capture-h200-agentic --replicas=0

agentic-capture-fetch out="/tmp/agentic-tap-trace.jsonl":
    kubectl -n {{namespace}} exec deploy/trace-capture-h200-agentic -c tap -- cat /trace/trace.jsonl > {{out}}
    @wc -l {{out}}

# Offline replay rig: python frontend + inference-sim, zero GPU (then: replay-load-trace).
replay-up:
    kubectl apply -f deploy/trace-capture/offline-replay.yaml

replay-down:
    kubectl -n {{namespace}} scale deploy offline-replay --replicas=0

# The sim waits for exactly /trace/trace.jsonl[.gz]; normalize the name on copy.
# Pod selection skips terminating pods (a recycle races .items[0]).
replay-load-trace trace:
    #!/usr/bin/env bash
    set -euo pipefail
    pod=$(kubectl -n {{namespace}} get pod -l llm-d.ai/guide=offline-replay -o json \
      | python3 -c "import sys,json; pods=[p['metadata']['name'] for p in json.load(sys.stdin)['items'] if not p['metadata'].get('deletionTimestamp')]; print(pods[0] if pods else '')")
    [ -n "$pod" ] || { echo "no live offline-replay pod"; exit 1; }
    dest=/trace/trace.jsonl
    [[ "{{trace}}" == *.gz ]] && dest=/trace/trace.jsonl.gz
    kubectl -n {{namespace}} cp {{trace}} "$pod:$dest" -c sim
    echo "loaded {{trace}} -> $pod:$dest"

# --- analysis ---------------------------------------------------------------------

# Summarize a trace (per-concurrency TTFT/ITL quantiles).
summarize trace:
    cargo run --release --bin inference-sim-trace -- summarize {{trace}}

# Model-level calibration: source vs replay vs knob-fit, request-total gate.
calibrate trace:
    cargo run --release --bin inference-sim-trace -- calibrate {{trace}}

# Rebuild every README/deck figure from the committed traces (~30 min: the
# arrival replays run in real time). sim-comparison.png is the one exception;
# it needs live stacks (commands in the README).
figures out_dir="docs/images":
    bash scripts/make_figures.sh {{out_dir}}

# Calibration plots (ITL replay fidelity + mean-vs-per-token) into out_dir.
# The calibrate verdict may fail on real loaded traces: the TTFT marginal is
# engine-mechanical (queueing), gated wire-level by `just replay`, not here.
plots trace out_dir="docs/images":
    -cargo run --release --bin inference-sim-trace -- calibrate {{trace}} \
        --dump-samples /tmp/calib-samples.json
    uv run scripts/plot_calibration.py --samples /tmp/calib-samples.json \
        --trace {{trace}} --out-dir {{out_dir}}

# Three-way survival comparison from labeled traces, e.g.:
#   just compare "real=tap.jsonl" "replay=ours.jsonl" "knobs=gosim.jsonl"
compare +labeled_traces:
    uv run scripts/plot_calibration.py \
        {{ replace_regex(labeled_traces, '(\S+)', '--compare $1') }} \
        --out-dir /tmp/sim-compare

# Open-loop arrival replay against a captured schedule (runs in real time).
replay trace latency_trace tolerance="0.10":
    cargo run --release --bin inference-sim-trace -- calibrate-e2e {{trace}} \
        --replay-arrivals --latency-trace {{latency_trace}} --tolerance {{tolerance}}

# Apply the rig with the engine's prefix cache DISABLED (counterfactual capture).
capture-up-nocache:
    kubectl apply -f deploy/trace-capture/h200-capture.yaml
    kubectl -n {{namespace}} patch deploy {{deploy}} --type=json \
        -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--no-enable-prefix-caching"}]'
    kubectl -n {{namespace}} scale deploy {{deploy}} --replicas=1

# --- kueue validation jobs ----------------------------------------------------------

# Queue the counterfactual-validation capture jobs (cached + nocache) on Kueue.
validation-up:
    kubectl create configmap validation-scripts -n {{namespace}} \
        --from-file=loadgen.py=deploy/trace-capture/loadgen.py \
        --from-file=runner.sh=deploy/trace-capture/validation-runner.sh \
        --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f deploy/trace-capture/validation-jobs.yaml

# Admission + pod + loadgen status for the validation jobs.
validation-status:
    kubectl -n {{namespace}} get workloads
    kubectl -n {{namespace}} get jobs -l llm-d.ai/guide=trace-capture
    kubectl -n {{namespace}} get pods -l llm-d.ai/guide=trace-capture
    -kubectl -n {{namespace}} logs -l llm-d.ai/guide=trace-capture -c loadgen --tail=2 --prefix

# Fetch a job's tap trace and release it (job completes, GPU freed).
validation-fetch job out:
    kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- cat /trace/trace.jsonl > {{out}}
    -kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- sh -c 'for f in /trace/marker-*; do echo "$f=$(cat $f)"; done'
    kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- touch /trace/fetched
    @wc -l {{out}}

# Delete the validation jobs (frees quota immediately).
validation-down:
    -kubectl -n {{namespace}} delete -f deploy/trace-capture/validation-jobs.yaml
