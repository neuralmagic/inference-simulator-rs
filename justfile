# vllm-vcr task runner.
#
# The capture workflow end to end:
#   just image-build && just image-push     # tap/frontend image (linux/amd64)
#   just capture-up                         # schedule the rig on the cluster
#   just capture-status                     # wait for "forwarding frames"
#   just capture-run                        # drive load, fetch trace + reports
#   just capture-down                       # release the GPU
#   just calibrate /tmp/trace-capture-h200/tap-trace.jsonl
#   just plots /tmp/trace-capture-h200/tap-trace.jsonl docs/images

image := env_var_or_default("IMAGE", "ghcr.io/neuralmagic/vllm-vcr:dev")
namespace := env_var_or_default("NAMESPACE", "inference-sim")
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
# Builds the compat.toml default line; use image-build-line for an older line.
image-build:
    podman build --platform linux/amd64 -t {{image}} .

# Build the capture image for a specific compat.toml line, e.g. `just image-build-line 0.22`.
# Pins Cargo.toml/Cargo.lock to the line's rev/fork, stamps VLLM_TARGET_VERSION for build.rs,
# and builds the vllm-rs frontend from the same source as the tap. Tags as <image>-vllm<line>.
# amd64 only; on Apple Silicon prefer a native remote builder over local emulation.
image-build-line line:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo xtask pin-vllm "{{line}}"
    cargo update -p vllm-engine-core-client
    eval "$(cargo xtask frontend-args "{{line}}")"
    podman build --platform linux/amd64 \
        --build-arg VLLM_TARGET_VERSION="$TAG" \
        --build-arg VLLM_REPO="$FREPO" \
        --build-arg VLLM_REF="$FREF" \
        -t {{image}}-vllm{{line}} .

# Push the capture image.
image-push:
    podman push {{image}}

# --- cluster capture rig ----------------------------------------------------------

# Apply the manifests via kustomize and scale the capture pod up (1x GPU; queues if none free).
capture-up:
    kustomize build deploy/trace-capture/overlays/{{namespace}} | kubectl apply -f -
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

# --- conformance captures (deploy/trace-capture/models.toml) -----------------------
# Kueue-serialized golden captures, one model x scenario per Job. See docs/conformance.md.

# Apply the one-GPU conformance queue (serializes captures; run once).
conformance-queue:
    kustomize build deploy/trace-capture/overlays/{{namespace}} | kubectl apply -f -

# List the capture targets defined in models.toml.
conformance-list:
    python3 deploy/trace-capture/gen-capture-jobs.py --list

# Submit conformance capture Job(s), e.g. `just conformance-capture qwen3-8b`. Ships the
# loadgen scripts as a configmap, then applies the generated Job (Kueue holds it until a
# GPU is free, so it's safe to submit several; they run one at a time).
conformance-capture +names:
    kubectl create configmap validation-scripts -n {{namespace}} \
        --from-file=loadgen.py=deploy/trace-capture/loadgen.py \
        --from-file=runner.sh=deploy/trace-capture/validation-runner.sh \
        --dry-run=client -o yaml | kubectl apply -f -
    python3 deploy/trace-capture/gen-capture-jobs.py {{names}} | kubectl apply -f -

# --- agentic capture + offline replay (docs/agentic-offline-replay.md) -------------

# Agentic capture rig: python frontend (/v1/messages) + tap + GPU engine.
agentic-capture-up:
    kustomize build deploy/trace-capture/overlays/{{namespace}} | kubectl apply -f -
    kubectl -n {{namespace}} scale deploy trace-capture-h200-agentic --replicas=1

agentic-capture-down:
    kubectl -n {{namespace}} scale deploy trace-capture-h200-agentic --replicas=0

agentic-capture-fetch out="/tmp/agentic-tap-trace.jsonl":
    kubectl -n {{namespace}} exec deploy/trace-capture-h200-agentic -c tap -- cat /trace/trace.jsonl > {{out}}
    @wc -l {{out}}

# Offline replay rig: python frontend + vllm-vcr play, zero GPU (then: replay-load-trace).
replay-up:
    kustomize build deploy/trace-capture/overlays/{{namespace}} | kubectl apply -f -

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
    cargo run --release --bin vllm-vcr -- inspect summarize {{trace}}

# Model-level calibration: source vs replay vs knob-fit, request-total gate.
calibrate trace:
    cargo run --release --bin vllm-vcr -- inspect calibrate {{trace}}

# Rebuild every README/deck figure from the committed traces (~30 min: the
# arrival replays run in real time). sim-comparison.png is the one exception;
# it needs live stacks (commands in the README).
figures out_dir="docs/images":
    bash scripts/make_figures.sh {{out_dir}}

# Calibration plots (ITL replay fidelity + mean-vs-per-token) into out_dir.
# The calibrate verdict may fail on real loaded traces: the TTFT marginal is
# engine-mechanical (queueing), gated wire-level by `just replay`, not here.
plots trace out_dir="docs/images":
    -cargo run --release --bin vllm-vcr -- inspect calibrate {{trace}} \
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
    cargo run --release --bin vllm-vcr -- inspect calibrate-e2e {{trace}} \
        --replay-arrivals --latency-trace {{latency_trace}} --tolerance {{tolerance}}

# Apply the rig with the engine's prefix cache DISABLED (counterfactual capture).
capture-up-nocache:
    kubectl apply -f deploy/trace-capture/h200-capture.yaml
    kubectl -n {{namespace}} patch deploy {{deploy}} --type=json \
        -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--no-enable-prefix-caching"}]'
    kubectl -n {{namespace}} scale deploy {{deploy}} --replicas=1

# Admission + pod + loadgen status for the conformance capture jobs.
conformance-status:
    kubectl -n {{namespace}} get workloads
    kubectl -n {{namespace}} get jobs -l llm-d.ai/guide=trace-capture
    kubectl -n {{namespace}} get pods -l llm-d.ai/guide=trace-capture
    -kubectl -n {{namespace}} logs -l llm-d.ai/guide=trace-capture -c loadgen --tail=2 --prefix

# Fetch a job's tap trace, optional step stats, and release it (job completes, GPU freed).
# e.g. `just conformance-fetch trace-qwen3-8b /tmp/qwen3-8b.jsonl`.
conformance-fetch job out:
    kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- cat /trace/trace.jsonl > {{out}}
    -kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- cat /trace/step-stats.jsonl > {{out}}.step-stats.jsonl
    -kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- sh -c 'for f in /trace/marker-*; do echo "$f=$(cat $f)"; done'
    kubectl -n {{namespace}} exec job/{{job}} -c loadgen -- touch /trace/fetched
    @wc -l {{out}}

# Delete all conformance capture jobs (frees quota immediately).
conformance-down:
    -kubectl -n {{namespace}} delete jobs -l llm-d.ai/guide=trace-capture
