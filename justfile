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

image := "quay.io/wseaton/mock-engine-nixl:trace-capture-v2"
namespace := "weaton-dev"
deploy := "trace-capture-h200"

# List available recipes.
default:
    @just --list

# --- repo gates -------------------------------------------------------------------

# Format, lint, and run the full test suite.
check:
    cargo fmt
    cargo clippy --all --benches --tests --examples --all-features
    cargo test

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

# --- analysis ---------------------------------------------------------------------

# Summarize a trace (per-concurrency TTFT/ITL quantiles).
summarize trace:
    cargo run --release --bin inference-sim-trace -- summarize {{trace}}

# Model-level calibration: source vs replay vs knob-fit, request-total gate.
calibrate trace:
    cargo run --release --bin inference-sim-trace -- calibrate {{trace}}

# Calibration plots (replay fidelity + mean-vs-per-token) into out_dir.
plots trace out_dir="docs/images":
    cargo run --release --bin inference-sim-trace -- calibrate {{trace}} \
        --dump-samples /tmp/calib-samples.json
    uv run scripts/plot_calibration.py --samples /tmp/calib-samples.json \
        --trace {{trace}} --out-dir {{out_dir}}

# Three-way survival comparison from labeled traces, e.g.:
#   just compare "real=tap.jsonl" "replay=ours.jsonl" "knobs=gosim.jsonl"
compare +labeled_traces:
    uv run scripts/plot_calibration.py \
        {{ replace_regex(labeled_traces, '(\S+)', '--compare $1') }} \
        --out-dir /tmp/sim-compare
