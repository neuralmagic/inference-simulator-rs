#!/usr/bin/env bash
#
# Live interop smoke test for the KV-cache event publisher: start the Rust emitter, then
# decode its messages with the REAL llm-d-kv-cache consumer (the Go harness in
# scripts/kv_events_smoke/). Proves our wire format is router-compatible end to end over a
# real ZMQ transport, not just by our own round-trip.
#
#   KV_CACHE_DIR=~/git/llm-d/llm-d-kv-cache ./scripts/kv_events_smoke.sh
#
# Needs Go (with network/module-cache access to go-zeromq/zmq4) and a local checkout of
# llm-d-kv-cache. The Rust-only gate (`cargo test --test kv_events_pubsub`) needs neither.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="$REPO_ROOT/scripts/kv_events_smoke"
KV_CACHE_DIR="${KV_CACHE_DIR:-$HOME/git/llm-d/llm-d-kv-cache}"
ENDPOINT_BIND="${ENDPOINT_BIND:-tcp://*:5556}"
ENDPOINT_DIAL="${ENDPOINT_DIAL:-tcp://127.0.0.1:5556}"
TOPIC="${TOPIC:-kv@127.0.0.1:8000@mock-model}"

if [[ ! -d "$KV_CACHE_DIR" ]]; then
    echo "error: llm-d-kv-cache checkout not found at $KV_CACHE_DIR (set KV_CACHE_DIR)" >&2
    exit 1
fi

echo "[smoke] building Rust emitter"
cargo build --example kv_event_emitter --manifest-path "$REPO_ROOT/Cargo.toml"

echo "[smoke] starting emitter on $ENDPOINT_BIND topic=$TOPIC"
"$REPO_ROOT/target/debug/examples/kv_event_emitter" "$ENDPOINT_BIND" "$TOPIC" &
EMITTER_PID=$!
trap 'kill "$EMITTER_PID" 2>/dev/null || true' EXIT
sleep 1 # let the PUB socket bind

echo "[smoke] pointing harness at local llm-d-kv-cache: $KV_CACHE_DIR"
( cd "$HARNESS_DIR" && go mod edit -replace "github.com/llm-d/llm-d-kv-cache=$KV_CACHE_DIR" && go mod tidy )

echo "[smoke] running real-consumer decode harness"
( cd "$HARNESS_DIR" && KV_ENDPOINT="$ENDPOINT_DIAL" go run . )

echo "[smoke] done"
