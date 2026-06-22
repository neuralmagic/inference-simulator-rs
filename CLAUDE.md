# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vllm-vcr` is a mock vLLM V1 engine-core backend that records, replays, and inspects vLLM engine-core traces. It speaks the real ZMQ + msgpack protocol and can operate as a GPU-free vLLM engine for testing. The binary has three subcommands:
- `record` - taps a live vLLM frontend ↔ engine-core link and writes JSONL traces
- `play` - runs a mock engine-core backend replaying traces or simulating from a latency model
- `inspect` - converts benchmark reports, summarizes traces, renders Perfetto timelines, runs calibration

## Build and Test Commands

### Basic Development
```bash
# Build the project
cargo build --release

# Run all tests
cargo test --workspace

# Format and lint (required before commits)
cargo fmt --all
cargo clippy --workspace --benches --tests --examples --all-features

# Full check (what CI runs)
just check
```

### Running the Binary
```bash
# Install locally
cargo install --path . --locked

# Run mock engine
vllm-vcr play --handshake-address tcp://127.0.0.1:29550 --log-requests

# Record a trace
vllm-vcr record --frontend-handshake tcp://127.0.0.1:29550 --engine-handshake tcp://127.0.0.1:29551 --out trace.jsonl

# Inspect/summarize a trace
vllm-vcr inspect summarize trace.jsonl

# Calibrate a trace
vllm-vcr inspect calibrate trace.jsonl
```

### Testing Specific Components
```bash
# Library tests only
cargo test --workspace --lib

# Specific integration test
cargo test --test engine_core_e2e -- --nocapture
cargo test --test tap_e2e -- --nocapture
cargo test --test conformance -- --nocapture
```

### NIXL Feature (requires libnixl + UCX)
```bash
# Build with real NIXL KV data plane (Linux only)
cargo build --features nixl

# Build with NIXL stubs for local development (macOS)
cargo build --features nixl-stub
```

## Architecture

### Workspace Structure
This is a Cargo workspace with member crates under `crates/`:
- **`sim-trace`** - trace schema, latency models, Perfetto rendering, guidellm converter (vLLM-free)
- **`sim-protocol`** - engine-core protocol glue, frontend handshake, wire conversions
- **`sim-tap`** - transparent ZMQ proxy for recording (used by `vllm-vcr record`)
- **`sim-compat`** - vLLM version compatibility manifest parser
- **`sim-s3`** - S3 trace I/O using AWS SDK
- **`xtask`** - repo automation tasks (`cargo xtask <cmd>`)

### Core Components
The main binary is in `src/` and `src/main.rs` dispatches to three subcommands:
- **`src/record.rs`** - recording tap front-end
- **`src/inspect/`** - trace inspection subcommands
- **`src/lib.rs`** exports the `play` backend and re-exports `sim-trace` and `sim-protocol` under original paths for backward compatibility

### Engine Architecture
The engine separates orchestration from request behavior:
- **`src/engine_core.rs`** - `EngineCore` trait and generic `run_loop` (tokio select over inputs/events/deadlines)
- **`src/engine.rs`** - `SimEngine` production implementation with three strategy traits:
  - `TokenSource` (`src/tokens.rs`) - which token IDs to emit (RandomTokens, EchoTokens)
  - `LatencyModel` (`crates/sim-trace/src/latency.rs`) - TTFT and inter-token pacing (KnobLatency, FixedLatency)
  - `Scheduler` (`src/sched.rs`) - admission order (Fcfs, Priority, ShortestPromptFirst)
- **`src/io.rs`** - decodes incoming ZMQ frames to `EngineInput`, encodes `EngineOutput` back to frontend
- **`src/dataplane.rs`** - prefill/decode KV transfer integration point (NoopDataPlane by default, NixlDataPlane with `nixl` feature)

### vLLM Version Compatibility
The project maintains a rolling support window for multiple vLLM versions defined in `compat.toml`:
- Each `[[vllm]]` entry specifies `line`, `tag`, `protocol_rev` (vllm.git commit)
- One line has `default = true` (currently v0.23.0)
- The `nightly` line tracks vLLM main to catch wire drift early
- `build.rs` stamps the default line's tag as `VLLM_TARGET_VERSION`
- Lines may require `patch_repo`/`patch_rev` for forks with backported fixes

### Key Patterns
- Protocol types come from vLLM's in-tree `vllm-engine-core-client` crate, pinned per supported line
- Frontend handshake and ready response in `sim-protocol::frontend_connect`
- Contract tests (`tests/engine_core_e2e.rs`) drive ZMQ + protocol framing end-to-end
- Unit tests in `src/engine.rs` cover engine internals

## Deployment

Trace capture deployments use **Kustomize** for multi-cluster configuration. Manifests are in `deploy/trace-capture/`:
- **`base/`** - Cluster-agnostic manifests (no hardcoded namespace/serviceAccount)
- **`overlays/inference-sim/`** - Cluster-specific configuration (namespace: `inference-sim`, serviceAccount: `modelexpress-p2p-nvidia-gpu-vllm-sa`)

To deploy to a different cluster, copy `overlays/inference-sim/` and update `namespace` + `serviceAccountName`. See `deploy/trace-capture/README.md` for details.

## Justfile Recipes

The project uses `just` for common workflows (see `justfile`). All cluster recipes use `kustomize build` under the hood.

### Cluster Capture
```bash
# Build and push capture image
just image-build && just image-push

# Deploy capture rig (1 GPU)
just capture-up

# Check status
just capture-status

# Run benchmark and fetch trace
just capture-run

# Cleanup (always do this)
just capture-down
```

### Conformance Testing
```bash
# List conformance capture targets
just conformance-list

# Submit conformance jobs
just conformance-capture qwen3-8b

# Check status
just conformance-status

# Fetch results
just conformance-fetch trace-qwen3-8b /tmp/qwen3-8b.jsonl

# Cleanup
just conformance-down
```

### Analysis
```bash
# Summarize trace
just summarize trace.jsonl

# Calibrate (model-level calibration)
just calibrate trace.jsonl

# Generate calibration plots
just plots trace.jsonl docs/images

# Compare multiple traces
just compare "real=tap.jsonl" "replay=ours.jsonl" "knobs=gosim.jsonl"
```

## Important Notes

- Rust version: 1.85+ (edition 2024, resolver "3" for MSRV-aware resolution)
- The in-tree `vllm-engine-core-client` dependency is a git dependency pinned to a specific vLLM commit
- CI runs format, clippy, workspace tests, and per-vLLM-line conformance suite
- Logs go to stderr (INFO default) so `inspect` stdout stays clean for piping
- The project is dual-licensed under Apache-2.0 or MIT
