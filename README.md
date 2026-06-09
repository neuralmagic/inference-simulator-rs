# inference-simulator-rs

A mock vLLM **V1 engine-core** backend that speaks the real ZMQ + msgpack protocol,
with a prefill/decode **KV data plane over NIXL** so you can exercise real-ish P/D
flows, including actual byte movement over CPU RDMA / shared memory, **without a GPU
or a model**.

## Why

`llm-d-inference-sim` today fakes prefill/decode purely in the control plane: it
adjusts the latency model and tags a finish reason, but no KV cache bytes ever move.
This project chases two birds with one stone:

1. **Faithful frontend testing.** Instead of reimplementing the OpenAI API surface,
   sit behind vLLM's *real* frontend (the in-tree Rust frontend, or the Python one)
   as a drop-in engine. The frontend does tokenization, chat templates, tool calling,
   streaming, all the edge cases, for free. We only fake the expensive part: the model.
2. **A real P/D data path.** The same fake engine moves real simulated-KV bytes
   between a prefill instance and a decode instance over [NIXL](https://github.com/ai-dynamo/nixl)
   (UCX backend: DRAM-to-DRAM or real RDMA NICs). No CUDA, no GPU.

## How it works

The whole protocol boundary is reused from vLLM's in-tree `vllm-engine-core-client`
crate (pulled as a pinned git dependency), so the wire format never drifts from
upstream:

```
            ZMQ + msgpack (real engine-core protocol)
 vLLM frontend  ◀──────────────────────────────────▶  inference-simulator-rs
 (Rust or Py)        handshake / ADD / ABORT / UTILITY        │
                                                              ▼
                                              ┌──────────────────────────────┐
                                              │ generation loop (fake tokens) │
                                              │           │                   │
                                              │           ▼                   │
                                              │   KvDataPlane (the boundary)  │
                                              │   • Noop  (default)           │
                                              │   • NIXL  (feature = "nixl")  │
                                              └──────────────────────────────┘
```

- `connect_to_frontend` (from the reused crate) joins the frontend-owned handshake,
  reports ready, and opens the DEALER/PUSH sockets.
- `src/io.rs` decodes frames into `EngineInput` and pushes `EngineOutput` back.
- `src/engine.rs` is the generation loop (random tokens to `max_tokens`), with the
  two data-plane hooks marked `=== DATA PLANE ===`.
- `src/dataplane.rs` is the integration point: prefill **advertises** KV via
  `kv_transfer_params`; decode **pulls** it. `NoopDataPlane` is byte-for-byte today's
  sim; `NixlDataPlane` (behind the `nixl` feature) is where real NIXL transfers land.

## Status

- **Bird one (protocol):** proven end-to-end. The real vLLM Rust frontend serves
  streaming and non-streaming OpenAI completions through this backend over the genuine
  ZMQ/msgpack protocol, real tokenizer/detokenizer and chat template included, no GPU,
  no model weights, no NIXL. Run it: `./scripts/e2e.sh`.
- **Wire-compat control plane:** done. The engine produces/consumes the real vLLM
  NixlConnector `kv_transfer_params` schema (`do_remote_prefill`/`do_remote_decode`,
  `remote_engine_id`/`remote_host`/`remote_port`/`remote_block_ids`/`remote_request_id`/
  `tp_size`/`remote_num_tokens`), driven per-request. Proven against a routing-sidecar
  emulation (`scripts/pd_control.sh`), no NIXL required.
- **Bird two (NIXL data plane):** the transfer mechanic is implemented and tested
  in-process (`tests/nixl_loopback.rs`: register → NIXL READ → verify, Linux + libnixl).
  Cross-pod pull over the ZMQ metadata side channel (the `get_meta_msg` handshake
  serving `NixlAgentMetadata`) is the remaining increment; the addressing it needs
  (`remote_host:remote_port:remote_engine_id`) is already produced/consumed.

  ```bash
  ./scripts/pd_control.sh              # macOS: control-plane schema round trip
  cargo check --features nixl-stub     # macOS gate: typecheck the NIXL path
  cargo test  --features nixl          # Linux: real NIXL transfer
  ```

## Test

```bash
./scripts/e2e.sh        # boots vllm-rs + this engine, asserts streaming + non-streaming flows
./scripts/e2e_lora.sh   # loads a LoRA adapter, asserts vllm:lora_requests_info names it
```

Needs the `vllm-rs` frontend built once (`cargo build --bin vllm-rs` in the vLLM
`rust/` workspace); override its path with `FRONTEND_BIN=...`. First run fetches the
tokenizer from HF.

`e2e_lora.sh` needs a `vllm-rs` built from the LoRA-gauge fork (`wseaton/vllm@lora-info-gauge`):
the upstream Rust frontend at `ba94a3b` does not export `vllm:lora_requests_info`, though the
engine and the Python frontend both speak it. The container image already builds `vllm-rs` from
that fork by default (`VLLM_REPO`/`VLLM_REF` Docker build args); point them back at
`vllm-project/vllm` once the gauge is upstream.

### LoRA simulation

The engine tracks LoRA adapters the frontend loads (`add_lora`/`remove_lora`), counts per-adapter
running/waiting requests for `vllm:lora_requests_info`, and honors `--max-loras` (distinct
adapters allowed in the running batch; `0` = no cap). In the image, set `MOCK_MAX_LORAS`.

## Build & run

Default build (bird one, no NIXL, runs anywhere):

```bash
cargo run -- --handshake-address tcp://127.0.0.1:29550 --log-requests
```

Then point a real frontend at the same handshake address (see vLLM's
`rust/src/mock-engine/README.md` for the `vllm-rs serve` / `vllm serve` invocations;
this binary is a drop-in for `vllm-mock-engine`).

Smoke request once a frontend is up:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"max_tokens":16,"stream":true}'
```

### NIXL data plane

The NIXL path needs `libnixl` + UCX installed (Linux; RDMA NICs or shared memory).
On a box without it, typecheck against stubs:

```bash
cargo check --features nixl-stub
```

On Linux with NIXL installed, split a prefill and a decode engine:

```bash
cargo run --features nixl -- --pd-role prefill ...
cargo run --features nixl -- --pd-role decode  ...
```

## Hacking the engine

The engine is split along a trait boundary so you can swap behaviors without
touching the core loop or the ZMQ transport.

**`EngineCore` (src/engine_core.rs)** is the top-level contract. The generic
`run_loop` owns the tokio `select!` over inputs, internal events, and deadline
ticks. Any struct implementing `EngineCore` plugs in unchanged. `SimEngine` is
the production implementation; `ConstantEngine` (test-only, same file) proves a
from-scratch engine reuses the loop with zero duplication.

**Three strategy traits on `SimEngine`** control its behavior without subclassing:

| Trait | File | Default | What it controls |
|---|---|---|---|
| `TokenSource` | `src/tokens.rs` | `RandomTokens` | Which token ids each request emits. `EchoTokens` replays the prompt. |
| `LatencyModel` | `src/latency.rs` | `KnobLatency` | TTFT and inter-token pacing. `FixedLatency` gives constant delays with no rng draws. |
| `Scheduler` | `src/sched.rs` | `Fcfs` | Waiting-queue admission order. `Priority` uses `(priority, arrival_time)`. `ShortestPromptFirst` picks the smallest prompt. |

Defaults are wired in `SimEngine::new` (from CLI flags) and in `run()`, so
nothing changes without opting in.

**The contract tests** live in `tests/engine_core_e2e.rs`. They drive the full
stack (real ZMQ, real protocol framing, real channels) and assert wire-level
behavior. If your change breaks those tests, the wire protocol regressed.
Unit tests in `src/engine.rs` cover engine internals at a finer grain.

## Dependencies of note

- `vllm-engine-core-client` — pinned git dep on `vllm-project/vllm` (`rev` in
  `Cargo.toml`). Bump the rev to track upstream protocol changes.
- `nixl-sys` — pinned git dep on `ai-dynamo/nixl` (`rev` in `Cargo.toml`), the same
  source the image builds `libnixl` from, so the crate resolves identically on macOS
  (stub) and in the container (real lib).

The binary is `inference-sim`; the k8s deployment lives in `deploy/llm-d-pd/`.
