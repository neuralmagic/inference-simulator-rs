# mock-engine-nixl

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
 vLLM frontend  ◀──────────────────────────────────▶  mock-engine-nixl
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
- **Bird two (NIXL data plane):** implemented. A prefill engine fabricates and
  registers fake KV, advertises `{agent_md, block descriptors, pattern}` as
  `kv_transfer_params`; a decode engine loads the remote metadata, posts a NIXL READ
  over UCX, polls to completion, and verifies the bytes. Proven by the loopback test
  (`tests/nixl_loopback.rs`). Runs on Linux with real libnixl+UCX; on macOS only the
  typecheck gate runs (the bindings link `-lstdc++`, which macOS lacks):

  ```bash
  cargo check --features nixl-stub     # macOS gate: typechecks the NIXL path
  cargo test  --features nixl          # Linux: actually moves bytes + verifies
  ```

  Still open: plumbing `kv_transfer_params` from a prefill output into a decode
  request across processes (today `extract_kv_params` returns `None`, so the live
  engine's decode-pull hook is dormant; the transfer mechanics are proven via the test).

## Test

```bash
./scripts/e2e.sh   # boots vllm-rs + this engine, asserts streaming + non-streaming flows
```

Needs the `vllm-rs` frontend built once (`cargo build --bin vllm-rs` in the vLLM
`rust/` workspace); override its path with `FRONTEND_BIN=...`. First run fetches the
tokenizer from HF.

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

## Dependencies of note

- `vllm-engine-core-client` — pinned git dep on `vllm-project/vllm` (`rev` in
  `Cargo.toml`). Bump the rev to track upstream protocol changes.
- `nixl-sys` — local **path** dependency on the nixl checkout
  (`/Users/weaton/git/nixl/src/bindings/rust`). Adjust the path for your machine, or
  repoint at the upstream repo.
