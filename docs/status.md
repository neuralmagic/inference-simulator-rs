# Status

`vllm-vcr` is usable today for protocol-level frontend testing, trace replay,
calibration, and GPU-free prefill/decode control-plane experiments. The NIXL data
plane is implemented behind an optional feature and needs a Linux host with libnixl
and UCX.

| Area | State | Validation |
| --- | --- | --- |
| Engine-core protocol | Streaming and non-streaming OpenAI flows work through the vLLM Rust frontend over ZMQ/msgpack, with tokenizer, detokenizer, chat template, and frontend metrics intact. | `./scripts/e2e.sh` |
| Trace timing | TTFT, inter-token gaps, multi-token chunks, prefix-cache structure, and arrival/session pacing can be captured, modeled, and replayed. | `inspect calibrate`, `inspect calibrate-e2e`, trace replay tests |
| Content replay | `record --record-tokens` plus `play --replay-tokens` can serve recorded token ids and finish reasons. | `tests/engine_core_e2e.rs`, `tests/closed_loop_prefix_replay.rs` |
| P/D control plane | The simulator produces and consumes vLLM NixlConnector `kv_transfer_params` per request. | `scripts/pd_control.sh` |
| NIXL data plane | Prefill registers a paged KV pool and serves metadata; decode fetches metadata and posts paged NIXL reads. | `tests/nixl_loopback.rs` on Linux + libnixl |
| Multi-version support | The build matrix pins one `vllm-engine-core-client` rev per supported line and uses conformance goldens when available. | CI matrix + `tests/conformance.rs` |

If NIXL initialization fails at runtime, the engine logs a warning and falls back to
`NoopDataPlane`, so protocol tests can still run.

```bash
./scripts/pd_control.sh              # macOS: control-plane schema round trip
cargo check --features nixl-stub     # macOS gate: typecheck the NIXL path
cargo test  --features nixl          # Linux: NIXL transfer
```
