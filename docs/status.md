# Status

- **Protocol:** implemented end-to-end. The vLLM Rust frontend serves streaming and
  non-streaming OpenAI completions through this backend over the ZMQ/msgpack
  protocol, with tokenizer/detokenizer and chat template, no GPU, no model
  weights, no NIXL. Run `./scripts/e2e.sh`.
- **Control plane (wire-compat):** the engine produces and consumes the vLLM
  NixlConnector `kv_transfer_params` schema
  (`do_remote_prefill`/`do_remote_decode`, `remote_engine_id`/`remote_host`/
  `remote_port`/`remote_block_ids`/`remote_request_id`/`tp_size`/`remote_num_tokens`),
  driven per-request. Exercised against a routing-sidecar emulation
  (`scripts/pd_control.sh`), no NIXL required.
- **NIXL data plane:** the implemented path registers a paged KV pool on prefill,
  serves a `PoolDescriptor` over a TCP metadata side channel, lets decode load the
  remote metadata, and posts paged NIXL READs. `tests/nixl_loopback.rs` covers this
  with distinct prefill/decode agents in one process over loopback (Linux + libnixl).
  If NIXL initialization fails, the runtime logs a warning and falls back to
  `NoopDataPlane`.

```bash
./scripts/pd_control.sh              # macOS: control-plane schema round trip
cargo check --features nixl-stub     # macOS gate: typecheck the NIXL path
cargo test  --features nixl          # Linux: NIXL transfer
```
