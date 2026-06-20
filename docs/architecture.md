# Architecture

The protocol boundary is reused from vLLM's in-tree `vllm-engine-core-client` crate
(pulled as a pinned git dependency):

```
            ZMQ + msgpack (engine-core protocol)
 vLLM frontend  ◀──────────────────────────────────▶  vllm-vcr play
 (Rust or Py)        handshake / ADD / ABORT / UTILITY        │
                                                              ▼
                                              ┌──────────────────────────────┐
                                              │ generation loop (sim tokens)  │
                                              │           │                   │
                                              │           ▼                   │
                                              │   KvDataPlane (the boundary)  │
                                              │   • Noop  (default)           │
                                              │   • NIXL  (feature = "nixl")  │
                                              └──────────────────────────────┘
```

- `connect_to_frontend` joins the frontend-owned handshake,
  reports ready, and opens the DEALER/PUSH sockets.
- `src/io.rs` decodes frames into `EngineInput` and pushes `EngineOutput` back.
- `src/engine.rs` is the generation loop (random tokens to `max_tokens`), with the
  two data-plane hooks marked `=== DATA PLANE ===`.
- `src/dataplane.rs` is the integration point: prefill **advertises** KV via
  `kv_transfer_params`; decode **pulls** it. `NoopDataPlane` performs no transfer;
  `NixlDataPlane` (behind the `nixl` feature) performs NIXL transfers.
