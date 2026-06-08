# mock-engine-nixl — next-session plan

## Where we are (done, all committed)

A GPU-free, model-free stand-in for a real llm-d prefill/decode engine, **proven end to
end on `coreweave-waldorf`**: a request through the real EPP/router drives prefill→decode
and the decode pod pulls 128 KB of KV over **real NIXL/UCX, cross-pod, CPU**, pattern-verified.

- Mock vLLM engine-core backend behind the **real** vLLM Rust frontend (`vllm-rs`). Bird one.
- Real two-agent NIXL KV transfer (UCX, via NIXL listener + `fetch_remote_md`). Bird two.
- Wire-compat `kv_transfer_params` (real NixlConnector schema), driven per-request.
- Fedora image: minimal CPU libnixl (UCX-only) + UCX 1.21.x from source + `vllm-rs` + engine.
- Vendored llm-d v0.7.0 P/D manifests (`deploy/llm-d-pd/`), deployed via helmfile.
- Local gates: `scripts/e2e.sh`, `scripts/pd_control.sh`, `cargo test --features nixl` (Linux),
  `cargo check --features nixl-stub` (macOS).

## Known rough edges (the honest list)

1. **Mock addr-piggyback.** `RemoteKv.addr/len/pattern` ride in `kv_transfer_params`
   (`remote_buf_addr` etc.). This works *mock-to-mock*; a real vLLM peer carries the base
   address in its own `NixlAgentMetadata` ZMQ channel, so mixed real/mock P/D won't interop yet.
2. **Single 128 KB block.** `block_ids` is cosmetic; one buffer regardless of real KV dims.
   No `block_id -> addr` paged-cache mapping.
3. **`fetch_remote_md` per request** with a best-effort invalidate. Untested under many
   requests / multiple remote prefills / high concurrency.
4. **Single-arch `:amd64` tag.** No proper multi-arch manifest (quay aggregated same-tag
   pushes into a broken list — that's why we pinned `:amd64`).
5. **No latency model.** Prefill/decode are instant; the original "task 2" (port TTFT/ITL
   from `llm-d-inference-sim`) is still pending.
6. Pins: `nixl-sys` = ai-dynamo/nixl@41685d39, `vllm-engine-core-client`/`vllm-rs` = vllm@ba94a3b.
   Need a bump strategy as upstream moves.

## Next session — proposed order

1. **Lock in CI.** A workflow that builds the image (proper `buildx` multi-arch) and runs
   `cargo test --features nixl` (the loopback) + `scripts/pd_control.sh` in the amd64 image.
   This is the regression net before we extend anything.
2. **Latency model (task 2).** Port the TTFT/ITL model from `llm-d-inference-sim` so the
   P/D path is useful for scheduler/router behavior under realistic timing, not just plumbing.
   This is high-value and self-contained.
3. **Robustness pass on the data plane.** Multiple blocks + a real `block_id -> addr` mapping;
   abort/release paths; repeated-request and multi-remote `fetch_remote_md` handling;
   a concurrency test (N requests in flight through one pod).
4. **Scale + benchmark.** Bump prefill/decode replicas, run `inference-perf` (the guide's
   `20_1_isl_osl` template) against the EPP to show the P/D control+data plane under load,
   GPU-free. Great demo + finds races.
5. **(Stretch) True wire-compat with real vLLM peers.** Replace the addr-piggyback with
   vLLM's `NixlAgentMetadata` ZMQ side channel (`get_meta_msg` -> msgpack payload) so a real
   vLLM prefill can feed a mock decode (and vice versa). This is the big interop unlock;
   reference: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/{scheduler,worker,metadata}.py`.
6. **Upstreaming question.** `vllm-mock-engine` is already upstream; decide whether the
   NIXL-extended mock belongs in llm-d as a test fixture for the P/D well-lit path.

## Operational notes

- Cluster: `kubectl config use-context coreweave-waldorf`, namespace `llm-d-pd-mock`.
- Deploy: `helmfile -f deploy/llm-d-pd/helmfile.yaml apply`. Teardown: `... destroy` + delete ns.
- The cluster's pre-existing `pd-disaggregation-epp` (ns `llm-d-pd-disaggregation`) is broken
  (CrashLoopBackOff, 37d) — not ours; our fresh EPP runs fine.
- Image: `quay.io/wseaton/mock-engine-nixl:amd64` (public). Rebuild is QEMU-slow on the Mac;
  prefer a native amd64 builder (or the CI from step 1).
