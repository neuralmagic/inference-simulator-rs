# inference-simulator-rs — next-session plan

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

1. ~~**Mock addr-piggyback.**~~ FIXED (Phase 4): the pool base + NIXL agent md travel over a
   mock-owned TCP metadata side channel (`PoolDescriptor`), the verify pattern is derived from
   `remote_request_id`, and `kv_transfer_params` carries only real vLLM `remote_*` fields.
   Byte-exact real-vLLM interop (vLLM's versioned `NixlAgentMetadata` v4) is still not attempted.
2. ~~**Single 128 KB block.**~~ FIXED (Phase 4): one registered KV pool, real per-block
   `block_id -> addr` paging (`pool_base + id*block_bytes`), multi-descriptor reads, block ids
   shared with the prefix cache + events.
3. **Metadata side channel per peer** (`load_remote_md`, cached by engine id). Untested under
   many requests / multiple remote prefills / high concurrency; peer cache has no invalidation.
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

## Feature roadmap (vs `llm-d-inference-sim`)

The Go `llm-d-inference-sim` is a self-contained HTTP server that fakes the **entire**
OpenAI surface in the control plane (no real frontend, no real KV bytes). We are the
opposite: the **real** vLLM engine-core behind the **real** vLLM frontend, faking only
the model, plus a **real NIXL KV data path** the Go sim doesn't have. That flips the
analysis: most of the Go feature list is stuff the frontend already gives us, so the
roadmap is only the handful of things that live below the frontend, at the engine layer.

### Free from the real frontend — do NOT port

OpenAI API surface (`/v1/chat/completions`, `/v1/completions`, `/v1/models`),
tokenization + chat templates, **real** tool-call parsing, logprobs decoding, streaming,
`/health`/`/ready`, the `/metrics` endpoint + full `vllm:*` Prometheus registration,
`X-Request-Id` headers, SSL/TLS. The metrics crate already registers
`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`,
prefix-cache hits, and the TTFT/ITL histograms, and populates them from `SchedulerStats`
+ `PrefillStats` carried on `EngineCoreOutputs`. So we don't reimplement metrics at all,
we just *fill in the stats* and they light up.

### Drop / never carry

`fake-metrics` (we get real metrics from real load), hand-rolled tool-call generation,
logprobs synthesis, the dataset/echo text machinery (the frontend detokenizes our token
IDs; realistic *text* is low-value for a load/timing/P-D testbed), LoRA metrics, the
SSE/HTTP error-body formatting.

### Build — engine-layer, frontend can't fake it

- **Phase 0 — CI regression net** (the "Lock in CI" item below): prerequisite.
- **Phase 1 — Latency model** ✅ DONE. Port `latencies.go`: TTFT + std-dev, ITL +
  std-dev (normal-truncated), token-count prefill (`prefill-overhead + (prompt-cached) *
  prefill-time-per-token`), KV-transfer latency used *instead of* TTFT when
  `do_remote_prefill` (hooks our real NIXL pull), and `time-factor-under-load`. Replaced
  the `yield_now()` tick with deadline-driven `sleep_until`. The frontend measures TTFT/ITL
  from when we emit, so this is also what makes the metrics realistic. (`src/latency.rs`)
- **Phase 2 — Scheduler stats** ✅ DONE. Populate `scheduler_stats` (`num_running_reqs`,
  `num_waiting_reqs`, `kv_cache_usage`) and per-request `prefill_stats` on outputs.
  Validated live on waldorf: the `vllm:*` gauges/histograms move under load.
- **Phase 3 — Scheduler model** ✅ DONE. Config surface + behavior matched to vLLM exactly
  (verified against `vllm/v1/core/sched/scheduler.py` + `request_queue.py` + `arg_utils.py`):
  - `--max-num-seqs` (default 128) caps the running batch.
  - `--max-num-batched-tokens` (default 2048): per-step token budget. Batch token demand
    (1/decoding req + each prefilling req's prompt chunk) can't exceed it, so prefill
    admission is throttled under load even with free seq slots. Budget frees when a prefill
    becomes a decode, then `schedule()` admits more.
  - `--long-prefill-token-threshold` (default 0 = off): caps a single prefill's per-step chunk.
  - `--scheduling-policy` (`fcfs` default | `priority`): waiting-queue order; `priority` admits
    smallest `(priority, arrival_time)` first, matching vLLM's priority queue.
  - The waiting queue is **unbounded** (`num_requests_waiting`); **vLLM never rejects on queue
    length**, so neither do we (the invented `max-waiting-queue-length` was removed).
  Modeling note: a prefilling request holds its budget demand for its whole TTFT window rather
  than literally chunking across forward passes; the Phase-1 latency formula still sets per-request
  TTFT once admitted. Full step-by-step chunked prefill would be the deeper-fidelity follow-up.
- **Phase 4 — KV-cache + prefix-cache model** ✅ DONE. A unified `BlockPool`
  (`src/blockpool.rs`, mirroring vLLM's `block_pool.py`) does prefix-hit accounting (feeds
  `prefix_cache_stats` + `num_local_cached_tokens`), physical block-slot allocation with
  ref-count pinning + LRU eviction, and KV-cache event generation. The publisher
  (`src/kvevents.rs`) emits `BlockStored`/`BlockRemoved`/`AllBlocksCleared` over a real ZMQ
  PUB socket, wire-compatible with vLLM's `ZmqEventPublisher` and the llm-d router's consumer
  (3-frame `[topic, seq, msgpack]`, tagged arrays, 8-byte big-endian hashes). The data plane
  is now paged: one registered KV pool, `addr = pool_base + block_id*bytes`, multi-descriptor
  reads, real `block_ids` shared with the events (retires the single-128KB-block hack,
  rough-edge #2). Verified end to end against the **real** `llm-d-kv-cache` Go decoder
  (`scripts/kv_events_smoke.sh`) plus a live Rust PUB→SUB test (`tests/kv_events_pubsub.rs`).
  The addr-piggyback (rough-edge #1) is also retired: the pool base address + NIXL agent
  metadata now travel over a minimal mock-owned TCP metadata side channel (`PoolDescriptor`
  via `get_local_md`/`load_remote_md`), and the verify pattern is derived from
  `remote_request_id` on both sides, so `kv_transfer_params` carries only real vLLM `remote_*`
  fields. Byte-exact interop with a real vLLM peer (matching vLLM's versioned `NixlAgentMetadata`
  v4 schema) is deliberately *not* attempted — that version-locked clone is the remaining gap.
- **Phase 5 — Failure injection** ✅ DONE. `--failure-injection-rate` + `--failure-types`
  (error/length/repetition) roll a per-request finish on arrival; `--max-model-len` fails
  `prompt + max_tokens > limit` with a length error. (`src/engine.rs::maybe_fail`)

Priority: **Phase 1 + Phase 2 together** is the high-leverage move (one focused session,
self-contained) and converts the project from "plumbing proven" to "GPU-free P/D testbed
with realistic timing and real vLLM metrics". Phase 4 is the most llm-d-specific value
but the heaviest lift.

## Operational notes

- Cluster: `kubectl config use-context coreweave-waldorf`, namespace `llm-d-pd-mock`.
- Deploy: `helmfile -f deploy/llm-d-pd/helmfile.yaml apply`. Teardown: `... destroy` + delete ns.
- The cluster's pre-existing `pd-disaggregation-epp` (ns `llm-d-pd-disaggregation`) is broken
  (CrashLoopBackOff, 37d) — not ours; our fresh EPP runs fine.
- Image: `quay.io/wseaton/mock-engine-nixl:amd64` (public). Rebuild is QEMU-slow on the Mac;
  prefer a native amd64 builder (or the CI from step 1).
