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

## Trace replay and calibration

The `inference-sim-trace` binary includes a calibration harness that proves two
claims about the latency models:

1. **TraceLatency replay reproduces source trace quantiles** within tolerance.
2. **KnobLatency structurally cannot reproduce heavy tails**: its `[0.3*mean, 1.7*mean]`
   clamp caps p99/p50 at roughly 1.7x for any knob settings.

Three-command demo:

```bash
# 1. Generate a synthetic heavy-tailed trace (lognormal TTFT/ITL).
cargo run --bin inference-sim-trace -- gen-demo -o /tmp/demo.jsonl

# 2. Model-level calibration (no transport, fast, exact).
cargo run --bin inference-sim-trace -- calibrate /tmp/demo.jsonl

# 3. Wire-level proof: spin the real simulator and measure client-side.
cargo run --bin inference-sim-trace -- calibrate-e2e /tmp/demo.jsonl --requests 60
```

Example verdict from the demo trace (seed 0, 200 records, 100k samples):

```
=== Verdict ===
Pooled p99/p50 ratios:
  source   TTFT=11.507  ITL=6.299
  replay   TTFT=11.997  ITL=6.308
  knob-fit TTFT=1.700  ITL=1.700

Replay max relative error: TTFT=0.0419  ITL=0.0046
Replay PASS (tolerance 0.10): PASS
Knob-fit tail capped at ~1.7x by construction (source TTFT ratio 11.507 > 1.75, knob-fit 1.700 <= 1.75)
```

Use `--fast` on `gen-demo` to produce a small-magnitude trace suitable for
quick e2e testing (TTFT ~15-40ms, ITL ~3-10ms). All subcommands accept `--json`
for machine-readable output and `--seed` for determinism.

### Calibration against a real engine

Both figures below come from one capture: the recording tap (`inference-sim-tap`)
sitting between the vLLM Rust frontend and a real headless vLLM engine
(Qwen3-8B, TP=1, H200), recording per-token inter-token gaps server-side over
in-pod localhost ZMQ. 1230 requests at concurrency 1 and 16, 118k token gaps.

Source vs `TraceLatency` replay vs best-fit `KnobLatency`, as survival curves
and Q-Q plots. Replay max relative error on this trace: TTFT 0.47%, ITL 0.21%.
The knob model's `[0.3*mean, 1.7*mean]` clamp is visible as a vertical cutoff
in both survival plots.

![Source vs replay vs knob-fit](docs/images/replay-fidelity.png)

The same trace viewed two ways: pooled per-token ITLs against per-request mean
ITLs (what client-side benchmark reports expose, e.g. guidellm, which records
only first/last token timestamps). On this run, 2.1% of all tokens are slower
than the largest per-request mean observed.

![Per-token vs per-request-mean ITL](docs/images/mean-vs-pertoken.png)

To regenerate from any trace with per-token `itl_ms` arrays:

```bash
cargo run --bin inference-sim-trace -- calibrate trace.jsonl --dump-samples samples.json
uv run scripts/plot_calibration.py --samples samples.json --trace trace.jsonl --out-dir docs/images
```

### Comparison with llm-d-inference-sim

Same workload (`deploy/trace-capture/loadgen.py`, concurrency 1 and 16, 512/128
tokens) against three targets: the real H200 engine (tap-recorded), this
simulator replaying the H200 trace, and the Go
[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) (v0.9.1)
with its latency knobs fit to the same trace (TTFT 132ms, ITL 11ms). Both
simulators ran natively on the same host, measured client-side by the same
load generator; the real-engine curves are the tap recording.

Fitting note: the trace's actual std-devs (TTFT 80ms, ITL 8ms) exceed
llm-d-inference-sim's config validation, which caps std-dev at 30% of the
mean, so it runs with the largest spread it accepts (39ms / 3.3ms).

![Real engine vs both simulators](docs/images/sim-comparison.png)

```bash
# llm-d-inference-sim invocation used above
llm-d-inference-sim --port 8001 --model Qwen/Qwen3-8B --mode random \
  --force-dummy-tokenizer --max-model-len 16384 --max-num-seqs 128 \
  --time-to-first-token 132ms --time-to-first-token-std-dev 39ms \
  --inter-token-latency 11ms --inter-token-latency-std-dev 3300us

# this simulator: vllm-rs frontend + trace replay, vLLM-default scheduler limits
inference-sim --handshake-address tcp://127.0.0.1:5571 \
  --latency-trace h200-qwen3-tap-trace.jsonl \
  --max-num-seqs 1024 --max-num-batched-tokens 8192
```

### Workload scenarios: open-loop arrival replay

The calibrations above sample the latency model closed-loop, which proves the
*distributions* are right but never stresses the reactive path: TTFT queueing,
prefill stalls, and concurrency mixing only emerge when an arrival process
drives the scheduler. `calibrate-e2e --replay-arrivals` validates that path:
it replays a captured arrival schedule in real time (each request sent at its
recorded offset, open loop, regardless of how earlier requests are going) and
compares client-side TTFT/ITL/request-total quantiles against the capture.
`--latency-trace` fits the sim's model from a *different* trace, so the gate
runs on an arrival process the model was never fitted on.

Setup: the same frontend → tap → engine stack as the capture rig, run locally
with `inference-sim` as the engine, its latency model fit from the original
H200 capture (fixed-concurrency closed loop). `deploy/trace-capture/loadgen.py
--pattern poisson|burst` drives arrival processes that capture never
contained, the tap records ground truth, and the replay must reproduce it:

| scenario                          | requests | concurrency seen | TTFT max err | ITL max err | req-total err | gate (10%) |
|-----------------------------------|----------|------------------|--------------|-------------|---------------|------------|
| poisson, 4 req/s                  | 483      | 1-22, median 9   | 9.3%         | 1.4%        | 3.8%          | PASS       |
| burst, 24 per 10s                 | 288      | 0 -> 24 spikes   | 2.8%         | 3.8%        | 8.0%          | PASS       |
| multiturn agentic (see below)     | 580      | 1-25             | 3.4%         | 0.5%        | 1.5%          | PASS       |

The burst scenario is the harsher test: each burst floods an idle engine, so
TTFT is dominated by queueing the latency model never saw as such (source p99
1.46s vs the fitting trace's ~250ms). The sim's scheduler recreates the queue
and the pooled distributions land on top of each other:

![Burst arrival replay](docs/images/replay-arrivals-burst.png)

![Poisson arrival replay](docs/images/replay-arrivals-poisson.png)

Per-concurrency-bucket rows shuffle under bursts (admission order inside a
burst is not deterministic, so per-request concurrency labels move), which is
why the gate compares pooled quantiles plus per-request decode totals.

Replayed prompts are unique-token synthetics: the captured workloads carry
`cached_tokens: 0`, and identical fill tokens would silently turn every
replayed request into a prefix-cache hit (TTFT collapses into the smallest
uncached-prompt bucket; this was a real bug). Workloads with genuine prefix
reuse (multiturn/agentic) need the prefix structure replayed too, which is the
next scenario.

To reproduce against any trace with `arrival_ms`:

```bash
# capture: any OpenAI-compatible target
uv run --with httpx deploy/trace-capture/loadgen.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --pattern poisson --rate 4 --duration 120 \
  --prompt-tokens 512 --output-tokens 128 --out run.json --trace-out client.jsonl

# replay the schedule, fitting the model from a different capture
just replay tap-poisson.jsonl h200-qwen3-tap-trace-v2.jsonl

# real-vs-replay survival curves (replay measurements via --dump-trace)
just compare "real=tap-poisson.jsonl" "replay=replay-measured.jsonl"
```

### Agentic multiturn and the prefix cache

The agentic scenario (`loadgen.py --pattern multiturn`): sessions arrive
poisson at `--rate`, each runs `--turns` closed-loop turns whose context grows
by the turn's prompt plus the model's response, on top of one of
`--prefix-count` shared `--prefix-tokens` prefixes. The validation run below
is 116 sessions x 5 turns over two ~10k-token shared prefixes; 578 of 580
requests were prefix-cache hits (warm median 10.6k of ~11k prompt tokens
cached).

Prefix caching is not a latency knob in this simulator. The engine runs a real
block-pool prefix cache; admission computes each request's actual cached-token
count, the trace-fitted TTFT model conditions on the *uncached* prompt size,
and a prefill admission stalls concurrent decodes by its uncached tokens only.
The perf gain emerges from workload structure, which is why replaying it needs
the workload's sharing structure: the tap fingerprints every prompt with
chained per-block hashes (`block_hashes`, mooncake-style), and the replay
expands each distinct hash to one deterministic token block, so replayed
prompts share prefixes exactly where the captured ones did and the sim's cache
reacts the same way.

Two replay modes matter here. Pure open-loop replay of this trace fails the
TTFT gate (16% err, concentrated in ramp-up): the capture's intra-session
closed loop backs off when the engine runs momentarily slow, open-loop
arrivals do not, and the divergence compounds. `--replay-sessions` restores
the generator's semantics (turn N+1 fires when turn N completes plus the
recorded think gap; sessions are inferred from the hash chains) and passes at
TTFT 3.4% / ITL 0.5% / request-totals 1.5%.

The shape proof, per turn cohort rather than pooled (compensating errors
would show here): real vs replay TTFT survival for turn-1 requests (shared
prefix hit only) and turns 2+ (the session's growing context), plus the same
schedule replayed with `--cold-prompts` (prefix reuse defeated, the cache-off
what-if). Without the cache, re-prefilling ~11k tokens per turn collapses the
queue: TTFT p50 goes from 0.37s to ~18s and p99 from 1.6s to ~29s.

![Multiturn cache effect](docs/images/multiturn-cache-effect.png)

```bash
# capture an agentic workload (10k-token shared prefixes at ~1.5 tokens/word)
uv run --with httpx deploy/trace-capture/loadgen.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --pattern multiturn --rate 1 --turns 5 \
  --prefix-tokens 6500 --prompt-tokens 128 --output-tokens 128 --duration 120 \
  --out run.json

# session-paced replay (the gate), then the cache-off what-if
cargo run --release --bin inference-sim-trace -- calibrate-e2e tap-multiturn.jsonl \
  --replay-arrivals --replay-sessions --latency-trace h200-qwen3-tap-trace-v2.jsonl \
  --sim-arg=--kv-cache-size --sim-arg=65536 --dump-trace replay-measured.jsonl
cargo run --release --bin inference-sim-trace -- calibrate-e2e tap-multiturn.jsonl \
  --replay-arrivals --replay-sessions --cold-prompts ... --dump-trace nocache-measured.jsonl

# the per-cohort figure
uv run scripts/plot_calibration.py --cache-effect real=tap-multiturn.jsonl \
  --cache-effect replay=replay-measured.jsonl --cache-effect nocache=nocache-measured.jsonl \
  --out-dir docs/images
```

Caveat for absolute cold numbers: the H200 capture's prompts top out around
800 uncached tokens, so a cold 11k prefill's park time extrapolates from the
nearest bucket; the ~50x what-if delta is dominated by mechanically simulated
queueing, but a prompt-length-sweep capture would pin it down. Mooncake-style
workload traces (block-hash ids + lengths + timestamps) map directly onto this
schema for replaying production workloads.
