# inference-simulator-rs

A vLLM **V1 engine-core** simulator that speaks the ZMQ + msgpack protocol.
It can run behind a vLLM frontend without model weights or a GPU. With the
`nixl` feature and a working libnixl/UCX runtime, it also moves simulated
KV-cache bytes over NIXL for prefill/decode testing.

## Table of contents

- [Purpose](#purpose)
- [Architecture](#architecture)
- [Status](#status)
- [Install](#install)
- [Quick start](#quick-start)
- [Testing](#testing)
- [LoRA simulation](#lora-simulation)
- [NIXL data plane](#nixl-data-plane)
- [Engine internals](#engine-internals)
- [Trace replay and calibration](#trace-replay-and-calibration)
  - [Concepts](#concepts)
  - [Calibration demo](#calibration-demo)
  - [Calibration with engine captures](#calibration-with-engine-captures)
  - [Open-loop arrival replay](#open-loop-arrival-replay)
  - [Prefix cache and agentic multiturn](#prefix-cache-and-agentic-multiturn)
  - [Content-identical replay](#content-identical-replay)
  - [Replay pacing](#replay-pacing)
  - [Speculative decoding and diffusion](#speculative-decoding-and-diffusion)
  - [Visualizing traces (Perfetto)](#visualizing-traces-perfetto)
- [Dependencies of note](#dependencies-of-note)
- [License](#license)

## Purpose

`llm-d-inference-sim` models prefill/decode in the control plane: it adjusts
latency and finish metadata, but it does not move KV-cache bytes. This simulator
adds two capabilities:

1. **Frontend compatibility.** It runs behind vLLM's Rust or Python frontend. The
   frontend still handles tokenization, chat templates, tool calling, streaming, and
   OpenAI-compatible request handling. The simulator replaces only the model backend.
2. **Prefill/decode data-plane testing.** It can move simulated KV-cache bytes
   between prefill and decode instances over [NIXL](https://github.com/ai-dynamo/nixl)
   using the UCX backend when the NIXL runtime initializes. CUDA and model weights
   are not required.

## Architecture

The protocol boundary is reused from vLLM's in-tree `vllm-engine-core-client` crate
(pulled as a pinned git dependency):

```
            ZMQ + msgpack (engine-core protocol)
 vLLM frontend  ◀──────────────────────────────────▶  inference-simulator-rs
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

## Status

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

## Install

Requires Rust 1.85 or newer. From a checkout:

```bash
cargo install --path . --locked
```

That installs `inference-sim` and `inference-sim-trace`. After the repository is
public, the same default no-NIXL build can be installed from Git:

```bash
cargo install --git https://github.com/neuralmagic/inference-simulator-rs \
  --locked inference-simulator-rs
```

For a NIXL-enabled install, build on Linux with `libnixl` and UCX available:

```bash
cargo install --path . --locked --features nixl
```

For the Kubernetes deployment, build the container image instead:

```bash
podman build -t ghcr.io/neuralmagic/inference-simulator-rs:dev .
```

## Quick start

### Protocol-only local run

Start the simulator:

```bash
inference-sim --handshake-address tcp://127.0.0.1:29550 --log-requests
```

Start a vLLM frontend with the same handshake address. See vLLM's
`rust/src/mock-engine/README.md` for the `vllm-rs serve` / `vllm serve`
invocations; this binary uses the same handshake role as `vllm-mock-engine`.

Send a streaming chat request once the frontend is up:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"max_tokens":16,"stream":true}'
```

### Prefill/decode control-plane smoke

Run the routing-sidecar schema round trip without NIXL:

```bash
./scripts/pd_control.sh
```

For the Kubernetes P/D deployment, build the image from the install section and use
[deploy/llm-d-pd/README.md](deploy/llm-d-pd/README.md).

## Testing

```bash
./scripts/e2e.sh        # boots vllm-rs + this engine, asserts streaming + non-streaming flows
./scripts/e2e_lora.sh   # loads a LoRA adapter, asserts vllm:lora_requests_info names it
```

Needs the `vllm-rs` frontend built once (`cargo build --bin vllm-rs` in the vLLM
`rust/` workspace); override its path with `FRONTEND_BIN=...`. First run fetches the
tokenizer from HF.

`e2e_lora.sh` needs a `vllm-rs` at or past vLLM #45030, which exports
`vllm:lora_requests_info` from the frontend (the engine no longer reports
per-adapter maps in `SchedulerStats`). The pinned commit in `Cargo.toml`/`Dockerfile`
qualifies.

## LoRA simulation

The engine tracks LoRA adapters the frontend loads (`add_lora`/`remove_lora`) and
honors `--max-loras` (distinct adapters allowed in the running batch; `0` = no cap).
In the image, set `MOCK_MAX_LORAS`. The `vllm:lora_requests_info` gauge is
frontend-derived as of vLLM #45030.

## NIXL data plane

The NIXL path needs `libnixl` + UCX installed (Linux; RDMA NICs or shared memory).
On a box without it, typecheck against stubs:

```bash
cargo check --features nixl-stub
```

On Linux with NIXL installed, split a prefill and a decode engine:

```bash
# prefill
cargo run --features nixl -- --pd-role prefill \
  --engine-id mock-prefill --side-channel-host 127.0.0.1 --side-channel-port 5600 ...

# decode
cargo run --features nixl -- --pd-role decode \
  --engine-id mock-decode --side-channel-port 5601 ...
```

The transfer path uses `remote_host`/`remote_port` from `kv_transfer_params` to fetch
the prefill's `PoolDescriptor` over TCP, then issues NIXL READs for the advertised
block ids. Decode receives those `remote_*` fields per request; the prefill address is
not a decode CLI argument. The loopback test validates the byte-transfer path;
Kubernetes deployment validation is separate.

## Engine internals

The engine separates loop orchestration from request behavior.

**`EngineCore` (src/engine_core.rs)** is the top-level contract. The generic
`run_loop` owns the tokio `select!` over inputs, internal events, and deadline
ticks. Any struct implementing `EngineCore` can use the loop. `SimEngine` is the
production implementation; `ConstantEngine` (test-only, same file) is a minimal
engine used by loop tests.

**Three strategy traits on `SimEngine`** control request behavior:

| Trait | File | Default | What it controls |
|---|---|---|---|
| `TokenSource` | `src/tokens.rs` | `RandomTokens` | Which token ids each request emits. `EchoTokens` replays the prompt. |
| `LatencyModel` | `crates/sim-trace/src/latency.rs` | `KnobLatency` | TTFT and inter-token pacing. `FixedLatency` gives constant delays with no rng draws. |
| `Scheduler` | `src/sched.rs` | `Fcfs` | Waiting-queue admission order. `Priority` uses `(priority, arrival_time)`. `ShortestPromptFirst` picks the smallest prompt. |

Defaults are wired in `SimEngine::new` (from CLI flags) and in `run()`.

**Contract tests** live in `tests/engine_core_e2e.rs`. They drive ZMQ, protocol
framing, and channels, then assert wire-level behavior. Unit tests in
`src/engine.rs` cover engine internals.

## Trace replay and calibration

Captured traces live under `traces/` (gitignored; see
[traces/README.md](traces/README.md) for the inventory and which captures are fitting
vs gate seeds).

### Concepts

This section uses three terms:

- **Captured** — per-token tap recordings from a vLLM engine, taken server-side on
  the engine-core protocol. Figures label these as "real" or "source".
- **Modeled** — latency the simulator emits. TTFT and per-token gaps are drawn from a
  statistical model fitted to a captured trace (conditioned on concurrency, context
  depth, and uncached prompt size). Captured timings are not played back verbatim,
  so a model fitted on one workload can be evaluated on another.
- **Direct replay** — recorded values used verbatim, no statistics: arrival
  timestamps (`--replay-arrivals`), session pacing (`--replay-sessions`), prefix
  structure (block hashes), and opt-in output token ids (`--replay-tokens`).

"Replay" in a figure or flag name refers to the workload side (the schedule being
replayed), not to the timing. Counterfactual gates fit on workload A, directly replay
workload B's schedule, and check the modeled timing against B's capture.

`just figures` rebuilds the figures from local trace files listed in
[traces/README.md](traces/README.md) (`scripts/make_figures.sh`; ~30 minutes, the
arrival replays run in real time). Those trace files are not committed. The
head-to-head comparison is the exception; it needs live serving stacks (commands in
that section).

### Calibration demo

The `inference-sim-trace` binary includes a calibration harness that checks two
properties of the latency models:

1. `TraceLatency` replay reproduces source-trace quantiles within tolerance.
2. `KnobLatency` cannot reproduce heavy tails: its `[0.3*mean, 1.7*mean]` clamp caps
   p99/p50 at roughly 1.7x for any knob settings.

This model-level check applies to ITL and to TTFT on unloaded traces. On loaded
captures, the TTFT marginal comes from queueing and chunk interference rather than a
sampled distribution, so this check can fail by design. Loaded TTFT is checked by the
arrival-replay scenarios below.

```bash
# 1. Generate a synthetic heavy-tailed trace (lognormal TTFT/ITL).
cargo run --bin inference-sim-trace -- gen-demo -o /tmp/demo.jsonl

# 2. Model-level calibration (no transport).
cargo run --bin inference-sim-trace -- calibrate /tmp/demo.jsonl

# 3. Wire-level: start the simulator and measure client-side.
cargo run --bin inference-sim-trace -- calibrate-e2e /tmp/demo.jsonl --requests 60
```

`--fast` on `gen-demo` produces a small-magnitude trace for quick e2e testing
(TTFT ~15-40ms, ITL ~3-10ms). All subcommands accept `--json` for machine-readable
output and `--seed` for determinism.

### Calibration with engine captures

The recording tap (`inference-sim-tap`, deployment manifests in
[deploy/trace-capture/](deploy/trace-capture/)) sits between the
vLLM Rust frontend and a headless vLLM engine (Qwen3-8B, TP=1, H200), recording
per-token inter-token gaps server-side over in-pod localhost ZMQ.

The figures below plot captured vs `TraceLatency` vs best-fit `KnobLatency` per-token
ITL (survival curve and Q-Q plot), and the same trace as pooled per-token ITLs vs
per-request mean ITLs. Client-side benchmark reports such as guidellm usually expose
per-request means because they record first/last token timestamps. The knob model's
`[0.3*mean, 1.7*mean]` clamp appears as a vertical cutoff before the captured tail.

![Source vs replay vs knob-fit](docs/images/replay-fidelity.png)

![Per-token vs per-request-mean ITL](docs/images/mean-vs-pertoken.png)

To regenerate from any trace with per-token `itl_ms` arrays:

```bash
cargo run --bin inference-sim-trace -- calibrate trace.jsonl --dump-samples samples.json
uv run scripts/plot_calibration.py --samples samples.json --trace trace.jsonl --out-dir docs/images
```

#### Comparison with llm-d-inference-sim

Same workload (`deploy/trace-capture/loadgen.py`, concurrency 1 and 16, 512/128
tokens) against three targets: the H200 engine (tap-recorded), this simulator with its
latency model fit from the canonical fitting set (a different workload, the
counterfactual setting), and the Go
[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) (v0.9.1) with its
latency knobs fit to the same trace (the in-sample setting). Both simulators ran on the
same host and were measured client-side by the same load generator. The engine curves
are the tap recording. Both simulators' timing is modeled.

![Real engine vs both simulators](docs/images/sim-comparison.png)

The step model over-predicts TTFT for this saturated fixed-concurrency workload by
~70ms at the median (a known calibration gap in the out-of-sample fit).
The knob model clamps both tails by construction.

Note: the trace's std-devs (TTFT 80ms, ITL 8ms) exceed llm-d-inference-sim's config
validation, which caps std-dev at 30% of the mean, so it runs with the largest spread
it accepts (39ms / 3.3ms).

```bash
# llm-d-inference-sim invocation used above
llm-d-inference-sim --port 8001 --model Qwen/Qwen3-8B --mode random \
  --force-dummy-tokenizer --max-model-len 16384 --max-num-seqs 128 \
  --time-to-first-token 132ms --time-to-first-token-std-dev 39ms \
  --inter-token-latency 11ms --inter-token-latency-std-dev 3300us

# this simulator: vllm-rs frontend + trace-fitted model, vLLM-default scheduler
# limits; the fit is the canonical set (sweep + warm multiturn + cold multiturn)
cat traces/h200-qwen3-8b/h200-sweep-full.jsonl \
    <(grep -v '"meta"' traces/h200-qwen3-8b/h200-multiturn-mtfit2.jsonl) \
    <(grep -v '"meta"' traces/h200-qwen3-8b/h200-multiturn-nocache4.jsonl) > /tmp/fit.jsonl
inference-sim --handshake-address tcp://127.0.0.1:5571 \
  --latency-trace /tmp/fit.jsonl \
  --max-num-seqs 1024 --max-num-batched-tokens 8192
```

#### Step-granular interference

The engine paces emission with a step clock that mirrors vLLM's per-step schedule:
decodes claim the shared token budget first, prefills chunk into whatever remains (in
admission order), and every co-running decode's gap is the composed step's duration.
Chunk compute is fitted from the trace as a depth-dependent function (attention makes deep
chunks cost more per token) plus a max-shape premium for budget-saturated steps; small
chunks hide under the batch's decode compute. Queueing, chunk serialization, and decode
elongation are produced by the step composer rather than by interference knobs.

The gate is counterfactual: fit on one workload (a constant-load sweep plus a warm
multiturn capture), then predict a cold-cache multiturn (~11k-token prompts, prefix
caching disabled) the model never saw, whose prefill chunks continuously interfere with
running decodes. The capture shows a two-shelf ITL band; the replay reproduces the
band's shape, mass (13.9% vs 14.1%), and tail.

![Counterfactual cold-multiturn replay](docs/images/step-model-counterfactual.png)

The warm-multiturn factual leg (99%+ prefix-cache hits) and a low-rate cold leg stay
calibrated under the same model:

![Factual warm-multiturn replay](docs/images/step-model-factual.png)

![Low-rate cold-multiturn replay](docs/images/step-model-lowrate.png)

The same fit procedure refits from a Qwen3-30B-A3B (MoE) sweep without constant
changes and reproduces its counterfactual band.

### Open-loop arrival replay

The calibrations above sample the latency model closed-loop. That validates
distributions, but it does not cover TTFT queueing, prefill stalls, or concurrency
mixing from an external arrival process.
`calibrate-e2e --replay-arrivals` direct-replays a captured arrival schedule in real
time (each request sent at its recorded offset, open loop) and compares client-side
TTFT/ITL/request-total quantiles against the capture. The arrivals are verbatim; every
latency is still modeled. `--latency-trace` fits the sim's model from a *different*
trace, so the gate runs on an arrival process outside the fitting set.

Setup: the same frontend → tap → engine stack as the capture rig, run locally with
`inference-sim` as the engine, its latency model fit from the canonical H200 fitting
set. `deploy/trace-capture/loadgen.py --pattern poisson|burst` drives arrival
processes the fitting set never contained.

| scenario                          | requests | concurrency seen | TTFT max err | ITL max err | req-total err |
|-----------------------------------|----------|------------------|--------------|-------------|---------------|
| poisson, 4 req/s                  | 464      | 1-15, median 6   | 36.1%*       | 1.1%        | 0.2%          |
| burst, 24 per 10s                 | 288      | 0 -> 24 spikes   | 0.4%         | 0.05%       | 0.5%          |
| multiturn agentic (see below)     | 495      | 1-13             | 26.0%*       | 0.9%        | 2.5%          |

The max-err columns are the worst single quantile across all concurrency buckets. The
starred cells are small-n tail artifacts: poisson's worst cell is its n=2
concurrency-1 bucket, multiturn's is a warm-TTFT p99 where captured 103ms vs modeled
76ms differ by transport jitter the in-process replay does not model. Medians and p90s
agree within ~1-2%, and request totals stay within 2.5%.

The burst scenario sends 24 simultaneous 512-token prefills to an idle engine, so TTFT
is queueing-dominated (burst TTFT p50 1.2s / p99 2.0s vs poisson's 58ms / 150ms on the
same config).

![Burst arrival replay](docs/images/replay-arrivals-burst.png)

![Poisson arrival replay](docs/images/replay-arrivals-poisson.png)

Per-concurrency-bucket rows shuffle under bursts (admission order inside a burst is not
deterministic), which is why the gate compares pooled quantiles plus per-request decode
totals.

Replayed prompts are unique-token synthetics: the captured workloads carry
`cached_tokens: 0`, and identical fill tokens would silently turn every replayed
request into a prefix-cache hit. Workloads with prefix reuse (multiturn/agentic) need
the prefix structure replayed too, which is the next
scenario.

To reproduce against any trace with `arrival_ms`:

```bash
# capture: any OpenAI-compatible target
uv run --with httpx deploy/trace-capture/loadgen.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --pattern poisson --rate 4 --duration 120 \
  --prompt-tokens 512 --output-tokens 128 --out run.json --trace-out client.jsonl

# replay the schedule, fitting the model from a different capture
just replay tap-poisson.jsonl /tmp/fit.jsonl

# real-vs-replay survival curves (replay measurements via --dump-trace)
just compare "real=tap-poisson.jsonl" "replay=replay-measured.jsonl"
```

### Prefix cache and agentic multiturn

The agentic scenario (`loadgen.py --pattern multiturn`): sessions arrive poisson at
`--rate`, each runs `--turns` closed-loop turns whose context grows by the turn's
prompt plus the model's response, on top of one of `--prefix-count` shared
`--prefix-tokens` prefixes. The validation run below is ~100 sessions x 5 turns over
two ~10k-token shared prefixes; 493 of 495 requests were prefix-cache hits.

Prefix caching is not a latency knob. The engine runs a block-pool prefix cache;
admission computes each request's cached-token count, the trace-fitted TTFT model
conditions on the uncached prompt size, and a prefill admission stalls concurrent
decodes by its uncached tokens. Replaying prefix-cache workloads requires the workload's
sharing structure. The tap fingerprints every prompt with chained per-block hashes
(`block_hashes`), and replay expands each distinct hash to one deterministic token
block. Replayed prompts therefore share prefixes at the same block boundaries as the
capture.

Two replay modes apply. Pure open-loop replay fires every turn at its recorded offset;
`--replay-sessions` restores the generator's semantics (turn N+1 fires when turn N
completes plus the recorded think gap; sessions are inferred from the hash chains).
Session pacing matches closed-loop client behavior: cold turns take seconds, so later
turns are delayed by prior responses. Open-loop replay would fire every turn on the
original warm schedule.

The figure shows captured vs modeled TTFT survival per turn cohort (turn-1 requests:
shared prefix hit only; turns 2+: growing context), plus the same schedule replayed
with `--cold-prompts` (prefix reuse disabled). Without the cache, every turn
re-prefills ~11k tokens and offered prefill load exceeds engine capacity. On turns 2+,
TTFT p50 changes from 36ms to ~24s and p99 from 87ms to ~59s, with closed-loop
sessions enabled.

![Multiturn cache effect](docs/images/multiturn-cache-effect.png)

```bash
# capture an agentic workload (10k-token shared prefixes at ~1.5 tokens/word)
uv run --with httpx deploy/trace-capture/loadgen.py --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --pattern multiturn --rate 1 --turns 5 \
  --prefix-tokens 6500 --prompt-tokens 128 --output-tokens 128 --duration 120 \
  --out run.json

# session-paced replay, then the cache-off replay
cargo run --release --bin inference-sim-trace -- calibrate-e2e tap-multiturn.jsonl \
  --replay-arrivals --replay-sessions --latency-trace /tmp/fit.jsonl \
  --sim-arg=--kv-cache-size --sim-arg=65536 --dump-trace replay-measured.jsonl
cargo run --release --bin inference-sim-trace -- calibrate-e2e tap-multiturn.jsonl \
  --replay-arrivals --replay-sessions --cold-prompts ... --dump-trace nocache-measured.jsonl

# per-cohort figure
uv run scripts/plot_calibration.py --cache-effect real=tap-multiturn.jsonl \
  --cache-effect replay=replay-measured.jsonl --cache-effect nocache=nocache-measured.jsonl \
  --out-dir docs/images
```

The cold replay uses the chunk-cost model validated by the cold-multiturn
counterfactual gate at this prompt scale. Workload traces with block-hash ids, lengths,
and timestamps map onto this schema.

### Content-identical replay

By default, traces include timing, shapes, and prefix structure (block hashes), but not
tokens. The tap's `--record-tokens` option adds each request's `output_token_ids` to
the trace. `finish_reason` is always recorded. With the same tokenizer, recorded token
ids decode back to generated text, so token-recording traces can contain user content.

On the replay side, `inference-sim --replay-tokens <trace>` serves the recorded ids
verbatim instead of random tokens, and ends each stream with the recorded finish
reason. `--replay-match` controls request-to-record matching:

- `index` (default): the trailing `-<index>` of the request id, where the index is the
  record's position in the arrival-ordered schedule (the replay harness names requests
  `replay-{i}`). This requires replay-generated request ids. Combined with arrival
  replay, it reproduces the captured token stream on the wire.
- `prefix`: the incoming prompt's chained block hashes are matched against the records'
  `block_hashes`, longest shared prefix wins, ties go to arrival order, and each record
  is consumed by its first match (a duplicate prompt takes the next duplicate record;
  once all are consumed, retries re-serve the best match). The matched stream ends where
  the capture did: the engine clamps the live request's `max_tokens` to the recorded
  length. This supports closed-loop clients with their own request ids, such as an agent
  loop re-run against the simulator. Because block hashes are chained, a tail change in
  a prompt shortens the match depth without changing earlier block matches.

Unmatched requests fall back to random tokens in both modes. These modes provide
deterministic streams for testing routers, EPPs, guardrails, and client SDK streaming
behavior without a GPU. Prefix mode can replay a closed-loop agentic workload offline
when the agent is deterministic; `tests/closed_loop_prefix_replay.rs` covers this case.

Every trace touchpoint (`--trace-out`, `--latency-trace`, `--replay-tokens`, trace
conversion and replay harnesses) reads and writes gzip transparently when the path ends
in `.gz`; token-recording traces grow by one integer per generated token, so
compressing them is recommended.

### Replay pacing

Content replay (`--replay-tokens`) and timing are configured independently:

| Mode | Invocation |
| --- | --- |
| Timing-modeled | `--replay-tokens trace.gz --latency-trace trace.gz` plus scheduler args matching the capture (`--max-num-seqs`, `--max-num-batched-tokens`, ...): gaps and burst sizes sampled from a model fitted to the trace |
| Timing-verbatim | `--replay-tokens trace.gz --replay-steps trace.gz`: each request replays its recorded per-chunk sizes and gaps |
| As fast as possible | `--replay-tokens trace.gz` and nothing else: all timing knobs default to 0, the instant model |
| Compressed but shaped | `--replay-tokens trace.gz --latency-trace trace.gz --time-scale 100`: same interleavings and relative ordering, 100x faster wall clock |
| Synthetic timing | `--replay-tokens trace.gz --time-to-first-token 50 --inter-token-latency 10` |

For the fast path, scheduler limits still apply at zero delay. `--max-num-seqs` and the
token budget control queueing and backpressure; increase them for pass-through replay.
`--output-token-chunk-size` controls output framing.

### Speculative decoding and diffusion

Speculative decoding and diffusion can emit multiple tokens from one engine step. A
step can deliver one verified token plus accepted drafts, or a diffusion block. Capture
and replay preserve this chunk structure.

On the capture side the tap records, per output chunk, one `itl_ms` gap and the number
of tokens that chunk delivered in a parallel `itl_tokens` array (omitted for plain
autoregressive captures, so old traces are unchanged). The first chunk has no gap; its
size is `output_tokens - sum(itl_tokens)`. With `--step-stats-out` the tap also writes
a per-step `SchedulerStats` sidecar, which under speculative decoding carries
`spec_decoding_stats` (per-position acceptance) from the vLLM engine.

There are two replay modes for this structure:

- **Modeled** (`--latency-trace`): the latency model draws the recorded `(gap, tokens)`
  pairs *jointly* from donor pools fitted to the capture. A step the capture saw deliver
  four tokens replays as one four-token message after a *sampled* gap, never four
  messages at gap/4. It reproduces the burst distribution, not an individual request's
  recorded sequence, so sampling drift is expected.
- **Replay** (`--replay-steps`): each matched request emits its recorded chunk sizes at
  its recorded gaps (the timing analogue of `--replay-tokens`; requests resolve to
  records via `--replay-match`). This mode replays recorded timing but does not
  transfer to a different workload.

Either way the simulator re-derives `spec_decoding_stats` from the bursts it emits
(speculative budget `K = max(itl_tokens) - 1`, a burst of N tokens reported as 1 target
token plus N-1 accepted drafts), so its scheduler stats use the same structure as the
capture. Autoregressive traces report no spec stats, matching a vLLM engine with
speculation off.

```bash
# 1. Capture: vLLM engine with ngram spec decode behind the tap (writes
#    tap-trace.jsonl + step-stats.jsonl). See deploy/trace-capture/ for manifests.
just capture-up && bash deploy/trace-capture/run-capture.sh && just capture-down

# 2. Replay the recorded schedule with verbatim per-request bursts and gaps.
cargo run --release --bin inference-sim-trace -- calibrate-e2e \
    /tmp/trace-capture-h200/tap-trace.jsonl --replay-arrivals \
    --sim-arg=--replay-steps=/tmp/trace-capture-h200/tap-trace.jsonl \
    --dump-trace /tmp/spec-replay.jsonl

# 3. Plot capture vs replay: burst sizes, per-chunk pacing, acceptance.
uv run scripts/plot_calibration.py \
    --spec-fidelity real=/tmp/trace-capture-h200/tap-trace.jsonl \
    --spec-fidelity replay=/tmp/spec-replay.jsonl \
    --spec-steps real=/tmp/trace-capture-h200/step-stats.jsonl \
    --out-dir docs/images
```

![Speculative decoding replay fidelity](docs/images/spec-decode-fidelity.png)

The figure is the verbatim `--replay-steps` path (a 4096-record Qwen3-8B run; ngram on
this workload accepts often, so ~45% of steps deliver the full 5 tokens). Left: tokens
delivered per decode step, captured vs replayed. Middle: step time vs burst size;
speculation verifies all K drafts in one target forward pass, so median step time is
~flat in the burst size (~12ms whether the step delivered 1 or 5 tokens). The dashed
line is the ~gap/N result that would appear if one chunk were split into N equal gaps.
Right: per-position draft acceptance read back from the `SchedulerStats` sidecar (pass
a second `--spec-steps replay=...` to overlay the simulator's own emitted stats).
Covered without a GPU by `tests/spec_replay_fidelity.rs` and `replay_steps`/engine unit
tests.

### Visualizing traces (Perfetto)

`inference-sim-trace perfetto` converts a trace to the Chrome Trace Event Format for
<https://ui.perfetto.dev>: per-request prefill/decode spans packed into peak-concurrency
lanes, batch-level counters, and (with the tap's `--step-stats` sidecar) a step-centric
track of what the scheduler actually ran each step. `--open` serves it and launches the
UI. See [docs/perfetto.md](docs/perfetto.md).

```bash
cargo run --bin inference-sim-trace -- perfetto trace.jsonl --step-stats trace-step-stats.jsonl --open
```

## Dependencies of note

- `vllm-engine-core-client` — pinned git dep on `vllm-project/vllm` (`rev` in
  `Cargo.toml`). Bump the rev to track upstream protocol changes.
- `nixl-sys` — pinned git dep on `ai-dynamo/nixl` (`rev` in `Cargo.toml`), the same
  source the image builds `libnixl` from, so the crate resolves identically on macOS
  (stub) and in the container (native library).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
