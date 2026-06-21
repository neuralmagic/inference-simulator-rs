# Perfetto trace viewer

Convert a JSONL engine trace into the [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
and view it on <https://ui.perfetto.dev>. The converter is the `perfetto`
subcommand of `vllm-vcr inspect`; it reads the same trace files the replay and
calibration paths use (`.gz` transparent), and optionally overlays the tap's
step-stats sidecar.

For the trace schema itself, see `crates/sim-trace/src/trace.rs`; for the sidecar,
`crates/sim-protocol/src/step_stats.rs`.

## Contents

- [Quick start](#quick-start)
- [What you see](#what-you-see)
- [Options](#options)
- [Fidelity: request shape vs scheduler steps](#fidelity-request-shape-vs-scheduler-steps)
- [Notes and limits](#notes-and-limits)

## Quick start

Write a Perfetto JSON file and drag it onto <https://ui.perfetto.dev>:

```bash
cargo run --bin vllm-vcr -- inspect perfetto trace.jsonl -o trace.perfetto.json
```

Or let it serve the trace and open the UI for you (blocks until Ctrl-C, since the
hosted UI fetches the file from this process):

```bash
cargo run --bin vllm-vcr -- inspect perfetto trace.jsonl --open
```

Overlay the step-stats sidecar (`vllm-vcr record --step-stats-out`) for the
batch-level counters and the per-step scheduler track:

```bash
cargo run --bin vllm-vcr -- inspect perfetto trace.jsonl \
  --step-stats trace-step-stats.jsonl --open
```

## What you see

Two process groups, both on one shared clock (milliseconds since capture start;
`arrival_ms` and the sidecar's `ts_ms` use the same zero).

**`inference trace`** — the per-request shapes. Each request is a `prefill` span
(`arrival → first token`) followed by one `decode` span per inter-token gap
(`itl_ms`), with multi-token chunks (spec decode, diffusion blocks) named
`decode xN`. Requests are packed into reusable lanes: a lane frees the moment its
request finishes, so the row count is the **peak concurrency**, not the request
count (a 2500-request trace becomes ~16 lanes). Under it sit counter tracks:

| Counter | Source | Meaning |
| --- | --- | --- |
| `active_requests` | swept from the spans | in-flight request depth over time |
| `engine_running` | `itl_ctx.num_running` | engine-reported running count per decode gap |
| `prefill_tokens` | `itl_ctx.prefill_tokens` | prompt tokens that finished prefill in the gap's step |
| `sched_running_reqs` / `sched_waiting_reqs` | sidecar | scheduler batch and queue depth |
| `sched_kv_cache_usage` | sidecar | KV-cache pressure (0–1) |
| `sched_accept_rate` | sidecar | spec-decode acceptance rate (only when drafting) |

**`scheduler steps`** (with `--step-stats`) — one span per executed scheduler
step, back to back on a single row (steps are sequential, so they never overlap).
Each step is classified and colored by what it ran: `decode B<n>`, `prefill`, or
`prefill+decode B<n> (+<r>r <t>t)`. A prefill step is visibly wider (it costs more),
and the args carry `running` / `waiting` / `prefill_requests` / `prefill_tokens` /
`kv_cache_usage` / `step_ms` and spec `accepted`/`draft`.

Spans are colored by phase so the language is consistent across both groups:
**orange = prefill, green = decode**, with a distinct shade for a mixed
prefill+decode step. Recorded output token ids (`tap --record-tokens`) ride each
span's `token_ids` arg, the hook for a future detokenize-to-text mode.

## Options

| Flag | Effect |
| --- | --- |
| `-o, --output <path>` | Write the JSON here (default: stdout, or nothing with `--open`) |
| `--step-stats <path>` | Overlay the step-stats sidecar (`.gz` ok): counters + the step track |
| `--name <label>` | Override the process-row label (default: the trace's model) |
| `--track-per-request` | One labelled row per request instead of packed lanes (good for small traces) |
| `--open` | Serve over localhost and open the Perfetto UI; blocks until Ctrl-C |
| `--port <n>` | Port for `--open`; default `0` lets the OS choose a free ephemeral port |

Records without an `arrival_ms` cannot be placed on a timeline and are dropped
(the command prints how many); guidellm-converted and `gen-demo` traces have none,
real tap captures do.

## Fidelity: request shape vs scheduler steps

The two views answer different questions, and the difference matters when reading
overlap.

The request shapes are a reconstruction of each request's **client-observed
latency envelope**: the `prefill` span is `arrival → first token`, which fuses
queue-wait and prefill compute into one contiguous bar, and the decode bar is the
observed inter-token cadence. Under load these bars overlap heavily, but that
**does not** mean the engine ran that many prefills at once. With chunked prefill,
the engine runs roughly one prefill chunk per step, interleaved with decodes, and
much of a prefill bar under saturation is queue-wait, not compute.

The `scheduler steps` track is the truthful counterpart: it shows what the engine
**actually executed each step**, sequential and non-overlapping, including where a
prefill chunk genuinely co-occurred with decodes in one step (`prefill+decode`).
Reach for the request shapes to see per-request experience (what replay cares
about), and the step track to see scheduler occupancy.

## Notes and limits

- Times are emitted in microseconds (the Chrome format's unit); `displayTimeUnit`
  is set to `ms` for the axis.
- Timestamps are relative to capture start, not wall-clock epoch, so a trace is
  self-contained but not directly correlatable with an external profiler.
- `--open` runs a minimal localhost HTTP server with permissive CORS so the hosted
  UI can fetch the file. The trace stays on your machine; the browser fetches it from
  `127.0.0.1`.
- Large traces produce large JSON (a ~2500-request capture is ~50 MB / ~370k
  events); the UI loads it fine but the default zoom fits the whole capture.
