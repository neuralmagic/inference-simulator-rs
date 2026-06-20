# Engine internals

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
