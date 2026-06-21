# Concepts

The trace docs use three terms consistently:

- **Captured** — per-token tap recordings from a vLLM engine, taken server-side on
  the engine-core protocol. Figures label these as "real" or "source".
- **Modeled** — latency the simulator emits. TTFT and per-token gaps are drawn from a
  statistical model fitted to a captured trace (conditioned on concurrency, context
  depth, and uncached prompt size). Captured timings are not played back verbatim,
  so a model fitted on one workload can be evaluated on another.
- **Direct replay** — recorded values used verbatim, no statistics: arrival
  timestamps (`--replay-arrivals`), session pacing (`--replay-sessions`), prefix
  structure (block hashes), per-step gaps (`--replay-steps`), and opt-in output token
  ids (`--replay-tokens`).

"Replay" in a figure or flag name refers to the workload side (the schedule being
replayed), not to the timing. Counterfactual gates fit on workload A, directly replay
workload B's schedule, and check the modeled timing against B's capture.

`just figures` rebuilds the figures from local trace files listed in
[traces/README.md](https://github.com/neuralmagic/vllm-vcr/blob/main/traces/README.md) (`scripts/make_figures.sh`; ~30 minutes, the
arrival replays run in real time). Those trace files are not committed. The
head-to-head comparison is the exception; it needs live serving stacks (commands in
that section).
