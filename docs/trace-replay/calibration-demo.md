# Calibration demo

The `vllm-vcr inspect` subcommands include a calibration harness that checks two
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
cargo run --bin vllm-vcr -- inspect gen-demo -o /tmp/demo.jsonl

# 2. Model-level calibration (no transport).
cargo run --bin vllm-vcr -- inspect calibrate /tmp/demo.jsonl

# 3. Wire-level: start the simulator and measure client-side.
cargo run --bin vllm-vcr -- inspect calibrate-e2e /tmp/demo.jsonl --requests 60
```

`--fast` on `gen-demo` produces a small-magnitude trace for quick e2e testing
(TTFT ~15-40ms, ITL ~3-10ms). All subcommands accept `--json` for machine-readable
output and `--seed` for determinism.
