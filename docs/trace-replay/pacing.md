# Replay pacing

Content replay (`--replay-tokens`) and timing are independent. Pick the content mode
first, then choose how quickly and in what shape the engine should emit chunks.

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
