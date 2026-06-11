# Trace inventory

Captured latency traces (tap schema, see `src/trace.rs`). The directory is
gitignored except for this file: traces are measurement data, some are large,
and token-recording ones carry model content. This README is the index of
what each capture is and how it's used; keep it current when adding captures.

Capture hygiene rules learned the hard way:

- One workload per pod lifetime; a half-rewired pod records garbage.
- Never fit and gate on the same trace.
- Never trust a single-seed gate: 3 of ~10 multiturn captures turned out
  anomalous (`mtfit`, `nocache3`, `nocachelo`), which only multi-seed
  averaging exposed.
- Fetch promptly; the cluster reaps idle GPU pods (~35 min) and emptyDir
  traces die with the pod.

## h200-qwen3-8b/ (Qwen3-8B, TP=1, H200, vLLM postmerge builds)

Calibration roles as of the step-model campaign (2026-06-11): **fitting** =
`sweep-full` + `mtfit2` + `nocache4`; **factual gate** = `mtfit3`;
**counterfactual gate** = seed-averaged `nocache{,2,3}`; **low-rate gate** =
seed-averaged `nocachelo{2,3,4}`.

| file | what it is |
| --- | --- |
| `h200-qwen3-tap-trace.jsonl`, `-v2` | original tap captures (first real-engine recordings); v2 is the README examples' latency trace |
| `h200-sweep-full.jsonl` | concurrency sweep, fitting |
| `h200-multiturn-mtfit.jsonl` | RETIRED: capture anomaly (was the factual gate seed) |
| `h200-multiturn-mtfit2.jsonl` | multiturn, fitting |
| `h200-multiturn-mtfit3.jsonl` | multiturn, factual gate |
| `h200-multiturn-nocache.jsonl` | cold-cache cf, seed 7, gate |
| `h200-multiturn-nocache2.jsonl` | cold-cache cf, seed 8, gate |
| `h200-multiturn-nocache3.jsonl` | cold-cache cf, seed 9, gate but ANOMALOUS (p75 30 vs 14.8/13.1) - retirement candidate |
| `h200-multiturn-nocache4.jsonl` | cold-cache cf, seed 10, fitting (cold donors) |
| `h200-multiturn-nocachelo.jsonl` | low-rate cf, seed 7, ANOMALOUS (14.4% 18-60ms shoulder) - anomaly exhibit, not in the gate |
| `h200-multiturn-nocachelo2.jsonl` | low-rate cf, seed 8, gate |
| `h200-multiturn-nocachelo3.jsonl` | low-rate cf, seed 9, gate |
| `h200-multiturn-nocachelo4.jsonl` | low-rate cf, seed 11, gate (adjudication capture; do NOT add to fitting - measured as a no-op) |
| `h200-multiturn-cached.jsonl` | warm multiturn (agentic prefix-cache scenario) |
| `h200-coldfloor.jsonl` | cold-floor probe |
| `h200-tokens-tap.jsonl` | first `--record-tokens` capture (2026-06-11, guidellm, trivia prompts): CARRIES MODEL CONTENT; drives `tests/real_trace_replay.rs` |

## h200-qwen3-30ba3b/ (Qwen3-30B-A3B MoE cross-model check)

| file | what it is |
| --- | --- |
| `h200-30ba3b-sweep.jsonl` | fitting sweep |
| `h200-30ba3b-mtfit.jsonl` | factual gate |
| `h200-30ba3b-nocache.jsonl` | counterfactual gate |

Known wart: the factual capture's first ~11s are a pod warm-up transient
that inflates the f-ttft gate.

## h200-scout/

`h200-scout-trace.jsonl`: Llama-4-Scout capture (early tap validation).

## local-sim/

Tap self-captures (tap recording the simulator itself, no GPU): `burst`,
`multiturn`, `poisson` arrival patterns. Used for tap/replay plumbing tests
and as cheap schema examples.
