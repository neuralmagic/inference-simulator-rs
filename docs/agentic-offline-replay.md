# Closed-loop agentic replay: SWE-bench offline

Goal: capture one real SWE-bench agent rollout against a live vLLM, then
re-run the *same agent* fully offline against the sim and get the same
patches, the same eval results, zero GPU, zero API spend.

The repo side is done: `--replay-tokens <trace> --replay-match prefix` turns
the sim into a content-keyed cassette player (see "Content-identical replay"
in the main README). This doc is the runbook for the demo around it.

## Why this works

SWE-bench splits into rollout (the agent loop, the only LLM-dependent phase)
and evaluation (`swebench.harness.run_evaluation`, which applies patches in
per-instance Docker images and runs tests, no model calls). So only the
rollout needs capture/replay.

The agent's message history is append-only: turn N+1's prompt = turn N's
prompt + turn N's response + new tool output. Replay serves turn N's response
byte-identically, so a deterministic agent reconstructs turn N+1's prompt
exactly as captured, and the loop closes. Prefix matching keys on the chained
block hashes the tap already records; tail noise in tool output (timing
strings like pytest's `in 0.42s`) only shortens the match depth, it cannot
change which record wins.

## Agent harness choice

What decides replayability: volatile content must not appear early in the
prompt (chained block hashes make an early difference fatal), and history
must be append-only (compaction rewrites the prefix mid-session). One hazard
is universal: a harness that stamps today's date into the system prompt
breaks matching when the replay happens on a different day; pin the clock in
the rollout container with libfaketime to the capture date.

1. [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
   (`mini-extra swebench`), the v1 and determinism control: strictly linear
   append-only history, deterministic truncation (pure char-count slicing at
   10k chars), no timestamps/hostnames in the default swebench templates,
   talks OpenAI chat completions via LiteLLM so it works against the vllm-rs
   frontend as-is, and writes `preds.json` directly consumable by the
   official eval harness.
2. [pi](https://github.com/badlogic/pi-mono), the "real agent" driver:
   native tool calling, append-only history, no compaction, and its only
   volatile prompt content is a day-precision date + cwd (cwd is constant
   inside a SWE-bench container, the date is faketime's job). Point
   `models.yml` at the frontend (`api: openai-completions`, `baseUrl`), set
   `temperature: 0`, run headless per instance, `git diff` for the patch
   (model the glue on badlogic/pi-terminal-bench).
3. Claude Code, the headline demo: set `CLAUDE_CODE_ATTRIBUTION_HEADER=0` to
   disable the attestation line it otherwise prepends to the system prompt
   (`x-anthropic-billing-header: ...`, per-conversation hash; vLLM > 0.17.1
   also strips it server-side). Its gitStatus snapshot is reproducible when
   replay starts from the same SWE-bench image; the date needs faketime. The
   remaining work is plumbing: it speaks the Anthropic `/v1/messages` API,
   so capture needs either the native vLLM frontend (>= 0.11.2) or a LiteLLM
   translation proxy in front of the Rust one.

Avoid opencode for this: it puts a live git-status block early in the system
prompt (changes as the agent edits files, poisoning the prefix on most
steps) and auto-compacts history by default. Codex CLI can't target vLLM at
all (Responses-API-only wire protocol).

## Capture (one GPU run)

Standard tap sidecar setup (manifests in `deploy/trace-capture/`), with token
recording on and the **python** vLLM frontend, which serves `/v1/messages`
for Claude Code (vllm-rs doesn't yet; the protocol-pin image c9340e6f3
already ships the anthropic entrypoint including the billing-header strip):

```
agent -> HTTP -> frontend (vllm serve --data-parallel-size-local 0) -> ZMQ :5570 -> tap --record-tokens -> :5580 -> engine (GPU)
```

1. `just agentic-capture-up` deploys
   `deploy/trace-capture/h200-capture-agentic.yaml` (tap has
   `--record-tokens` on; the trace carries user content, token ids decode
   back to text, treat it accordingly). First deploy: verify the API-only
   frontend boots without a GPU; vLLM's platform probe has historically
   wanted CUDA, and the fallback is a CPU-target build of the same commit.
2. Point mini-swe-agent at the frontend and run a few instances:

   ```bash
   mini-extra swebench --subset verified --split test --slice 0:5 \
     --model hosted_vllm/Qwen/Qwen3-8B --workers 1
   # model base URL via the usual LiteLLM env (HOSTED_VLLM_API_BASE=http://<frontend>:8000/v1)
   ```

   `--workers 1` for the first capture: concurrent instances interleave fine
   (prompts don't share prefixes across instances past the system prompt),
   but single-stream makes the trace easy to eyeball.
3. Keep the agent's sampling deterministic-ish (`temperature: 0` in the model
   kwargs). Not strictly required for replay (the recorded tokens are served
   regardless of what sampling the replayed client asks for), but it makes
   capture-vs-replay diffs meaningful.
4. Save `preds.json` from the capture run; it's the ground truth the replay
   must reproduce.

## Replay (no GPU)

On the cluster: `just replay-up` deploys
`deploy/trace-capture/offline-replay.yaml` (the same python frontend in
front of `vllm-vcr play --replay-match prefix`, zero GPU), then
`just replay-load-trace <trace>` copies the capture in; the sim starts as
soon as the file appears. Or run the same pair locally:

```bash
vllm-vcr play --handshake-address tcp://127.0.0.1:5570 \
  --replay-tokens swebench-capture.jsonl.gz --replay-match prefix \
  --latency-trace swebench-capture.jsonl.gz   # optional: replay timing too
```

Same tokenizer is load-bearing: matching happens on token ids, so the
frontend must run the same model name and tokenize identical text to
identical ids (keep both rigs on the protocol-pin image).

**WARNING: keep `--latency-trace` on when replaying tool-calling models
through the python frontend.** Unpaced replay delivers entire responses
faster than the frontend's streaming consumer drains them; the per-request
`RequestOutputCollector` then merges deltas, and vLLM's `qwen3_coder`
streaming tool parser corrupts (>= ~8 tokens/delta) or silently drops
(whole response in one delta) tool calls, which kills the agent loop after
one turn. Paced replay reproduces the captured ~1 token/delta layout and
sidesteps the bug (upstream: vllm#45256; local repros in
`demo/repro-qwen3coder-burst.py` and `demo/bench-parser-quadratic.py`; the
parser's full-text rescans are also O(n^2) in response length). The Rust
frontend's tool parsers are delta-layout invariant and don't need this.

Then re-run the exact same agent command against it and compare:
`preds.json` (replay) == `preds.json` (capture), then run the official eval
on the replayed predictions:

```bash
python -m swebench.harness.run_evaluation \
  --predictions_path preds.json --run_id offline-replay ...
```

Watch the sim logs: every "no trace record shares a prompt prefix" or
"prompt shorter than one block" warning is a request that fell back to random
tokens, i.e. a divergence to explain (usually nondeterministic tool output
deep enough in the prompt to flip an entire turn, or a tokenizer mismatch).

## Known hazards

- Tool-output nondeterminism inside the SWE-bench containers (timestamps,
  partial output on command timeouts, unsorted `find`/`ls`). Tail noise is
  absorbed by longest-prefix matching; noise *early* in a turn's tool output
  shifts every later block and degrades the match to the turns before it.
- Aborted requests are dropped by the tap, so a client-side timeout during
  capture leaves a hole in the trace and the replayed agent gets random
  tokens for that turn.
- Concurrent identical prompts (agent retries) consume duplicate records in
  arrival order; once records run dry, retries re-serve the best match.
