# Agent replay: live agent capture -> zero-GPU byte-identical replay

One demo, two acts: Claude Code does a coding task against a real H200, every
token is recorded on the wire by the tap, the GPU is scaled to zero on
camera, and the *same* agent command replays the entire loop against the
simulator and produces the identical file.

## Beat sheet

| Beat | What's on screen | Why it lands |
| --- | --- | --- |
| 1 | `kubectl get pods` showing the GPU capture rig | establishes the real hardware |
| 2 | `demo/run-agent.sh http://localhost:8000 /tmp/demo-live` | a real agent doing real work on a real model |
| 3 | `cat greet.py` + run it | the work product |
| 4 | `just agentic-capture-fetch` + head of the trace | the recording exists, tokens and timing |
| 5 | `just agentic-capture-down` + empty `kubectl get pods` | **the GPU is gone** |
| 6 | same `run-agent.sh` against `:8001` | same agent, simulator behind the curtain |
| 7 | `diff` of the two work products printing nothing | byte-identical, zero GPU |

## Pre-flight (slow stuff, off camera)

```bash
just agentic-capture-up            # wait for 4/4 (engine downloads ~61GB once)
kubectl -n "${NAMESPACE:-inference-sim}" port-forward deploy/trace-capture-h200-agentic 8000:8000 &
just replay-up                     # sim waits for a trace; frontend boots
# after act 1 produces the trace:
kubectl -n "${NAMESPACE:-inference-sim}" port-forward deploy/offline-replay 8001:8000 &
```

## Recording

- `vhs demo/agent-replay.tape` renders `demo/agent-replay.gif` deterministically
  from the tape. Re-render any time the sim improves; no human in the loop.
  Trim the act-1 `Sleep` to match real agent latency.
- For an interactive-feel take instead, `asciinema rec demo.cast`, drive the
  beats by hand, convert with `agg demo.cast demo.gif`. Use this for the live
  act if you want visible streaming; VHS is better for the replay act since
  it's reproducible.

## Determinism notes (why act 2 matches act 1)

- Replay serves the *recorded* output tokens per request (prefix-matched on
  the prompt's block-hash chain), so the agent reconstructs the same next
  prompt and the loop closes. See docs/agentic-offline-replay.md.
- Same-day replay matters: Claude Code stamps today's date into its system
  prompt. Different day = first blocks diverge = no match. (libfaketime if
  you need to re-record later.)
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0` in run-agent.sh kills the per-session
  attestation hash; the vLLM frontend would strip it anyway, belt and braces.
- The workspace is wiped before each act (`run-agent.sh` does this), so the
  agent's `ls`/`cat` tool outputs match between acts.
- If a turn falls back to random tokens, the sim logs "no trace record shares
  a prompt prefix": that's the divergence alarm.
