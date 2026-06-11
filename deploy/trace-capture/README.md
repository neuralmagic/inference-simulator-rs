# `inference-sim-tap`: the engine-core recording tap

A transparent ZMQ proxy that sits between a vLLM frontend and a real
engine-core, recording per-request timing (and optionally content) into a
JSONL trace. It plays both protocol roles: an engine toward the frontend, a
frontend toward the engine. Frames move verbatim in both directions; the tap
decodes copies for observation only, after forwarding.

```text
  clients (guidellm / loadgen.py)
      | HTTP
  frontend   vllm-rs serve --data-parallel-size-local 0
      | ZMQ (handshake :5570)
  tap        inference-sim-tap                            trace -> /trace
      | ZMQ (handshake :5580)
  engine     vllm serve --headless                        the GPU
```

## Running as a sidecar

The tap is a sidecar by design: it has no GPU, no model, and no state beyond
the trace file. The only requirement is that the frontend and engine-core
talk ZMQ across a boundary the tap can sit on - which means a *disaggregated*
vLLM deployment (`vllm serve --headless` + a separate frontend). A monolithic
`vllm serve` keeps the engine in-process and exposes no wire to tap.

Wiring rules:

1. The tap binds the engine-side handshake (`--engine-handshake`) and its own
   input/output sockets; the headless engine connects to it
   (`--data-parallel-rpc-port` pointing at the tap, not the frontend).
2. The tap dials the frontend (`--frontend-handshake`) only after the engine
   registers, relaying the engine's real ready response (max_model_len,
   num_gpu_blocks) so the frontend validates against the actual engine.
3. The three processes handshake exactly once, in order. If any container
   restarts, delete the pod: a half-rewired pod records garbage.
4. Version pinning is load-bearing: the msgspec data-path structs are
   positional, so the engine image must match the protocol commit the tap was
   built against (see `Cargo.toml`'s `vllm-engine-core-client` rev).

Container snippet (see `h200-capture.yaml` for the full three-container pod):

```yaml
- name: tap
  image: quay.io/wseaton/mock-engine-nixl:trace-capture-v5
  command: ["/usr/local/bin/inference-sim-tap"]
  args:
    - --frontend-handshake=tcp://127.0.0.1:5570  # frontend's --handshake-port
    - --engine-handshake=tcp://127.0.0.1:5580    # engine's --data-parallel-rpc-port
    - --input-address=tcp://127.0.0.1:29560
    - --output-address=tcp://127.0.0.1:29561
    - --trace-out=/trace/trace.jsonl             # .gz to compress
    - --model=Qwen/Qwen3-8B
    - --gpu=H200
    - --tp=1
    # opt-in content capture; the trace then carries user/model content
    # - --record-tokens
  env:
    - { name: RUST_LOG, value: info }
  resources:
    requests: { cpu: "2", memory: 1Gi }
    limits: { cpu: "4", memory: 2Gi }
  volumeMounts:
    - { mountPath: /trace, name: trace }
```

## Does it add latency?

Per message the tap adds one loopback TCP hop in each direction plus a tokio
scheduling wake: tens of microseconds against inter-token gaps of ~10ms, so
well under 1% on the engine data path. Two design choices keep it that way:
frames are forwarded before the observation decode runs (timestamps are
stamped at the wire, decode happens on refcounted copies off the critical
path), and the trace write lands at request completion, buffered.

The honest caveats:

- The proxy is a single forwarding loop, so observation work (msgpack decode,
  prompt block-hashing) delays the *next* frame, not the current one. At
  ~10ms ITLs this is noise; at multi-thousand-tokens/s aggregate throughput
  the loop becomes a serialization point worth measuring before trusting.
- Empirically the overhead has never surfaced: tap-recorded TTFT/ITL agrees
  with client-side measurements of the same run, and every calibration gate
  in the main README was scored against tap traces.

## Recording content

`--record-tokens` adds each request's `output_token_ids` to its trace record
(`finish_reason` is always recorded; it carries no content). With the same
tokenizer the ids decode back to the generated text: such traces carry user
and model content and lose the share-freely property of the default
hash-only schema. Compress them (`--trace-out=...jsonl.gz`); token recording
grows the trace by one integer per generated token.

These traces drive content-identical replay: `inference-sim
--replay-tokens trace.jsonl.gz` serves the recorded streams byte for byte
(see "Content-identical replay" in the main README).

## Fetching traces

```bash
kubectl -n weaton-dev exec deploy/trace-capture-h200 -c tap -- cat /trace/trace.jsonl > trace.jsonl
```

For `.gz` traces the stream is finalized on Ctrl-C/SIGTERM (the pod's normal
termination path); a SIGKILL leaves a truncated gzip file, so fetch before
deleting the pod, or rely on the per-record sync flush and accept a
truncated-trailer warning from forgiving decoders.

## Tutorial: from live traffic to byte-identical replay

The 15-minute version of what the calibration figures are built on, using a
content capture you can verify by reading it. Everything below is the exact
sequence from the first real token capture (Qwen3-8B, H200, 2026-06-11).

### 1. Bring up the capture rig

```bash
kubectl apply -f deploy/trace-capture/h200-capture.yaml
kubectl -n weaton-dev scale deploy/trace-capture-h200 --replicas=1
kubectl -n weaton-dev wait --for=condition=ready pod \
  -l llm-d.ai/guide=trace-capture --timeout=600s   # weight download ~5 min
kubectl -n weaton-dev port-forward deploy/trace-capture-h200 8000:8000 &
curl -sf http://127.0.0.1:8000/v1/models   # frontend is up
```

The manifest already passes `--record-tokens` to the tap. Two operational
rules: the cluster reaps idle GPU pods after ~35 minutes ("Pod was not using
GPU" event) and the trace lives in an emptyDir, so run your load promptly and
fetch the trace before walking away.

### 2. Drive verifiable load with guidellm

Use prompts whose answers you can check by eye, with a per-row output budget:

```bash
python3 - <<'EOF'
import json
qs = ["What is the capital of France?", "List the first five prime numbers.",
      "Name the planets of the solar system in order from the sun."]
with open("prompts.jsonl", "w") as f:
    for q in qs:
        f.write(json.dumps({"prompt": f"Question: {q}\nAnswer:",
                            "output_tokens_count": 48}) + "\n")
EOF

uvx guidellm@0.6.0 benchmark run \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --processor Qwen/Qwen3-8B \
  --data prompts.jsonl --request-type /v1/completions \
  --profile constant --rate 2 --max-seconds 45 --data-samples 20 \
  --output-dir . --outputs benchmarks.json
```

(Streaming guidellm needs a frontend at >= vLLM #43965; against an older
vllm-rs every request 400s on `stream_options.continuous_usage_stats` - add
`--backend-kwargs '{"stream": false}'` there. The tap records server-side, so
the trace is identical either way.)

### 3. Fetch the trace, release the GPU

```bash
kubectl -n weaton-dev exec deploy/trace-capture-h200 -c tap -- \
  cat /trace/trace.jsonl > h200-tokens-tap.jsonl
kubectl -n weaton-dev scale deploy/trace-capture-h200 --replicas=0
```

Each line is one request: timing (`ttft_ms`, per-token `itl_ms`, batch
context), the arrival schedule, prefix-structure hashes, and - because the
tap ran with `--record-tokens` - `output_token_ids` and `finish_reason`.

### 4. Read the capture back as text

```bash
uv run --with tokenizers --with huggingface_hub python3 - <<'EOF'
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-8B", "tokenizer.json"))
for line in open("h200-tokens-tap.jsonl"):
    r = json.loads(line)
    if "output_token_ids" in r:
        print(tok.decode(r["output_token_ids"])[:120])
EOF
```

Expect actual model output: " The capital of France is Paris. ...". This is
the moment the trace stops being shareable - it carries content now.

### 5. Replay it byte-identically

The repo ships a verification test that boots the sim on any token trace and
asserts every replayed stream equals the capture, finish reasons included:

```bash
REPLAY_TRACE=h200-tokens-tap.jsonl cargo test --test real_trace_replay -- --nocapture
# replaying 20 records (20 with tokens) from h200-tokens-tap.jsonl
# all 20 token streams byte-identical, finish reasons match
```

For your own harness, run the sim directly and pick a pacing mode (see
"Replay pacing" in the main README):

```bash
# as fast as possible (client harness verification): instant timing
inference-sim --handshake-address tcp://127.0.0.1:29550 \
  --replay-tokens h200-tokens-tap.jsonl

# timing-accurate: same content AND the capture's latency model
inference-sim --handshake-address tcp://127.0.0.1:29550 \
  --replay-tokens h200-tokens-tap.jsonl \
  --latency-trace h200-tokens-tap.jsonl
```

Name requests `replay-{i}` (index = position in the arrival-ordered
schedule) to hit specific records; anything else falls back to random
tokens. The 45-second capture replays in ~40ms in fast mode.

## Limitations

- Single engine, single client (`client_index` 0); no coordinator
  pass-through (DP > 1 untested).
- Aborted requests are discarded, not recorded.
- Multi-token output chunks divide their gap evenly across the chunk's ITL
  entries.
