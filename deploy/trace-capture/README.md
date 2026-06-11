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

## Limitations

- Single engine, single client (`client_index` 0); no coordinator
  pass-through (DP > 1 untested).
- Aborted requests are discarded, not recorded.
- Multi-token output chunks divide their gap evenly across the chunk's ITL
  entries.
