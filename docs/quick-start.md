# Quick start

This smoke test runs the real vLLM frontend and replaces only the engine-core
backend. It does not need a GPU or model weights, but the frontend still downloads
the tokenizer on first use.

Start the simulator:

```bash
vllm-vcr play --handshake-address tcp://127.0.0.1:29550 --log-requests
```

In another shell, start a vLLM frontend with the same handshake port and no local
engine rank. The exact command depends on the vLLM line and frontend you are testing;
`vllm-vcr` uses the same external-engine role as vLLM's mock-engine harness. The repo
script wraps the common Rust-frontend path:

```bash
./scripts/e2e.sh
```

Send a streaming chat request once the frontend is up:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"max_tokens":16,"stream":true}'
```

## Prefill/decode control-plane smoke

Run the routing-sidecar schema round trip without NIXL:

```bash
./scripts/pd_control.sh
```

For the Kubernetes P/D deployment, build the image from the install section and use
[deploy/llm-d-pd/README.md](https://github.com/neuralmagic/vllm-vcr/blob/main/deploy/llm-d-pd/README.md).

## Replay a trace

Once you have a tap trace, use it as the latency source:

```bash
vllm-vcr play --handshake-address tcp://127.0.0.1:29550 \
  --latency-trace trace.jsonl.gz --log-requests
```

If the trace was captured with `record --record-tokens`, add
`--replay-tokens trace.jsonl.gz` to serve the recorded token ids instead of random
tokens.
