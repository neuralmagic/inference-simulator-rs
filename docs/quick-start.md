# Quick start

## Protocol-only local run

Start the simulator:

```bash
vllm-vcr play --handshake-address tcp://127.0.0.1:29550 --log-requests
```

Start a vLLM frontend with the same handshake address. See vLLM's
`rust/src/mock-engine/README.md` for the `vllm-rs serve` / `vllm serve`
invocations; this binary uses the same handshake role as `vllm-mock-engine`.

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
