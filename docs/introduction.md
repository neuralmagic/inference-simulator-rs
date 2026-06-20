# vllm-vcr

Record, play, and inspect vLLM **V1 engine-core** traces. One binary with three
subcommands, named for the VCR metaphor:

- **`record`** taps a live vLLM frontend ↔ engine-core link (a transparent ZMQ
  proxy) and writes a JSONL trace.
- **`play`** runs a mock engine-core backend that speaks the real ZMQ + msgpack
  protocol, replaying a recorded trace or simulating from a latency model. No model
  weights, no GPU.
- **`inspect`** converts benchmark reports, summarizes traces, renders Perfetto
  timelines, and runs calibration.

With the `nixl` feature and a working libnixl/UCX runtime, `play` also moves
simulated KV-cache bytes between prefill and decode instances over NIXL.

## Why

`llm-d-inference-sim` models prefill/decode in the control plane: it adjusts
latency and finish metadata, but it does not move KV-cache bytes. `vllm-vcr` adds
two capabilities:

1. **Frontend compatibility.** It runs behind vLLM's Rust or Python frontend. The
   frontend still handles tokenization, chat templates, tool calling, streaming, and
   OpenAI-compatible request handling. The simulator replaces only the model backend.
2. **Prefill/decode data-plane testing.** It can move simulated KV-cache bytes
   between prefill and decode instances over [NIXL](https://github.com/ai-dynamo/nixl)
   using the UCX backend when the NIXL runtime initializes. CUDA and model weights
   are not required.

## Where to start

- New here? Read [Architecture](./architecture.md), then [Install](./install.md)
  and the [Quick start](./quick-start.md).
- Capturing and replaying real traces? Jump to
  [Trace replay and calibration](./trace-replay/index.md).
- Running it across vLLM versions in CI? See [Versioning](./versioning.md) and
  [Conformance](./conformance.md).
