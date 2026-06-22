# vllm-vcr

<div class="vcr-hero">
  <p class="eyebrow">GPU-free vLLM engine-core replay</p>
  <p class="lead">Record a real vLLM engine once, then replay its protocol behavior, timing, and optional output tokens behind the real vLLM frontend on ordinary CPU hosts.</p>
</div>

`vllm-vcr` is a single binary with three subcommands:

- **`record`** taps a live vLLM frontend ↔ engine-core link as a transparent ZMQ
  proxy and writes a JSONL trace.
- **`play`** runs a mock engine-core backend that speaks the real ZMQ + msgpack
  protocol. It can generate synthetic tokens, sample timing from a fitted trace,
  replay recorded step timing, or serve recorded token ids.
- **`inspect`** converts benchmark reports, summarizes traces, renders Perfetto
  timelines, and runs calibration checks.

With the optional `nixl` feature and a working libnixl/UCX runtime, `play` can also
move simulated KV-cache bytes between prefill and decode instances over NIXL.

## What it is for

Testing the software around a vLLM engine usually means provisioning GPUs and model
weights before you can exercise frontends, cache-aware routers, schedulers,
autoscalers, or CI compatibility matrices. `vllm-vcr` keeps the real frontend and
wire protocol in the loop, but replaces the model backend with a CPU simulator.

Use it when you need to:

- replay captured TTFT and inter-token behavior without a GPU;
- run OpenAI-compatible frontend, streaming, LoRA, scheduler, and router tests
  against the real engine-core protocol;
- validate trace fidelity and version compatibility in CI;
- test prefill/decode control-plane behavior, and optionally the NIXL data plane,
  without model weights.

It is not a model-quality simulator: generated tokens are random unless you record
and replay token ids, and latency fidelity depends on traces captured from the
engine/configuration you care about.

## How it fits

The vLLM frontend remains responsible for tokenization, chat templates, tool calling,
streaming, metrics, and OpenAI-compatible HTTP handling. `vllm-vcr play` only replaces
the engine-core process behind that frontend. For prefill/decode work, the default
data plane is a no-op; the NIXL path is opt-in and still runs without CUDA or model
weights.

<div class="vcr-grid">
  <div class="vcr-card">
    <h3>New setup</h3>
    <p>Read <a href="./architecture.html">Architecture</a>, then install the binary and run the quick start.</p>
  </div>
  <div class="vcr-card">
    <h3>Trace replay</h3>
    <p>Start with <a href="./trace-replay/index.html">Trace replay and calibration</a> for capture, model fit, and replay modes.</p>
  </div>
  <div class="vcr-card">
    <h3>Operations</h3>
    <p>Use <a href="./versioning.html">Versioning</a> and <a href="./conformance.html">Conformance</a> for multi-line vLLM support.</p>
  </div>
</div>
