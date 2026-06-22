# vllm-vcr

Record, play, and inspect vLLM **V1 engine-core** traces. One binary, three
subcommands (the VCR metaphor):

- **`record`** taps a live vLLM frontend ↔ engine-core link (a transparent ZMQ
  proxy) and writes a JSONL trace.
- **`play`** runs a mock engine-core backend that speaks the real ZMQ + msgpack
  protocol, replaying a trace or simulating from a latency model. No model weights,
  no GPU. With the `nixl` feature it also moves simulated KV-cache bytes between
  prefill and decode over [NIXL](https://github.com/ai-dynamo/nixl).
- **`inspect`** converts benchmark reports, summarizes traces, renders Perfetto
  timelines, and runs calibration.

It runs behind vLLM's Rust or Python frontend unchanged: the frontend still owns
tokenization, chat templates, tool calling, streaming, and OpenAI-compatible request
handling; `vllm-vcr` replaces only the model backend.

## Documentation

**📖 Full docs: <https://neuralmagic.github.io/vllm-vcr/>**

The site covers architecture, install, the quick start, trace replay and
calibration, versioning and conformance, and operations. Source lives in
[`docs/`](docs/) and is built with [mdBook](https://rust-lang.github.io/mdBook/).

## Install

Requires Rust 1.85 or newer. From a checkout:

```bash
cargo install --path . --locked
```

That installs the single `vllm-vcr` binary. See the
[Install guide](https://neuralmagic.github.io/vllm-vcr/install.html) for the
NIXL-enabled build, the container image, and installing from Git.

## Quick start

```bash
# Run the mock engine; point a vLLM frontend at the same handshake address.
vllm-vcr play --handshake-address tcp://127.0.0.1:29550 --log-requests
```

Full walkthrough (frontend wiring, prefill/decode smoke, capture and replay) in the
[Quick start](https://neuralmagic.github.io/vllm-vcr/quick-start.html).

## Contributing

Run `cargo fmt` and `cargo clippy --all --benches --tests --examples --all-features`
before sending a change; CI runs the same plus the per-vLLM-line conformance suite
(see [`.github/workflows`](.github/workflows)).

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.
