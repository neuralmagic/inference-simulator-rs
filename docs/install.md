# Install

`vllm-vcr` requires Rust 1.85 or newer. The default build is pure Rust and does not
include NIXL.

From a checkout:

```bash
cargo install --path . --locked
```

That installs the single `vllm-vcr` binary with `record`, `play`, `inspect`, and
`completions` subcommands.

To install the default no-NIXL build directly from Git:

```bash
cargo install --git https://github.com/neuralmagic/vllm-vcr \
  --locked vllm-vcr
```

For a NIXL-enabled binary, build on Linux with `libnixl` and UCX available:

```bash
cargo install --path . --locked --features nixl
```

For local development on a machine without libnixl, use the stub feature to typecheck
the NIXL code path without enabling real transfers:

```bash
cargo check --features nixl-stub
```

For Kubernetes deployments, build the container image instead. The image includes the
vLLM Rust frontend, the simulator, libnixl, and UCX:

```bash
podman build -t ghcr.io/neuralmagic/vllm-vcr:dev .
```
