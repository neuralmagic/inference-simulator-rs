# Install

Requires Rust 1.85 or newer. From a checkout:

```bash
cargo install --path . --locked
```

That installs the single `vllm-vcr` binary, with `record`, `play`, and `inspect`
subcommands. After the repository is public, the same default no-NIXL build can
be installed from Git:

```bash
cargo install --git https://github.com/neuralmagic/vllm-vcr \
  --locked vllm-vcr
```

For a NIXL-enabled install, build on Linux with `libnixl` and UCX available:

```bash
cargo install --path . --locked --features nixl
```

For the Kubernetes deployment, build the container image instead:

```bash
podman build -t ghcr.io/neuralmagic/vllm-vcr:dev .
```
