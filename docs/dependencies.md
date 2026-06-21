# Dependencies of note

The dependency graph is intentionally split so offline trace tooling stays light while
protocol-facing binaries remain pinned to the vLLM line they speak.

- `vllm-engine-core-client` is a pinned git dependency on `vllm-project/vllm`
  (`rev` in `Cargo.toml`). The supported-line matrix rewrites this rev from
  `compat.toml`; do not let routine dependency tooling bump it.
- `nixl-sys` is a pinned git dependency on `ai-dynamo/nixl`, matching the source used
  when the image builds `libnixl`. The default build does not compile it; enable
  `nixl` for real transfers or `nixl-stub` for typechecking without libnixl.
- `sim-trace` is deliberately vLLM-protocol-free. It owns the trace schema,
  calibration models, Perfetto conversion, and guidellm conversion so analysis tools
  can build without compiling the protocol stack.
