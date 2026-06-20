# Dependencies of note

- `vllm-engine-core-client` — pinned git dep on `vllm-project/vllm` (`rev` in
  `Cargo.toml`). Bump the rev to track upstream protocol changes.
- `nixl-sys` — pinned git dep on `ai-dynamo/nixl` (`rev` in `Cargo.toml`), the same
  source the image builds `libnixl` from, so the crate resolves identically on macOS
  (stub) and in the container (native library).
