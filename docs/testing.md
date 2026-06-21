# Testing

The main local gate is:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --no-deps -- -D warnings
cargo test --workspace --locked
```

The full smoke scripts also boot a real vLLM frontend:

```bash
./scripts/e2e.sh        # boots vllm-rs + this engine, asserts streaming + non-streaming flows
./scripts/e2e_lora.sh   # loads a LoRA adapter, asserts vllm:lora_requests_info names it
./scripts/e2e_generate.sh # exercises /inference/v1/generate token-in/token-out
```

These scripts need `vllm-rs` built once (`cargo build --bin vllm-rs` in the vLLM
`rust/` workspace). Override its path with `FRONTEND_BIN=...`. The first run fetches
the tokenizer from Hugging Face.

`e2e_lora.sh` needs a frontend that exports `vllm:lora_requests_info` from the
frontend metrics path. The image and current default protocol pin qualify; if you use
your own checkout, point `FRONTEND_BIN` at a compatible build.
