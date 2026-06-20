# Testing

```bash
./scripts/e2e.sh        # boots vllm-rs + this engine, asserts streaming + non-streaming flows
./scripts/e2e_lora.sh   # loads a LoRA adapter, asserts vllm:lora_requests_info names it
```

Needs the `vllm-rs` frontend built once (`cargo build --bin vllm-rs` in the vLLM
`rust/` workspace); override its path with `FRONTEND_BIN=...`. First run fetches the
tokenizer from HF.

`e2e_lora.sh` needs a `vllm-rs` at or past vLLM #45030, which exports
`vllm:lora_requests_info` from the frontend (the engine no longer reports
per-adapter maps in `SchedulerStats`). The pinned commit in `Cargo.toml`/`Dockerfile`
qualifies.
