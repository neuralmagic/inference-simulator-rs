# LoRA simulation

The engine tracks LoRA adapters the frontend loads (`add_lora`/`remove_lora`) and
honors `--max-loras` (distinct adapters allowed in the running batch; `0` = no cap).
In the image, set `MOCK_MAX_LORAS`. The `vllm:lora_requests_info` gauge is
frontend-derived as of vLLM #45030.
