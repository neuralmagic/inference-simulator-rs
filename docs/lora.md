# LoRA simulation

The simulator tracks LoRA adapters loaded by the frontend through the engine utility
path (`add_lora` / `remove_lora`) and uses that registry while scheduling requests.

`--max-loras` mirrors vLLM's running-batch diversity cap: it limits how many distinct
adapters may be resident in the running batch at once. `0` disables the cap, but LoRA
accounting still runs so frontend metrics can report running and waiting adapters.

In the container image, set `MOCK_MAX_LORAS`; the entrypoint maps it to
`vllm-vcr play --max-loras`.

The `vllm:lora_requests_info` metric is frontend-derived on current supported vLLM
lines. The simulator's job is to keep scheduler stats and adapter state consistent
enough for that frontend metric to reflect the request mix.
