# Content-identical replay

By default, traces include timing, shapes, and prefix structure (block hashes), but not
tokens. The tap's `--record-tokens` option adds each request's `output_token_ids` to
the trace. `finish_reason` is always recorded. With the same tokenizer, recorded token
ids decode back to generated text, so token-recording traces can contain user content.

On the replay side, `vllm-vcr play --replay-tokens <trace>` serves the recorded ids
verbatim instead of random tokens, and ends each stream with the recorded finish
reason. `--replay-match` controls request-to-record matching:

- `index` (default): the trailing `-<index>` of the request id, where the index is the
  record's position in the arrival-ordered schedule (the replay harness names requests
  `replay-{i}`). This requires replay-generated request ids. Combined with arrival
  replay, it reproduces the captured token stream on the wire.
- `prefix`: the incoming prompt's chained block hashes are matched against the records'
  `block_hashes`, longest shared prefix wins, ties go to arrival order, and each record
  is consumed by its first match (a duplicate prompt takes the next duplicate record;
  once all are consumed, retries re-serve the best match). The matched stream ends where
  the capture did: the engine clamps the live request's `max_tokens` to the recorded
  length. This supports closed-loop clients with their own request ids, such as an agent
  loop re-run against the simulator. Because block hashes are chained, a tail change in
  a prompt shortens the match depth without changing earlier block matches.

Unmatched requests fall back to random tokens in both modes. These modes provide
deterministic streams for testing routers, EPPs, guardrails, and client SDK streaming
behavior without a GPU. Prefix mode can replay a closed-loop agentic workload offline
when the agent is deterministic; `tests/closed_loop_prefix_replay.rs` covers this case.

Every trace touchpoint (`--trace-out`, `--latency-trace`, `--replay-tokens`, trace
conversion and replay harnesses) reads and writes gzip transparently when the path ends
in `.gz`; token-recording traces grow by one integer per generated token, so
compressing them is recommended.
