# Architecture

`vllm-vcr` sits at the engine-core boundary. The frontend is still a normal vLLM
frontend; the simulator connects where a headless engine would connect and speaks the
same ZMQ + msgpack protocol. The protocol types come from vLLM's in-tree
`vllm-engine-core-client` crate, pinned per supported vLLM line.

<div class="vcr-flow vcr-flow-arch" role="img" aria-label="vLLM frontend connects to vllm-vcr play over the engine-core protocol, then the simulator runs its generation loop and data-plane hook.">
  <div class="vcr-node">
    <span class="vcr-node-kicker">frontend</span>
    <strong>vLLM frontend</strong>
    <span>Rust or Python</span>
  </div>
  <div class="vcr-connector"><span>ZMQ + msgpack</span></div>
  <div class="vcr-node">
    <span class="vcr-node-kicker">backend</span>
    <strong>vllm-vcr play</strong>
    <span>mock engine-core</span>
  </div>
  <div class="vcr-connector"><span>EngineInput</span></div>
  <div class="vcr-node">
    <span class="vcr-node-kicker">sim</span>
    <strong>generation loop</strong>
    <span>tokens, latency, scheduler</span>
  </div>
  <div class="vcr-connector"><span>KV hooks</span></div>
  <div class="vcr-node-stack">
    <div class="vcr-node">
      <span class="vcr-node-kicker">default</span>
      <strong>NoopDataPlane</strong>
      <span>control plane only</span>
    </div>
    <div class="vcr-node vcr-node-accent">
      <span class="vcr-node-kicker">feature = nixl</span>
      <strong>NixlDataPlane</strong>
      <span>CPU KV transfers</span>
    </div>
  </div>
</div>

The main pieces are:

- `connect_to_frontend` joins the frontend-owned handshake, reports ready, and
  opens the DEALER/PUSH data sockets.
- `src/io.rs` decodes incoming frames into `EngineInput` and writes `EngineOutput`
  messages back to the frontend.
- `src/engine.rs` owns scheduling, latency, token emission, LoRA accounting,
  prefix-cache state, and failure injection.
- `src/dataplane.rs` is the prefill/decode integration point. Prefill advertises
  KV metadata through `kv_transfer_params`; decode pulls those blocks. The default
  `NoopDataPlane` only exercises the control plane, while `NixlDataPlane` performs
  real NIXL reads when the `nixl` feature is enabled.

`record` uses the same boundary in proxy form: it presents as an engine to the
frontend, presents as a frontend to the real engine, relays frames unchanged, and
records timing/token metadata from decoded copies.
