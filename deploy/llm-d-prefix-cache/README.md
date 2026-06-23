# Mock precise-prefix-cache-routing deployment

A GPU-free stand-in for llm-d's **precise-prefix-cache-routing** deployment. A single

pool of mock model servers (real vLLM Rust frontend + our mock engine) publish KV-cache

events over ZMQ; the real llm-d Router (EPP) indexes them with `precise-prefix-cache-producer`

and routes new requests to whichever pod already holds the longest matching prompt prefix.

```text

client ─▶ Router/EPP ──(prefix-cache-scorer)──▶ mock pod (vllm-rs + mock-engine)

              ▲                                        │

              └────── KV-cache events (ZMQ :5556) ─────┘

        (precise-prefix-cache-producer subscribes, indexes BlockStored/BlockRemoved)

```

This is the control-plane counterpart to `../llm-d-pd` (which exercises the NIXL **data**

plane for P/D). Here there is no KV byte transfer: the value is the **event stream** that

drives cache-aware routing.

## What the mock emits

Each pod's mock engine (Phase 4) publishes, per scheduler step:

- `BlockStored` when prompt blocks are cached (carries the real prompt `token_ids`, which the

  EPP re-hashes at `blockSize: 64` to build its prefix tree),

- `BlockRemoved` on eviction,

- `AllBlocksCleared` on `reset_prefix_cache`.

Wire format is vLLM-compatible (verified against the real `llm-d-kv-cache` decoder via

`scripts/kv_events_smoke.sh`). Key knobs (set in `modelserver/deployment.yaml`):

| env | value | must match |

| --- | --- | --- |

| `MOCK_ENABLE_KV_EVENTS` | `1` | — |

| `MOCK_KV_EVENTS_ENDPOINT` | `tcp://*:5556` | EPP `podDiscoveryConfig.socketPort: 5556` |

| `MOCK_KV_EVENTS_TOPIC` | `kv@$(POD_IP):8000@<model>` | EPP `topicFilter: "kv@"` |

| `MOCK_TOKENS_PER_BLOCK` | `64` | EPP `tokenProcessorConfig.blockSize: 64` |

The pods carry `llm-d.ai/inference-serving: "true"` so the EPP's pod reconciler discovers

them and dials each one's `:5556` socket.

## Deploy

```bash

kubectl create namespace llm-d-pc-mock

helmfile -f deploy/llm-d-prefix-cache/helmfile.yaml apply

```

Teardown: `helmfile -f deploy/llm-d-prefix-cache/helmfile.yaml destroy` and delete the namespace.

## Verifying routing

Send several requests sharing a long common prefix; the EPP should pin them to the same pod

(watch the EPP logs / the `prefix-cache-scorer` scores, and the pods' `vllm:prefix_cache_*`

metrics). A cold prefix lands on any pod (broken ties via `no-hit-lru-scorer`), then repeats

of that prefix follow it.