# vLLM version mapping and release automation

Goal: support a rolling **N-3** window (latest vLLM release plus the three
before it) from a single repo, with automation that catches protocol drift the
day a new vLLM lands and ships one clearly-labelled artifact per supported line.

## Where vLLM version actually bites

Only one axis is a hard compile-time coupling. The rest are behavioral and get
caught by tests, not the compiler.

| Axis | Where it lives | Breaks on version bump? |
| --- | --- | --- |
| **Wire protocol** | `vllm-engine-core-client` git rev → the ~6 imported types (`HandshakeInitMessage`, `ReadyMessage`, `EngineCoreFinishReason`, `ModelDtype`, `encode/decode_msgpack`) | **Hard.** Cargo allows exactly one rev of a git dep per build. This is the whole problem. |
| **Registration schema** | `SimReadyResponse` in `crates/sim-protocol/src/frontend_connect.rs` | Soft, but already drifted (see `block_size` note below). We own a superset, so we absorb adds. |
| **Metrics surface** | `vllm:*` Prometheus gauges (e.g. the #45030 `lora_requests_info` move) | Soft. Behavioral, caught by `e2e*.sh`. |
| **Frontend** (`vllm-rs` / python) for e2e | `scripts/e2e*.sh` | Must match the protocol rev under test. |
| **Scheduler / step model** | `src/` step engine | Soft. Doesn't break the build, shifts replay error (chunked-prefill defaults etc.). Re-validated per line via the figure/gate harness. |

The trace and modeling crates (`sim-trace`, `src/` model) are already
vLLM-protocol-free. So "support N-3" is almost entirely a `sim-protocol` +
build-matrix problem, which is what makes the matrix approach cheap.

## Decision: build matrix, one artifact per line

Cargo cannot hold two revs of the same git dep in one build. A single binary
speaking four protocols would mean vendoring the wire structs into per-version
modules behind a trait. The surface is small (~6 types) so it isn't insane, but
msgpack shapes can diverge in ways a trait can't paper over, and it is real
standing maintenance.

We commit to the **build matrix**: one image per supported vLLM line, a
manifest-driven CI matrix, and the handshake `vllm_version` field used to reject
mismatches loudly rather than silently mis-speak the wire. Revisit the
single-binary path only if the protocol surface stays this small and we have a
concrete reason to ship one image.

## Source of truth: `compat.toml`

A manifest at repo root defines the support window. The manifest diff *is* the
release.

```toml
# compat.toml — the N-3 support window. Oldest line drops as N advances.

[[vllm]]
line = "0.10"                 # N (head)
tag  = "v0.10.1"              # vLLM release tag; also the e2e frontend version
protocol_rev = "abc123..."    # rev for vllm-engine-core-client at this line
fidelity_validated = true     # replay gates green against this line's captures
default = true                # what :latest / unsuffixed builds point at

[[vllm]]
line = "0.9"                  # N-1
tag  = "v0.9.2"
protocol_rev = "c9340e6f350a009cf835878abad2a0e379b9e6a4"
fidelity_validated = true

# ... N-2, N-3
```

Rules:

- **Pin to vLLM release tags, not arbitrary revs.** `protocol_rev` is the rev of
  the in-tree Rust crate that ships with that tag. Today's `c9340e6f` becomes
  "the rev for the 0.9 line."
- **Exactly one `default = true`.** That line is `:latest` and the unsuffixed
  build.
- **A line enters the window only when `fidelity_validated = true`.** New lines
  land as `false`, get capture+replay validation, then flip.

## Versioning: keep the two axes orthogonal

The sim's own semver (`0.1.0` → ...) tracks *its* features. vLLM compatibility
is build metadata, expressed in the image tag, never baked into the sim semver.
Conflating them is the classic mistake.

Image tags:

- `inference-sim:0.3.0-vllm0.10` — immutable, the real artifact (sim version ×
  vLLM line).
- `inference-sim:vllm0.10` — floating, → latest sim for that line.
- `inference-sim:latest` — sim-head × the `default = true` line.

## CI matrix mechanics

Cargo's git rev isn't env-overridable on stable, but `--config` patching is. A
small step reads `compat.toml` and, per matrix entry, overrides the rev:

```sh
cargo build --config \
  'patch."https://github.com/vllm-project/vllm.git".vllm-engine-core-client.rev="<protocol_rev>"'
```

(Alternative: template the workspace `Cargo.toml` rev per entry in a throwaway
build dir. Either works; the artifact is not committed.)

On `main` and tags the matrix builds + runs `e2e*.sh` + the replay gates against
**all four lines**. That is the payoff: the day vLLM N+1 lands, the matrix tells
us whether the wire still parses before we promote anything.

## Handshake version guard (do regardless of path)

The frontend registration already carries `vllm_version`
(`frontend_connect.rs:43`). On connect, assert the peer's version is in the
artifact's supported set and refuse with a clear error otherwise. That turns
"silent msgpack corruption" into "this image speaks vLLM 0.10, peer is 0.8,
abort." Cheap, high value, independent of the matrix.

## Rotation when N advances

When vLLM cuts N+1:

1. Add it to `compat.toml` with `fidelity_validated = false`.
2. Matrix builds it; run capture + replay to validate fidelity.
3. Flip `fidelity_validated = true`, move `default` to the new line.
4. Drop the now-N-4 line from the manifest.

## Build order

1. `compat.toml` + handshake version guard. Small, immediately useful, makes
   mismatches loud.
2. Manifest-driven CI matrix. The real automation payoff.
3. Per-version protocol modules behind a trait — only if we later want a single
   multi-version binary, and only while the surface stays ~6 types.

## Open coupling note: the `block_size` / registration drift

The python frontend's `EngineCoreReadyResponse` requires six fields including
`block_size` (tokens per KV block). The upstream Rust
`vllm-engine-core-client::EngineCoreReadyResponse` still has only five and is
**missing `block_size`** — confirmed at both the pinned `c9340e6` rev and
upstream `main` as of 2026-06-13. Re-encoding through the crate's struct silently
drops the field and the python frontend rejects the registration.

Our workaround is the forward-compatible path that already exists: the sim emits
its own complete `SimReadyResponse` superset, and the tap relays the real
engine's response bytes verbatim (immune to future adds). This is the kind of
schema drift the matrix is meant to track; if upstream ever adds `block_size` to
the Rust struct we can drop `SimReadyResponse` for that line, but until then the
superset stays. Keep the `sim_ready_response_carries_all_python_required_fields`
test as the canary.
