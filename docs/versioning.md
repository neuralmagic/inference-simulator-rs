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
# compat.toml — the rolling support window. Oldest release line drops as N advances.

[[vllm]]
line = "nightly"              # tracks vLLM main for drift detection
tag  = "nightly"
protocol_rev = "9c7c74bf..."
fidelity_validated = false

[[vllm]]
line = "0.23"                 # current default line
tag  = "v0.23.0"              # vLLM release tag; also the e2e frontend version
protocol_rev = "17bc1445..."  # rev for vllm-engine-core-client at this line
fidelity_validated = false    # flips true once replay gates validate goldens
default = true                # what :latest / unsuffixed builds point at

[[vllm]]
line = "0.22"                 # older supported release line
tag  = "v0.22.1"
protocol_rev = "0decac0d..."
patch_repo = "https://github.com/wseaton/vllm.git"
patch_rev = "b48f2434..."
fidelity_validated = false
```

Rules:

- **Pin release lines to release tags, not arbitrary labels.** `tag` is the vLLM
  release label for the line. `protocol_rev` is the vLLM git rev whose in-tree
  `vllm-engine-core-client` crate the simulator builds against for that line.
- **Exactly one `default = true`.** That line is `:latest` and the unsuffixed
  build.
- **A line enters the window only when `fidelity_validated = true`.** New lines
  land as `false`, get capture+replay validation, then flip.

## Versioning: keep the two axes orthogonal

The sim's own semver (`0.1.0` → ...) tracks *its* features. vLLM compatibility
is build metadata, expressed in the image tag, never baked into the sim semver.
Conflating them is the classic mistake.

Image tags:

- `vllm-vcr:0.3.0-vllm0.23` — immutable, the real artifact (sim version ×
  vLLM line).
- `vllm-vcr:vllm0.23` — floating, latest sim for that line.
- `vllm-vcr:latest` — sim-head × the `default = true` line.

## CI matrix mechanics

The rev is swapped in `Cargo.toml`, NOT via `--config` patching. Cargo rejects a
`[patch]` that points a git dependency at a different rev of the **same** source
("patches must point to different sources"), so `--config 'patch...rev=...'`
fails for every line (including the head). The per-line build instead rewrites
the manifest in a throwaway checkout:

```sh
cargo xtask pin-vllm "<line>"   # reads compat.toml, edits Cargo.toml
cargo build --workspace               # no --locked: the rev changed
```

`cargo xtask pin-vllm` sets the rev in `[workspace.dependencies]` and rewrites or
removes the fork `[patch]` from the line's `patch_repo`/`patch_rev` (the head
line carries the #45848 fork; lines without a fork build against `protocol_rev`
upstream). `compat.toml` stays the single source of truth.

On `main` and tags the matrix builds + runs the replay gates against every line.
That is the payoff: the day vLLM N+1 lands, the matrix tells us whether the wire
still parses before we promote anything. Lines that are not yet
`fidelity_validated` run non-gating (job-level `continue-on-error`), so a line
with real API drift (e.g. a removed `mock_engine` module) surfaces as a
non-blocking annotation rather than blocking the merge.

## Handshake version guard (do regardless of path)

The frontend registration already carries `vllm_version`
(`frontend_connect.rs:43`). On connect, assert the peer's version is in the
artifact's supported set and refuse with a clear error otherwise. That turns
"silent msgpack corruption" into "this image speaks vLLM 0.23, peer is 0.22,
abort." Cheap, high value, independent of the matrix.

## Rotation when N advances

When vLLM cuts N+1:

1. Add it to `compat.toml` with `fidelity_validated = false`.
2. Matrix builds it; run capture + replay to validate fidelity.
3. Flip `fidelity_validated = true`, move `default` to the new line.
4. Drop the now-N-4 line from the manifest.

## Build order

1. `compat.toml` + handshake version guard. Small, immediately useful, makes
   mismatches loud. **Done.**
2. Manifest-driven CI matrix + conformance runner. The real automation payoff.
   **Done** (per-line build/unit/conformance; see `conformance.md`).
3. Compatibility shim for the protocol crate's per-line API drift. **Done** for
   the 0.22 line; see `multi-version-shim.md` (capability cfgs + owned/tolerant
   decodes, `cargo xtask pin-vllm`). This is what lets one `main` build against
   multiple lines without a single multi-version binary.
4. Nightly canary (`.github/workflows/nightly-canary.yml`). The `nightly` line is
   pinned and only moves when bumped; the canary instead pins to the LIVE upstream
   main HEAD each night (`cargo xtask pin-vllm nightly --rev <sha>`), builds +
   unit-tests, and publishes a rolling `nightly` prerelease with the sha in its
   notes. A red scheduled run is the early warning that upstream moved the
   engine-core protocol. **Done.**

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

## Conformance testing

The matrix above answers "does this line still build and parse the wire." Fidelity
(does the sim still reproduce the engine's behavior for a line) is answered by
conformance testing: a profile-once/replay-many loop that pairs a manifest of golden
captures with a GPU-free replay in CI. The capture runbook is
[conformance.md](conformance.md); this section ties the pieces together.

Three artifacts cooperate:

- **`compat.toml`** — the N-3 window (above). Each line carries `fidelity_validated`,
  which gates whether its conformance failures block promotion.
- **`conformance/manifest.toml`** — one `[[golden]]` entry per captured trace, with
  `line`, `bucket_path`, `sha256`, `config_hash`, `workload`, and `role` (`schema` or
  `fidelity`). The captures themselves are NOT in the repo; they live in a private
  bucket and CI fetches them by sha.
- **The capture runbook** ([conformance.md](conformance.md)) — how a golden gets
  captured on the GPU cluster, uploaded, and registered.

The CI flow (`.github/workflows/ci.yml`):

1. `compat-matrix` parses `compat.toml` into a per-line build matrix.
2. The `conformance` matrix job builds + tests each line against its own
   `protocol_rev` (the `--config` patch from "CI matrix mechanics"), then fetches that
   line's goldens by sha, verifies the sha256, and replays them GPU-free, asserting the
   trace's `config_hash` (the profile-once/replay-many cache key, `--expect-config-hash`).
3. Lines with `fidelity_validated = false` build and run conformance, but their
   fidelity failures are continue-on-error (see "Rotation when N advances"): a freshly
   added line gets signal without blocking the merge. Flip `fidelity_validated = true`
   once the golden validates, and the leg becomes a hard gate.

The replay-many half needs no GPU and is the same mechanism as the offline replay rig
(`deploy/trace-capture/offline-replay.yaml`): the python frontend talks to
`vllm-vcr play` serving the captured trace, with no real engine behind it. CI runs it
headlessly; the rig serves a live agent the same byte-identical streams.
