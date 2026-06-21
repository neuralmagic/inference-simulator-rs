# Multi-version vLLM support: the compatibility shim

How the simulator builds against more than one vLLM line from a single `main`
branch. Read `versioning.md` first for the strategy (build matrix, one image per
line, `compat.toml` as source of truth) and `conformance.md` for capture/replay.
This doc covers the *code* side: how we absorb the protocol crate's API drift.

## Contents

- [The shape of the problem](#the-shape-of-the-problem)
- [Per-line builds: pin, don't patch](#per-line-builds-pin-dont-patch)
- [Capability cfgs](#capability-cfgs)
- [What the shim owns](#what-the-shim-owns)
- [Testing across lines](#testing-across-lines)
- [Current window](#current-window)
- [Follow-ups](#follow-ups)

## The shape of the problem

The wire protocol comes from one git dependency, `vllm-engine-core-client`,
which lives in the vLLM repo (`rust/src/engine-core-client/`). Its API drifts
across releases. Cargo can hold only one rev of a git dep per build, so each line
is a separate build (the matrix). The job of the shim is to let the same source
compile against each line's crate, isolating the divergences in one place.

## Per-line builds: pin, don't patch

Cargo rejects a `[patch]` that redirects a git dependency to a different rev of
the **same** source ("patches must point to different sources"). So the per-line
rev is swapped in `[workspace.dependencies]`, not via `--config patch`. `build.rs`
cannot do it either: dependency resolution happens before any build script runs.

`cargo xtask pin-vllm <line>` reads `compat.toml` and rewrites `Cargo.toml`: it sets
the `vllm-engine-core-client` rev to the line's `protocol_rev`, and inserts,
rewrites, or removes the fork `[patch]` to match the line's `patch_repo`/`patch_rev`
(a fork is a *different* source, so it is allowed to `[patch]`). The committed
`Cargo.toml` carries **no** `[patch]` block (the default line builds upstream), so
a forked line's block is inserted, not rewritten; the script strips any existing
block first, so it's idempotent. After the rewrite the rev no longer matches
`Cargo.lock`, so per-line builds omit `--locked`.

**Gotcha:** the script changes the manifest but not the environment, and
`build.rs` reads the line from `VLLM_TARGET_VERSION` (falling back to the
`compat.toml` default). So a per-line build must set **both**: run the script
*and* export `VLLM_TARGET_VERSION=<tag>`. The CI matrix does both; a local
older-line build must too:

```sh
cargo xtask pin-vllm 0.22
VLLM_TARGET_VERSION=v0.22.1 cargo build --workspace   # no --locked
```

## Capability cfgs

Where the crate's API genuinely diverges in a way owning a type can't hide (a
field whose *type* differs per line), the engine gates on a discrete capability,
not a version number. `build.rs` maps the target line to cfgs and declares them
with `cargo::rustc-check-cfg`:

- `vllm_lora_typed` — the crate exposes a typed `protocol::lora` module and
  `EngineCoreRequest.lora_request: Option<LoraRequest>`. On 0.23+. On 0.22 lora is
  opaque `rmpv::Value`.

cfgs from `build.rs` only reach the crate that owns the build script (the root
crate) and its targets (incl. its `tests/`). Keep cfg-gated code in the root crate.
rust-analyzer does not run `build.rs`, so it shows false-positive errors on the
inactive cfg branch; `cargo build` is the truth.

## What the shim owns

The principle: **own a tolerant decode wherever possible (no cfg), and reach for a
capability cfg only when a field's type differs per line.** Owned types
deserialize the same wire on every line (serde ignores unknown fields).

| Concern | Divergence | Shim |
| --- | --- | --- |
| Handshake harness types | `mock_engine` module absent before 0.23 (we never used its behavior, only structs) | `sim-protocol::mock_engine` owns `MockEngineSockets`/`MockEngineDataSockets`/`MockCoordinatorSockets` + `DEFAULT_MOCK_MAX_MODEL_LEN` + `default_dtype()` |
| Request-type frame | `EngineCoreRequestType::from_frame` is head-only | `sim-protocol::wire::request_type_from_frame` (1-byte decode) |
| Lora request | typed `LoraRequest` (0.23+) vs opaque rmpv (0.22) | `LoraSpec{lora_int_id,lora_name}` (own, decodes both) for the add_lora call + registry; `request_lora_name()` is the one `vllm_lora_typed`-gated fn for the `lora_request` field |
| Ready response | `EngineCoreReadyResponse.vllm_version` absent before 0.23 | tap decodes its own tolerant `CapturedReadyInfo{vllm_version:Option<String>}` |
| Utility request | `EngineCoreUtilityRequest` derives `Deserialize` only on 0.23+ (crate was client-only) | `engine_core::UtilityRequestSpec` (`Deserialize_tuple`, matches the wire tuple) |

The wire types still come from the crate (the matrix's whole point: catch drift at
compile time). The shim only covers the spots where our *decoding/server* role
needs something the client-oriented crate lacks on an older line.

## Testing across lines

The matrix runs per line (see `ci.yml`):

- `cargo build --workspace` — the "does the wire still compile" gate.
- `cargo test --workspace --lib` — unit tests (compile + pass on every line).
- `cargo test --test conformance` — the conformance runner (skips until goldens).

The full-stack e2e integration tests (`tests/engine_core_e2e.rs`,
`tests/tap_e2e.rs`) drive the *real* `EngineCoreClient`, whose API
is incomplete on older lines, so they are HEAD-client-targeted and run on the
default line via the `build-and-test` job, not per matrix leg. The lora lifecycle
e2e test is `#[cfg(vllm_lora_typed)]` so the workspace still compiles tests on
lines that have the typed client.

## Current window

- **nightly** (`nightly`): tracks vLLM main, `protocol_rev` is the latest post-merge
  commit (bumped regularly). No fork (main carries everything). build.rs treats the
  non-`vX.Y` tag as the newest line, so all capability cfgs are on. It exists to catch
  wire drift before a release lands: the live-HEAD nightly canary pins to upstream
  `main`, builds, runs unit tests, runs the HEAD-client protocol e2e suite, and runs
  the conformance runner. `fidelity_validated = false` because nightly has no golden
  capture set yet.
- **0.23** (`v0.23.0`, default): builds against upstream `17bc1445`, unit tests +
  conformance green. No fork (`#45848` is upstream here).
- **0.22** (`v0.22.1`): library + bins + unit tests + conformance all build; the
  shim absorbs the build-time drift. Builds against the 0.22 crate (`0decac0d`)
  `[patch]`ed to the `wseaton/vllm` serde-defaults fork (see below).
  `fidelity_validated = false` (no captures yet).
- **0.21** (`v0.21.0`): the Rust crate did not exist at the v0.21.0 tag, so there
  is no 0.21 rev to build against. It builds against the **same** 0.22 crate +
  fork as the 0.22 line; only its tag label and goldens differ. The capture is
  what proves the 0.21 engine's wire actually matches the 0.22 client types; if it
  diverges, 0.21 grows its own shim. `fidelity_validated = false`.

### The 0.22/0.21 serde-defaults fork

`EngineCoreSamplingParams` has no `#[serde(default)]` on its omittable fields at
`0decac0d` (the 0.22 crate), so the tap can't decode a real Python frontend's
msgspec `omit_defaults` request at *capture* time. This is a runtime decode issue,
not a build one, so it's a `[patch]` (a different source), not a `protocol_rev`
bump. The fix is `vllm-project/vllm#45848` backported onto the 0.22 base:

- branch `wseaton/serde-defaults-0.22` on `wseaton/vllm`, rev `b48f2434`.
- it adds the field defaults (temperature/top_p/repetition_penalty=1.0,
  max_tokens=16, the rest zero/None/empty) and nothing else.
- both the 0.21 and 0.22 lines carry it via `patch_repo`/`patch_rev` in
  `compat.toml`; `cargo xtask pin-vllm` inserts the `[patch]` block per leg (the
  committed `Cargo.toml` has none, since the default line builds upstream).

## Follow-ups

- **Capture a 0.22 golden** and flip `fidelity_validated = true` once it's
  uploaded, registered in `conformance/manifest.toml`, and the replay gate passes.
- **Capture a 0.21 golden** against the 0.22-crate frontend image to confirm the
  wire matches; flip `fidelity_validated = true` (or add a 0.21 shim if it
  diverges).
