# Conformance capture runbook

How to capture a golden trace for one vLLM line, upload it to the private golden
bucket, register it in `conformance/manifest.toml`, and let CI flip that line to
`fidelity_validated = true`. This is the "profile-once" half of the
profile-once/replay-many model. The "replay-many" half is GPU-free and runs in CI
and on the offline replay rig.

For the version-mapping strategy this runbook serves (the N-3 window, `compat.toml`,
the build matrix, image tagging), see [versioning.md](versioning.md). For the trace
schema, see `crates/sim-trace/src/trace.rs`.

## Contents

- [Capture topology](#capture-topology)
- [What pins the vLLM version](#what-pins-the-vllm-version)
- [Capture hygiene](#capture-hygiene)
- [Stand up a capture per line](#stand-up-a-capture-per-line)
- [Fetch the trace + step stats](#fetch-the-trace--step-stats)
- [Compute the config hash](#compute-the-config-hash)
- [Upload + register the golden](#upload--register-the-golden)
- [Flip the line to validated](#flip-the-line-to-validated)
- [The GPU-free replay half](#the-gpu-free-replay-half)
- [Building the capture image on waldorf](#building-the-capture-image-on-waldorf)

## Capture topology

The capture stack records the real engine's wire traffic without changing it. Three
processes run as sidecars, the load generator drives them, and the tap writes the
trace:

```
loadgen (HTTP)            real vLLM engine (--headless, owns the GPU)
    |                          ^
    | /v1/...                  | ZMQ (engine handshake :5580)
    v                          |
vLLM frontend  --ZMQ-->  inference-sim-tap  (relays bytes verbatim, writes trace.jsonl)
 (vllm-rs serve)         frontend handshake :5570
```

The engine runs `--headless` and binds the GPU. The tap dials the engine's
handshake, the frontend dials the tap's handshake, so the tap sits on the wire
between frontend and engine and copies every frame to `trace.jsonl` (and, with
`--step-stats-out`, the per-step `SchedulerStats` sidecar). Load comes from
`deploy/trace-capture/loadgen.py`, driven in-pod by `validation-runner.sh`.

This is the same topology described in `traces/README.md` (the local-sim self-captures
swap the real engine for the sim, no GPU). For conformance goldens we always use the
real engine, because the point is to measure the engine vLLM actually ships for a line.

## What pins the vLLM version

The vLLM version under capture is pinned by the engine container image digest, not by
a tag. The per-line engine + tap/frontend images live in `models.toml` under
`[lines.<vllm_tag>]`, e.g.:

```toml
[lines."v0.23.0"]
engine_image = "docker.io/vllm/vllm-openai:v0.23.0@sha256:6d8429e3..."  # release, by digest
tap_image    = "ghcr.io/neuralmagic/inference-simulator-rs:vllm0.23"     # built for this line
```

For a conformance capture, pin the engine to the **release tag's** published image for
the line you are validating (the `tag` field in `compat.toml`, e.g. `v0.23.0`), by its
digest, that digest is the ground truth for "which vLLM this golden measures" (record it
in the manifest entry's provenance). The `nightly` line instead points at the post-merge
image at its `protocol_rev`. The tap and frontend stay on the per-line capture image
(`inference-sim-tap` + `vllm-rs`), built against that line's `protocol_rev` so the wire
parses, which is exactly what CI's `docker.yml` publishes as `:vllm<line>`.

## Capture hygiene

These rules are carried from `traces/README.md`; they are the difference between a
golden you can gate on and noise:

- **One workload per pod lifetime.** A half-rewired pod records garbage. Each capture
  pod runs exactly one workload and is torn down.
- **Never fit and gate on the same trace.** A trace used to fit the latency model
  cannot also be the fidelity gate; that is grading your own homework. Keep fitting
  captures and gate captures separate (the manifest `role` field: `schema` vs
  `fidelity`).
- **Never trust a single seed.** 3 of ~10 multiturn captures turned out anomalous and
  only multi-seed averaging exposed them. Capture multiple seeds for any gate; a lone
  capture is an anomaly waiting to happen.
- **Fetch promptly.** The cluster reaps idle GPU pods (~35 min) and `emptyDir` traces
  die with the pod. The loadgen container self-terminates after 2h if never fetched,
  so an abandoned run cannot squat on the GPU.

## Stand up a capture per line

Each capture is a Kueue-admitted Job on the GPU cluster (waldorf), so Kueue holds the Job
until GPU quota admits it and releases the GPU the moment the capture completes. The
capture matrix lives in `deploy/trace-capture/models.toml`: one `[[capture]]` per
model × scenario (`baseline` / `nocache` / `specdecode` / `multimodal`), each pinned to a
line's engine + tap/frontend images. `gen-capture-jobs.py` turns a target into a Job; the
scenario drives the engine *and* tap flags so the `config_hash` matches the engine config
(prefix-cache, spec-decode). To add a model or scenario, edit `models.toml`, no YAML by hand.

All capture Jobs target `conformance-queue` (`conformance-queue.yaml`), a dedicated Kueue
queue with a one-GPU quota, so they run **one at a time**: submit as many as you like and
they serialize, the rest wait pending (capture hygiene: one workload per pod lifetime, no
cross-capture interference). Each pod runs engine/tap/frontend as sidecars and
`validation-runner.sh` as the loadgen container, which runs `$PHASES`, marks the trace
line count at each phase boundary, then idles until the trace is fetched.

The flow (wrapped by the justfile):

```bash
just conformance-queue                  # apply the one-GPU queue (once)
just conformance-list                   # see the targets in models.toml
just conformance-capture qwen3-8b       # ship loadgen scripts + submit the Job

# raw equivalent:
python3 deploy/trace-capture/gen-capture-jobs.py qwen3-8b | kubectl apply -f -

# wait for the loadgen to finish (it logs "waiting for fetch"):
kubectl logs -f job/trace-qwen3-8b -c loadgen
```

To capture a new line (e.g. a new release or `nightly`), add a `[lines.<vllm_tag>]` entry
(engine image digest + the tap/frontend image built for that line) and point captures at it.

## Fetch the trace + step stats

Fetch before the reaper window closes. The marker is the loadgen log line "waiting for
fetch":

```bash
POD=$(kubectl get pod -n weaton-dev -l job-name=trace-validation-cached -o name)

# The trace, and the per-step SchedulerStats sidecar if captured.
kubectl exec -n weaton-dev "$POD" -c loadgen -- cat /trace/trace.jsonl > trace.jsonl
kubectl exec -n weaton-dev "$POD" -c loadgen -- cat /trace/step-stats.jsonl > step-stats.jsonl

# Let the Job complete and release the GPU.
kubectl exec -n weaton-dev "$POD" -c loadgen -- touch /trace/fetched
```

Compress before upload (`.jsonl.gz`); the trace tooling and the sim read gzip
transparently.

## Compute the config hash

The `config_hash` is the profile-once/replay-many cache key. It fingerprints the
capture config (model, GPU, TP, scheduler flags) so a trace cannot be replayed against
a config it was not captured for. The tap stamps it into the trace metadata line via
`--config-hash`, and the sim asserts it at replay via `--expect-config-hash` (see
`crates/sim-tap/src/bin/inference_sim_tap.rs` and `crates/sim-trace/src/trace.rs`).

The recipe is `ConfigFingerprint` in `crates/sim-trace/src/config_hash.rs`: a lowercase-hex
SHA-256 over a versioned, order-fixed canonical form (scheme tag `config-fingerprint-v3`)
of three inputs:

- `gpu` (hardware, e.g. `H200`)
- `vllm_tag` (the line tag, e.g. `v0.23.0`)
- `engine_config` — a canonical, sorted digest of the *deployed behavioral* engine flags
  (model, tensor-parallel, gpu-mem-util, max-model-len, max-num-seqs, block-size,
  enforce-eager, prefix-caching, speculative config, ...), built by the capture driver.

The design rule is **hardware + version + deployed behavioral flags**, deliberately *not*
a hand-curated field list (the per-knob trap, you'd forever be adding the next one) and
*not* the entire resolved config: engine defaults aren't enumerated because `vllm_tag`
pins them, and transport/addressing flags are excluded. `vllm_tag` is the line tag, NOT
the engine's raw reported version (a dev build `0.23.0.dev1+g...`, not reproducible across
rebuilds); the reported version is recorded separately in the trace meta (`vllm_version`).
`engine_config` is what keeps a cache-off, spec-decode, or eager-vs-graph capture from
sharing a fingerprint with another deployment of the same model/hardware. If the input set
ever changes, bump the scheme so old hashes deliberately stop matching. (Older goldens keep
their own scheme's hashes and stay valid: the sim compares the stamped hash, never recomputes.)

Two ways to get the hash:

- Run the tap with `--vllm-version <tag>`, `--gpu <type>`, and `--engine-config <canonical>`
  (plus `--model`/`--tp`/`--block-size`/`--max-num-seqs` for the readable trace meta) and
  let it compute the fingerprint, stamping `config_hash` into the trace `meta`. This is the
  default; `gen-capture-jobs.py` builds the `--engine-config` string (it owns the engine
  flags, the tap can't observe them on the wire), and the manifest entry just copies the hash.
- Or pass `--config-hash <hash>` explicitly to override the computed value.

## Upload + register the golden

Goldens are NOT committed to the repo: they are measurement data, some are large, and
token-recording captures carry model content. They live under the `conformance/` prefix
of `s3://llm-d-artifacts-783952637884`; CI fetches them by sha. The GHA runner assumes
the least-privilege `llm-d-conformance-ci` role (GitHub OIDC, GetObject on
`conformance/*` only), defined in the llm-d-infra terraform
(`aws/783952637884/bootstrap/iam.tf`).

```bash
# 1. Upload to the conformance/ prefix (use credentials with write access; the CI
#    role is read-only).
# Path convention: conformance/<vllm_tag>/<gpu>/<model>/<workload>[-<seed>].jsonl.gz
# where <vllm_tag> is the release tag (v0.23.0) or "nightly" (tracks main). These
# mirror config_hash inputs (vllm_tag/gpu/model), so captures across builds, hardware,
# and models don't collide. CI mirrors the full key locally.
aws s3 cp trace.jsonl.gz \
  "s3://llm-d-artifacts-783952637884/conformance/<vllm_tag>/<gpu>/<model>/<workload>-<seed>.jsonl.gz"

# 2. Record the sha256 (the manifest key CI verifies after fetch).
sha256sum trace.jsonl.gz
```

Add a `[[golden]]` entry to `conformance/manifest.toml`:

```toml
[[golden]]
line        = "0.23"                                  # matches a compat.toml line
bucket_path = "conformance/v0.23.0/H200/Qwen/Qwen3-8B/multiturn-seed7.jsonl.gz" # key within the bucket
sha256      = "<sha256 of the uploaded .gz>"          # CI verifies this post-fetch
config_hash = "<the trace's config_hash>"             # replay asserts --expect-config-hash
workload    = "multiturn"                              # human-readable workload label
role        = "fidelity"                              # "schema" or "fidelity"
```

A line gains `has_goldens = true` in the CI matrix automatically once it has a
`[[golden]]` entry; that is what turns on the AWS fetch leg for it (lines without
goldens skip credentials entirely).

`role = "schema"` captures gate the wire protocol parsing (cheap, never a fidelity
gate); `role = "fidelity"` captures gate replay accuracy. Per the hygiene rules,
fitting captures and fidelity-gate captures must be different traces, and a fidelity
gate should reference multiple seeds.

## Flip the line to validated

A new line lands in `compat.toml` with `fidelity_validated = false`. The matrix builds
it and runs conformance, but a fidelity failure does not block promotion
(continue-on-error in the non-gating conformance step, see `.github/workflows/ci.yml`).

Once the golden(s) are uploaded, registered, and the replay gates green for that line:

1. Flip `fidelity_validated = true` for the line in `compat.toml`.
2. On the next run the conformance leg for that line becomes a hard gate.
3. When the line becomes the head, move `default = true` to it (`:latest` follows the
   default line) and drop the now-N-4 line per `versioning.md`.

## The conformance runner

The runner is the `tests/conformance.rs` integration test. It is manifest-driven and
runs entirely on CPU, so the CI matrix invokes it per line after building against that
line's `protocol_rev`:

```bash
CONFORMANCE_GOLDENS=/path/to/fetched cargo test --test conformance -- --nocapture
```

For the line this build targets (`VLLM_TARGET_VERSION`, stamped from `compat.toml`), it
reads `conformance/manifest.toml`, and for each `[[golden]]` on that line resolves the
fetched file at `$CONFORMANCE_GOLDENS/<basename(bucket_path)>`. It then asserts:

- line: the golden's recorded `vllm_version` is on the same `major.minor` line as the
  build (`assert_same_line`).
- provenance: the trace's `config_hash` equals the manifest entry's `config_hash`.
- schema (`role = "schema"`): the sim's `SimReadyResponse` carries every field the
  captured engine emitted (`assert_ready_response_schema`, decoded from the meta's
  `ready_response_hex`). This is the automated, per-line generalization of the
  `block_size` canary: a new line that grows a registration field fails here.
- fidelity (`role = "fidelity"`): boots the sim on the golden under the `config_hash`
  gate and asserts every recorded token stream replays byte-identically.

It skips cleanly (passes without asserting) when there are no goldens for the line or
`$CONFORMANCE_GOLDENS` is unset, which is the normal state until captures exist. Set
`$CONFORMANCE_MANIFEST` to point at an alternate manifest. The pure assertions live in
`src/conformance.rs` and are unit-tested independently of any real capture.

## The GPU-free replay half

The replay-many half needs no GPU anywhere. `deploy/trace-capture/offline-replay.yaml`
runs the same python frontend with `inference-sim` (not a real engine) in the engine
slot, serving a captured trace with content-keyed matching:

```
agent (port-forward :8000)
    | HTTP
frontend  (vllm serve --data-parallel-size-local 0, no GPU)
    | ZMQ
sim       (inference-sim --replay-tokens trace.jsonl --replay-match prefix)
```

The frontend MUST run the same model/tokenizer as the capture (prefix matching is on
token ids), and stays on the protocol-pin image for the line. This is the same
mechanism CI's conformance step uses headlessly: fetch the golden by sha, then replay
it against the sim built for that line and assert `--expect-config-hash`.

## Building the capture image on waldorf

The tap + frontend capture image is `linux/amd64`; cross-building it on Apple Silicon
under QEMU is unreliable (rustc SIGSEGVs under emulation). Build natively:

- `just image-build && just image-push` builds and pushes the `linux/amd64` image
  for the compat.toml default line (slow under emulation; run on an amd64 host).
- `just image-build-line <line>` builds the image for an older line, e.g.
  `just image-build-line 0.22`: it pins `Cargo.toml` to that line's rev/fork with
  `ci/pin-vllm-rev.py`, stamps `VLLM_TARGET_VERSION`, and builds the vllm-rs frontend
  from the same source as the tap. (It leaves `Cargo.toml`/`Cargo.lock` rewritten;
  `git checkout Cargo.toml Cargo.lock` to restore.)
- On Apple Silicon, use the **build-on-waldorf** flow to build natively on the cluster
  with an unprivileged kaniko pod instead of cross-building locally.

CI publishes these per line automatically (`.github/workflows/docker.yml`): the same
pin step, `VLLM_TARGET_VERSION`, and frontend-source wiring run per matrix leg, so the
floating `vllm<line>` image is built against that line's wire. Build locally only when
you need an image ahead of a CI publish.
