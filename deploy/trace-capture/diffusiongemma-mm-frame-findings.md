# DiffusionGemma multimodal trace-capture: frame-level findings

Date: 2026-06-16. Cluster: coreweave-waldorf, ns `weaton-dev`.
Model: `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` (FP8, H200).
Stack: vLLM `0.23.1rc1.dev32+g16e91176` (main `16e91176`), Python frontend +
headless engine, recording tap (`inference-sim-tap`) rebuilt against the same rev.

## TL;DR

Image-in / text-out **works** through a direct frontend→engine path (model
correctly described a red circle, HTTP 200, ~19s). Through the **tap**, image
requests hang. At the ZMQ frame level, the tap sees a multimodal request as a
**2-frame message with no multimodal tensor payload** — the same frame shape as a
plain text request. vLLM's own encoder is supposed to append the mm tensors as
**aux frames**, and they are not present in what the tap relays. That missing
aux-frame transport is the frame-data crux, and it lines up with a standing TODO
in vLLM's Rust engine-core-client.

## The ZMQ frame model (engine-core input socket)

The frontend sends each EngineCoreRequest to the engine's input socket as a
multipart message. From vLLM `v1/engine/core_client.py`:

```python
msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
self.input_socket.send_multipart(msg, copy=False)
```

`MsgpackEncoder.encode()` (`v1/serial_utils.py`) returns a **list of buffers**:
`[msgpack_blob, aux_buffer_0, aux_buffer_1, ...]`. Large tensors (e.g. multimodal
pixel values) are extracted into the aux buffers rather than inlined into the
msgpack blob, and ride as **additional ZMQ frames**.

After ROUTER→DEALER routing strips the identity frame, the engine (and the tap,
which acts as the engine toward the frontend) receives:

```
[ request_type (1 byte) , msgpack_blob , aux_tensor_frame_0 , ... ]
```

So: **text request → 2 frames; multimodal request → 3+ frames** (extra aux frames
for the image tensors).

## What the tap actually observed

Instrumented the tap forwarding loop (`crates/sim-tap/src/tap.rs`) to log frame
counts and sizes per message:

| Request | downstream→upstream frames | sizes (bytes) | engine output | client result |
|---|---|---|---|---|
| text ("Hi") | 2 | `[1, 27]` | immediate | 200 OK |
| text ("primary colors") | 2 | `[1, 32]` | immediate | 200 OK, ~20s |
| **image (red circle)** | **2** | **`[1, 378]`** | started ~18s later | **hangs (000)**, no trace record |

The image request arrives as **2 frames** — `request_type` (1B) + a **378-byte
msgpack blob** — with **no aux tensor frames**. 378 bytes cannot contain image
pixel data (the source JPEG alone was ~4 KB; processed pixel tensors are MB), nor
the full image-expanded prompt. So the multimodal tensors the Python encoder is
supposed to emit as aux frames are **absent from the message the tap relays**.

Consequence: the engine receives a request referencing multimodal content whose
tensors never arrive. It emits some output frames but the request **never
completes** (no `finish_reason`, no trace record, client times out).

## The open question (frame-attribution)

Two candidates for *why* the aux frames are missing at the tap:

1. **The frontend never emitted aux frames over this socket** (mm transferred by a
   different mechanism / stripped) — the issue is upstream of the tap.
2. **The frontend emitted them; the tap's recv dropped them.** The tap uses the
   pure-Rust `zeromq` crate (zmq.rs); libzmq (real engine, no-tap) receives all
   frames, zmq.rs may not for this multipart/large-frame shape.

This is **not yet resolved.** The decisive test — decode the 378B request in the
tap and log `mm_features.is_some()` + prompt-token count — was instrumented
(`tap-16e91176-diag2`) but we could not reliably capture an image `Add` at the
tap because the **frontend stalls before sending** it (see flakiness below), so
the `mm_features` field of the actual image request remains unconfirmed.

Relevant upstream signal — vLLM `rust/src/engine-core-client/src/client/imp.rs:212`:

```rust
// TODO: for `EngineCoreRequest`, split outbound tensor raw views into aux
// frames instead of always producing a single msgpack frame.
```

The Rust engine-core-client's request **encode** path does not yet split tensors
into aux frames. The tap forwards verbatim (it does not re-encode), but the
broader aux-frame transport for `EngineCoreRequest` is unfinished on the Rust
side, and a faithful tap/sim needs to handle it. This is the natural **upstream
PR target**.

## Confounder: rig instability (not just the tap)

The disaggregated rig (Python frontend + headless engine over ZMQ DP, ± tap) is
**flaky**: requests hang inconsistently, including **text**, even when both
containers report `ready=true` (the `/v1/models` readiness probe passes while
generation hangs). A/B run (2026-06-16 03:13Z), no-tap rig:

```
notap-text   STATUS=000 T=150s   (had answered text in ~20s earlier)
notap-img-1  STATUS=000 T=150s
notap-img-2  STATUS=000 T=150s
```

The one clean image success and the diag1 image capture were not reproducible on
later fresh pods. The rig README warns that a half-completed frontend↔engine
handshake hangs ("delete the pod"); readiness passing while generation hangs is
consistent with that. So the 2-frame finding is **real and repeatable when the
image request reaches the tap**, but reaching that point is itself unreliable.

## What is solid vs unconfirmed

Solid:
- Multimodal request reaches the tap as a 2-frame `[1, 378]` message, no aux
  tensor frames (diag1, repeatable when the request arrives).
- Text works through the tap; image does not complete.
- Image-in/text-out works on a direct (no-tap) path at least once → model + flags
  + vLLM 0.23.1 are correct; `min_vllm_version 0.24.0` from the recipe is
  conservative.
- Handshake-capability ruled out: the tap relays the **real engine's**
  `ready_response` verbatim (`connect_to_frontend_raw`), so the frontend sees the
  same capabilities as no-tap.

Unconfirmed:
- Whether the frontend emits mm aux frames over the input socket (and the tap
  drops them) vs never emits them. Needs a reliably-captured decoded image `Add`.
- Whether the rig flakiness (hangs on text too) shares a root cause with the mm
  issue or is independent (handshake fragility).

## Next steps

1. **Stabilize the rig** enough to reliably get one image `Add` to the tap
   (delete-pod-per-run per the handshake warning; confirm text is reliable first).
2. With a captured image `Add`: log `mm_features` + prompt-token count to settle
   frontend-strip vs tap-drop.
3. If tap-drop: inspect zmq.rs multipart recv for the aux-frame case; if
   frontend-strip / Rust-encode gap: the `client/imp.rs:212` aux-frame work is the
   upstream PR.

## Update 2026-06-16 (afternoon): transport ruled out, focus shifts to serde decode

New cluster-free evidence narrows the two candidates from "The open question"
above.

**zmq.rs transports large multipart aux frames fine.** New test
`crates/sim-tap/tests/multipart_frames.rs` drives the tap's exact downstream
pairing (libzmq-style ROUTER bind ↔ zmq.rs `DealerSocket` connect with the
engine `PeerIdentity`) and pushes 3-frame and 5-frame messages with large
(2 MiB, 1 MiB×3) aux frames. All arrive intact. So candidate #2 ("zmq.rs's recv
dropped the aux frames") is **refuted** for the pure-Rust path — the tap's
forwarder does not lose frames. (Caveat: the recv must run concurrently with the
send; a single-task send-then-recv deadlocks on backpressure for frames that
exceed the socket buffer — that is a *test harness* footgun, not a zmq.rs bug,
and not how the tap is structured.)

**Request/output serde structs match Python field-for-field at rev 16e91176.**
Compared the Rust `EngineCoreRequest` / `EngineCoreOutput` / `EngineCoreOutputs`
tuples against the deployed Python `msgspec.Struct`s (same checkout). Field order
and count match; Python's `omit_defaults` trailing-field shortening is covered by
the Rust `#[serde(default)]` + `DefaultFromSerde`. The multimodal subtree also
lines up at the type level: Python `MultiModalFeatureSpec` is a `@dataclass`
(msgspec → map) and the Rust `MmFeatureSpec` decodes a map; the mm field
factories (`batched`/`flat`/`shared`) and their dataclass fields
(`keep_on_cpu`, `slices`, `dim`, `batch_size`) match the Rust `deny_unknown_fields`
structs; tensor wire tuple `(dtype, shape, data)` with `data` as inline `Ext(3)`
or an integer aux-index matches `WireArrayData`.

**Why this still points at serde (the real failure mode).** Field-matching does
NOT prove decode: msgspec encodes subtleties (slice-as-tuple, set, enum-as-int,
the diffusion `is_embed` tensor, nested `NestedTensors` lists) that only real
bytes exercise. And the failure chain is precise: if `observe_request` fails to
decode the `Add`, the request is never inserted into the tap's `requests` map, so
every later engine output for it hits `requests.get(id) == None` and is dropped
silently → **engine emits output but no trace record**, exactly as observed. The
crate's `python_compat` test only covers a synthetic *inline-tensor* mm request,
never the large-tensor aux-frame path nor a real DiffusionGemma item.

**Decisive next test (in progress): ground-truth encode.**
`deploy/trace-capture/mm_encode_groundtruth.py` runs vLLM's REAL `MsgpackEncoder`
in a CPU-only pod (no GPU, pruner-immune, no flaky DP handshake) to emit the
exact wire bytes for an image `EngineCoreRequest`, printed as hex. Feeding frame 0
into `decode_msgpack::<EngineCoreRequest>` settles whether the Rust model
round-trips a real multimodal request. If it fails, the failing field is the fix
target (likely in the vendored `vllm-engine-core-client` crate → upstream PR). If
it succeeds, the blocker is upstream of the tap entirely (frontend wedges before
sending / mm preprocessing stall), not serde.

**Operational note:** the with-tap deployment (`trace-capture-diffusiongemma`) is
gpu-pruner-gated and gets scaled to `0/0` when GPU sits idle; the pruner keys off
GPU utilization with no clean annotation opt-out. Combined with the frontend
wedge, live tap capture is unreliable — hence the offline ground-truth approach.

## RESOLVED 2026-06-16: serde decode gap in `EngineCoreSamplingParams`

Ground-truth confirmed and fixed. The offline encoder
(`mm_encode_groundtruth.py`, run in a CPU-only pod with the deployed vLLM image)
emitted the exact wire bytes for an image `EngineCoreRequest`. Feeding frame 0
into the tap's own `decode_msgpack::<EngineCoreRequest>` reproduced the failure:

```
Decode error: missing field `temperature`
```

**Root cause.** Python `SamplingParams` is `msgspec.Struct(omit_defaults=True)`,
so it serializes as a MAP that omits every field sitting at its default. A
typical request's `sampling_params` is just
`{stop, stop_token_ids, bad_words, skip_reading_prefix_cache}` — `temperature`,
`max_tokens`, `seed`, `frequency_penalty`, `presence_penalty`, `logprobs`,
`prompt_logprobs`, `stop_token_ids`, `_eos_token_id`, `_all_stop_token_ids` are
all dropped. The Rust `EngineCoreSamplingParams` had `#[serde(default)]` on only
*some* fields, so `decode_msgpack` died on the first omitted required field. A
failed `observe_request` decode means the request is never inserted into the
tap's `requests` map, so every later engine output for it is dropped silently →
**output emitted, no trace record**. This is NOT multimodal-specific (any
default-sampling request breaks); it surfaced on the image path because that was
under investigation, and the tap's e2e test missed it (it encodes via the Rust
serializer, which writes all fields — only Python's `omit_defaults` exposes it).

The `EngineCoreOutput` / `EngineCoreOutputs` structs were already fully defended
with `#[serde(default)]`, so the output decode path needed no change.

**Fix.** Add `#[serde(default)]` (with `default_temperature` = 1.0 and
`default_max_tokens` = 16 for the non-zero Python defaults) to every omittable
field of `EngineCoreSamplingParams`. Lives in the vendored
`vllm-engine-core-client` crate. Patch:
`deploy/trace-capture/engine-core-sampling-params-serde-default.patch` (target
for the upstream PR). Verified locally via a `[patch]` in the workspace
`Cargo.toml` pointing at a fork of the rust workspace: the ground-truth decode
tests `crates/sim-tap/tests/mm_decode_groundtruth.rs` (large-tensor aux-frame +
inline-tensor) now pass, and the full `sim-tap` suite is green.

**Replay validated (code level).** The simulator's replay path already handles
diffusion blocks: `src/replay_steps.rs` consumes `itl_tokens` (several tokens per
step, "a whole block for diffusion"), and `tests/diffusion_replay_demo.rs`
replays a Gemma-diffusion-shaped trace (multimodal `prompt_tokens=259`, 8-token
blocks, recorded `output_token_ids`) byte-identically with the correct finish
reason. Burst-structure replay is covered by `tests/spec_replay_fidelity.rs`. So
both halves of the goal are demonstrated offline; full workspace suite is green
with the patch applied.

**Remaining to close the goal LIVE (operational, needs creds + a cooperative
rig):** push the fork branch to a git remote, repoint the dep at it (replace the
local-path `[patch]`), rebuild + push the tap image, redeploy, and capture a real
multimodal trace end to end. Plus submit the upstream PR
(`UPSTREAM-PR-engine-core-sampling-params.md`). Note the frontend-wedge confounder
is independent of the (now-fixed) serde bug.

## Update 2026-06-16 (late): libzmq->zmq.rs transport REFUTED as the blocker

A live tap log (ANSI-stripped) suggested the image request `chatcmpl-8b7c…` was
sent by the frontend (MMDBG: 6-frame `track=True`, "send await DONE") but never
arrived at the tap's downstream DEALER, while 2-frame text arrived fine — pointing
at the **libzmq ROUTER -> zmq.rs DEALER** hop, which the earlier `multipart_frames.rs`
test (zmq.rs -> zmq.rs) never exercised.

That hop is now tested directly with **real libzmq** and it does NOT lose frames.
`crates/sim-tap/tests/libzmq_interop.rs` drives a real-libzmq ROUTER (pyzmq via
`uv run --with pyzmq`, fixture `tests/fixtures/libzmq_router.py`) into the tap's
exact downstream socket (zmq.rs `DealerSocket` + engine `PeerIdentity`, as
`connect_to_frontend_raw` builds it). It sends, **copy=False** like vLLM:

- a 5-frame multimodal message FIRST — `[rt, msgpack(378), aux 16 MiB, aux 8 MiB,
  aux 8 MiB]` (32 MiB total) — then a 2-frame text message.

Both arrive **intact and in order**: the DEALER receives `[1, 378, 16M, 8M, 8M]`
then `[1, 27]`. So:

- zmq.rs does NOT drop or truncate large multipart on the real libzmq path.
- A big mm message does NOT wedge a following small text message (head-of-line
  ordering preserved).

So "image never reaches the tap" is **not a zmq.rs framing/recv bug**. The
remaining candidates are upstream of the tap's recv:

1. **libzmq ROUTER silently dropped it.** A ROUTER discards messages addressed to
   an identity it has no live route for (unless `ROUTER_MANDATORY`). If the tap's
   DEALER connection flapped/reconnected at send time, the mm `Add` is dropped at
   the frontend with no error — MMDBG's "send DONE" only means libzmq *accepted it
   into its send queue*, not that it was delivered.
2. **The frontend's actual wire send never happened** (encode wedge on the large
   tensors, or routed to a different DP socket), with "DONE" measuring an enqueue
   rather than the send.

Decisive next capture: correlate by `request_id` the frontend's send (instrument
the real `send_multipart` return, not `add_request`) against the tap's downstream
recv, AND set `ROUTER_MANDATORY` on a diagnostic frontend so an unroutable mm send
raises `EHOSTUNREACH` instead of vanishing. That distinguishes drop-at-router (1)
from never-sent (2). This is the disaggregation-stability confounder the README
warns about, independent of the (fixed) serde bug.

## Update 2026-06-16 (later still): silent ROUTER drop is the mechanism, and the
## frontend mm send-path differs from text — both verified against the deployed source

Two locally-verified facts close the loop on "image sent (DONE) but never arrives,
no error":

**1. A plain libzmq ROUTER silently drops a send to a missing/stale route; only
`ROUTER_MANDATORY` surfaces it.** Demonstrated with real libzmq (a pyzmq probe via
`uv run --with pyzmq`):

```
CASE1 plain-router send to missing route: RETURNED (no error) -> dropped silently
CASE2 mandatory-router raised: errno=65 (Host unreachable)
```

**2. The deployed frontend's engine-input ROUTER is created WITHOUT
`ROUTER_MANDATORY`, and without `ROUTER_HANDOVER` unless elastic-EP is on (it is
not for this deploy).** From the deployed rev (`vllm/utils/network_utils.py`
`make_zmq_socket`, `vllm/v1/engine/core_client.py:507,518-523`):

- `make_zmq_socket` never sets `ROUTER_MANDATORY`. So a send to an absent route is
  silently swallowed and `send_multipart` returns success.
- `enable_input_socket_handover = parallel_config.enable_elastic_ep` → handover is
  OFF for a normal DiffusionGemma deploy. So if the tap's zmq.rs DEALER (identity
  `engine_index 0`) reconnects, the ROUTER does not cleanly transfer the identity;
  sends can target the stale/dead pipe and drop.

**3. The multimodal send path is DISTINCT from text** (`core_client.py:861-873`
`_send_input`):

```python
msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
if len(msg) <= 3:                       # text: identity + type + 1 msgpack frame
    self.input_socket.send_multipart(msg, copy=False)        # fire-and-forget
    return
tracker = self.input_socket.send_multipart(msg, copy=False, track=True)  # mm: aux bufs
self.add_pending_message(tracker, request)
```

Text is fire-and-forget; multimodal uses `copy=False, track=True` plus
pending-message bookkeeping (the `track=True` MMDBG showed). Awaiting that tracker
("send await DONE") means **libzmq released the zero-copy tensor buffers, NOT that
the engine received the message.** A dropped (unroutable) mm send releases the
buffers too, so "DONE" is fully consistent with a silent drop.

**Conclusion.** The image-request hang is NOT a tap/zmq.rs framing bug (refuted
above) and NOT serde (fixed earlier). It is the frontend's ROUTER silently
dropping the mm `Add` when the tap's DEALER route is missing/stale at send time,
with no `ROUTER_MANDATORY` to raise and no handover to repair the route on
reconnect. Text survives because its send lands while the route is live (and is a
different, simpler code path).

**Decisive live confirmation + fix (rig-side, no code change here):**
1. Run a diagnostic frontend with `ROUTER_MANDATORY=1` on the engine-input socket
   → the dropped mm send raises `EHOSTUNREACH` instead of hanging. That proves
   drop-at-router and converts the silent hang into a loud, debuggable error.
2. Stabilize the tap's downstream DEALER so its route never goes stale (avoid
   reconnects), or enable input-socket handover, so the ROUTER always has a live
   route for `engine_index 0` when the frontend sends.

## RESOLVED (root cause) 2026-06-16 (evening): receiver-side wedge in the tap's
## zmq.rs DEALER, localized live with ROUTER_MANDATORY + a TCP Recv-Q snapshot

Brought the with-tap rig up (`trace-capture-diffusiongemma`, coreweave-waldorf)
with a one-line frontend `sitecustomize` monkeypatch: patch
`core_client.make_zmq_socket` to set `ROUTER_MANDATORY=1` on the frontend's input
ROUTER (handshake ROUTER untouched). Confirmed in the frontend log:
`[MMDBG] set ROUTER_MANDATORY=1 on input ROUTER socket`.

**Live results:**

- Text request: full path works, traced (`finish_reason=length`, TTFT 19.5s).
- Image request (`mm=True`, `_send_input_message len=6 track=True`):
  `send await DONE`, **no `EHOSTUNREACH`**. So the route is LIVE →
  **missing-route silent-drop (the earlier "ROUTER drop" theory) is REFUTED.**
- The image `Add` never reaches the tap (no `n_frames>2`, no decoded Add), AND the
  first image request **wedges the whole frontend→tap pipe**: every later request,
  *including text*, hangs with nothing reaching the tap. Deterministic
  head-of-line block; needs a pod restart.

**The decisive probe — TCP Recv-Q on the tap's data socket** (`ss -tnp` from an
ephemeral container in the pod netns; the data path is TCP loopback, distinct from
the `:5570` handshake port):

```
ESTAB  0         653548     127.0.0.1:45863  127.0.0.1:55868   ← frontend, 653 KB Send-Q (blocked)
ESTAB  11082163  0          127.0.0.1:55868  127.0.0.1:45863   ← tap fd=14, 11 MB UNREAD Recv-Q
```

The multimodal request's **~11 MB arrived at the tap's TCP socket and sits unread
in the kernel receive queue**; the frontend is flow-controlled (Send-Q backed up,
zero window) because the tap never drains. The tap process is alive but its
forwarding loop logged nothing after the text generation ended. So:

**Root cause: the tap's zmq.rs `DealerSocket` stops draining TCP on a large
multi-frame message. Not serde, not missing-route, not the engine, not the
libzmq→zmq.rs wire (bytes cross fine).**

**Where the wedge is — confirmed; exact trigger — partially open.** The tap's
forwarding loop (`tap.rs run_tap`) awaited `downstream_dealer.recv()` directly
inside a `tokio::select!`, which DROPS (cancels) the non-winning recv future every
iteration. zmq.rs's DEALER read is poll-driven through its `FairQueue` (no
independent reader task: `dealer.rs recv()` just awaits `fair_queue.next()`), so a
socket only drains its TCP buffer while its `recv()` is being polled. That made
`select!`-over-sockets the prime suspect.

Honest status on the trigger: three offline reproductions on macOS did NOT trigger
it, which rules out the simplest stories:
- `recv()` cancelled repeatedly via short `timeout`s — delivered fine (zmq.rs recv
  is cancel-safe in isolation).
- multi-socket `select!` (DEALER + PULL) with a 300-message output flood, then a
  32 MiB multimodal message — delivered fine.
- same, with a 15 s idle gap before the multimodal message — delivered fine.

So on macOS/kqueue the `select!`-over-sockets pattern survives. The live wedge is
real and deterministic (Linux/epoll, full 4-socket forwarding topology, real async
libzmq frontend) — at least one of those, most likely the full forwarding topology
or epoll readiness, is the remaining variable. (A Linux/docker repro of the
2-socket case was attempted to test the platform axis.) The 11 MB-unread Recv-Q
snapshot is unambiguous regardless of which: **the tap's zmq.rs DEALER stopped
draining its socket.**

**Fix (implemented in `tap.rs run_tap`).** Stop awaiting `Socket::recv()` inside
`select!` at all. Each receive socket (downstream DEALER, upstream PULL) now has a
dedicated drain task that calls `recv()` in a tight, never-cancelled loop and
forwards `(arrival, message)` over an unbounded `mpsc` channel. The proxy loop
selects over the CHANNELS (whose `recv()` is a cancel-safe queue pop) and does the
forwarding sends in the body, so both sockets are always being drained regardless
of which side is busy. This is exactly the tight-loop pattern the offline interop
test proves works at 32 MB, now applied to the real proxy. `tap_e2e` (the real
`run_tap` integration test) still passes. **Live verification (capturing a real
multimodal trace end to end through the rebuilt tap) is the remaining proof** —
the offline tests can't distinguish old vs new on macOS since both pass there.

## RESOLVED LIVE 2026-06-16 (evening): two fixes, end-to-end multimodal trace captured

Built the fixed tap on the cluster (kaniko, unprivileged; QEMU cross-build on the
mac segfaults rustc) and redeployed. A real inline-image request now flows AND is
recorded:

```
HTTP 200, 19 s, "This is a solid red circle on a white background."
tap: DIAG downstream->upstream request frames n_frames=5 sizes=[1, 6224, 161280, 15482880, 1091]
tap: DIAG decoded Add request mm_features=true prompt_tokens=1114
trace.jsonl: {meta} + {prompt_tokens:1114, output_tokens:64, ttft_ms:18821, finish:length, 64 output_token_ids}
```

The 15.5 MB aux pixel tensor (`15482880`) is exactly the large multi-frame message
that wedged before. It took TWO fixes, both in `crates/sim-tap/src/tap.rs`:

1. **Transport wedge (`run_tap`).** The proxy loop awaited `Socket::recv()` inside
   a `tokio::select!`. zmq.rs reads are poll-driven (the DEALER drains TCP only
   while its `recv()` is polled), so the cancel-every-iteration `select!` let a
   large multi-frame message stall mid-arrival — 11 MB sat unread in the DEALER's
   TCP Recv-Q, head-of-line-blocking the whole proxy. Fix: each receive socket is
   drained by a dedicated, never-cancelled task that forwards `(arrival, message)`
   over an unbounded `mpsc`; the proxy loop selects over the (cancel-safe) channels
   and does the forwarding sends in the body.

2. **Observation guard (`observe_request`).** Even once the message flowed, the
   request wasn't traced: `observe_request` required exactly 2 frames and dropped
   the 5-frame multimodal request (`WARN unexpected request frame count
   frame_count=5`), so it never entered the `requests` map and no record was
   written. Fix: accept `>= 2` frames and decode from `frames[1]` (the aux tensor
   frames are payload the tap does not need for timing).

Both are covered by tests: `tap_e2e` (real `run_tap`), the `tap::tests::
observe_request_tracks_multimodal_add_with_aux_frames` unit test (real groundtruth
blob + aux frame), and the `libzmq_interop` / `tap_topology_wedge` interop tests.
Saved artifact: `/tmp/dg-mm-trace/trace.jsonl` (one real DiffusionGemma multimodal
record). Image tag: `quay.io/wseaton/mock-engine-nixl:tap-16e91176-mmfix`.

Note on the earlier offline-repro failures: neither bug reproduces on macOS in
isolation (the wedge needs the real Linux/epoll + 4-socket forwarding runtime; the
observe guard only bites a real multi-frame request). Deploying to the rig where it
actually reproduces is what exposed both — offline tests never ran `observe_request`
against a real frame count.

## Repro artifacts (in `deploy/trace-capture/`)

- `diffusiongemma-capture.yaml` — frontend ⇄ tap ⇄ engine (tap image
  `tap-16e91176-diag2` has the frame logging).
- `diffusiongemma-notap.yaml` — frontend ⇄ engine direct (the working path).
- Tap instrumentation: `crates/sim-tap/src/tap.rs` `DIAG` log lines.
- Weights cached on PVC `dg-model-cache` (RWX) so redeploys skip the 25 GB pull.
- `crates/sim-tap/tests/libzmq_interop.rs` (+ `tests/fixtures/libzmq_router.py`) —
  real-libzmq ROUTER → tap zmq.rs DEALER multipart fidelity test (refutes the
  transport hypothesis). Needs `uv`; skips if absent.
- `crates/sim-tap/tests/tap_topology_wedge.rs` (+ `tests/fixtures/tap_topology.py`)
  — rebuilds the tap's DEALER+PULL `select!` shape against a real libzmq ROUTER+PUSH
  (text, output flood, then a 32 MiB multimodal message). Passes on macOS (the
  wedge did not reproduce there); kept as a regression guard for the receive path.
