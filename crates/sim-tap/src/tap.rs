//! Engine-core recording tap: a transparent ZMQ proxy that sits between a real
//! frontend and a real engine-core, recording per-request timing into a JSONL
//! trace file.
//!
//! The tap plays both protocol roles:
//!   - Downstream (toward the real frontend): acts as an ENGINE, using the same
//!     `connect_to_frontend` handshake that `run_engine` uses.
//!   - Upstream (toward the real engine): acts as a FRONTEND, binding its own
//!     handshake/input/output sockets and running the HELLO/INIT/READY
//!     choreography.
//!
//! Frames move VERBATIM in both directions. The tap decodes COPIES of each
//! frame for observation only; a frame that fails to decode is still forwarded.
//!
//! ## Limitations (prototype)
//!
//! - Single engine, single client (client_index 0).
//! - `parallel_config_hash` from the real engine's ReadyMessage is not relayed
//!   downstream (MockEngineConfig does not surface it); only relevant for DP > 1.
//! - Abort handling: aborted requests are discarded with a debug log and do not
//!   appear in the trace. This avoids polluting TTFT/ITL stats with incomplete
//!   data.
//! - No coordinator pass-through.
//!
//! ## Multi-token chunks
//!
//! When a single output carries N > 1 tokens (speculative decoding, diffusion
//! blocks, or frontend output batching), the chunk contributes ONE `itl_ms`
//! gap (the full time since the prior chunk) and its token count goes into
//! `itl_tokens`, preserving the burst structure. `itl_tokens` is omitted from
//! records whose every chunk carried one token, so plain autoregressive
//! captures are unchanged.

use std::collections::HashMap;
use std::io::Write;

use anyhow::{Context as _, Result, anyhow, bail};
use sim_protocol::mock_engine::MockEngineSockets;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::protocol::handshake::{
    HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreRequestType, decode_engine_core_outputs,
    decode_msgpack, encode_msgpack,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSocket, ZmqMessage};

use sim_protocol::kvparams::{extract_kv_params, kv_flag};
use sim_protocol::wire::{request_type_from_frame, trace_finish_reason};
use sim_trace::config_hash::ConfigFingerprint;
use sim_trace::trace::{ItlContext, TraceMeta, TraceRecord, append_record};

use sim_protocol::step_stats::{StepStatsRecord, append_step_stats};

/// Batch context attached to a single ITL gap (one gap per output chunk).
struct GapSample {
    gap_ms: f64,
    /// Tokens delivered by the chunk that closed this gap (> 1 under
    /// speculative decoding or diffusion blocks).
    tokens: u32,
    /// Engine-reported running count for the step closing this gap.
    running: u32,
    /// Prompt tokens that finished prefill in the step closing this gap.
    prefill_tokens: u32,
}

/// Per-request observation state maintained by the tap.
struct RequestState {
    arrival: Instant,
    prompt_tokens: usize,
    /// Chained per-block prefix fingerprints of the prompt (see
    /// `trace::prompt_block_hashes`).
    block_hashes: Option<Vec<u64>>,
    /// `do_remote_prefill` request (P/D decode side): its first output reflects a
    /// KV pull, not local prefill compute, so it never counts as interference.
    remote_prefill: bool,
    /// Last output instant; `None` until the first token arrives.
    last_output: Option<Instant>,
    /// Accumulated output token count.
    output_tokens: usize,
    /// Time to first token in milliseconds (set on first output).
    ttft_ms: Option<f64>,
    /// One entry per inter-token gap, with its step's batch context.
    gaps: Vec<GapSample>,
    /// Cached token count from prefill_stats.
    cached_tokens: usize,
    /// Concurrency snapshot at arrival.
    concurrency: u64,
    /// Accumulated output token ids; only filled when the tap records tokens.
    output_token_ids: Vec<u32>,
}

impl RequestState {
    fn new(req: &EngineCoreRequest, arrival: Instant, block_size: usize, concurrency: u64) -> Self {
        // P/D decode side: this request's prefill ran on another node, so its
        // first output here is a KV-pull completion, not local prefill compute,
        // and must not count as batch interference.
        let remote_prefill = extract_kv_params(req)
            .map(|kv| kv_flag(&kv, "do_remote_prefill"))
            .unwrap_or(false);
        Self {
            arrival,
            prompt_tokens: req.prompt_token_ids.as_ref().map(Vec::len).unwrap_or(0),
            block_hashes: req
                .prompt_token_ids
                .as_deref()
                .and_then(|tokens| sim_trace::trace::prompt_block_hashes(tokens, block_size)),
            remote_prefill,
            last_output: None,
            output_tokens: 0,
            ttft_ms: None,
            gaps: Vec::new(),
            cached_tokens: 0,
            concurrency,
            output_token_ids: Vec::new(),
        }
    }

    fn awaiting_first_token(&self) -> bool {
        self.last_output.is_none()
    }

    /// Fold one non-empty output chunk into the timing state: the first chunk
    /// sets TTFT, later chunks add ITL gaps with their step's batch context.
    fn record_chunk(
        &mut self,
        arrival: Instant,
        num_new_tokens: usize,
        step_running: u32,
        step_prefill_tokens: u32,
    ) {
        match self.last_output {
            None => self.ttft_ms = Some(ms_between(self.arrival, arrival)),
            Some(prev) => {
                // One gap per chunk, however many tokens it carried: the burst
                // structure of multi-token steps (spec decode, diffusion
                // blocks) is data, not noise.
                //
                // scheduler_stats are post-step, so a step in which requests
                // finish undercounts what ran during it (down to 0 on the last
                // step). The request owning this gap was certainly running, so
                // floor at 1.
                self.gaps.push(GapSample {
                    gap_ms: ms_between(prev, arrival),
                    tokens: num_new_tokens as u32,
                    running: step_running.max(1),
                    prefill_tokens: step_prefill_tokens,
                });
            }
        }
        self.output_tokens += num_new_tokens;
        self.last_output = Some(arrival);
    }

    /// Finalize into a trace record. `capture_start` is the zero point for the
    /// arrival_ms column.
    fn into_record(
        self,
        capture_start: Instant,
        finish_reason: EngineCoreFinishReason,
        record_tokens: TokenRecording,
    ) -> TraceRecord {
        let (itl_ms, itl_tokens, itl_ctx) = if self.gaps.is_empty() {
            (None, None, None)
        } else {
            let mut itl_ms = Vec::with_capacity(self.gaps.len());
            let mut tokens = Vec::with_capacity(self.gaps.len());
            let mut num_running = Vec::with_capacity(self.gaps.len());
            let mut prefill_tokens = Vec::with_capacity(self.gaps.len());
            for gap in &self.gaps {
                itl_ms.push(gap.gap_ms);
                tokens.push(gap.tokens);
                num_running.push(gap.running);
                prefill_tokens.push(gap.prefill_tokens);
            }
            // All-ones is the schema default; omit it so plain autoregressive
            // captures serialize exactly as before.
            let itl_tokens = tokens.iter().any(|&t| t > 1).then_some(tokens);
            (
                Some(itl_ms),
                itl_tokens,
                Some(ItlContext {
                    num_running,
                    prefill_tokens,
                }),
            )
        };
        TraceRecord {
            prompt_tokens: self.prompt_tokens,
            cached_tokens: self.cached_tokens,
            output_tokens: self.output_tokens,
            ttft_ms: self.ttft_ms.unwrap_or(0.0),
            itl_ms,
            itl_tokens,
            itl_summary: None,
            concurrency: self.concurrency,
            arrival_ms: Some(ms_between(capture_start, self.arrival)),
            itl_ctx,
            block_hashes: self.block_hashes,
            output_token_ids: (record_tokens == TokenRecording::On)
                .then_some(self.output_token_ids),
            finish_reason: Some(trace_finish_reason(finish_reason)),
        }
    }
}

fn ms_between(from: Instant, to: Instant) -> f64 {
    to.duration_since(from).as_secs_f64() * 1000.0
}

/// Whether the tap writes output token ids into trace records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum TokenRecording {
    /// Timing + prefix hashes only (default): the trace stays free of user
    /// content and can be shared freely.
    #[default]
    Off,
    /// Also record each request's `output_token_ids`. With the same tokenizer
    /// these decode back to the generated text, so the trace carries user
    /// content.
    On,
}

/// Configuration for the tap proxy.
pub struct TapConfig {
    /// Handshake address of the real frontend to connect to (downstream).
    pub frontend_handshake: String,
    /// Handshake address the tap binds for the real engine (upstream).
    pub engine_handshake: String,
    /// Address the tap binds for engine input (ROUTER socket, upstream).
    pub input_address: String,
    /// Address the tap binds for engine output (PULL socket, upstream).
    pub output_address: String,
    /// Token-block size for prompt prefix fingerprints (should match the
    /// engine's prefix-cache block size).
    pub block_size: usize,
    /// Record each request's output token ids into the trace
    /// (`TraceRecord::output_token_ids`).
    pub record_tokens: TokenRecording,
}

/// Sockets and handshake payloads captured while connecting the real engine.
struct UpstreamEngine {
    input: RouterSocket,
    output: PullSocket,
    ready_message: ReadyMessage,
    /// The engine's input-socket registration payload (python
    /// `EngineCoreReadyResponse`), kept as raw bytes and relayed verbatim to
    /// the downstream frontend: re-encoding through the crate's struct drops
    /// fields the python frontend requires (`block_size`), and raw relay is
    /// immune to schema drift entirely.
    ready_response_payload: Vec<u8>,
    /// The engine's reported vLLM version, decoded from the registration ready
    /// response. Ground truth for the version guard and stamped into the trace
    /// meta so a capture is self-describing.
    vllm_version: String,
}

/// Split a message into its expected `[identity, payload]` frame pair.
fn into_two_frames<T>(frames: Vec<T>, what: &str) -> Result<[T; 2]> {
    <[T; 2]>::try_from(frames)
        .map_err(|frames| anyhow!("expected 2 frames for {what}, got {}", frames.len()))
}

/// Run the upstream handshake: bind the handshake socket, wait for the engine's
/// HELLO, reply with INIT pointing to the tap-bound input/output sockets, wait
/// for READY, then wait for the engine to register on the input socket.
async fn upstream_handshake(config: &TapConfig) -> Result<UpstreamEngine> {
    // Bind input (ROUTER) and output (PULL) sockets for the upstream engine.
    let mut input_socket = RouterSocket::new();
    input_socket
        .bind(&config.input_address)
        .await
        .with_context(|| format!("binding tap input socket at {}", config.input_address))?;

    let mut output_socket = PullSocket::new();
    output_socket
        .bind(&config.output_address)
        .await
        .with_context(|| format!("binding tap output socket at {}", config.output_address))?;

    // Bind handshake socket and wait for the engine's HELLO.
    let mut handshake_socket = RouterSocket::new();
    handshake_socket
        .bind(&config.engine_handshake)
        .await
        .with_context(|| {
            format!(
                "binding tap handshake socket at {}",
                config.engine_handshake
            )
        })?;

    info!(
        engine_handshake = %config.engine_handshake,
        "waiting for real engine HELLO"
    );

    let hello_msg = handshake_socket
        .recv()
        .await
        .context("receiving engine HELLO")?;
    let [engine_identity, hello_payload] = into_two_frames(hello_msg.into_vec(), "engine HELLO")?;
    let ready_message: ReadyMessage =
        decode_msgpack(&hello_payload).map_err(|e| anyhow!("decoding HELLO: {e}"))?;

    debug!(?ready_message, "received engine HELLO");

    // Send INIT with our bound input/output addresses.
    let init_message = HandshakeInitMessage {
        addresses: HandshakeAddresses {
            inputs: vec![config.input_address.clone()],
            outputs: vec![config.output_address.clone()],
            coordinator_input: None,
            coordinator_output: None,
            frontend_stats_publish_address: None,
        },
        parallel_config: Default::default(),
    };
    let init_payload = encode_msgpack(&init_message).map_err(|e| anyhow!("encoding INIT: {e}"))?;
    // ROUTER messages are [identity, payload].
    let init_zmq = ZmqMessage::try_from(vec![engine_identity, init_payload.into()])
        .map_err(|e| anyhow!("building INIT message: {e}"))?;

    handshake_socket
        .send(init_zmq)
        .await
        .context("sending INIT to engine")?;
    debug!("sent INIT to engine");

    // Wait for READY. The decode is validation only; an incompatible protocol
    // version should fail loudly instead of recording a trace against a broken
    // pair.
    let ready_msg = handshake_socket
        .recv()
        .await
        .context("receiving engine READY")?;
    let [_, ready_payload] = into_two_frames(ready_msg.into_vec(), "engine READY")?;
    let ready: ReadyMessage =
        decode_msgpack(&ready_payload).map_err(|e| anyhow!("decoding READY: {e}"))?;
    debug!(ready_payload = ?ready, "received engine READY");

    // Wait for the engine to register on the input socket (sends its identity +
    // ready response). The decoded copy is observation-only; the raw bytes are
    // what get relayed.
    let reg_msg = input_socket
        .recv()
        .await
        .context("receiving engine registration on input socket")?;
    let [_, reg_payload] = into_two_frames(reg_msg.into_vec(), "engine registration")?;
    // Decode our own tolerant view rather than the crate's EngineCoreReadyResponse:
    // the only field we read is vllm_version, and that field is absent from the
    // struct before 0.23, so decoding the crate type would fail to build on older
    // lines. Owning the subset (serde ignores the rest) keeps the tap version-portable.
    let info: CapturedReadyInfo = decode_msgpack(&reg_payload)
        .map_err(|e| anyhow!("decoding engine registration ready response: {e}"))?;
    debug!(vllm_version = ?info.vllm_version, "engine registered on tap input socket");

    Ok(UpstreamEngine {
        input: input_socket,
        output: output_socket,
        ready_message,
        ready_response_payload: reg_payload.to_vec(),
        vllm_version: info.vllm_version.unwrap_or_default(),
    })
}

/// The single field the tap reads off the engine's registration ready response.
/// Decoded tolerantly so it works on lines whose `EngineCoreReadyResponse` lacks
/// `vllm_version` (absent before 0.23); there it stays `None`.
#[derive(serde::Deserialize)]
struct CapturedReadyInfo {
    #[serde(default)]
    vllm_version: Option<String>,
}

/// Refuse to record when the engine speaks a different vLLM line than the
/// capture is labelled for. Matching is on the `major.minor` line (the engine
/// reports a full version like `0.23.0.dev1+g...`, the label is a tag like
/// `v0.23.0`), so patch/dev/build suffixes don't trip a false mismatch. This
/// turns "silently recorded a mislabelled capture" into a loud abort.
fn guard_vllm_version(expected_tag: &str, engine_reported: &str) -> Result<()> {
    match (
        sim_compat::minor_line(expected_tag),
        sim_compat::minor_line(engine_reported),
    ) {
        (Some(want), Some(got)) if want == got => Ok(()),
        (Some(want), Some(got)) => bail!(
            "vLLM line mismatch: tap labelled for {expected_tag} (line {want}), but the engine \
             reported {engine_reported} (line {got}); refusing to record a mislabelled capture"
        ),
        _ => {
            // Can't parse one side; don't hard-fail on an unexpected version
            // string, just warn so the capture still happens.
            warn!(
                expected_tag,
                engine_reported,
                "could not parse a major.minor line from a vLLM version; skipping the guard"
            );
            Ok(())
        }
    }
}

/// Connect downstream to the real frontend as an engine, presenting the real
/// engine's handshake payloads verbatim.
async fn downstream_connect(
    frontend_handshake: &str,
    ready_message: &ReadyMessage,
    ready_response_payload: &[u8],
) -> Result<MockEngineSockets> {
    sim_protocol::frontend_connect::connect_to_frontend_raw(
        frontend_handshake,
        EngineId::from_engine_index(0),
        ready_message.local.unwrap_or(false),
        ready_message.headless.unwrap_or(true),
        ready_response_payload,
        std::time::Duration::from_secs(5),
    )
    .await
    .map_err(|e| anyhow!("connecting to frontend: {e}"))
}

/// Run the tap proxy: forward frames between the real frontend and the real
/// engine, recording per-request timing into a trace writer. When
/// `step_writer` is given, every engine output message carrying scheduler
/// stats also appends a [`StepStatsRecord`] line to it (requires the engine
/// to run with stats logging enabled; without it the stream stays empty).
///
/// Writes the trace meta line itself, after the handshake, so the meta can
/// carry the engine's reported vLLM version and raw ready-response bytes.
pub async fn run_tap<W: Write, S: Write>(
    config: TapConfig,
    meta: TapMetaConfig,
    writer: &mut W,
    mut step_writer: Option<S>,
    shutdown: CancellationToken,
) -> Result<()> {
    // Step 1: Bind upstream side and wait for the real engine's HELLO first.
    let UpstreamEngine {
        input: mut upstream_input,
        output: mut upstream_output,
        ready_message,
        ready_response_payload,
        vllm_version: engine_vllm_version,
    } = upstream_handshake(&config).await?;

    // Guard before recording anything: a capture labelled for one vLLM line but
    // served by another mislabels the golden, which is worse than no capture.
    if let Some(tag) = meta.vllm_tag.as_deref() {
        guard_vllm_version(tag, &engine_vllm_version)?;
    }

    // The meta line carries handshake-derived fields (engine version, raw
    // ready-response bytes), so it is written here, after the handshake and
    // before any records.
    let trace_meta = build_meta(&meta, &engine_vllm_version, &ready_response_payload);
    write_meta_line(writer, &trace_meta)?;

    info!(
        engine_vllm_version = %engine_vllm_version,
        "upstream engine connected, connecting downstream to frontend"
    );

    // Step 2: Connect downstream to the real frontend.
    let MockEngineSockets { data_sockets, .. } = downstream_connect(
        &config.frontend_handshake,
        &ready_message,
        &ready_response_payload,
    )
    .await?;

    // Single client, single engine: take the first data socket pair.
    let sockets = data_sockets
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no data sockets from frontend"))?;
    let (mut downstream_dealer, mut downstream_push) = (sockets.dealer, sockets.push);

    info!("tap proxy connected to both sides, forwarding frames");

    let mut requests: HashMap<String, RequestState> = HashMap::new();
    // Zero point for the arrival_ms column: trace arrival times are relative to
    // the moment the proxy went live.
    let capture_start = Instant::now();
    // The engine registered with the engine_index 0 identity; the upstream
    // ROUTER socket needs it as the first frame of every forwarded request.
    let engine_identity = EngineId::from_engine_index(0).to_frame();

    // Drain the two RECEIVE sockets in dedicated tasks rather than awaiting their
    // `recv()` directly in the `select!` below.
    //
    // zmq.rs reads are poll-driven (no background reader): a socket only drains
    // its TCP buffer while its `recv()` future is being polled. A `select!` over
    // two `Socket::recv()` futures DROPS (cancels) the non-winning future every
    // iteration, so a large multi-frame message (e.g. multimodal pixel tensors,
    // tens of MB across several frames) that needs many polls to arrive can stall
    // mid-flight while the other branch keeps firing. Observed live: an 11 MB
    // multimodal request sat unread in the dealer's TCP Recv-Q while the loop
    // serviced engine outputs, wedging the whole proxy (every later request,
    // text included, head-of-line blocked behind it).
    //
    // Each drain task owns its socket and calls `recv()` in a tight, never-
    // cancelled loop, forwarding (arrival_instant, message) over an unbounded
    // channel. The proxy loop then selects over the CHANNELS, whose `recv()` is
    // cancel-safe (a queue pop), and does the forwarding sends in the body. The
    // sockets are therefore always being drained regardless of which side is
    // busy. Arrival is stamped inside the drain task, closest to the wire.
    let (req_tx, mut req_rx) = tokio::sync::mpsc::unbounded_channel::<(Instant, ZmqMessage)>();
    let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<(Instant, ZmqMessage)>();

    let req_drain = tokio::spawn(async move {
        loop {
            match downstream_dealer.recv().await {
                Ok(message) => {
                    if req_tx.send((Instant::now(), message)).is_err() {
                        break; // proxy loop gone
                    }
                }
                Err(e) => {
                    warn!(%e, "tap: downstream dealer recv error, stopping request drain");
                    break;
                }
            }
        }
    });

    let out_drain = tokio::spawn(async move {
        loop {
            match upstream_output.recv().await {
                Ok(message) => {
                    if out_tx.send((Instant::now(), message)).is_err() {
                        break; // proxy loop gone
                    }
                }
                Err(e) => {
                    warn!(%e, "tap: upstream output recv error, stopping output drain");
                    break;
                }
            }
        }
    });

    let proxy_result: Result<()> = loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                info!("tap shutting down");
                break Ok(());
            }

            // Downstream -> Upstream: frames from the frontend to the engine.
            // The dealer receives [request_type, payload] frames.
            //
            // Forward first, observe after: the observation decode stays out of
            // the forwarding path. Frame clones are Bytes refcount bumps, not
            // copies.
            req = req_rx.recv() => {
                let Some((arrival, message)) = req else {
                    break Err(anyhow!("request drain task ended (frontend disconnected)"));
                };
                let frames = message.into_vec();

                // DIAG: frame structure of each frontend->engine message. Multimodal
                // requests carry mm tensors as extra aux frames; if image requests show
                // the same frame count as text, the mm data is on a channel we don't tap.
                info!(
                    n_frames = frames.len(),
                    sizes = ?frames.iter().map(|f| f.len()).collect::<Vec<_>>(),
                    "DIAG downstream->upstream request frames"
                );
                // DIAG: decode the request to see if it carries mm_features (and how
                // many prompt tokens).
                if frames.len() >= 2 && request_type_from_frame(frames[0].as_ref()) == Some(EngineCoreRequestType::Add) {
                    match decode_msgpack::<EngineCoreRequest>(frames[1].as_ref()) {
                        Ok(req) => info!(
                            request_id = %req.request_id,
                            mm_features = req.mm_features.is_some(),
                            prompt_tokens = req.prompt_token_ids.as_ref().map(|p| p.len()).unwrap_or(0),
                            "DIAG decoded Add request"
                        ),
                        Err(e) => warn!(%e, "DIAG failed to decode Add request"),
                    }
                }

                // Forward verbatim to the upstream engine, prefixed with the
                // engine's ROUTER identity.
                let mut router_frames = vec![engine_identity.clone()];
                router_frames.extend(frames.iter().cloned());
                let fwd = match ZmqMessage::try_from(router_frames) {
                    Ok(fwd) => fwd,
                    Err(e) => break Err(anyhow!("building router forward message: {e}")),
                };
                if let Err(e) = upstream_input.send(fwd).await {
                    break Err(anyhow!("forwarding request to engine: {e}"));
                }

                // Decode a copy for observation.
                observe_request(&frames, &mut requests, arrival, config.block_size);
            }

            // Upstream -> Downstream: frames from the engine to the frontend.
            out = out_rx.recv() => {
                let Some((arrival, message)) = out else {
                    break Err(anyhow!("output drain task ended (engine disconnected)"));
                };
                let frames = message.into_vec();

                // DIAG: any engine->frontend output means the engine processed a request.
                info!(n_frames = frames.len(), "DIAG upstream->downstream output frames");

                // Forward verbatim to the frontend via the PUSH socket.
                let fwd = match ZmqMessage::try_from(frames.clone()) {
                    Ok(fwd) => fwd,
                    Err(e) => break Err(anyhow!("building push forward message: {e}")),
                };
                if let Err(e) = downstream_push.send(fwd).await {
                    break Err(anyhow!("forwarding output to frontend: {e}"));
                }

                // Decode a copy for observation; the trace write at request
                // completion also lands here, after the frame is already out.
                observe_output(
                    &frames,
                    &mut requests,
                    writer,
                    step_writer.as_mut(),
                    arrival,
                    capture_start,
                    config.record_tokens,
                );
            }
        }
    };

    // Stop the drain tasks: they park in `recv()` and won't notice the dropped
    // channel, so abort them outright.
    req_drain.abort();
    out_drain.abort();
    proxy_result?;

    writer.flush().context("flushing trace writer")?;
    Ok(())
}

/// Observe (decode a copy of) an incoming request for recording. `arrival` is
/// the instant the frames came off the wire, stamped before forwarding.
fn observe_request<F: AsRef<[u8]>>(
    frames: &[F],
    requests: &mut HashMap<String, RequestState>,
    arrival: Instant,
    block_size: usize,
) {
    // A request is `[request_type, msgpack_blob, aux_tensor_frame...]`: text
    // requests are 2 frames, but a multimodal request appends one aux frame per
    // large tensor (image pixel values, tens of MB), so it carries 3+ frames. The
    // msgpack blob in frames[1] decodes on its own (large tensors ride as aux
    // indices, resolved later by the engine), so observation only needs the first
    // two frames; the aux tensor payload is irrelevant to timing. Requiring
    // exactly 2 frames silently dropped every multimodal request from the trace.
    if frames.len() < 2 {
        warn!(
            frame_count = frames.len(),
            "request has fewer than 2 frames, skipping observation"
        );
        return;
    }

    let Some(request_type) = request_type_from_frame(frames[0].as_ref()) else {
        warn!("unknown request type frame, skipping observation");
        return;
    };

    match request_type {
        EngineCoreRequestType::Add => {
            match decode_msgpack::<EngineCoreRequest>(frames[1].as_ref()) {
                Ok(req) => {
                    let concurrency = (requests.len() as u64) + 1;
                    let state = RequestState::new(&req, arrival, block_size, concurrency);
                    debug!(
                        request_id = %req.request_id,
                        prompt_tokens = state.prompt_tokens,
                        concurrency,
                        remote_prefill = state.remote_prefill,
                        "tap: observed Add request"
                    );
                    requests.insert(req.request_id, state);
                }
                Err(e) => {
                    warn!(%e, "tap: failed to decode Add request for observation");
                }
            }
        }
        EngineCoreRequestType::Abort => match decode_msgpack::<Vec<String>>(frames[1].as_ref()) {
            Ok(ids) => {
                for id in &ids {
                    if requests.remove(id).is_some() {
                        debug!(request_id = %id, "tap: discarding aborted request from trace");
                    }
                }
            }
            Err(e) => {
                warn!(%e, "tap: failed to decode Abort request for observation");
            }
        },
        // Utility and StartDpWave are not tracked.
        _ => {}
    }
}

/// Observe (decode a copy of) an engine output for recording. Finalized
/// records are written to the trace writer. `arrival` is the instant the
/// frames came off the wire, stamped before forwarding; `capture_start` is the
/// zero point for the trace's arrival_ms column.
fn observe_output<W: Write, S: Write, F: AsRef<[u8]>>(
    frames: &[F],
    requests: &mut HashMap<String, RequestState>,
    writer: &mut W,
    step_writer: Option<&mut S>,
    arrival: Instant,
    capture_start: Instant,
    record_tokens: TokenRecording,
) {
    let outputs = match decode_engine_core_outputs(frames) {
        Ok(outputs) => outputs,
        Err(e) => {
            warn!(%e, "tap: failed to decode engine output for observation (still forwarded)");
            return;
        }
    };

    if let Some(step_writer) = step_writer
        && let Some(stats) = &outputs.scheduler_stats
    {
        let record = StepStatsRecord {
            ts_ms: ms_between(capture_start, arrival),
            scheduler: stats.as_ref().clone(),
        };
        // Flush per line for the same crash-safety the trace writer gets: a
        // killed pod keeps everything observed so far.
        if let Err(e) = append_step_stats(&mut *step_writer, &record) {
            warn!(%e, "tap: failed to write step-stats record");
        } else if let Err(e) = step_writer.flush() {
            warn!(%e, "tap: failed to flush step-stats writer");
        }
    }

    // Step-level batch context for every gap closed by this message: the engine's
    // own running count when it attaches scheduler stats (tap in-flight count as
    // fallback), and the prompt tokens that finished prefill in this step (the
    // requests receiving their first tokens here).
    let step_running = outputs
        .scheduler_stats
        .as_ref()
        .map(|s| s.num_running_reqs as u32)
        .unwrap_or(requests.len() as u32);
    let step_prefill_tokens: u32 = outputs
        .outputs
        .iter()
        .filter(|o| !o.new_token_ids.is_empty())
        .filter_map(|o| requests.get(&o.request_id))
        .filter(|s| s.awaiting_first_token() && !s.remote_prefill)
        .map(|s| s.prompt_tokens as u32)
        .sum();

    for output in &outputs.outputs {
        let request_id = &output.request_id;

        let Some(state) = requests.get_mut(request_id) else {
            // Could be a request that started before the tap, or a response to
            // a utility call. Ignore silently.
            continue;
        };

        if output.finish_reason == Some(EngineCoreFinishReason::Abort) {
            debug!(request_id, "tap: discarding aborted request from trace");
            requests.remove(request_id);
            continue;
        }

        if record_tokens == TokenRecording::On {
            state.output_token_ids.extend(&output.new_token_ids);
        }

        // Extract cached_tokens from prefill_stats if present (first output).
        if let Some(ref stats) = output.prefill_stats {
            state.cached_tokens =
                (stats.num_local_cached_tokens + stats.num_external_cached_tokens) as usize;
        }

        if !output.new_token_ids.is_empty() {
            state.record_chunk(
                arrival,
                output.new_token_ids.len(),
                step_running,
                step_prefill_tokens,
            );
        }

        if let Some(reason) = output.finish_reason {
            let Some(state) = requests.remove(request_id) else {
                continue;
            };
            let record = state.into_record(capture_start, reason, record_tokens);
            debug!(
                request_id = %request_id,
                prompt_tokens = record.prompt_tokens,
                output_tokens = record.output_tokens,
                ttft_ms = record.ttft_ms,
                "tap: writing trace record"
            );
            if let Err(e) = append_record(writer, &record) {
                warn!(%e, "tap: failed to write trace record");
            }
            if let Err(e) = writer.flush() {
                warn!(%e, "tap: failed to flush trace writer");
            }
        }
    }
}

/// Inputs for the trace meta line, supplied by the caller. The handshake-derived
/// fields (engine version, raw ready-response bytes) are filled in by the tap.
#[derive(Debug, Clone, Default)]
pub struct TapMetaConfig {
    /// Model identifier recorded in the meta line.
    pub model: String,
    /// GPU type (e.g. `"H200"`).
    pub gpu: Option<String>,
    /// Tensor-parallel degree.
    pub tp: Option<u32>,
    /// Scheduler concurrency ceiling, a `config_hash` input.
    pub max_num_seqs: Option<u64>,
    /// Token-block size for prefix fingerprints.
    pub block_size: usize,
    /// Explicit `config_hash` override. When `None` it is computed from a
    /// [`ConfigFingerprint`] over the inputs here plus `vllm_tag`.
    pub config_hash: Option<String>,
    /// vLLM tag this capture targets (e.g. `"v0.23.0"`). Two roles: it guards
    /// the engine's reported version (by `major.minor` line), and it is the
    /// stable `config_hash` input. The engine's own version string is a dev
    /// build (`0.23.0.dev1+g...`) and not reproducible across rebuilds, so the
    /// tag is what capture and replay must agree on.
    pub vllm_tag: Option<String>,
    /// Canonical string of the deployed behavioral engine flags (a `config_hash`
    /// input). The capture driver builds it (it owns the engine args); the tap can't
    /// observe the engine config on the wire, so an empty string means "unknown".
    pub engine_config: String,
}

/// Build the trace meta from the caller's inputs plus what the handshake
/// observed. `config_hash` is the explicit override when set, otherwise the
/// canonical fingerprint over the deployment inputs.
fn build_meta(
    meta: &TapMetaConfig,
    engine_vllm_version: &str,
    ready_response_payload: &[u8],
) -> TraceMeta {
    let config_hash = meta.config_hash.clone().or_else(|| {
        let vllm_tag = meta
            .vllm_tag
            .clone()
            .unwrap_or_else(|| engine_vllm_version.to_string());
        Some(
            ConfigFingerprint {
                gpu: meta.gpu.clone().unwrap_or_default(),
                vllm_tag,
                engine_config: meta.engine_config.clone(),
            }
            .hash(),
        )
    });

    TraceMeta {
        source: Some("tap".to_string()),
        model: if meta.model.is_empty() {
            None
        } else {
            Some(meta.model.clone())
        },
        gpu: meta.gpu.clone(),
        tp: meta.tp,
        max_num_seqs: meta.max_num_seqs,
        block_size: Some(meta.block_size),
        config_hash,
        vllm_version: Some(engine_vllm_version.to_string()),
        ready_response_hex: Some(hex_encode(ready_response_payload)),
        ..TraceMeta::default()
    }
}

/// Serialize the `{"meta": ...}` line to the writer.
fn write_meta_line<W: Write>(writer: &mut W, meta: &TraceMeta) -> Result<()> {
    let wrapper = serde_json::json!({ "meta": meta });
    serde_json::to_writer(&mut *writer, &wrapper)?;
    writeln!(writer)?;
    Ok(())
}

/// Lowercase-hex encode bytes (no extra dependency for a one-shot payload).
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn guard_passes_on_same_line_across_suffix_differences() {
        // Tag is a release label, engine reports a dev build of the same line.
        assert!(guard_vllm_version("v0.23.0", "0.23.0.dev1+g16e9117").is_ok());
        assert!(guard_vllm_version("v0.23.1", "0.23.0").is_ok());
    }

    #[test]
    fn guard_rejects_a_different_line() {
        let err = guard_vllm_version("v0.23.0", "0.22.1")
            .expect_err("0.22 engine under a 0.23 label must abort");
        assert!(err.to_string().contains("line mismatch"), "got: {err}");
    }

    #[test]
    fn guard_is_lenient_when_a_version_is_unparseable() {
        // Don't fail a capture just because a version string is exotic.
        assert!(guard_vllm_version("v0.23.0", "nightly-weirdness").is_ok());
    }

    #[test]
    fn build_meta_computes_config_hash_from_the_tag_not_the_engine_dev_string() {
        // The computed hash must be stable across engine dev-build suffixes:
        // capture (dev string A) and replay (tag) have to agree.
        let cfg = TapMetaConfig {
            model: "Qwen/Qwen3-8B".to_string(),
            gpu: Some("H200".to_string()),
            tp: Some(1),
            max_num_seqs: Some(256),
            block_size: 16,
            config_hash: None,
            vllm_tag: Some("v0.23.0".to_string()),
            engine_config: "model=Qwen/Qwen3-8B;tensor_parallel=1".to_string(),
        };
        let a = build_meta(&cfg, "0.23.0.dev1+gAAAA", b"payload-a");
        let b = build_meta(&cfg, "0.23.0.dev9+gZZZZ", b"payload-b");
        assert_eq!(
            a.config_hash, b.config_hash,
            "config hash must come from the tag, not the engine dev string"
        );
        // But the recorded engine version + raw bytes do reflect what was seen.
        assert_eq!(a.vllm_version.as_deref(), Some("0.23.0.dev1+gAAAA"));
        assert_eq!(a.ready_response_hex.as_deref(), Some("7061796c6f61642d61"));
    }

    #[test]
    fn build_meta_honours_an_explicit_config_hash_override() {
        let cfg = TapMetaConfig {
            config_hash: Some("manual-override".to_string()),
            vllm_tag: Some("v0.23.0".to_string()),
            ..TapMetaConfig::default()
        };
        let meta = build_meta(&cfg, "0.23.0", b"");
        assert_eq!(meta.config_hash.as_deref(), Some("manual-override"));
    }

    fn state_at(arrival: Instant) -> RequestState {
        RequestState {
            arrival,
            prompt_tokens: 10,
            block_hashes: None,
            remote_prefill: false,
            last_output: None,
            output_tokens: 0,
            ttft_ms: None,
            gaps: Vec::new(),
            cached_tokens: 0,
            concurrency: 1,
            output_token_ids: Vec::new(),
        }
    }

    #[test]
    fn multi_token_chunks_record_one_gap_each_with_their_sizes() {
        let t0 = Instant::now();
        let mut state = state_at(t0);

        // First chunk (1 token) sets TTFT; then a 4-token burst after 20ms and
        // a single token after 10ms.
        state.record_chunk(t0 + Duration::from_millis(5), 1, 3, 0);
        state.record_chunk(t0 + Duration::from_millis(25), 4, 3, 0);
        state.record_chunk(t0 + Duration::from_millis(35), 1, 3, 0);

        let record = state.into_record(t0, EngineCoreFinishReason::Stop, TokenRecording::Off);
        assert_eq!(record.output_tokens, 6);
        let gaps = record.itl_ms.unwrap();
        assert_eq!(gaps.len(), 2, "one gap per chunk, not per token");
        assert!((gaps[0] - 20.0).abs() < 1.0, "burst keeps its full gap");
        assert!((gaps[1] - 10.0).abs() < 1.0);
        assert_eq!(record.itl_tokens, Some(vec![4, 1]));
        let ctx = record.itl_ctx.unwrap();
        assert_eq!(ctx.num_running.len(), 2);
        assert_eq!(ctx.prefill_tokens.len(), 2);
    }

    #[test]
    fn single_token_chunks_omit_itl_tokens() {
        let t0 = Instant::now();
        let mut state = state_at(t0);
        for i in 0..3u64 {
            state.record_chunk(t0 + Duration::from_millis(5 + 10 * i), 1, 1, 0);
        }
        let record = state.into_record(t0, EngineCoreFinishReason::Stop, TokenRecording::Off);
        assert_eq!(record.output_tokens, 3);
        assert_eq!(record.itl_ms.as_ref().map(Vec::len), Some(2));
        assert_eq!(
            record.itl_tokens, None,
            "plain autoregressive captures keep the pre-spec schema"
        );
    }

    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        let hex = hex.trim();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex"))
            .collect()
    }

    /// A real DiffusionGemma multimodal `Add` arrives as `[request_type,
    /// msgpack_blob, aux_tensor...]` (3+ frames, the pixel tensor riding as an
    /// aux frame). `observe_request` must track it; the old `frames.len() != 2`
    /// guard dropped every multimodal request, so the engine generated tokens but
    /// no trace record was ever written.
    #[test]
    fn observe_request_tracks_multimodal_add_with_aux_frames() {
        // frames[1] is the genuine MsgpackEncoder blob (rev 16e91176); the aux
        // pixel tensor is a stand-in here since observation never touches it.
        let msgpack = hex_to_bytes(include_str!("../tests/fixtures/mm_request_large_aux.hex"));
        let frames: Vec<Vec<u8>> = vec![
            EngineCoreRequestType::Add.to_frame().to_vec(), // request_type
            msgpack,                                        // msgpack blob
            vec![0xABu8; 4096],                             // aux pixel tensor (ignored)
        ];

        let mut requests: HashMap<String, RequestState> = HashMap::new();
        observe_request(&frames, &mut requests, Instant::now(), 16);

        assert_eq!(requests.len(), 1, "multimodal Add must be tracked");
        let state = requests.get("req-mm-large").expect("request_id tracked");
        // 2 text + 256 image placeholders + 1 text (matches the groundtruth test).
        assert_eq!(state.prompt_tokens, 259);
    }
}
