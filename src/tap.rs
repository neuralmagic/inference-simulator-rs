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

use anyhow::{Context as _, Result, anyhow};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::MockEngineSockets;
use vllm_engine_core_client::protocol::handshake::{
    EngineCoreReadyResponse, HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreRequestType, decode_engine_core_outputs,
    decode_msgpack, encode_msgpack,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSocket, ZmqMessage};

use crate::trace::{
    ItlContext, StepStatsRecord, TraceMeta, TraceRecord, append_record, append_step_stats,
};

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
        let remote_prefill = crate::engine::extract_kv_params(req)
            .map(|kv| crate::engine::kv_flag(&kv, "do_remote_prefill"))
            .unwrap_or(false);
        Self {
            arrival,
            prompt_tokens: req.prompt_token_ids.as_ref().map(Vec::len).unwrap_or(0),
            block_hashes: req
                .prompt_token_ids
                .as_deref()
                .and_then(|tokens| crate::trace::prompt_block_hashes(tokens, block_size)),
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
            finish_reason: Some(finish_reason.into()),
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
    let ready_response: EngineCoreReadyResponse = decode_msgpack(&reg_payload)
        .map_err(|e| anyhow!("decoding engine registration ready response: {e}"))?;
    debug!(?ready_response, "engine registered on tap input socket");

    Ok(UpstreamEngine {
        input: input_socket,
        output: output_socket,
        ready_message,
        ready_response_payload: reg_payload.to_vec(),
    })
}

/// Connect downstream to the real frontend as an engine, presenting the real
/// engine's handshake payloads verbatim.
async fn downstream_connect(
    frontend_handshake: &str,
    ready_message: &ReadyMessage,
    ready_response_payload: &[u8],
) -> Result<MockEngineSockets> {
    crate::frontend_connect::connect_to_frontend_raw(
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
/// The caller is responsible for writing the meta line before calling this.
pub async fn run_tap<W: Write, S: Write>(
    config: TapConfig,
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
    } = upstream_handshake(&config).await?;

    info!("upstream engine connected, connecting downstream to frontend");

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

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                info!("tap shutting down");
                break;
            }

            // Downstream -> Upstream: frames from the frontend to the engine.
            // The dealer receives [request_type, payload] frames.
            //
            // Stamp arrival, forward first, observe after: the timestamp is taken
            // at the wire, but the observation decode stays out of the forwarding
            // path. Frame clones are Bytes refcount bumps, not copies.
            result = downstream_dealer.recv() => {
                let message = result.context("receiving from frontend")?;
                let arrival = Instant::now();
                let frames = message.into_vec();

                // Forward verbatim to the upstream engine, prefixed with the
                // engine's ROUTER identity.
                let mut router_frames = vec![engine_identity.clone()];
                router_frames.extend(frames.iter().cloned());
                let fwd = ZmqMessage::try_from(router_frames)
                    .map_err(|e| anyhow!("building router forward message: {e}"))?;
                upstream_input.send(fwd).await
                    .context("forwarding request to engine")?;

                // Decode a copy for observation.
                observe_request(&frames, &mut requests, arrival, config.block_size);
            }

            // Upstream -> Downstream: frames from the engine to the frontend.
            result = upstream_output.recv() => {
                let message = result.context("receiving from engine")?;
                let arrival = Instant::now();
                let frames = message.into_vec();

                // Forward verbatim to the frontend via the PUSH socket.
                let fwd = ZmqMessage::try_from(frames.clone())
                    .map_err(|e| anyhow!("building push forward message: {e}"))?;
                downstream_push.send(fwd).await
                    .context("forwarding output to frontend")?;

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
    }

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
    if frames.len() != 2 {
        warn!(
            frame_count = frames.len(),
            "unexpected request frame count, skipping observation"
        );
        return;
    }

    let Some(request_type) = EngineCoreRequestType::from_frame(frames[0].as_ref()) else {
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

/// Write the trace metadata line.
pub fn write_meta<W: Write>(
    writer: &mut W,
    model: &str,
    gpu: Option<&str>,
    tp: Option<u32>,
    block_size: usize,
) -> Result<()> {
    let meta = TraceMeta {
        source: Some("tap".to_string()),
        model: if model.is_empty() {
            None
        } else {
            Some(model.to_string())
        },
        gpu: gpu.map(str::to_string),
        tp,
        block_size: Some(block_size),
        ..TraceMeta::default()
    };
    let wrapper = serde_json::json!({"meta": meta});
    serde_json::to_writer(&mut *writer, &wrapper)?;
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

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
}
