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
//! - Multi-token chunks: when a single output carries N > 1 tokens, the ITL gap
//!   is divided evenly across N-1 intervals (the first token in the chunk gets
//!   the TTFT or the gap since the prior chunk). This is documented in each
//!   record's ITL array.
//! - No coordinator pass-through.

use std::collections::HashMap;
use std::io::Write;

use anyhow::{Context as _, Result, anyhow};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineSockets, connect_to_frontend,
};
use vllm_engine_core_client::protocol::handshake::{
    EngineCoreReadyResponse, HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreRequestType, decode_engine_core_outputs,
    decode_msgpack, encode_msgpack,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSocket, ZmqMessage};

use crate::trace::{TraceMeta, TraceRecord, append_record};

/// Per-request observation state maintained by the tap.
struct RequestState {
    arrival: Instant,
    prompt_tokens: usize,
    /// Last output instant (for ITL computation).
    last_output: Option<Instant>,
    /// Accumulated output token count.
    output_tokens: usize,
    /// Time to first token in milliseconds (set on first output).
    ttft_ms: Option<f64>,
    /// Inter-token latency gaps in milliseconds.
    itl_ms: Vec<f64>,
    /// Cached token count from prefill_stats.
    cached_tokens: usize,
    /// Concurrency snapshot at arrival.
    concurrency: u64,
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
    /// Model name for the trace metadata.
    pub model: String,
}

/// Sockets and handshake payloads captured while connecting the real engine.
struct UpstreamEngine {
    input: RouterSocket,
    output: PullSocket,
    ready_message: ReadyMessage,
    /// Post-initialization config from the engine's input-socket registration
    /// (max_model_len, num_gpu_blocks, dtype, vllm_version). Relayed verbatim
    /// to the downstream frontend so it validates against the real engine.
    ready_response: EngineCoreReadyResponse,
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
    let hello_frames = hello_msg.into_vec();
    if hello_frames.len() != 2 {
        return Err(anyhow!(
            "expected 2 frames for engine HELLO, got {}",
            hello_frames.len()
        ));
    }
    let engine_identity = hello_frames[0].clone();
    let ready_message: ReadyMessage =
        decode_msgpack(&hello_frames[1]).map_err(|e| anyhow!("decoding HELLO: {e}"))?;

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
    // Build a two-frame ROUTER message: [identity, payload].
    // Start with the identity as a single-frame message, then push the payload.
    let mut init_zmq = ZmqMessage::from(engine_identity.as_ref().to_vec());
    let payload_msg = ZmqMessage::from(init_payload);
    // push_back needs Bytes; extract from a single-frame ZmqMessage.
    let payload_frame = payload_msg
        .into_vec()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("empty payload message"))?;
    init_zmq.push_back(payload_frame);

    handshake_socket
        .send(init_zmq)
        .await
        .context("sending INIT to engine")?;
    debug!("sent INIT to engine");

    // Wait for READY.
    let ready_msg = handshake_socket
        .recv()
        .await
        .context("receiving engine READY")?;
    let ready_frames = ready_msg.into_vec();
    if ready_frames.len() != 2 {
        return Err(anyhow!(
            "expected 2 frames for engine READY, got {}",
            ready_frames.len()
        ));
    }
    let ready_payload: ReadyMessage =
        decode_msgpack(&ready_frames[1]).map_err(|e| anyhow!("decoding READY: {e}"))?;
    debug!(?ready_payload, "received engine READY");

    // Wait for the engine to register on the input socket (sends its identity + ready response).
    let reg_msg = input_socket
        .recv()
        .await
        .context("receiving engine registration on input socket")?;
    let reg_frames = reg_msg.into_vec();
    if reg_frames.len() != 2 {
        return Err(anyhow!(
            "expected 2 frames for engine registration, got {}",
            reg_frames.len()
        ));
    }
    // A decode failure here means the engine speaks an incompatible protocol
    // version; fail loudly instead of recording a trace against a broken pair.
    let ready_response: EngineCoreReadyResponse = decode_msgpack(&reg_frames[1])
        .map_err(|e| anyhow!("decoding engine registration ready response: {e}"))?;
    debug!(?ready_response, "engine registered on tap input socket");

    Ok(UpstreamEngine {
        input: input_socket,
        output: output_socket,
        ready_message,
        ready_response,
    })
}

/// Connect downstream to the real frontend as an engine, presenting the real
/// engine's handshake payloads.
async fn downstream_connect(
    frontend_handshake: &str,
    ready_message: &ReadyMessage,
    ready_response: EngineCoreReadyResponse,
) -> Result<MockEngineSockets> {
    let config = MockEngineConfig {
        local: ready_message.local.unwrap_or(false),
        headless: ready_message.headless.unwrap_or(true),
        ready_response,
        ..MockEngineConfig::default()
    };

    connect_to_frontend(frontend_handshake, EngineId::from_engine_index(0), config)
        .await
        .map_err(|e| anyhow!("connecting to frontend: {e}"))
}

/// Run the tap proxy: forward frames between the real frontend and the real
/// engine, recording per-request timing into a trace writer.
///
/// The caller is responsible for writing the meta line before calling this.
pub async fn run_tap<W: Write>(
    config: TapConfig,
    writer: &mut W,
    shutdown: CancellationToken,
) -> Result<()> {
    // Step 1: Bind upstream side and wait for the real engine's HELLO first.
    let UpstreamEngine {
        input: mut upstream_input,
        output: mut upstream_output,
        ready_message,
        ready_response,
    } = upstream_handshake(&config).await?;

    info!("upstream engine connected, connecting downstream to frontend");

    // Step 2: Connect downstream to the real frontend.
    let MockEngineSockets { data_sockets, .. } =
        downstream_connect(&config.frontend_handshake, &ready_message, ready_response).await?;

    // Single client, single engine: take the first data socket pair.
    let (mut downstream_dealer, mut downstream_push) =
        if let Some(sockets) = data_sockets.into_iter().next() {
            (sockets.dealer, sockets.push)
        } else {
            return Err(anyhow!("no data sockets from frontend"));
        };

    info!("tap proxy connected to both sides, forwarding frames");

    let mut requests: HashMap<String, RequestState> = HashMap::new();
    // Zero point for the arrival_ms column: trace arrival times are relative to
    // the moment the proxy went live.
    let capture_start = Instant::now();

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

                // Forward verbatim to the upstream engine via the ROUTER socket.
                // The ROUTER needs [identity, ...frames]. The engine registered
                // with engine_index 0 identity.
                let engine_id = EngineId::from_engine_index(0);
                let mut router_frames = vec![engine_id.to_frame()];
                router_frames.extend(frames.iter().cloned());
                let fwd = ZmqMessage::try_from(router_frames)
                    .map_err(|e| anyhow!("building router forward message: {e}"))?;
                upstream_input.send(fwd).await
                    .context("forwarding request to engine")?;

                // Decode a copy for observation.
                observe_request(&frames, &mut requests, arrival);
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
                observe_output(&frames, &mut requests, writer, arrival, capture_start);
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
                    let prompt_tokens = req.prompt_token_ids.as_ref().map(Vec::len).unwrap_or(0);
                    let concurrency = (requests.len() as u64) + 1;
                    debug!(
                        request_id = %req.request_id,
                        prompt_tokens,
                        concurrency,
                        "tap: observed Add request"
                    );
                    requests.insert(
                        req.request_id.clone(),
                        RequestState {
                            arrival,
                            prompt_tokens,
                            last_output: None,
                            output_tokens: 0,
                            ttft_ms: None,
                            itl_ms: Vec::new(),
                            cached_tokens: 0,
                            concurrency,
                        },
                    );
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
fn observe_output<W: Write, F: AsRef<[u8]>>(
    frames: &[F],
    requests: &mut HashMap<String, RequestState>,
    writer: &mut W,
    arrival: Instant,
    capture_start: Instant,
) {
    let outputs = match decode_engine_core_outputs(frames) {
        Ok(outputs) => outputs,
        Err(e) => {
            warn!(%e, "tap: failed to decode engine output for observation (still forwarded)");
            return;
        }
    };

    for output in &outputs.outputs {
        let request_id = &output.request_id;
        let num_new_tokens = output.new_token_ids.len();

        let Some(state) = requests.get_mut(request_id) else {
            // Could be a request that started before the tap, or a response to
            // a utility call. Ignore silently.
            continue;
        };

        // Extract cached_tokens from prefill_stats if present (first output).
        if let Some(ref stats) = output.prefill_stats {
            state.cached_tokens =
                (stats.num_local_cached_tokens + stats.num_external_cached_tokens) as usize;
        }

        if num_new_tokens == 0 {
            // No tokens in this output (e.g. an abort confirmation with empty tokens).
            if output.finish_reason == Some(EngineCoreFinishReason::Abort) {
                debug!(request_id, "tap: discarding aborted request from trace");
                requests.remove(request_id);
            }
            continue;
        }

        if state.ttft_ms.is_none() {
            // First token(s) for this request.
            let ttft = arrival.duration_since(state.arrival);
            state.ttft_ms = Some(ttft.as_secs_f64() * 1000.0);
            state.output_tokens += num_new_tokens;
            state.last_output = Some(arrival);
        } else {
            // Subsequent tokens: compute ITL gap.
            let gap = arrival.duration_since(state.last_output.unwrap_or(state.arrival));
            let gap_ms = gap.as_secs_f64() * 1000.0;

            // For multi-token chunks, divide the gap evenly across the tokens.
            // The first token in the chunk gets one share, the remaining get one each.
            // This produces num_new_tokens ITL entries for this chunk.
            let per_token_gap = gap_ms / num_new_tokens as f64;
            for _ in 0..num_new_tokens {
                state.itl_ms.push(per_token_gap);
            }

            state.output_tokens += num_new_tokens;
            state.last_output = Some(arrival);
        }

        // Check for finish.
        if let Some(reason) = &output.finish_reason {
            if *reason == EngineCoreFinishReason::Abort {
                debug!(request_id, "tap: discarding aborted request from trace");
                requests.remove(request_id);
                continue;
            }

            let request_id_owned = request_id.clone();
            if let Some(state) = requests.remove(&request_id_owned) {
                let record = TraceRecord {
                    prompt_tokens: state.prompt_tokens,
                    cached_tokens: state.cached_tokens,
                    output_tokens: state.output_tokens,
                    ttft_ms: state.ttft_ms.unwrap_or(0.0),
                    itl_ms: if state.itl_ms.is_empty() {
                        None
                    } else {
                        Some(state.itl_ms)
                    },
                    itl_summary: None,
                    concurrency: state.concurrency,
                    arrival_ms: Some(
                        state.arrival.duration_since(capture_start).as_secs_f64() * 1000.0,
                    ),
                };
                debug!(
                    request_id = %request_id_owned,
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
}

/// Write the trace metadata line.
pub fn write_meta<W: Write>(
    writer: &mut W,
    model: &str,
    gpu: Option<&str>,
    tp: Option<u32>,
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
        ..TraceMeta::default()
    };
    let wrapper = serde_json::json!({"meta": meta});
    serde_json::to_writer(&mut *writer, &wrapper)?;
    writeln!(writer)?;
    Ok(())
}
