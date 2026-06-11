//! A mock vLLM V1 engine-core backend.
//!
//! It speaks the real ZMQ + msgpack engine-core protocol (reusing the in-tree
//! `vllm-engine-core-client` crate, so it sits behind the real vLLM Rust or Python
//! frontend unmodified), and carries a prefill/decode data plane that, with the
//! `nixl` feature, moves real KV-cache bytes over NIXL (UCX/DRAM or RDMA) without a GPU.
//!
//! Two birds:
//!   1. faithful frontend testing without a model or GPU, and
//!   2. a real P/D data path exercised over CPU RDMA.

use anyhow::{Context as _, Result, bail};
use clap::Parser;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineSockets, connect_to_frontend, default_ready_response,
};
use vllm_engine_core_client::protocol::EngineCoreFinishReason;

pub mod blockpool;
pub mod calibrate;
pub mod dataplane;
mod engine;
mod engine_core;
mod io;
pub mod kvevents;
pub mod latency;
pub mod lora;
mod sched;
pub mod tap;
mod tokens;
pub mod trace;
pub mod trace_convert;

use dataplane::PdRole;
use latency::KnobLatency;

/// A failure the engine can inject at the configured rate (Phase 5). Maps to the engine-core
/// finish reason a real vLLM engine would return for that class of failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FailureType {
    /// Retryable request-level internal error (the frontend may resubmit).
    Error,
    /// Truncation / context-length error.
    Length,
    /// A repetitive output pattern was detected.
    Repetition,
}

impl FailureType {
    /// The engine-core finish reason this failure surfaces as.
    pub fn finish_reason(self) -> EngineCoreFinishReason {
        match self {
            FailureType::Error => EngineCoreFinishReason::Error,
            FailureType::Length => EngineCoreFinishReason::Length,
            FailureType::Repetition => EngineCoreFinishReason::Repetition,
        }
    }
}

/// Waiting-queue ordering, matching vLLM's `--scheduling-policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SchedulingPolicy {
    /// First-come, first-served: requests are admitted in arrival order.
    Fcfs,
    /// Priority order: smaller `priority` value first, ties broken by earlier arrival.
    Priority,
}

/// Mock engine-core backend for frontend + prefill/decode data-plane testing.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "inference-sim",
    about = "Run a mock vLLM engine-core backend with an optional NIXL KV data plane."
)]
pub struct Opt {
    /// Frontend-owned ZMQ handshake address.
    #[arg(long, default_value = "tcp://127.0.0.1:29550")]
    pub handshake_address: String,

    /// Number of mock engine identities to register with the frontend.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,

    /// Prefill/decode role this process plays.
    #[arg(long, value_enum, default_value_t = PdRole::Both)]
    pub pd_role: PdRole,

    /// Number of accepted output tokens included in each EngineCoreOutput.
    #[arg(long, default_value_t = 1)]
    pub output_token_chunk_size: usize,

    /// Random token IDs are sampled uniformly from 0..vocab_size.
    #[arg(long, default_value_t = 32_000)]
    pub vocab_size: u32,

    /// Path to a JSONL trace whose recorded output token ids (tap
    /// `--record-tokens`) are served verbatim instead of random tokens, making
    /// replayed streams content-identical to the capture. A request resolves
    /// to its record through the trailing `-<index>` of its request id, where
    /// the index is the record's position in the arrival-ordered subset (the
    /// arrival-replay harness names them `replay-{i}`). Unmatched requests
    /// fall back to random tokens.
    #[arg(long, default_value = "")]
    pub replay_tokens: String,

    /// Base seed for deterministic random token generation.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Bytes per fabricated KV block (NIXL data plane only).
    #[arg(long, default_value_t = 128 * 1024)]
    pub kv_block_bytes: usize,

    /// Prompt tokens that map to one KV block (NIXL data plane only).
    #[arg(long, default_value_t = 16)]
    pub tokens_per_block: usize,

    /// This engine's id, advertised as `remote_engine_id` in kv_transfer_params.
    /// Set per-pod (e.g. from POD_NAME) so a decode peer can address this prefill.
    #[arg(long, env = "MOCK_ENGINE_ID", default_value = "mock-engine-0")]
    pub engine_id: String,

    /// Host advertised as `remote_host` for the NIXL metadata side channel
    /// (set to the pod IP in k8s).
    #[arg(long, env = "MOCK_SIDE_CHANNEL_HOST", default_value = "127.0.0.1")]
    pub side_channel_host: String,

    /// Port advertised as `remote_port` for the NIXL metadata side channel.
    #[arg(long, env = "MOCK_SIDE_CHANNEL_PORT", default_value_t = 5600)]
    pub side_channel_port: u32,

    /// Log a summary line for each request.
    #[arg(long)]
    pub log_requests: bool,

    // === Latency model (all milliseconds; 0 = instant, the default) ===
    // Ported from llm-d-inference-sim. The real frontend measures TTFT/ITL from when we
    // emit tokens, so these knobs drive both response timing and the vllm:* latency metrics.
    /// Path to a JSONL trace file for replay-based latency. Mutually exclusive with the
    /// timing knobs below (time_to_first_token, inter_token_latency, prefill_*). When set,
    /// first-token and inter-token delays are sampled from recorded observations rather
    /// than synthesized from knob parameters. KV-transfer timing for do_remote_prefill
    /// requests still uses the kv_cache_transfer_* knobs (the trace does not cover P/D
    /// transfer latencies).
    #[arg(long, default_value = "")]
    pub latency_trace: String,

    /// Fixed time-to-first-token. When this and its std-dev are 0, the token-count prefill
    /// model (`--prefill-overhead` + `--prefill-time-per-token`) is used instead.
    #[arg(long, default_value_t = 0)]
    pub time_to_first_token: u64,

    /// Standard deviation for time-to-first-token.
    #[arg(long, default_value_t = 0)]
    pub time_to_first_token_std_dev: u64,

    /// Time to generate one output token (decode step).
    #[arg(long, default_value_t = 0)]
    pub inter_token_latency: u64,

    /// Standard deviation for inter-token latency.
    #[arg(long, default_value_t = 0)]
    pub inter_token_latency_std_dev: u64,

    /// Fixed prefill overhead, added once per request in the token-count prefill model.
    #[arg(long, default_value_t = 0)]
    pub prefill_overhead: u64,

    /// Per-(uncached-)prompt-token prefill cost in the token-count prefill model.
    #[arg(long, default_value_t = 0)]
    pub prefill_time_per_token: u64,

    /// Standard deviation for the token-count prefill time.
    #[arg(long, default_value_t = 0)]
    pub prefill_time_std_dev: u64,

    /// Fixed KV-cache transfer time for a `do_remote_prefill` decode request. When this and
    /// its std-dev are 0, the per-token transfer model is used instead.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_latency: u64,

    /// Standard deviation for the fixed KV-cache transfer time.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_latency_std_dev: u64,

    /// Per-prompt-token KV-cache transfer cost for a `do_remote_prefill` decode request.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_time_per_token: u64,

    /// Standard deviation for the per-token KV-cache transfer cost.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_transfer_time_std_dev: u64,

    /// Latency multiplier at full load (`>= 1.0`; 1.0 disables load scaling). Latency grows
    /// linearly from 1.0 with one request to this value at `--max-num-seqs` concurrent ones.
    #[arg(long, default_value_t = 1.0)]
    pub time_factor_under_load: f64,

    /// Maximum requests running (in the model batch) at once (vLLM `--max-num-seqs`). Excess
    /// requests wait in an unbounded FIFO/priority queue, producing `vllm:num_requests_waiting`
    /// and realistic backpressure. Also the load-factor denominator. vLLM never rejects on
    /// queue length, so neither do we.
    #[arg(long, default_value_t = 128)]
    pub max_num_seqs: u64,

    /// Per-step token budget (vLLM `--max-num-batched-tokens`): the running batch's per-step
    /// token demand (1 per decoding request + each prefilling request's prompt chunk) cannot
    /// exceed this. Throttles prefill admission under load even when batch slots are free.
    #[arg(long, default_value_t = 2048)]
    pub max_num_batched_tokens: u64,

    /// Chunked-prefill cap (vLLM `--long-prefill-token-threshold`): a single prefill consumes
    /// at most this many tokens of budget per step. `0` (default) disables the cap (a prefill
    /// is bounded only by the token budget).
    #[arg(long, default_value_t = 0)]
    pub long_prefill_token_threshold: u64,

    /// Waiting-queue ordering (vLLM `--scheduling-policy`): `fcfs` (arrival order) or
    /// `priority` (smaller `priority` value first, ties by earlier arrival).
    #[arg(long, value_enum, default_value_t = SchedulingPolicy::Fcfs)]
    pub scheduling_policy: SchedulingPolicy,

    /// KV-cache capacity in blocks, used to report `vllm:kv_cache_usage_perc`.
    #[arg(long, default_value_t = 1024)]
    pub kv_cache_size: u64,

    /// Maximum distinct LoRA adapters allowed in the running batch at once (vLLM
    /// `--max-loras`). A request needing an adapter not already resident waits once the batch
    /// holds this many distinct adapters. `0` (default) disables the cap; adapter accounting
    /// (the `running_lora_adapters`/`waiting_lora_adapters` stats behind
    /// `vllm:lora_requests_info`) is always on regardless.
    #[arg(long, default_value_t = 0)]
    pub max_loras: u64,

    // === KV-cache events (vLLM `--kv-events-config`) ===
    /// Publish KV-cache events (BlockStored/BlockRemoved/AllBlocksCleared) over ZMQ so the
    /// llm-d cache-aware router can index this engine's prefix cache.
    #[arg(long, default_value_t = false)]
    pub enable_kv_cache_events: bool,

    /// ZMQ PUB endpoint for KV-cache events. Wildcards (`tcp://*:5557`) bind; concrete hosts
    /// connect. Offset by engine index when running multiple engines in one process.
    #[arg(long, default_value = "tcp://*:5557")]
    pub kv_events_endpoint: String,

    /// KV-cache event topic. The llm-d router expects `kv@<pod-id>@<model-name>` (its SUB
    /// filter defaults to `kv@`). Leave empty to auto-build from `--engine-id`/`--model-name`.
    #[arg(long, default_value = "")]
    pub kv_events_topic: String,

    /// Served model name, used only to build the default KV-event topic.
    #[arg(long, env = "MODEL", default_value = "")]
    pub model_name: String,

    /// Fixed seed chaining the first block of every sequence's hash (vLLM's `NONE_HASH`).
    /// Pinned (not random) so block hashes are reproducible across restarts and peers.
    #[arg(long, default_value_t = 0)]
    pub kv_cache_none_seed: u64,

    // === Failure injection (Phase 5) ===
    /// Maximum context length (prompt + output tokens). `0` (default) disables the check; a
    /// request whose `prompt + max_tokens` exceeds it finishes immediately with a length error.
    #[arg(long, default_value_t = 0)]
    pub max_model_len: u64,

    /// Probability in `[0, 1]` that an incoming request is failed on arrival. `0` disables
    /// injection.
    #[arg(long, default_value_t = 0.0)]
    pub failure_injection_rate: f64,

    /// Divide every model delay by this factor (time compression for fast
    /// calibration loops). Wire-level measurements must be re-multiplied by
    /// the same factor; timer granularity and transport jitter do not scale,
    /// so fidelity degrades as the factor grows. `1.0` is real time.
    #[arg(long, default_value_t = 1.0)]
    pub time_scale: f64,

    /// Failure kinds to inject (one is chosen uniformly per injected failure).
    #[arg(long, value_enum, value_delimiter = ',', default_value = "error")]
    pub failure_types: Vec<FailureType>,

    /// Graceful-shutdown grace period in seconds (vLLM `shutdown_timeout`). On
    /// SIGTERM/SIGINT the engine rejects new requests and lets in-flight ones finish for
    /// up to this long; whatever remains is then aborted. `0` (vLLM's default) aborts
    /// every in-flight request immediately.
    #[arg(long, default_value_t = 0)]
    pub shutdown_timeout: u64,
}

/// Offset the port of a `tcp://host:port` endpoint by `n` (no-op for `n == 0` or non-tcp),
/// mirroring vLLM's per-DP-rank port offset so multiple engines in one process don't clash.
fn offset_endpoint_port(endpoint: &str, n: u32) -> String {
    if n == 0 || !endpoint.starts_with("tcp://") {
        return endpoint.to_string();
    }
    match endpoint.rsplit_once(':') {
        Some((base, port)) => match port.parse::<u32>() {
            Ok(port) => format!("{base}:{}", port + n),
            Err(_) => endpoint.to_string(),
        },
        None => endpoint.to_string(),
    }
}

impl Opt {
    /// Build the KV-cache event publisher config for one engine. The endpoint port and the
    /// topic's pod id are offset by `engine_index` so several engines in one process publish
    /// on distinct sockets/streams.
    pub fn kv_events_config(&self, engine_index: u32) -> kvevents::KvEventsConfig {
        let endpoint = offset_endpoint_port(&self.kv_events_endpoint, engine_index);
        let topic = if !self.kv_events_topic.is_empty() {
            self.kv_events_topic.clone()
        } else {
            let pod = if engine_index == 0 {
                self.engine_id.clone()
            } else {
                format!("{}-{engine_index}", self.engine_id)
            };
            format!("kv@{pod}@{}", self.model_name)
        };
        kvevents::KvEventsConfig {
            enabled: self.enable_kv_cache_events,
            endpoint,
            topic,
        }
    }

    /// Build the knob-based latency model from the configured timing knobs.
    pub fn latency_model(&self) -> KnobLatency {
        KnobLatency {
            time_to_first_token: self.time_to_first_token,
            time_to_first_token_std_dev: self.time_to_first_token_std_dev,
            inter_token_latency: self.inter_token_latency,
            inter_token_latency_std_dev: self.inter_token_latency_std_dev,
            prefill_overhead: self.prefill_overhead,
            prefill_time_per_token: self.prefill_time_per_token,
            prefill_time_std_dev: self.prefill_time_std_dev,
            kv_cache_transfer_latency: self.kv_cache_transfer_latency,
            kv_cache_transfer_latency_std_dev: self.kv_cache_transfer_latency_std_dev,
            kv_cache_transfer_time_per_token: self.kv_cache_transfer_time_per_token,
            kv_cache_transfer_time_std_dev: self.kv_cache_transfer_time_std_dev,
            time_factor_under_load: self.time_factor_under_load,
            max_num_seqs: self.max_num_seqs,
        }
    }

    /// Build the token source: replay recorded output ids (from `--replay-tokens`) or
    /// random draws. Returns an error if the trace is unreadable or carries no tokens.
    pub(crate) fn build_token_source(&self) -> Result<Box<dyn tokens::TokenSource>> {
        if self.replay_tokens.is_empty() {
            return Ok(Box::new(tokens::RandomTokens {
                vocab_size: self.vocab_size,
            }));
        }
        let (_, records) = trace::read_trace_file(std::path::Path::new(&self.replay_tokens))?;
        let subset = trace::replay_subset(records);
        if subset.is_empty() {
            bail!(
                "--replay-tokens trace {} has no records with arrival_ms (nothing to map \
                 request indices onto)",
                self.replay_tokens
            );
        }
        if !subset.iter().any(|r| r.output_token_ids.is_some()) {
            bail!(
                "--replay-tokens trace {} has no output_token_ids; capture it with the \
                 tap's --record-tokens",
                self.replay_tokens
            );
        }
        Ok(Box::new(tokens::ReplayTokens::from_records(
            &subset,
            self.vocab_size,
        )))
    }

    /// Whether any of the timing knobs that are mutually exclusive with `--latency-trace`
    /// have been set to a nonzero value.
    fn has_timing_knobs(&self) -> bool {
        self.time_to_first_token != 0
            || self.time_to_first_token_std_dev != 0
            || self.inter_token_latency != 0
            || self.inter_token_latency_std_dev != 0
            || self.prefill_overhead != 0
            || self.prefill_time_per_token != 0
            || self.prefill_time_std_dev != 0
    }

    /// Build the latency model: either trace-replay (from `--latency-trace`) or
    /// knob-based. Returns an error if both are configured or if the trace file is
    /// unreadable.
    pub fn build_latency(&self) -> Result<Box<dyn latency::LatencyModel>> {
        if self.latency_trace.is_empty() {
            return Ok(Box::new(self.latency_model()));
        }

        if self.has_timing_knobs() {
            bail!(
                "--latency-trace is mutually exclusive with timing knobs \
                 (time_to_first_token, inter_token_latency, prefill_*). \
                 Remove the knobs or the trace path."
            );
        }

        let (meta, records) = trace::read_trace_file(std::path::Path::new(&self.latency_trace))?;

        // The KV-transfer knobs are NOT mutually exclusive: the trace does not cover
        // P/D transfer timing, so the knob model handles do_remote_prefill requests.
        let kv_fallback = self.latency_model();
        let trace_model = latency::TraceLatency::from_records(
            meta,
            &records,
            kv_fallback,
            self.max_num_batched_tokens as usize,
        )
        .with_context(|| format!("building trace latency model from: {}", self.latency_trace))?;

        Ok(Box::new(trace_model))
    }
}

/// Run one mock engine until shutdown completes or the transport fails. Shutdown is
/// graceful: the engine loop drains or aborts in-flight requests per `--shutdown-timeout`
/// (mirroring vLLM's engine core), and the IO loop flushes the final outputs before exit.
async fn run_engine(engine_index: u32, opt: Opt, shutdown: CancellationToken) -> Result<()> {
    // Advertise the sim's actual configured limits in the registration ready
    // response so the frontend validates against what this engine enforces.
    let mut ready_response = default_ready_response();
    ready_response.num_gpu_blocks = opt.kv_cache_size;
    if opt.max_model_len > 0 {
        ready_response.max_model_len = opt.max_model_len;
    }
    let config = MockEngineConfig {
        ready_response,
        ..MockEngineConfig::default()
    };

    // A shutdown signal during the handshake means there is nothing to drain; just leave.
    let connect = connect_to_frontend(
        &opt.handshake_address,
        EngineId::from_engine_index(engine_index),
        config,
    );
    let MockEngineSockets { data_sockets, .. } = tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            info!(engine_index, "shutdown requested before frontend handshake completed");
            return Ok(());
        }
        result = connect => result
            .with_context(|| format!("engine {engine_index} failed to connect to frontend"))?,
    };

    info!(engine_index, role = ?opt.pd_role, "engine connected to frontend");

    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (output_tx, output_rx) = mpsc::unbounded_channel();

    let mut io_loop = tokio::spawn(io::run_io_loop(data_sockets, input_tx, output_rx));
    // The publisher's token is cancelled only after the engine loop exits, so KV events
    // keep flowing while in-flight requests drain (the publisher also exits on its own
    // once the engine drops its KvEventTx).
    let engine_done = CancellationToken::new();
    let events = kvevents::spawn(opt.kv_events_config(engine_index), engine_done.clone())
        .await
        .unwrap_or_else(|error| {
            tracing::warn!(%error, "kv-event publisher failed to start; continuing without events");
            None
        });
    let shutdown_timeout = std::time::Duration::from_secs(opt.shutdown_timeout);
    let sim_engine = engine::SimEngine::new(engine_index, opt, events).await?;
    let mut engine_loop = tokio::spawn(engine_core::run_loop(
        sim_engine,
        input_rx,
        output_tx,
        shutdown.clone(),
        shutdown_timeout,
    ));

    tokio::select! {
        biased;
        result = &mut engine_loop => {
            engine_done.cancel();
            if !shutdown.is_cancelled() {
                error!(engine_index, "engine loop exited unexpectedly");
            }
            // The engine loop dropped its output sender; the IO loop exits on its own
            // after flushing the remaining outputs (final tokens and abort notices).
            // Bound the flush so a wedged peer socket cannot hold up process exit.
            let flushed = tokio::time::timeout(std::time::Duration::from_secs(5), &mut io_loop).await;
            match flushed {
                Ok(Ok(Ok(()))) => {}
                Ok(Ok(Err(error))) => {
                    tracing::warn!(engine_index, %error, "IO loop errored while flushing final outputs");
                }
                Ok(Err(error)) => {
                    tracing::warn!(engine_index, %error, "IO loop task join failed during flush");
                }
                Err(_) => {
                    error!(engine_index, "IO loop failed to flush within 5s; aborting it");
                    io_loop.abort();
                    io_loop.await.ok();
                }
            }
            result??;
        }
        result = &mut io_loop => {
            error!(engine_index, "engine IO loop exited unexpectedly");
            engine_done.cancel();
            engine_loop.abort();
            engine_loop.await.ok();
            result??;
        }
    }

    info!(engine_index, "engine shut down");
    Ok(())
}

/// Run all requested mock engines until cancellation or one engine task fails.
pub async fn run(opt: Opt, shutdown: CancellationToken) -> Result<()> {
    // Validate the latency configuration (knob/trace conflict, trace parse) before any
    // transport setup, so a bad config fails immediately instead of after the 30s
    // frontend handshake timeout. Engines rebuild their own copy in SimEngine::new.
    opt.build_latency()?;

    info!(?opt, "starting mock engine");

    let mut engines = JoinSet::new();
    for engine_index in 0..opt.engine_count {
        engines.spawn(run_engine(
            engine_index as u32,
            opt.clone(),
            shutdown.clone(),
        ));
    }

    // No abort-on-shutdown here: each engine loop observes `shutdown` itself and drains
    // or aborts its in-flight requests per `--shutdown-timeout`, so this just waits for
    // every engine to finish. An engine failing (or exiting without a shutdown request)
    // is an error; returning early drops the JoinSet, which aborts the survivors.
    while let Some(joined) = engines.join_next().await {
        match joined {
            Ok(Ok(())) => {
                if !shutdown.is_cancelled() {
                    bail!("engine exited unexpectedly");
                }
            }
            Ok(Err(error)) => return Err(error),
            Err(error) => return Err(error).context("engine task join failed"),
        }
    }
    Ok(())
}
