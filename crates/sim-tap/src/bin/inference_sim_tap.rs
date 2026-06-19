//! Engine-core recording tap: a transparent ZMQ proxy between a real vLLM
//! frontend and a real engine-core, recording per-request timing into a JSONL
//! trace file.
//!
//! ## Usage
//!
//! ```text
//! inference-sim-tap --trace-out /tmp/trace.jsonl --model my-model
//! ```
//!
//! Addresses default to the sidecar convention (frontend handshake
//! tcp://127.0.0.1:5570, engine handshake tcp://127.0.0.1:5580, tap-bound
//! data sockets :29560/:29561); override with --frontend-handshake /
//! --engine-handshake / --input-address / --output-address (ipc:// works too).
//!
//! ## Topology
//!
//! ```text
//!   real frontend <--[downstream]--> TAP <--[upstream]--> real engine
//! ```
//!
//! The tap presents itself as an engine to the frontend (downstream) and as a
//! frontend to the engine (upstream). All frames pass through verbatim; the tap
//! decodes copies for timing observation only.
//!
//! ## Limitations (prototype)
//!
//! - Single engine, single client (client_index 0).
//! - `parallel_config_hash` is not relayed downstream (only relevant for DP > 1).
//! - No coordinator pass-through.
//! - Multi-token output chunks (spec decode, diffusion blocks): one ITL gap
//!   per chunk, with token counts in the record's `itl_tokens`.
//! - Aborted requests are silently discarded (no trace record emitted).

use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use sim_s3::TraceUri;
use tokio_util::sync::CancellationToken;
use tracing::{Level, info};

use sim_tap::tap::{TapConfig, TapMetaConfig, TokenRecording, run_tap};
use sim_trace::trace::TraceWriter;

#[derive(Debug, Parser)]
#[command(
    name = "inference-sim-tap",
    about = "Transparent recording proxy between a vLLM frontend and an engine-core."
)]
struct TapOpt {
    /// Handshake address of the real frontend to connect to (the tap acts as
    /// an engine). The default matches the sidecar convention: the frontend
    /// binds its handshake on :5570 (`vllm-rs serve --handshake-port 5570`).
    #[arg(long, default_value = "tcp://127.0.0.1:5570")]
    frontend_handshake: String,

    /// Handshake address the tap binds for the real engine (the tap acts as a
    /// frontend). The engine connects here: `vllm serve --headless
    /// --data-parallel-rpc-port 5580`.
    #[arg(long, default_value = "tcp://127.0.0.1:5580")]
    engine_handshake: String,

    /// Address the tap binds for the upstream engine's input (ROUTER socket).
    #[arg(long, default_value = "tcp://127.0.0.1:29560")]
    input_address: String,

    /// Address the tap binds for the upstream engine's output (PULL socket).
    #[arg(long, default_value = "tcp://127.0.0.1:29561")]
    output_address: String,

    /// Path or `s3://bucket/key` URI to write the JSONL trace output.
    /// Gzip-compressed when the path ends in `.gz` (recommended with
    /// --record-tokens, which grows traces by one integer per generated token);
    /// the stream is finalized on SIGINT/SIGTERM, so don't SIGKILL the tap if
    /// you want a well-terminated gzip file. An `s3://` target is written to a
    /// local scratch file and uploaded as one object after finalize.
    #[arg(long)]
    trace_out: TraceUri,

    /// Model name recorded in the trace metadata line.
    #[arg(long, default_value = "")]
    model: String,

    /// GPU type recorded in the trace metadata line (e.g. "H200").
    #[arg(long)]
    gpu: Option<String>,

    /// Tensor-parallel size recorded in the trace metadata line.
    #[arg(long)]
    tp: Option<u32>,

    /// Config hash recorded in the trace metadata line. This is the CI
    /// profile-once/replay-many cache key; the sim checks it at replay
    /// (`--expect-config-hash`) so a trace cannot be replayed against a config
    /// it was not captured for. When omitted, the tap computes the canonical
    /// fingerprint from --model/--gpu/--tp/--block-size/--max-num-seqs and the
    /// --vllm-version tag, so capture and replay derive the same value.
    #[arg(long)]
    config_hash: Option<String>,

    /// vLLM tag this capture targets (e.g. "v0.23.0"). Guards the real engine's
    /// reported version (matched on the major.minor line) so a mislabelled
    /// capture aborts, and is the reproducible vLLM input to the computed
    /// config hash. The engine's own reported version is recorded separately.
    #[arg(long)]
    vllm_version: Option<String>,

    /// Scheduler concurrency ceiling (max_num_seqs) recorded in the meta line
    /// and folded into the computed config hash.
    #[arg(long)]
    max_num_seqs: Option<u64>,

    /// Canonical string of the deployed behavioral engine flags (model, tp,
    /// gpu-mem-util, max-model-len, max-num-seqs, block-size, enforce-eager,
    /// prefix-caching, speculative, ...), folded into the config hash so two
    /// different deployments of the same model/hardware get distinct fingerprints.
    /// The capture driver (gen-capture-jobs.py) builds it; the tap can't observe the
    /// engine config on the wire. See docs/conformance.md.
    #[arg(long, default_value = "")]
    engine_config: String,

    /// Token-block size for prompt prefix fingerprints (block_hashes in the
    /// trace). Should match the engine's prefix-cache block size.
    #[arg(long, default_value_t = 16)]
    block_size: usize,

    /// Record each request's output token ids (`output_token_ids` in the
    /// trace), enabling content-identical replay via the sim's
    /// `--replay-tokens`. Off by default: with the same tokenizer the ids
    /// decode back to the generated text, so such traces carry user content.
    #[arg(
        long,
        value_enum,
        default_value_t = TokenRecording::Off,
        default_missing_value = "on",
        num_args = 0..=1
    )]
    record_tokens: TokenRecording,

    /// Path or `s3://bucket/key` URI to also write per-step scheduler-stats
    /// snapshots (JSONL, one line per engine output message that carried stats;
    /// gzip when the path ends in `.gz`). Includes spec-decoding
    /// draft/acceptance counts when the engine runs with speculative decoding.
    /// Requires the engine's stats logging to be enabled, otherwise the file
    /// stays empty. `s3://` targets upload after finalize, like --trace-out.
    #[arg(long)]
    step_stats_out: Option<TraceUri>,
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(Level::INFO.to_string())),
        )
        .init();
}

/// A cancellation token triggered by SIGINT (Ctrl-C) or SIGTERM. SIGTERM is the
/// signal Kubernetes sends on `scale 0`/pod delete, so without it the tap's
/// finalize (gzip trailer + S3 upload) never runs on cluster teardown. Mirrors
/// the main sim's `wait_for_signal`.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    tokio::spawn(async move {
        let signal = wait_for_signal().await;
        info!(signal, "received shutdown signal, shutting down tap");
        shutdown.cancel();
    });
    token
}

#[cfg(unix)]
async fn wait_for_signal() -> &'static str {
    use tokio::signal::unix::{SignalKind, signal};
    match signal(SignalKind::terminate()) {
        Ok(mut sigterm) => {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => "SIGINT",
                _ = sigterm.recv() => "SIGTERM",
            }
        }
        Err(error) => {
            tracing::warn!(%error, "failed to install SIGTERM handler; handling SIGINT only");
            let _ = tokio::signal::ctrl_c().await;
            "SIGINT"
        }
    }
}

#[cfg(not(unix))]
async fn wait_for_signal() -> &'static str {
    let _ = tokio::signal::ctrl_c().await;
    "ctrl-c"
}

fn main() -> ExitCode {
    init_tracing();

    let opt = TapOpt::parse();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to build Tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = runtime.block_on(async move { run_main(opt).await });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inference-sim-tap error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_main(opt: TapOpt) -> Result<()> {
    let scratch_dir = std::env::temp_dir();
    let trace_local = opt.trace_out.write_path(&scratch_dir);
    let step_local = opt
        .step_stats_out
        .as_ref()
        .map(|uri| uri.write_path(&scratch_dir));

    let mut writer = TraceWriter::create(&trace_local)?;

    // The meta line is written by run_tap after the handshake, so it can stamp
    // the engine's reported version and raw ready-response bytes.
    let meta = TapMetaConfig {
        model: opt.model,
        gpu: opt.gpu,
        tp: opt.tp,
        max_num_seqs: opt.max_num_seqs,
        block_size: opt.block_size,
        config_hash: opt.config_hash,
        vllm_tag: opt.vllm_version,
        engine_config: opt.engine_config,
    };

    let config = TapConfig {
        frontend_handshake: opt.frontend_handshake,
        engine_handshake: opt.engine_handshake,
        input_address: opt.input_address,
        output_address: opt.output_address,
        block_size: opt.block_size,
        record_tokens: opt.record_tokens,
    };

    let mut step_writer = step_local.as_deref().map(TraceWriter::create).transpose()?;

    let shutdown = shutdown_signal();
    // Finalize and upload even on transport error: an untrailered gzip stream
    // reads as truncated, so a clean shutdown must always land a complete object.
    let result = run_tap(config, meta, &mut writer, step_writer.as_mut(), shutdown).await;
    writer.finish()?;
    opt.trace_out.upload(&trace_local).await?;
    if let Some(step_writer) = step_writer {
        step_writer.finish()?;
        if let (Some(uri), Some(local)) = (&opt.step_stats_out, &step_local) {
            uri.upload(local).await?;
        }
    }
    result
}
