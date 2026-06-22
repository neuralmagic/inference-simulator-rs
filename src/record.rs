//! `vllm-vcr record`: engine-core recording tap. A transparent ZMQ proxy
//! between a real vLLM frontend and a real engine-core that records per-request
//! timing into a JSONL trace file.
//!
//! ## Usage
//!
//! ```text
//! vllm-vcr record --trace-out /tmp/trace.jsonl --model my-model
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

use anyhow::{Context as _, Result};
use clap::Args;
use sim_s3::TraceUri;

use sim_tap::tap::{TapConfig, TapMetaConfig, TokenRecording, run_tap};
use sim_trace::trace::TraceWriter;

#[derive(Debug, Args)]
pub struct RecordArgs {
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

/// Run the recording tap on a multi-thread runtime until a shutdown signal
/// (SIGINT/SIGTERM) or transport failure, then finalize and upload the trace.
pub fn run(args: RecordArgs) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(run_record(args))
}

async fn run_record(args: RecordArgs) -> Result<()> {
    let scratch_dir = std::env::temp_dir();
    let trace_local = args.trace_out.write_path(&scratch_dir);
    let step_local = args
        .step_stats_out
        .as_ref()
        .map(|uri| uri.write_path(&scratch_dir));

    let mut writer = TraceWriter::create(&trace_local)?;

    // The meta line is written by run_tap after the handshake, so it can stamp
    // the engine's reported version and raw ready-response bytes.
    let meta = TapMetaConfig {
        model: args.model,
        gpu: args.gpu,
        tp: args.tp,
        max_num_seqs: args.max_num_seqs,
        block_size: args.block_size,
        config_hash: args.config_hash,
        vllm_tag: args.vllm_version,
        engine_config: args.engine_config,
    };

    let config = TapConfig {
        frontend_handshake: args.frontend_handshake,
        engine_handshake: args.engine_handshake,
        input_address: args.input_address,
        output_address: args.output_address,
        block_size: args.block_size,
        record_tokens: args.record_tokens,
    };

    let mut step_writer = step_local.as_deref().map(TraceWriter::create).transpose()?;

    let shutdown = vllm_vcr::shutdown_signal();
    // Finalize and upload even on transport error: an untrailered gzip stream
    // reads as truncated, so a clean shutdown must always land a complete object.
    let result = run_tap(config, meta, &mut writer, step_writer.as_mut(), shutdown).await;
    writer.finish()?;
    args.trace_out.upload(&trace_local).await?;
    if let Some(step_writer) = step_writer {
        step_writer.finish()?;
        if let (Some(uri), Some(local)) = (&args.step_stats_out, &step_local) {
            uri.upload(local).await?;
        }
    }
    result
}
