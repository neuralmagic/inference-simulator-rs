//! Engine-core recording tap: a transparent ZMQ proxy between a real vLLM
//! frontend and a real engine-core, recording per-request timing into a JSONL
//! trace file.
//!
//! ## Usage
//!
//! ```text
//! inference-sim-tap \
//!   --frontend-handshake ipc:///tmp/frontend.ipc \
//!   --engine-handshake ipc:///tmp/tap-engine.ipc \
//!   --input-address tcp://127.0.0.1:29560 \
//!   --output-address tcp://127.0.0.1:29561 \
//!   --trace-out /tmp/trace.jsonl \
//!   --model my-model
//! ```
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
//! - Multi-token output chunks: ITL is divided evenly across the tokens in the chunk.
//! - Aborted requests are silently discarded (no trace record emitted).

use std::path::Path;
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing::{Level, info};

use inference_simulator_rs::tap::{TapConfig, TokenRecording, run_tap, write_meta};
use inference_simulator_rs::trace::TraceWriter;

#[derive(Debug, Parser)]
#[command(
    name = "inference-sim-tap",
    about = "Transparent recording proxy between a vLLM frontend and an engine-core."
)]
struct TapOpt {
    /// Handshake address of the real frontend to connect to (the tap acts as an engine).
    #[arg(long)]
    frontend_handshake: String,

    /// Handshake address the tap binds for the real engine (the tap acts as a frontend).
    #[arg(long)]
    engine_handshake: String,

    /// Address the tap binds for the upstream engine's input (ROUTER socket).
    #[arg(long, default_value = "tcp://127.0.0.1:29560")]
    input_address: String,

    /// Address the tap binds for the upstream engine's output (PULL socket).
    #[arg(long, default_value = "tcp://127.0.0.1:29561")]
    output_address: String,

    /// Path to write the JSONL trace output. Gzip-compressed when the path
    /// ends in `.gz` (recommended with --record-tokens, which grows traces by
    /// one integer per generated token); the stream is finalized on Ctrl-C,
    /// so don't SIGKILL the tap if you want a well-terminated gzip file.
    #[arg(long)]
    trace_out: String,

    /// Model name recorded in the trace metadata line.
    #[arg(long, default_value = "")]
    model: String,

    /// GPU type recorded in the trace metadata line (e.g. "H200").
    #[arg(long)]
    gpu: Option<String>,

    /// Tensor-parallel size recorded in the trace metadata line.
    #[arg(long)]
    tp: Option<u32>,

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
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(Level::INFO.to_string())),
        )
        .init();
}

fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            info!("received Ctrl-C, shutting down tap");
            shutdown.cancel();
        }
    });
    token
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
    let mut writer = TraceWriter::create(Path::new(&opt.trace_out))?;

    write_meta(
        &mut writer,
        &opt.model,
        opt.gpu.as_deref(),
        opt.tp,
        opt.block_size,
    )?;

    let config = TapConfig {
        frontend_handshake: opt.frontend_handshake,
        engine_handshake: opt.engine_handshake,
        input_address: opt.input_address,
        output_address: opt.output_address,
        model: opt.model,
        block_size: opt.block_size,
        record_tokens: opt.record_tokens,
    };

    let shutdown = shutdown_signal();
    // Finalize the trace even when the proxy dies on a transport error: a
    // gzip stream without its trailer reads as truncated.
    let result = run_tap(config, &mut writer, shutdown).await;
    writer.finish()?;
    result
}
