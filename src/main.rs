//! `vllm-vcr`: record, play, and inspect vLLM engine-core traces.
//!
//! One binary, three subcommands (the VCR metaphor):
//!   - `record`  — tap a live vLLM frontend ↔ engine-core link and write a trace.
//!   - `play`    — run the mock engine-core backend (replay a trace or simulate).
//!   - `inspect` — convert, summarize, Perfetto-render, and calibrate traces.
//!
//! `record` and `play` bake the vLLM engine-core protocol in (per build line);
//! `inspect` never handshakes, so it runs on any build.

mod inspect;
mod record;

use std::process::ExitCode;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(
    name = "vllm-vcr",
    version,
    about = "Record, play, and inspect vLLM engine-core traces."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Record a trace by tapping a live vLLM frontend ↔ engine-core link
    /// (transparent ZMQ proxy; frames relayed verbatim, timing observed).
    Record(record::RecordArgs),

    /// Play a trace back through the mock engine-core backend, or run it as a
    /// GPU-free vLLM engine for frontend / prefill-decode testing.
    Play(Box<vllm_vcr::Opt>),

    /// Inspect traces: convert benchmark reports, summarize, render Perfetto,
    /// and run calibration.
    #[command(subcommand)]
    Inspect(inspect::InspectCommand),

    /// Print a shell completion script to stdout. Source it from your shell rc,
    /// e.g. `vllm-vcr completions fish > ~/.config/fish/completions/vllm-vcr.fish`.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
    },
}

/// Logs go to stderr so `inspect`'s stdout stays clean for piping (Perfetto
/// JSON, summaries). INFO default keeps `record`/`play` debuggable; override
/// with `RUST_LOG`.
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
}

/// Run the mock engine-core backend on a multi-thread runtime until a shutdown
/// signal (SIGINT/SIGTERM) or transport failure.
fn play(opt: vllm_vcr::Opt) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(async move {
        let shutdown: CancellationToken = vllm_vcr::shutdown_signal();
        vllm_vcr::run(opt, shutdown).await
    })
}

fn main() -> ExitCode {
    init_tracing();
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Record(args) => record::run(args).map(|()| ExitCode::SUCCESS),
        Command::Play(opt) => play(*opt).map(|()| ExitCode::SUCCESS),
        Command::Inspect(command) => inspect::run(command),
        Command::Completions { shell } => {
            let mut cmd = <Cli as clap::CommandFactory>::command();
            clap_complete::generate(shell, &mut cmd, "vllm-vcr", &mut std::io::stdout());
            Ok(ExitCode::SUCCESS)
        }
    };

    match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
