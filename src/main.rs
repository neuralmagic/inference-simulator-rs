use anyhow::{Context as _, Result};
use clap::Parser as _;
use inference_simulator_rs::Opt;
use tokio_util::sync::CancellationToken;
use tracing::{Level, info};

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(Level::INFO.to_string())),
        )
        .init();
}

/// A cancellation token triggered by SIGINT (Ctrl-C) or SIGTERM, mirroring vLLM's
/// engine-core signal handlers (k8s sends SIGTERM on pod termination). What happens
/// next is up to `--shutdown-timeout`: drain in-flight requests or abort them.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();
    tokio::spawn(async move {
        let signal = wait_for_signal().await;
        info!(signal, "received shutdown signal");
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

fn main() -> Result<()> {
    init_tracing();
    let opt = Opt::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime")?;

    runtime.block_on(async move {
        let shutdown = shutdown_signal();
        inference_simulator_rs::run(opt, shutdown).await
    })
}
