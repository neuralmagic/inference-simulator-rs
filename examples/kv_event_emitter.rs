//! Standalone KV-cache event emitter: stands up the real publisher and emits one batch of
//! each event type (BlockStored, BlockRemoved, AllBlocksCleared) on a loop. No frontend or
//! engine handshake needed, so an external subscriber (the Go smoke harness in
//! `scripts/kv_events_smoke/`, which drives the real llm-d-kv-cache decoder) can verify our
//! wire bytes against the actual router consumer.
//!
//! Usage: `cargo run --example kv_event_emitter -- [endpoint] [topic]`
//!   endpoint default: tcp://*:5556   (the precise-prefix-cache-routing guide's port)
//!   topic    default: kv@127.0.0.1:8000@mock-model

use std::time::Duration;

use inference_simulator_rs::blockpool::KvCacheEvent;
use inference_simulator_rs::kvevents::{KvEventsConfig, spawn};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let endpoint = args.next().unwrap_or_else(|| "tcp://*:5556".to_string());
    let topic = args
        .next()
        .unwrap_or_else(|| "kv@127.0.0.1:8000@mock-model".to_string());

    // Block size 64 matches the precise-prefix-cache-routing guide's tokenProcessor config.
    const BLOCK_SIZE: usize = 64;
    let token_ids: Vec<u32> = (0..BLOCK_SIZE as u32).collect();

    let shutdown = CancellationToken::new();
    let tx = spawn(
        KvEventsConfig {
            enabled: true,
            endpoint: endpoint.clone(),
            topic: topic.clone(),
        },
        shutdown.clone(),
    )
    .await?
    .expect("publisher should be enabled");

    tracing::info!(endpoint, topic, "emitting KV-cache events; Ctrl-C to stop");

    let mut ticks: u64 = 0;
    loop {
        // One batch exercising all three event shapes the router decodes.
        tx.publish(vec![
            KvCacheEvent::Stored {
                block_hashes: vec![0xAA00 + ticks],
                parent_hash: None,
                token_ids: token_ids.clone(),
                block_size: BLOCK_SIZE,
            },
            KvCacheEvent::Removed {
                block_hashes: vec![0xAA00 + ticks],
            },
            KvCacheEvent::AllCleared,
        ]);
        ticks += 1;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
