//! KV-cache event publishing over ZMQ, wire-compatible with vLLM's `ZmqEventPublisher`
//! and the llm-d cache-aware router's subscriber (`llm-d-kv-cache/pkg/kvevents`).
//!
//! The router builds its prefix tree from these events, so the wire format has to match
//! exactly. Confirmed against both the vLLM producer (`vllm/distributed/kv_events.py`) and
//! the Go consumer + the reference Go publisher (`llm-d-inference-sim`):
//!
//!   * Socket: ZMQ **PUB**. Bind when the endpoint is a wildcard (`*`, `::`, `ipc://`,
//!     `inproc://`), else connect, matching vLLM's heuristic.
//!   * Each published message is a **3-frame** multipart: frame 0 is the topic bytes
//!     (llm-d convention `kv@<pod-id>@<model-name>`), frame 1 is the sequence number as an
//!     8-byte big-endian integer (monotonic), frame 2 is the msgpack payload.
//!   * Payload: a msgpack array `[ts: f64, [events...], dp_rank: int]`.
//!   * Each event: a msgpack array with the type tag string first, then positional fields
//!     (msgspec `array_like=True, tag=True`). The consumer is positional and length-guarded,
//!     so trailing optional fields may be omitted:
//!
//! ```text
//! BlockStored:      ["BlockStored", block_hashes, parent_hash|nil, token_ids, block_size, lora_id|nil, "GPU"]
//! BlockRemoved:     ["BlockRemoved", block_hashes, "GPU"]
//! AllBlocksCleared: ["AllBlocksCleared"]
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use rmpv::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zeromq::{PubSocket, Socket as _, SocketSend as _, ZmqMessage};

use crate::blockpool::KvCacheEvent;

/// vLLM's `MEDIUM_GPU`: the device tier reported on every store/remove. The mock has no
/// real device, but the router keys on this string, so we report what a GPU engine would.
const MEDIUM_GPU: &str = "GPU";

/// Configuration for KV-cache event publishing, mirroring vLLM's `--kv-events-config`.
#[derive(Debug, Clone)]
pub struct KvEventsConfig {
    /// When false, nothing is published and [`spawn`] returns `None`.
    pub enabled: bool,
    /// ZMQ endpoint. Wildcard endpoints (`tcp://*:5557`, `ipc://...`) bind; concrete hosts
    /// connect, matching vLLM.
    pub endpoint: String,
    /// Topic the router subscribes to. The llm-d router expects `kv@<pod-id>@<model-name>`
    /// (its default SUB filter is `kv@`); the deploy layer fills the pod id and model name.
    pub topic: String,
}

/// Handle the engine uses to publish event batches. Cloneable and cheap; dropping all
/// clones lets the publisher task drain and exit.
#[derive(Debug, Clone)]
pub struct KvEventTx {
    tx: mpsc::Sender<Vec<KvCacheEvent>>,
}

impl KvEventTx {
    /// Queue a batch of events for publishing. Non-blocking: like vLLM's high-water-mark
    /// behavior, a full queue drops the batch (with a warning) rather than stalling the
    /// engine loop. No-op for an empty batch.
    pub fn publish(&self, events: Vec<KvCacheEvent>) {
        if events.is_empty() {
            return;
        }
        if let Err(err) = self.tx.try_send(events) {
            warn!(%err, "kv-event queue full or closed; dropping batch");
        }
    }
}

/// Current UNIX time in fractional seconds for the batch `ts`.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// True if the endpoint should `bind` (a stable wildcard) rather than `connect`.
fn should_bind(endpoint: &str) -> bool {
    endpoint.contains('*')
        || endpoint.contains("::")
        || endpoint.starts_with("ipc://")
        || endpoint.starts_with("inproc://")
}

/// The address to actually bind. vLLM/pyzmq accept `tcp://*:PORT`, but the pure-Rust
/// `zeromq` crate tries to resolve `*` as a hostname, so translate the wildcard host to
/// `0.0.0.0` (all interfaces) for tcp. Other schemes pass through unchanged.
fn bind_addr(endpoint: &str) -> String {
    if endpoint.starts_with("tcp://") {
        endpoint.replacen("://*:", "://0.0.0.0:", 1)
    } else {
        endpoint.to_string()
    }
}

/// Encode a block hash as an 8-byte big-endian msgpack binary. The router's consumer reads
/// hashes as `uint64`, `int64`, or `[]byte` (taking the last 8 bytes big-endian); a compact
/// msgpack int would decode to a narrower Go type (e.g. uint16) that it rejects, and our
/// FNV hashes aren't guaranteed full-width like vLLM's, so we always send the fixed-width
/// byte form (vLLM's `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` representation).
fn hash_value(hash: u64) -> Value {
    Value::Binary(hash.to_be_bytes().to_vec())
}

/// Encode one event as its msgpack array value (tag string first, then positional fields).
fn encode_event(event: &KvCacheEvent) -> Value {
    let hashes = |hs: &[u64]| Value::Array(hs.iter().map(|&h| hash_value(h)).collect());
    match event {
        KvCacheEvent::Stored {
            block_hashes,
            parent_hash,
            token_ids,
            block_size,
        } => Value::Array(vec![
            Value::from("BlockStored"),
            hashes(block_hashes),
            parent_hash.map(hash_value).unwrap_or(Value::Nil),
            Value::Array(token_ids.iter().map(|&t| Value::from(t)).collect()),
            Value::from(*block_size as u64),
            Value::Nil, // lora_id
            Value::from(MEDIUM_GPU),
        ]),
        KvCacheEvent::Removed { block_hashes } => Value::Array(vec![
            Value::from("BlockRemoved"),
            hashes(block_hashes),
            Value::from(MEDIUM_GPU),
        ]),
        KvCacheEvent::AllCleared => Value::Array(vec![Value::from("AllBlocksCleared")]),
    }
}

/// Encode a full event batch (`[ts, [events...], dp_rank]`) to msgpack bytes.
fn encode_batch(events: &[KvCacheEvent]) -> Result<Vec<u8>> {
    let batch = Value::Array(vec![
        Value::from(now_secs()),
        Value::Array(events.iter().map(encode_event).collect()),
        Value::from(0u64), // data_parallel_rank
    ]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &batch).context("encode kv-event batch")?;
    Ok(buf)
}

/// Build the 3-frame ZMQ message `[topic, seq(8, big-endian), payload]`.
fn build_message(topic: &str, seq: u64, payload: Vec<u8>) -> ZmqMessage {
    let mut msg = ZmqMessage::from(topic.as_bytes().to_vec());
    msg.push_back(seq.to_be_bytes().to_vec().into());
    msg.push_back(payload.into());
    msg
}

/// Start the KV-event publisher. Returns `None` (and binds nothing) when disabled. On
/// success, returns a [`KvEventTx`] the engine clones to publish batches; the background
/// task owns the PUB socket and the sequence counter and runs until `shutdown` or until
/// every `KvEventTx` is dropped.
pub async fn spawn(cfg: KvEventsConfig, shutdown: CancellationToken) -> Result<Option<KvEventTx>> {
    if !cfg.enabled {
        debug!("kv-cache events disabled");
        return Ok(None);
    }

    let mut socket = PubSocket::new();
    if should_bind(&cfg.endpoint) {
        let addr = bind_addr(&cfg.endpoint);
        socket
            .bind(&addr)
            .await
            .with_context(|| format!("bind kv-event PUB socket to {addr}"))?;
        info!(
            endpoint = addr,
            topic = cfg.topic,
            "kv-event publisher bound"
        );
    } else {
        socket
            .connect(&cfg.endpoint)
            .await
            .with_context(|| format!("connect kv-event PUB socket to {}", cfg.endpoint))?;
        info!(
            endpoint = cfg.endpoint,
            topic = cfg.topic,
            "kv-event publisher connected"
        );
    }

    // A bounded queue gives us vLLM-style backpressure: the engine drops on a full queue
    // rather than blocking generation on a slow subscriber.
    let (tx, mut rx) = mpsc::channel::<Vec<KvCacheEvent>>(1024);

    tokio::spawn(async move {
        let topic = cfg.topic;
        let mut seq: u64 = 0;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                batch = rx.recv() => {
                    let Some(events) = batch else { break };
                    let payload = match encode_batch(&events) {
                        Ok(p) => p,
                        Err(err) => {
                            warn!(%err, "failed to encode kv-event batch; skipping");
                            continue;
                        }
                    };
                    let message = build_message(&topic, seq, payload);
                    if let Err(err) = socket.send(message).await {
                        warn!(%err, "failed to send kv-event batch");
                        continue;
                    }
                    seq = seq.wrapping_add(1);
                }
            }
        }
        debug!("kv-event publisher task exiting");
    });

    Ok(Some(KvEventTx { tx }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the bytes back into an rmpv Value for structural assertions.
    fn decode(bytes: &[u8]) -> Value {
        let mut cursor = std::io::Cursor::new(bytes);
        rmpv::decode::read_value(&mut cursor).expect("decode msgpack")
    }

    fn as_array(v: &Value) -> &Vec<Value> {
        v.as_array().expect("expected array")
    }

    /// Decode the 8-byte big-endian binary hash form back to u64.
    fn hash_u64(v: &Value) -> u64 {
        let bytes = v.as_slice().expect("hash is binary");
        assert_eq!(bytes.len(), 8, "hash is 8 bytes");
        u64::from_be_bytes(bytes.try_into().unwrap())
    }

    fn hash_list(v: &Value) -> Vec<u64> {
        as_array(v).iter().map(hash_u64).collect()
    }

    #[test]
    fn batch_round_trips_to_the_expected_array_shape() {
        let events = vec![KvCacheEvent::Stored {
            block_hashes: vec![10, 20, 30],
            parent_hash: Some(7),
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
        }];
        let bytes = encode_batch(&events).unwrap();
        let decoded = decode(&bytes);

        let batch = as_array(&decoded);
        assert_eq!(batch.len(), 3, "[ts, events, dp_rank]");
        assert!(batch[0].is_f64(), "ts is a float");
        assert_eq!(batch[2].as_u64(), Some(0), "dp_rank = 0");

        let evs = as_array(&batch[1]);
        assert_eq!(evs.len(), 1);
        let stored = as_array(&evs[0]);
        // Tag, block_hashes, parent_hash, token_ids, block_size, lora_id, medium
        assert_eq!(stored[0].as_str(), Some("BlockStored"));
        assert_eq!(hash_list(&stored[1]), vec![10, 20, 30]);
        assert_eq!(hash_u64(&stored[2]), 7, "parent hash");
        assert_eq!(
            as_array(&stored[3])
                .iter()
                .filter_map(Value::as_u64)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(stored[4].as_u64(), Some(4), "block size");
        assert!(stored[5].is_nil(), "lora_id");
        assert_eq!(stored[6].as_str(), Some("GPU"), "medium");
    }

    #[test]
    fn no_parent_block_encodes_nil() {
        let events = vec![KvCacheEvent::Stored {
            block_hashes: vec![1],
            parent_hash: None,
            token_ids: vec![0, 0, 0, 0],
            block_size: 4,
        }];
        let decoded = decode(&encode_batch(&events).unwrap());
        let stored = as_array(&as_array(&as_array(&decoded)[1])[0]);
        assert!(stored[2].is_nil(), "no parent -> nil");
    }

    #[test]
    fn removed_and_cleared_have_their_tags() {
        let decoded = decode(
            &encode_batch(&[
                KvCacheEvent::Removed {
                    block_hashes: vec![42],
                },
                KvCacheEvent::AllCleared,
            ])
            .unwrap(),
        );
        let evs = as_array(&as_array(&decoded)[1]);
        let removed = as_array(&evs[0]);
        assert_eq!(removed[0].as_str(), Some("BlockRemoved"));
        assert_eq!(hash_list(&removed[1]), vec![42]);
        assert_eq!(removed[2].as_str(), Some("GPU"));

        let cleared = as_array(&evs[1]);
        assert_eq!(cleared[0].as_str(), Some("AllBlocksCleared"));
        assert_eq!(cleared.len(), 1);
    }

    #[test]
    fn bind_vs_connect_heuristic() {
        assert!(should_bind("tcp://*:5557"));
        assert!(should_bind("ipc:///tmp/kv"));
        assert!(should_bind("tcp://[::]:5557"));
        assert!(!should_bind("tcp://10.0.0.5:5557"));
    }

    #[test]
    fn bind_addr_translates_wildcard_host() {
        // The pure-Rust zeromq crate can't resolve `*`, so we bind all interfaces instead.
        assert_eq!(bind_addr("tcp://*:5556"), "tcp://0.0.0.0:5556");
        assert_eq!(bind_addr("tcp://10.0.0.5:5556"), "tcp://10.0.0.5:5556");
        assert_eq!(bind_addr("ipc:///tmp/kv"), "ipc:///tmp/kv");
    }

    #[test]
    fn message_has_three_frames_with_big_endian_seq() {
        let msg = build_message("kv@pod-a@model", 0x0102, vec![0xaa, 0xbb]);
        let frames = msg.into_vec();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].as_ref(), b"kv@pod-a@model");
        assert_eq!(frames[1].as_ref(), &[0, 0, 0, 0, 0, 0, 1, 2]);
        assert_eq!(frames[2].as_ref(), &[0xaa, 0xbb]);
    }
}
