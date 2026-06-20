//! Live smoke test for the KV-cache event publisher: stand up the real publisher on a real
//! ZMQ PUB socket, connect a real SUB socket (the same role the llm-d router's subscriber
//! plays), and confirm a published batch arrives with the exact 3-frame layout and
//! tagged-array msgpack the router decodes. Pure-Rust `zeromq` on both ends, so it runs
//! anywhere (no libzmq, no GPU, no NIXL).
//!
//! This proves our side of the wire contract end to end over a real transport. The
//! authoritative interop check against the actual Go decoder lives in
//! `scripts/kv_events_smoke/` (needs a checkout of llm-d-kv-cache + Go).

use std::time::Duration;

use tokio_util::sync::CancellationToken;
use vllm_vcr::blockpool::KvCacheEvent;
use vllm_vcr::kvevents::{KvEventsConfig, spawn};
use zeromq::{Socket as _, SocketRecv as _, SubSocket};

#[tokio::test]
async fn events_decode_over_a_real_zmq_sub_socket() {
    // Unique per-process ipc endpoint so parallel test binaries don't collide.
    let endpoint = format!("ipc:///tmp/inf-sim-kv-{}.ipc", std::process::id());
    let topic = "kv@smoke-pod@smoke-model".to_string();

    let token = CancellationToken::new();
    let tx = spawn(
        KvEventsConfig {
            enabled: true,
            endpoint: endpoint.clone(),
            topic: topic.clone(),
        },
        token.clone(),
    )
    .await
    .expect("publisher should start")
    .expect("publisher should be enabled");

    // The subscriber side, exactly as the router connects: SUB + the `kv@` topic filter.
    let mut sub = SubSocket::new();
    sub.connect(&endpoint).await.expect("sub connect");
    sub.subscribe("kv@").await.expect("subscribe");
    // Give the subscription time to propagate to the publisher before the first send.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let event = KvCacheEvent::Stored {
        block_hashes: vec![111, 222],
        parent_hash: Some(7),
        token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
        block_size: 4,
        lora_name: None,
    };

    // PUB/SUB is a slow joiner: re-publish until the SUB actually receives one (or give up).
    let mut message = None;
    for _ in 0..50 {
        tx.publish(vec![event.clone()]);
        if let Ok(Ok(m)) = tokio::time::timeout(Duration::from_millis(200), sub.recv()).await {
            message = Some(m);
            break;
        }
    }
    let message = message.expect("subscriber should receive a published batch");

    // Frame layout: [topic, seq(8, big-endian), payload].
    let frames = message.into_vec();
    assert_eq!(frames.len(), 3, "topic, seq, payload");
    assert_eq!(frames[0].as_ref(), topic.as_bytes());
    assert_eq!(frames[1].len(), 8, "seq is 8 bytes big-endian");

    // Payload: [ts, [events...], dp_rank], events tagged-array.
    let mut cursor = std::io::Cursor::new(frames[2].as_ref());
    let batch = rmpv::decode::read_value(&mut cursor).expect("decode msgpack batch");
    let batch = batch.as_array().expect("batch is an array");
    assert!(batch[0].is_f64(), "ts");

    let events = batch[1].as_array().expect("events array");
    assert_eq!(events.len(), 1);
    // Hashes are 8-byte big-endian binaries (the consumer reads uint64/int64/[]byte).
    let hash_u64 = |v: &rmpv::Value| -> u64 {
        u64::from_be_bytes(v.as_slice().expect("binary hash").try_into().unwrap())
    };
    let stored = events[0].as_array().expect("event array");
    assert_eq!(stored[0].as_str(), Some("BlockStored"));
    assert_eq!(
        stored[1]
            .as_array()
            .unwrap()
            .iter()
            .map(hash_u64)
            .collect::<Vec<_>>(),
        vec![111, 222]
    );
    assert_eq!(hash_u64(&stored[2]), 7, "parent hash");
    assert_eq!(
        stored[3]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(rmpv::Value::as_u64)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(stored[4].as_u64(), Some(4), "block size");
    assert_eq!(stored[6].as_str(), Some("GPU"), "medium");

    token.cancel();
}
