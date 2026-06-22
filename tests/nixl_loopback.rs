//! The first real NIXL transfer: a prefill agent fills + advertises patterned KV slots in
//! its registered pool and serves its metadata over a listener; a separate decode agent
//! fetches that metadata by host:port, posts a multi-descriptor NIXL READ (one descriptor
//! per paged block), and verifies the bytes. Two distinct agents in one process (same shape
//! as the cross-pod path, just over loopback).
//!
//! Needs real libnixl + UCX, so it runs on Linux and skips under the stub bindings.
//! Run it: `cargo test --features nixl` on a box with NIXL installed.

#![cfg(feature = "nixl")]

use vllm_vcr::dataplane::{NixlConfig, PdRole, RequestKv, make_data_plane, nixl_is_stub};

fn cfg(engine_id: &str, port: u32) -> NixlConfig {
    NixlConfig {
        kv_block_bytes: 4096,
        kv_cache_blocks: 16,
        engine_id: engine_id.to_string(),
        side_channel_host: "127.0.0.1".to_string(),
        side_channel_port: port,
    }
}

#[test]
fn loopback_paged_dram_transfer() {
    if nixl_is_stub() {
        eprintln!("skipping NIXL loopback: built against stub bindings (no real libnixl)");
        return;
    }

    // Prefill: listens on 5600 and serves its pool. Decode: a distinct agent that pulls.
    let mut prefill = make_data_plane(PdRole::Prefill, cfg("mock-prefill", 5600));
    let mut decode = make_data_plane(PdRole::Decode, cfg("mock-decode", 5601));

    // Three non-contiguous slots, to prove the paging maps block_id -> addr (not a single
    // contiguous span): blocks 2, 5, 9 in the prefill's pool.
    let prefill_slots = [2usize, 5, 9];
    let prefill_kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40, // 40 tokens / 16 -> 3 blocks
        block_ids: &prefill_slots,
    };

    let remote = prefill
        .advertise_prefilled(prefill_kv)
        .expect("prefill should fill + advertise paged KV");
    assert_eq!(remote.engine_id, "mock-prefill");
    assert_eq!(remote.block_ids, vec![2, 5, 9]);
    assert_eq!(remote.request_id, "req-loopback-1");
    // No mock-extension addressing in the advertised params: the decode learns the pool base
    // from the metadata side channel (PoolDescriptor) during the pull below.

    // The decode lands the blocks into its own (different) pool slots.
    let decode_slots = [0usize, 1, 2];
    let decode_kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40,
        block_ids: &decode_slots,
    };

    let bytes = decode
        .pull_prefilled(decode_kv, &remote)
        .expect("decode should fetch md, multi-desc READ over NIXL, and verify the pattern");
    assert_eq!(
        bytes,
        3 * 4096,
        "expected all three paged blocks to transfer"
    );
}
