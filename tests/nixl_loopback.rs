//! The NIXL transfer mechanic: a prefill registers and stages a patterned KV buffer,
//! then a decode posts a real NIXL READ and verifies the bytes. Exercised in-process
//! (one agent reading its own staged buffer) so it needs no ZMQ side channel; the
//! cross-pod path reuses the same `read_staged` mechanic over the side channel.
//!
//! Needs real libnixl + UCX, so it runs on Linux and skips under the stub bindings.
//! Run it: `cargo test --features nixl` on a box with NIXL installed.

#![cfg(feature = "nixl")]

use mock_engine_nixl::dataplane::{NixlConfig, PdRole, RequestKv, make_data_plane, nixl_is_stub};

#[test]
fn loopback_dram_transfer() {
    if nixl_is_stub() {
        eprintln!("skipping NIXL loopback: built against stub bindings (no real libnixl)");
        return;
    }

    let cfg = NixlConfig {
        kv_block_bytes: 4096,
        tokens_per_block: 16,
        engine_id: "mock-loopback".to_string(),
        side_channel_host: "127.0.0.1".to_string(),
        side_channel_port: 5600,
    };

    // A prefill plane registers + stages the KV; the same agent reads it back, which
    // drives the real register -> descriptor -> create_xfer_req(READ) -> poll -> verify
    // path without needing the cross-pod side channel.
    let mut plane = make_data_plane(PdRole::Prefill, cfg);
    let kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40,
    };

    let remote = plane
        .advertise_prefilled(kv)
        .expect("prefill should register + advertise KV");
    assert_eq!(remote.engine_id, "mock-loopback");
    // 40 tokens / 16 per block = ceil -> 3 blocks.
    assert_eq!(remote.block_ids, vec![0, 1, 2]);

    let bytes = plane
        .pull_prefilled(kv, &remote)
        .expect("decode should READ + verify the staged KV over NIXL");
    assert_eq!(
        bytes,
        3 * 4096,
        "expected the full fabricated KV to transfer"
    );
}
