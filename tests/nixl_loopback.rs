//! The first real NIXL transfer: a prefill engine fabricates and advertises KV,
//! a decode engine pulls it over NIXL (UCX) and verifies the bytes, all in one
//! process with two distinct agents (mirrors NIXL's own single-process example).
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
    };
    let mut prefill = make_data_plane(PdRole::Prefill, cfg);
    let mut decode = make_data_plane(PdRole::Decode, cfg);

    let kv = RequestKv {
        request_id: "req-loopback-1",
        num_tokens: 40,
    };

    let params = prefill
        .advertise_prefilled(kv)
        .expect("prefill should advertise KV params");

    let bytes = decode
        .pull_prefilled(kv, &params)
        .expect("decode should pull KV over NIXL and verify the pattern");

    // 40 tokens / 16 per block = ceil -> 3 blocks * 4096 bytes.
    assert_eq!(
        bytes,
        3 * 4096,
        "expected the full fabricated KV to transfer"
    );

    prefill.release(kv.request_id);
}
