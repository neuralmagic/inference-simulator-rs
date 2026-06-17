//! Does the tap's serde model decode a REAL DiffusionGemma multimodal request?
//!
//! The fixtures in `tests/fixtures/*.hex` are the primary msgpack frame
//! (`frames[0]`) produced by vLLM's REAL `MsgpackEncoder` at the deployed rev
//! 16e91176, captured offline (CPU, no GPU) by
//! `deploy/trace-capture/mm_encode_groundtruth.py`. This is exactly the blob the
//! tap feeds to `decode_msgpack::<EngineCoreRequest>` in `observe_request`.
//!
//! If this fails, the multimodal blocker is a serde gap in the tap's request
//! model: a decode failure there means the request is never tracked, so every
//! engine output for it is dropped and no trace record is written. If it
//! succeeds, the request path is sound and the blocker lives upstream of the tap
//! (frontend mm-preprocessing stall / DP handshake wedge), not in serde.

use vllm_engine_core_client::protocol::{EngineCoreRequest, decode_msgpack};

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    assert!(hex.len() % 2 == 0, "odd-length hex");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

/// Large pixel tensor rides as an aux frame; the blob carries an integer
/// aux-index in its place. This is the path the crate's `python_compat` test
/// never exercises and the real DiffusionGemma image request takes.
#[test]
fn decodes_real_multimodal_request_with_aux_tensor() {
    let bytes = hex_to_bytes(include_str!("fixtures/mm_request_large_aux.hex"));
    let req: EngineCoreRequest = decode_msgpack(&bytes)
        .expect("tap must decode a real multimodal EngineCoreRequest (aux-tensor)");

    assert_eq!(req.request_id, "req-mm-large");
    let mm = req.mm_features.as_ref().expect("mm_features present");
    assert_eq!(mm.len(), 1, "one image feature");
    assert_eq!(mm[0].modality, "image");
    // 2 text + 256 image placeholders + 1 text.
    assert_eq!(req.prompt_token_ids.as_ref().map(Vec::len), Some(259));
}

/// Small tensor inlined as `Ext(3)` in the blob (single frame, no aux).
#[test]
fn decodes_real_multimodal_request_with_inline_tensor() {
    let bytes = hex_to_bytes(include_str!("fixtures/mm_request_small_inline.hex"));
    let req: EngineCoreRequest = decode_msgpack(&bytes)
        .expect("tap must decode a real multimodal EngineCoreRequest (inline tensor)");
    assert_eq!(req.request_id, "req-mm-small");
    assert!(req.mm_features.is_some(), "mm_features present");
}
