//! End-to-end demo: a Gemma-diffusion-shaped trace replays byte-identically.
//!
//! This stitches together both halves of the DiffusionGemma goal on realistic
//! data, with no live rig:
//!   - TRACE shape: a multimodal request's worth of `prompt_tokens` (259 = a few
//!     text tokens + 256 image-placeholder tokens) and DIFFUSION-BLOCK output
//!     timing (`itl_tokens` carries several tokens per step, not one), exactly
//!     what the tap writes after the serde-decode fix lets it track the request.
//!   - REPLAY: booting the sim with `--replay-tokens` on that trace must
//!     reproduce the recorded `output_token_ids` byte-for-byte and the recorded
//!     finish reason.
//!
//! The decode of a *real* vLLM-encoded multimodal request is verified separately
//! by `crates/sim-tap/tests/mm_decode_groundtruth.rs`; block-burst timing replay
//! by `tests/spec_replay_fidelity.rs`. This test ties the pipeline together.

use std::io::Write;
use std::time::Duration;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

use vllm_vcr::trace::{
    TraceFinishReason, TraceMeta, TraceRecord, append_record, prompt_block_hashes,
};
use vllm_vcr::{Opt, run};

const BLOCK_SIZE: usize = 16;

/// One multimodal, diffusion-block trace record as the tap would write it.
fn diffusion_record() -> TraceRecord {
    // 2 text + 256 image placeholders + 1 text = 259 prompt tokens.
    let prompt: Vec<u32> = std::iter::once(2u32)
        .chain(std::iter::once(108))
        .chain(std::iter::repeat_n(262144, 256))
        .chain(std::iter::once(108))
        .collect();
    assert_eq!(prompt.len(), 259);

    // 24 output tokens delivered in 8-token diffusion blocks: prefill chunk of 8,
    // then two decode steps of 8 each.
    let output_token_ids: Vec<u32> = (5000u32..5024).collect();
    TraceRecord {
        prompt_tokens: prompt.len(),
        cached_tokens: 0,
        output_tokens: output_token_ids.len(),
        ttft_ms: 600.0,
        itl_ms: Some(vec![150.0, 150.0]),
        itl_tokens: Some(vec![8, 8]),
        itl_summary: None,
        concurrency: 1,
        arrival_ms: Some(0.0),
        itl_ctx: None,
        block_hashes: prompt_block_hashes(&prompt, BLOCK_SIZE),
        output_token_ids: Some(output_token_ids),
        finish_reason: Some(TraceFinishReason::Stop),
    }
}

fn write_trace(path: &std::path::Path, record: &TraceRecord) {
    let mut f = std::fs::File::create(path).expect("create trace");
    let meta = TraceMeta {
        source: Some("tap".to_string()),
        model: Some("RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic".to_string()),
        gpu: Some("H200".to_string()),
        block_size: Some(BLOCK_SIZE),
        ..TraceMeta::default()
    };
    writeln!(f, "{}", serde_json::json!({ "meta": meta })).expect("write meta");
    append_record(&mut f, record).expect("write record");
    f.flush().expect("flush");
}

#[tokio::test]
async fn diffusion_multimodal_trace_replays_byte_identical() {
    let record = diffusion_record();
    let expected = record.output_token_ids.clone().unwrap();

    let trace_path =
        std::env::temp_dir().join(format!("dg-diffusion-trace-{}.jsonl", std::process::id()));
    write_trace(&trace_path, &record);
    let trace_str = trace_path.to_str().unwrap();

    let addr = format!("ipc:///tmp/inf-sim-dg-replay-{}.ipc", std::process::id());
    let opt = Opt::parse_from([
        "play",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        trace_str,
    ]);
    let token = CancellationToken::new();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(opt, sim_token).await;
    });

    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .expect("connect timed out")
        .expect("connect failed");

    // Index-matched to record 0; prompt length mirrors the multimodal request.
    let req = EngineCoreRequest {
        request_id: "replay-0".to_string(),
        prompt_token_ids: Some(vec![42u32; record.prompt_tokens]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens: expected.len() as u32,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    };
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(Duration::from_secs(30), stream.collect::<Vec<_>>())
        .await
        .expect("stream timed out");

    let tokens: Vec<u32> = outputs
        .iter()
        .flat_map(|r| r.as_ref().expect("stream error").new_token_ids.clone())
        .collect();
    assert_eq!(
        tokens, expected,
        "replayed stream must match the recorded diffusion output byte-for-byte"
    );

    let last = outputs.last().unwrap().as_ref().unwrap();
    assert_eq!(
        last.finish_reason,
        Some(EngineCoreFinishReason::Stop),
        "finish reason must match the capture"
    );

    token.cancel();
}
