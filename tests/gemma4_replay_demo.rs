//! End-to-end demo: a REAL gemma-4 multimodal capture replays byte-identically.
//!
//! Unlike `diffusion_replay_demo` (synthetic record), this drives the committed
//! fixture `tests/fixtures/gemma4_mm_trace.jsonl`, captured live on coreweave from
//! `google/gemma-4-E4B-it` (vLLM 16e91176, H200) through the recording tap with
//! `--record-tokens`. gemma-4 is a standard AUTOREGRESSIVE multimodal model (no
//! diffusion blocks), so the records carry plain per-token ITL (`itl_tokens`
//! absent) and ~271-281 prompt tokens (a few text tokens + 280 vision soft-tokens
//! per image).
//!
//! Booting the sim with `--replay-tokens` on the fixture must reproduce every
//! recorded `output_token_ids` byte-for-byte and the recorded finish reason.
//!
//! Why E4B and not the 26B: `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` emits
//! gibberish image output at every shippable vLLM rev (open bug
//! vllm-project/vllm#40106: its `use_bidirectional_attention="vision"` flag is
//! silently ignored, so vision tokens get causal attention). E4B does NOT carry
//! that flag and produces coherent image->text, so it is the canonical gemma-4
//! multimodal capture here. Either way replay fidelity is the same: the tap
//! records the engine's REAL token ids and timing and the sim replays them
//! exactly.

use std::time::Duration;

use clap::Parser as _;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

use inference_simulator_rs::trace::read_trace_file;
use inference_simulator_rs::{Opt, run};

const FIXTURE: &str = "tests/fixtures/gemma4_mm_trace.jsonl";
const TIMEOUT: Duration = Duration::from_secs(30);

#[tokio::test]
async fn gemma4_multimodal_capture_replays_byte_identical() {
    let (meta, records) = read_trace_file(std::path::Path::new(FIXTURE)).expect("read fixture");
    assert!(
        meta.model.as_deref() == Some("google/gemma-4-E4B-it"),
        "fixture must be the gemma-4 E4B capture"
    );
    assert!(!records.is_empty(), "fixture has no records");

    // Every fixture record is a real image request: a few text tokens plus 280
    // vision soft-tokens, and all carry recorded output token ids.
    for (i, r) in records.iter().enumerate() {
        assert!(
            r.prompt_tokens >= 256,
            "record {i} prompt_tokens {} is not multimodal (expected ~280 image soft-tokens)",
            r.prompt_tokens
        );
        assert!(
            r.output_token_ids.is_some(),
            "record {i} has no recorded tokens; capture must use --record-tokens"
        );
    }

    let addr = format!(
        "ipc:///tmp/inf-sim-gemma4-replay-{}.ipc",
        std::process::id()
    );
    let opt = Opt::parse_from([
        "inference-sim",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        FIXTURE,
    ]);
    let token = CancellationToken::new();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(opt, sim_token).await;
    });

    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(TIMEOUT, EngineCoreClient::connect(config))
        .await
        .expect("connect timed out")
        .expect("connect failed");

    for (i, record) in records.iter().enumerate() {
        let expected = record
            .output_token_ids
            .as_deref()
            .expect("record has tokens");
        let req = EngineCoreRequest {
            request_id: format!("gemma4-replay-{i}"),
            // The sim replays by record index; prompt length mirrors the capture.
            prompt_token_ids: Some(vec![42u32; record.prompt_tokens]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: expected.len() as u32,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        };
        let stream = client.call(req).await.expect("call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("stream timed out");

        let tokens: Vec<u32> = outputs
            .iter()
            .flat_map(|r| r.as_ref().expect("stream error").new_token_ids.clone())
            .collect();
        assert_eq!(
            tokens, expected,
            "record {i}: replayed stream must match the captured output_token_ids byte-for-byte"
        );

        let last = outputs.last().unwrap().as_ref().unwrap();
        let expected_finish = record
            .finish_reason
            .map(inference_simulator_rs::wire::engine_finish_reason)
            .unwrap_or(EngineCoreFinishReason::Length);
        assert_eq!(
            last.finish_reason,
            Some(expected_finish),
            "record {i}: finish reason must match the capture"
        );
    }

    token.cancel();
}
