//! Replay verification against a real captured trace.
//!
//! Skipped unless `REPLAY_TRACE` points at a token-recording trace (tap
//! `--record-tokens`). Boots the sim with `--replay-tokens` on that trace and
//! asserts every replayed stream is byte-identical to the recorded one:
//!
//! ```bash
//! REPLAY_TRACE=h200-tokens-tap.jsonl cargo test --test real_trace_replay -- --nocapture
//! ```

use std::time::Duration;

use clap::Parser as _;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

use inference_simulator_rs::trace::{read_trace_file, replay_subset};
use inference_simulator_rs::{Opt, run};

const TIMEOUT: Duration = Duration::from_secs(30);

#[tokio::test]
async fn replays_real_trace_byte_identical() {
    let Ok(trace_path) = std::env::var("REPLAY_TRACE") else {
        eprintln!("REPLAY_TRACE not set; skipping real-trace replay verification");
        return;
    };

    let (_, records) = read_trace_file(std::path::Path::new(&trace_path)).expect("read trace");
    let subset = replay_subset(records);
    let with_tokens = subset
        .iter()
        .filter(|r| r.output_token_ids.is_some())
        .count();
    assert!(with_tokens > 0, "trace has no output_token_ids");
    eprintln!(
        "replaying {} records ({} with tokens) from {trace_path}",
        subset.len(),
        with_tokens
    );

    let addr = format!("ipc:///tmp/inf-sim-real-replay-{}.ipc", std::process::id());
    let args = vec![
        "inference-sim",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        &trace_path,
    ];
    let opt = Opt::parse_from(&args);
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

    for (i, record) in subset.iter().enumerate() {
        let Some(expected) = record.output_token_ids.as_deref() else {
            continue;
        };
        let req = EngineCoreRequest {
            request_id: format!("replay-{i}"),
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
            "replay-{i}: stream must be byte-identical to the capture"
        );
        let last = outputs.last().unwrap().as_ref().unwrap();
        let expected_finish = record
            .finish_reason
            .map(EngineCoreFinishReason::from)
            .unwrap_or(EngineCoreFinishReason::Length);
        assert_eq!(
            last.finish_reason,
            Some(expected_finish),
            "replay-{i}: finish reason must match the capture"
        );
    }
    eprintln!("all {with_tokens} token streams byte-identical, finish reasons match");
    token.cancel();
}
