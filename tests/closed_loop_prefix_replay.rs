//! Closed-loop replay through the full ZMQ path with `--replay-match prefix`.
//!
//! This is the offline-agent scenario in miniature: a "live" client whose
//! turn N+1 prompt is built from the response it got for turn N, exactly like
//! an agent loop re-run against the sim. The client knows nothing about
//! replay indices; it uses its own request ids and a huge max_tokens. The sim
//! must match each incoming prompt to the captured record by block-hash
//! prefix and serve the recorded stream, which in turn makes the client's
//! next prompt identical to the captured one, and the loop closes.

use std::time::Duration;

use clap::Parser as _;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

use inference_simulator_rs::trace::{
    TraceFinishReason, TraceMeta, TraceRecord, prompt_block_hashes, write_trace,
};
use inference_simulator_rs::{Opt, run};

const TIMEOUT: Duration = Duration::from_secs(30);
const BLOCK_SIZE: usize = 16;

/// Deterministic stand-ins for tokenized text. The values don't matter, only
/// that turn N+1's prompt extends turn N's prompt + response.
fn user_tokens(turn: usize, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| 1000 * (turn as u32 + 1) + i)
        .collect()
}

#[tokio::test]
async fn agent_loop_replays_offline_via_prefix_matching() {
    // === Synthesize the "capture": a 3-turn agent conversation ===
    let system: Vec<u32> = (0..32).collect();
    let responses: Vec<Vec<u32>> = vec![
        (500..520).collect(),
        (600..650).collect(),
        (700..707).collect(),
    ];

    let mut prompt = system.clone();
    let mut records = Vec::new();
    let mut prompts = Vec::new();
    for (turn, response) in responses.iter().enumerate() {
        prompt.extend(user_tokens(turn, 24));
        prompts.push(prompt.clone());
        records.push(TraceRecord {
            prompt_tokens: prompt.len(),
            output_tokens: response.len(),
            ttft_ms: 1.0,
            itl_ms: Some(vec![1.0; response.len() - 1]),
            arrival_ms: Some(turn as f64 * 100.0),
            block_hashes: prompt_block_hashes(&prompt, BLOCK_SIZE),
            output_token_ids: Some(response.clone()),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        });
        prompt.extend(response);
    }

    let meta = TraceMeta {
        block_size: Some(BLOCK_SIZE),
        ..Default::default()
    };
    let trace_path =
        std::env::temp_dir().join(format!("closed-loop-prefix-{}.jsonl", std::process::id()));
    let mut file = std::fs::File::create(&trace_path).expect("create trace");
    write_trace(&mut file, &meta, &records).expect("write trace");
    drop(file);

    // === Boot the sim in prefix-match mode ===
    let addr = format!("ipc:///tmp/inf-sim-closed-loop-{}.ipc", std::process::id());
    let trace_arg = trace_path.to_str().unwrap().to_string();
    let args = vec![
        "inference-sim",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        &trace_arg,
        "--replay-match",
        "prefix",
    ];
    let opt = Opt::parse_from(&args);
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

    // === The agent loop: each turn's prompt is built from live responses ===
    let mut live_prompt = system.clone();
    for (turn, expected) in responses.iter().enumerate() {
        live_prompt.extend(user_tokens(turn, 24));
        assert_eq!(
            live_prompt, prompts[turn],
            "turn {turn}: the loop diverged from the capture"
        );
        let req = EngineCoreRequest {
            // A live client's id carries no replay index.
            request_id: format!("chatcmpl-live-{turn}"),
            prompt_token_ids: Some(live_prompt.clone()),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 4096,
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
            &tokens, expected,
            "turn {turn}: response must be byte-identical to the capture"
        );
        assert_eq!(
            outputs.last().unwrap().as_ref().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Stop),
            "turn {turn}: stream must end with the recorded finish reason"
        );
        // The agent appends the response and goes around again.
        live_prompt.extend(&tokens);
    }

    token.cancel();
    let _ = std::fs::remove_file(&trace_path);
}
