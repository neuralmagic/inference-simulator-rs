//! End-to-end integration tests for the simulator over real ZMQ transport.
//!
//! Each test spins up the full `inference_simulator_rs::run` task (the real engine loop,
//! IO loop, and ZMQ handshake) and connects a real `EngineCoreClient` from the
//! `vllm-engine-core-client` dependency. No mocks, no stubs: this is the only coverage
//! of the engine-core protocol path end to end, including the two-phase async KV-pull
//! admission that landed in `src/engine.rs`.
//!
//! The contract under test:
//!   - The ZMQ handshake completes and the client can submit requests.
//!   - Token streams deliver the correct count of tokens with the right finish reason.
//!   - Abort, prefix-cache reset, and LoRA lifecycle utilities round-trip correctly.
//!   - The P/D handoff (prefill advertise, then decode pull) exercises the real engine
//!     loop select branch for two-phase admission.

use std::collections::HashMap;
use std::time::Duration;

use clap::Parser as _;
use futures::StreamExt;
use inference_simulator_rs::{Opt, run};
use serde_json::json;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

const TIMEOUT: Duration = Duration::from_secs(10);

/// RAII guard that cancels the simulator task on drop, even if the test panics.
struct SimGuard {
    token: CancellationToken,
}

impl Drop for SimGuard {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

/// Spin up the simulator with the given CLI flags and connect a real client.
/// Returns `(client, guard)`. The guard cancels the sim on drop.
async fn harness(name: &str, extra_flags: &[&str]) -> (EngineCoreClient, SimGuard) {
    let addr = format!("ipc:///tmp/inf-sim-e2e-{}-{name}.ipc", std::process::id());

    let mut args: Vec<&str> = vec!["inference-sim", "--handshake-address", &addr];
    args.extend_from_slice(extra_flags);
    let opt = Opt::parse_from(&args);

    let token = CancellationToken::new();
    let guard = SimGuard {
        token: token.clone(),
    };

    // Spawn the simulator task.
    let sim_opt = opt.clone();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(sim_opt, sim_token).await;
    });

    // Connect the real client. The client binds the handshake address and waits
    // for the engine to dial in.
    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .expect("client connect timed out")
        .expect("client connect failed");

    (client, guard)
}

/// Build a request with a prompt of `prompt_len` identical tokens and the given max_tokens.
fn make_request(id: &str, prompt_len: usize, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.to_string(),
        prompt_token_ids: Some(vec![42u32; prompt_len]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    }
}

/// Build a request whose prompt tokens are all set to `token_id` (for cache isolation).
fn make_request_with_token(
    id: &str,
    prompt_len: usize,
    max_tokens: u32,
    token_id: u32,
) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.to_string(),
        prompt_token_ids: Some(vec![token_id; prompt_len]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn token_stream_round_trip() {
    let (client, _guard) = harness("token_stream_round_trip", &[]).await;

    let req = make_request("rt-1", 8, 5);
    let stream = client.call(req).await.expect("call failed");

    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");

    let total_tokens: usize = outputs
        .iter()
        .map(|r| {
            let o = r.as_ref().expect("stream item error");
            assert_eq!(o.request_id, "rt-1");
            o.new_token_ids.len()
        })
        .sum();

    assert_eq!(total_tokens, 5, "expected exactly 5 output tokens");

    let last = outputs.last().unwrap().as_ref().unwrap();
    assert_eq!(
        last.finish_reason,
        Some(EngineCoreFinishReason::Length),
        "final output should have finish_reason Length"
    );
}

#[tokio::test]
async fn replay_tokens_serves_recorded_stream() {
    use inference_simulator_rs::trace::{
        TraceFinishReason, TraceMeta, TraceRecord, TraceWriter, write_trace,
    };

    // A two-record trace, written out of arrival order to prove the sim maps
    // replay indices through the canonical arrival ordering.
    let recorded_late: Vec<u32> = vec![900, 901, 902];
    let recorded_early: Vec<u32> = vec![800, 801, 802, 803];
    let records = vec![
        TraceRecord {
            prompt_tokens: 8,
            output_tokens: recorded_late.len(),
            ttft_ms: 1.0,
            itl_ms: Some(vec![1.0; recorded_late.len() - 1]),
            arrival_ms: Some(100.0),
            output_token_ids: Some(recorded_late.clone()),
            finish_reason: Some(TraceFinishReason::Length),
            ..Default::default()
        },
        TraceRecord {
            prompt_tokens: 8,
            output_tokens: recorded_early.len(),
            ttft_ms: 1.0,
            itl_ms: Some(vec![1.0; recorded_early.len() - 1]),
            arrival_ms: Some(0.0),
            output_token_ids: Some(recorded_early.clone()),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        },
    ];
    // The .gz extension also covers the compressed read path end to end.
    let trace_path = format!(
        "/tmp/inf-sim-e2e-replay-tokens-{}.jsonl.gz",
        std::process::id()
    );
    let mut writer =
        TraceWriter::create(std::path::Path::new(&trace_path)).expect("create trace file");
    write_trace(&mut writer, &TraceMeta::default(), &records).expect("write trace");
    writer.finish().expect("finish trace file");

    let (client, _guard) = harness("replay_tokens", &["--replay-tokens", &trace_path]).await;

    // replay-0 = the arrival_ms 0.0 record (file order is finish order).
    let req = make_request("replay-0", 8, recorded_early.len() as u32);
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");
    let tokens: Vec<u32> = outputs
        .iter()
        .flat_map(|r| r.as_ref().expect("stream item error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens, recorded_early, "replay-0 serves the early arrival");
    assert_eq!(
        outputs.last().unwrap().as_ref().unwrap().finish_reason,
        Some(EngineCoreFinishReason::Stop),
        "stream ends with the recorded finish reason"
    );

    let req = make_request("replay-1", 8, recorded_late.len() as u32);
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");
    let tokens: Vec<u32> = outputs
        .iter()
        .flat_map(|r| r.as_ref().expect("stream item error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens, recorded_late, "replay-1 serves the late arrival");

    // An unmatched request still streams (random fallback).
    let req = make_request("adhoc", 8, 2);
    let stream = client.call(req).await.expect("call failed");
    let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("stream collect timed out");
    let total: usize = outputs
        .iter()
        .map(|r| r.as_ref().expect("stream item error").new_token_ids.len())
        .sum();
    assert_eq!(total, 2, "unmatched request falls back to random tokens");

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn abort_terminates_stream() {
    let (client, _guard) = harness(
        "abort_terminates_stream",
        &["--inter-token-latency", "200", "--max-num-seqs", "1"],
    )
    .await;

    let req = make_request("abort-1", 4, 10000);
    let mut stream = client.call(req).await.expect("call failed");

    // Take the first output to prove it started generating.
    let first = tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("first output timed out")
        .expect("stream ended unexpectedly")
        .expect("first output error");
    assert_eq!(first.request_id, "abort-1");

    // Abort the request.
    client
        .abort(&["abort-1".to_string()])
        .await
        .expect("abort failed");

    // Collect remaining outputs; one of them should carry finish_reason Abort.
    let remaining: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("remaining stream timed out");

    let got_abort = remaining.iter().any(|r| {
        r.as_ref()
            .map(|o| o.finish_reason == Some(EngineCoreFinishReason::Abort))
            .unwrap_or(false)
    });

    assert!(
        got_abort,
        "expected an output with finish_reason Abort after calling abort"
    );
}

#[tokio::test]
async fn graceful_shutdown_drains_in_flight() {
    let (client, guard) = harness(
        "graceful_shutdown_drains_in_flight",
        &["--shutdown-timeout", "10", "--inter-token-latency", "50"],
    )
    .await;

    let req = make_request("drain-1", 4, 10);
    let mut stream = client.call(req).await.expect("call failed");

    // Wait for the first output so the request is in flight, then request shutdown.
    let first = tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("first output timed out")
        .expect("stream ended unexpectedly")
        .expect("first output error");
    assert_eq!(first.request_id, "drain-1");
    guard.token.cancel();

    // A request arriving during the drain is rejected with an immediate Abort.
    let late = make_request("late-1", 4, 5);
    let late_stream = client.call(late).await.expect("late call failed");
    let late_outputs: Vec<_> = tokio::time::timeout(TIMEOUT, late_stream.collect::<Vec<_>>())
        .await
        .expect("late stream timed out");
    let late_final = late_outputs
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .find(|o| o.finish_reason.is_some())
        .expect("late request should get a terminal output");
    assert_eq!(
        late_final.finish_reason,
        Some(EngineCoreFinishReason::Abort),
        "a request arriving during drain is rejected with Abort"
    );

    // The in-flight request still completes naturally, through the real ZMQ path,
    // even though shutdown was requested mid-stream.
    let remaining: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("drained stream timed out");
    let total_tokens = first.new_token_ids.len()
        + remaining
            .iter()
            .map(|r| r.as_ref().expect("stream item error").new_token_ids.len())
            .sum::<usize>();
    assert_eq!(total_tokens, 10, "drain delivers the full token stream");
    assert_eq!(
        remaining.last().unwrap().as_ref().unwrap().finish_reason,
        Some(EngineCoreFinishReason::Length),
        "drained request finishes with its natural reason"
    );
}

#[tokio::test]
async fn shutdown_default_aborts_in_flight() {
    // Default --shutdown-timeout 0 (vLLM's default): shutdown aborts immediately.
    let (client, guard) = harness(
        "shutdown_default_aborts_in_flight",
        &["--inter-token-latency", "200"],
    )
    .await;

    let req = make_request("abort-on-shutdown-1", 4, 10000);
    let mut stream = client.call(req).await.expect("call failed");

    let first = tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .expect("first output timed out")
        .expect("stream ended unexpectedly")
        .expect("first output error");
    assert_eq!(first.request_id, "abort-on-shutdown-1");
    guard.token.cancel();

    let remaining: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
        .await
        .expect("remaining stream timed out");
    let got_abort = remaining.iter().any(|r| {
        r.as_ref()
            .map(|o| o.finish_reason == Some(EngineCoreFinishReason::Abort))
            .unwrap_or(false)
    });
    assert!(
        got_abort,
        "shutdown with timeout 0 must abort the in-flight request"
    );
}

#[tokio::test]
async fn reset_prefix_cache_busy_vs_idle() {
    let (client, _guard) = harness(
        "reset_prefix_cache_busy_vs_idle",
        &["--time-to-first-token", "10000"],
    )
    .await;

    // Submit a request that will park in prefill for 10 seconds (never emits a token
    // during this test).
    let req = make_request("rpc-1", 4, 5);
    let _stream = client.call(req).await.expect("call failed");

    // Give the engine a moment to admit the request into the batch.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // reset_prefix_cache should fail (return false) because the request holds pinned blocks.
    let busy_result = tokio::time::timeout(TIMEOUT, client.reset_prefix_cache(false, false))
        .await
        .expect("reset timed out")
        .expect("reset call failed");
    assert!(
        !busy_result,
        "reset_prefix_cache should return false while a request is running"
    );

    // Abort the request so the engine is idle.
    client
        .abort(&["rpc-1".to_string()])
        .await
        .expect("abort failed");

    // Drain the abort output so the engine fully processes it.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Now reset should succeed.
    let idle_result = tokio::time::timeout(TIMEOUT, client.reset_prefix_cache(false, false))
        .await
        .expect("idle reset timed out")
        .expect("idle reset call failed");
    assert!(
        idle_result,
        "reset_prefix_cache should return true when the engine is idle"
    );
}

#[tokio::test]
async fn lora_load_unload_lifecycle() {
    let (client, _guard) = harness("lora_load_unload_lifecycle", &[]).await;

    let lora = LoraRequest::new(
        "test-adapter".to_string(),
        42,
        "/fake/path".to_string(),
        false,
        false,
    );

    // add_lora -> true
    let added = tokio::time::timeout(TIMEOUT, client.add_lora(&lora))
        .await
        .expect("add_lora timed out")
        .expect("add_lora failed");
    assert!(added, "add_lora should return true");

    // remove_lora(42) -> true (it was loaded)
    let removed = tokio::time::timeout(TIMEOUT, client.remove_lora(42))
        .await
        .expect("remove_lora timed out")
        .expect("remove_lora failed");
    assert!(removed, "first remove_lora should return true");

    // remove_lora(42) again -> false (already gone)
    let removed_again = tokio::time::timeout(TIMEOUT, client.remove_lora(42))
        .await
        .expect("remove_lora again timed out")
        .expect("remove_lora again failed");
    assert!(
        !removed_again,
        "second remove_lora should return false (adapter already removed)"
    );
}

#[tokio::test]
async fn pd_handoff_advertise_then_pull() {
    let (client, _guard) = harness(
        "pd_handoff_advertise_then_pull",
        &["--tokens-per-block", "16"],
    )
    .await;

    // Request A: prefill-side. 32 prompt tokens with default 16 tokens/block = 2 blocks.
    // Carries do_remote_decode so the engine advertises its KV on completion.
    let mut req_a = make_request_with_token("pd-a", 32, 5, 100);
    let mut extra_a = HashMap::new();
    extra_a.insert(
        "kv_transfer_params".to_string(),
        json!({
            "do_remote_decode": true,
            "do_remote_prefill": false
        }),
    );
    req_a.sampling_params.as_mut().unwrap().extra_args = Some(extra_a);

    let stream_a = client.call(req_a).await.expect("call A failed");
    let outputs_a: Vec<_> = tokio::time::timeout(TIMEOUT, stream_a.collect::<Vec<_>>())
        .await
        .expect("stream A timed out");

    // Find the final output (the one with finish_reason).
    let final_a = outputs_a
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .find(|o| o.finish_reason.is_some())
        .expect("no final output for request A");
    assert_eq!(final_a.finish_reason, Some(EngineCoreFinishReason::Length));

    // The final output must carry kv_transfer_params with the prefill descriptor.
    let kv_params_a = final_a
        .kv_transfer_params
        .as_ref()
        .expect("final output of a do_remote_decode request must carry kv_transfer_params");

    assert_eq!(
        kv_params_a
            .get("do_remote_prefill")
            .and_then(|v| v.as_bool()),
        Some(true),
        "kv_transfer_params.do_remote_prefill should be true"
    );

    let remote_block_ids = kv_params_a
        .get("remote_block_ids")
        .and_then(|v| v.as_array())
        .expect("kv_transfer_params must have remote_block_ids");
    assert!(
        !remote_block_ids.is_empty(),
        "remote_block_ids should be non-empty for a 32-token prompt with 16 tokens/block"
    );

    let remote_engine_id = kv_params_a
        .get("remote_engine_id")
        .and_then(|v| v.as_str())
        .expect("kv_transfer_params must have remote_engine_id");
    let remote_host = kv_params_a
        .get("remote_host")
        .and_then(|v| v.as_str())
        .expect("kv_transfer_params must have remote_host");
    let remote_port = kv_params_a
        .get("remote_port")
        .and_then(|v| v.as_u64())
        .expect("kv_transfer_params must have remote_port");
    let remote_request_id = kv_params_a
        .get("remote_request_id")
        .and_then(|v| v.as_str())
        .expect("kv_transfer_params must have remote_request_id");

    // Request B: decode-side pull. Uses a DIFFERENT prompt (token_id 200 instead of 100)
    // so the local prefix cache has zero hits, making the full prompt external-cached.
    // This exercises the two-phase pull admission in the real engine loop.
    let mut req_b = make_request_with_token("pd-b", 32, 3, 200);
    let mut extra_b = HashMap::new();
    extra_b.insert(
        "kv_transfer_params".to_string(),
        json!({
            "do_remote_prefill": true,
            "do_remote_decode": false,
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": remote_engine_id,
            "remote_host": remote_host,
            "remote_port": remote_port,
            "remote_request_id": remote_request_id
        }),
    );
    req_b.sampling_params.as_mut().unwrap().extra_args = Some(extra_b);

    let stream_b = client.call(req_b).await.expect("call B failed");
    let outputs_b: Vec<_> = tokio::time::timeout(TIMEOUT, stream_b.collect::<Vec<_>>())
        .await
        .expect("stream B timed out");

    // B should complete with Length.
    let final_b = outputs_b
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .find(|o| o.finish_reason.is_some())
        .expect("no final output for request B");
    assert_eq!(
        final_b.finish_reason,
        Some(EngineCoreFinishReason::Length),
        "request B should complete with Length"
    );

    // The first output of B should carry prefill_stats with external cached tokens.
    // With a different prompt (no local cache hit), all 32 tokens come from external.
    let first_b = outputs_b
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .find(|o| o.prefill_stats.is_some())
        .expect("request B should have prefill_stats on its first output");
    let stats_b = first_b.prefill_stats.as_ref().unwrap();

    assert_eq!(stats_b.num_prompt_tokens, 32, "B has 32 prompt tokens");
    // With a different prompt token, there are no local cache hits.
    // num_external_cached_tokens = prompt - local_cached = 32 - 0 = 32
    assert_eq!(
        stats_b.num_external_cached_tokens, 32,
        "all prompt tokens should be externally cached (different prompt, zero local hits)"
    );
    assert_eq!(
        stats_b.num_computed_tokens, 0,
        "no tokens should be locally computed when fully covered by external cache"
    );
}
