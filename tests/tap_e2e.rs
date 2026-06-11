//! End-to-end integration test for the engine-core recording tap.
//!
//! Topology (no GPUs, no real vLLM):
//!
//! ```text
//!   EngineCoreClient (acts as "real frontend")
//!       |
//!       v
//!   inference-sim-tap  (the binary under test)
//!       |
//!       v
//!   inference_simulator_rs::run (acts as "real engine")
//! ```
//!
//! The test drives 2-3 normal requests plus one aborted request through the
//! full chain and asserts:
//!   - Tokens arrive at the client with the correct count and Length finish.
//!   - The trace file contains one record per finished request.
//!   - Each record has correct prompt_tokens, output_tokens, ttft_ms > 0,
//!     and itl_ms count == output_tokens - 1 (chunk size 1).
//!   - The aborted request does NOT appear in the trace.

use std::io::BufReader;
use std::process::Command;
use std::time::Duration;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use inference_simulator_rs::trace::{TraceFinishReason, TraceRecord, read_trace};
use inference_simulator_rs::{Opt, run};

use clap::Parser as _;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

const TIMEOUT: Duration = Duration::from_secs(15);

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

/// Spin up the simulator as the "real engine" that the tap will connect to.
/// Returns a guard that cancels on drop.
///
/// The max-model-len/kv-cache-size values are distinct from the mock-engine
/// defaults so the test can prove the tap relays the engine's real ready
/// response downstream instead of fabricating one.
async fn start_sim(handshake_address: &str) -> CancellationToken {
    let args = vec![
        "inference-sim",
        "--handshake-address",
        handshake_address,
        "--inter-token-latency",
        "30",
        "--max-model-len",
        "16384",
        "--kv-cache-size",
        "2048",
    ];
    let opt = Opt::parse_from(&args);
    let token = CancellationToken::new();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = run(opt, sim_token).await;
    });
    token
}

#[tokio::test]
async fn tap_records_trace() {
    let pid = std::process::id();

    // Unique IPC endpoints scoped by pid.
    let frontend_handshake = format!("ipc:///tmp/tap-e2e-frontend-{pid}.ipc");
    let tap_engine_handshake = format!("ipc:///tmp/tap-e2e-engine-{pid}.ipc");
    let tap_input = format!("ipc:///tmp/tap-e2e-input-{pid}.ipc");
    let tap_output = format!("ipc:///tmp/tap-e2e-output-{pid}.ipc");
    let trace_path = format!("/tmp/tap-e2e-trace-{pid}.jsonl");

    // Clean up leftover IPC files from prior runs.
    for path in [
        &frontend_handshake,
        &tap_engine_handshake,
        &tap_input,
        &tap_output,
    ] {
        if let Some(p) = path.strip_prefix("ipc://") {
            let _ = std::fs::remove_file(p);
        }
    }
    let _ = std::fs::remove_file(&trace_path);

    // Step 1: Start the tap binary. It will bind the upstream side first, then
    // connect downstream. The tap binary path comes from CARGO_BIN_EXE.
    let tap_bin = env!("CARGO_BIN_EXE_inference-sim-tap");
    let mut tap_child = Command::new(tap_bin)
        .arg("--frontend-handshake")
        .arg(&frontend_handshake)
        .arg("--engine-handshake")
        .arg(&tap_engine_handshake)
        .arg("--input-address")
        .arg(&tap_input)
        .arg("--output-address")
        .arg(&tap_output)
        .arg("--trace-out")
        .arg(&trace_path)
        .arg("--model")
        .arg("test-model")
        .arg("--record-tokens")
        .env("RUST_LOG", "info")
        .spawn()
        .expect("failed to spawn inference-sim-tap");

    // Give the tap a moment to bind its upstream sockets.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Step 2: Start the simulator as the "real engine", connecting to the tap's
    // engine handshake address.
    let sim_token = start_sim(&tap_engine_handshake).await;

    // Give the sim a moment to connect to the tap.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Step 3: Connect the EngineCoreClient to the tap's frontend-facing side.
    // The tap connects to this address as an engine.
    let config = EngineCoreClientConfig::new_single(&frontend_handshake);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .expect("client connect timed out")
        .expect("client connect failed");

    // The tap must relay the engine's real registration ready response, not
    // the mock-engine defaults (1Mi max_model_len / default block count).
    assert_eq!(
        client.max_model_len(),
        16384,
        "frontend should see the engine's real max_model_len through the tap"
    );
    assert_eq!(
        client.total_num_gpu_blocks(),
        2048,
        "frontend should see the engine's real num_gpu_blocks through the tap"
    );

    // Step 4: Send requests through the chain.

    // Request 1: 10 prompt tokens, 5 output tokens.
    let req1 = make_request("tap-req-1", 10, 5);
    let stream1 = client.call(req1).await.expect("call 1 failed");
    let outputs1: Vec<_> = tokio::time::timeout(TIMEOUT, stream1.collect::<Vec<_>>())
        .await
        .expect("stream 1 timed out");
    let tokens1: Vec<u32> = outputs1
        .iter()
        .flat_map(|r| r.as_ref().expect("output error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens1.len(), 5, "request 1 should produce 5 tokens");
    let last1 = outputs1.last().unwrap().as_ref().unwrap();
    assert_eq!(last1.finish_reason, Some(EngineCoreFinishReason::Length));

    // Request 2: 20 prompt tokens, 3 output tokens.
    let req2 = make_request("tap-req-2", 20, 3);
    let stream2 = client.call(req2).await.expect("call 2 failed");
    let outputs2: Vec<_> = tokio::time::timeout(TIMEOUT, stream2.collect::<Vec<_>>())
        .await
        .expect("stream 2 timed out");
    let tokens2: Vec<u32> = outputs2
        .iter()
        .flat_map(|r| r.as_ref().expect("output error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens2.len(), 3, "request 2 should produce 3 tokens");
    let last2 = outputs2.last().unwrap().as_ref().unwrap();
    assert_eq!(last2.finish_reason, Some(EngineCoreFinishReason::Length));

    // Request 3 (aborted): large max_tokens, abort after first output.
    let req3 = make_request("tap-req-abort", 4, 10000);
    let mut stream3 = client.call(req3).await.expect("call 3 failed");
    let first3 = tokio::time::timeout(TIMEOUT, stream3.next())
        .await
        .expect("first output timed out")
        .expect("stream ended")
        .expect("output error");
    assert_eq!(first3.request_id, "tap-req-abort");
    client
        .abort(&["tap-req-abort".to_string()])
        .await
        .expect("abort failed");
    // Drain remaining outputs.
    let _remaining: Vec<_> = tokio::time::timeout(TIMEOUT, stream3.collect::<Vec<_>>())
        .await
        .expect("remaining stream timed out");

    // Request 4: one more to make sure the tap is still working after an abort.
    let req4 = make_request("tap-req-3", 15, 4);
    let stream4 = client.call(req4).await.expect("call 4 failed");
    let outputs4: Vec<_> = tokio::time::timeout(TIMEOUT, stream4.collect::<Vec<_>>())
        .await
        .expect("stream 4 timed out");
    let tokens4: Vec<u32> = outputs4
        .iter()
        .flat_map(|r| r.as_ref().expect("output error").new_token_ids.clone())
        .collect();
    assert_eq!(tokens4.len(), 4, "request 3 should produce 4 tokens");

    // Step 5: Shut down the simulator and tap.
    sim_token.cancel();
    // Give the sim a moment to shut down so the tap sees the connection drop.
    tokio::time::sleep(Duration::from_millis(200)).await;
    // Kill the tap process.
    let _ = tap_child.kill();
    let _ = tap_child.wait();

    // Step 6: Read and validate the trace file.
    let file = std::fs::File::open(&trace_path).expect("trace file should exist");
    let reader = BufReader::new(file);
    let (meta, records) = read_trace(reader).expect("trace should be valid JSONL");

    // Meta line.
    assert_eq!(meta.source.as_deref(), Some("tap"));
    assert_eq!(meta.model.as_deref(), Some("test-model"));

    // Should have exactly 3 records (the abort should NOT be present).
    assert_eq!(
        records.len(),
        3,
        "expected 3 trace records (no aborted request), got {}",
        records.len()
    );

    // Validate each record.
    let record_by_prompt: std::collections::HashMap<usize, &TraceRecord> =
        records.iter().map(|r| (r.prompt_tokens, r)).collect();

    // Request 1: 10 prompt, 5 output.
    let r1 = record_by_prompt
        .get(&10)
        .expect("should have record with 10 prompt tokens");
    assert_eq!(r1.output_tokens, 5);
    assert!(
        r1.ttft_ms > 0.0,
        "ttft_ms should be > 0, got {}",
        r1.ttft_ms
    );
    let itl1 = r1.itl_ms.as_ref().expect("should have itl_ms");
    assert_eq!(
        itl1.len(),
        4,
        "itl_ms count should be output_tokens - 1 = 4, got {}",
        itl1.len()
    );
    assert!(r1.concurrency >= 1, "concurrency should be >= 1");

    // --record-tokens: the trace carries exactly the token ids the client saw,
    // plus the finish reason.
    assert_eq!(
        r1.output_token_ids.as_deref(),
        Some(tokens1.as_slice()),
        "recorded ids must be content-identical to the delivered stream"
    );
    assert_eq!(r1.finish_reason, Some(TraceFinishReason::Length));
    assert!(
        records.iter().all(
            |r| r.output_token_ids.as_ref().map(Vec::len) == Some(r.output_tokens)
                && r.finish_reason == Some(TraceFinishReason::Length)
        ),
        "every record carries ids matching output_tokens and a finish reason"
    );

    // Per-gap batch context parallels the gap array and carries a sane
    // running count for every step.
    let ctx1 = r1.itl_ctx.as_ref().expect("should have itl_ctx");
    assert_eq!(ctx1.num_running.len(), itl1.len());
    assert_eq!(ctx1.prefill_tokens.len(), itl1.len());
    assert!(
        ctx1.num_running.iter().all(|&n| n >= 1),
        "running count should be >= 1 while this request decodes, got {:?}",
        ctx1.num_running
    );

    // Arrival times are recorded relative to capture start and must respect the
    // order the requests were sent in (10-prompt, then 20-prompt, then 15-prompt).
    let arrivals: Vec<f64> = [10usize, 20, 15]
        .iter()
        .map(|p| {
            record_by_prompt[p]
                .arrival_ms
                .expect("tap records should carry arrival_ms")
        })
        .collect();
    assert!(
        arrivals[0] >= 0.0 && arrivals.windows(2).all(|w| w[0] <= w[1]),
        "arrival_ms should be non-negative and ordered by send time, got {arrivals:?}"
    );

    // Request 2: 20 prompt, 3 output.
    let r2 = record_by_prompt
        .get(&20)
        .expect("should have record with 20 prompt tokens");
    assert_eq!(r2.output_tokens, 3);
    assert!(r2.ttft_ms > 0.0);
    let itl2 = r2.itl_ms.as_ref().expect("should have itl_ms");
    assert_eq!(
        itl2.len(),
        2,
        "itl_ms count should be 2, got {}",
        itl2.len()
    );

    // Request 3: 15 prompt, 4 output.
    let r3 = record_by_prompt
        .get(&15)
        .expect("should have record with 15 prompt tokens");
    assert_eq!(r3.output_tokens, 4);
    assert!(r3.ttft_ms > 0.0);
    let itl3 = r3.itl_ms.as_ref().expect("should have itl_ms");
    assert_eq!(
        itl3.len(),
        3,
        "itl_ms count should be 3, got {}",
        itl3.len()
    );

    // Step 7: full record -> replay round trip. Boot a FRESH sim on the trace
    // the tap just captured (fast mode: no latency config) and verify each
    // replayed stream is byte-identical to what the live client saw.
    let replay_addr = format!("ipc:///tmp/tap-e2e-replay-{pid}.ipc");
    if let Some(p) = replay_addr.strip_prefix("ipc://") {
        let _ = std::fs::remove_file(p);
    }
    let replay_args = vec![
        "inference-sim",
        "--handshake-address",
        &replay_addr,
        "--replay-tokens",
        &trace_path,
    ];
    let replay_opt = Opt::parse_from(&replay_args);
    let replay_guard = CancellationToken::new();
    let replay_sim_token = replay_guard.clone();
    tokio::spawn(async move {
        let _ = run(replay_opt, replay_sim_token).await;
    });
    let config = EngineCoreClientConfig::new_single(&replay_addr);
    let replay_client =
        tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
            .await
            .expect("replay client connect timed out")
            .expect("replay client connect failed");

    // Replay indices follow arrival order, which is the send order above.
    let captured: Vec<(usize, &Vec<u32>)> = vec![(10, &tokens1), (20, &tokens2), (15, &tokens4)];
    for (i, (prompt_len, expected)) in captured.into_iter().enumerate() {
        let req = make_request(&format!("replay-{i}"), prompt_len, expected.len() as u32);
        let stream = replay_client.call(req).await.expect("replay call failed");
        let outputs: Vec<_> = tokio::time::timeout(TIMEOUT, stream.collect::<Vec<_>>())
            .await
            .expect("replay stream timed out");
        let tokens: Vec<u32> = outputs
            .iter()
            .flat_map(|r| {
                r.as_ref()
                    .expect("replay output error")
                    .new_token_ids
                    .clone()
            })
            .collect();
        assert_eq!(
            &tokens, expected,
            "replay-{i} must reproduce the captured stream byte for byte"
        );
        assert_eq!(
            outputs.last().unwrap().as_ref().unwrap().finish_reason,
            Some(EngineCoreFinishReason::Length),
            "replay-{i} must end with the recorded finish reason"
        );
    }
    replay_guard.cancel();

    // Clean up.
    let _ = std::fs::remove_file(&trace_path);
    if let Some(p) = replay_addr.strip_prefix("ipc://") {
        let _ = std::fs::remove_file(p);
    }
    for path in [
        &frontend_handshake,
        &tap_engine_handshake,
        &tap_input,
        &tap_output,
    ] {
        if let Some(p) = path.strip_prefix("ipc://") {
            let _ = std::fs::remove_file(p);
        }
    }
}
