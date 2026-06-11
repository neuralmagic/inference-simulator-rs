//! Integration tests for the calibration harness.
//!
//! These tests exercise the full calibration pipeline as library calls, proving
//! the two core claims:
//!   1. TraceLatency replay reproduces source trace quantiles within tolerance.
//!   2. KnobLatency structurally cannot reproduce heavy tails (capped at ~1.7x).

use std::time::Duration;

use inference_simulator_rs::calibrate;

// Test (a): gen-demo + model-level calibrate
//
// This single test IS the theorem made executable:
//   - Generate a heavy-tailed demo trace.
//   - Run model-level calibration.
//   - Assert replay passes tolerance.
//   - Assert the source trace has a heavy tail (p99/p50 >= 3.0).
//   - Assert the knob-fit tail is capped (<= 1.75).

#[test]
fn calibrate_model_level_proves_both_claims() {
    let (_, records) = calibrate::gen_demo(200, 0);

    let report =
        calibrate::calibrate(&records, 100_000, 0, 0.10).expect("calibrate should succeed");

    // Claim 1: replay reproduces the source within tolerance.
    assert!(
        report.verdict.replay_pass,
        "replay should PASS with tolerance 0.10: TTFT err={:.4}, ITL err={:.4}",
        report.verdict.replay_ttft_max_error, report.verdict.replay_itl_max_error,
    );

    // The source trace must have a genuinely heavy tail.
    assert!(
        report.verdict.source_ttft_tail_ratio >= 3.0,
        "source TTFT p99/p50 should be >= 3.0, got {:.3}",
        report.verdict.source_ttft_tail_ratio,
    );

    // Claim 2: knob-fit tail is capped by the [0.3*mean, 1.7*mean] clamp.
    let knob_ttft = report
        .verdict
        .knobfit_ttft_tail_ratio
        .expect("model-level calibrate always carries knob-fit data");
    let knob_itl = report
        .verdict
        .knobfit_itl_tail_ratio
        .expect("model-level calibrate always carries knob-fit data");
    assert!(
        knob_ttft <= 1.75,
        "knob-fit TTFT p99/p50 should be <= 1.75, got {knob_ttft:.3}",
    );
    assert!(
        knob_itl <= 1.75,
        "knob-fit ITL p99/p50 should be <= 1.75, got {knob_itl:.3}",
    );

    // Print the report for visibility in test output.
    let mut buf = Vec::new();
    calibrate::write_report(&mut buf, &report).unwrap();
    let text = String::from_utf8(buf).unwrap();
    eprintln!("\n{text}");
}

// Test (b): calibrate-e2e smoke test
//
// Generates a small, fast-magnitude trace, spins the real simulator in-process,
// sends requests through it, and checks that measured TTFT p50s are within 40%
// of the source. Wrapped in a 120s timeout because CI can be slow.

#[tokio::test]
async fn calibrate_e2e_smoke() {
    use clap::Parser as _;
    use futures::StreamExt;
    use inference_simulator_rs::{Opt, run};
    use rand::Rng;
    use rand::SeedableRng as _;
    use tokio_util::sync::CancellationToken;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
    use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

    let result = tokio::time::timeout(Duration::from_secs(120), async {
        // Generate a fast trace to a temp file.
        let trace_path = std::env::temp_dir().join(format!(
            "inf-sim-cal-smoke-{}.jsonl",
            std::process::id()
        ));
        let (meta, records) = calibrate::gen_demo_fast(40, 42);
        calibrate::write_demo_trace(&trace_path, &meta, &records)
            .expect("writing demo trace");

        let num_requests = 20usize;
        let tolerance = 0.40;

        // Spin up the simulator
        let addr = format!(
            "ipc:///tmp/inf-sim-e2e-cal-{}-smoke.ipc",
            std::process::id()
        );
        let trace_str = trace_path.to_string_lossy().to_string();
        let args = vec![
            "inference-sim",
            "--handshake-address",
            &addr,
            "--max-num-seqs",
            "64",
            "--latency-trace",
            &trace_str,
        ];
        let opt = Opt::parse_from(&args);
        let token = CancellationToken::new();
        let sim_token = token.clone();
        let sim_opt = opt.clone();

        tokio::spawn(async move {
            let _ = run(sim_opt, sim_token).await;
        });

        let config = EngineCoreClientConfig::new_single(&addr);
        let client = tokio::time::timeout(
            Duration::from_secs(30),
            EngineCoreClient::connect(config),
        )
        .await
        .expect("client connect timed out")
        .expect("client connect failed");

        // Sample records and send requests
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut measured_ttfts: Vec<f64> = Vec::new();

        for i in 0..num_requests {
            let idx = rng.random_range(0..records.len());
            let rec = &records[idx];
            let max_tokens = rec.output_tokens.min(16) as u32;

            let request = EngineCoreRequest {
                request_id: format!("smoke-{i}"),
                prompt_token_ids: Some(calibrate::synthetic_prompt(i, rec.prompt_tokens)),
                sampling_params: Some(EngineCoreSamplingParams {
                    max_tokens,
                    ..EngineCoreSamplingParams::for_test()
                }),
                ..Default::default()
            };

            let call_start = std::time::Instant::now();
            let mut stream = client.call(request).await.expect("call failed");

            let mut got_first = false;
            while let Some(item) = stream.next().await {
                let output = item.expect("stream error");
                if !output.new_token_ids.is_empty() && !got_first {
                    let ttft_ms = call_start.elapsed().as_secs_f64() * 1000.0;
                    measured_ttfts.push(ttft_ms);
                    got_first = true;
                }
                if output.finish_reason.is_some() {
                    break;
                }
            }
        }

        token.cancel();

        // Check that we got measurements
        assert!(
            !measured_ttfts.is_empty(),
            "should have measured at least one TTFT"
        );

        // Check that measured p50 TTFT is within tolerance of source p50.
        measured_ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let measured_p50 = measured_ttfts[measured_ttfts.len() / 2];

        let mut source_ttfts: Vec<f64> = records.iter().map(|r| r.ttft_ms).collect();
        source_ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let source_p50 = source_ttfts[source_ttfts.len() / 2];

        let rel_err = (measured_p50 - source_p50).abs() / source_p50.max(1.0);
        eprintln!(
            "e2e smoke: measured_p50={:.1}ms, source_p50={:.1}ms, rel_err={:.3}",
            measured_p50, source_p50, rel_err,
        );

        assert!(
            rel_err <= tolerance,
            "measured p50 TTFT ({measured_p50:.1}ms) should be within {:.0}% of source ({source_p50:.1}ms), got {:.1}%",
            tolerance * 100.0,
            rel_err * 100.0,
        );
    })
    .await;

    result.expect("calibrate_e2e_smoke timed out (120s)");
}

// Test (c): open-loop arrival replay (calibrate-e2e --replay-arrivals)
//
// A constant trace (80ms TTFT, 7x20ms gaps) with arrivals every 250ms: every
// value the replay model can draw IS the source value, so the comparison
// isolates the harness's own transport and scheduling overhead. Arrivals are
// spaced past each request's 220ms duration so no prefill lands mid-decode
// (admission blocks running decodes by the prefill's service time, which the
// constant source values do not include). Also pins the open-loop property:
// wall time must cover the arrival schedule's span.

#[tokio::test]
async fn replay_arrivals_constant_trace() {
    use inference_simulator_rs::trace::{TraceMeta, TraceRecord};

    let result = tokio::time::timeout(Duration::from_secs(120), async {
        let n = 25usize;
        let records: Vec<TraceRecord> = (0..n)
            .map(|i| TraceRecord {
                prompt_tokens: 128,
                cached_tokens: 0,
                output_tokens: 8,
                ttft_ms: 80.0,
                itl_ms: Some(vec![20.0; 7]),
                itl_summary: None,
                concurrency: 1,
                arrival_ms: Some(i as f64 * 250.0),
                itl_ctx: None,
                ..Default::default()
            })
            .collect();
        let path = std::env::temp_dir().join(format!(
            "inf-sim-replay-const-{}.jsonl",
            std::process::id()
        ));
        calibrate::write_demo_trace(&path, &TraceMeta::default(), &records)
            .expect("writing trace");

        let cfg = calibrate::ReplayArrivalsConfig {
            trace_path: &path,
            latency_trace: None,
            max_requests: None,
            tolerance: 0.25,
            driver: calibrate::SimDriver::TraceReplay,
            ipc_tag: "test-const".to_string(),
            extra_sim_args: Vec::new(),
            pacing: calibrate::SessionPacing::OpenLoop,
            prompts: calibrate::PromptReplay::SharedPrefixes,
            time_scale: 1.0,
        };
        let outcome = calibrate::replay_arrivals(&cfg).await.expect("replay should run");

        assert_eq!(outcome.requests_replayed, n);
        assert_eq!(outcome.requests_completed, n, "all requests should complete");

        let mut buf = Vec::new();
        calibrate::write_report(&mut buf, &outcome.report).unwrap();
        eprintln!("\n{}", String::from_utf8(buf).unwrap());

        let rt = outcome
            .report
            .request_total
            .as_ref()
            .expect("request totals should be measured");
        assert!(
            outcome.report.verdict.replay_pass,
            "constant trace should replay within 25%: ttft err {:.3}, itl err {:.3}, totals err {:.3}",
            outcome.report.verdict.replay_ttft_max_error,
            outcome.report.verdict.replay_itl_max_error,
            rt.max_error,
        );

        // Open loop in real time: 25 arrivals at 100ms spacing span 2.4s.
        assert!(
            outcome.wall_time_s >= 2.4,
            "open-loop replay should honor the arrival schedule, took {:.2}s",
            outcome.wall_time_s,
        );
    })
    .await;

    result.expect("replay_arrivals_constant_trace timed out (120s)");
}

// Test (d): replay refuses traces without an arrival schedule, before
// spinning anything up.

#[tokio::test]
async fn replay_arrivals_requires_arrival_ms() {
    let (meta, records) = calibrate::gen_demo_fast(10, 7);
    let path = std::env::temp_dir().join(format!(
        "inf-sim-replay-noarrivals-{}.jsonl",
        std::process::id()
    ));
    calibrate::write_demo_trace(&path, &meta, &records).expect("writing trace");

    let cfg = calibrate::ReplayArrivalsConfig {
        trace_path: &path,
        latency_trace: None,
        max_requests: None,
        tolerance: 0.25,
        driver: calibrate::SimDriver::TraceReplay,
        ipc_tag: "test-noarrivals".to_string(),
        extra_sim_args: Vec::new(),
        pacing: calibrate::SessionPacing::OpenLoop,
        prompts: calibrate::PromptReplay::SharedPrefixes,
        time_scale: 1.0,
    };
    let err = calibrate::replay_arrivals(&cfg)
        .await
        .expect_err("traces without arrival_ms must be rejected");
    assert!(
        format!("{err:#}").contains("arrival_ms"),
        "error should name the missing field: {err:#}"
    );
}
