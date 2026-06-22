//! Replay-fidelity verification for multi-token-step captures: speculative
//! decoding and diffusion blocks.
//!
//! Boots the full sim on in-tree synthetic fixtures (`--latency-trace`) and
//! asserts the client-observed stream reproduces the capture's burst
//! structure end to end: chunk sizes stay within the recorded set, decode
//! paces at the recorded per-chunk gaps (not gap/N flattened), and token
//! totals land exactly.
//!
//! The fixtures mirror what the tap records from real engines:
//! - `spec_decode_trace.jsonl`: EAGLE-ish K=4, ~10ms steps delivering 1-5
//!   accepted tokens per chunk.
//! - `diffusion_block_trace.jsonl`: vLLM dLLM-path shape, 64-token blocks
//!   committed as one chunk every ~80ms.

use std::time::Duration;

use futures::StreamExt;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

use vllm_vcr::trace::{TraceRecord, read_trace_file};
use vllm_vcr::{Opt, run};

const TIMEOUT: Duration = Duration::from_secs(60);

/// One replayed request as the client saw it: tokens per stream message and
/// the gaps (ms) between consecutive token-bearing messages.
struct ObservedStream {
    chunks: Vec<usize>,
    gaps_ms: Vec<f64>,
}

/// Mean per-chunk gap and mean tokens-per-chunk recorded in a fixture.
fn capture_means(records: &[TraceRecord]) -> (f64, f64) {
    let mut gaps = Vec::new();
    let mut tokens = Vec::new();
    for r in records {
        gaps.extend(r.itl_ms.clone().unwrap_or_default());
        tokens.extend(
            r.itl_tokens
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(f64::from),
        );
    }
    let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let mean_tokens = tokens.iter().sum::<f64>() / tokens.len() as f64;
    (mean_gap, mean_tokens)
}

/// Boot the sim on `fixture`, replay each record's shape (prompt length and
/// output budget) sequentially, and collect the observed streams.
async fn replay_fixture(fixture: &str, requests: usize) -> (Vec<TraceRecord>, Vec<ObservedStream>) {
    let (_, records) = read_trace_file(std::path::Path::new(fixture)).expect("read fixture");
    assert!(!records.is_empty());

    let addr = format!(
        "ipc:///tmp/inf-sim-spec-fidelity-{}-{}.ipc",
        std::process::id(),
        fixture.len(),
    );
    let args = vec![
        "play",
        "--handshake-address",
        &addr,
        "--latency-trace",
        fixture,
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

    let mut observed = Vec::new();
    for i in 0..requests {
        let record = &records[i % records.len()];
        let req = EngineCoreRequest {
            request_id: format!("fidelity-{i}"),
            prompt_token_ids: Some(vec![42u32; record.prompt_tokens]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: record.output_tokens as u32,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        };
        let mut stream = client.call(req).await.expect("call failed");
        let mut chunks = Vec::new();
        let mut gaps_ms = Vec::new();
        let mut last: Option<Instant> = None;
        loop {
            let item = tokio::time::timeout(TIMEOUT, stream.next())
                .await
                .expect("stream timed out");
            let Some(item) = item else { break };
            let output = item.expect("stream error");
            if output.new_token_ids.is_empty() {
                continue;
            }
            let now = Instant::now();
            if let Some(prev) = last {
                gaps_ms.push(now.duration_since(prev).as_secs_f64() * 1000.0);
            }
            last = Some(now);
            chunks.push(output.new_token_ids.len());
        }
        observed.push(ObservedStream { chunks, gaps_ms });
    }
    token.cancel();
    (records, observed)
}

#[tokio::test]
async fn spec_decode_fixture_replays_burst_structure() {
    let (records, observed) = replay_fixture("tests/fixtures/spec_decode_trace.jsonl", 4).await;
    let (capture_gap, capture_tokens) = capture_means(&records);

    let mut all_decode_chunks = Vec::new();
    let mut all_gaps = Vec::new();
    for (i, stream) in observed.iter().enumerate() {
        let expected: usize = records[i % records.len()].output_tokens;
        let total: usize = stream.chunks.iter().sum();
        assert_eq!(total, expected, "request {i}: token total must match");
        // First chunk is the prefill step's single token; the rest are the
        // replayed bursts (the last may be capped by max_tokens).
        all_decode_chunks.extend(stream.chunks[1..].iter().copied());
        all_gaps.extend(stream.gaps_ms.iter().copied());
    }

    // Every decode burst must be a size the capture actually produced (1-5
    // accepted tokens for the K=4 fixture).
    assert!(
        all_decode_chunks.iter().all(|&c| (1..=5).contains(&c)),
        "burst sizes must come from the recorded chunk set, got {all_decode_chunks:?}"
    );
    let mean_chunk =
        all_decode_chunks.iter().sum::<usize>() as f64 / all_decode_chunks.len() as f64;
    assert!(
        (mean_chunk - capture_tokens).abs() <= 0.35 * capture_tokens,
        "mean burst size {mean_chunk:.2} must track the capture's {capture_tokens:.2}"
    );

    // Decode must pace at the recorded PER-CHUNK gaps. A flattened (gap/N)
    // replay would pace ~3x faster here; a one-token-per-step engine would
    // emit single-token chunks instead.
    let mean_gap = all_gaps.iter().sum::<f64>() / all_gaps.len() as f64;
    assert!(
        mean_gap >= 0.6 * capture_gap && mean_gap <= 2.5 * capture_gap,
        "mean observed gap {mean_gap:.2}ms must track the capture's {capture_gap:.2}ms"
    );
}

#[tokio::test]
async fn diffusion_fixture_replays_block_commits() {
    let (records, observed) = replay_fixture("tests/fixtures/diffusion_block_trace.jsonl", 3).await;
    let (capture_gap, _) = capture_means(&records);

    let mut all_gaps = Vec::new();
    for (i, stream) in observed.iter().enumerate() {
        let expected: usize = records[i % records.len()].output_tokens;
        let total: usize = stream.chunks.iter().sum();
        assert_eq!(total, expected, "request {i}: token total must match");

        // First chunk rides the prefill step (the recorded first block's size
        // is a known replay gap, see trace.rs itl_tokens docs); every decode
        // chunk after it must be a full 64-token block except a final
        // remainder capped by max_tokens.
        let decode = &stream.chunks[1..];
        let (body, tail) = decode.split_at(decode.len() - 1);
        assert!(
            body.iter().all(|&c| c == 64),
            "request {i}: blocks must commit whole, got {:?}",
            stream.chunks
        );
        assert!(tail[0] <= 64, "request {i}: tail block may only be capped");
        all_gaps.extend(stream.gaps_ms.iter().copied());
    }

    // ~80ms per block, not 80/64 per token: the flattened replay would pace
    // this two orders of magnitude faster.
    let mean_gap = all_gaps.iter().sum::<f64>() / all_gaps.len() as f64;
    assert!(
        mean_gap >= 0.6 * capture_gap && mean_gap <= 2.5 * capture_gap,
        "mean observed gap {mean_gap:.2}ms must track the capture's {capture_gap:.2}ms"
    );
}
