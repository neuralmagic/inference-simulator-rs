//! Manifest-driven conformance runner (capture-once-GPU / replay-many-CPU).
//!
//! For the vLLM line this build targets (`VLLM_TARGET_VERSION`), this reads
//! `conformance/manifest.toml`, locates each golden the CI fetch step placed
//! under `$CONFORMANCE_GOLDENS`, and asserts the build conforms:
//!
//!   - line: the golden was captured from an engine on this build's line.
//!   - provenance: the trace's `config_hash` matches what the manifest claims.
//!   - schema: the sim's `SimReadyResponse` covers every field the captured
//!     engine emitted (schema-role goldens).
//!   - fidelity: replayed token streams are byte-identical to the capture
//!     (fidelity-role goldens that recorded tokens).
//!
//! It skips cleanly (returns, doesn't fail) when there are no goldens for this
//! line or `$CONFORMANCE_GOLDENS` is unset, which is the normal state until
//! captures exist. The CI matrix sets `$CONFORMANCE_GOLDENS` after fetching by
//! sha; locally:
//!
//! ```bash
//! CONFORMANCE_GOLDENS=/path/to/fetched cargo test --test conformance -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser as _;
use futures::StreamExt as _;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams, ModelDtype,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

use inference_simulator_rs::conformance::{assert_ready_response_schema, assert_same_line};
use inference_simulator_rs::frontend_connect::SimReadyResponse;
use inference_simulator_rs::trace::{TraceMeta, read_trace_file, replay_subset};
use inference_simulator_rs::{Opt, VLLM_TARGET_VERSION, run};
use sim_compat::{GoldenManifest, GoldenRole};

const TIMEOUT: Duration = Duration::from_secs(30);

/// Manifest path: `$CONFORMANCE_MANIFEST` override, else the repo's
/// `conformance/manifest.toml`.
fn default_manifest_path() -> PathBuf {
    std::env::var("CONFORMANCE_MANIFEST")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/manifest.toml"))
}

fn hex_decode(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "hex string has odd length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// A representative ready response with the sim's registration field set. Only
/// the key set matters for schema conformance, not the values.
fn sim_ready_response() -> SimReadyResponse {
    SimReadyResponse {
        max_model_len: 32768,
        num_gpu_blocks: 1000,
        block_size: 16,
        dp_stats_address: None,
        dtype: ModelDtype::Float32,
        vllm_version: VLLM_TARGET_VERSION.to_string(),
    }
}

/// Static checks derivable straight from the trace meta.
fn check_meta(meta: &TraceMeta, expected_config_hash: &str, role: GoldenRole) {
    let captured = meta
        .vllm_version
        .as_deref()
        .expect("golden trace meta must record the engine's vllm_version");
    assert_same_line(VLLM_TARGET_VERSION, captured).expect("vLLM line conformance");

    let trace_hash = meta
        .config_hash
        .as_deref()
        .expect("golden trace meta must carry a config_hash");
    assert_eq!(
        trace_hash, expected_config_hash,
        "golden provenance: trace config_hash must match the manifest entry"
    );

    if role == GoldenRole::Schema {
        let recorded = hex_decode(
            meta.ready_response_hex
                .as_deref()
                .expect("schema-role golden must record ready_response_hex"),
        );
        assert_ready_response_schema(&recorded, &sim_ready_response())
            .expect("ready-response schema conformance");
    }
}

/// Boot the sim on this golden and assert every recorded token stream replays
/// byte-identically, gated on the trace's `config_hash` so a mislabelled golden
/// can't slip through.
async fn check_fidelity(trace_path: &Path) {
    let path = trace_path.to_string_lossy().to_string();
    let (_, records) = read_trace_file(trace_path).expect("read golden trace");
    let subset = replay_subset(records);
    let with_tokens = subset
        .iter()
        .filter(|r| r.output_token_ids.is_some())
        .count();
    if with_tokens == 0 {
        eprintln!("  fidelity: golden has no recorded tokens, skipping byte-identical replay");
        return;
    }

    let addr = format!("ipc:///tmp/inf-sim-conformance-{}.ipc", std::process::id());
    let opt = Opt::parse_from([
        "inference-sim",
        "--handshake-address",
        &addr,
        "--replay-tokens",
        &path,
    ]);
    // The provenance gate: refuse to replay a trace from a different config.
    opt.verify_config_hash().expect("config-hash gate");

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
        let last = outputs.last().expect("at least one output");
        let last = last.as_ref().expect("stream error");
        let expected_finish = record
            .finish_reason
            .map(inference_simulator_rs::wire::engine_finish_reason)
            .unwrap_or(EngineCoreFinishReason::Length);
        assert_eq!(
            last.finish_reason,
            Some(expected_finish),
            "replay-{i}: finish reason must match the capture"
        );
    }
    token.cancel();
    eprintln!("  fidelity: {with_tokens} token streams byte-identical, finish reasons match");
}

/// Run conformance against a manifest. Returns the number of goldens actually
/// checked (0 means cleanly skipped: none for this line, or none fetched).
async fn run_conformance(manifest_path: &Path, goldens_dir: Option<&Path>) -> usize {
    let manifest = GoldenManifest::load(manifest_path).expect("load conformance manifest");
    // The build's compat line: the major.minor for a release tag (v0.23.0 -> 0.23), or
    // the tag verbatim when it has no major.minor (e.g. "nightly", which tracks main).
    // Matches the `line` field on golden entries (compat.toml uses "0.23" / "nightly").
    let line = sim_compat::minor_line(VLLM_TARGET_VERSION)
        .unwrap_or_else(|| VLLM_TARGET_VERSION.to_string());

    let goldens: Vec<_> = manifest.for_line(&line).collect();
    if goldens.is_empty() {
        eprintln!("no conformance goldens for vLLM line {line}; skipping (none captured yet)");
        return 0;
    }

    let Some(dir) = goldens_dir else {
        eprintln!(
            "no goldens dir; {} golden(s) listed for line {line} but not fetched, skipping",
            goldens.len()
        );
        return 0;
    };

    let mut checked = 0;
    for golden in goldens {
        // The CI fetch step mirrors the bucket key under $CONFORMANCE_GOLDENS, so the
        // local path is the bucket_path joined onto the fetch dir. Keeping the full key
        // (not just the basename) lets the path carry the model/workload structure
        // (conformance/<line>/<model>/<workload>...) without filename collisions.
        let trace_path = dir.join(&golden.bucket_path);
        if !trace_path.exists() {
            eprintln!(
                "  {} ({:?}) not present at {}; skipping",
                golden.workload,
                golden.role,
                trace_path.display()
            );
            continue;
        }
        eprintln!(
            "conformance: {} [{:?}] {}",
            golden.workload, golden.role, golden.bucket_path
        );

        let (meta, _) = read_trace_file(&trace_path).expect("read golden meta");
        check_meta(&meta, &golden.config_hash, golden.role);
        if golden.role == GoldenRole::Fidelity {
            check_fidelity(&trace_path).await;
        }
        checked += 1;
    }
    checked
}

#[tokio::test]
async fn build_conforms_to_its_vllm_line_goldens() {
    let dir = std::env::var("CONFORMANCE_GOLDENS").ok().map(PathBuf::from);
    run_conformance(&default_manifest_path(), dir.as_deref()).await;
}

/// Drive the schema/line/provenance path end to end against a synthetic golden,
/// so the runner's plumbing is exercised even before real captures exist.
#[tokio::test]
async fn synthetic_schema_golden_passes_the_static_checks() {
    use std::io::Write as _;

    let line = sim_compat::minor_line(VLLM_TARGET_VERSION)
        .unwrap_or_else(|| VLLM_TARGET_VERSION.to_string());
    // A release line (0.23) needs a captured version on that line; nightly tracks main,
    // so assert_same_line accepts any version there (use a representative dev build).
    let captured_version = if sim_compat::minor_line(VLLM_TARGET_VERSION).is_some() {
        format!("{line}.0.dev1+gTEST")
    } else {
        "0.99.0.dev1+gTEST".to_string()
    };
    let config_hash = "synthetic-config-hash";

    // A ready response carrying exactly the sim's registration field set.
    let ready_keys = [
        "max_model_len",
        "num_gpu_blocks",
        "block_size",
        "dp_stats_address",
        "dtype",
        "vllm_version",
    ];
    let map = rmpv::Value::Map(
        ready_keys
            .iter()
            .map(|k| (rmpv::Value::from(*k), rmpv::Value::Nil))
            .collect(),
    );
    let mut ready_bytes = Vec::new();
    rmpv::encode::write_value(&mut ready_bytes, &map).expect("encode ready response");
    let ready_hex: String = ready_bytes.iter().map(|b| format!("{b:02x}")).collect();

    // Unique temp dir for this test run.
    let dir = std::env::temp_dir().join(format!("conformance-synthetic-{}", std::process::id()));
    // The golden lives at the bucket key under the fetch dir (mirrors the CI fetch).
    let bucket_path = format!("conformance/{line}/test-gpu/test-org/test-model/golden.jsonl");
    let golden_path = dir.join(&bucket_path);
    std::fs::create_dir_all(golden_path.parent().expect("golden has a parent"))
        .expect("create temp dir");

    let meta = TraceMeta {
        source: Some("tap".to_string()),
        config_hash: Some(config_hash.to_string()),
        vllm_version: Some(captured_version),
        ready_response_hex: Some(ready_hex),
        ..TraceMeta::default()
    };
    {
        let mut f = std::fs::File::create(&golden_path).expect("create golden");
        let wrapper = serde_json::json!({ "meta": meta });
        serde_json::to_writer(&mut f, &wrapper).expect("write meta");
        writeln!(f).expect("newline");
    }

    let manifest_path = dir.join("manifest.toml");
    std::fs::write(
        &manifest_path,
        format!(
            "[[golden]]\n\
             line = \"{line}\"\n\
             bucket_path = \"{bucket_path}\"\n\
             sha256 = \"deadbeef\"\n\
             config_hash = \"{config_hash}\"\n\
             workload = \"synthetic\"\n\
             role = \"schema\"\n"
        ),
    )
    .expect("write manifest");

    let checked = run_conformance(&manifest_path, Some(&dir)).await;
    assert_eq!(checked, 1, "the synthetic schema golden must be checked");

    let _ = std::fs::remove_dir_all(&dir);
}
