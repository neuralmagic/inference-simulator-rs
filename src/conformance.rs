//! Conformance assertions: does this build faithfully reproduce the vLLM line
//! it targets?
//!
//! Both checks are derivable from a captured trace's meta (recorded by the tap),
//! so the conformance runner can apply them offline on CPU:
//!
//!   - **schema**: the sim's registration ready response must carry every field
//!     the real engine emitted for this line (a superset). This is the
//!     automated, per-line generalization of the hand-written `block_size`
//!     canary in `sim-protocol`: when a new vLLM line adds a registration
//!     field, conformance flags that the sim does not emit it yet, instead of
//!     the frontend silently rejecting the registration at runtime.
//!   - **line**: the engine the golden was captured from must be on the same
//!     `major.minor` line this build targets, so a golden can't be replayed
//!     against the wrong build.

use anyhow::{Result, anyhow, bail};

use crate::frontend_connect::SimReadyResponse;

/// The msgpack-map keys of an encoded registration ready response.
fn ready_response_keys(payload: &[u8]) -> Result<Vec<String>> {
    let value = rmpv::decode::read_value(&mut &payload[..])
        .map_err(|e| anyhow!("decoding ready response msgpack: {e}"))?;
    let map = value
        .as_map()
        .ok_or_else(|| anyhow!("ready response is not a msgpack map"))?;
    Ok(map
        .iter()
        .filter_map(|(k, _)| k.as_str().map(str::to_string))
        .collect())
}

/// Assert the sim's ready response carries every field the real engine emitted
/// for this line. `recorded` is the raw `EngineCoreReadyResponse` bytes the tap
/// captured (trace meta `ready_response_hex`, hex-decoded).
pub fn assert_ready_response_schema(recorded: &[u8], sim_ready: &SimReadyResponse) -> Result<()> {
    let recorded_keys = ready_response_keys(recorded)?;
    let sim_payload = sim_ready.encode()?;
    let sim_keys = ready_response_keys(&sim_payload)?;

    let missing: Vec<&str> = recorded_keys
        .iter()
        .filter(|k| !sim_keys.iter().any(|s| s == *k))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        bail!(
            "ready-response schema drift: the captured engine emits fields the sim does not: \
             {missing:?} (sim emits {sim_keys:?}); the SimReadyResponse superset needs updating \
             for this vLLM line"
        );
    }
    Ok(())
}

/// Assert the golden was captured from an engine on the same `major.minor` line
/// this build targets. Strict: an unparseable *captured* version is a conformance
/// failure (unlike the tap's capture guard, which is lenient so a capture still runs).
///
/// Exception: a build target with no major.minor (e.g. `"nightly"`, which tracks vLLM
/// main) matches any captured version. Nightly *is* whatever main currently reports,
/// so there is no fixed line to compare against.
pub fn assert_same_line(build_target: &str, captured_version: &str) -> Result<()> {
    if sim_compat::minor_line(build_target).is_none() {
        return Ok(());
    }
    match (
        sim_compat::minor_line(build_target),
        sim_compat::minor_line(captured_version),
    ) {
        (Some(want), Some(got)) if want == got => Ok(()),
        (Some(want), Some(got)) => bail!(
            "vLLM line mismatch: build targets {build_target} (line {want}), golden captured \
             from {captured_version} (line {got})"
        ),
        _ => bail!(
            "could not parse a major.minor line (build target {build_target}, captured \
             {captured_version})"
        ),
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;
    use vllm_engine_core_client::protocol::ModelDtype;

    use crate::conformance::{assert_ready_response_schema, assert_same_line};
    use crate::frontend_connect::SimReadyResponse;

    /// Encode a msgpack map with the given keys (nil values) as a stand-in for a
    /// recorded `EngineCoreReadyResponse`.
    fn recorded_with_keys(keys: &[&str]) -> Vec<u8> {
        let map = Value::Map(keys.iter().map(|k| (Value::from(*k), Value::Nil)).collect());
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &map).expect("encode recorded ready response");
        buf
    }

    fn sim_ready() -> SimReadyResponse {
        SimReadyResponse {
            max_model_len: 32768,
            num_gpu_blocks: 1000,
            block_size: 16,
            dp_stats_address: None,
            dtype: ModelDtype::Float32,
            vllm_version: "v0.23.0".to_string(),
        }
    }

    #[test]
    fn schema_passes_when_sim_covers_every_recorded_field() {
        // The recorded engine emits exactly the fields the sim does.
        let recorded = recorded_with_keys(&[
            "max_model_len",
            "num_gpu_blocks",
            "block_size",
            "dp_stats_address",
            "dtype",
            "vllm_version",
        ]);
        assert_ready_response_schema(&recorded, &sim_ready()).expect("schema conformance");
    }

    #[test]
    fn schema_fails_when_a_new_line_adds_a_field_the_sim_lacks() {
        // A future vLLM line grew a registration field; the sim must be updated.
        let recorded = recorded_with_keys(&[
            "max_model_len",
            "num_gpu_blocks",
            "block_size",
            "dp_stats_address",
            "dtype",
            "vllm_version",
            "kv_cache_dtype",
        ]);
        let err = assert_ready_response_schema(&recorded, &sim_ready())
            .expect_err("must flag the missing field");
        assert!(err.to_string().contains("kv_cache_dtype"), "got: {err}");
    }

    #[test]
    fn line_check_matches_on_minor_across_suffixes() {
        assert_same_line("v0.23.0", "0.23.0.dev1+g16e9117").expect("same line");
    }

    #[test]
    fn line_check_rejects_a_different_minor() {
        let err = assert_same_line("v0.23.0", "0.22.1").expect_err("different line");
        assert!(err.to_string().contains("line mismatch"), "got: {err}");
    }

    #[test]
    fn line_check_is_strict_on_unparseable_versions() {
        assert!(assert_same_line("v0.23.0", "nightly-weird").is_err());
    }

    #[test]
    fn nightly_build_matches_any_captured_version() {
        // "nightly" tracks main, so it has no fixed line and accepts whatever the
        // captured engine reported (here a main dev build).
        assert_same_line("nightly", "0.24.0.dev1+gABCDEF").expect("nightly matches main");
        assert_same_line("nightly", "0.23.0").expect("nightly matches a release too");
    }
}
