//! Parsing for `compat.toml` (the vLLM support window) and the conformance
//! golden manifest.
//!
//! Both are pure data files with no vLLM dependency, so this crate stays
//! protocol-free and can be used from three places that must agree on the same
//! source of truth:
//!
//!   - the root crate's `build.rs`, which stamps the default line's tag into the
//!     binary as `VLLM_TARGET_VERSION`,
//!   - the runtime handshake version guard, which refuses peers outside the
//!     build's supported line, and
//!   - the conformance runner, which replays each line's goldens.
//!
//! The manifest diff *is* the release: adding/removing a `[[vllm]]` line or
//! flipping `fidelity_validated` is what advances the N-3 window.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};

/// One supported vLLM line in the rolling N-3 window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VllmLine {
    /// Minor line, e.g. `"0.10"`. The grouping key the build matrix and image
    /// tags use (`inference-sim:<simver>-vllm0.10`).
    pub line: String,
    /// vLLM release tag, e.g. `"v0.10.1"`. Also the e2e frontend version and the
    /// string the handshake guard checks the peer against.
    pub tag: String,
    /// Git rev of the in-tree `vllm-engine-core-client` shipping with `tag`. The
    /// one axis Cargo can't multiplex, so each line builds against its own rev.
    /// This is the rev written into `[workspace.dependencies]` for the line's
    /// build (Cargo rejects patching a git dep to a different rev of the same
    /// source, so the rev is swapped in the dependency, not via `[patch]`).
    pub protocol_rev: String,
    /// Optional fork that `[patch]`-overrides `vllm-engine-core-client` for this
    /// line (a DIFFERENT source than `vllm.git`, e.g. a fork carrying a fix not
    /// yet upstream). The head line uses one for vllm-project/vllm#45848; lines
    /// without a fork build against `protocol_rev` upstream directly. Both must
    /// be set together or both omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_rev: Option<String>,
    /// True once the replay gates pass against this line's captured goldens. A
    /// line enters the window as `false` and is promoted once conformance is green.
    #[serde(default)]
    pub fidelity_validated: bool,
    /// Exactly one line carries `default = true`; it is `:latest` and the
    /// unsuffixed build.
    #[serde(default)]
    pub default: bool,
}

/// The parsed `compat.toml` support window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatManifest {
    #[serde(rename = "vllm", default)]
    pub lines: Vec<VllmLine>,
}

impl CompatManifest {
    /// Parse and validate `compat.toml` text.
    pub fn parse(text: &str) -> Result<Self> {
        let manifest: CompatManifest = toml::from_str(text).context("parsing compat.toml")?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Load and validate `compat.toml` from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text)
    }

    /// Enforce the invariants the build matrix relies on: at least one line, and
    /// exactly one `default = true`.
    pub fn validate(&self) -> Result<()> {
        if self.lines.is_empty() {
            bail!("compat.toml defines no [[vllm]] lines");
        }
        let defaults = self.lines.iter().filter(|l| l.default).count();
        if defaults != 1 {
            bail!("compat.toml must have exactly one line with default = true, found {defaults}");
        }
        for l in &self.lines {
            if l.patch_repo.is_some() != l.patch_rev.is_some() {
                bail!(
                    "compat.toml line {} sets only one of patch_repo/patch_rev; set both or neither",
                    l.line
                );
            }
        }
        Ok(())
    }

    /// The single `default = true` line (`:latest` / the unsuffixed build target).
    pub fn default_line(&self) -> Result<&VllmLine> {
        self.lines
            .iter()
            .find(|l| l.default)
            .context("compat.toml has no default line")
    }

    /// Look up a line by its minor (`"0.10"`).
    pub fn line(&self, minor: &str) -> Option<&VllmLine> {
        self.lines.iter().find(|l| l.line == minor)
    }

    /// The set of vLLM release tags this window supports.
    pub fn supported_tags(&self) -> BTreeSet<&str> {
        self.lines.iter().map(|l| l.tag.as_str()).collect()
    }
}

/// Extract the `major.minor` line from a vLLM version string, tolerating a
/// leading `v` and trailing patch/pre-release/build suffixes. Returns `None`
/// when the string has no parseable `X.Y`. Used to compare a release tag
/// (`"v0.23.0"`) against an engine's reported version (`"0.23.0.dev1+g16e9"`):
/// both reduce to `"0.23"`, so patch/dev/build differences don't trip a false
/// mismatch. Shared by the tap's capture guard and the conformance runner so
/// they agree on what "same line" means.
pub fn minor_line(version: &str) -> Option<String> {
    let mut parts = version.trim().trim_start_matches('v').split('.');
    let major = parts.next()?;
    let minor = parts.next()?;
    if major.is_empty() || !major.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let minor: String = minor.chars().take_while(|c| c.is_ascii_digit()).collect();
    if minor.is_empty() {
        return None;
    }
    Some(format!("{major}.{minor}"))
}

/// What a golden capture is used to assert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GoldenRole {
    /// Structural/handshake conformance: small, cheap, runs on every CI build.
    Schema,
    /// Behavioral replay fidelity: larger captures, tolerance-banded gates.
    Fidelity,
}

/// One golden capture, stored in the private bucket and referenced by sha.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenEntry {
    /// vLLM line this golden validates; matches a [`VllmLine::line`].
    pub line: String,
    /// Where the golden lives, fetched directly via sim-s3 by the conformance
    /// runner. Either a full `s3://bucket/key` URI, or a bare key resolved under
    /// `$CONFORMANCE_BUCKET` (e.g. `conformance/0.23/sweep.jsonl.gz`).
    pub bucket_path: String,
    /// Content hash. CI fetches and verifies the capture against this.
    pub sha256: String,
    /// Deployment-config fingerprint the capture was recorded under. The replay
    /// gate passes this as `--expect-config-hash` so a trace can't be replayed
    /// against a config it wasn't captured for.
    pub config_hash: String,
    /// Workload label, e.g. `"sweep"` or `"multiturn"`.
    pub workload: String,
    /// Whether this golden drives schema or fidelity conformance.
    pub role: GoldenRole,
}

/// The parsed conformance golden manifest (`conformance/manifest.toml`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GoldenManifest {
    #[serde(rename = "golden", default)]
    pub goldens: Vec<GoldenEntry>,
}

impl GoldenManifest {
    /// Parse manifest text.
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("parsing conformance manifest")
    }

    /// Load the manifest from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text)
    }

    /// Goldens that validate a given vLLM line.
    pub fn for_line<'a>(&'a self, line: &str) -> impl Iterator<Item = &'a GoldenEntry> {
        self.goldens.iter().filter(move |g| g.line == line)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CompatManifest, GoldenManifest, GoldenRole, minor_line};

    #[test]
    fn minor_line_strips_prefix_and_suffixes() {
        assert_eq!(minor_line("v0.23.0").as_deref(), Some("0.23"));
        assert_eq!(minor_line("0.23.0").as_deref(), Some("0.23"));
        assert_eq!(minor_line("0.23.0.dev1+g16e9117").as_deref(), Some("0.23"));
        assert_eq!(minor_line("v0.9.2").as_deref(), Some("0.9"));
        assert_eq!(minor_line("garbage").as_deref(), None);
        assert_eq!(minor_line("").as_deref(), None);
    }

    const SAMPLE: &str = r#"
[[vllm]]
line = "0.10"
tag = "v0.10.1"
protocol_rev = "aaaa"
fidelity_validated = true
default = true

[[vllm]]
line = "0.9"
tag = "v0.9.2"
protocol_rev = "bbbb"
fidelity_validated = false
"#;

    #[test]
    fn parses_and_picks_default_line() {
        let manifest = CompatManifest::parse(SAMPLE).expect("parse");
        assert_eq!(manifest.lines.len(), 2);
        let default = manifest.default_line().expect("default");
        assert_eq!(default.line, "0.10");
        assert_eq!(default.tag, "v0.10.1");
        assert!(manifest.line("0.9").is_some());
        assert!(!manifest.line("0.9").expect("0.9 line").fidelity_validated);
    }

    #[test]
    fn supported_tags_covers_every_line() {
        let manifest = CompatManifest::parse(SAMPLE).expect("parse");
        let tags = manifest.supported_tags();
        assert!(tags.contains("v0.10.1"));
        assert!(tags.contains("v0.9.2"));
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn rejects_zero_defaults() {
        let text = r#"
[[vllm]]
line = "0.10"
tag = "v0.10.1"
protocol_rev = "aaaa"
"#;
        let err = CompatManifest::parse(text).expect_err("must reject zero defaults");
        assert!(err.to_string().contains("exactly one"), "got: {err}");
    }

    #[test]
    fn rejects_multiple_defaults() {
        let text = r#"
[[vllm]]
line = "0.10"
tag = "v0.10.1"
protocol_rev = "aaaa"
default = true

[[vllm]]
line = "0.9"
tag = "v0.9.2"
protocol_rev = "bbbb"
default = true
"#;
        let err = CompatManifest::parse(text).expect_err("must reject multiple defaults");
        assert!(err.to_string().contains("exactly one"), "got: {err}");
    }

    #[test]
    fn rejects_half_specified_patch() {
        let text = r#"
[[vllm]]
line = "0.23"
tag = "v0.23.0"
protocol_rev = "aaaa"
patch_rev = "bbbb"
default = true
"#;
        let err = CompatManifest::parse(text).expect_err("must reject half-specified patch");
        assert!(
            err.to_string().contains("patch_repo/patch_rev"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_empty_manifest() {
        let err = CompatManifest::parse("").expect_err("must reject empty");
        assert!(err.to_string().contains("no [[vllm]] lines"), "got: {err}");
    }

    #[test]
    fn golden_manifest_round_trips_and_filters_by_line() {
        let text = r#"
[[golden]]
line = "0.10"
bucket_path = "s3://bucket/0.10/sweep.jsonl.gz"
sha256 = "deadbeef"
config_hash = "cafef00d"
workload = "sweep"
role = "fidelity"

[[golden]]
line = "0.10"
bucket_path = "s3://bucket/0.10/handshake.jsonl"
sha256 = "0011"
config_hash = "cafef00d"
workload = "handshake"
role = "schema"

[[golden]]
line = "0.9"
bucket_path = "s3://bucket/0.9/sweep.jsonl.gz"
sha256 = "2233"
config_hash = "99aa"
workload = "sweep"
role = "fidelity"
"#;
        let manifest = GoldenManifest::parse(text).expect("parse");
        assert_eq!(manifest.goldens.len(), 3);
        let head: Vec<_> = manifest.for_line("0.10").collect();
        assert_eq!(head.len(), 2);
        assert!(head.iter().any(|g| g.role == GoldenRole::Schema));
        assert!(head.iter().any(|g| g.role == GoldenRole::Fidelity));
        assert_eq!(manifest.for_line("0.9").count(), 1);
    }
}
