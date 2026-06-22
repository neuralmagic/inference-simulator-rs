//! S3 object I/O for vllm-vcr trace files: a [`TraceUri`] is a local path
//! or an `s3://` object, fetched/uploaded via the AWS default credential chain.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use tracing::{debug, info};
use url::Url;

/// Whether a raw path string is an `s3://` URI rather than a local path.
pub fn is_remote(uri: &str) -> bool {
    uri.len() >= 5 && uri[..5].eq_ignore_ascii_case("s3://")
}

/// A trace location, parsed (and validated) at the CLI boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceUri {
    Local(PathBuf),
    S3 { bucket: String, key: String },
}

impl FromStr for TraceUri {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if is_remote(s) {
            let (bucket, key) = parse_s3_uri(s).map_err(|e| format!("{e:#}"))?;
            Ok(TraceUri::S3 { bucket, key })
        } else {
            Ok(TraceUri::Local(PathBuf::from(s)))
        }
    }
}

impl fmt::Display for TraceUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceUri::Local(path) => write!(f, "{}", path.display()),
            TraceUri::S3 { bucket, key } => write!(f, "s3://{bucket}/{key}"),
        }
    }
}

impl TraceUri {
    pub fn is_remote(&self) -> bool {
        matches!(self, TraceUri::S3 { .. })
    }

    /// The local path, when this is a local target (`None` for S3).
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            TraceUri::Local(path) => Some(path),
            TraceUri::S3 { .. } => None,
        }
    }

    /// A local path holding this trace's bytes: the path itself when local, or a
    /// scratch file fetched from S3.
    pub async fn materialize(&self, scratch_dir: &Path) -> Result<PathBuf> {
        match self {
            TraceUri::Local(path) => Ok(path.clone()),
            TraceUri::S3 { bucket, key } => self.fetch(bucket, key, scratch_dir).await,
        }
    }

    /// Where to write this trace locally before upload: its own path when local,
    /// else a scratch path under `scratch_dir`.
    pub fn write_path(&self, scratch_dir: &Path) -> PathBuf {
        match self {
            TraceUri::Local(path) => path.clone(),
            TraceUri::S3 { key, .. } => scratch_path(&self.to_string(), key, scratch_dir),
        }
    }

    /// Upload a finalized local file to this target; a no-op when local.
    pub async fn upload(&self, local: &Path) -> Result<()> {
        let TraceUri::S3 { bucket, key } = self else {
            return Ok(());
        };
        let size = std::fs::metadata(local).map(|m| m.len()).ok();
        info!(local = %local.display(), uri = %self, bucket, key, bytes = size, "S3 PUT: uploading trace");
        let started = Instant::now();
        let body = ByteStream::from_path(local)
            .await
            .with_context(|| format!("opening {} for upload", local.display()))?;
        s3_client()
            .await
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(body)
            .send()
            .await
            .with_context(|| format!("S3 PUT {self}"))?;
        info!(uri = %self, bytes = size, elapsed_ms = started.elapsed().as_millis(), "S3 PUT: trace uploaded");
        Ok(())
    }

    async fn fetch(&self, bucket: &str, key: &str, scratch_dir: &Path) -> Result<PathBuf> {
        let dest = scratch_path(&self.to_string(), key, scratch_dir);
        info!(uri = %self, bucket, key, dest = %dest.display(), "S3 GET: fetching trace to scratch");
        let started = Instant::now();
        let response = s3_client()
            .await
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("S3 GET {self}"))?;
        let content_length = response.content_length();
        let bytes = response
            .body
            .collect()
            .await
            .with_context(|| format!("reading S3 object body: {self}"))?
            .into_bytes();
        std::fs::write(&dest, &bytes)
            .with_context(|| format!("writing scratch {} for {self}", dest.display()))?;
        info!(uri = %self, bytes = bytes.len(), content_length, dest = %dest.display(), elapsed_ms = started.elapsed().as_millis(), "S3 GET: trace materialized");
        Ok(dest)
    }
}

fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let url = Url::parse(uri).with_context(|| format!("parsing S3 URI: {uri}"))?;
    if url.scheme() != "s3" {
        bail!(
            "expected an s3:// URI, got scheme {:?}: {uri}",
            url.scheme()
        );
    }
    let bucket = url
        .host_str()
        .filter(|host| !host.is_empty())
        .with_context(|| format!("S3 URI has no bucket: {uri}"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    if key.is_empty() {
        bail!("S3 URI has no object key: {uri}");
    }
    Ok((bucket, key))
}

fn key_basename(key: &str) -> &str {
    key.rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or("trace.jsonl")
}

/// Scratch path for a remote object: basename (keeping its suffix for gzip
/// detection) tagged with a hash of the URI so distinct objects don't collide.
fn scratch_path(uri: &str, key: &str, scratch_dir: &Path) -> PathBuf {
    use std::hash::{Hash as _, Hasher as _};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    uri.hash(&mut hasher);
    scratch_dir.join(format!(
        "sim-s3-{:016x}-{}",
        hasher.finish(),
        key_basename(key)
    ))
}

async fn s3_client() -> Client {
    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    // S3-compatible endpoints (MinIO/LocalStack) only serve path-style; real AWS
    // (no endpoint override) uses virtual-host style.
    let force_path_style = config.endpoint_url().is_some();
    debug!(
        region = config.region().map(|r| r.as_ref()),
        endpoint = config.endpoint_url(),
        force_path_style,
        "built S3 client from default credential chain"
    );
    Client::from_conf(
        aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(force_path_style)
            .build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_remote_only_matches_s3_scheme() {
        assert!(is_remote("s3://bucket/key"));
        assert!(is_remote("S3://Bucket/Key"));
        assert!(!is_remote("/tmp/trace.jsonl.gz"));
        assert!(!is_remote("trace.jsonl"));
        assert!(!is_remote("file:///tmp/trace.jsonl"));
        assert!(!is_remote(""));
        assert!(!is_remote("s3:"));
    }

    #[test]
    fn parses_s3_uri_into_typed_variant() {
        let uri: TraceUri = "s3://my-bucket/traces/abc/tap-trace.jsonl.gz"
            .parse()
            .unwrap();
        assert_eq!(
            uri,
            TraceUri::S3 {
                bucket: "my-bucket".to_string(),
                key: "traces/abc/tap-trace.jsonl.gz".to_string(),
            }
        );
        assert!(uri.is_remote());
        assert!(uri.local_path().is_none());
        assert_eq!(
            uri.to_string(),
            "s3://my-bucket/traces/abc/tap-trace.jsonl.gz"
        );
    }

    #[test]
    fn parses_bare_path_as_local() {
        let uri: TraceUri = "/tmp/trace.jsonl".parse().unwrap();
        assert_eq!(uri, TraceUri::Local(PathBuf::from("/tmp/trace.jsonl")));
        assert!(!uri.is_remote());
        assert_eq!(uri.local_path(), Some(Path::new("/tmp/trace.jsonl")));
    }

    #[test]
    fn rejects_malformed_s3_uri() {
        assert!("s3://bucket".parse::<TraceUri>().is_err()); // no key
        assert!("s3://bucket/".parse::<TraceUri>().is_err()); // empty key
        assert!("s3:///key".parse::<TraceUri>().is_err()); // no bucket
    }

    #[test]
    fn key_basename_keeps_gz_suffix() {
        assert_eq!(
            key_basename("traces/abc/tap-trace.jsonl.gz"),
            "tap-trace.jsonl.gz"
        );
        assert_eq!(key_basename("flat.jsonl"), "flat.jsonl");
        assert_eq!(key_basename("trailing/"), "trailing");
    }

    #[test]
    fn write_path_is_stable_per_uri_and_collision_free() {
        let dir = Path::new("/tmp/scratch");
        let a1: TraceUri = "s3://b/traces/a/tap-trace.jsonl.gz".parse().unwrap();
        let a2: TraceUri = "s3://b/traces/a/tap-trace.jsonl.gz".parse().unwrap();
        let b: TraceUri = "s3://b/traces/b/tap-trace.jsonl.gz".parse().unwrap();

        assert_eq!(a1.write_path(dir), a2.write_path(dir));
        assert_ne!(a1.write_path(dir), b.write_path(dir));
        assert!(
            a1.write_path(dir)
                .to_string_lossy()
                .ends_with("-tap-trace.jsonl.gz")
        );

        let local: TraceUri = "/tmp/x.jsonl".parse().unwrap();
        assert_eq!(local.write_path(dir), PathBuf::from("/tmp/x.jsonl"));
    }
}
