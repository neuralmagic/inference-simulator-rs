//! JSONL trace schema for recording and replaying engine request latencies.
//!
//! The trace format is shared by the replay latency model (`TraceLatency`), the guidellm
//! converter binary, and the recording tap binary. Format:
//!
//!   Line 1 (optional): `{"meta": {...}}` with fields like model, gpu, tp, max_num_seqs,
//!   source, plus a freeform `extra` map.
//!
//!   Subsequent lines: one `TraceRecord` per completed request, carrying observed prompt
//!   size, cache hit, output length, TTFT, and inter-token latencies (either an array or
//!   a summary).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};

/// Metadata header optionally present as the first line of a trace file.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TraceMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_seqs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Token-block size behind the records' `block_hashes` (prefix-sharing
    /// fingerprints). Hashes are chained per block, mooncake-style.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// Hash of the deployment config this trace was captured under (the CI
    /// profile-once/replay-many cache key). Stamped by the tap at capture and
    /// checked by the sim at replay, so a trace cannot be replayed against a
    /// config it was not recorded for. `None` on traces captured before this
    /// field existed, or outside the cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_hash: Option<String>,
    /// Freeform key-value pairs for any fields not covered above.
    #[serde(default, skip_serializing_if = "HashMap::is_empty", flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Wrapper for the meta line: `{"meta": {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaWrapper {
    meta: TraceMeta,
}

/// Summary of inter-token latencies when the full array is not available.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ItlSummary {
    pub mean_ms: f64,
    pub count: usize,
}

/// Why a request finished, mirroring vLLM's finish reasons in a
/// human-readable wire form.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TraceFinishReason {
    Stop,
    Length,
    Abort,
    Error,
    Repetition,
}

/// Per-gap batch context, parallel to `itl_ms`: the engine state under which each
/// gap was measured. This is what lets a replay model separate clean decode steps
/// from prefill-interfered ones and condition on the *simulated* scheduler state
/// instead of replaying a steady-state marginal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ItlContext {
    /// Engine-reported running-request count for the step that closed each gap
    /// (falls back to the recorder's own in-flight count when the engine doesn't
    /// attach scheduler stats).
    pub num_running: Vec<u32>,
    /// Prompt tokens that finished prefill in the step that closed each gap
    /// (0 = clean decode step). Chunked prefill attributes the whole prompt to
    /// the step that emitted the first token.
    pub prefill_tokens: Vec<u32>,
}

/// One completed request's observed latency. At least one of `itl_ms` or `itl_summary`
/// must be present when `output_tokens > 1`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceRecord {
    pub prompt_tokens: usize,
    #[serde(default)]
    pub cached_tokens: usize,
    pub output_tokens: usize,
    pub ttft_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_ms: Option<Vec<f64>>,
    /// Tokens delivered by the chunk that closed each gap, parallel to
    /// `itl_ms`. Absent means one token per gap (every gap is a plain
    /// autoregressive step). Speculative decoding and diffusion-block engines
    /// emit several tokens per step as one chunk; each such chunk contributes
    /// ONE `itl_ms` gap (the full step time) and its token count here, so the
    /// burst structure survives capture. The first chunk has no gap; its size
    /// is `output_tokens - sum(itl_tokens)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_tokens: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_summary: Option<ItlSummary>,
    #[serde(default = "default_concurrency")]
    pub concurrency: u64,
    /// Request arrival time in milliseconds since the start of the capture
    /// (first observed request = ~0). Records are written at completion, so file
    /// order is finish order; this field carries the arrival schedule, which is
    /// what an open-loop workload replay needs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrival_ms: Option<f64>,
    /// Per-gap batch context, parallel to `itl_ms` when both are present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_ctx: Option<ItlContext>,
    /// Chained hashes of the prompt's full token blocks (block size in
    /// `TraceMeta::block_size`), recording prefix-sharing structure without the
    /// tokens themselves: two requests share a prompt prefix exactly where
    /// their hash chains agree. Lets a replay reconstruct prompts whose prefix
    /// overlap (and thus prefix-cache behavior) matches the capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_hashes: Option<Vec<u64>>,
    /// The request's actual output token ids, recorded only when the capture
    /// opts in (tap `--record-tokens`). Length must equal `output_tokens`.
    /// CAUTION: with the same tokenizer these decode back to the generated
    /// text, so traces carrying this field contain user content and lose the
    /// share-freely property of the hash-only schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    /// Why the request finished. Recorded by the tap regardless of
    /// `--record-tokens` (it carries no content).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<TraceFinishReason>,
}

fn default_concurrency() -> u64 {
    1
}

/// Matches the serde field defaults (notably `concurrency: 1`), so literal
/// initializers can spell out only what they care about.
impl Default for TraceRecord {
    fn default() -> Self {
        TraceRecord {
            prompt_tokens: 0,
            cached_tokens: 0,
            output_tokens: 0,
            ttft_ms: 0.0,
            itl_ms: None,
            itl_tokens: None,
            itl_summary: None,
            concurrency: default_concurrency(),
            arrival_ms: None,
            itl_ctx: None,
            block_hashes: None,
            output_token_ids: None,
            finish_reason: None,
        }
    }
}

/// Chained FNV-1a fingerprints of a prompt's full token blocks, for
/// `TraceRecord::block_hashes`. Block i's hash folds block i-1's hash, so two
/// records' chains agree exactly as far as their prompts share a prefix (the
/// same chaining idea as vLLM's prefix-cache blocks and mooncake's trace
/// format). The partial tail block is not hashed (it can never be a cache
/// hit). Returns None when the prompt has no full block.
pub fn prompt_block_hashes(tokens: &[u32], block_size: usize) -> Option<Vec<u64>> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    if block_size == 0 || tokens.len() < block_size {
        return None;
    }
    let mut hashes = Vec::with_capacity(tokens.len() / block_size);
    let mut prev: u64 = FNV_OFFSET;
    for block in tokens.chunks_exact(block_size) {
        let mut h = FNV_OFFSET;
        for byte in prev
            .to_le_bytes()
            .into_iter()
            .chain(block.iter().flat_map(|t| t.to_le_bytes()))
        {
            h = (h ^ byte as u64).wrapping_mul(FNV_PRIME);
        }
        prev = h;
        hashes.push(h);
    }
    Some(hashes)
}

/// Whether a trace path names a gzip-compressed file. Token-recording traces
/// get large (one integer per generated token), so every trace touchpoint
/// reads and writes `.gz` transparently.
fn is_gz(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "gz")
}

/// Open a trace file for buffered reading, transparently decompressing when
/// the path ends in `.gz`.
pub fn open_trace_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file =
        File::open(path).with_context(|| format!("opening trace file: {}", path.display()))?;
    Ok(if is_gz(path) {
        Box::new(BufReader::new(MultiGzDecoder::new(BufReader::new(file))))
    } else {
        Box::new(BufReader::new(file))
    })
}

/// Open and parse a trace file (gzip-aware, see [`open_trace_reader`]).
pub fn read_trace_file(path: &Path) -> Result<(TraceMeta, Vec<TraceRecord>)> {
    read_trace(open_trace_reader(path)?)
        .with_context(|| format!("parsing trace file: {}", path.display()))
}

/// Read only the metadata header of a trace, without parsing records. Returns a
/// default (empty) meta when the file has no `{"meta": ...}` header line. Used
/// for cheap provenance checks (e.g. config-hash verification at replay).
pub fn read_trace_meta(path: &Path) -> Result<TraceMeta> {
    read_meta(open_trace_reader(path)?)
        .with_context(|| format!("reading trace meta: {}", path.display()))
}

/// Parse just the meta header from a reader: scan to the first non-blank line,
/// and parse it as meta only if it carries a `"meta"` key (otherwise it is a
/// record and the trace has no header).
fn read_meta(reader: impl BufRead) -> Result<TraceMeta> {
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .with_context(|| format!("line {}: invalid JSON", idx + 1))?;
        if value.get("meta").is_some() {
            let wrapper: MetaWrapper = serde_json::from_value(value)
                .with_context(|| format!("line {}: invalid meta object", idx + 1))?;
            return Ok(wrapper.meta);
        }
        // First non-blank line is a record: no header.
        return Ok(TraceMeta::default());
    }
    Ok(TraceMeta::default())
}

/// A trace file writer: plain JSONL, or gzip when the path ends in `.gz`.
///
/// Call [`TraceWriter::finish`] when done - a gzip stream needs its trailer
/// written, and dropping the encoder swallows any error doing so.
pub enum TraceWriter {
    Plain(BufWriter<File>),
    Gzip(Box<GzEncoder<BufWriter<File>>>),
}

impl TraceWriter {
    /// Create (truncate) a trace file, gzip-compressing when the path ends in
    /// `.gz`.
    pub fn create(path: &Path) -> Result<TraceWriter> {
        let file = File::create(path)
            .with_context(|| format!("creating trace file: {}", path.display()))?;
        let buffered = BufWriter::new(file);
        Ok(if is_gz(path) {
            TraceWriter::Gzip(Box::new(GzEncoder::new(buffered, Compression::default())))
        } else {
            TraceWriter::Plain(buffered)
        })
    }

    /// Finalize the file: write the gzip trailer (when compressing) and flush.
    pub fn finish(self) -> Result<()> {
        match self {
            TraceWriter::Plain(mut writer) => writer.flush().context("flushing trace file"),
            TraceWriter::Gzip(encoder) => encoder
                .finish()
                .context("finishing gzip trace stream")?
                .flush()
                .context("flushing trace file"),
        }
    }
}

impl Write for TraceWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            TraceWriter::Plain(writer) => writer.write(buf),
            TraceWriter::Gzip(encoder) => encoder.write(buf),
        }
    }

    /// Flushing a gzip stream emits a sync point so the data written so far is
    /// readable (the tap flushes per record); this costs a little compression
    /// ratio compared to one final flush.
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            TraceWriter::Plain(writer) => writer.flush(),
            TraceWriter::Gzip(encoder) => encoder.flush(),
        }
    }
}

/// Parse a trace from any `BufRead`. The first line is checked for a `"meta"` key; if
/// present it is parsed as `TraceMeta`, otherwise the whole file is treated as records
/// with a default meta. Blank lines are skipped. Errors include line numbers.
pub fn read_trace(reader: impl BufRead) -> Result<(TraceMeta, Vec<TraceRecord>)> {
    let mut meta = TraceMeta::default();
    let mut records = Vec::new();
    let mut lines = reader.lines();
    let mut line_num: usize = 0;

    // Try to parse the first non-blank line as meta.
    let first_line = loop {
        line_num += 1;
        match lines.next() {
            Some(Ok(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                break Some((line_num, trimmed.to_string()));
            }
            Some(Err(e)) if is_unfinalized_gz_eof(&e) => {
                warn_unfinalized_trace(line_num, records.len());
                break None;
            }
            Some(Err(e)) => return Err(e).with_context(|| format!("reading line {line_num}")),
            None => break None,
        }
    };

    let Some((first_num, first_text)) = first_line else {
        return Ok((meta, records));
    };

    // Check if the first line has a "meta" key.
    let first_value: serde_json::Value = serde_json::from_str(&first_text)
        .with_context(|| format!("line {first_num}: invalid JSON"))?;

    if first_value.get("meta").is_some() {
        let wrapper: MetaWrapper = serde_json::from_value(first_value)
            .with_context(|| format!("line {first_num}: invalid meta object"))?;
        meta = wrapper.meta;
    } else {
        // First line is a record.
        let record: TraceRecord = serde_json::from_value(first_value)
            .with_context(|| format!("line {first_num}: invalid trace record"))?;
        validate_record(&record, first_num)?;
        records.push(record);
    }

    for item in lines {
        line_num += 1;
        let line = match item {
            Ok(line) => line,
            Err(e) if is_unfinalized_gz_eof(&e) => {
                warn_unfinalized_trace(line_num, records.len());
                break;
            }
            Err(e) => return Err(e).with_context(|| format!("reading line {line_num}")),
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: TraceRecord = serde_json::from_str(trimmed)
            .with_context(|| format!("line {line_num}: invalid trace record"))?;
        validate_record(&record, line_num)?;
        records.push(record);
    }

    Ok((meta, records))
}

/// A gzip trace flushed per-record for live capture (see [`TraceWriter`]) has no
/// final trailer until [`TraceWriter::finish`] runs. When the tap is still up
/// (e.g. captured via `kubectl exec cat` under `--no-cleanup`), the trailer is
/// absent and `MultiGzDecoder` reports the missing trailer as `UnexpectedEof`.
/// Every complete record precedes it, so this is a clean end-of-stream, not a
/// parse failure.
fn is_unfinalized_gz_eof(err: &std::io::Error) -> bool {
    err.kind() == std::io::ErrorKind::UnexpectedEof
}

fn warn_unfinalized_trace(line_num: usize, records_read: usize) {
    tracing::warn!(
        line = line_num,
        records = records_read,
        "trace stream ended without a gzip trailer (live/unfinalized capture); \
         using records read so far"
    );
}

/// Validate that a record has ITL data when output_tokens > 1, and that any
/// per-gap context arrays line up with the gap array.
fn validate_record(record: &TraceRecord, line_num: usize) -> Result<()> {
    if record.output_tokens > 1 && record.itl_ms.is_none() && record.itl_summary.is_none() {
        bail!(
            "line {line_num}: output_tokens={} but neither itl_ms nor itl_summary is present",
            record.output_tokens
        );
    }
    if let Some(ref ctx) = record.itl_ctx {
        let gaps = record.itl_ms.as_ref().map(Vec::len).unwrap_or(0);
        if ctx.num_running.len() != gaps || ctx.prefill_tokens.len() != gaps {
            bail!(
                "line {line_num}: itl_ctx arrays (running={}, prefill={}) must parallel itl_ms (len={gaps})",
                ctx.num_running.len(),
                ctx.prefill_tokens.len(),
            );
        }
    }
    if let Some(ref tokens) = record.itl_tokens {
        let gaps = record.itl_ms.as_ref().map(Vec::len).unwrap_or(0);
        if tokens.len() != gaps {
            bail!(
                "line {line_num}: itl_tokens (len={}) must parallel itl_ms (len={gaps})",
                tokens.len(),
            );
        }
        let chunk_total: u64 = tokens.iter().map(|&t| u64::from(t)).sum();
        if tokens.contains(&0) || chunk_total >= record.output_tokens as u64 {
            bail!(
                "line {line_num}: itl_tokens must be positive and sum below output_tokens={} \
                 (the first chunk owns the remainder), got sum={chunk_total}",
                record.output_tokens,
            );
        }
    }
    if let Some(ref ids) = record.output_token_ids
        && ids.len() != record.output_tokens
    {
        bail!(
            "line {line_num}: output_token_ids has {} ids but output_tokens={}",
            ids.len(),
            record.output_tokens,
        );
    }
    Ok(())
}

/// The canonical replay ordering of a trace: records that carry an arrival
/// time, sorted by it (stable, so finish-order ties keep file order). A
/// record's index in this subset is its replay identity - the arrival-replay
/// harness names request `i` `replay-{i}`, and `ReplayTokens` resolves the
/// trailing index back to the record. Both sides MUST use this function so
/// the mapping cannot drift.
pub fn replay_subset(records: Vec<TraceRecord>) -> Vec<TraceRecord> {
    let mut subset: Vec<TraceRecord> = records
        .into_iter()
        .filter(|r| r.arrival_ms.is_some())
        .collect();
    subset.sort_by(|a, b| {
        a.arrival_ms
            .partial_cmp(&b.arrival_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    subset
}

/// Write a trace meta line followed by records to a writer.
pub fn write_trace(
    writer: &mut impl Write,
    meta: &TraceMeta,
    records: &[TraceRecord],
) -> Result<()> {
    let wrapper = MetaWrapper { meta: meta.clone() };
    serde_json::to_writer(&mut *writer, &wrapper)?;
    writeln!(writer)?;
    for record in records {
        serde_json::to_writer(&mut *writer, record)?;
        writeln!(writer)?;
    }
    Ok(())
}

/// Append a single record to a writer (no meta line).
pub fn append_record(writer: &mut impl Write, record: &TraceRecord) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    fn sample_meta() -> TraceMeta {
        TraceMeta {
            model: Some("llama-7b".to_string()),
            gpu: Some("A100".to_string()),
            tp: Some(2),
            max_num_seqs: Some(128),
            source: Some("guidellm".to_string()),
            block_size: None,
            config_hash: None,
            extra: HashMap::new(),
        }
    }

    fn sample_records() -> Vec<TraceRecord> {
        vec![
            TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 20,
                output_tokens: 5,
                ttft_ms: 50.0,
                itl_ms: Some(vec![10.0, 12.0, 11.0, 9.0]),
                itl_summary: None,
                concurrency: 3,
                arrival_ms: Some(1234.5),
                itl_ctx: None,
                ..Default::default()
            },
            TraceRecord {
                prompt_tokens: 200,
                cached_tokens: 0,
                output_tokens: 1,
                ttft_ms: 80.0,
                itl_ms: None,
                itl_summary: None,
                concurrency: 1,
                arrival_ms: None,
                itl_ctx: None,
                ..Default::default()
            },
            TraceRecord {
                prompt_tokens: 50,
                cached_tokens: 0,
                output_tokens: 10,
                ttft_ms: 30.0,
                itl_ms: None,
                itl_summary: Some(ItlSummary {
                    mean_ms: 15.0,
                    count: 9,
                }),
                concurrency: 5,
                arrival_ms: None,
                itl_ctx: None,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn round_trip_with_meta() {
        let meta = sample_meta();
        let records = sample_records();
        let mut buf = Vec::new();
        write_trace(&mut buf, &meta, &records).unwrap();

        let (parsed_meta, parsed_records) = read_trace(io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_records, records);
    }

    #[test]
    fn read_trace_tolerates_unfinalized_gz() {
        use std::io::Read as _;

        // The tap flushes per record but only writes the gzip trailer on finish().
        // Captured live (tap still up under --no-cleanup), the trailer is absent.
        let meta = sample_meta();
        let records = sample_records();
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        write_trace(&mut enc, &meta, &records).unwrap();
        enc.flush().unwrap(); // sync flush, NO finish() -> no trailer
        let trailerless = enc.get_ref().clone();

        // It really is trailer-less: a raw decode to EOF errors (UnexpectedEof).
        let mut sink = Vec::new();
        let raw = MultiGzDecoder::new(trailerless.as_slice()).read_to_end(&mut sink);
        assert!(raw.is_err(), "expected unfinalized gz to error at EOF");

        // read_trace recovers every record despite the missing trailer.
        let (parsed_meta, parsed_records) = read_trace(io::BufReader::new(MultiGzDecoder::new(
            trailerless.as_slice(),
        )))
        .unwrap();
        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_records, records);
    }

    #[test]
    fn config_hash_round_trips() {
        let meta = TraceMeta {
            config_hash: Some("deadbeef".to_string()),
            ..sample_meta()
        };
        let mut buf = Vec::new();
        write_trace(&mut buf, &meta, &sample_records()).unwrap();
        // It serializes under the plain `config_hash` key (not nested in extra).
        let text = String::from_utf8(buf.clone()).unwrap();
        assert!(
            text.lines()
                .next()
                .unwrap()
                .contains("\"config_hash\":\"deadbeef\"")
        );

        let (parsed, _) = read_trace(io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(parsed.config_hash.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn read_meta_reads_header_only() {
        let meta = TraceMeta {
            config_hash: Some("abc123".to_string()),
            ..sample_meta()
        };
        let mut buf = Vec::new();
        write_trace(&mut buf, &meta, &sample_records()).unwrap();

        let parsed = read_meta(io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(parsed, meta);
    }

    #[test]
    fn read_meta_headerless_is_default() {
        // A trace whose first line is a record (no `{"meta": ...}`) has no header.
        let mut buf = Vec::new();
        write_trace(&mut buf, &TraceMeta::default(), &sample_records()).unwrap();
        let parsed = read_meta(io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(parsed, TraceMeta::default());
        assert!(parsed.config_hash.is_none());
    }

    #[test]
    fn read_meta_empty_is_default() {
        let parsed = read_meta(io::BufReader::new(b"".as_slice())).unwrap();
        assert_eq!(parsed, TraceMeta::default());
    }

    #[test]
    fn read_trace_meta_file_gz() {
        // The cheap file reader works through gzip too (config-hash check reads
        // the .gz traces stored in S3).
        let meta = TraceMeta {
            config_hash: Some("hash-in-gz".to_string()),
            ..sample_meta()
        };
        let path = std::env::temp_dir().join(format!(
            "inference-sim-meta-gz-test-{}.jsonl.gz",
            std::process::id()
        ));
        let mut writer = TraceWriter::create(&path).unwrap();
        write_trace(&mut writer, &meta, &sample_records()).unwrap();
        writer.finish().unwrap();

        let parsed = read_trace_meta(&path).unwrap();
        assert_eq!(parsed.config_hash.as_deref(), Some("hash-in-gz"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trip_without_meta() {
        let records = sample_records();
        let mut buf = Vec::new();
        // Write records directly, no meta wrapper.
        for record in &records {
            append_record(&mut buf, record).unwrap();
        }

        let (parsed_meta, parsed_records) = read_trace(io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(parsed_meta, TraceMeta::default());
        assert_eq!(parsed_records, records);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let meta = sample_meta();
        let records = sample_records();
        let mut buf = Vec::new();
        write_trace(&mut buf, &meta, &records).unwrap();
        // Insert blank lines.
        let text = String::from_utf8(buf).unwrap();
        let with_blanks = text.replace('\n', "\n\n");
        let (parsed_meta, parsed_records) =
            read_trace(io::BufReader::new(with_blanks.as_bytes())).unwrap();
        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_records, records);
    }

    #[test]
    fn empty_input_returns_default() {
        let (meta, records) = read_trace(io::BufReader::new(b"" as &[u8])).unwrap();
        assert_eq!(meta, TraceMeta::default());
        assert!(records.is_empty());
    }

    #[test]
    fn malformed_json_includes_line_number() {
        let input = b"{\"meta\": {}}\n{bad json}\n";
        let err = read_trace(io::BufReader::new(input.as_slice())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("line 2"), "error should mention line 2: {msg}");
    }

    #[test]
    fn missing_itl_with_multiple_output_tokens_is_error() {
        let input = br#"{"prompt_tokens":10,"output_tokens":5,"ttft_ms":10.0,"concurrency":1}"#;
        let err = read_trace(io::BufReader::new(input.as_slice())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("itl"), "error should mention itl: {msg}");
    }

    #[test]
    fn itl_ctx_round_trips_and_validates_lengths() {
        let input = br#"{"prompt_tokens":10,"output_tokens":3,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"itl_ctx":{"num_running":[4,4],"prefill_tokens":[0,800]},"concurrency":4}"#;
        let (_, records) = read_trace(io::BufReader::new(input.as_slice())).unwrap();
        let ctx = records[0].itl_ctx.as_ref().unwrap();
        assert_eq!(ctx.num_running, vec![4, 4]);
        assert_eq!(ctx.prefill_tokens, vec![0, 800]);

        // Context arrays that do not parallel itl_ms are rejected.
        let bad = br#"{"prompt_tokens":10,"output_tokens":3,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"itl_ctx":{"num_running":[4],"prefill_tokens":[0,800]},"concurrency":4}"#;
        let err = read_trace(io::BufReader::new(bad.as_slice())).unwrap_err();
        assert!(format!("{err:#}").contains("itl_ctx"));
    }

    #[test]
    fn itl_tokens_round_trips_and_validates() {
        // 1 (first chunk) + 4 + 1 = 6 output tokens over two gaps.
        let input = br#"{"prompt_tokens":10,"output_tokens":6,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"itl_tokens":[4,1],"concurrency":1}"#;
        let (_, records) = read_trace(io::BufReader::new(input.as_slice())).unwrap();
        assert_eq!(records[0].itl_tokens, Some(vec![4, 1]));

        // Absent itl_tokens parses as None (all chunks carried one token).
        let plain = br#"{"prompt_tokens":10,"output_tokens":3,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"concurrency":1}"#;
        let (_, records) = read_trace(io::BufReader::new(plain.as_slice())).unwrap();
        assert_eq!(records[0].itl_tokens, None);

        // Length must parallel itl_ms.
        let bad_len = br#"{"prompt_tokens":10,"output_tokens":6,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"itl_tokens":[4],"concurrency":1}"#;
        let err = read_trace(io::BufReader::new(bad_len.as_slice())).unwrap_err();
        assert!(format!("{err:#}").contains("itl_tokens"));

        // Chunk totals must leave at least one token for the first chunk.
        let bad_sum = br#"{"prompt_tokens":10,"output_tokens":5,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"itl_tokens":[4,1],"concurrency":1}"#;
        let err = read_trace(io::BufReader::new(bad_sum.as_slice())).unwrap_err();
        assert!(format!("{err:#}").contains("itl_tokens"));

        // Zero-token chunks cannot exist.
        let bad_zero = br#"{"prompt_tokens":10,"output_tokens":6,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"itl_tokens":[4,0],"concurrency":1}"#;
        let err = read_trace(io::BufReader::new(bad_zero.as_slice())).unwrap_err();
        assert!(format!("{err:#}").contains("itl_tokens"));
    }

    #[test]
    fn single_output_token_needs_no_itl() {
        let input = br#"{"prompt_tokens":10,"output_tokens":1,"ttft_ms":10.0,"concurrency":1}"#;
        let (_, records) = read_trace(io::BufReader::new(input.as_slice())).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].output_tokens, 1);
    }

    #[test]
    fn default_concurrency_is_one() {
        let input = br#"{"prompt_tokens":10,"output_tokens":1,"ttft_ms":10.0}"#;
        let (_, records) = read_trace(io::BufReader::new(input.as_slice())).unwrap();
        assert_eq!(records[0].concurrency, 1);
    }

    #[test]
    fn default_cached_tokens_is_zero() {
        let input = br#"{"prompt_tokens":10,"output_tokens":1,"ttft_ms":10.0}"#;
        let (_, records) = read_trace(io::BufReader::new(input.as_slice())).unwrap();
        assert_eq!(records[0].cached_tokens, 0);
    }

    #[test]
    fn meta_with_extra_fields() {
        let input = br#"{"meta":{"model":"llama","custom_key":"custom_value"}}"#;
        let (meta, _) = read_trace(io::BufReader::new(input.as_slice())).unwrap();
        assert_eq!(meta.model, Some("llama".to_string()));
        assert_eq!(
            meta.extra.get("custom_key"),
            Some(&serde_json::Value::String("custom_value".to_string()))
        );
    }

    #[test]
    fn append_record_format() {
        let record = TraceRecord {
            prompt_tokens: 42,
            cached_tokens: 0,
            output_tokens: 1,
            ttft_ms: 5.0,
            itl_ms: None,
            itl_summary: None,
            concurrency: 1,
            arrival_ms: None,
            itl_ctx: None,
            ..Default::default()
        };
        let mut buf = Vec::new();
        append_record(&mut buf, &record).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.ends_with('\n'));
        let parsed: TraceRecord = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(parsed, record);
    }

    #[test]
    fn gzip_trace_file_round_trips() {
        let meta = sample_meta();
        let records = sample_records();
        let path = std::env::temp_dir().join(format!(
            "inference-sim-trace-gz-test-{}.jsonl.gz",
            std::process::id()
        ));

        let mut writer = TraceWriter::create(&path).unwrap();
        write_trace(&mut writer, &meta, &records).unwrap();
        writer.finish().unwrap();

        // The bytes on disk are actually gzip, not plain JSONL.
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..2], &[0x1f, 0x8b], "gzip magic bytes");

        let (parsed_meta, parsed_records) = read_trace_file(&path).unwrap();
        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_records, records);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn plain_trace_file_round_trips() {
        let meta = sample_meta();
        let records = sample_records();
        let path = std::env::temp_dir().join(format!(
            "inference-sim-trace-plain-test-{}.jsonl",
            std::process::id()
        ));

        let mut writer = TraceWriter::create(&path).unwrap();
        write_trace(&mut writer, &meta, &records).unwrap();
        writer.finish().unwrap();

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw[0], b'{', "plain path stays uncompressed JSONL");

        let (parsed_meta, parsed_records) = read_trace_file(&path).unwrap();
        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_records, records);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn output_token_ids_and_finish_reason_round_trip() {
        let record = TraceRecord {
            prompt_tokens: 10,
            output_tokens: 3,
            ttft_ms: 12.0,
            itl_ms: Some(vec![5.0, 6.0]),
            output_token_ids: Some(vec![7, 8, 9]),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        };
        let mut buf = Vec::new();
        append_record(&mut buf, &record).unwrap();
        let text = String::from_utf8(buf).unwrap();
        // Wire form is human-readable lowercase, not the protocol's int repr.
        assert!(text.contains(r#""finish_reason":"stop""#), "{text}");
        assert!(text.contains(r#""output_token_ids":[7,8,9]"#), "{text}");

        let (_, records) = read_trace(io::BufReader::new(text.as_bytes())).unwrap();
        assert_eq!(records[0], record);
    }

    #[test]
    fn records_without_tokens_serialize_without_the_fields() {
        let mut buf = Vec::new();
        append_record(
            &mut buf,
            &TraceRecord {
                prompt_tokens: 10,
                output_tokens: 1,
                ttft_ms: 5.0,
                ..Default::default()
            },
        )
        .unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(!text.contains("output_token_ids"), "{text}");
        assert!(!text.contains("finish_reason"), "{text}");
    }

    #[test]
    fn output_token_ids_length_mismatch_is_rejected() {
        let input = br#"{"prompt_tokens":10,"output_tokens":3,"ttft_ms":10.0,"itl_ms":[5.0,6.0],"output_token_ids":[7,8]}"#;
        let err = read_trace(io::BufReader::new(input.as_slice())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("output_token_ids"), "{msg}");
    }

    #[test]
    fn replay_subset_orders_by_arrival_and_drops_unscheduled() {
        let rec = |arrival: Option<f64>, prompt: usize| TraceRecord {
            prompt_tokens: prompt,
            output_tokens: 1,
            ttft_ms: 1.0,
            arrival_ms: arrival,
            ..Default::default()
        };
        // Finish order on disk: arrivals 50, none, 10, 50 (tie keeps file order).
        let records = vec![
            rec(Some(50.0), 1),
            rec(None, 2),
            rec(Some(10.0), 3),
            rec(Some(50.0), 4),
        ];
        let subset = replay_subset(records);
        let prompts: Vec<usize> = subset.iter().map(|r| r.prompt_tokens).collect();
        assert_eq!(prompts, vec![3, 1, 4]);
    }

    #[test]
    fn block_hashes_agree_exactly_on_shared_prefixes() {
        let block = 4usize;
        let shared: Vec<u32> = (0..12).collect();

        // Same prefix, diverging after block 2: chains agree for 2 blocks.
        let mut a = shared.clone();
        a.extend([100, 101, 102, 103]);
        let mut b = shared.clone();
        b[8] = 999; // diverge inside block 2
        b.extend([100, 101, 102, 103]);

        let ha = prompt_block_hashes(&a, block).unwrap();
        let hb = prompt_block_hashes(&b, block).unwrap();
        assert_eq!(ha.len(), 4);
        assert_eq!(ha[..2], hb[..2], "shared prefix blocks must hash equal");
        assert_ne!(ha[2], hb[2], "divergent block must hash differently");
        // Chaining: identical block CONTENT after the divergence still differs,
        // because the previous hash is folded in.
        assert_ne!(ha[3], hb[3], "chains must stay diverged");

        // Identical prompts hash identically end to end.
        assert_eq!(prompt_block_hashes(&a, block).unwrap(), ha);
    }

    #[test]
    fn block_hashes_ignore_partial_tail() {
        let tokens: Vec<u32> = (0..10).collect();
        let hashes = prompt_block_hashes(&tokens, 4).unwrap();
        assert_eq!(hashes.len(), 2, "10 tokens = 2 full blocks of 4 + tail");
        assert!(prompt_block_hashes(&tokens[..3], 4).is_none());
        assert!(prompt_block_hashes(&tokens, 0).is_none());
    }
}
