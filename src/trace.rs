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
use std::io::{BufRead, Write};

use anyhow::{Context as _, Result, bail};
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
}

fn default_concurrency() -> u64 {
    1
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
            Some(line) => {
                let line = line.with_context(|| format!("reading line {line_num}"))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                break Some((line_num, trimmed.to_string()));
            }
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

    for line in lines {
        line_num += 1;
        let line = line.with_context(|| format!("reading line {line_num}"))?;
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
    Ok(())
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
        };
        let mut buf = Vec::new();
        append_record(&mut buf, &record).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.ends_with('\n'));
        let parsed: TraceRecord = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(parsed, record);
    }
}
