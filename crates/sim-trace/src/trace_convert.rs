//! Convert guidellm benchmark reports to the vllm-vcr trace format.
//!
//! guidellm (v0.6.0, https://github.com/vllm-project/guidellm) serializes results as
//! `GenerativeBenchmarksReport` JSON. The per-request data lives at:
//!
//!   report.benchmarks[].requests.successful[] : GenerativeRequestStats
//!
//! Source files in guidellm v0.6.0 that define this schema:
//!   - src/guidellm/benchmark/schemas/generative/report.py   (GenerativeBenchmarksReport)
//!   - src/guidellm/benchmark/schemas/generative/benchmark.py (GenerativeBenchmark)
//!   - src/guidellm/schemas/request_stats.py                  (GenerativeRequestStats)
//!   - src/guidellm/schemas/info.py                           (RequestInfo, RequestTimings)
//!   - src/guidellm/schemas/request.py                        (UsageMetrics)
//!   - src/guidellm/scheduler/strategies.py                   (ConcurrentStrategy, etc.)
//!
//! We only parse the fields we need, letting serde skip the rest.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use anyhow::{Context as _, Result, bail};
use serde::Deserialize;

use crate::trace::{ItlSummary, TraceMeta, TraceRecord, read_trace, write_trace};

// guidellm JSON schema (minimal subset we actually use)

/// Top-level report container.
/// Source: src/guidellm/benchmark/schemas/generative/report.py
#[derive(Debug, Deserialize)]
pub struct GuidellmReport {
    #[serde(default)]
    pub args: Option<GuidellmArgs>,
    pub benchmarks: Vec<GuidellmBenchmark>,
}

/// Benchmark-level args (carries model/target info).
/// Source: src/guidellm/benchmark/schemas/generative/entrypoints.py
#[derive(Debug, Deserialize)]
pub struct GuidellmArgs {
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

/// One benchmark run in a sweep.
/// Source: src/guidellm/benchmark/schemas/generative/benchmark.py
#[derive(Debug, Deserialize)]
pub struct GuidellmBenchmark {
    pub config: GuidellmBenchmarkConfig,
    pub requests: GuidellmStatusBreakdown,
}

/// Strategy configuration carried on each benchmark.
/// Source: src/guidellm/benchmark/schemas/base.py (BenchmarkConfig)
#[derive(Debug, Deserialize)]
pub struct GuidellmBenchmarkConfig {
    pub strategy: GuidellmStrategy,
}

/// Scheduling strategy. We care about the type and, for concurrent, the stream count.
/// Source: src/guidellm/scheduler/strategies.py
#[derive(Debug, Deserialize)]
pub struct GuidellmStrategy {
    #[serde(rename = "type_")]
    pub type_name: String,
    /// Only present for ConcurrentStrategy.
    #[serde(default)]
    pub streams: Option<u64>,
    /// Only present for ThroughputStrategy.
    #[serde(default)]
    pub max_concurrency: Option<u64>,
}

/// Requests grouped by status. We only consume successful.
/// Source: src/guidellm/schemas/__init__.py (StatusBreakdown)
#[derive(Debug, Deserialize)]
pub struct GuidellmStatusBreakdown {
    #[serde(default)]
    pub successful: Vec<GuidellmRequestStats>,
}

/// Per-request stats from a successful generative request.
/// Source: src/guidellm/schemas/request_stats.py (GenerativeRequestStats)
///
/// guidellm serializes computed_fields into JSON, so prompt_tokens, output_tokens,
/// time_to_first_token_ms, and inter_token_latency_ms appear as regular keys.
#[derive(Debug, Deserialize)]
pub struct GuidellmRequestStats {
    /// Prompt (input) token count. Computed from input_metrics.total_tokens in guidellm.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Output token count. Computed from output_metrics.total_tokens (fallback:
    /// info.timings.token_iterations) in guidellm.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Time-to-first-token in milliseconds. Computed from
    /// (info.timings.first_token_iteration - info.timings.request_start) * 1000.
    #[serde(default)]
    pub time_to_first_token_ms: Option<f64>,
    /// Mean inter-token latency in milliseconds (excluding first token). Computed from
    /// (last_token_iteration - first_token_iteration) / (output_tokens - 1) * 1000.
    #[serde(default)]
    pub inter_token_latency_ms: Option<f64>,
}

/// Options for `from_guidellm` that override or supplement report metadata.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    pub model: Option<String>,
    pub gpu: Option<String>,
    pub tp: Option<u32>,
}

/// Parse a guidellm JSON report and produce trace (meta, records).
///
/// All benchmarks in the report are flattened into one trace. Concurrency is derived
/// from each benchmark's scheduling strategy:
///   - `concurrent` -> strategy.streams
///   - `throughput` -> strategy.max_concurrency (if set), else 1 with warning
///   - `synchronous` -> 1
///   - anything else -> 1 with warning
pub fn convert_guidellm(
    report_json: &str,
    opts: &ConvertOptions,
) -> Result<(TraceMeta, Vec<TraceRecord>)> {
    let report: GuidellmReport =
        serde_json::from_str(report_json).context("parsing guidellm report JSON")?;

    if report.benchmarks.is_empty() {
        bail!("guidellm report contains no benchmarks");
    }

    // Build meta from report args + overrides.
    let report_model = report
        .args
        .as_ref()
        .and_then(|a| a.model.clone())
        .or_else(|| report.args.as_ref().and_then(|a| a.target.clone()));

    let meta = TraceMeta {
        model: opts.model.clone().or(report_model),
        gpu: opts.gpu.clone(),
        tp: opts.tp,
        max_num_seqs: None,
        source: Some("guidellm".to_string()),
        block_size: None,
        config_hash: None,
        vllm_version: None,
        ready_response_hex: None,
        extra: HashMap::new(),
    };

    let mut records = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    for (bench_idx, bench) in report.benchmarks.iter().enumerate() {
        let concurrency = derive_concurrency(&bench.config.strategy, bench_idx, &mut warnings);

        for (req_idx, req) in bench.requests.successful.iter().enumerate() {
            let ctx = || format!("benchmark[{bench_idx}].requests.successful[{req_idx}]");

            let prompt_tokens = req
                .prompt_tokens
                .with_context(ctx)
                .context("prompt_tokens is null")?;
            let output_tokens = req
                .output_tokens
                .with_context(ctx)
                .context("output_tokens is null")?;
            let ttft_ms = req
                .time_to_first_token_ms
                .with_context(ctx)
                .context("time_to_first_token_ms is null")?;

            // ITL: guidellm only provides the mean inter_token_latency_ms, not per-token
            // timings. Build an ItlSummary when output_tokens > 1.
            let itl_summary = if output_tokens > 1 {
                let mean_itl = req.inter_token_latency_ms.unwrap_or(0.0);
                Some(ItlSummary {
                    mean_ms: mean_itl,
                    count: (output_tokens - 1) as usize,
                })
            } else {
                None
            };

            records.push(TraceRecord {
                prompt_tokens: prompt_tokens as usize,
                cached_tokens: 0,
                output_tokens: output_tokens as usize,
                ttft_ms,
                itl_ms: None,
                itl_summary,
                concurrency,
                arrival_ms: None,
                itl_ctx: None,
                ..Default::default()
            });
        }
    }

    for w in &warnings {
        tracing::warn!("{w}");
    }

    Ok((meta, records))
}

/// Derive concurrency from the benchmark's scheduling strategy.
fn derive_concurrency(
    strategy: &GuidellmStrategy,
    bench_idx: usize,
    warnings: &mut Vec<String>,
) -> u64 {
    match strategy.type_name.as_str() {
        "concurrent" => strategy.streams.unwrap_or_else(|| {
            warnings.push(format!(
                "benchmark[{bench_idx}]: concurrent strategy missing 'streams', defaulting to 1"
            ));
            1
        }),
        "synchronous" => 1,
        "throughput" => strategy.max_concurrency.unwrap_or_else(|| {
            warnings.push(format!(
                "benchmark[{bench_idx}]: throughput strategy missing 'max_concurrency', defaulting to 1"
            ));
            1
        }),
        other => {
            warnings.push(format!(
                "benchmark[{bench_idx}]: unknown strategy type '{other}', defaulting concurrency to 1"
            ));
            1
        }
    }
}

/// Per-concurrency-bucket statistics.
#[derive(Debug)]
pub struct BucketStats {
    pub concurrency: u64,
    pub count: usize,
    pub prompt_min: usize,
    pub prompt_max: usize,
    pub output_min: usize,
    pub output_max: usize,
    pub ttft_p50: f64,
    pub ttft_p90: f64,
    pub ttft_p99: f64,
    pub itl_p50: f64,
    pub itl_p90: f64,
    pub itl_p99: f64,
}

/// Compute summary stats from a trace file's records, bucketed by concurrency.
pub fn summarize_trace(reader: impl BufRead) -> Result<(TraceMeta, Vec<BucketStats>)> {
    let (meta, records) = read_trace(reader)?;

    if records.is_empty() {
        bail!("trace contains no records");
    }

    // Group by concurrency.
    let mut buckets: HashMap<u64, Vec<&TraceRecord>> = HashMap::new();
    for r in &records {
        buckets.entry(r.concurrency).or_default().push(r);
    }

    let mut sorted_keys: Vec<u64> = buckets.keys().copied().collect();
    sorted_keys.sort();

    let mut stats = Vec::new();
    for conc in sorted_keys {
        let recs = &buckets[&conc];

        let mut ttfts: Vec<f64> = recs.iter().map(|r| r.ttft_ms).collect();
        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Pool per-token ITL samples, matching calibrate::pool_samples: full arrays
        // contribute every token's gap; summary-only records (e.g. guidellm, which has
        // no per-token timings) contribute their mean once per token so weighting is
        // consistent. Percentiles below are therefore per-token, not per-request.
        let mut itls: Vec<f64> = Vec::new();
        for r in recs {
            if let Some(ref arr) = r.itl_ms {
                itls.extend(arr.iter().copied());
            } else if let Some(ref s) = r.itl_summary {
                for _ in 0..s.count {
                    itls.push(s.mean_ms);
                }
            }
        }
        itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let prompt_tokens: Vec<usize> = recs.iter().map(|r| r.prompt_tokens).collect();
        let output_tokens: Vec<usize> = recs.iter().map(|r| r.output_tokens).collect();

        stats.push(BucketStats {
            concurrency: conc,
            count: recs.len(),
            prompt_min: prompt_tokens.iter().copied().min().unwrap_or(0),
            prompt_max: prompt_tokens.iter().copied().max().unwrap_or(0),
            output_min: output_tokens.iter().copied().min().unwrap_or(0),
            output_max: output_tokens.iter().copied().max().unwrap_or(0),
            ttft_p50: percentile(&ttfts, 50.0),
            ttft_p90: percentile(&ttfts, 90.0),
            ttft_p99: percentile(&ttfts, 99.0),
            itl_p50: percentile(&itls, 50.0),
            itl_p90: percentile(&itls, 90.0),
            itl_p99: percentile(&itls, 99.0),
        });
    }

    Ok((meta, stats))
}

/// Write the summary as aligned plain text.
pub fn write_summary(
    writer: &mut impl Write,
    meta: &TraceMeta,
    stats: &[BucketStats],
) -> Result<()> {
    if let Some(ref m) = meta.model {
        writeln!(writer, "model: {m}")?;
    }
    if let Some(ref g) = meta.gpu {
        writeln!(writer, "gpu:   {g}")?;
    }
    if let Some(tp) = meta.tp {
        writeln!(writer, "tp:    {tp}")?;
    }
    if let Some(ref s) = meta.source {
        writeln!(writer, "src:   {s}")?;
    }
    writeln!(writer)?;

    // Header
    writeln!(
        writer,
        "{:>5}  {:>6}  {:>14}  {:>14}  {:>28}  {:>28}",
        "conc",
        "count",
        "prompt_tok",
        "output_tok",
        "ttft_ms (p50/p90/p99)",
        "itl_ms (p50/p90/p99)"
    )?;
    writeln!(
        writer,
        "{:>5}  {:>6}  {:>14}  {:>14}  {:>28}  {:>28}",
        "-----",
        "------",
        "--------------",
        "--------------",
        "----------------------------",
        "----------------------------"
    )?;

    for s in stats {
        writeln!(
            writer,
            "{:>5}  {:>6}  {:>6}-{:<6}  {:>6}-{:<6}  {:>8.1}/{:>8.1}/{:>8.1}  {:>8.1}/{:>8.1}/{:>8.1}",
            s.concurrency,
            s.count,
            s.prompt_min,
            s.prompt_max,
            s.output_min,
            s.output_max,
            s.ttft_p50,
            s.ttft_p90,
            s.ttft_p99,
            s.itl_p50,
            s.itl_p90,
            s.itl_p99,
        )?;
    }

    Ok(())
}

/// Nearest-rank percentile on a sorted slice.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = idx.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Write the conversion result (meta + records) to a JSONL writer.
pub fn write_conversion(
    writer: &mut impl Write,
    meta: &TraceMeta,
    records: &[TraceRecord],
) -> Result<()> {
    write_trace(writer, meta, records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report_json() -> &'static str {
        // Minimal guidellm report matching v0.6.0 schema.
        // Source files: see module doc.
        r#"{
            "args": {
                "target": "http://localhost:8000/v1",
                "model": "meta-llama/Llama-3-8B"
            },
            "benchmarks": [
                {
                    "config": {
                        "strategy": {
                            "type_": "concurrent",
                            "streams": 4
                        }
                    },
                    "requests": {
                        "successful": [
                            {
                                "prompt_tokens": 128,
                                "output_tokens": 50,
                                "time_to_first_token_ms": 45.2,
                                "inter_token_latency_ms": 12.5
                            },
                            {
                                "prompt_tokens": 256,
                                "output_tokens": 1,
                                "time_to_first_token_ms": 80.0,
                                "inter_token_latency_ms": null
                            }
                        ],
                        "incomplete": [],
                        "errored": []
                    }
                },
                {
                    "config": {
                        "strategy": {
                            "type_": "synchronous"
                        }
                    },
                    "requests": {
                        "successful": [
                            {
                                "prompt_tokens": 64,
                                "output_tokens": 20,
                                "time_to_first_token_ms": 30.0,
                                "inter_token_latency_ms": 8.0
                            }
                        ],
                        "incomplete": [],
                        "errored": []
                    }
                }
            ]
        }"#
    }

    #[test]
    fn convert_basic_report() {
        let (meta, records) =
            convert_guidellm(sample_report_json(), &ConvertOptions::default()).unwrap();
        assert_eq!(meta.source.as_deref(), Some("guidellm"));
        assert_eq!(meta.model.as_deref(), Some("meta-llama/Llama-3-8B"));
        assert_eq!(records.len(), 3);

        // First record from concurrent benchmark.
        assert_eq!(records[0].prompt_tokens, 128);
        assert_eq!(records[0].output_tokens, 50);
        assert!((records[0].ttft_ms - 45.2).abs() < 0.01);
        assert_eq!(records[0].concurrency, 4);
        assert_eq!(records[0].cached_tokens, 0);
        let itl = records[0].itl_summary.as_ref().unwrap();
        assert!((itl.mean_ms - 12.5).abs() < 0.01);
        assert_eq!(itl.count, 49);

        // Single-token output has no ITL.
        assert_eq!(records[1].output_tokens, 1);
        assert!(records[1].itl_summary.is_none());
        assert!(records[1].itl_ms.is_none());

        // Third record from synchronous benchmark.
        assert_eq!(records[2].concurrency, 1);
        assert_eq!(records[2].prompt_tokens, 64);
    }

    #[test]
    fn convert_with_overrides() {
        let opts = ConvertOptions {
            model: Some("my-model".to_string()),
            gpu: Some("H100".to_string()),
            tp: Some(4),
        };
        let (meta, _) = convert_guidellm(sample_report_json(), &opts).unwrap();
        assert_eq!(meta.model.as_deref(), Some("my-model"));
        assert_eq!(meta.gpu.as_deref(), Some("H100"));
        assert_eq!(meta.tp, Some(4));
    }

    #[test]
    fn convert_empty_benchmarks_is_error() {
        let json = r#"{"benchmarks": []}"#;
        let err = convert_guidellm(json, &ConvertOptions::default()).unwrap_err();
        assert!(format!("{err:#}").contains("no benchmarks"));
    }

    #[test]
    fn convert_missing_prompt_tokens_is_error() {
        let json = r#"{
            "benchmarks": [{
                "config": {"strategy": {"type_": "synchronous"}},
                "requests": {"successful": [{"output_tokens": 10, "time_to_first_token_ms": 5.0, "inter_token_latency_ms": 2.0}]}
            }]
        }"#;
        let err = convert_guidellm(json, &ConvertOptions::default()).unwrap_err();
        assert!(format!("{err:#}").contains("prompt_tokens"));
    }

    #[test]
    fn round_trip_conversion() {
        let (meta, records) =
            convert_guidellm(sample_report_json(), &ConvertOptions::default()).unwrap();
        let mut buf = Vec::new();
        write_conversion(&mut buf, &meta, &records).unwrap();

        let (parsed_meta, parsed_records) =
            read_trace(std::io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_records.len(), records.len());
        assert_eq!(parsed_records, records);
    }

    #[test]
    fn summarize_basic() {
        let (meta, records) =
            convert_guidellm(sample_report_json(), &ConvertOptions::default()).unwrap();
        let mut buf = Vec::new();
        write_trace(&mut buf, &meta, &records).unwrap();

        let (smeta, stats) = summarize_trace(std::io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(smeta, meta);
        // Two concurrency buckets: 1 and 4.
        assert_eq!(stats.len(), 2);

        let bucket1 = stats.iter().find(|s| s.concurrency == 1).unwrap();
        assert_eq!(bucket1.count, 1);
        assert!((bucket1.ttft_p50 - 30.0).abs() < 0.01);

        let bucket4 = stats.iter().find(|s| s.concurrency == 4).unwrap();
        assert_eq!(bucket4.count, 2);
    }

    #[test]
    fn percentile_edge_cases() {
        assert!((percentile(&[], 50.0) - 0.0).abs() < f64::EPSILON);
        assert!((percentile(&[42.0], 50.0) - 42.0).abs() < f64::EPSILON);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0], 50.0) - 2.0).abs() < f64::EPSILON);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0], 99.0) - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn throughput_strategy_concurrency() {
        let json = r#"{
            "benchmarks": [{
                "config": {"strategy": {"type_": "throughput", "max_concurrency": 8}},
                "requests": {"successful": [{"prompt_tokens": 10, "output_tokens": 1, "time_to_first_token_ms": 5.0}]}
            }]
        }"#;
        let (_, records) = convert_guidellm(json, &ConvertOptions::default()).unwrap();
        assert_eq!(records[0].concurrency, 8);
    }

    #[test]
    fn unknown_strategy_defaults_to_one() {
        let json = r#"{
            "benchmarks": [{
                "config": {"strategy": {"type_": "async_poisson"}},
                "requests": {"successful": [{"prompt_tokens": 10, "output_tokens": 1, "time_to_first_token_ms": 5.0}]}
            }]
        }"#;
        let (_, records) = convert_guidellm(json, &ConvertOptions::default()).unwrap();
        assert_eq!(records[0].concurrency, 1);
    }

    #[test]
    fn write_summary_output() {
        let (meta, records) =
            convert_guidellm(sample_report_json(), &ConvertOptions::default()).unwrap();
        let mut buf = Vec::new();
        write_trace(&mut buf, &meta, &records).unwrap();

        let (smeta, stats) = summarize_trace(std::io::BufReader::new(buf.as_slice())).unwrap();
        let mut out = Vec::new();
        write_summary(&mut out, &smeta, &stats).unwrap();
        let text = String::from_utf8(out).unwrap();
        // Should contain model name and concurrency values.
        assert!(text.contains("meta-llama/Llama-3-8B"));
        assert!(text.contains("guidellm"));
    }
}
