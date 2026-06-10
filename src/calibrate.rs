//! Calibration and comparison harness for trace-replay vs knob-fit latency models.
//!
//! Proves two claims:
//!   1. `TraceLatency` replay reproduces a source trace's latency quantiles (within tolerance).
//!   2. `KnobLatency` structurally cannot reproduce heavy tails: its `[0.3*mean, 1.7*mean]`
//!      clamp caps p99/p50 at roughly 1.7 for any knob settings.
//!
//! Three entry points, each exposed as a subcommand on the `inference-sim-trace` binary:
//!   - `gen_demo`: synthesize a heavy-tailed demo trace (lognormal TTFT/ITL).
//!   - `calibrate`: model-level quantile comparison (source vs replay vs knob-fit).
//!   - `calibrate_e2e`: wire-level proof using the real simulator in-process.

use std::io::Write;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use rand::Rng;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::latency::{
    CONCURRENCY_RANGES, DecodePacing, FirstTokenCtx, KnobLatency, LatencyModel,
    NUM_CONCURRENCY_BUCKETS, TraceLatency, concurrency_bucket, random_norm,
};
use crate::trace::{TraceMeta, TraceRecord, read_trace, write_trace};

// ---------------------------------------------------------------------------
// Quantile helpers
// ---------------------------------------------------------------------------

/// Nearest-rank percentile on a sorted slice. Returns 0.0 for empty input.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = idx.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Quantile triplet (p50, p90, p99).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Quantiles {
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

impl Quantiles {
    pub fn from_sorted(sorted: &[f64]) -> Self {
        Self {
            p50: percentile(sorted, 50.0),
            p90: percentile(sorted, 90.0),
            p99: percentile(sorted, 99.0),
        }
    }

    /// p99/p50 ratio. Returns 0.0 if p50 is zero.
    pub fn tail_ratio(&self) -> f64 {
        if self.p50 == 0.0 {
            0.0
        } else {
            self.p99 / self.p50
        }
    }

    /// Max relative error vs another quantile triplet.
    pub fn max_relative_error(&self, other: &Quantiles) -> f64 {
        let mut max_err = 0.0_f64;
        for (a, b) in [
            (self.p50, other.p50),
            (self.p90, other.p90),
            (self.p99, other.p99),
        ] {
            if a == 0.0 && b == 0.0 {
                continue;
            }
            let denom = a.max(b).max(f64::MIN_POSITIVE);
            let err = (a - b).abs() / denom;
            max_err = max_err.max(err);
        }
        max_err
    }
}

// ---------------------------------------------------------------------------
// Bucket-level stats
// ---------------------------------------------------------------------------

/// Stats for one concurrency bucket from one source (source trace, replay, or knob-fit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketQuantiles {
    pub concurrency_label: String,
    pub count: usize,
    pub ttft: Quantiles,
    pub itl: Quantiles,
}

/// Full calibration report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// What the middle (measured) column is: "replay" for trace replay, "knobfit" when the
    /// e2e harness ran the sim with fitted knobs instead.
    pub measured_label: String,
    pub source: Vec<BucketQuantiles>,
    pub replay: Vec<BucketQuantiles>,
    pub knobfit: Option<Vec<BucketQuantiles>>,
    /// Per-request decode-total comparison (model-level calibration only).
    #[serde(default)]
    pub request_total: Option<RequestTotalReport>,
    pub verdict: Verdict,
}

/// Per-request decode-total (sum of a request's ITL gaps) quantiles, pooled across
/// all multi-token records.
///
/// Per-token quantiles alone cannot catch a model that loses within-request
/// correlation: i.i.d. per-token sampling reproduces them while compressing
/// per-request totals (a sum of n independent draws concentrates ~sqrt(n) tighter
/// than n correlated gaps). Hierarchical donor-based pacing preserves the
/// correlation; this metric gates `replay_pass` to keep it that way.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTotalReport {
    /// Number of multi-token requests contributing totals.
    pub count: usize,
    pub source: Quantiles,
    pub replay: Quantiles,
    pub max_error: f64,
    /// Whether `max_error` is within the run's tolerance (part of `replay_pass`).
    pub within_tolerance: bool,
}

/// The verdict block printed at the bottom of the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub source_ttft_tail_ratio: f64,
    pub source_itl_tail_ratio: f64,
    pub replay_ttft_tail_ratio: f64,
    pub replay_itl_tail_ratio: f64,
    pub knobfit_ttft_tail_ratio: Option<f64>,
    pub knobfit_itl_tail_ratio: Option<f64>,
    pub replay_ttft_max_error: f64,
    pub replay_itl_max_error: f64,
    pub replay_pass: bool,
    pub knobfit_tail_capped: bool,
    pub tolerance: f64,
}

// ---------------------------------------------------------------------------
// gen-demo: synthesize a heavy-tailed demo trace
// ---------------------------------------------------------------------------

/// Lognormal sample: exp(N(mu, sigma)).
fn lognormal(rng: &mut StdRng, mu: f64, sigma: f64) -> f64 {
    random_norm(rng, mu, sigma).exp()
}

/// Parameters for one prompt-length group in the demo trace.
struct DemoGroup {
    prompt_tokens: usize,
    concurrency: u64,
    /// Lognormal mu for TTFT.
    ttft_mu: f64,
    /// Lognormal sigma for TTFT.
    ttft_sigma: f64,
    /// Lognormal mu for ITL.
    itl_mu: f64,
    /// Lognormal sigma for ITL.
    itl_sigma: f64,
    /// Output token range [lo, hi].
    output_range: (usize, usize),
}

/// Generate a deterministic, heavy-tailed demo trace.
///
/// Lognormal parameters are chosen so the pooled p99/p50 ratio exceeds 3.0.
///
/// Lognormal params chosen after experimentation:
///   TTFT: mu=3.5 sigma=0.9 gives mean ~50ms, heavy right tail.
///   ITL:  mu=2.0 sigma=0.8 gives mean ~11ms, heavy right tail.
/// The sigma values produce p99/p50 well above 3.0 when pooled across groups.
pub fn gen_demo(num_records: usize, seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // 4 groups across 3 concurrency levels, mixing prompt lengths.
    let groups = [
        DemoGroup {
            prompt_tokens: 64,
            concurrency: 1,
            ttft_mu: 3.2,
            ttft_sigma: 0.9,
            itl_mu: 1.8,
            itl_sigma: 0.8,
            output_range: (8, 24),
        },
        DemoGroup {
            prompt_tokens: 256,
            concurrency: 4,
            ttft_mu: 3.8,
            ttft_sigma: 0.9,
            itl_mu: 2.0,
            itl_sigma: 0.8,
            output_range: (10, 32),
        },
        DemoGroup {
            prompt_tokens: 512,
            concurrency: 4,
            ttft_mu: 4.0,
            ttft_sigma: 0.85,
            itl_mu: 2.2,
            itl_sigma: 0.75,
            output_range: (12, 28),
        },
        DemoGroup {
            prompt_tokens: 1024,
            concurrency: 16,
            ttft_mu: 4.3,
            ttft_sigma: 0.95,
            itl_mu: 2.3,
            itl_sigma: 0.8,
            output_range: (8, 20),
        },
    ];

    let records_per_group = num_records / groups.len();
    let remainder = num_records % groups.len();

    let mut records = Vec::with_capacity(num_records);

    for (gi, group) in groups.iter().enumerate() {
        let n = records_per_group + if gi < remainder { 1 } else { 0 };
        for _ in 0..n {
            let ttft_ms = lognormal(&mut rng, group.ttft_mu, group.ttft_sigma).max(1.0);

            let out_range = group.output_range.1 - group.output_range.0 + 1;
            let output_tokens = group.output_range.0 + (rng.random_range(0..out_range));

            let mut itl_ms = Vec::with_capacity(output_tokens.saturating_sub(1));
            for _ in 0..output_tokens.saturating_sub(1) {
                itl_ms.push(lognormal(&mut rng, group.itl_mu, group.itl_sigma).max(0.5));
            }

            records.push(TraceRecord {
                prompt_tokens: group.prompt_tokens,
                cached_tokens: 0,
                output_tokens,
                ttft_ms,
                itl_ms: if itl_ms.is_empty() {
                    None
                } else {
                    Some(itl_ms)
                },
                itl_summary: None,
                concurrency: group.concurrency,
                arrival_ms: None,
                itl_ctx: None,
            });
        }
    }

    let meta = TraceMeta {
        source: Some("gen-demo".to_string()),
        ..TraceMeta::default()
    };

    (meta, records)
}

/// Generate a demo trace with small magnitudes suitable for fast e2e testing.
/// TTFT ~15-40ms, ITL ~3-10ms. Same structure as gen_demo but dialed down.
pub fn gen_demo_fast(num_records: usize, seed: u64) -> (TraceMeta, Vec<TraceRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let groups = [
        DemoGroup {
            prompt_tokens: 32,
            concurrency: 1,
            ttft_mu: 2.7, // exp(2.7) ~ 15ms
            ttft_sigma: 0.5,
            itl_mu: 1.1, // exp(1.1) ~ 3ms
            itl_sigma: 0.5,
            output_range: (4, 8),
        },
        DemoGroup {
            prompt_tokens: 64,
            concurrency: 4,
            ttft_mu: 3.0, // exp(3.0) ~ 20ms
            ttft_sigma: 0.5,
            itl_mu: 1.4, // exp(1.4) ~ 4ms
            itl_sigma: 0.5,
            output_range: (4, 8),
        },
        DemoGroup {
            prompt_tokens: 128,
            concurrency: 4,
            ttft_mu: 3.2, // exp(3.2) ~ 25ms
            ttft_sigma: 0.4,
            itl_mu: 1.6, // exp(1.6) ~ 5ms
            itl_sigma: 0.4,
            output_range: (4, 8),
        },
        DemoGroup {
            prompt_tokens: 256,
            concurrency: 16,
            ttft_mu: 3.5, // exp(3.5) ~ 33ms
            ttft_sigma: 0.5,
            itl_mu: 1.8, // exp(1.8) ~ 6ms
            itl_sigma: 0.5,
            output_range: (4, 8),
        },
    ];

    let records_per_group = num_records / groups.len();
    let remainder = num_records % groups.len();

    let mut records = Vec::with_capacity(num_records);

    for (gi, group) in groups.iter().enumerate() {
        let n = records_per_group + if gi < remainder { 1 } else { 0 };
        for _ in 0..n {
            let ttft_ms = lognormal(&mut rng, group.ttft_mu, group.ttft_sigma).max(1.0);
            let out_range = group.output_range.1 - group.output_range.0 + 1;
            let output_tokens = group.output_range.0 + (rng.random_range(0..out_range));

            let mut itl_ms = Vec::with_capacity(output_tokens.saturating_sub(1));
            for _ in 0..output_tokens.saturating_sub(1) {
                itl_ms.push(lognormal(&mut rng, group.itl_mu, group.itl_sigma).max(0.5));
            }

            records.push(TraceRecord {
                prompt_tokens: group.prompt_tokens,
                cached_tokens: 0,
                output_tokens,
                ttft_ms,
                itl_ms: if itl_ms.is_empty() {
                    None
                } else {
                    Some(itl_ms)
                },
                itl_summary: None,
                concurrency: group.concurrency,
                arrival_ms: None,
                itl_ctx: None,
            });
        }
    }

    let meta = TraceMeta {
        source: Some("gen-demo-fast".to_string()),
        ..TraceMeta::default()
    };

    (meta, records)
}

/// Write a generated demo trace to a file.
pub fn write_demo_trace(path: &Path, meta: &TraceMeta, records: &[TraceRecord]) -> Result<()> {
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    write_trace(&mut writer, meta, records)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// calibrate: model-level comparison
// ---------------------------------------------------------------------------

/// Collect TTFT and ITL samples from trace records, grouped by concurrency bucket.
pub fn source_samples_by_bucket(records: &[TraceRecord]) -> Vec<(String, Vec<f64>, Vec<f64>)> {
    // Group by concurrency bucket index.
    let mut buckets: Vec<(Vec<f64>, Vec<f64>)> = (0..NUM_CONCURRENCY_BUCKETS)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();

    for r in records {
        let cb = concurrency_bucket(r.concurrency);
        buckets[cb].0.push(r.ttft_ms);

        if let Some(ref itls) = r.itl_ms {
            buckets[cb].1.extend(itls.iter().copied());
        } else if let Some(ref summary) = r.itl_summary {
            for _ in 0..summary.count {
                buckets[cb].1.push(summary.mean_ms);
            }
        }
    }

    let mut result = Vec::new();
    for (i, (mut ttfts, mut itls)) in buckets.into_iter().enumerate() {
        if ttfts.is_empty() {
            continue;
        }
        let (lo, hi) = CONCURRENCY_RANGES[i];
        let label = if hi == u64::MAX {
            format!("{lo}+")
        } else if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}-{hi}")
        };
        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.push((label, ttfts, itls));
    }
    result
}

/// Pool all TTFT and ITL samples across records.
pub fn pool_samples(records: &[TraceRecord]) -> (Vec<f64>, Vec<f64>) {
    let mut ttfts = Vec::new();
    let mut itls = Vec::new();
    for r in records {
        ttfts.push(r.ttft_ms);
        if let Some(ref arr) = r.itl_ms {
            itls.extend(arr.iter().copied());
        } else if let Some(ref s) = r.itl_summary {
            for _ in 0..s.count {
                itls.push(s.mean_ms);
            }
        }
    }
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (ttfts, itls)
}

/// Compute quantiles per bucket plus a pooled row from sorted sample vecs.
pub fn quantiles_from_buckets(
    buckets: &[(String, Vec<f64>, Vec<f64>)],
    pooled_ttft: &[f64],
    pooled_itl: &[f64],
) -> Vec<BucketQuantiles> {
    let mut out: Vec<BucketQuantiles> = buckets
        .iter()
        .map(|(label, ttfts, itls)| BucketQuantiles {
            concurrency_label: label.clone(),
            count: ttfts.len(),
            ttft: Quantiles::from_sorted(ttfts),
            itl: Quantiles::from_sorted(itls),
        })
        .collect();

    out.push(BucketQuantiles {
        concurrency_label: "pooled".to_string(),
        count: pooled_ttft.len(),
        ttft: Quantiles::from_sorted(pooled_ttft),
        itl: Quantiles::from_sorted(pooled_itl),
    });

    out
}

/// Fit a KnobLatency model from source trace statistics (steelmanned: true mean + std dev).
pub fn fit_knob_from_trace(records: &[TraceRecord]) -> KnobLatency {
    let (ttfts, itls) = pool_samples(records);

    let ttft_mean = if ttfts.is_empty() {
        0.0
    } else {
        ttfts.iter().sum::<f64>() / ttfts.len() as f64
    };
    let ttft_std = std_dev(&ttfts, ttft_mean);

    let itl_mean = if itls.is_empty() {
        0.0
    } else {
        itls.iter().sum::<f64>() / itls.len() as f64
    };
    let itl_std = std_dev(&itls, itl_mean);

    KnobLatency {
        time_to_first_token: ttft_mean as u64,
        time_to_first_token_std_dev: ttft_std as u64,
        inter_token_latency: itl_mean as u64,
        inter_token_latency_std_dev: itl_std as u64,
        prefill_overhead: 0,
        prefill_time_per_token: 0,
        prefill_time_std_dev: 0,
        kv_cache_transfer_latency: 0,
        kv_cache_transfer_latency_std_dev: 0,
        kv_cache_transfer_time_per_token: 0,
        kv_cache_transfer_time_std_dev: 0,
        time_factor_under_load: 1.0,
        max_num_seqs: 128,
    }
}

fn std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Labeled buckets of (label, ttft_samples, itl_samples) plus pooled ttft and itl vecs.
type BucketedSamples = (Vec<(String, Vec<f64>, Vec<f64>)>, Vec<f64>, Vec<f64>);

/// Sample TTFT and ITL from a model, bucketed to match the source trace structure.
///
/// For each source record, draws `samples_per_record` TTFT samples using matching
/// (uncached prompt, concurrency) context. ITL samples are drawn per-concurrency.
fn sample_model_to_buckets(
    model: &dyn LatencyModel,
    records: &[TraceRecord],
    samples_per_record: usize,
    seed: u64,
) -> BucketedSamples {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut bucket_ttfts: Vec<Vec<f64>> =
        (0..NUM_CONCURRENCY_BUCKETS).map(|_| Vec::new()).collect();
    let mut bucket_itls: Vec<Vec<f64>> = (0..NUM_CONCURRENCY_BUCKETS).map(|_| Vec::new()).collect();

    for record in records {
        let cb = concurrency_bucket(record.concurrency);
        let ctx = FirstTokenCtx {
            num_prompt_tokens: record.prompt_tokens,
            num_cached_tokens: record.cached_tokens,
            do_remote_prefill: false,
            num_running: record.concurrency,
        };

        for _ in 0..samples_per_record {
            let ttft = model.first_token_delay(&mut rng, &ctx);
            bucket_ttfts[cb].push(ttft.as_secs_f64() * 1000.0);

            let itl = model.inter_token_delay(&mut rng, record.concurrency);
            bucket_itls[cb].push(itl.as_secs_f64() * 1000.0);
        }
    }

    let mut pooled_ttft = Vec::new();
    let mut pooled_itl = Vec::new();
    let mut result = Vec::new();

    for (i, (mut ttfts, mut itls)) in bucket_ttfts.into_iter().zip(bucket_itls).enumerate() {
        if ttfts.is_empty() {
            continue;
        }
        pooled_ttft.extend(ttfts.iter().copied());
        pooled_itl.extend(itls.iter().copied());

        let (lo, hi) = CONCURRENCY_RANGES[i];
        let label = if hi == u64::MAX {
            format!("{lo}+")
        } else if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}-{hi}")
        };
        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.push((label, ttfts, itls));
    }

    pooled_ttft.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pooled_itl.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    (result, pooled_ttft, pooled_itl)
}

/// A KnobLatency that adds nothing: the TraceLatency fallback for traces with no
/// P/D transfer data, and the base for replay-only comparisons.
fn zero_knob() -> KnobLatency {
    KnobLatency {
        time_to_first_token: 0,
        time_to_first_token_std_dev: 0,
        inter_token_latency: 0,
        inter_token_latency_std_dev: 0,
        prefill_overhead: 0,
        prefill_time_per_token: 0,
        prefill_time_std_dev: 0,
        kv_cache_transfer_latency: 0,
        kv_cache_transfer_latency_std_dev: 0,
        kv_cache_transfer_time_per_token: 0,
        kv_cache_transfer_time_std_dev: 0,
        time_factor_under_load: 1.0,
        max_num_seqs: 128,
    }
}

/// Pooled TTFT/ITL sample arrays for one model, sorted ascending.
#[derive(Debug, Serialize)]
pub struct ModelSamples {
    pub ttft_ms: Vec<f64>,
    pub itl_ms: Vec<f64>,
}

/// Raw pooled samples behind a calibration run, for external plotting: the source
/// trace observations plus replay and knob-fit draws using the same seeds and
/// per-record contexts as `calibrate`.
#[derive(Debug, Serialize)]
pub struct SampleDump {
    pub source: ModelSamples,
    pub replay: ModelSamples,
    pub knobfit: ModelSamples,
}

/// Produce the pooled sample arrays that `calibrate` reduces to quantiles.
pub fn dump_samples(records: &[TraceRecord], num_samples: usize, seed: u64) -> Result<SampleDump> {
    if records.is_empty() {
        bail!("no records in trace");
    }
    let samples_per_record = (num_samples / records.len()).max(10);

    let (mut source_ttft, mut source_itl) = pool_samples(records);
    source_ttft.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    source_itl.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let trace_model = TraceLatency::from_records(TraceMeta::default(), records, zero_knob())
        .context("building TraceLatency for replay")?;
    let (_, replay_ttft, replay_itl) =
        sample_model_to_buckets(&trace_model, records, samples_per_record, seed);

    let knob_model = fit_knob_from_trace(records);
    let (_, knob_ttft, knob_itl) =
        sample_model_to_buckets(&knob_model, records, samples_per_record, seed + 1);

    Ok(SampleDump {
        source: ModelSamples {
            ttft_ms: source_ttft,
            itl_ms: source_itl,
        },
        replay: ModelSamples {
            ttft_ms: replay_ttft,
            itl_ms: replay_itl,
        },
        knobfit: ModelSamples {
            ttft_ms: knob_ttft,
            itl_ms: knob_itl,
        },
    })
}

/// Run the model-level calibration. Returns the full report.
pub fn calibrate(
    records: &[TraceRecord],
    num_samples: usize,
    seed: u64,
    tolerance: f64,
) -> Result<CalibrationReport> {
    if records.is_empty() {
        bail!("no records in trace");
    }

    let samples_per_record = (num_samples / records.len()).max(10);

    // SOURCE quantiles
    let source_buckets = source_samples_by_bucket(records);
    let (source_pooled_ttft, source_pooled_itl) = pool_samples(records);
    let source = quantiles_from_buckets(&source_buckets, &source_pooled_ttft, &source_pooled_itl);

    // REPLAY: build TraceLatency with a zero KnobLatency fallback
    let trace_model = TraceLatency::from_records(TraceMeta::default(), records, zero_knob())
        .context("building TraceLatency for replay")?;

    let (replay_buckets, replay_pooled_ttft, replay_pooled_itl) =
        sample_model_to_buckets(&trace_model, records, samples_per_record, seed);
    let replay = quantiles_from_buckets(&replay_buckets, &replay_pooled_ttft, &replay_pooled_itl);

    let request_total =
        request_total_report(records, &trace_model, samples_per_record, seed, tolerance);

    // KNOB-FIT: fit KnobLatency from source stats
    let knob_model = fit_knob_from_trace(records);

    let (knob_buckets, knob_pooled_ttft, knob_pooled_itl) =
        sample_model_to_buckets(&knob_model, records, samples_per_record, seed + 1);
    let knobfit = quantiles_from_buckets(&knob_buckets, &knob_pooled_ttft, &knob_pooled_itl);

    // Pooled quantiles (last entry in each vec)
    let source_pooled = source.last().map(|b| (&b.ttft, &b.itl));
    let replay_pooled = replay.last().map(|b| (&b.ttft, &b.itl));
    let knobfit_pooled = knobfit.last().map(|b| (&b.ttft, &b.itl));

    let (src_ttft_q, src_itl_q) = source_pooled.map(|(t, i)| (*t, *i)).unwrap_or((
        Quantiles {
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        },
        Quantiles {
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        },
    ));
    let (rep_ttft_q, rep_itl_q) = replay_pooled.map(|(t, i)| (*t, *i)).unwrap_or((
        Quantiles {
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        },
        Quantiles {
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        },
    ));
    let (knb_ttft_q, knb_itl_q) = knobfit_pooled.map(|(t, i)| (*t, *i)).unwrap_or((
        Quantiles {
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        },
        Quantiles {
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        },
    ));

    let replay_ttft_err = src_ttft_q.max_relative_error(&rep_ttft_q);
    let replay_itl_err = src_itl_q.max_relative_error(&rep_itl_q);

    let replay_pass = replay_ttft_err <= tolerance
        && replay_itl_err <= tolerance
        && request_total.as_ref().is_none_or(|rt| rt.within_tolerance);
    let knobfit_tail_capped = knb_ttft_q.tail_ratio() <= 1.75 && src_ttft_q.tail_ratio() > 1.75;

    let verdict = Verdict {
        source_ttft_tail_ratio: src_ttft_q.tail_ratio(),
        source_itl_tail_ratio: src_itl_q.tail_ratio(),
        replay_ttft_tail_ratio: rep_ttft_q.tail_ratio(),
        replay_itl_tail_ratio: rep_itl_q.tail_ratio(),
        knobfit_ttft_tail_ratio: Some(knb_ttft_q.tail_ratio()),
        knobfit_itl_tail_ratio: Some(knb_itl_q.tail_ratio()),
        replay_ttft_max_error: replay_ttft_err,
        replay_itl_max_error: replay_itl_err,
        replay_pass,
        knobfit_tail_capped,
        tolerance,
    };

    Ok(CalibrationReport {
        measured_label: "replay".to_string(),
        source,
        replay,
        knobfit: Some(knobfit),
        request_total,
        verdict,
    })
}

/// Compare per-request decode totals: observed sums of each record's ITL gaps vs
/// totals assembled by drawing the same number of gaps from the replay model.
fn request_total_report(
    records: &[TraceRecord],
    model: &dyn LatencyModel,
    samples_per_record: usize,
    seed: u64,
    tolerance: f64,
) -> Option<RequestTotalReport> {
    let mut source_totals: Vec<f64> = Vec::new();
    for r in records {
        if let Some(ref gaps) = r.itl_ms {
            if !gaps.is_empty() {
                source_totals.push(gaps.iter().sum());
            }
        } else if let Some(ref s) = r.itl_summary
            && s.count > 0
        {
            source_totals.push(s.mean_ms * s.count as f64);
        }
    }
    if source_totals.is_empty() {
        return None;
    }

    // Bound the per-record repetitions: totals need far fewer draws than
    // per-token quantiles to stabilize, and each repetition costs n_gaps draws.
    let reps = samples_per_record.clamp(3, 20);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(2));
    let mut replay_totals: Vec<f64> = Vec::new();
    for r in records {
        let n_gaps = match (&r.itl_ms, &r.itl_summary) {
            (Some(gaps), _) => gaps.len(),
            (None, Some(s)) => s.count,
            (None, None) => 0,
        };
        if n_gaps == 0 {
            continue;
        }
        for _ in 0..reps {
            let mut pacing = DecodePacing::default();
            let total_ms: f64 = (0..n_gaps)
                .map(|i| {
                    // Model-level calibration has no scheduler to generate prefill
                    // interference, so replay the record's own recorded event
                    // sequence: stall exactly where the source stalled.
                    if let Some(ctx) = &r.itl_ctx
                        && ctx.prefill_tokens[i] > 0
                    {
                        pacing.note_prefill(ctx.prefill_tokens[i]);
                    }
                    model
                        .paced_inter_token_delay(&mut rng, r.concurrency, &mut pacing)
                        .as_secs_f64()
                        * 1000.0
                })
                .sum();
            replay_totals.push(total_ms);
        }
    }

    source_totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    replay_totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let source = Quantiles::from_sorted(&source_totals);
    let replay = Quantiles::from_sorted(&replay_totals);
    let max_error = source.max_relative_error(&replay);

    Some(RequestTotalReport {
        count: source_totals.len(),
        source,
        replay,
        max_error,
        within_tolerance: max_error <= tolerance,
    })
}

// ---------------------------------------------------------------------------
// Report rendering
// ---------------------------------------------------------------------------

/// Render an aligned text table and verdict block. The knob-fit columns appear only when
/// the report carries knob-fit data (model-level calibration); the e2e harness measures one
/// model per run and labels the measured column accordingly.
pub fn write_report(writer: &mut impl Write, report: &CalibrationReport) -> Result<()> {
    let m = &report.measured_label;
    let has_knob = report.knobfit.is_some();

    let header = |writer: &mut dyn Write| -> Result<()> {
        write!(
            writer,
            "{:<10} {:>6}  {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}",
            "conc",
            "n",
            "src p50",
            "src p90",
            "src p99",
            format!("{m} p50"),
            format!("{m} p90"),
            format!("{m} p99"),
        )?;
        if has_knob {
            write!(
                writer,
                "  {:>10} {:>10} {:>10}",
                "knb p50", "knb p90", "knb p99"
            )?;
        }
        writeln!(writer)?;
        Ok(())
    };

    let row = |writer: &mut dyn Write,
               label: &str,
               count: usize,
               s: &Quantiles,
               r: &Quantiles,
               k: Option<&Quantiles>|
     -> Result<()> {
        write!(
            writer,
            "{label:<10} {count:>6}  {:>10.2} {:>10.2} {:>10.2}  {:>10.2} {:>10.2} {:>10.2}",
            s.p50, s.p90, s.p99, r.p50, r.p90, r.p99,
        )?;
        if let Some(k) = k {
            write!(writer, "  {:>10.2} {:>10.2} {:>10.2}", k.p50, k.p90, k.p99)?;
        }
        writeln!(writer)?;
        Ok(())
    };

    let knob_at = |i: usize| report.knobfit.as_deref().map(|kb| &kb[i]);

    writeln!(writer, "=== TTFT (ms) ===")?;
    header(writer)?;
    for (i, (s, r)) in report.source.iter().zip(&report.replay).enumerate() {
        row(
            writer,
            &s.concurrency_label,
            s.count,
            &s.ttft,
            &r.ttft,
            knob_at(i).map(|k| &k.ttft),
        )?;
    }

    writeln!(writer)?;
    writeln!(writer, "=== ITL (ms) ===")?;
    header(writer)?;
    for (i, (s, r)) in report.source.iter().zip(&report.replay).enumerate() {
        row(
            writer,
            &s.concurrency_label,
            s.count,
            &s.itl,
            &r.itl,
            knob_at(i).map(|k| &k.itl),
        )?;
    }

    if let Some(ref rt) = report.request_total {
        writeln!(writer)?;
        writeln!(writer, "=== Request decode total (ms) ===")?;
        writeln!(
            writer,
            "{:<10} {:>6}  {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}",
            "",
            "n",
            "src p50",
            "src p90",
            "src p99",
            format!("{m} p50"),
            format!("{m} p90"),
            format!("{m} p99"),
        )?;
        writeln!(
            writer,
            "{:<10} {:>6}  {:>10.2} {:>10.2} {:>10.2}  {:>10.2} {:>10.2} {:>10.2}",
            "pooled",
            rt.count,
            rt.source.p50,
            rt.source.p90,
            rt.source.p99,
            rt.replay.p50,
            rt.replay.p90,
            rt.replay.p99,
        )?;
    }

    writeln!(writer)?;
    write_verdict(writer, &report.verdict, m)?;

    if let Some(ref rt) = report.request_total {
        writeln!(
            writer,
            "Request-total max relative error: {:.4} ({})",
            rt.max_error,
            if rt.within_tolerance {
                "within tolerance"
            } else {
                "EXCEEDS tolerance"
            },
        )?;
    }

    Ok(())
}

fn write_verdict(writer: &mut impl Write, v: &Verdict, measured_label: &str) -> Result<()> {
    writeln!(writer, "=== Verdict ===")?;
    writeln!(writer, "Pooled p99/p50 ratios:")?;
    writeln!(
        writer,
        "  source   TTFT={:.3}  ITL={:.3}",
        v.source_ttft_tail_ratio, v.source_itl_tail_ratio
    )?;
    writeln!(
        writer,
        "  {:<8} TTFT={:.3}  ITL={:.3}",
        measured_label, v.replay_ttft_tail_ratio, v.replay_itl_tail_ratio
    )?;
    if let (Some(ttft), Some(itl)) = (v.knobfit_ttft_tail_ratio, v.knobfit_itl_tail_ratio) {
        writeln!(writer, "  knob-fit TTFT={ttft:.3}  ITL={itl:.3}")?;
    }
    let mut cap = measured_label.to_string();
    if let Some(first) = cap.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    writeln!(writer)?;
    writeln!(
        writer,
        "{cap} max relative error: TTFT={:.4}  ITL={:.4}",
        v.replay_ttft_max_error, v.replay_itl_max_error
    )?;
    writeln!(
        writer,
        "{cap} PASS (tolerance {:.2}): {}",
        v.tolerance,
        if v.replay_pass { "PASS" } else { "FAIL" }
    )?;

    if v.knobfit_tail_capped
        && let Some(knob_ttft) = v.knobfit_ttft_tail_ratio
    {
        writeln!(
            writer,
            "Knob-fit tail capped at ~1.7x by construction (source TTFT ratio {:.3} > 1.75, knob-fit {:.3} <= 1.75)",
            v.source_ttft_tail_ratio, knob_ttft
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// calibrate from file
// ---------------------------------------------------------------------------

/// Run calibrate from a trace file path. Convenience for the CLI.
pub fn calibrate_from_file(
    path: &Path,
    num_samples: usize,
    seed: u64,
    tolerance: f64,
) -> Result<CalibrationReport> {
    let file =
        std::fs::File::open(path).with_context(|| format!("opening trace: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let (_meta, records) =
        read_trace(reader).with_context(|| format!("parsing trace: {}", path.display()))?;
    calibrate(&records, num_samples, seed, tolerance)
}

// ---------------------------------------------------------------------------
// calibrate-e2e (wire-level proof)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_on_known_samples() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let q = Quantiles::from_sorted(&sorted);
        assert!((q.p50 - 5.0).abs() < 0.01, "p50={}", q.p50);
        assert!((q.p90 - 9.0).abs() < 0.01, "p90={}", q.p90);
        assert!((q.p99 - 10.0).abs() < 0.01, "p99={}", q.p99);
    }

    #[test]
    fn quantile_empty() {
        let q = Quantiles::from_sorted(&[]);
        assert_eq!(q.p50, 0.0);
        assert_eq!(q.p90, 0.0);
        assert_eq!(q.p99, 0.0);
    }

    #[test]
    fn tail_ratio_basic() {
        let q = Quantiles {
            p50: 10.0,
            p90: 50.0,
            p99: 100.0,
        };
        assert!((q.tail_ratio() - 10.0).abs() < 0.001);
    }

    #[test]
    fn tail_ratio_zero_p50() {
        let q = Quantiles {
            p50: 0.0,
            p90: 1.0,
            p99: 2.0,
        };
        assert_eq!(q.tail_ratio(), 0.0);
    }

    #[test]
    fn max_relative_error_identical() {
        let a = Quantiles {
            p50: 10.0,
            p90: 20.0,
            p99: 30.0,
        };
        assert!(a.max_relative_error(&a) < f64::EPSILON);
    }

    #[test]
    fn max_relative_error_different() {
        let a = Quantiles {
            p50: 10.0,
            p90: 20.0,
            p99: 30.0,
        };
        let b = Quantiles {
            p50: 12.0,
            p90: 22.0,
            p99: 33.0,
        };
        let err = a.max_relative_error(&b);
        // max is at p50: |10-12|/12 = 0.1667
        assert!(err > 0.15 && err < 0.20, "err={err}");
    }

    #[test]
    fn knob_fit_math() {
        // Constant trace: all TTFTs=50ms, all ITLs=10ms
        let records: Vec<TraceRecord> = (0..20)
            .map(|_| TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 0,
                output_tokens: 5,
                ttft_ms: 50.0,
                itl_ms: Some(vec![10.0; 4]),
                itl_summary: None,
                concurrency: 1,
                arrival_ms: None,
                itl_ctx: None,
            })
            .collect();

        let knob = fit_knob_from_trace(&records);
        assert_eq!(knob.time_to_first_token, 50);
        assert_eq!(knob.inter_token_latency, 10);
        assert_eq!(knob.time_to_first_token_std_dev, 0);
        assert_eq!(knob.inter_token_latency_std_dev, 0);
    }

    #[test]
    fn gen_demo_determinism() {
        let (meta1, recs1) = gen_demo(100, 42);
        let (meta2, recs2) = gen_demo(100, 42);
        assert_eq!(meta1, meta2);
        assert_eq!(recs1.len(), recs2.len());
        for (a, b) in recs1.iter().zip(recs2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn gen_demo_different_seeds() {
        let (_, recs1) = gen_demo(50, 0);
        let (_, recs2) = gen_demo(50, 99);
        // At least some records should differ
        let any_diff = recs1
            .iter()
            .zip(recs2.iter())
            .any(|(a, b)| a.ttft_ms != b.ttft_ms);
        assert!(any_diff, "different seeds should produce different traces");
    }

    #[test]
    fn report_pass_on_constant_trace() {
        // A constant trace should pass calibration trivially: replay reproduces exact values.
        let records: Vec<TraceRecord> = (0..50)
            .map(|_| TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 0,
                output_tokens: 5,
                ttft_ms: 50.0,
                itl_ms: Some(vec![10.0; 4]),
                itl_summary: None,
                concurrency: 1,
                arrival_ms: None,
                itl_ctx: None,
            })
            .collect();

        let report = calibrate(&records, 10000, 0, 0.10).unwrap();
        assert!(
            report.verdict.replay_pass,
            "constant trace should pass: {:?}",
            report.verdict
        );
    }

    #[test]
    fn report_fail_on_shifted_source() {
        // Source trace has one distribution, but we'll create a second set of records
        // with a very different TTFT and compare manually.
        let records: Vec<TraceRecord> = (0..50)
            .map(|_| TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 0,
                output_tokens: 5,
                ttft_ms: 50.0,
                itl_ms: Some(vec![10.0; 4]),
                itl_summary: None,
                concurrency: 1,
                arrival_ms: None,
                itl_ctx: None,
            })
            .collect();

        // Build replay model from the original records
        let report = calibrate(&records, 10000, 0, 0.10).unwrap();
        // Replay of a constant trace with itself should pass
        assert!(report.verdict.replay_pass);

        // Now create a "shifted" trace where TTFT is 200ms instead of 50ms.
        // Calibrate from the shifted trace: replay will match the shifted trace,
        // but let's check the knob-fit tail ratio vs source.
        let shifted: Vec<TraceRecord> = (0..50)
            .map(|_| TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 0,
                output_tokens: 5,
                ttft_ms: 200.0,
                itl_ms: Some(vec![40.0; 4]),
                itl_summary: None,
                concurrency: 1,
                arrival_ms: None,
                itl_ctx: None,
            })
            .collect();

        // Build TraceLatency from original records, but compare against shifted source.
        // The replay (from original records) should NOT match shifted source quantiles.
        let replay_model =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();

        // Sample from the replay model using the shifted records' contexts
        let samples_per_record = 200;
        let mut rng = StdRng::seed_from_u64(0);
        let mut replay_ttfts = Vec::new();
        for r in &shifted {
            let ctx = FirstTokenCtx {
                num_prompt_tokens: r.prompt_tokens,
                num_cached_tokens: r.cached_tokens,
                do_remote_prefill: false,
                num_running: r.concurrency,
            };
            for _ in 0..samples_per_record {
                replay_ttfts
                    .push(replay_model.first_token_delay(&mut rng, &ctx).as_secs_f64() * 1000.0);
            }
        }
        replay_ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let replay_q = Quantiles::from_sorted(&replay_ttfts);

        let mut shifted_ttfts: Vec<f64> = shifted.iter().map(|r| r.ttft_ms).collect();
        shifted_ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let source_q = Quantiles::from_sorted(&shifted_ttfts);

        let err = source_q.max_relative_error(&replay_q);
        assert!(
            err > 0.10,
            "replay from original trace should NOT match shifted source (err={err})"
        );
    }

    #[test]
    fn gen_demo_heavy_tail() {
        let (_, records) = gen_demo(200, 0);
        let (ttfts, itls) = pool_samples(&records);
        let ttft_q = Quantiles::from_sorted(&ttfts);
        let itl_q = Quantiles::from_sorted(&itls);

        assert!(
            ttft_q.tail_ratio() >= 3.0,
            "demo TTFT p99/p50 should be >= 3.0, got {:.3}",
            ttft_q.tail_ratio()
        );
        assert!(
            itl_q.tail_ratio() >= 2.0,
            "demo ITL p99/p50 should be heavy-tailed, got {:.3}",
            itl_q.tail_ratio()
        );
    }

    #[test]
    fn write_report_produces_output() {
        let records: Vec<TraceRecord> = (0..50)
            .map(|_| TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 0,
                output_tokens: 5,
                ttft_ms: 50.0,
                itl_ms: Some(vec![10.0; 4]),
                itl_summary: None,
                concurrency: 1,
                arrival_ms: None,
                itl_ctx: None,
            })
            .collect();

        let report = calibrate(&records, 5000, 0, 0.10).unwrap();
        let mut buf = Vec::new();
        write_report(&mut buf, &report).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("TTFT"), "report should contain TTFT table");
        assert!(text.contains("Verdict"), "report should contain verdict");
        assert!(text.contains("PASS") || text.contains("FAIL"));
        assert!(
            text.contains("Request decode total"),
            "report should contain request-total table"
        );
    }

    /// The trap this metric exists for: requests that are internally consistent
    /// but bimodal ACROSS requests (all-fast vs all-slow). Per-token quantiles
    /// pass under any sampling scheme; per-request totals only reproduce when
    /// within-request correlation is preserved (hierarchical donor pacing).
    /// i.i.d. totals would concentrate near 64 * 27.5 = 1760ms; the source is
    /// bimodal at 320ms / 3200ms.
    #[test]
    fn hierarchical_sampling_preserves_request_totals() {
        let mut records: Vec<TraceRecord> = Vec::new();
        for i in 0..100 {
            // Half the requests decode at 5ms/token, half at 50ms/token.
            let gap = if i % 2 == 0 { 5.0 } else { 50.0 };
            records.push(TraceRecord {
                prompt_tokens: 100,
                cached_tokens: 0,
                output_tokens: 65,
                ttft_ms: 40.0,
                itl_ms: Some(vec![gap; 64]),
                itl_summary: None,
                concurrency: 4,
                arrival_ms: None,
                itl_ctx: None,
            });
        }

        let report = calibrate(&records, 50_000, 0, 0.10).unwrap();
        let rt = report
            .request_total
            .expect("request totals should be computed");
        assert_eq!(rt.count, 100);

        assert!(
            rt.within_tolerance,
            "donor pacing should reproduce bimodal request totals, got max_error {:.3} \
             (src p50/p99 {:.0}/{:.0}, replay p50/p99 {:.0}/{:.0})",
            rt.max_error, rt.source.p50, rt.source.p99, rt.replay.p50, rt.replay.p99,
        );
        assert!(
            report.verdict.replay_pass,
            "request totals now gate replay_pass"
        );
        assert!(
            report.verdict.replay_itl_max_error <= 0.10,
            "per-token ITL quantiles must still pass, got {:.3}",
            report.verdict.replay_itl_max_error
        );
    }
}
