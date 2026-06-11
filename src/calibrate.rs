//! Calibration and comparison harness for trace-replay vs knob-fit latency models.
//!
//! Proves two claims:
//!   1. `TraceLatency` replay reproduces a source trace's latency quantiles (within tolerance).
//!   2. `KnobLatency` structurally cannot reproduce heavy tails: its `[0.3*mean, 1.7*mean]`
//!      clamp caps p99/p50 at roughly 1.7 for any knob settings.
//!
//! Four entry points, each exposed on the `inference-sim-trace` binary:
//!   - `gen_demo`: synthesize a heavy-tailed demo trace (lognormal TTFT/ITL).
//!   - `calibrate`: model-level quantile comparison (source vs replay vs knob-fit).
//!   - `calibrate_e2e`: wire-level proof using the real simulator in-process.
//!   - `replay_arrivals`: wire-level open-loop replay of a recorded arrival schedule.

use std::io::Write;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use rand::Rng;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::latency::{
    DecodePacing, FirstTokenCtx, KnobLatency, LatencyModel, NUM_CONCURRENCY_BUCKETS, TraceLatency,
    concurrency_bucket, concurrency_label, random_norm,
};
use crate::trace::{TraceMeta, TraceRecord, read_trace, write_trace};

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
                block_hashes: None,
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
                block_hashes: None,
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
        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.push((concurrency_label(i), ttfts, itls));
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
pub type BucketedSamples = (Vec<(String, Vec<f64>, Vec<f64>)>, Vec<f64>, Vec<f64>);

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

    buckets_from_measured(bucket_ttfts, bucket_itls)
}

/// Sort per-concurrency-bucket TTFT/ITL sample arrays into labeled buckets plus
/// pooled arrays, dropping buckets with no TTFT observations. Input vecs are
/// indexed by concurrency bucket.
pub fn buckets_from_measured(
    bucket_ttfts: Vec<Vec<f64>>,
    bucket_itls: Vec<Vec<f64>>,
) -> BucketedSamples {
    let mut pooled_ttft = Vec::new();
    let mut pooled_itl = Vec::new();
    let mut result = Vec::new();

    for (i, (mut ttfts, mut itls)) in bucket_ttfts.into_iter().zip(bucket_itls).enumerate() {
        if ttfts.is_empty() {
            continue;
        }
        pooled_ttft.extend(ttfts.iter().copied());
        pooled_itl.extend(itls.iter().copied());

        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.push((concurrency_label(i), ttfts, itls));
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
            let mut pacing = DecodePacing::for_prompt(r.prompt_tokens);
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

// SplitMix64-style mix constants (Steele et al., "Fast Splittable Pseudorandom
// Number Generators"): GOLDEN_GAMMA is 2^64/phi, the stream increment; MIX is
// the first finalizer multiplier. One multiply-add plus a xor-shift is enough
// here: we only need (seed, position) -> token to not produce identical
// token blocks across different seeds, not statistical quality.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX: u64 = 0xBF58_476D_1CE4_E5B9;

/// Deterministic token at `position` of the stream keyed by `seed`.
fn mix_token(seed: u64, position: u64) -> u32 {
    let mut x = seed
        .wrapping_mul(GOLDEN_GAMMA)
        .wrapping_add(position.wrapping_mul(MIX));
    x ^= x >> 33;
    // Stay well under the sim's default vocab size (32000).
    (x % 31_000) as u32
}

/// Deterministic per-request prompt tokens for the wire-level harnesses.
///
/// Each request gets a unique token sequence so the sim's prefix cache treats
/// it as a cold prompt, matching captures that record cached_tokens=0.
/// Identical fill tokens would silently turn every replayed request into a
/// prefix-cache hit, collapsing TTFT into the smallest uncached-prompt bucket.
/// Records that carry `block_hashes` get their prefix sharing reconstructed by
/// [`prompt_from_block_hashes`] instead.
pub fn synthetic_prompt(request_index: usize, len: usize) -> Vec<u32> {
    (0..len as u64)
        .map(|j| mix_token(request_index as u64, j))
        .collect()
}

/// Reconstruct a prompt whose block-level prefix sharing matches the recorded
/// fingerprints: each distinct hash deterministically expands to one block of
/// tokens, so two records agree on tokens exactly where their hash chains
/// agree, and the replay sim's prefix cache reproduces the captured engine's
/// hits reactively. Tokens past the last full block are unique per request
/// (a partial tail block can never be a cache hit).
pub fn prompt_from_block_hashes(
    hashes: &[u64],
    prompt_tokens: usize,
    block_size: usize,
    request_index: usize,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(prompt_tokens);
    for &h in hashes {
        tokens.extend((0..block_size as u64).map(|j| mix_token(h, j)));
    }
    // Defensive: a malformed record could carry more hashed blocks than
    // prompt_tokens; the prompt length is the schema's source of truth.
    tokens.truncate(prompt_tokens);
    let tail = prompt_tokens - tokens.len();
    tokens.extend(synthetic_prompt(request_index, tail));
    tokens
}

/// Config for the open-loop arrival replay harness ([`replay_arrivals`]).
pub struct ReplayArrivalsConfig<'a> {
    /// Trace supplying the arrival schedule, request shapes, and ground-truth
    /// quantiles. Records without `arrival_ms` are skipped.
    pub trace_path: &'a Path,
    /// Build the sim's latency model from this trace instead of `trace_path`,
    /// validating the model against an arrival process it was not fitted on.
    pub latency_trace: Option<&'a Path>,
    /// Replay only the first N arrivals (default: the whole schedule).
    pub max_requests: Option<usize>,
    /// Max allowed relative error on pooled quantiles and request totals.
    pub tolerance: f64,
    /// Drive the sim with fitted knobs instead of trace replay.
    pub use_knob_fit: bool,
    /// Disambiguates the IPC socket when several replays share a process.
    pub ipc_tag: String,
    /// Extra CLI flags for the in-process sim (e.g. --kv-cache-size). The
    /// replay sim must match the capture engine's scheduler/cache config for
    /// the reactive comparison to be apples-to-apples.
    pub extra_sim_args: Vec<String>,
    /// Replay multiturn sessions closed-loop: a turn fires when its parent
    /// turn completes plus the recorded think gap, instead of at its recorded
    /// wall-clock offset. Session roots still fire open-loop. This mirrors how
    /// agentic workloads actually pace turns; pure open-loop replay of a
    /// closed-loop-generated schedule amplifies any transient divergence
    /// (arrivals do not back off when the replay runs momentarily slower).
    /// Sessions are inferred from block-hash chains; requires a trace with
    /// `block_hashes`.
    pub session_replay: bool,
    /// Cache-off what-if: replay every prompt as unique tokens even when the
    /// trace carries block_hashes (the chains still drive session inference).
    /// The TTFT delta vs a normal replay is the prefix cache's contribution.
    pub cold_prompts: bool,
    /// Time compression factor: the sim divides its delays by this, the
    /// harness compresses the schedule and re-multiplies measurements. Inner
    /// calibration loops run faster at the cost of timer-granularity fidelity;
    /// final validation should use 1.0.
    pub time_scale: f64,
}

/// For each record (arrival-ordered), the index of its session parent: the
/// most recent earlier record whose full hash chain is a proper prefix of this
/// record's chain (a multiturn child's prompt extends its parent's context).
/// Records without hashes or without such a predecessor are session roots.
fn infer_session_parents(records: &[TraceRecord]) -> Vec<Option<usize>> {
    use std::collections::HashMap;
    let mut by_last_hash: HashMap<u64, usize> = HashMap::new();
    let mut parents = Vec::with_capacity(records.len());
    for (i, r) in records.iter().enumerate() {
        let mut parent = None;
        if let Some(chain) = r.block_hashes.as_deref() {
            for k in (1..chain.len()).rev() {
                if let Some(&p) = by_last_hash.get(&chain[k - 1])
                    && records[p].block_hashes.as_deref().map(<[u64]>::len) == Some(k)
                {
                    parent = Some(p);
                    break;
                }
            }
            if let Some(&last) = chain.last() {
                by_last_hash.insert(last, i);
            }
        }
        parents.push(parent);
    }
    parents
}

/// When a request finished, in source-trace time (arrival + TTFT + decode).
fn source_completion_ms(r: &TraceRecord) -> f64 {
    r.arrival_ms.unwrap_or(0.0)
        + r.ttft_ms
        + r.itl_ms
            .as_ref()
            .map(|g| g.iter().sum::<f64>())
            .or_else(|| r.itl_summary.as_ref().map(|s| s.mean_ms * s.count as f64))
            .unwrap_or(0.0)
}

/// Send one replayed request and collect client-side timing. Returns
/// (ttft_ms, gaps_ms); a None ttft means the request failed or timed out.
async fn measure_one(
    client: &vllm_engine_core_client::EngineCoreClient,
    request: vllm_engine_core_client::protocol::EngineCoreRequest,
    timeout_dur: std::time::Duration,
) -> (Option<f64>, Vec<f64>) {
    use std::time::Instant;

    use futures::StreamExt as _;

    let rid = request.request_id.clone();
    let call_start = Instant::now();
    let mut stream = match client.call(request).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("{rid}: call failed: {e:#}");
            return (None, Vec::new());
        }
    };
    // TTFT needs the client clock (it spans the send), but GAPS use the
    // engine-side batch timestamps: client-receive stamps coalesce whenever
    // the client task lags and buffered outputs drain at once, fabricating
    // zero-gaps the engine never produced (measured 15.6% of gaps vs <=0.2%
    // in tap-measured real captures).
    let mut first_token_at: Option<Instant> = None;
    let mut token_stamps: Vec<f64> = Vec::new();
    let result = tokio::time::timeout(timeout_dur, async {
        while let Some(item) = stream.next().await {
            let output = item.context("stream item error")?;
            if !output.new_token_ids.is_empty() {
                first_token_at.get_or_insert_with(Instant::now);
                token_stamps.push(output.timestamp);
            }
            if output.finish_reason.is_some() {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    })
    .await;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!("{rid}: stream error: {e:#}");
            return (None, Vec::new());
        }
        Err(_) => {
            tracing::warn!("{rid}: timed out after {timeout_dur:?}");
            return (None, Vec::new());
        }
    }
    let ttft = first_token_at.map(|f| (f - call_start).as_secs_f64() * 1000.0);
    let gaps = token_stamps
        .windows(2)
        .map(|w| (w[1] - w[0]) * 1000.0)
        .collect();
    (ttft, gaps)
}

/// Outcome of an open-loop arrival replay: the calibration report plus
/// schedule-fidelity diagnostics.
#[derive(Debug)]
pub struct ArrivalReplayOutcome {
    pub report: CalibrationReport,
    /// Client-side measurements of completed requests in trace schema, ordered
    /// by arrival, for external plotting against the source trace.
    pub measured: Vec<TraceRecord>,
    /// Requests in the replayed schedule.
    pub requests_replayed: usize,
    /// Requests that produced a first token (no error, no timeout).
    pub requests_completed: usize,
    /// Worst observed lag between a request's scheduled and actual send time.
    /// Large values mean the harness could not keep up with the schedule and
    /// the measured quantiles are suspect.
    pub max_send_lag_ms: f64,
    pub wall_time_s: f64,
}

/// One replayed request's client-side measurements.
struct ReplayedRequest {
    index: usize,
    send_lag_ms: f64,
    ttft_ms: Option<f64>,
    gaps_ms: Vec<f64>,
}

/// Reactive wire-level validation: replay a trace's recorded arrival schedule
/// against the real in-process simulator over ZMQ. Each request is sent at its
/// recorded offset regardless of how earlier requests are progressing (open
/// loop), so the sim's scheduler state (num_running, prefill admissions)
/// emerges from the workload instead of being pinned by closed-loop batching.
/// Client-side TTFT/ITL/request-totals are then compared against the source
/// quantiles.
pub async fn replay_arrivals(cfg: &ReplayArrivalsConfig<'_>) -> Result<ArrivalReplayOutcome> {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use clap::Parser as _;
    use tokio_util::sync::CancellationToken;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
    use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

    let file = std::fs::File::open(cfg.trace_path)
        .with_context(|| format!("opening trace: {}", cfg.trace_path.display()))?;
    let (meta, all_records) = read_trace(std::io::BufReader::new(file))
        .with_context(|| format!("parsing trace: {}", cfg.trace_path.display()))?;
    let block_size = meta.block_size.unwrap_or(16);
    let time_scale = if cfg.time_scale > 0.0 {
        cfg.time_scale
    } else {
        1.0
    };

    let mut subset: Vec<TraceRecord> = all_records
        .into_iter()
        .filter(|r| r.arrival_ms.is_some())
        .collect();
    if subset.is_empty() {
        bail!(
            "trace has no records with arrival_ms; capture with a tap/loadgen that \
             records the arrival schedule"
        );
    }
    subset.sort_by(|a, b| {
        a.arrival_ms
            .partial_cmp(&b.arrival_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(n) = cfg.max_requests {
        subset.truncate(n);
    }
    let first_arrival = subset[0].arrival_ms.unwrap_or(0.0);
    let span_ms = subset
        .last()
        .and_then(|r| r.arrival_ms)
        .unwrap_or(first_arrival)
        - first_arrival;
    tracing::info!(
        "replaying {} arrivals spanning {:.1}s (real time)",
        subset.len(),
        span_ms / 1000.0
    );

    // Build the simulator's latency model from the fitting source, which is the
    // replayed trace itself unless the caller supplies a separate one.
    let latency_path = cfg.latency_trace.unwrap_or(cfg.trace_path);
    let max_num_seqs = meta.max_num_seqs.unwrap_or(64);

    let addr = format!(
        "ipc:///tmp/inf-sim-replay-{}-{}.ipc",
        std::process::id(),
        cfg.ipc_tag
    );
    let mut args: Vec<String> = vec![
        "inference-sim".to_string(),
        "--handshake-address".to_string(),
        addr.clone(),
    ];
    // The trace meta's max_num_seqs (default 64) is only a fallback; an
    // explicit --sim-arg override wins, and duplicating the flag would make
    // clap reject the args.
    if !cfg
        .extra_sim_args
        .iter()
        .any(|a| a.contains("--max-num-seqs"))
    {
        args.extend(["--max-num-seqs".to_string(), max_num_seqs.to_string()]);
    }
    if cfg.use_knob_fit {
        let lat_file = std::fs::File::open(latency_path)
            .with_context(|| format!("opening latency trace: {}", latency_path.display()))?;
        let (_, lat_records) = read_trace(std::io::BufReader::new(lat_file))
            .with_context(|| format!("parsing latency trace: {}", latency_path.display()))?;
        let knob = fit_knob_from_trace(&lat_records);
        args.extend([
            "--time-to-first-token".to_string(),
            knob.time_to_first_token.to_string(),
            "--time-to-first-token-std-dev".to_string(),
            knob.time_to_first_token_std_dev.to_string(),
            "--inter-token-latency".to_string(),
            knob.inter_token_latency.to_string(),
            "--inter-token-latency-std-dev".to_string(),
            knob.inter_token_latency_std_dev.to_string(),
        ]);
    } else {
        args.extend([
            "--latency-trace".to_string(),
            latency_path.to_string_lossy().to_string(),
        ]);
    }
    if time_scale != 1.0 {
        args.extend(["--time-scale".to_string(), time_scale.to_string()]);
    }
    args.extend(cfg.extra_sim_args.iter().cloned());

    let opt = crate::Opt::parse_from(&args);
    let token = CancellationToken::new();
    let sim_token = token.clone();
    tokio::spawn(async move {
        let _ = crate::run(opt, sim_token).await;
    });

    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(Duration::from_secs(30), EngineCoreClient::connect(config))
        .await
        .map_err(|_| anyhow::anyhow!("client connect timed out"))?
        .context("client connect failed")?;
    let client = Arc::new(client);

    let wall_start = Instant::now();
    let base = tokio::time::Instant::now();

    // Session mode: a child turn fires when its parent completes plus the
    // recorded think gap; roots (and every request, in pure arrival mode)
    // fire at recorded offsets.
    let parents = if cfg.session_replay {
        infer_session_parents(&subset)
    } else {
        vec![None; subset.len()]
    };
    let mut child_txs: Vec<Vec<tokio::sync::oneshot::Sender<()>>> =
        (0..subset.len()).map(|_| Vec::new()).collect();
    let mut triggers: Vec<Option<(tokio::sync::oneshot::Receiver<()>, f64)>> =
        (0..subset.len()).map(|_| None).collect();
    if cfg.session_replay {
        let mut roots = 0usize;
        for (i, parent) in parents.iter().enumerate() {
            let Some(p) = *parent else {
                roots += 1;
                continue;
            };
            let think_ms = (subset[i].arrival_ms.unwrap_or(0.0) - source_completion_ms(&subset[p]))
                .max(0.0)
                / time_scale;
            let (tx, rx) = tokio::sync::oneshot::channel();
            child_txs[p].push(tx);
            triggers[i] = Some((rx, think_ms));
        }
        tracing::info!(
            "session replay: {} session roots, {} chained turns",
            roots,
            subset.len() - roots
        );
    }

    let mut handles = Vec::with_capacity(subset.len());
    for (i, rec) in subset.iter().enumerate() {
        let offset_ms = (rec.arrival_ms.unwrap_or(first_arrival) - first_arrival).max(0.0);
        let target = base + Duration::from_secs_f64(offset_ms / 1000.0 / time_scale);
        // Reconstruct prefix sharing when the capture fingerprinted it;
        // otherwise every prompt is unique (cold cache, matching the capture).
        let prompt = match rec.block_hashes.as_deref() {
            Some(hashes) if !cfg.cold_prompts => {
                prompt_from_block_hashes(hashes, rec.prompt_tokens, block_size, i)
            }
            _ => synthetic_prompt(i, rec.prompt_tokens),
        };
        let max_tokens = rec.output_tokens.max(1).min(u32::MAX as usize) as u32;
        // Generous per-request timeout scaled to the source duration: open-loop
        // replay can queue far past the source under model mismatch, and a
        // timeout there should fail the gate, not hang the harness.
        let source_ms = rec.ttft_ms
            + rec
                .itl_ms
                .as_ref()
                .map(|g| g.iter().sum::<f64>())
                .or_else(|| rec.itl_summary.as_ref().map(|s| s.mean_ms * s.count as f64))
                .unwrap_or(0.0);
        let timeout_dur = Duration::from_secs_f64(60.0 + 5.0 * source_ms / 1000.0 / time_scale);
        let client = Arc::clone(&client);
        let trigger = triggers[i].take();
        let txs = std::mem::take(&mut child_txs[i]);

        handles.push(tokio::spawn(async move {
            let send_lag_ms = match trigger {
                None => {
                    tokio::time::sleep_until(target).await;
                    (tokio::time::Instant::now() - target).as_secs_f64() * 1000.0
                }
                Some((rx, think_ms)) => {
                    // A dead parent task drops the sender; fire immediately then.
                    let _ = rx.await;
                    if think_ms > 0.0 {
                        tokio::time::sleep(Duration::from_secs_f64(think_ms / 1000.0)).await;
                    }
                    // Schedule lag is meaningful only for wall-clock targets.
                    0.0
                }
            };

            let request = EngineCoreRequest {
                request_id: format!("replay-{i}"),
                prompt_token_ids: Some(prompt),
                sampling_params: Some(EngineCoreSamplingParams {
                    max_tokens,
                    ..EngineCoreSamplingParams::for_test()
                }),
                ..Default::default()
            };
            let (ttft_ms, mut gaps_ms) = measure_one(&client, request, timeout_dur).await;
            let ttft_ms = ttft_ms.map(|t| t * time_scale);
            if time_scale != 1.0 {
                for g in &mut gaps_ms {
                    *g *= time_scale;
                }
            }
            // Release dependent turns even when this one failed: the session
            // must keep pacing or the replay deadlocks on errors.
            for tx in txs {
                let _ = tx.send(());
            }
            ReplayedRequest {
                index: i,
                send_lag_ms,
                ttft_ms,
                gaps_ms,
            }
        }));
    }

    let mut completed_requests: Vec<ReplayedRequest> = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(m) => completed_requests.push(m),
            Err(e) => tracing::warn!("replay task failed to join: {e}"),
        }
    }
    let wall_time_s = wall_start.elapsed().as_secs_f64();
    token.cancel();

    // Bucket measurements by the source record's concurrency so they compare
    // against the same source bucketing.
    let mut measured_ttfts: Vec<Vec<f64>> = vec![Vec::new(); NUM_CONCURRENCY_BUCKETS];
    let mut measured_itls: Vec<Vec<f64>> = vec![Vec::new(); NUM_CONCURRENCY_BUCKETS];
    let mut replay_totals: Vec<f64> = Vec::new();
    let mut measured_records: Vec<TraceRecord> = Vec::new();
    let mut requests_completed = 0usize;
    let mut max_send_lag_ms = 0.0_f64;

    for m in &completed_requests {
        max_send_lag_ms = max_send_lag_ms.max(m.send_lag_ms);
        let Some(ttft) = m.ttft_ms else { continue };
        requests_completed += 1;
        let Some(rec) = subset.get(m.index) else {
            continue;
        };
        let cb = concurrency_bucket(rec.concurrency);
        measured_ttfts[cb].push(ttft);
        measured_itls[cb].extend(m.gaps_ms.iter().copied());
        if !m.gaps_ms.is_empty() {
            replay_totals.push(m.gaps_ms.iter().sum());
        }
        measured_records.push(TraceRecord {
            prompt_tokens: rec.prompt_tokens,
            cached_tokens: 0,
            output_tokens: m.gaps_ms.len() + 1,
            ttft_ms: ttft,
            itl_ms: (!m.gaps_ms.is_empty()).then(|| m.gaps_ms.clone()),
            itl_summary: None,
            concurrency: rec.concurrency,
            arrival_ms: rec.arrival_ms,
            itl_ctx: None,
            block_hashes: None,
        });
    }

    let (measured_buckets, pooled_ttft, pooled_itl) =
        buckets_from_measured(measured_ttfts, measured_itls);
    let replay_stats = quantiles_from_buckets(&measured_buckets, &pooled_ttft, &pooled_itl);

    let source_buckets = source_samples_by_bucket(&subset);
    let (source_pooled_ttft, source_pooled_itl) = pool_samples(&subset);
    let source_stats =
        quantiles_from_buckets(&source_buckets, &source_pooled_ttft, &source_pooled_itl);

    // Per-request decode totals: the reactive gate that catches lost
    // within-request correlation and mistimed stall draws.
    let mut source_totals: Vec<f64> = subset
        .iter()
        .filter_map(|r| {
            r.itl_ms
                .as_ref()
                .filter(|g| !g.is_empty())
                .map(|g| g.iter().sum())
                .or_else(|| {
                    r.itl_summary
                        .as_ref()
                        .filter(|s| s.count > 0)
                        .map(|s| s.mean_ms * s.count as f64)
                })
        })
        .collect();
    source_totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    replay_totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let request_total = if source_totals.is_empty() || replay_totals.is_empty() {
        None
    } else {
        let source = Quantiles::from_sorted(&source_totals);
        let replay = Quantiles::from_sorted(&replay_totals);
        let max_error = source.max_relative_error(&replay);
        Some(RequestTotalReport {
            count: source_totals.len(),
            source,
            replay,
            max_error,
            within_tolerance: max_error <= cfg.tolerance,
        })
    };

    let zero_q = Quantiles {
        p50: 0.0,
        p90: 0.0,
        p99: 0.0,
    };
    let (src_ttft_q, src_itl_q) = source_stats
        .last()
        .map(|b| (b.ttft, b.itl))
        .unwrap_or((zero_q, zero_q));
    let (rep_ttft_q, rep_itl_q) = replay_stats
        .last()
        .map(|b| (b.ttft, b.itl))
        .unwrap_or((zero_q, zero_q));

    let replay_ttft_err = src_ttft_q.max_relative_error(&rep_ttft_q);
    let replay_itl_err = src_itl_q.max_relative_error(&rep_itl_q);
    let replay_pass = replay_ttft_err <= cfg.tolerance
        && replay_itl_err <= cfg.tolerance
        && request_total.as_ref().is_none_or(|rt| rt.within_tolerance);

    let measured_label = if cfg.use_knob_fit {
        "knobfit"
    } else {
        "replay"
    }
    .to_string();
    let verdict = Verdict {
        source_ttft_tail_ratio: src_ttft_q.tail_ratio(),
        source_itl_tail_ratio: src_itl_q.tail_ratio(),
        replay_ttft_tail_ratio: rep_ttft_q.tail_ratio(),
        replay_itl_tail_ratio: rep_itl_q.tail_ratio(),
        // One model measured per run; no separate knob-fit sample set.
        knobfit_ttft_tail_ratio: None,
        knobfit_itl_tail_ratio: None,
        replay_ttft_max_error: replay_ttft_err,
        replay_itl_max_error: replay_itl_err,
        replay_pass,
        knobfit_tail_capped: false,
        tolerance: cfg.tolerance,
    };

    Ok(ArrivalReplayOutcome {
        report: CalibrationReport {
            measured_label,
            source: source_stats,
            replay: replay_stats,
            knobfit: None,
            request_total,
            verdict,
        },
        measured: measured_records,
        requests_replayed: subset.len(),
        requests_completed,
        max_send_lag_ms,
        wall_time_s,
    })
}

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
    fn session_parents_follow_hash_chains() {
        use crate::trace::prompt_block_hashes;

        let block = 4usize;
        let rec = |tokens: &[u32], arrival: f64| TraceRecord {
            prompt_tokens: tokens.len(),
            cached_tokens: 0,
            output_tokens: 2,
            ttft_ms: 10.0,
            itl_ms: Some(vec![5.0]),
            itl_summary: None,
            concurrency: 1,
            arrival_ms: Some(arrival),
            itl_ctx: None,
            block_hashes: prompt_block_hashes(tokens, block),
        };

        // Two interleaved sessions; session B shares session A's first block
        // (a shared prefix) but is not a chain extension of any A turn.
        let a1: Vec<u32> = (0..8).collect();
        let mut a2 = a1.clone();
        a2.extend(100..108);
        let mut b1: Vec<u32> = (0..4).collect();
        b1.extend(200..208);
        let mut b2 = b1.clone();
        b2.extend(300..308);

        let records = vec![rec(&a1, 0.0), rec(&b1, 1.0), rec(&a2, 2.0), rec(&b2, 3.0)];
        let parents = infer_session_parents(&records);
        assert_eq!(parents, vec![None, None, Some(0), Some(1)]);
    }

    #[test]
    fn prompt_reconstruction_mirrors_hash_chains() {
        use crate::trace::prompt_block_hashes;

        let block = 16usize;
        // Two "captured" prompts sharing 32 tokens then diverging.
        let shared: Vec<u32> = (0..32).collect();
        let mut a = shared.clone();
        a.extend(200..232);
        let mut b = shared.clone();
        b.extend(300..340);

        let ha = prompt_block_hashes(&a, block).unwrap();
        let hb = prompt_block_hashes(&b, block).unwrap();

        let ra = prompt_from_block_hashes(&ha, a.len(), block, 0);
        let rb = prompt_from_block_hashes(&hb, b.len(), block, 1);

        assert_eq!(ra.len(), a.len());
        assert_eq!(rb.len(), b.len());
        // Reconstructed prompts share tokens exactly where the originals did.
        assert_eq!(ra[..32], rb[..32], "shared prefix must reconstruct shared");
        assert_ne!(ra[32..48], rb[32..48], "divergent blocks must differ");
        // Re-fingerprinting the reconstruction yields a chain with the same
        // sharing structure (equal where the original chains were equal).
        let ra2 = prompt_block_hashes(&ra, block).unwrap();
        let rb2 = prompt_block_hashes(&rb, block).unwrap();
        assert_eq!(ra2[..2], rb2[..2]);
        assert_ne!(ra2[2], rb2[2]);
    }

    #[test]
    fn prompt_reconstruction_tail_is_unique_per_request() {
        // 2 full blocks + 5-token tail; same hashes, different request index.
        let hashes = vec![7u64, 8u64];
        let x = prompt_from_block_hashes(&hashes, 37, 16, 10);
        let y = prompt_from_block_hashes(&hashes, 37, 16, 11);
        assert_eq!(x[..32], y[..32], "hashed blocks are shared");
        assert_ne!(x[32..], y[32..], "tails must not collide across requests");

        // Malformed record: more hashed blocks than prompt_tokens. Length wins.
        let z = prompt_from_block_hashes(&hashes, 20, 16, 12);
        assert_eq!(z.len(), 20);
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
                block_hashes: None,
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
                block_hashes: None,
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
                block_hashes: None,
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
                block_hashes: None,
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
                block_hashes: None,
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
                block_hashes: None,
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
