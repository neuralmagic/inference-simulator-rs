//! The response-timing model, ported from `llm-d-inference-sim`'s `latencies.go`.
//!
//! We are the real engine-core behind the real vLLM frontend, and the frontend measures
//! time-to-first-token and inter-token latency from *when we emit tokens*. So pacing our
//! output is what produces realistic timing AND realistic `vllm:*` latency histograms.
//!
//! Two delays drive the engine step loop:
//!   - **first-token delay** (prefill): waited once, before the first output token. Models
//!     prefill compute (`prefill_overhead + (prompt - cached) * prefill_time_per_token`),
//!     or, for a disaggregated decode request (`do_remote_prefill`), the KV-cache transfer
//!     time instead. A fixed `time_to_first_token` overrides the token-count model.
//!   - **inter-token delay** (decode): waited between subsequent output tokens.
//!
//! Both are sampled from a truncated normal (mean clamped to `[0.3*mean, 1.7*mean]`, matching
//! upstream `RandomNormTruncated`) and scaled by a load factor that grows with concurrency.
//! Every knob defaults to 0 (no std-dev, no overhead), so an unconfigured engine is instant,
//! byte-for-byte the pre-latency behavior.

use std::f64::consts::PI;
use std::time::Duration;

use rand::Rng as _;
use rand::rngs::StdRng;

/// Context for computing the first-token delay of a request.
pub struct FirstTokenCtx {
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub do_remote_prefill: bool,
    pub num_running: u64,
}

/// Per-request decode pacing state, owned by the engine alongside the request.
///
/// For hierarchical models (trace replay) this pins the request to one source
/// "donor" request per concurrency bucket, so consecutive gaps stay correlated
/// the way real decode gaps are (a slow request is slow throughout). Stateless
/// models ignore it.
#[derive(Debug, Clone)]
pub struct DecodePacing {
    /// Chosen donor index per concurrency bucket, lazily picked on first use.
    donors: [Option<u32>; NUM_CONCURRENCY_BUCKETS],
    /// Prefill interference noted by the engine since this request's last gap
    /// draw: prompt tokens of a request admitted to prefill. Consumed by the
    /// next draw, which samples the stall distribution instead of the clean one.
    pending_stall: Option<u32>,
}

impl DecodePacing {
    /// Note that the engine admitted a prefill of `prompt_tokens` while this
    /// request decodes. A real prefill blocks one engine step, spiking exactly
    /// one gap per concurrent decode request, so the flag is consumed by a
    /// single draw. Multiple admissions before the next draw keep the largest.
    pub fn note_prefill(&mut self, prompt_tokens: u32) {
        self.pending_stall = Some(self.pending_stall.unwrap_or(0).max(prompt_tokens));
    }
}

impl Default for DecodePacing {
    fn default() -> Self {
        Self {
            donors: [None; NUM_CONCURRENCY_BUCKETS],
            pending_stall: None,
        }
    }
}

/// Strategy for pacing token emission (TTFT and inter-token delays).
pub trait LatencyModel: Send {
    fn first_token_delay(&self, rng: &mut StdRng, ctx: &FirstTokenCtx) -> Duration;
    fn inter_token_delay(&self, rng: &mut StdRng, num_running: u64) -> Duration;

    /// Inter-token delay with per-request pacing state. Stateless models fall
    /// back to the marginal distribution; hierarchical models use `pacing` to
    /// keep a request's gaps internally correlated.
    fn paced_inter_token_delay(
        &self,
        rng: &mut StdRng,
        num_running: u64,
        _pacing: &mut DecodePacing,
    ) -> Duration {
        self.inter_token_delay(rng, num_running)
    }
}

/// All timing knobs, in milliseconds (except the unitless `time_factor_under_load`).
/// Mirrors the `llm-d-inference-sim` configuration one-for-one. The default
/// [`LatencyModel`] behind every CLI-configured engine.
#[derive(Debug, Clone)]
pub struct KnobLatency {
    /// Fixed time-to-first-token. When this and its std-dev are both 0, the token-count
    /// prefill model (`prefill_overhead` + per-token) is used instead.
    pub time_to_first_token: u64,
    pub time_to_first_token_std_dev: u64,
    /// Time to generate one output token (decode).
    pub inter_token_latency: u64,
    pub inter_token_latency_std_dev: u64,
    /// Token-count prefill model: fixed overhead plus a per-(uncached-)prompt-token cost.
    pub prefill_overhead: u64,
    pub prefill_time_per_token: u64,
    pub prefill_time_std_dev: u64,
    /// Fixed KV-cache transfer time for a `do_remote_prefill` decode request. When this and
    /// its std-dev are both 0, the per-token transfer model is used instead.
    pub kv_cache_transfer_latency: u64,
    pub kv_cache_transfer_latency_std_dev: u64,
    pub kv_cache_transfer_time_per_token: u64,
    pub kv_cache_transfer_time_std_dev: u64,
    /// Latency multiplier at full load. `1.0` means load has no effect. Must be `>= 1.0`.
    pub time_factor_under_load: f64,
    /// Concurrency at which the load factor reaches `time_factor_under_load`.
    pub max_num_seqs: u64,
}

impl KnobLatency {
    /// Multiplier in `[1.0, time_factor_under_load]` that grows linearly with the number of
    /// running requests, reaching `time_factor_under_load` at `max_num_seqs` concurrent
    /// requests. With one request in flight (or `max_num_seqs <= 1`) it is exactly `1.0`.
    fn load_factor(&self, num_running: u64) -> f64 {
        if self.max_num_seqs <= 1 {
            return 1.0;
        }
        let extra = num_running.saturating_sub(1) as f64;
        1.0 + (self.time_factor_under_load - 1.0) * extra / (self.max_num_seqs - 1) as f64
    }
}

impl LatencyModel for KnobLatency {
    fn first_token_delay(&self, rng: &mut StdRng, ctx: &FirstTokenCtx) -> Duration {
        let load = self.load_factor(ctx.num_running);

        let millis = if ctx.do_remote_prefill {
            if self.kv_cache_transfer_latency == 0 && self.kv_cache_transfer_latency_std_dev == 0 {
                let mean = self.kv_cache_transfer_time_per_token * ctx.num_prompt_tokens as u64;
                random_norm_truncated(rng, mean, self.kv_cache_transfer_time_std_dev)
            } else {
                random_norm_truncated(
                    rng,
                    self.kv_cache_transfer_latency,
                    self.kv_cache_transfer_latency_std_dev,
                )
            }
        } else if self.time_to_first_token == 0 && self.time_to_first_token_std_dev == 0 {
            let uncached = ctx.num_prompt_tokens.saturating_sub(ctx.num_cached_tokens) as u64;
            let mean = scale(self.prefill_overhead, load)
                + uncached * scale(self.prefill_time_per_token, load);
            random_norm_truncated(rng, mean, self.prefill_time_std_dev)
        } else {
            random_norm_truncated(
                rng,
                scale(self.time_to_first_token, load),
                self.time_to_first_token_std_dev,
            )
        };

        Duration::from_millis(millis)
    }

    fn inter_token_delay(&self, rng: &mut StdRng, num_running: u64) -> Duration {
        let load = self.load_factor(num_running);
        let millis = random_norm_truncated(
            rng,
            scale(self.inter_token_latency, load),
            self.inter_token_latency_std_dev,
        );
        Duration::from_millis(millis)
    }
}

/// Apply the load factor to a millisecond mean, truncating toward zero.
fn scale(millis: u64, load: f64) -> u64 {
    (millis as f64 * load) as u64
}

/// Sample a normal deviate via the Box-Muller transform. Returns `mean` exactly when
/// `stddev == 0` (matching upstream, and keeping the zero-config path deterministic).
pub(crate) fn random_norm(rng: &mut StdRng, mean: f64, stddev: f64) -> f64 {
    if stddev == 0.0 {
        return mean;
    }
    // u1 in (0, 1]: nudge off zero so ln() stays finite.
    let u1: f64 = (1.0 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    let u2: f64 = rng.random::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    z * stddev + mean
}

/// Truncated-normal millisecond sample, clamped to `[0.3*mean, 1.7*mean]` like upstream's
/// `RandomNormTruncated`. A zero mean yields zero (instant).
fn random_norm_truncated(rng: &mut StdRng, mean: u64, stddev: u64) -> u64 {
    let mean_f = mean as f64;
    let value = random_norm(rng, mean_f, stddev as f64);
    let clamped = value.clamp(0.3 * mean_f, 1.7 * mean_f);
    clamped as u64
}

/// Constant first-token and inter-token delays. No rng draws, no load scaling.
/// Useful for deterministic timing tests where you want exact control.
pub struct FixedLatency {
    pub first_token: Duration,
    pub inter_token: Duration,
}

impl LatencyModel for FixedLatency {
    fn first_token_delay(&self, _rng: &mut StdRng, _ctx: &FirstTokenCtx) -> Duration {
        self.first_token
    }

    fn inter_token_delay(&self, _rng: &mut StdRng, _num_running: u64) -> Duration {
        self.inter_token
    }
}

// ---------------------------------------------------------------------------
// TraceLatency: replay latency model driven by recorded traces
// ---------------------------------------------------------------------------

use crate::trace::{TraceMeta, TraceRecord};

/// Prompt-token bucket edges (powers of two). A value `v` falls into bucket `i` where
/// `PROMPT_EDGES[i] <= v < PROMPT_EDGES[i+1]`. The last bucket is uncapped.
const PROMPT_EDGES: &[usize] = &[0, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769];

/// Number of prompt buckets (one per interval between edges, plus the uncapped tail).
const NUM_PROMPT_BUCKETS: usize = PROMPT_EDGES.len();

/// Map an uncached prompt token count to a bucket index.
pub fn prompt_bucket(uncached_prompt_tokens: usize) -> usize {
    // Find the last edge the value is >= to.
    let mut bucket = 0;
    for (i, &edge) in PROMPT_EDGES.iter().enumerate() {
        if uncached_prompt_tokens >= edge {
            bucket = i;
        } else {
            break;
        }
    }
    bucket
}

/// Concurrency bucket boundaries. A concurrency value maps to the first range it fits in.
pub const CONCURRENCY_RANGES: &[(u64, u64)] = &[
    (1, 1),
    (2, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, u64::MAX),
];

pub const NUM_CONCURRENCY_BUCKETS: usize = 7;

pub fn concurrency_bucket(concurrency: u64) -> usize {
    for (i, &(lo, hi)) in CONCURRENCY_RANGES.iter().enumerate() {
        if concurrency >= lo && concurrency <= hi {
            return i;
        }
    }
    NUM_CONCURRENCY_BUCKETS - 1
}

/// A grid cell holding sorted TTFT samples and pooled ITL samples.
#[derive(Debug, Clone, Default)]
struct Cell {
    ttft_samples: Vec<f64>,
    itl_samples: Vec<f64>,
}

/// One source request's gap distribution, used as a pacing donor.
#[derive(Debug, Clone)]
struct Donor {
    /// The request's own ITL gaps, sorted for inverse-CDF sampling. Summary-only
    /// records collapse to a single repeated mean.
    gaps: Vec<f64>,
}

/// Per-concurrency-bucket donor pool with token-count weighting, so the marginal
/// per-token distribution of hierarchical sampling matches the pooled samples.
#[derive(Debug, Clone, Default)]
struct DonorBucket {
    donors: Vec<Donor>,
    /// Cumulative token-count weights aligned with `donors`.
    cum_weights: Vec<f64>,
}

impl DonorBucket {
    fn push(&mut self, donor: Donor, weight: f64) {
        let total = self.cum_weights.last().copied().unwrap_or(0.0);
        self.donors.push(donor);
        self.cum_weights.push(total + weight);
    }

    /// Pick a donor index, weighted by token count.
    fn pick(&self, rng: &mut StdRng) -> usize {
        debug_assert!(!self.donors.is_empty());
        let total = self.cum_weights.last().copied().unwrap_or(0.0);
        let u: f64 = rng.random::<f64>() * total;
        self.cum_weights
            .partition_point(|&w| w <= u)
            .min(self.donors.len() - 1)
    }
}

/// Replay latency model: samples timing from a grid of recorded observations, bucketed
/// by (uncached prompt tokens, concurrency). For `do_remote_prefill` requests where the
/// trace has no transfer-time data, falls back to a wrapped `KnobLatency` for KV-transfer
/// timing only.
pub struct TraceLatency {
    /// Grid indexed by [prompt_bucket][concurrency_bucket].
    grid: Vec<Vec<Cell>>,
    /// Pooled ITL samples per concurrency bucket (union over all prompt buckets).
    itl_by_concurrency: Vec<Vec<f64>>,
    /// Per-request donor pools per concurrency bucket, for hierarchical pacing.
    /// When the trace carries `itl_ctx`, donors hold only CLEAN gaps (no prefill
    /// in the step); prefill-interfered gaps go to `stalls_by_concurrency`.
    donors_by_concurrency: Vec<DonorBucket>,
    /// Sorted prefill-interfered gap samples per concurrency bucket, drawn when
    /// the engine notes a prefill admission on the request's pacing state. Empty
    /// for traces without `itl_ctx`.
    stalls_by_concurrency: Vec<Vec<f64>>,
    /// Fallback for do_remote_prefill KV-transfer timing (the trace does not record
    /// transfer latencies from disaggregated decode pulls).
    kv_transfer_fallback: KnobLatency,
    #[allow(dead_code)]
    meta: TraceMeta,
}

impl TraceLatency {
    /// Build from parsed trace records. Bins records into the prompt x concurrency grid.
    /// Returns an error if the trace has no records.
    pub fn from_records(
        meta: TraceMeta,
        records: &[TraceRecord],
        kv_transfer_fallback: KnobLatency,
    ) -> anyhow::Result<Self> {
        if records.is_empty() {
            anyhow::bail!("trace has no records; cannot build a replay latency model");
        }

        let mut grid = vec![vec![Cell::default(); NUM_CONCURRENCY_BUCKETS]; NUM_PROMPT_BUCKETS];
        let mut donors_by_concurrency = vec![DonorBucket::default(); NUM_CONCURRENCY_BUCKETS];
        let mut stalls_by_concurrency = vec![Vec::new(); NUM_CONCURRENCY_BUCKETS];

        for record in records {
            let uncached = record.prompt_tokens.saturating_sub(record.cached_tokens);
            let pb = prompt_bucket(uncached);
            let cb = concurrency_bucket(record.concurrency);
            let cell = &mut grid[pb][cb];

            cell.ttft_samples.push(record.ttft_ms);

            // Expand ITL: prefer the array, fall back to the summary. Each record
            // also becomes a pacing donor weighted by its token count. With per-gap
            // context, prefill-interfered gaps leave the donor pool and feed the
            // stall distribution instead, so replay re-adds them only when the
            // simulated scheduler actually creates the interference.
            if let Some(itls) = &record.itl_ms {
                if !itls.is_empty() {
                    cell.itl_samples.extend(itls.iter().copied());
                    let mut gaps: Vec<f64> = match &record.itl_ctx {
                        Some(ctx) => {
                            let (stalled, clean): (Vec<usize>, Vec<usize>) =
                                (0..itls.len()).partition(|&i| ctx.prefill_tokens[i] > 0);
                            stalls_by_concurrency[cb].extend(stalled.iter().map(|&i| itls[i]));
                            clean.iter().map(|&i| itls[i]).collect()
                        }
                        None => itls.clone(),
                    };
                    if !gaps.is_empty() {
                        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let weight = gaps.len() as f64;
                        donors_by_concurrency[cb].push(Donor { gaps }, weight);
                    }
                }
            } else if let Some(summary) = &record.itl_summary
                && summary.count > 0
            {
                for _ in 0..summary.count {
                    cell.itl_samples.push(summary.mean_ms);
                }
                donors_by_concurrency[cb].push(
                    Donor {
                        gaps: vec![summary.mean_ms],
                    },
                    summary.count as f64,
                );
            }
        }
        for stalls in &mut stalls_by_concurrency {
            stalls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Sort TTFT samples for inverse-CDF sampling.
        for row in &mut grid {
            for cell in row.iter_mut() {
                cell.ttft_samples
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                cell.itl_samples
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // Pool ITL samples per concurrency bucket (union over prompt buckets).
        let mut itl_by_concurrency = vec![Vec::new(); NUM_CONCURRENCY_BUCKETS];
        for row in &grid {
            for (cb, cell) in row.iter().enumerate() {
                itl_by_concurrency[cb].extend(cell.itl_samples.iter().copied());
            }
        }
        for samples in &mut itl_by_concurrency {
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        // A trace where every record is a single-token output carries no decode pacing
        // information; refuse it up front rather than replaying instant inter-token times.
        if itl_by_concurrency.iter().all(|samples| samples.is_empty()) {
            anyhow::bail!(
                "trace contains no inter-token latency data (all records are single-token \
                 outputs); cannot pace decode from it"
            );
        }

        Ok(TraceLatency {
            grid,
            itl_by_concurrency,
            donors_by_concurrency,
            stalls_by_concurrency,
            kv_transfer_fallback,
            meta,
        })
    }
}

/// Sample from sorted samples using inverse CDF: draw u in [0,1), interpolate.
pub(crate) fn sample_inverse_cdf(rng: &mut StdRng, samples: &[f64]) -> f64 {
    debug_assert!(!samples.is_empty());
    if samples.len() == 1 {
        return samples[0];
    }
    let u: f64 = rng.random::<f64>();
    let pos = u * (samples.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(samples.len() - 1);
    let frac = pos - lo as f64;
    samples[lo] * (1.0 - frac) + samples[hi] * frac
}

/// Find the nearest non-empty cell by Manhattan distance on bucket indices, preferring
/// the same prompt bucket (tie-break). Returns None only if the entire grid is empty
/// (which from_records prevents).
fn nearest_ttft(grid: &[Vec<Cell>], pb: usize, cb: usize) -> Option<&[f64]> {
    if !grid[pb][cb].ttft_samples.is_empty() {
        return Some(&grid[pb][cb].ttft_samples);
    }
    let mut best: Option<(usize, usize, usize)> = None; // (distance, prompt_dist, indices)
    for (pi, row) in grid.iter().enumerate() {
        for (ci, cell) in row.iter().enumerate() {
            if cell.ttft_samples.is_empty() {
                continue;
            }
            let pd = pi.abs_diff(pb);
            let cd = ci.abs_diff(cb);
            let dist = pd + cd;
            match best {
                None => best = Some((dist, pd, pi * NUM_CONCURRENCY_BUCKETS + ci)),
                Some((bd, bpd, _)) => {
                    if dist < bd || (dist == bd && pd < bpd) {
                        best = Some((dist, pd, pi * NUM_CONCURRENCY_BUCKETS + ci));
                    }
                }
            }
        }
    }
    best.map(|(_, _, idx)| {
        let pi = idx / NUM_CONCURRENCY_BUCKETS;
        let ci = idx % NUM_CONCURRENCY_BUCKETS;
        grid[pi][ci].ttft_samples.as_slice()
    })
}

/// Find the nearest concurrency bucket with at least one donor.
fn nearest_donor_bucket(donors_by_concurrency: &[DonorBucket], cb: usize) -> Option<usize> {
    if !donors_by_concurrency[cb].donors.is_empty() {
        return Some(cb);
    }
    let mut best: Option<(usize, usize)> = None;
    for (ci, bucket) in donors_by_concurrency.iter().enumerate() {
        if bucket.donors.is_empty() {
            continue;
        }
        let dist = ci.abs_diff(cb);
        match best {
            None => best = Some((dist, ci)),
            Some((bd, _)) if dist < bd => best = Some((dist, ci)),
            _ => {}
        }
    }
    best.map(|(_, ci)| ci)
}

/// Find nearest non-empty ITL concurrency bucket.
fn nearest_itl(itl_by_concurrency: &[Vec<f64>], cb: usize) -> Option<&[f64]> {
    if !itl_by_concurrency[cb].is_empty() {
        return Some(&itl_by_concurrency[cb]);
    }
    let mut best: Option<(usize, usize)> = None;
    for (ci, samples) in itl_by_concurrency.iter().enumerate() {
        if samples.is_empty() {
            continue;
        }
        let dist = ci.abs_diff(cb);
        match best {
            None => best = Some((dist, ci)),
            Some((bd, _)) if dist < bd => best = Some((dist, ci)),
            _ => {}
        }
    }
    best.map(|(_, ci)| itl_by_concurrency[ci].as_slice())
}

impl LatencyModel for TraceLatency {
    fn first_token_delay(&self, rng: &mut StdRng, ctx: &FirstTokenCtx) -> Duration {
        if ctx.do_remote_prefill {
            return self.kv_transfer_fallback.first_token_delay(rng, ctx);
        }

        let uncached = ctx.num_prompt_tokens.saturating_sub(ctx.num_cached_tokens);
        let pb = prompt_bucket(uncached);
        let cb = concurrency_bucket(ctx.num_running);

        // from_records rejects traces with no records, so some cell is non-empty and the
        // nearest-cell search always resolves; the zero fallback is unreachable belt-and-braces.
        let Some(samples) = nearest_ttft(&self.grid, pb, cb) else {
            return Duration::ZERO;
        };
        let ms = sample_inverse_cdf(rng, samples);
        Duration::from_secs_f64(ms / 1000.0)
    }

    fn inter_token_delay(&self, rng: &mut StdRng, num_running: u64) -> Duration {
        let cb = concurrency_bucket(num_running);
        // from_records rejects traces with no ITL data, so some bucket is non-empty.
        let Some(samples) = nearest_itl(&self.itl_by_concurrency, cb) else {
            return Duration::ZERO;
        };
        let ms = sample_inverse_cdf(rng, samples);
        Duration::from_secs_f64(ms / 1000.0)
    }

    /// Hierarchical sampling: pin the request to one source request ("donor") per
    /// concurrency bucket, picked token-count-weighted on first use, then draw
    /// gaps from that donor's own distribution. Marginal per-token quantiles match
    /// the pooled path (same token weighting); within-request correlation matches
    /// the source (a slow request stays slow), so per-request decode totals
    /// reproduce instead of concentrating around the grand mean.
    fn paced_inter_token_delay(
        &self,
        rng: &mut StdRng,
        num_running: u64,
        pacing: &mut DecodePacing,
    ) -> Duration {
        let cb = concurrency_bucket(num_running);

        // A prefill admission noted by the engine spikes exactly one gap: draw it
        // from the recorded prefill-interfered distribution when the trace has one.
        // Without itl_ctx data the flag falls through to the clean path, whose
        // donor gaps still contain the stalls (they were never separated out).
        if pacing.pending_stall.take().is_some()
            && let Some(stalls) = nearest_itl(&self.stalls_by_concurrency, cb)
        {
            let ms = sample_inverse_cdf(rng, stalls);
            return Duration::from_secs_f64(ms / 1000.0);
        }

        // The resolved bucket is deterministic for a given cb, so the cached donor
        // index below always refers to the same bucket's pool.
        let Some(bucket_idx) = nearest_donor_bucket(&self.donors_by_concurrency, cb) else {
            return self.inter_token_delay(rng, num_running);
        };
        let bucket = &self.donors_by_concurrency[bucket_idx];
        let donor_idx = match pacing.donors[cb] {
            Some(idx) => idx as usize,
            None => {
                let idx = bucket.pick(rng);
                pacing.donors[cb] = Some(idx as u32);
                idx
            }
        };
        let ms = sample_inverse_cdf(rng, &bucket.donors[donor_idx].gaps);
        Duration::from_secs_f64(ms / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng as _;

    use crate::latency::{
        DecodePacing, FirstTokenCtx, KnobLatency, LatencyModel, random_norm_truncated,
    };

    fn model() -> KnobLatency {
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
            max_num_seqs: 5,
        }
    }

    fn ctx(prompt: usize, cached: usize, remote: bool, running: u64) -> FirstTokenCtx {
        FirstTokenCtx {
            num_prompt_tokens: prompt,
            num_cached_tokens: cached,
            do_remote_prefill: remote,
            num_running: running,
        }
    }

    #[test]
    fn zero_config_is_instant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let m = model();
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(100, 0, false, 1)),
            std::time::Duration::ZERO
        );
        assert_eq!(m.inter_token_delay(&mut rng, 1), std::time::Duration::ZERO);
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(100, 0, true, 1)),
            std::time::Duration::ZERO
        );
    }

    #[test]
    fn fixed_ttft_no_stddev_is_exact() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut m = model();
        m.time_to_first_token = 200;
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(50, 0, false, 1)),
            std::time::Duration::from_millis(200)
        );
    }

    #[test]
    fn token_count_prefill_model() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut m = model();
        m.prefill_overhead = 100;
        m.prefill_time_per_token = 2;
        // overhead + (prompt - cached) * per_token = 100 + (50 - 10) * 2 = 180.
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(50, 10, false, 1)),
            std::time::Duration::from_millis(180)
        );
    }

    #[test]
    fn remote_prefill_uses_transfer_per_token() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut m = model();
        m.kv_cache_transfer_time_per_token = 3;
        m.prefill_overhead = 999;
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(20, 0, true, 1)),
            std::time::Duration::from_millis(60)
        );
    }

    #[test]
    fn fixed_transfer_latency_overrides_per_token() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut m = model();
        m.kv_cache_transfer_latency = 500;
        m.kv_cache_transfer_time_per_token = 3;
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(20, 0, true, 1)),
            std::time::Duration::from_millis(500)
        );
    }

    #[test]
    fn load_factor_scales_between_one_and_max() {
        let mut m = model();
        m.time_factor_under_load = 3.0;
        m.max_num_seqs = 5;
        assert_eq!(m.load_factor(1), 1.0);
        assert_eq!(m.load_factor(5), 3.0);
        assert_eq!(m.load_factor(3), 2.0);
    }

    #[test]
    fn load_factor_disabled_when_single_slot() {
        let mut m = model();
        m.time_factor_under_load = 9.0;
        m.max_num_seqs = 1;
        assert_eq!(m.load_factor(4), 1.0);
    }

    #[test]
    fn fixed_ttft_scales_with_load() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut m = model();
        m.time_to_first_token = 100;
        m.time_factor_under_load = 2.0;
        m.max_num_seqs = 5;
        assert_eq!(
            m.first_token_delay(&mut rng, &ctx(10, 0, false, 5)),
            std::time::Duration::from_millis(200)
        );
    }

    #[test]
    fn truncated_normal_stays_within_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..10_000 {
            let v = random_norm_truncated(&mut rng, 100, 50);
            assert!((30..=170).contains(&v), "out of truncation bounds: {v}");
        }
    }

    #[test]
    fn fixed_latency_ignores_context_and_rng() {
        use crate::latency::FixedLatency;

        let fixed = FixedLatency {
            first_token: std::time::Duration::from_millis(42),
            inter_token: std::time::Duration::from_millis(7),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);

        // TTFT is constant regardless of prompt size, cache, remote, or concurrency.
        for &(prompt, cached, remote, running) in
            &[(100, 0, false, 1), (1, 1, true, 999), (0, 0, false, 0)]
        {
            assert_eq!(
                fixed.first_token_delay(&mut rng, &ctx(prompt, cached, remote, running)),
                std::time::Duration::from_millis(42),
            );
        }

        // Inter-token is constant regardless of concurrency.
        for &num_running in &[1, 10, 1000] {
            assert_eq!(
                fixed.inter_token_delay(&mut rng, num_running),
                std::time::Duration::from_millis(7),
            );
        }
    }

    // -----------------------------------------------------------------------
    // TraceLatency tests
    // -----------------------------------------------------------------------

    use crate::latency::TraceLatency;
    use crate::trace::{ItlSummary, TraceMeta, TraceRecord};

    fn zero_knob() -> KnobLatency {
        model()
    }

    fn make_record(
        prompt: usize,
        output: usize,
        ttft: f64,
        itl: Vec<f64>,
        conc: u64,
    ) -> TraceRecord {
        TraceRecord {
            prompt_tokens: prompt,
            cached_tokens: 0,
            output_tokens: output,
            ttft_ms: ttft,
            itl_ms: if itl.is_empty() { None } else { Some(itl) },
            itl_summary: None,
            concurrency: conc,
            arrival_ms: None,
            itl_ctx: None,
        }
    }

    #[test]
    fn trace_empty_records_is_error() {
        let result = TraceLatency::from_records(TraceMeta::default(), &[], zero_knob());
        assert!(result.is_err());
    }

    #[test]
    fn trace_constant_ttft_reproduces_exactly() {
        // All records in the same bucket have the same TTFT; sampling must return it exactly.
        let records = vec![
            make_record(50, 2, 42.0, vec![9.0], 1),
            make_record(60, 1, 42.0, vec![], 1),
        ];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for _ in 0..100 {
            let d = trace.first_token_delay(&mut rng, &ctx(55, 0, false, 1));
            assert_eq!(d, std::time::Duration::from_secs_f64(0.042));
        }
    }

    #[test]
    fn trace_samples_stay_within_min_max() {
        let records = vec![
            make_record(50, 3, 10.0, vec![5.0, 15.0], 1),
            make_record(60, 3, 20.0, vec![8.0, 12.0], 1),
            make_record(40, 3, 15.0, vec![6.0, 10.0], 1),
        ];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let ttft = trace.first_token_delay(&mut rng, &ctx(50, 0, false, 1));
            let ms = ttft.as_secs_f64() * 1000.0;
            assert!(
                (10.0 - 0.001..=20.0 + 0.001).contains(&ms),
                "ttft out of range: {ms}"
            );

            let itl = trace.inter_token_delay(&mut rng, 1);
            let ms = itl.as_secs_f64() * 1000.0;
            assert!(
                (5.0 - 0.001..=15.0 + 0.001).contains(&ms),
                "itl out of range: {ms}"
            );
        }
    }

    #[test]
    fn trace_determinism_same_seed() {
        let records = vec![
            make_record(100, 5, 25.0, vec![3.0, 4.0, 5.0, 6.0], 2),
            make_record(120, 3, 35.0, vec![7.0, 8.0], 2),
        ];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();

        let run = |seed: u64| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut results = Vec::new();
            for _ in 0..20 {
                results.push(trace.first_token_delay(&mut rng, &ctx(110, 0, false, 2)));
                results.push(trace.inter_token_delay(&mut rng, 2));
            }
            results
        };

        assert_eq!(
            run(42),
            run(42),
            "same seed must produce identical sequence"
        );
        assert_ne!(
            run(42),
            run(99),
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn trace_fallback_to_nearest_bucket() {
        // Only populate a single cell in the grid: prompt bucket 0 (0-64), concurrency bucket 0 (1).
        let records = vec![make_record(30, 2, 77.0, vec![11.0], 1)];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);

        // Query a very different bucket (large prompt, high concurrency). Should fall back
        // to the only populated cell.
        let d = trace.first_token_delay(&mut rng, &ctx(10000, 0, false, 50));
        assert_eq!(d, std::time::Duration::from_secs_f64(0.077));

        let itl = trace.inter_token_delay(&mut rng, 50);
        assert_eq!(itl, std::time::Duration::from_secs_f64(0.011));
    }

    #[test]
    fn trace_itl_summary_expansion() {
        // Use itl_summary instead of itl_ms; verify it expands correctly.
        let records = vec![TraceRecord {
            prompt_tokens: 50,
            cached_tokens: 0,
            output_tokens: 6,
            ttft_ms: 20.0,
            itl_ms: None,
            itl_summary: Some(ItlSummary {
                mean_ms: 9.0,
                count: 5,
            }),
            concurrency: 1,
            arrival_ms: None,
            itl_ctx: None,
        }];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);

        // All ITL samples are 9.0, so sampling always returns 9.0.
        for _ in 0..50 {
            let itl = trace.inter_token_delay(&mut rng, 1);
            assert_eq!(itl, std::time::Duration::from_secs_f64(0.009));
        }
    }

    #[test]
    fn stall_conditioning_draws_from_interfered_gaps_once() {
        // One source request whose trace marks gaps 3 and 7 as prefill-interfered
        // (150ms) among clean 10ms gaps.
        let gaps = vec![10.0, 10.0, 10.0, 150.0, 10.0, 10.0, 10.0, 150.0, 10.0];
        let prefill = vec![0u32, 0, 0, 800, 0, 0, 0, 800, 0];
        let records = vec![TraceRecord {
            prompt_tokens: 100,
            cached_tokens: 0,
            output_tokens: 10,
            ttft_ms: 40.0,
            itl_ms: Some(gaps),
            itl_summary: None,
            concurrency: 4,
            arrival_ms: None,
            itl_ctx: Some(crate::trace::ItlContext {
                num_running: vec![4; 9],
                prefill_tokens: prefill,
            }),
        }];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut pacing = DecodePacing::default();

        let draw = |rng: &mut rand::rngs::StdRng, pacing: &mut DecodePacing| -> f64 {
            trace.paced_inter_token_delay(rng, 4, pacing).as_secs_f64() * 1000.0
        };

        // Clean draws never produce stall gaps: those left the donor pool.
        for _ in 0..30 {
            let ms = draw(&mut rng, &mut pacing);
            assert!(
                (ms - 10.0).abs() < 1e-9,
                "clean draw should be 10ms, got {ms}"
            );
        }

        // A noted prefill spikes exactly the next draw, then the flag is spent.
        pacing.note_prefill(800);
        let stalled = draw(&mut rng, &mut pacing);
        assert!(
            (stalled - 150.0).abs() < 1e-9,
            "stall draw should come from interfered gaps, got {stalled}"
        );
        let after = draw(&mut rng, &mut pacing);
        assert!(
            (after - 10.0).abs() < 1e-9,
            "flag must be consumed, got {after}"
        );
    }

    #[test]
    fn paced_sampling_sticks_to_one_donor_per_request() {
        // Two donor requests with disjoint gap levels. A single request's paced
        // draws must all come from ONE donor (within-request correlation), while
        // many requests must hit both donors (across-request variation).
        let records = vec![
            make_record(100, 10, 40.0, vec![5.0; 9], 4),
            make_record(100, 10, 40.0, vec![50.0; 9], 4),
        ];
        let trace =
            TraceLatency::from_records(TraceMeta::default(), &records, zero_knob()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        let mut seen_fast_request = false;
        let mut seen_slow_request = false;
        for _ in 0..50 {
            let mut pacing = DecodePacing::default();
            let draws: Vec<f64> = (0..20)
                .map(|_| {
                    trace
                        .paced_inter_token_delay(&mut rng, 4, &mut pacing)
                        .as_secs_f64()
                        * 1000.0
                })
                .collect();
            let all_fast = draws.iter().all(|&d| (d - 5.0).abs() < 1e-9);
            let all_slow = draws.iter().all(|&d| (d - 50.0).abs() < 1e-9);
            assert!(
                all_fast || all_slow,
                "one request's draws must stay at one donor's level, got {draws:?}"
            );
            seen_fast_request |= all_fast;
            seen_slow_request |= all_slow;
        }
        assert!(
            seen_fast_request && seen_slow_request,
            "across requests both donors must be picked"
        );

        // The stateless marginal path still mixes both levels.
        let mut seen_fast_gap = false;
        let mut seen_slow_gap = false;
        for _ in 0..100 {
            let ms = trace.inter_token_delay(&mut rng, 4).as_secs_f64() * 1000.0;
            seen_fast_gap |= (ms - 5.0).abs() < 1.0;
            seen_slow_gap |= (ms - 50.0).abs() < 1.0;
        }
        assert!(seen_fast_gap && seen_slow_gap);
    }

    #[test]
    fn trace_without_itl_data_is_rejected() {
        // Single-token outputs carry no decode pacing data; the model must refuse to build
        // rather than silently replay instant inter-token times.
        let records = vec![
            make_record(50, 1, 42.0, vec![], 1),
            make_record(60, 1, 42.0, vec![], 1),
        ];
        let result = TraceLatency::from_records(TraceMeta::default(), &records, zero_knob());
        let error = result.err().expect("ITL-less trace must be rejected");
        assert!(error.to_string().contains("inter-token"), "got: {error}");
    }

    #[test]
    fn trace_remote_prefill_delegates_to_knob() {
        let records = vec![make_record(50, 2, 10.0, vec![1.0], 1)];
        let mut knob = zero_knob();
        knob.kv_cache_transfer_latency = 500;
        let trace = TraceLatency::from_records(TraceMeta::default(), &records, knob).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);

        let d = trace.first_token_delay(&mut rng, &ctx(50, 0, true, 1));
        assert_eq!(d, std::time::Duration::from_millis(500));
    }

    #[test]
    fn prompt_bucket_edges() {
        use crate::latency::prompt_bucket;
        assert_eq!(prompt_bucket(0), 0);
        assert_eq!(prompt_bucket(64), 0);
        assert_eq!(prompt_bucket(65), 1);
        assert_eq!(prompt_bucket(128), 1);
        assert_eq!(prompt_bucket(129), 2);
        assert_eq!(prompt_bucket(50000), 10); // last bucket
    }

    #[test]
    fn concurrency_bucket_edges() {
        use crate::latency::concurrency_bucket;
        assert_eq!(concurrency_bucket(1), 0);
        assert_eq!(concurrency_bucket(2), 1);
        assert_eq!(concurrency_bucket(4), 1);
        assert_eq!(concurrency_bucket(5), 2);
        assert_eq!(concurrency_bucket(8), 2);
        assert_eq!(concurrency_bucket(9), 3);
        assert_eq!(concurrency_bucket(65), 6);
        assert_eq!(concurrency_bucket(1000), 6);
    }
}
