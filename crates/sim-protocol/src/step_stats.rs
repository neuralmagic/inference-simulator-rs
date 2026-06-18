//! Step-level scheduler-stats sidecar stream.
//!
//! Separate from the per-request trace because these snapshots are step- and
//! batch-level: nothing here can be attributed to a single request, and the
//! payload carries vLLM's `SchedulerStats` verbatim off the wire. It lives in
//! this crate (not the protocol-free `sim-trace`) because it depends on the
//! engine-core `SchedulerStats` type; the tap writes the stream during capture,
//! and the offline perfetto converter reads it back into counter tracks.

use std::io::{BufRead, Write};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use sim_trace::perfetto::{CounterSeries, StepSpan};
use vllm_engine_core_client::protocol::stats::SchedulerStats;

/// One scheduler-stats snapshot observed during capture, written to the
/// step-stats sidecar stream (one JSONL line per engine output message that
/// carried stats). These are step-level and batch-level: nothing here can be
/// attributed to a single request, which is why they live next to the trace
/// instead of inside `TraceRecord`. Under speculative decoding or diffusion
/// blocks, `scheduler.spec_decoding_stats` carries the per-step draft and
/// acceptance counts (including per-position acceptance), the raw material
/// for fitting an acceptance model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepStatsRecord {
    /// Milliseconds since capture start (same zero point as
    /// `TraceRecord::arrival_ms`).
    pub ts_ms: f64,
    /// The engine's scheduler stats, verbatim from the wire.
    pub scheduler: SchedulerStats,
}

/// Append one step-stats snapshot to the sidecar stream.
pub fn append_step_stats(writer: &mut impl Write, record: &StepStatsRecord) -> Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writeln!(writer)?;
    Ok(())
}

/// Turn a step-stats sidecar stream into Perfetto counter tracks, on the same
/// capture-start clock as the trace. These are the batch-level series that don't
/// fit on any single request's track: scheduler queue depths, KV-cache pressure,
/// and (under spec decode) the live acceptance rate. Empty series are skipped, so
/// `accept_rate` only shows up when the run actually drafted tokens.
pub fn step_stats_counters(records: &[StepStatsRecord]) -> Vec<CounterSeries> {
    let mut running = Vec::with_capacity(records.len());
    let mut waiting = Vec::with_capacity(records.len());
    let mut kv_usage = Vec::with_capacity(records.len());
    let mut accept_rate = Vec::new();

    for record in records {
        let ts = record.ts_ms;
        let s = &record.scheduler;
        running.push((ts, s.num_running_reqs as f64));
        waiting.push((ts, s.num_waiting_reqs as f64));
        kv_usage.push((ts, s.kv_cache_usage));
        // Acceptance rate is only meaningful on steps that drafted tokens.
        if let Some(spec) = &s.spec_decoding_stats
            && spec.num_draft_tokens > 0
        {
            accept_rate.push((
                ts,
                spec.num_accepted_tokens as f64 / spec.num_draft_tokens as f64,
            ));
        }
    }

    [
        ("sched_running_reqs", running),
        ("sched_waiting_reqs", waiting),
        ("sched_kv_cache_usage", kv_usage),
        ("sched_accept_rate", accept_rate),
    ]
    .into_iter()
    .filter(|(_, samples)| !samples.is_empty())
    .map(|(name, samples)| CounterSeries {
        name: name.to_string(),
        samples,
    })
    .collect()
}

/// Turn the step-stats stream into one [`StepSpan`] per executed scheduler
/// step, for the step-centric Perfetto track. Each record reports the stats of
/// the step that just finished, so step `i`'s wall-clock interval is
/// `[ts[i-1], ts[i]]`; the prefill signal is the per-step prefix-cache counters
/// (`requests`/`queries` are reset each step, nonzero only when a prefill ran).
/// The first record has no prior boundary, so it seeds the clock and is not
/// emitted as a span (one step out of thousands).
pub fn step_spans(records: &[StepStatsRecord]) -> Vec<StepSpan> {
    records
        .windows(2)
        .map(|pair| {
            let (prev, cur) = (&pair[0], &pair[1]);
            let s = &cur.scheduler;
            let pcs = &s.prefix_cache_stats.base;
            StepSpan {
                start_ms: prev.ts_ms,
                end_ms: cur.ts_ms,
                running: s.num_running_reqs as u32,
                waiting: s.num_waiting_reqs as u32,
                prefill_requests: pcs.requests as u32,
                prefill_tokens: pcs.queries as u32,
                kv_cache_usage: s.kv_cache_usage,
                spec: s
                    .spec_decoding_stats
                    .as_ref()
                    .map(|sp| (sp.num_accepted_tokens as u32, sp.num_draft_tokens as u32)),
            }
        })
        .collect()
}

/// Read step-stats records from any JSONL reader, skipping blank lines.
pub fn read_step_stats(reader: impl BufRead) -> Result<Vec<StepStatsRecord>> {
    let mut records = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading step-stats line {}", i + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: StepStatsRecord = serde_json::from_str(trimmed)
            .with_context(|| format!("step-stats line {}: invalid record", i + 1))?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use std::io;

    use vllm_engine_core_client::protocol::stats::SpecDecodingStats;

    use super::*;

    #[test]
    fn step_stats_round_trip() {
        let record = StepStatsRecord {
            ts_ms: 123.5,
            scheduler: SchedulerStats {
                num_running_reqs: 3,
                num_waiting_reqs: 1,
                kv_cache_usage: 0.25,
                spec_decoding_stats: Some(SpecDecodingStats {
                    num_spec_tokens: 3,
                    num_drafts: 3,
                    num_draft_tokens: 9,
                    num_accepted_tokens: 5,
                    num_accepted_tokens_per_pos: vec![3, 1, 1],
                }),
                ..Default::default()
            },
        };
        let mut buf = Vec::new();
        append_step_stats(&mut buf, &record).unwrap();
        append_step_stats(&mut buf, &record).unwrap();

        let records = read_step_stats(io::BufReader::new(buf.as_slice())).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].ts_ms, 123.5);
        assert_eq!(records[0].scheduler.num_running_reqs, 3);
        let spec = records[0].scheduler.spec_decoding_stats.as_ref().unwrap();
        assert_eq!(spec.num_accepted_tokens, 5);
        assert_eq!(spec.num_accepted_tokens_per_pos, vec![3, 1, 1]);
    }

    #[test]
    fn counters_carry_core_series_and_spec_accept_rate() {
        let records = vec![
            StepStatsRecord {
                ts_ms: 0.0,
                scheduler: SchedulerStats {
                    num_running_reqs: 2,
                    num_waiting_reqs: 1,
                    kv_cache_usage: 0.25,
                    ..Default::default()
                },
            },
            StepStatsRecord {
                ts_ms: 10.0,
                scheduler: SchedulerStats {
                    num_running_reqs: 3,
                    num_waiting_reqs: 0,
                    kv_cache_usage: 0.5,
                    spec_decoding_stats: Some(SpecDecodingStats {
                        num_draft_tokens: 8,
                        num_accepted_tokens: 6,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            },
        ];
        let series: std::collections::HashMap<_, _> = step_stats_counters(&records)
            .into_iter()
            .map(|c| (c.name, c.samples))
            .collect();

        assert_eq!(series["sched_running_reqs"], vec![(0.0, 2.0), (10.0, 3.0)]);
        assert_eq!(series["sched_waiting_reqs"], vec![(0.0, 1.0), (10.0, 0.0)]);
        assert_eq!(
            series["sched_kv_cache_usage"],
            vec![(0.0, 0.25), (10.0, 0.5)]
        );
        // Only the second step drafted tokens, so accept_rate has one sample.
        assert_eq!(series["sched_accept_rate"], vec![(10.0, 6.0 / 8.0)]);
    }

    #[test]
    fn step_spans_bracket_steps_and_classify_prefill() {
        use vllm_engine_core_client::protocol::stats::{BaseCacheStats, PrefixCacheStats};
        let pcs = |requests: u64, queries: u64| PrefixCacheStats {
            base: BaseCacheStats {
                reset: false,
                requests,
                queries,
                hits: 0,
            },
            ..Default::default()
        };
        let records = vec![
            StepStatsRecord {
                ts_ms: 0.0,
                scheduler: SchedulerStats {
                    num_running_reqs: 16,
                    ..Default::default()
                },
            },
            // Step ending at 12ms: pure decode (no prefill counters).
            StepStatsRecord {
                ts_ms: 12.0,
                scheduler: SchedulerStats {
                    num_running_reqs: 16,
                    ..Default::default()
                },
            },
            // Step ending at 51ms: a prefill ran (requests/queries nonzero).
            StepStatsRecord {
                ts_ms: 51.0,
                scheduler: SchedulerStats {
                    num_running_reqs: 14,
                    prefix_cache_stats: pcs(2, 1600),
                    ..Default::default()
                },
            },
        ];
        let spans = step_spans(&records);
        // Two spans for three records (first seeds the clock).
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].start_ms, spans[0].end_ms), (0.0, 12.0));
        assert_eq!(spans[0].prefill_requests, 0); // decode step
        assert_eq!((spans[1].start_ms, spans[1].end_ms), (12.0, 51.0));
        assert_eq!(spans[1].prefill_requests, 2);
        assert_eq!(spans[1].prefill_tokens, 1600);
    }

    #[test]
    fn accept_rate_absent_without_spec_decode() {
        let records = vec![StepStatsRecord {
            ts_ms: 0.0,
            scheduler: SchedulerStats {
                num_running_reqs: 1,
                ..Default::default()
            },
        }];
        let names: Vec<_> = step_stats_counters(&records)
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert!(!names.contains(&"sched_accept_rate".to_string()));
    }
}
