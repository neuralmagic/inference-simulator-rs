//! Step-level scheduler-stats sidecar stream.
//!
//! Separate from the per-request trace because these snapshots are step- and
//! batch-level: nothing here can be attributed to a single request, and the
//! payload carries vLLM's `SchedulerStats` verbatim off the wire. Keeping it
//! out of `trace` lets the trace schema stay free of the engine-core protocol
//! dependency.

use std::io::{BufRead, Write};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
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
}
