//! Verbatim decode-schedule replay: the timing/framing analogue of the
//! content replay in [`crate::tokens`].
//!
//! `--latency-trace` builds a *modeled* latency, per-request gaps and burst
//! sizes are sampled from donor pools fitted to a capture, so a model fit on
//! one workload transfers to a different one. `--replay-steps` is the opposite
//! axis, *replay*: each matched request emits its recorded per-chunk token
//! counts at its recorded gaps, bit-for-bit. Use it as a fidelity ceiling and
//! for exact reproduction of multi-token steps (speculative decoding, diffusion
//! blocks).
//!
//! Matching mirrors [`crate::tokens`] and shares the `--replay-match` flag:
//! `Index` trusts the request id's trailing `-<i>` (our own arrival-replay
//! harness), `Prefix` matches the prompt's block-hash chain (a live client),
//! consume-once.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use tracing::{debug, warn};

use crate::latency::StepDraw;
use crate::trace::{TraceRecord, prompt_block_hashes};

/// One request's recorded decode timeline, consumed step by step at replay.
#[derive(Debug)]
pub(crate) struct ScriptedDecode {
    /// Size of the first output chunk (the prefill step's emission), recorded
    /// as `output_tokens - sum(itl_tokens)`. Usually 1 for autoregressive and
    /// speculative decoding; a whole block for diffusion.
    pub first_chunk_tokens: u32,
    /// Decode steps after the first chunk, in order: each carries the recorded
    /// gap and the token count that chunk delivered.
    pub steps: VecDeque<StepDraw>,
}

impl ScriptedDecode {
    /// Total recorded output length, used to clamp the live request's
    /// `max_tokens` so the replayed stream ends exactly where the capture did.
    pub fn output_tokens(&self) -> usize {
        self.first_chunk_tokens as usize
            + self.steps.iter().map(|s| s.tokens as usize).sum::<usize>()
    }
}

/// Build a request's decode schedule from its trace record. `None` when the
/// record carries no per-chunk timing (`itl_ms`) to replay.
fn schedule_from_record(r: &TraceRecord) -> Option<ScriptedDecode> {
    let gaps = r.itl_ms.as_ref()?;
    // `itl_tokens` is parallel to `itl_ms`; absent means one token per gap.
    let tokens: Vec<u32> = match &r.itl_tokens {
        Some(t) if t.len() == gaps.len() => t.clone(),
        Some(_) => return None,
        None => vec![1; gaps.len()],
    };
    let decode_total: usize = tokens.iter().map(|&t| t as usize).sum();
    // The first chunk owns the remainder (`output_tokens - sum(itl_tokens)`);
    // at least one token rode the prefill step.
    let first_chunk_tokens = r.output_tokens.saturating_sub(decode_total).max(1) as u32;
    let steps = gaps
        .iter()
        .zip(&tokens)
        .map(|(&ms, &t)| StepDraw {
            delay: Duration::from_secs_f64((ms / 1000.0).max(0.0)),
            tokens: t.max(1),
        })
        .collect();
    Some(ScriptedDecode {
        first_chunk_tokens,
        steps,
    })
}

/// Configured speculative budget K = widest recorded burst - 1, or `None` when
/// every recorded chunk is a single token. Mirrors `TraceLatency::spec_tokens`
/// so a `--replay-steps` run emits `spec_decoding_stats` even without a
/// `--latency-trace` model.
fn derive_spec_tokens(records: &[TraceRecord]) -> Option<u32> {
    let max_chunk = records
        .iter()
        .filter_map(|r| r.itl_tokens.as_ref())
        .flat_map(|t| t.iter().copied())
        .max()
        .unwrap_or(1);
    (max_chunk > 1).then(|| max_chunk - 1)
}

/// Hands an incoming request its verbatim decode schedule at admission.
pub(crate) trait StepSource: Send {
    /// Match the request and return its decode schedule, or `None` to leave it
    /// on the modeled latency path.
    fn take_schedule(
        &mut self,
        request_id: &str,
        prompt_token_ids: &[u32],
    ) -> Option<ScriptedDecode>;

    /// Configured speculative budget K for `spec_decoding_stats`, when the
    /// replayed bursts are multi-token.
    fn spec_tokens(&self) -> Option<u32>;
}

/// Resolves a request to its record by the trailing `-<index>` of its id (the
/// arrival-replay harness names requests `replay-{i}`), where the index is the
/// record's position in [`crate::trace::replay_subset`] order.
pub(crate) struct IndexSteps {
    records: Vec<TraceRecord>,
    spec_tokens: Option<u32>,
}

impl IndexSteps {
    pub(crate) fn from_records(records: Vec<TraceRecord>) -> IndexSteps {
        let spec_tokens = derive_spec_tokens(&records);
        IndexSteps {
            records,
            spec_tokens,
        }
    }

    fn record_index(request_id: &str) -> Option<usize> {
        let tail = request_id.rsplit_once('-').map_or(request_id, |(_, t)| t);
        tail.parse().ok()
    }
}

impl StepSource for IndexSteps {
    fn take_schedule(&mut self, request_id: &str, _prompt: &[u32]) -> Option<ScriptedDecode> {
        let i = Self::record_index(request_id)?;
        schedule_from_record(self.records.get(i)?)
    }

    fn spec_tokens(&self) -> Option<u32> {
        self.spec_tokens
    }
}

/// Resolves a *live* request to its record by longest block-hash prefix, the
/// same consume-once matching as [`crate::tokens::PrefixMatchTokens`]. Only
/// records carrying both `itl_ms` and `block_hashes` are matchable.
pub(crate) struct PrefixSteps {
    records: Vec<TraceRecord>,
    by_hash: HashMap<u64, Vec<usize>>,
    consumed: Vec<bool>,
    block_size: usize,
    spec_tokens: Option<u32>,
}

impl PrefixSteps {
    pub(crate) fn from_records(records: Vec<TraceRecord>, block_size: usize) -> PrefixSteps {
        let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, record) in records.iter().enumerate() {
            if record.itl_ms.is_none() {
                continue;
            }
            for &hash in record.block_hashes.iter().flatten() {
                by_hash.entry(hash).or_default().push(i);
            }
        }
        let spec_tokens = derive_spec_tokens(&records);
        let consumed = vec![false; records.len()];
        PrefixSteps {
            records,
            by_hash,
            consumed,
            block_size,
            spec_tokens,
        }
    }
}

impl StepSource for PrefixSteps {
    fn take_schedule(
        &mut self,
        request_id: &str,
        prompt_token_ids: &[u32],
    ) -> Option<ScriptedDecode> {
        let chain = prompt_block_hashes(prompt_token_ids, self.block_size)?;
        // Deepest hash first: chaining makes the first hit the longest-prefix match.
        for (depth, hash) in chain.iter().enumerate().rev() {
            let Some(candidates) = self.by_hash.get(hash) else {
                continue;
            };
            let idx = candidates
                .iter()
                .copied()
                .find(|&i| !self.consumed[i])
                .or_else(|| candidates.first().copied())?;
            self.consumed[idx] = true;
            debug!(
                request_id,
                record = idx,
                matched_blocks = depth + 1,
                "prefix-matched request to recorded decode schedule"
            );
            return schedule_from_record(&self.records[idx]);
        }
        warn!(
            request_id,
            prompt_blocks = chain.len(),
            "no trace record shares a prompt prefix; decode stays on the modeled latency path"
        );
        None
    }

    fn spec_tokens(&self) -> Option<u32> {
        self.spec_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{TraceRecord, prompt_block_hashes};

    fn record(prompt: usize, out: usize, gaps: Vec<f64>, toks: Option<Vec<u32>>) -> TraceRecord {
        TraceRecord {
            prompt_tokens: prompt,
            output_tokens: out,
            ttft_ms: 5.0,
            itl_ms: Some(gaps),
            itl_tokens: toks,
            ..Default::default()
        }
    }

    #[test]
    fn schedule_reproduces_recorded_bursts_and_first_chunk() {
        // out=10: first chunk 1, then bursts 4+5 = 9.
        let r = record(8, 10, vec![10.0, 12.0], Some(vec![4, 5]));
        let s = schedule_from_record(&r).unwrap();
        assert_eq!(s.first_chunk_tokens, 1);
        assert_eq!(s.output_tokens(), 10);
        let sizes: Vec<u32> = s.steps.iter().map(|d| d.tokens).collect();
        assert_eq!(sizes, vec![4, 5]);
        let gaps_ms: Vec<f64> = s
            .steps
            .iter()
            .map(|d| d.delay.as_secs_f64() * 1000.0)
            .collect();
        assert!((gaps_ms[0] - 10.0).abs() < 1e-6 && (gaps_ms[1] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn diffusion_first_chunk_owns_the_remainder() {
        // 320 tokens in 64-token blocks: first chunk 64, then four 64s.
        let r = record(64, 320, vec![80.0; 4], Some(vec![64; 4]));
        let s = schedule_from_record(&r).unwrap();
        assert_eq!(s.first_chunk_tokens, 64);
        assert_eq!(s.output_tokens(), 320);
    }

    #[test]
    fn autoregressive_record_yields_single_token_steps() {
        let r = record(8, 5, vec![10.0; 4], None);
        let s = schedule_from_record(&r).unwrap();
        assert_eq!(s.first_chunk_tokens, 1);
        assert!(s.steps.iter().all(|d| d.tokens == 1));
    }

    #[test]
    fn index_match_resolves_by_request_id_tail() {
        let recs = vec![
            record(8, 5, vec![10.0; 4], Some(vec![1, 1, 1, 1])),
            record(8, 10, vec![10.0, 12.0], Some(vec![4, 5])),
        ];
        let mut src = IndexSteps::from_records(recs);
        assert_eq!(src.spec_tokens(), Some(4)); // max chunk 5 - 1
        let s = src.take_schedule("replay-1", &[]).unwrap();
        assert_eq!(s.output_tokens(), 10);
        assert!(src.take_schedule("replay-99", &[]).is_none());
    }

    #[test]
    fn prefix_match_consumes_once_in_arrival_order() {
        let prompt: Vec<u32> = (0..8).collect();
        let mk = || {
            let mut r = record(8, 10, vec![10.0, 12.0], Some(vec![4, 5]));
            r.block_hashes = prompt_block_hashes(&prompt, 4);
            r
        };
        let mut src = PrefixSteps::from_records(vec![mk(), mk()], 4);
        assert!(src.take_schedule("a", &prompt).is_some());
        assert!(src.take_schedule("b", &prompt).is_some());
        // Both records consumed; a third request re-serves the earliest.
        assert!(src.take_schedule("c", &prompt).is_some());
        // A prompt sharing no block matches nothing.
        let alien: Vec<u32> = (900..908).collect();
        assert!(src.take_schedule("alien", &alien).is_none());
    }
}
