//! Token-generation strategy for the engine.
//!
//! The `TokenSource` trait decouples *what tokens a request emits* from the engine loop
//! that paces them. The default `RandomTokens` reproduces the original behavior: uniform
//! random draws from `0..vocab_size` using the per-request seeded rng. `ReplayTokens`
//! serves the output ids recorded in a trace (tap `--record-tokens`), making replayed
//! streams content-identical to the capture. `HFDatasetTokens` loads a HuggingFace dataset
//! in memory (via `--replay-tokens`) and serves tokenized responses, matching rows by
//! request id or prompt block-hash prefix per `--replay-match`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use hf_hub::api::sync::Api;
use rand::Rng as _;
use rand::rngs::StdRng;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};
use vllm_engine_core_client::protocol::EngineCoreFinishReason;

use crate::trace::TraceRecord;
use crate::ReplayMatch;

/// Context passed to `TokenSource::next_tokens` so implementations can condition on
/// request state without holding a reference to the full `ActiveRequest`.
#[allow(dead_code)]
pub(crate) struct TokenCtx<'a> {
    pub request_id: &'a str,
    pub prompt_token_ids: &'a [u32],
    pub num_generated: usize,
}

/// Strategy for producing output tokens for a request.
///
/// `rng` is the per-request seeded rng; implementations that do not need randomness must
/// still not draw from it (callers rely on a fixed draw order for determinism).
pub(crate) trait TokenSource: Send {
    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32>;

    /// Called at admission, before any generation. Sources that pin a request
    /// to a recorded stream by inspecting its prompt do the matching here and
    /// return the recorded output length, which the engine uses to clamp
    /// `max_tokens` so a live client's stream ends exactly where the capture
    /// did. The default matches nothing.
    fn on_request_added(&mut self, _request_id: &str, _prompt_token_ids: &[u32]) -> Option<usize> {
        None
    }

    /// Called when a request finishes or is aborted, so stateful sources can drop any
    /// per-request bookkeeping. The default is a no-op.
    fn on_request_finished(&mut self, _request_id: &str) {}

    /// The finish reason this request's stream should end with, when the source knows
    /// it (replayed traces record one). `None` keeps the engine's own semantics
    /// (`Length` at max_tokens).
    fn finish_reason(&self, _request_id: &str) -> Option<EngineCoreFinishReason> {
        None
    }
}

/// The original token strategy: each token is drawn uniformly from `0..vocab_size`.
pub(crate) struct RandomTokens {
    pub vocab_size: u32,
}

impl TokenSource for RandomTokens {
    fn next_tokens(&mut self, _ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            tokens.push(rng.random_range(0..self.vocab_size));
        }
        tokens
    }
}

/// One trace record's recorded output, in canonical replay order.
struct RecordedOutput {
    /// The recorded output ids; empty when the capture did not record tokens
    /// for this request (the fallback source covers it).
    token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
}

/// Serves the output token ids recorded in a trace, making a replayed stream
/// content-identical to the capture.
///
/// A request resolves to its record through the trailing `-<index>` of its
/// request id (the arrival-replay harness names them `replay-{i}`), where the
/// index is the record's position in [`crate::trace::replay_subset`] order.
/// Requests that don't resolve, and generation past the end of a record's ids,
/// fall back to random tokens.
pub(crate) struct ReplayTokens {
    records: Vec<RecordedOutput>,
    fallback: RandomTokens,
}

impl ReplayTokens {
    /// Build from records already in [`crate::trace::replay_subset`] order.
    pub(crate) fn from_records(records: &[TraceRecord], vocab_size: u32) -> ReplayTokens {
        ReplayTokens {
            records: records
                .iter()
                .map(|r| RecordedOutput {
                    token_ids: r.output_token_ids.clone().unwrap_or_default(),
                    finish_reason: r.finish_reason.map(crate::wire::engine_finish_reason),
                })
                .collect(),
            fallback: RandomTokens { vocab_size },
        }
    }

    /// The replay-subset index encoded in a request id: everything after the
    /// last `-` parsed as an integer (the whole id when there is no `-`).
    pub(crate) fn record_index(request_id: &str) -> Option<usize> {
        let tail = request_id.rsplit_once('-').map_or(request_id, |(_, t)| t);
        tail.parse().ok()
    }

    fn record(&self, request_id: &str) -> Option<&RecordedOutput> {
        Self::record_index(request_id).and_then(|i| self.records.get(i))
    }
}

impl TokenSource for ReplayTokens {
    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32> {
        let recorded = match self.record(ctx.request_id) {
            Some(record) => {
                let start = ctx.num_generated.min(record.token_ids.len());
                let end = (ctx.num_generated + n).min(record.token_ids.len());
                &record.token_ids[start..end]
            }
            None => &[],
        };
        let mut tokens = recorded.to_vec();
        // The client asked for more than was recorded (or nothing was recorded
        // for this request): pad with random draws rather than starving it.
        if tokens.len() < n {
            tokens.extend(self.fallback.next_tokens(ctx, n - tokens.len(), rng));
        }
        tokens
    }

    fn finish_reason(&self, request_id: &str) -> Option<EngineCoreFinishReason> {
        self.record(request_id).and_then(|r| r.finish_reason)
    }
}

/// Serves recorded outputs to *live* requests by matching each incoming
/// prompt's block-hash chain against the trace, instead of trusting request
/// ids. This is what lets a real client (an agent loop re-run offline) talk to
/// the sim through a real frontend and get back the captured streams.
///
/// Matching: the incoming prompt is hashed with [`crate::trace::prompt_block_hashes`]
/// (same chain the tap stored in `TraceRecord::block_hashes`). Because block
/// hashes are chained, one hash value identifies the entire prefix up to its
/// block, so the deepest incoming hash that any record contains IS the
/// longest-common-prefix match. Noise at the prompt tail (a timing string in
/// the last tool output) only shortens the match depth; it doesn't change
/// which record wins.
///
/// Each record is consumed by its first match so duplicate prompts map 1:1 to
/// duplicate records in arrival order. When every candidate at the deepest
/// depth is already consumed (a client retry after an abort), the earliest one
/// is re-served rather than falling back to random tokens.
pub(crate) struct PrefixMatchTokens {
    records: Vec<RecordedOutput>,
    /// Every hash in every record's chain, mapped to the records containing it
    /// (ascending = arrival order, the tie-break).
    by_hash: HashMap<u64, Vec<usize>>,
    /// Live request id -> matched record, assigned at admission.
    assigned: HashMap<String, usize>,
    consumed: Vec<bool>,
    block_size: usize,
    fallback: RandomTokens,
}

impl PrefixMatchTokens {
    /// Build from records in arrival order. Records without `block_hashes` or
    /// recorded token ids can never match and are not indexed.
    pub(crate) fn from_records(
        records: &[TraceRecord],
        block_size: usize,
        vocab_size: u32,
    ) -> PrefixMatchTokens {
        let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, record) in records.iter().enumerate() {
            let matchable = record
                .output_token_ids
                .as_ref()
                .is_some_and(|t| !t.is_empty());
            if !matchable {
                continue;
            }
            for &hash in record.block_hashes.iter().flatten() {
                by_hash.entry(hash).or_default().push(i);
            }
        }
        PrefixMatchTokens {
            records: records
                .iter()
                .map(|r| RecordedOutput {
                    token_ids: r.output_token_ids.clone().unwrap_or_default(),
                    finish_reason: r.finish_reason.map(crate::wire::engine_finish_reason),
                })
                .collect(),
            by_hash,
            assigned: HashMap::new(),
            consumed: vec![false; records.len()],
            block_size,
            fallback: RandomTokens { vocab_size },
        }
    }

    fn record(&self, request_id: &str) -> Option<&RecordedOutput> {
        self.assigned
            .get(request_id)
            .and_then(|&i| self.records.get(i))
    }
}

impl TokenSource for PrefixMatchTokens {
    fn on_request_added(&mut self, request_id: &str, prompt_token_ids: &[u32]) -> Option<usize> {
        let Some(chain) = crate::trace::prompt_block_hashes(prompt_token_ids, self.block_size)
        else {
            warn!(
                request_id,
                prompt_tokens = prompt_token_ids.len(),
                "prompt shorter than one block; cannot prefix-match, serving random tokens"
            );
            return None;
        };
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
            self.assigned.insert(request_id.to_string(), idx);
            debug!(
                request_id,
                record = idx,
                matched_blocks = depth + 1,
                prompt_blocks = chain.len(),
                "prefix-matched request to trace record"
            );
            return Some(self.records[idx].token_ids.len());
        }
        warn!(
            request_id,
            prompt_blocks = chain.len(),
            "no trace record shares a prompt prefix; serving random tokens"
        );
        None
    }

    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32> {
        let recorded = match self.record(ctx.request_id) {
            Some(record) => {
                let start = ctx.num_generated.min(record.token_ids.len());
                let end = (ctx.num_generated + n).min(record.token_ids.len());
                &record.token_ids[start..end]
            }
            None => &[],
        };
        let mut tokens = recorded.to_vec();
        if tokens.len() < n {
            tokens.extend(self.fallback.next_tokens(ctx, n - tokens.len(), rng));
        }
        tokens
    }

    fn on_request_finished(&mut self, request_id: &str) {
        // The record stays consumed: a later identical prompt is a genuine
        // duplicate and should take the next record, not replay this one.
        self.assigned.remove(request_id);
    }

    fn finish_reason(&self, request_id: &str) -> Option<EngineCoreFinishReason> {
        self.record(request_id).and_then(|r| r.finish_reason)
    }
}

/// One dataset row with tokenized prompt and response.
struct DatasetRow {
    #[allow(dead_code)]
    prompt_token_ids: Vec<u32>,
    block_hashes: Option<Vec<u64>>,
    response_tokens: Vec<u32>,
}

/// Serves tokens from a HuggingFace dataset loaded in memory at init.
///
/// Prompt and response text are tokenized with the GPT-2 tokenizer. Requests
/// resolve to rows via [`ReplayMatch`]: by trailing request-id index (`index`)
/// or by longest block-hash prefix of the incoming prompt (`prefix`).
pub(crate) struct HFDatasetTokens {
    rows: Vec<DatasetRow>,
    assignments: HashMap<String, usize>,
    positions: HashMap<String, usize>,
    by_hash: HashMap<u64, Vec<usize>>,
    consumed: Vec<bool>,
    replay_match: ReplayMatch,
    block_size: usize,
    fallback: RandomTokens,
}

impl HFDatasetTokens {
    /// Load a dataset file and tokenize prompts and responses using GPT-2 from HuggingFace.
    pub(crate) fn from_file(
        dataset_path: &Path,
        block_size: usize,
        vocab_size: u32,
        replay_match: ReplayMatch,
    ) -> Result<Self> {
        info!(
            dataset = %dataset_path.display(),
            replay_match = ?replay_match,
            "loading dataset with GPT-2 tokenizer from HuggingFace (openai-community/gpt2)"
        );

        let api = Api::new()
            .map_err(|e| anyhow::anyhow!("initializing HuggingFace API: {}", e))?;
        let repo = api.model("openai-community/gpt2".to_string());
        let tokenizer_file = repo.get("tokenizer.json").map_err(|e| {
            anyhow::anyhow!("downloading GPT-2 tokenizer from HuggingFace: {}", e)
        })?;

        info!(
            tokenizer_cache = %tokenizer_file.display(),
            "GPT-2 tokenizer cached"
        );

        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("loading GPT-2 tokenizer: {}", e))?;

        let json_rows = sim_trace::dataset_convert::parse_dataset(dataset_path)
            .with_context(|| format!("parsing dataset {}", dataset_path.display()))?;

        if json_rows.is_empty() {
            anyhow::bail!("dataset {} contains no rows", dataset_path.display());
        }

        let mut rows = Vec::with_capacity(json_rows.len());
        for (idx, row) in json_rows.iter().enumerate() {
            let prompt_text = sim_trace::dataset_convert::extract_prompt(row)
                .with_context(|| format!("extracting prompt from row {idx}"))?;
            let response_text = sim_trace::dataset_convert::extract_response(row)
                .with_context(|| format!("extracting response from row {idx}"))?;

            let prompt_encoding = tokenizer
                .encode(prompt_text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing prompt for row {idx}: {e}"))?;
            let prompt_token_ids: Vec<u32> = prompt_encoding.get_ids().to_vec();

            let response_encoding = tokenizer
                .encode(response_text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing response for row {idx}: {e}"))?;
            let response_tokens: Vec<u32> = response_encoding.get_ids().to_vec();

            if response_tokens.is_empty() {
                warn!(row = idx, "row has empty response after tokenization, skipping");
                continue;
            }

            let block_hashes =
                crate::trace::prompt_block_hashes(&prompt_token_ids, block_size);

            rows.push(DatasetRow {
                prompt_token_ids,
                block_hashes,
                response_tokens,
            });
        }

        if rows.is_empty() {
            anyhow::bail!(
                "dataset {} has no rows with tokenizable responses",
                dataset_path.display()
            );
        }

        let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
        if replay_match == ReplayMatch::Prefix {
            for (i, row) in rows.iter().enumerate() {
                for &hash in row.block_hashes.iter().flatten() {
                    by_hash.entry(hash).or_default().push(i);
                }
            }
            if by_hash.is_empty() {
                anyhow::bail!(
                    "dataset {} has no rows with a full prompt block (block_size={}); \
                     prefix matching requires at least one block of prompt tokens per row",
                    dataset_path.display(),
                    block_size,
                );
            }
        }

        info!(rows = rows.len(), "loaded dataset with tokenized prompts and responses");

        let row_count = rows.len();
        Ok(HFDatasetTokens {
            rows,
            assignments: HashMap::new(),
            positions: HashMap::new(),
            by_hash,
            consumed: vec![false; row_count],
            replay_match,
            block_size,
            fallback: RandomTokens { vocab_size },
        })
    }

    fn row(&self, request_id: &str) -> Option<&DatasetRow> {
        self.assignments
            .get(request_id)
            .and_then(|&idx| self.rows.get(idx))
    }

    fn assign_index(&mut self, request_id: &str, row_idx: usize) -> Option<usize> {
        self.assignments.insert(request_id.to_string(), row_idx);
        self.positions.insert(request_id.to_string(), 0);
        debug!(
            request_id,
            dataset_row = row_idx,
            tokens = self.rows[row_idx].response_tokens.len(),
            "assigned request to dataset row by index"
        );
        Some(self.rows[row_idx].response_tokens.len())
    }

    fn match_prefix(&mut self, request_id: &str, prompt_token_ids: &[u32]) -> Option<usize> {
        let chain = crate::trace::prompt_block_hashes(prompt_token_ids, self.block_size)?;
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
            self.assignments.insert(request_id.to_string(), idx);
            self.positions.insert(request_id.to_string(), 0);
            debug!(
                request_id,
                dataset_row = idx,
                matched_blocks = depth + 1,
                prompt_blocks = chain.len(),
                "prefix-matched request to dataset row"
            );
            return Some(self.rows[idx].response_tokens.len());
        }
        warn!(
            request_id,
            prompt_blocks = chain.len(),
            "no dataset row shares a prompt prefix; serving random tokens"
        );
        None
    }

    #[cfg(test)]
    fn from_rows_for_test(
        rows: Vec<DatasetRow>,
        block_size: usize,
        vocab_size: u32,
        replay_match: ReplayMatch,
    ) -> Self {
        let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
        if replay_match == ReplayMatch::Prefix {
            for (i, row) in rows.iter().enumerate() {
                for &hash in row.block_hashes.iter().flatten() {
                    by_hash.entry(hash).or_default().push(i);
                }
            }
        }
        let consumed_len = rows.len();
        HFDatasetTokens {
            rows,
            assignments: HashMap::new(),
            positions: HashMap::new(),
            by_hash,
            consumed: vec![false; consumed_len],
            replay_match,
            block_size,
            fallback: RandomTokens { vocab_size },
        }
    }
}

impl TokenSource for HFDatasetTokens {
    fn on_request_added(&mut self, request_id: &str, prompt_token_ids: &[u32]) -> Option<usize> {
        match self.replay_match {
            ReplayMatch::Index => {
                let row_idx = ReplayTokens::record_index(request_id)?;
                if row_idx >= self.rows.len() {
                    return None;
                }
                self.assign_index(request_id, row_idx)
            }
            ReplayMatch::Prefix => self.match_prefix(request_id, prompt_token_ids),
        }
    }

    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32> {
        let row = match self.row(ctx.request_id) {
            Some(r) => r,
            None => {
                return self.fallback.next_tokens(ctx, n, rng);
            }
        };

        let pos = self.positions.get(ctx.request_id).copied().unwrap_or(0);
        let available = &row.response_tokens[pos.min(row.response_tokens.len())..];
        let count = n.min(available.len());

        let mut tokens = available[..count].to_vec();

        // Update position
        if let Some(p) = self.positions.get_mut(ctx.request_id) {
            *p += count;
        }

        // Pad with fallback if we've exhausted the dataset row
        if tokens.len() < n {
            tokens.extend(self.fallback.next_tokens(ctx, n - tokens.len(), rng));
        }

        tokens
    }

    fn on_request_finished(&mut self, request_id: &str) {
        self.assignments.remove(request_id);
        self.positions.remove(request_id);
    }

    fn finish_reason(&self, request_id: &str) -> Option<EngineCoreFinishReason> {
        let row = self.row(request_id)?;
        let pos = self.positions.get(request_id).copied().unwrap_or(0);

        if pos >= row.response_tokens.len() {
            Some(EngineCoreFinishReason::Stop)
        } else {
            None
        }
    }
}

/// Replays the request's prompt tokens as output, cycling from the start when
/// `max_tokens` exceeds the prompt length. Draws nothing from the rng.
#[cfg(test)]
pub(crate) struct EchoTokens;

#[cfg(test)]
impl TokenSource for EchoTokens {
    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, _rng: &mut StdRng) -> Vec<u32> {
        let prompt = ctx.prompt_token_ids;
        if prompt.is_empty() {
            return vec![0; n];
        }
        (0..n)
            .map(|i| {
                let idx = (ctx.num_generated + i) % prompt.len();
                prompt[idx]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng as _;

    use super::{DatasetRow, HFDatasetTokens, RandomTokens, TokenCtx, TokenSource};

    #[test]
    fn random_tokens_draws_correct_count() {
        let mut src = RandomTokens { vocab_size: 100 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let ctx = TokenCtx {
            request_id: "t1",
            prompt_token_ids: &[0, 1, 2],
            num_generated: 0,
        };
        let tokens = src.next_tokens(&ctx, 5, &mut rng);
        assert_eq!(tokens.len(), 5);
        assert!(tokens.iter().all(|&t| t < 100));
    }

    #[test]
    fn random_tokens_deterministic_with_same_seed() {
        let make = || {
            let mut src = RandomTokens { vocab_size: 32000 };
            let mut rng = rand::rngs::StdRng::seed_from_u64(7);
            let ctx = TokenCtx {
                request_id: "det",
                prompt_token_ids: &[0],
                num_generated: 0,
            };
            src.next_tokens(&ctx, 10, &mut rng)
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn on_request_finished_default_is_noop() {
        let mut src = RandomTokens { vocab_size: 10 };
        // Should not panic.
        src.on_request_finished("anything");
    }

    fn replay_source() -> crate::tokens::ReplayTokens {
        use crate::trace::{TraceFinishReason, TraceRecord};
        let records = vec![
            TraceRecord {
                prompt_tokens: 4,
                output_tokens: 5,
                ttft_ms: 1.0,
                itl_ms: Some(vec![1.0; 4]),
                output_token_ids: Some(vec![100, 101, 102, 103, 104]),
                finish_reason: Some(TraceFinishReason::Stop),
                ..Default::default()
            },
            // Captured without --record-tokens: reason only, no ids.
            TraceRecord {
                prompt_tokens: 4,
                output_tokens: 2,
                ttft_ms: 1.0,
                itl_ms: Some(vec![1.0]),
                finish_reason: Some(TraceFinishReason::Length),
                ..Default::default()
            },
        ];
        crate::tokens::ReplayTokens::from_records(&records, 50)
    }

    #[test]
    fn replay_tokens_serves_recorded_ids_at_the_generation_offset() {
        let mut src = replay_source();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let ctx = TokenCtx {
            request_id: "replay-0",
            prompt_token_ids: &[],
            num_generated: 0,
        };
        assert_eq!(src.next_tokens(&ctx, 2, &mut rng), vec![100, 101]);
        let ctx = TokenCtx {
            request_id: "replay-0",
            prompt_token_ids: &[],
            num_generated: 2,
        };
        assert_eq!(src.next_tokens(&ctx, 3, &mut rng), vec![102, 103, 104]);
    }

    #[test]
    fn replay_tokens_pads_past_recorded_end_with_fallback() {
        let mut src = replay_source();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let ctx = TokenCtx {
            request_id: "replay-0",
            prompt_token_ids: &[],
            num_generated: 3,
        };
        // 2 recorded ids remain; the other 3 come from the fallback (vocab 50).
        let tokens = src.next_tokens(&ctx, 5, &mut rng);
        assert_eq!(tokens.len(), 5);
        assert_eq!(&tokens[..2], &[103, 104]);
        assert!(tokens[2..].iter().all(|&t| t < 50));
    }

    #[test]
    fn replay_tokens_falls_back_for_unmatched_and_tokenless_requests() {
        let mut src = replay_source();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for id in ["not-a-replay-id", "replay-99", "replay-1"] {
            let ctx = TokenCtx {
                request_id: id,
                prompt_token_ids: &[],
                num_generated: 0,
            };
            let tokens = src.next_tokens(&ctx, 4, &mut rng);
            assert_eq!(tokens.len(), 4, "{id}");
            assert!(tokens.iter().all(|&t| t < 50), "{id}");
        }
    }

    #[test]
    fn replay_tokens_reports_recorded_finish_reasons() {
        use vllm_engine_core_client::protocol::EngineCoreFinishReason;
        let src = replay_source();
        assert_eq!(
            src.finish_reason("replay-0"),
            Some(EngineCoreFinishReason::Stop)
        );
        assert_eq!(
            src.finish_reason("replay-1"),
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(src.finish_reason("replay-99"), None);
        assert_eq!(src.finish_reason("other"), None);
    }

    #[test]
    fn default_finish_reason_is_none() {
        let src = RandomTokens { vocab_size: 10 };
        assert_eq!(src.finish_reason("anything"), None);
    }

    /// Two multi-turn records over a shared prefix: turn 1's prompt is blocks
    /// A|B, turn 2's prompt extends it to A|B|C|D (the agentic append-only
    /// shape). Block size 4.
    fn prefix_records() -> (Vec<u32>, Vec<u32>, crate::tokens::PrefixMatchTokens) {
        use crate::trace::{TraceFinishReason, TraceRecord, prompt_block_hashes};
        let turn1: Vec<u32> = (0..8).collect();
        let turn2: Vec<u32> = (0..16).collect();
        let records = vec![
            TraceRecord {
                prompt_tokens: turn1.len(),
                output_tokens: 3,
                block_hashes: prompt_block_hashes(&turn1, 4),
                output_token_ids: Some(vec![100, 101, 102]),
                finish_reason: Some(TraceFinishReason::Stop),
                ..Default::default()
            },
            TraceRecord {
                prompt_tokens: turn2.len(),
                output_tokens: 2,
                block_hashes: prompt_block_hashes(&turn2, 4),
                output_token_ids: Some(vec![200, 201]),
                finish_reason: Some(TraceFinishReason::Length),
                ..Default::default()
            },
        ];
        let src = crate::tokens::PrefixMatchTokens::from_records(&records, 4, 50);
        (turn1, turn2, src)
    }

    fn drain(src: &mut dyn TokenSource, request_id: &str, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let ctx = TokenCtx {
            request_id,
            prompt_token_ids: prompt,
            num_generated: 0,
        };
        src.next_tokens(&ctx, n, &mut rng)
    }

    #[test]
    fn prefix_match_serves_recorded_ids_and_reports_length() {
        use vllm_engine_core_client::protocol::EngineCoreFinishReason;
        let (turn1, _, mut src) = prefix_records();
        assert_eq!(src.on_request_added("live-abc", &turn1), Some(3));
        assert_eq!(drain(&mut src, "live-abc", &turn1, 3), vec![100, 101, 102]);
        assert_eq!(
            src.finish_reason("live-abc"),
            Some(EngineCoreFinishReason::Stop)
        );
    }

    #[test]
    fn prefix_match_prefers_deepest_record() {
        // Turn 2's prompt also contains turn 1's chain as a prefix; the
        // deeper record must win, not the first indexed one.
        let (_, turn2, mut src) = prefix_records();
        assert_eq!(src.on_request_added("live-t2", &turn2), Some(2));
        assert_eq!(drain(&mut src, "live-t2", &turn2, 2), vec![200, 201]);
    }

    #[test]
    fn prefix_match_survives_tail_divergence() {
        // Same first three blocks as turn 2, but the last block differs (a
        // timing string in the final tool output). Still matches turn 2.
        let (_, turn2, mut src) = prefix_records();
        let mut noisy = turn2.clone();
        for t in &mut noisy[12..] {
            *t += 1000;
        }
        assert_eq!(src.on_request_added("live-noisy", &noisy), Some(2));
        assert_eq!(drain(&mut src, "live-noisy", &noisy, 2), vec![200, 201]);
    }

    #[test]
    fn prefix_match_consumes_records_in_arrival_order_and_reserves_when_dry() {
        use crate::trace::{TraceFinishReason, TraceRecord, prompt_block_hashes};
        let prompt: Vec<u32> = (0..8).collect();
        let record = |ids: Vec<u32>| TraceRecord {
            prompt_tokens: prompt.len(),
            output_tokens: ids.len(),
            block_hashes: prompt_block_hashes(&prompt, 4),
            output_token_ids: Some(ids),
            finish_reason: Some(TraceFinishReason::Stop),
            ..Default::default()
        };
        let records = vec![record(vec![1, 2]), record(vec![3, 4])];
        let mut src = crate::tokens::PrefixMatchTokens::from_records(&records, 4, 50);
        // Identical prompts consume distinct records in arrival order.
        assert_eq!(src.on_request_added("a", &prompt), Some(2));
        assert_eq!(src.on_request_added("b", &prompt), Some(2));
        assert_eq!(drain(&mut src, "a", &prompt, 2), vec![1, 2]);
        assert_eq!(drain(&mut src, "b", &prompt, 2), vec![3, 4]);
        // All consumed: a retry re-serves the earliest match instead of going random.
        src.on_request_finished("a");
        src.on_request_finished("b");
        assert_eq!(src.on_request_added("retry", &prompt), Some(2));
        assert_eq!(drain(&mut src, "retry", &prompt, 2), vec![1, 2]);
    }

    #[test]
    fn prefix_match_unmatched_requests_fall_back_to_random() {
        let (_, _, mut src) = prefix_records();
        // Shares no block with the trace.
        let alien: Vec<u32> = (5000..5016).collect();
        assert_eq!(src.on_request_added("alien", &alien), None);
        let tokens = drain(&mut src, "alien", &alien, 4);
        assert_eq!(tokens.len(), 4);
        assert!(tokens.iter().all(|&t| t < 50));
        // Shorter than one block: no chain to match on.
        assert_eq!(src.on_request_added("tiny", &[1, 2]), None);
    }

    #[test]
    fn prefix_match_skips_records_without_tokens_or_hashes() {
        use crate::trace::{TraceFinishReason, TraceRecord, prompt_block_hashes};
        let prompt: Vec<u32> = (0..8).collect();
        let records = vec![
            // Hash-only capture (no --record-tokens): must never match.
            TraceRecord {
                prompt_tokens: prompt.len(),
                output_tokens: 2,
                block_hashes: prompt_block_hashes(&prompt, 4),
                finish_reason: Some(TraceFinishReason::Stop),
                ..Default::default()
            },
            // Tokens but no hashes (short prompt at capture time): unreachable.
            TraceRecord {
                prompt_tokens: 2,
                output_tokens: 2,
                output_token_ids: Some(vec![7, 8]),
                finish_reason: Some(TraceFinishReason::Stop),
                ..Default::default()
            },
        ];
        let mut src = crate::tokens::PrefixMatchTokens::from_records(&records, 4, 50);
        assert_eq!(src.on_request_added("x", &prompt), None);
    }

    #[test]
    fn echo_tokens_replays_prompt_ids() {
        use crate::tokens::EchoTokens;
        let mut src = EchoTokens;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let ctx = TokenCtx {
            request_id: "echo",
            prompt_token_ids: &prompt,
            num_generated: 0,
        };
        let tokens = src.next_tokens(&ctx, 8, &mut rng);
        assert_eq!(tokens, prompt, "first 8 tokens echo the prompt exactly");
    }

    #[test]
    fn echo_tokens_cycles_when_exceeding_prompt_len() {
        use crate::tokens::EchoTokens;
        let mut src = EchoTokens;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let prompt: Vec<u32> = vec![1, 2, 3];
        // Already generated 3 (one full cycle), now generating 4 more: should cycle.
        let ctx = TokenCtx {
            request_id: "cycle",
            prompt_token_ids: &prompt,
            num_generated: 3,
        };
        let tokens = src.next_tokens(&ctx, 4, &mut rng);
        assert_eq!(tokens, vec![1, 2, 3, 1]);
    }

    #[test]
    fn echo_tokens_empty_prompt_returns_zeros() {
        use crate::tokens::EchoTokens;
        let mut src = EchoTokens;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let ctx = TokenCtx {
            request_id: "empty",
            prompt_token_ids: &[],
            num_generated: 0,
        };
        let tokens = src.next_tokens(&ctx, 3, &mut rng);
        assert_eq!(tokens, vec![0, 0, 0]);
    }

    #[test]
    fn hf_dataset_rows_carry_prompt_block_hashes() {
        use crate::trace::prompt_block_hashes;
        let prompt: Vec<u32> = (0..8).collect();
        let row = DatasetRow {
            prompt_token_ids: prompt.clone(),
            block_hashes: prompt_block_hashes(&prompt, 4),
            response_tokens: vec![1, 2, 3],
        };
        assert_eq!(row.prompt_token_ids.len(), 8);
        assert_eq!(row.block_hashes.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn hf_dataset_index_match_serves_row_by_request_id() {
        use crate::ReplayMatch;
        let prompt0: Vec<u32> = (0..8).collect();
        let prompt1: Vec<u32> = (100..108).collect();
        let rows = vec![
            DatasetRow {
                prompt_token_ids: prompt0.clone(),
                block_hashes: crate::trace::prompt_block_hashes(&prompt0, 4),
                response_tokens: vec![10, 11, 12],
            },
            DatasetRow {
                prompt_token_ids: prompt1.clone(),
                block_hashes: crate::trace::prompt_block_hashes(&prompt1, 4),
                response_tokens: vec![20, 21],
            },
        ];
        let mut src = HFDatasetTokens::from_rows_for_test(rows, 4, 50, ReplayMatch::Index);
        assert_eq!(src.on_request_added("replay-0", &prompt0), Some(3));
        assert_eq!(drain(&mut src, "replay-0", &prompt0, 3), vec![10, 11, 12]);
        assert_eq!(src.on_request_added("replay-1", &prompt1), Some(2));
        assert_eq!(drain(&mut src, "replay-1", &prompt1, 2), vec![20, 21]);
    }

    #[test]
    fn hf_dataset_prefix_match_serves_matching_row() {
        use crate::ReplayMatch;
        let prompt: Vec<u32> = (0..8).collect();
        let rows = vec![DatasetRow {
            prompt_token_ids: prompt.clone(),
            block_hashes: crate::trace::prompt_block_hashes(&prompt, 4),
            response_tokens: vec![42, 43],
        }];
        let mut src = HFDatasetTokens::from_rows_for_test(rows, 4, 50, ReplayMatch::Prefix);
        assert_eq!(src.on_request_added("live", &prompt), Some(2));
        assert_eq!(drain(&mut src, "live", &prompt, 2), vec![42, 43]);
    }

    #[test]
    fn hf_dataset_index_unmatched_falls_back_to_random() {
        use crate::ReplayMatch;
        let prompt: Vec<u32> = (0..8).collect();
        let rows = vec![DatasetRow {
            prompt_token_ids: prompt.clone(),
            block_hashes: crate::trace::prompt_block_hashes(&prompt, 4),
            response_tokens: vec![1, 2],
        }];
        let mut src = HFDatasetTokens::from_rows_for_test(rows, 4, 50, ReplayMatch::Index);
        assert_eq!(src.on_request_added("replay-99", &prompt), None);
        let tokens = drain(&mut src, "replay-99", &prompt, 3);
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&t| t < 50));
    }
}
