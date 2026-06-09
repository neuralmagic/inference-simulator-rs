//! The per-engine generation loop. Adapted from vLLM's in-tree `vllm-mock-engine`
//! (`rust/src/mock-engine/src/engine.rs`), with the prefill/decode data-plane hooks
//! added at the two points where real KV bytes would move.
//!
//! Everything wire-facing comes from the `vllm-engine-core-client` crate, so this
//! stays correct as the protocol evolves upstream.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value as MsgpackValue;
use serde::Serialize;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::protocol::stats::{
    BaseCacheStats, PrefillStats, PrefixCacheStats, SchedulerStats,
};
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
};

use crate::blockpool::BlockPool;
use crate::dataplane::{KvDataPlane, NixlConfig, RemoteKv, RequestKv, make_data_plane};
use crate::kvevents::{self, KvEventTx};
use crate::latency::LatencyModel;
use crate::{Opt, SchedulingPolicy};

/// Per-step token demand of a single prefilling request: a (possibly chunked) slice of its
/// prompt, capped by the chunked-prefill threshold and the overall token budget. At least 1
/// so a request always makes progress.
fn prefill_token_demand(
    prompt_len: usize,
    long_prefill_threshold: usize,
    max_batched_tokens: usize,
) -> usize {
    let chunk = if long_prefill_threshold > 0 {
        prompt_len.min(long_prefill_threshold)
    } else {
        prompt_len
    };
    chunk.min(max_batched_tokens).max(1)
}

/// Derive a stable per-request seed from the CLI seed, engine, and request id.
fn request_seed(base_seed: u64, engine_index: u32, request_id: &str) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = std::hash::DefaultHasher::new();
    base_seed.hash(&mut hasher);
    engine_index.hash(&mut hasher);
    request_id.hash(&mut hasher);
    hasher.finish()
}

/// Current UNIX timestamp in seconds for engine-core output envelopes.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

/// Build one request output with only token IDs and terminal status populated.
fn request_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        finish_reason,
        ..Default::default()
    }
}

/// Produce an empty output with a terminal finish reason for an invalid request.
fn empty_finish_outputs(
    engine_index: u32,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
) -> EngineCoreOutputs {
    let output = request_output(request_id, Vec::new(), Some(finish_reason));
    let finished_requests = BTreeSet::from([output.request_id.clone()]);

    EngineCoreOutputs {
        engine_index,
        outputs: vec![output],
        timestamp: now_secs(),
        finished_requests: Some(finished_requests),
        ..Default::default()
    }
}

/// Encode a utility result into the protocol's msgpack value envelope.
fn utility_envelope<T>(value: T) -> Result<UtilityResultEnvelope>
where
    T: Serialize,
{
    Ok(UtilityResultEnvelope::without_type_info(
        rmpv::ext::to_value(value)?,
    ))
}

/// Produce the minimal utility responses needed by a real frontend.
fn utility_response(
    engine_index: u32,
    request: EngineCoreUtilityRequest,
) -> Result<EngineCoreOutputs> {
    let result = match request.method_name.as_str() {
        "get_supported_tasks" => utility_envelope(vec!["generate"]),
        "is_sleeping" => utility_envelope(false),
        "reset_prefix_cache" => utility_envelope(true),
        "reset_mm_cache"
        | "reset_encoder_cache"
        | "profile"
        | "sleep"
        | "wake_up"
        | "execute_dummy_batch" => utility_envelope(()),
        _ => utility_envelope(MsgpackValue::Nil),
    }?;

    Ok(EngineCoreOutputs {
        engine_index,
        utility_output: Some(UtilityOutput {
            call_id: request.call_id,
            failure_message: None,
            result: Some(result),
        }),
        timestamp: now_secs(),
        ..Default::default()
    })
}

/// The `kv_transfer_params` the frontend ferries down from the OpenAI request. The
/// server merges them into `sampling_params.extra_args["kv_transfer_params"]`
/// (mirroring Python vLLM), so that is where the P/D intent (`do_remote_prefill` /
/// `do_remote_decode` / `remote_*`) arrives. In real vLLM the produce/consume logic
/// lives in the NixlConnector inside the engine; here our data plane plays that role.
fn extract_kv_params(request: &EngineCoreRequest) -> Option<JsonValue> {
    request
        .sampling_params
        .as_ref()?
        .extra_args
        .as_ref()?
        .get("kv_transfer_params")
        .cloned()
}

/// Read a boolean flag out of a `kv_transfer_params` object.
fn kv_flag(kv: &JsonValue, key: &str) -> bool {
    kv.get(key).and_then(JsonValue::as_bool).unwrap_or(false)
}

/// Parse the `remote_*` addressing out of a decode request's `kv_transfer_params`
/// (set by the routing sidecar from the prefill response). `None` if this is not a
/// `do_remote_prefill` request or the fields are missing.
fn parse_remote_kv(kv: &JsonValue) -> Option<RemoteKv> {
    if !kv_flag(kv, "do_remote_prefill") {
        return None;
    }
    Some(RemoteKv {
        engine_id: kv.get("remote_engine_id")?.as_str()?.to_string(),
        host: kv.get("remote_host")?.as_str()?.to_string(),
        port: kv.get("remote_port")?.as_u64()? as u32,
        block_ids: kv
            .get("remote_block_ids")?
            .as_array()?
            .iter()
            .filter_map(JsonValue::as_i64)
            .collect(),
        // The prefill's request id; the data plane derives the verify pattern from it and the
        // pool base/addressing comes over the NIXL metadata side channel, so no mock-specific
        // fields ride in kv_transfer_params.
        request_id: kv
            .get("remote_request_id")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

/// Build the `kv_transfer_params` a prefill engine returns, matching vLLM's
/// NixlConnector schema (`scheduler.py:664`). The routing sidecar relays these
/// `remote_*` fields into the decode request.
fn build_prefill_kv_params(
    remote: &RemoteKv,
    request_id: &str,
    remote_num_tokens: usize,
) -> JsonValue {
    serde_json::json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_block_ids": remote.block_ids,
        "remote_engine_id": remote.engine_id,
        "remote_request_id": request_id,
        "remote_host": remote.host,
        "remote_port": remote.port,
        "tp_size": 1,
        "remote_num_tokens": remote_num_tokens,
    })
}

/// Message sent from the IO loop to the engine task to drive the engine loop.
pub(crate) enum EngineInput {
    Request(Box<EngineCoreRequest>),
    Abort(Vec<String>),
    Utility(EngineCoreUtilityRequest),
    StartDpWave,
}

/// Message sent from the engine task to the IO loop for one engine output batch.
pub(crate) struct EngineOutput {
    pub client_index: u32,
    pub outputs: EngineCoreOutputs,
}

/// Per-request decode state owned by one engine.
#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    client_index: u32,
    prompt_len: usize,
    max_tokens: usize,
    generated: usize,
    rng: StdRng,
    /// This request asked us to prefill for a remote decoder (`do_remote_decode`), so on
    /// finish we register its KV and stamp the `remote_*` descriptor onto its output.
    prefill_advertise: bool,
    /// This request's KV was prefilled remotely and pulled in (`do_remote_prefill`), so its
    /// prompt tokens count as externally cached for prefill stats / metrics.
    remote_prefill: bool,
    /// Prompt tokens served from this engine's local prefix cache (the block-pool prefix
    /// hit), feeding `num_local_cached_tokens` and the first-token (TTFT) timing.
    num_local_cached_tokens: usize,
    /// Physical block-pool slot ids this request pins (prompt blocks). Unpinned on finish or
    /// abort so they become evictable; also the `remote_block_ids` the data plane pages.
    block_ids: Vec<usize>,
    /// When the next output token is due. Set to `now + first-token delay` at admission, then
    /// advanced by the inter-token delay after each emitted token. The engine loop sleeps
    /// until the earliest deadline across all active requests, so this is the timing model.
    next_at: Instant,
}

impl ActiveRequest {
    /// Create a new active request, or return an immediate finish reason if invalid.
    ///
    /// `num_running` is the running-request count *including* this one, used to scale the
    /// first-token delay under load.
    fn new(
        engine_index: u32,
        request: Box<EngineCoreRequest>,
        opt: &Opt,
        latency: &LatencyModel,
        num_running: u64,
        num_local_cached_tokens: usize,
        block_ids: Vec<usize>,
    ) -> Result<Self, EngineCoreFinishReason> {
        let incoming_kv = extract_kv_params(&request);
        let prefill_advertise = incoming_kv
            .as_ref()
            .map(|kv| kv_flag(kv, "do_remote_decode"))
            .unwrap_or(false);
        let remote_prefill = incoming_kv
            .as_ref()
            .map(|kv| kv_flag(kv, "do_remote_prefill"))
            .unwrap_or(false);
        let request_id = request.request_id;
        let client_index = request.client_index;
        let prompt_len = request
            .prompt_token_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or_default();

        let Some(sampling_params) = request.sampling_params else {
            warn!(
                request_id,
                "request has no sampling params; returning engine error"
            );
            return Err(EngineCoreFinishReason::Error);
        };
        let max_tokens = sampling_params.max_tokens as usize;

        if let Some(kv) = &incoming_kv {
            info!(request_id, %kv, "received kv_transfer_params from frontend");
        }

        if opt.log_requests {
            info!(
                request_id,
                prompt_len,
                max_tokens,
                chunk_size = opt.output_token_chunk_size,
                "mock request started"
            );
        }

        if max_tokens == 0 {
            return Err(EngineCoreFinishReason::Length);
        }

        let mut rng = StdRng::seed_from_u64(request_seed(opt.seed, engine_index, &request_id));
        // Prompt tokens served from the local prefix cache are not recomputed, so they
        // shorten the prefill (TTFT). The block pool measured the hit at admission.
        let first_delay = latency.first_token_delay(
            &mut rng,
            prompt_len,
            num_local_cached_tokens,
            remote_prefill,
            num_running,
        );

        Ok(ActiveRequest {
            rng,
            request_id,
            client_index,
            prompt_len,
            max_tokens,
            generated: 0,
            prefill_advertise,
            remote_prefill,
            num_local_cached_tokens,
            block_ids,
            next_at: Instant::now() + first_delay,
        })
    }

    /// Advance this request by one engine step.
    fn step(&mut self, opt: &Opt) -> EngineCoreOutput {
        let remaining = self.max_tokens - self.generated;
        let chunk_len = remaining.min(opt.output_token_chunk_size);
        let mut new_token_ids = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            new_token_ids.push(self.rng.random_range(0..opt.vocab_size));
        }
        self.generated += chunk_len;

        let finished = self.generated >= self.max_tokens;
        request_output(
            self.request_id.clone(),
            new_token_ids,
            finished.then_some(EngineCoreFinishReason::Length),
        )
    }

    /// Per-step token demand for the batch token budget: a decoding request needs 1 token, a
    /// prefilling request (no output yet) needs its prompt chunk. Mirrors how vLLM's scheduler
    /// charges `num_new_tokens` against `max_num_batched_tokens` each step.
    fn token_demand(&self, long_prefill_threshold: usize, max_batched_tokens: usize) -> usize {
        if self.generated > 0 {
            1
        } else {
            prefill_token_demand(self.prompt_len, long_prefill_threshold, max_batched_tokens)
        }
    }

    /// Prefill breakdown for this request's first output, feeding the prefix-cache and
    /// KV-transfer metrics. Prompt tokens split three ways: local prefix-cache hits (the
    /// block pool), external KV transfer (a `do_remote_prefill` decode pulls the rest from
    /// the prefill peer), and fresh local compute (whatever neither covered).
    fn prefill_stats(&self) -> PrefillStats {
        let prompt = self.prompt_len as u32;
        let local = (self.num_local_cached_tokens as u32).min(prompt);
        let external = if self.remote_prefill {
            prompt.saturating_sub(local)
        } else {
            0
        };
        let cached = local + external;
        PrefillStats {
            num_prompt_tokens: prompt,
            num_computed_tokens: prompt.saturating_sub(cached),
            num_cached_tokens: cached,
            num_local_cached_tokens: local,
            num_external_cached_tokens: external,
        }
    }
}

/// Internal state for one engine instance, owned by the engine loop task.
struct Engine {
    engine_index: u32,
    opt: Opt,
    latency: LatencyModel,
    data_plane: Box<dyn KvDataPlane>,
    /// Prefix-cache + block-slot pool. Drives local cache hits, `kv_cache_usage`,
    /// `prefix_cache_stats`, and the KV-cache events the cache-aware router consumes.
    pool: BlockPool,
    /// Publisher for KV-cache events; `None` when events are disabled.
    events: Option<KvEventTx>,
    /// RNG for failure injection (Phase 5), seeded per engine for reproducibility.
    failure_rng: StdRng,
    /// The running batch: requests being actively decoded. Capped at `max_num_seqs`.
    active_requests: HashMap<String, ActiveRequest>,
    /// Admitted-but-not-yet-running requests, in arrival order. Drained into `active_requests`
    /// as slots free up; its length is `vllm:num_requests_waiting`.
    waiting: VecDeque<Box<EngineCoreRequest>>,
}

impl Engine {
    /// Decide whether to fail a request on arrival (Phase 5). First the deterministic
    /// context-length check (`prompt + max_tokens > max_model_len`), then the random
    /// injection at `failure_injection_rate`. Returns the finish reason to fail with, if any.
    fn maybe_fail(&mut self, request: &EngineCoreRequest) -> Option<EngineCoreFinishReason> {
        if self.opt.max_model_len > 0 {
            let prompt_len = request.prompt_token_ids.as_ref().map(Vec::len).unwrap_or(0) as u64;
            let max_tokens = request
                .sampling_params
                .as_ref()
                .map(|s| s.max_tokens as u64)
                .unwrap_or(0);
            if prompt_len + max_tokens > self.opt.max_model_len {
                return Some(EngineCoreFinishReason::Length);
            }
        }

        let rate = self.opt.failure_injection_rate;
        if rate > 0.0
            && !self.opt.failure_types.is_empty()
            && self.failure_rng.random::<f64>() < rate
        {
            let idx = self
                .failure_rng
                .random_range(0..self.opt.failure_types.len());
            return Some(self.opt.failure_types[idx].finish_reason());
        }
        None
    }

    /// Drain one frontend request message.
    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        let mut outputs = Vec::new();

        match input {
            EngineInput::Request(request) => {
                let request_id = request.request_id.clone();
                let client_index = request.client_index;

                // Dedup against both the running batch and the waiting queue.
                if self.active_requests.contains_key(&request_id)
                    || self.waiting.iter().any(|r| r.request_id == request_id)
                {
                    warn!(
                        engine_index = self.engine_index,
                        request_id, "duplicate request id"
                    );
                    return Ok(vec![EngineOutput {
                        client_index,
                        outputs: empty_finish_outputs(
                            self.engine_index,
                            request_id,
                            EngineCoreFinishReason::Error,
                        ),
                    }]);
                }

                // Phase 5: fail the request on arrival if the context-length check trips or
                // the injector rolls a failure. It never enters the queue or the pool.
                if let Some(reason) = self.maybe_fail(&request) {
                    if self.opt.log_requests {
                        info!(request_id, ?reason, "request failed on arrival (injected)");
                    }
                    return Ok(vec![EngineOutput {
                        client_index,
                        outputs: empty_finish_outputs(self.engine_index, request_id, reason),
                    }]);
                }

                // vLLM never rejects on queue length, so the queue is unbounded. Enqueue, then
                // admit into the batch if the seq cap and token budget allow; else it waits.
                self.waiting.push_back(request);
                outputs.extend(self.schedule());
                if self.opt.log_requests && self.waiting.iter().any(|r| r.request_id == request_id)
                {
                    info!(
                        request_id,
                        waiting = self.waiting.len(),
                        "request queued (batch/budget full)"
                    );
                }
            }

            EngineInput::Abort(request_ids) => {
                let mut by_client =
                    BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
                for request_id in request_ids {
                    self.data_plane.release(&request_id);
                    // A request is either running or waiting (never both); abort whichever.
                    let client_index =
                        if let Some(request) = self.active_requests.remove(&request_id) {
                            // Free its prefix-cache pins so the blocks can be evicted later.
                            self.pool.unpin(&request.block_ids);
                            Some(request.client_index)
                        } else if let Some(pos) =
                            self.waiting.iter().position(|r| r.request_id == request_id)
                        {
                            self.waiting.remove(pos).map(|r| r.client_index)
                        } else {
                            None
                        };

                    if let Some(client_index) = client_index {
                        let output = request_output(
                            request_id.clone(),
                            Vec::new(),
                            Some(EngineCoreFinishReason::Abort),
                        );
                        let (outs, finished) = by_client
                            .entry(client_index)
                            .or_insert_with(|| (Vec::new(), BTreeSet::new()));
                        outs.push(output);
                        finished.insert(request_id.clone());
                        if self.opt.log_requests {
                            info!(request_id, finish_reason = "abort", "request aborted");
                        }
                    }
                }
                for (client_index, (client_outputs, finished_requests)) in by_client {
                    outputs.push(EngineOutput {
                        client_index,
                        outputs: EngineCoreOutputs {
                            engine_index: self.engine_index,
                            outputs: client_outputs,
                            timestamp: now_secs(),
                            finished_requests: Some(finished_requests),
                            ..Default::default()
                        },
                    });
                }
                // Aborting running requests frees batch slots; admit any waiting requests.
                outputs.extend(self.schedule());
            }

            EngineInput::Utility(request) => {
                debug!(
                    engine_index = self.engine_index,
                    call_id = %request.call_id,
                    method = request.method_name,
                    "utility request"
                );
                // Mirror vLLM: resetting the prefix cache clears the block pool and emits a
                // single AllBlocksCleared so the router drops this engine's whole index.
                if request.method_name == "reset_prefix_cache"
                    && let Some(event) = self.pool.reset()
                    && let Some(events) = &self.events
                {
                    events.publish(vec![event]);
                }
                let client_index = request.client_index;
                outputs.push(EngineOutput {
                    client_index,
                    outputs: utility_response(self.engine_index, request)?,
                });
            }

            EngineInput::StartDpWave => {
                debug!(engine_index = self.engine_index, "ignoring START_DP_WAVE");
            }
        }

        Ok(outputs)
    }

    /// Maximum requests allowed in the running batch. Clamped to at least 1 so a misconfigured
    /// `max_num_seqs = 0` can never wedge every request in the queue forever.
    fn running_capacity(&self) -> usize {
        self.opt.max_num_seqs.max(1) as usize
    }

    /// Move a request from the waiting queue into the running batch: run the decode-side KV
    /// pull (its TTFT models the transfer, so the pull belongs at admission, not arrival),
    /// then build its `ActiveRequest` (which starts the first-token clock). Returns an
    /// immediate-finish output if the request is invalid (e.g. `max_tokens == 0`).
    fn admit(&mut self, request: Box<EngineCoreRequest>) -> Option<EngineOutput> {
        let request_id = request.request_id.clone();
        let client_index = request.client_index;
        let remote = extract_kv_params(&request)
            .as_ref()
            .and_then(parse_remote_kv);
        let prompt_tokens: Vec<u32> = request.prompt_token_ids.clone().unwrap_or_default();

        // === BLOCK POOL: cache this prompt locally ===
        // Measure the local prefix hit, allocate slots for the new blocks, pin them all, and
        // emit BlockStored/BlockRemoved for the router. The slot ids are what the data plane
        // pages over NIXL and what we advertise as remote_block_ids.
        let outcome = self.pool.cache_prompt(&prompt_tokens);
        if let Some(events) = &self.events {
            events.publish(outcome.events);
        }
        let num_local_cached = outcome.num_cached_tokens;
        let block_ids = outcome.block_ids;

        // === DATA PLANE: decode-side pull ===
        // A do_remote_prefill request carries the prefill engine's remote_* descriptor; pull
        // its KV over NIXL into our slots before we start generating.
        if let Some(remote) = remote {
            let kv = RequestKv {
                request_id: &request_id,
                num_tokens: prompt_tokens.len(),
                block_ids: &block_ids,
            };
            match self.data_plane.pull_prefilled(kv, &remote) {
                Ok(bytes) => info!(
                    request_id,
                    bytes,
                    engine_id = remote.engine_id,
                    "pulled remote KV before decode"
                ),
                Err(error) => warn!(request_id, %error, "remote KV pull failed"),
            }
        }

        // Count this request among the running set for its own load-factor scaling.
        let num_running = self.active_requests.len() as u64 + 1;
        match ActiveRequest::new(
            self.engine_index,
            request,
            &self.opt,
            &self.latency,
            num_running,
            num_local_cached,
            block_ids.clone(),
        ) {
            Ok(active) => {
                self.active_requests.insert(request_id, active);
                None
            }
            Err(finish_reason) => {
                // The request never runs, so release the pins we just took.
                self.pool.unpin(&block_ids);
                Some(EngineOutput {
                    client_index,
                    outputs: empty_finish_outputs(self.engine_index, request_id, finish_reason),
                })
            }
        }
    }

    /// Current per-step token demand of the running batch (sum of each request's `token_demand`),
    /// charged against `max_num_batched_tokens`.
    fn scheduled_token_demand(&self) -> usize {
        let threshold = self.opt.long_prefill_token_threshold as usize;
        let budget = self.opt.max_num_batched_tokens as usize;
        self.active_requests
            .values()
            .map(|request| request.token_demand(threshold, budget))
            .sum()
    }

    /// Pop the next request to admit, honoring the scheduling policy: FIFO for `fcfs`, or the
    /// smallest `(priority, arrival_time)` for `priority` (matching vLLM's priority queue).
    fn pop_next_waiting(&mut self) -> Option<Box<EngineCoreRequest>> {
        match self.opt.scheduling_policy {
            SchedulingPolicy::Fcfs => self.waiting.pop_front(),
            SchedulingPolicy::Priority => {
                let idx = self
                    .waiting
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.priority.cmp(&b.priority).then_with(|| {
                            a.arrival_time
                                .partial_cmp(&b.arrival_time)
                                .unwrap_or(Ordering::Equal)
                        })
                    })
                    .map(|(i, _)| i)?;
                self.waiting.remove(idx)
            }
        }
    }

    /// Admit waiting requests into the running batch (in policy order) until the seq cap
    /// (`max_num_seqs`) or the per-step token budget (`max_num_batched_tokens`) is reached, or
    /// the queue empties. This mirrors vLLM's scheduler filling the batch under both bounds.
    /// Returns any immediate-finish outputs produced during admission.
    fn schedule(&mut self) -> Vec<EngineOutput> {
        let mut outputs = Vec::new();
        let budget = self.opt.max_num_batched_tokens as usize;
        let threshold = self.opt.long_prefill_token_threshold as usize;
        let mut demand = self.scheduled_token_demand();

        while self.active_requests.len() < self.running_capacity() && demand < budget {
            let Some(request) = self.pop_next_waiting() else {
                break;
            };
            // The token demand this request adds once admitted (it starts by prefilling).
            let prompt_len = request
                .prompt_token_ids
                .as_ref()
                .map(Vec::len)
                .unwrap_or_default();
            match self.admit(request) {
                // Admitted and now prefilling: it occupies budget until its first token.
                None => demand += prefill_token_demand(prompt_len, threshold, budget),
                // Invalid request finished immediately; it never entered the batch.
                Some(output) => outputs.push(output),
            }
        }
        outputs
    }

    /// Advance every request whose token is due, returning one batched engine output per
    /// client. Each emitted token reschedules the request by the inter-token delay; the
    /// engine loop sleeps until the earliest deadline, so this is paced by the latency model.
    fn step(&mut self) -> Vec<EngineOutput> {
        if self.active_requests.is_empty() {
            return Vec::new();
        }

        let now = Instant::now();
        // Running-request count drives the inter-token load factor; snapshot before the loop.
        let num_running = self.active_requests.len() as u64;

        let mut by_client = BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
        let mut finished_ids = BTreeSet::new();

        // Collect (client_index, request_id, num_tokens, block_ids) for finished prefill
        // requests so we can advertise their KV after the borrow on active_requests ends.
        let mut to_advertise: Vec<(u32, String, usize, Vec<usize>)> = Vec::new();

        for request in self.active_requests.values_mut() {
            if request.next_at > now {
                // Not due yet; its deadline keeps the engine loop's sleep honest.
                continue;
            }
            let client_index = request.client_index;
            let is_first = request.generated == 0;
            let mut output = request.step(&self.opt);
            let request_id = request.request_id.clone();
            let finished = output.finished();

            // The first output of a request carries its prefill breakdown for metrics.
            if is_first {
                output.prefill_stats = Some(request.prefill_stats());
            }

            if finished {
                finished_ids.insert(request_id.clone());
                if request.prefill_advertise {
                    to_advertise.push((
                        client_index,
                        request_id.clone(),
                        request.prompt_len,
                        request.block_ids.clone(),
                    ));
                }
                if self.opt.log_requests {
                    info!(
                        request_id,
                        prompt_len = request.prompt_len,
                        output_tokens = request.generated,
                        finish_reason = "length",
                        "request finished"
                    );
                }
            } else {
                // Schedule the next decode token.
                request.next_at = now
                    + self
                        .latency
                        .inter_token_delay(&mut request.rng, num_running);
            }

            let (outs, finished_set) = by_client
                .entry(client_index)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            if finished {
                finished_set.insert(request_id);
            }
            // Hold the output so we can attach kv_transfer_params below if advertised.
            outs.push(std::mem::take(&mut output));
        }

        for request_id in &finished_ids {
            if let Some(request) = self.active_requests.remove(request_id) {
                // Release the prefix-cache pins; the blocks stay cached (evictable) for hits.
                // (A prefill-advertised block could in principle be evicted before its decode
                // peer pulls it under heavy cache pressure; size the pool to avoid that.)
                self.pool.unpin(&request.block_ids);
            }
        }

        // === DATA PLANE: prefill-side advertise ===
        // Once prefill finishes, register the fake KV and stamp the real kv_transfer_params
        // (remote_engine_id/host/port/block_ids) onto the finishing output for the decoder.
        for (client_index, request_id, num_tokens, block_ids) in to_advertise {
            let kv = RequestKv {
                request_id: &request_id,
                num_tokens,
                block_ids: &block_ids,
            };
            match self.data_plane.advertise_prefilled(kv) {
                Ok(remote) => {
                    if let Some((outs, _)) = by_client.get_mut(&client_index)
                        && let Some(out) = outs.iter_mut().find(|o| o.request_id == request_id)
                    {
                        out.kv_transfer_params =
                            Some(build_prefill_kv_params(&remote, &request_id, num_tokens));
                    }
                }
                Err(error) => warn!(request_id, %error, "prefill KV advertise failed"),
            }
        }

        // Refill freed batch slots from the waiting queue before snapshotting stats, so the
        // gauges reflect the post-step batch and queue depth.
        let admit_outputs = self.schedule();

        // Computed after removals and admission so the gauges reflect post-step state (e.g.
        // the batch that finishes the last request reports num_running = 0). Cloned per client.
        let stats = self.scheduler_stats();

        let mut result: Vec<EngineOutput> = by_client
            .into_iter()
            .filter_map(|(client_index, (outputs, finished_requests))| {
                (!outputs.is_empty()).then(|| EngineOutput {
                    client_index,
                    outputs: EngineCoreOutputs {
                        engine_index: self.engine_index,
                        outputs,
                        scheduler_stats: Some(Box::new(stats.clone())),
                        timestamp: now_secs(),
                        finished_requests: (!finished_requests.is_empty())
                            .then_some(finished_requests),
                        ..Default::default()
                    },
                })
            })
            .collect();
        // Immediate-finish outputs from admission (rare: an invalid request was queued).
        result.extend(admit_outputs);
        result
    }

    /// Snapshot of scheduler state for the frontend's `vllm:*` gauges: the running batch size,
    /// the waiting-queue depth, KV-cache utilization, and the prefix-cache hit counters.
    ///
    /// Takes `&mut self` because the prefix-cache counters are per-report deltas (the frontend
    /// does `inc_by` on them), so each snapshot drains them.
    fn scheduler_stats(&mut self) -> SchedulerStats {
        let prefix = self.pool.take_stats();
        SchedulerStats {
            num_running_reqs: self.active_requests.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            kv_cache_usage: self.pool.usage(),
            prefix_cache_stats: PrefixCacheStats {
                base: BaseCacheStats {
                    reset: false,
                    requests: prefix.requests,
                    queries: prefix.queries,
                    hits: prefix.hits,
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// The soonest a token is due across all active requests; `None` when idle. The engine
    /// loop sleeps until this instant before calling `step`.
    fn earliest_deadline(&self) -> Option<Instant> {
        self.active_requests
            .values()
            .map(|request| request.next_at)
            .min()
    }
}

/// Run the main loop for one engine, receiving `EngineInput` and sending `EngineOutput`
/// until `shutdown` is cancelled.
pub(crate) async fn run_engine_loop(
    engine_index: u32,
    opt: Opt,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::Sender<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let role = opt.pd_role;
    let cfg = NixlConfig {
        kv_block_bytes: opt.kv_block_bytes,
        tokens_per_block: opt.tokens_per_block,
        kv_cache_blocks: opt.kv_cache_size as usize,
        engine_id: opt.engine_id.clone(),
        side_channel_host: opt.side_channel_host.clone(),
        side_channel_port: opt.side_channel_port,
    };
    let latency = opt.latency_model();
    let opt_seed = opt.seed;
    let pool = BlockPool::new(
        opt.tokens_per_block,
        opt.kv_cache_size as usize,
        opt.kv_cache_none_seed,
    );
    let events = kvevents::spawn(opt.kv_events_config(engine_index), shutdown.clone())
        .await
        .unwrap_or_else(|error| {
            warn!(%error, "kv-event publisher failed to start; continuing without events");
            None
        });
    let mut engine = Engine {
        engine_index,
        opt,
        latency,
        data_plane: make_data_plane(role, cfg),
        pool,
        events,
        failure_rng: StdRng::seed_from_u64(
            opt_seed ^ (engine_index as u64).wrapping_mul(0x9e3779b9),
        ),
        active_requests: HashMap::new(),
        waiting: VecDeque::new(),
    };

    loop {
        // Wake at the soonest token deadline; disabled (no timer branch) when idle.
        let next_deadline = engine.earliest_deadline();
        let outputs = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,

            input = input_rx.recv() => {
                let input = input.ok_or_else(|| anyhow!("engine input channel closed"))?;
                engine.handle_input(input)?
            }

            _ = async { sleep_until(next_deadline.unwrap_or_else(Instant::now)).await },
                if next_deadline.is_some() =>
            {
                engine.step()
            }
        };

        for output in outputs {
            output_tx
                .send(output)
                .await
                .map_err(|_| anyhow!("engine IO task shut down"))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use clap::Parser as _;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};

    use super::*;
    use crate::dataplane::{NixlConfig, PdRole, make_data_plane};

    fn test_opt() -> Opt {
        // clap fills every field with its declared default (all latency knobs = 0 / instant).
        Opt::parse_from(["inference-sim"])
    }

    fn test_engine(opt: Opt) -> Engine {
        let cfg = NixlConfig {
            kv_block_bytes: opt.kv_block_bytes,
            tokens_per_block: opt.tokens_per_block,
            kv_cache_blocks: opt.kv_cache_size as usize,
            engine_id: opt.engine_id.clone(),
            side_channel_host: opt.side_channel_host.clone(),
            side_channel_port: opt.side_channel_port,
        };
        let pool = crate::blockpool::BlockPool::new(
            opt.tokens_per_block,
            opt.kv_cache_size as usize,
            opt.kv_cache_none_seed,
        );
        Engine {
            engine_index: 0,
            latency: opt.latency_model(),
            data_plane: make_data_plane(PdRole::Both, cfg),
            pool,
            events: None,
            failure_rng: StdRng::seed_from_u64(opt.seed),
            active_requests: HashMap::new(),
            waiting: VecDeque::new(),
            opt,
        }
    }

    fn request(id: &str, prompt_len: usize, max_tokens: u32) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some(vec![0u32; prompt_len]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        }
    }

    fn add(engine: &mut Engine, req: EngineCoreRequest) {
        engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input");
    }

    /// Drain steps until the engine is idle, returning the flat output list. Safe only when
    /// the latency model is instant (deadlines never in the future), as in these tests.
    fn drain(engine: &mut Engine) -> Vec<EngineCoreOutput> {
        let mut all = Vec::new();
        while !engine.active_requests.is_empty() {
            let batch = engine.step();
            assert!(
                !batch.is_empty(),
                "instant model must make progress each step"
            );
            for output in batch {
                all.extend(output.outputs.outputs);
            }
        }
        all
    }

    #[test]
    fn unconfigured_engine_is_instant() {
        let mut engine = test_engine(test_opt());
        add(&mut engine, request("r1", 4, 3));
        // Instant model: the first token is due immediately.
        assert!(engine.earliest_deadline().unwrap() <= Instant::now());
        let outputs = drain(&mut engine);
        let tokens: usize = outputs.iter().map(|o| o.new_token_ids.len()).sum();
        assert_eq!(tokens, 3);
        assert!(outputs.last().unwrap().finished());
    }

    #[test]
    fn ttft_delays_the_first_token() {
        let mut opt = test_opt();
        opt.time_to_first_token = 10_000; // 10s, no std-dev -> exact, comfortably in the future
        let mut engine = test_engine(opt);

        let before = Instant::now();
        add(&mut engine, request("r1", 4, 2));
        let deadline = engine.earliest_deadline().expect("a deadline");
        assert!(deadline >= before + Duration::from_millis(9_000));

        // The token is not due yet, so a step right now produces nothing and keeps the request.
        assert!(engine.step().is_empty());
        assert_eq!(engine.active_requests.len(), 1);
    }

    #[test]
    fn prefill_stats_only_on_first_output() {
        let mut engine = test_engine(test_opt());
        add(&mut engine, request("r1", 7, 4));
        let outputs = drain(&mut engine);

        let with_prefill: Vec<_> = outputs
            .iter()
            .filter(|o| o.prefill_stats.is_some())
            .collect();
        assert_eq!(
            with_prefill.len(),
            1,
            "exactly one output carries prefill stats"
        );
        let stats = with_prefill[0].prefill_stats.as_ref().unwrap();
        assert_eq!(stats.num_prompt_tokens, 7);
        assert_eq!(stats.num_computed_tokens, 7);
        assert_eq!(stats.num_external_cached_tokens, 0);
    }

    #[test]
    fn scheduler_stats_report_running_and_kv_usage() {
        let mut opt = test_opt();
        opt.tokens_per_block = 16;
        opt.kv_cache_size = 100; // blocks
        let mut engine = test_engine(opt);

        // One request, 32 prompt tokens -> ceil(32/16) = 2 blocks while running.
        add(&mut engine, request("r1", 32, 5));
        let stats = engine.scheduler_stats();
        assert_eq!(stats.num_running_reqs, 1);
        assert_eq!(stats.num_waiting_reqs, 0);
        assert!(
            (stats.kv_cache_usage - 0.02).abs() < 1e-9,
            "got {}",
            stats.kv_cache_usage
        );

        // Drain to completion; the engine is empty so usage is back to zero.
        let _ = drain(&mut engine);
        let idle = engine.scheduler_stats();
        assert_eq!(idle.num_running_reqs, 0);
        assert_eq!(idle.kv_cache_usage, 0.0);
    }

    #[test]
    fn final_batch_reports_zero_running() {
        let mut engine = test_engine(test_opt());
        add(&mut engine, request("r1", 4, 1)); // single output token, finishes in one step
        let batch = engine.step();
        let stats = batch[0]
            .outputs
            .scheduler_stats
            .as_ref()
            .expect("stats on batch");
        // Computed after the finished request is removed, so the gauge drops to 0.
        assert_eq!(stats.num_running_reqs, 0);
        assert!(engine.active_requests.is_empty());
    }

    #[test]
    fn remote_prefill_request_counts_prompt_as_external_cached() {
        let mut engine = test_engine(test_opt());
        let mut req = request("r1", 9, 2);
        let mut extra = HashMap::new();
        extra.insert(
            "kv_transfer_params".to_string(),
            serde_json::json!({ "do_remote_prefill": true }),
        );
        req.sampling_params.as_mut().unwrap().extra_args = Some(extra);
        add(&mut engine, req);

        let outputs = drain(&mut engine);
        let stats = outputs
            .iter()
            .find_map(|o| o.prefill_stats.as_ref())
            .expect("prefill stats");
        assert_eq!(stats.num_external_cached_tokens, 9);
        assert_eq!(stats.num_computed_tokens, 0);
    }

    #[test]
    fn shared_prompt_prefix_counts_as_local_cached() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4; // small blocks so an 8-token prompt is 2 full blocks
        let mut engine = test_engine(opt);

        // First request is cold: both blocks are computed, none cached.
        add(&mut engine, request("r1", 8, 1));
        let first = drain(&mut engine);
        let s1 = first
            .iter()
            .find_map(|o| o.prefill_stats.as_ref())
            .expect("prefill stats");
        assert_eq!(s1.num_local_cached_tokens, 0);
        assert_eq!(s1.num_computed_tokens, 8);

        // r1's blocks stay cached after it finishes. The identical prompt now fully hits.
        add(&mut engine, request("r2", 8, 1));
        let second = drain(&mut engine);
        let s2 = second
            .iter()
            .find_map(|o| o.prefill_stats.as_ref())
            .expect("prefill stats");
        assert_eq!(
            s2.num_local_cached_tokens, 8,
            "both blocks served from cache"
        );
        assert_eq!(s2.num_computed_tokens, 0);
        assert_eq!(s2.num_cached_tokens, 8);
    }

    #[test]
    fn scheduler_stats_carry_prefix_cache_counters() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4;
        let mut engine = test_engine(opt);

        // Cold request: its single step reports 2 queries and 0 hits (delta since last drain).
        add(&mut engine, request("r1", 8, 1));
        let batch = engine.step();
        let stats = batch[0]
            .outputs
            .scheduler_stats
            .as_ref()
            .expect("stats on batch");
        assert_eq!(stats.prefix_cache_stats.base.requests, 1);
        assert_eq!(stats.prefix_cache_stats.base.queries, 2);
        assert_eq!(stats.prefix_cache_stats.base.hits, 0);
    }

    #[test]
    fn max_model_len_fails_oversized_request_with_length() {
        let mut opt = test_opt();
        opt.max_model_len = 10;
        let mut engine = test_engine(opt);

        // prompt 8 + max_tokens 5 = 13 > 10: context-length error, never queued.
        let out = submit(&mut engine, request("big", 8, 5));
        assert_eq!(finish_reason(&out[0]), Some(EngineCoreFinishReason::Length));
        assert!(engine.active_requests.is_empty());
        assert!(engine.waiting.is_empty());
    }

    #[test]
    fn max_model_len_admits_a_fitting_request() {
        let mut opt = test_opt();
        opt.max_model_len = 100;
        let mut engine = test_engine(opt);
        submit(&mut engine, request("ok", 8, 5)); // 13 <= 100
        assert_eq!(engine.active_requests.len(), 1);
    }

    #[test]
    fn failure_injection_at_rate_one_fails_every_request() {
        let mut opt = test_opt();
        opt.failure_injection_rate = 1.0;
        opt.failure_types = vec![crate::FailureType::Error];
        let mut engine = test_engine(opt);

        let out = submit(&mut engine, request("doomed", 4, 5));
        assert_eq!(finish_reason(&out[0]), Some(EngineCoreFinishReason::Error));
        assert!(engine.active_requests.is_empty());
    }

    #[test]
    fn failure_injection_at_rate_zero_never_fails() {
        let opt = test_opt(); // rate defaults to 0.0
        let mut engine = test_engine(opt);
        let out = submit(&mut engine, request("safe", 4, 5));
        assert!(
            out.iter()
                .all(|o| finish_reason(o) != Some(EngineCoreFinishReason::Error))
        );
        assert_eq!(engine.active_requests.len(), 1);
    }

    fn submit(engine: &mut Engine, req: EngineCoreRequest) -> Vec<EngineOutput> {
        engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input")
    }

    fn abort(engine: &mut Engine, ids: &[&str]) -> Vec<EngineOutput> {
        let ids = ids.iter().map(|s| s.to_string()).collect();
        engine
            .handle_input(EngineInput::Abort(ids))
            .expect("handle_input")
    }

    /// The finish reason on a single-output engine batch (used to inspect rejections/aborts).
    fn finish_reason(out: &EngineOutput) -> Option<EngineCoreFinishReason> {
        out.outputs.outputs.first().and_then(|o| o.finish_reason)
    }

    #[test]
    fn batch_capped_at_max_num_seqs_rest_wait() {
        let mut opt = test_opt();
        opt.max_num_seqs = 2;
        let mut engine = test_engine(opt);

        // Five long requests, batch holds 2; the other three wait.
        for i in 0..5 {
            submit(&mut engine, request(&format!("r{i}"), 4, 50));
        }
        assert_eq!(engine.active_requests.len(), 2);
        assert_eq!(engine.waiting.len(), 3);

        let stats = engine.scheduler_stats();
        assert_eq!(stats.num_running_reqs, 2);
        assert_eq!(stats.num_waiting_reqs, 3);
    }

    #[test]
    fn queue_drains_fifo_as_running_finish() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        let mut engine = test_engine(opt);

        // Three single-token requests through a batch of 1: each step finishes one and admits
        // the next, in arrival order.
        for i in 0..3 {
            submit(&mut engine, request(&format!("r{i}"), 4, 1));
        }
        assert_eq!(engine.active_requests.len(), 1);
        assert_eq!(engine.waiting.len(), 2);

        let finished_order: Vec<String> = drain(&mut engine)
            .into_iter()
            .filter(|o| o.finished())
            .map(|o| o.request_id)
            .collect();
        assert_eq!(finished_order, vec!["r0", "r1", "r2"]);
        assert!(engine.waiting.is_empty());
    }

    #[test]
    fn aborting_a_waiting_request_removes_it_from_the_queue() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        let mut engine = test_engine(opt);

        submit(&mut engine, request("running", 4, 50));
        submit(&mut engine, request("queued", 4, 50));
        assert_eq!(engine.waiting.len(), 1);

        let out = abort(&mut engine, &["queued"]);
        assert_eq!(finish_reason(&out[0]), Some(EngineCoreFinishReason::Abort));
        assert!(engine.waiting.is_empty());
        assert!(engine.active_requests.contains_key("running"));
    }

    #[test]
    fn aborting_a_running_request_admits_a_waiting_one() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        let mut engine = test_engine(opt);

        submit(&mut engine, request("running", 4, 50));
        submit(&mut engine, request("queued", 4, 50));
        assert!(engine.active_requests.contains_key("running"));
        assert_eq!(engine.waiting.len(), 1);

        // Freeing the only batch slot pulls the queued request into the batch.
        abort(&mut engine, &["running"]);
        assert!(engine.active_requests.contains_key("queued"));
        assert!(engine.waiting.is_empty());
    }

    #[test]
    fn unbounded_queue_never_rejects() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1; // vLLM never rejects on queue length; neither do we
        let mut engine = test_engine(opt);

        for i in 0..10 {
            let out = submit(&mut engine, request(&format!("r{i}"), 4, 50));
            assert!(
                out.iter()
                    .all(|o| finish_reason(o) != Some(EngineCoreFinishReason::Error)),
                "no request should ever be rejected on queue length"
            );
        }
        assert_eq!(engine.active_requests.len(), 1);
        assert_eq!(engine.waiting.len(), 9);
    }

    #[test]
    fn token_budget_caps_batch_even_with_free_seq_slots() {
        let mut opt = test_opt();
        opt.max_num_seqs = 100; // plenty of seq slots; the token budget is the binding limit
        opt.max_num_batched_tokens = 20;
        let mut engine = test_engine(opt);

        // Each prefilling request demands its 10 prompt tokens. Budget 20 admits two (demand
        // 0 -> 10 -> 20), then 20 < 20 is false so the rest wait, despite 98 free seq slots.
        for i in 0..5 {
            submit(&mut engine, request(&format!("r{i}"), 10, 50));
        }
        assert_eq!(engine.active_requests.len(), 2);
        assert_eq!(engine.waiting.len(), 3);
        assert_eq!(engine.scheduler_stats().num_waiting_reqs, 3);
    }

    #[test]
    fn budget_frees_as_prefill_becomes_decode() {
        let mut opt = test_opt();
        opt.max_num_seqs = 100;
        opt.max_num_batched_tokens = 20;
        let mut engine = test_engine(opt);

        for i in 0..5 {
            submit(&mut engine, request(&format!("r{i}"), 10, 50));
        }
        assert_eq!(engine.active_requests.len(), 2);

        // One step: the two prefilling requests emit their first token (instant model) and
        // become decoders (demand 1 each), freeing budget to admit more from the queue.
        engine.step();
        assert!(engine.active_requests.len() > 2);
        assert!(engine.waiting.len() < 3);
    }

    #[test]
    fn priority_policy_admits_lowest_priority_value_first() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        opt.scheduling_policy = SchedulingPolicy::Priority;
        let mut engine = test_engine(opt);

        let with_priority = |id: &str, p: i32| {
            let mut req = request(id, 4, 50);
            req.priority = p;
            req
        };

        submit(&mut engine, with_priority("blocker", 0)); // admitted immediately
        submit(&mut engine, with_priority("p10", 10));
        submit(&mut engine, with_priority("p1", 1));
        submit(&mut engine, with_priority("p5", 5));
        assert_eq!(engine.waiting.len(), 3);

        // Each freed slot admits the smallest remaining priority value, not arrival order.
        abort(&mut engine, &["blocker"]);
        assert!(engine.active_requests.contains_key("p1"));
        abort(&mut engine, &["p1"]);
        assert!(engine.active_requests.contains_key("p5"));
        abort(&mut engine, &["p5"]);
        assert!(engine.active_requests.contains_key("p10"));
    }
}
