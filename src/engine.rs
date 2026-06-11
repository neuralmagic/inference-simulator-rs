//! The per-engine generation loop. Adapted from vLLM's in-tree `vllm-mock-engine`
//! (`rust/src/mock-engine/src/engine.rs`), with the prefill/decode data-plane hooks
//! added at the two points where real KV bytes would move.
//!
//! Everything wire-facing comes from the `vllm-engine-core-client` crate, so this
//! stays correct as the protocol evolves upstream.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value as MsgpackValue;
use serde::Serialize;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{debug, info, warn};
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::stats::{
    BaseCacheStats, PrefillStats, PrefixCacheStats, SchedulerStats,
};
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityCallId, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
};

use crate::blockpool::BlockPool;
use crate::dataplane::{KvDataPlane, NixlConfig, RemoteKv, RequestKv, make_data_plane};
use crate::kvevents::KvEventTx;
use crate::latency::{DecodePacing, FirstTokenCtx, LatencyModel};
use crate::lora::LoraRegistry;
use crate::sched::{self, Scheduler};
use crate::tokens::{RandomTokens, TokenCtx, TokenSource};
use crate::{Opt, SchedulingPolicy};

/// Per-step token demand of a single prefilling request: a (possibly chunked) slice of its
/// prompt, capped by the chunked-prefill threshold and the overall token budget, then halved
/// to model vLLM's chunk packing: the tail chunk of one prefill shares a step's budget with
/// the head chunk of the next, so roughly two partial prefills are in flight at the rate the
/// budget sustains. Without the halving, one budget-sized prefill serializes all admissions
/// behind its whole park and the simulated queue grows ~10x past the real engine's (H200
/// counterfactual validation). At least 1 so a request always makes progress.
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
    chunk.min(max_batched_tokens).div_ceil(2).max(1)
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

/// Wrap a utility result envelope in the `EngineCoreOutputs` the frontend awaits for `call_id`.
fn utility_result_outputs(
    engine_index: u32,
    call_id: UtilityCallId,
    result: UtilityResultEnvelope,
) -> EngineCoreOutputs {
    EngineCoreOutputs {
        engine_index,
        utility_output: Some(UtilityOutput {
            call_id,
            failure_message: None,
            result: Some(result),
        }),
        timestamp: now_secs(),
        ..Default::default()
    }
}

/// Produce the minimal utility responses needed by a real frontend. Stateful utilities
/// (`add_lora`/`remove_lora`, `reset_prefix_cache`) are handled on `Engine` instead.
fn utility_response(
    engine_index: u32,
    request: EngineCoreUtilityRequest,
) -> Result<EngineCoreOutputs> {
    let result = match request.method_name.as_str() {
        "get_supported_tasks" => utility_envelope(vec!["generate"]),
        "is_sleeping" => utility_envelope(false),
        "reset_mm_cache"
        | "reset_encoder_cache"
        | "profile"
        | "sleep"
        | "wake_up"
        | "execute_dummy_batch" => utility_envelope(()),
        other => {
            warn!(
                engine_index,
                method = other,
                "unknown utility method; returning Nil"
            );
            utility_envelope(MsgpackValue::Nil)
        }
    }?;

    Ok(utility_result_outputs(
        engine_index,
        request.call_id,
        result,
    ))
}

/// The `kv_transfer_params` the frontend ferries down from the OpenAI request. The
/// server merges them into `sampling_params.extra_args["kv_transfer_params"]`
/// (mirroring Python vLLM), so that is where the P/D intent (`do_remote_prefill` /
/// `do_remote_decode` / `remote_*`) arrives. In real vLLM the produce/consume logic
/// lives in the NixlConnector inside the engine; here our data plane plays that role.
pub(crate) fn extract_kv_params(request: &EngineCoreRequest) -> Option<JsonValue> {
    request
        .sampling_params
        .as_ref()?
        .extra_args
        .as_ref()?
        .get("kv_transfer_params")
        .cloned()
}

/// Read a boolean flag out of a `kv_transfer_params` object.
pub(crate) fn kv_flag(kv: &JsonValue, key: &str) -> bool {
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

use crate::engine_core::{EngineCore, EngineInput, EngineOutput};

/// Per-request decode state owned by one engine.
#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    client_index: u32,
    prompt_len: usize,
    /// The original prompt token ids, retained so the `TokenSource` can condition on them.
    prompt_token_ids: Vec<u32>,
    max_tokens: usize,
    generated: usize,
    rng: StdRng,
    /// Per-request decode pacing state (hierarchical latency models pin a request
    /// to one source request's gap distribution; stateless models ignore it).
    pacing: DecodePacing,
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
    /// The LoRA adapter this request runs against, if any (from `EngineCoreRequest.lora_request`).
    /// Drives the per-adapter running count and the LoRA slot cap. `None` is the base model.
    lora_name: Option<String>,
    /// Sampled prefill service time (the first-token delay draw). Kept so admission can
    /// block every running decode by the same amount: prefill chunks and decode steps run
    /// in one serial engine step stream, so a prefill delays everyone, not just itself.
    prefill_service: Duration,
    /// Per-step interference owed to prefill chunks admitted while this request decodes.
    /// Each admitted chunk pushes one surcharge; each emitted token pops one and stretches
    /// that gap by it, smearing the prefill's service across the chunk window the way real
    /// chunked prefill elongates the steps it rides in (instead of one giant gap).
    step_surcharges: VecDeque<Duration>,
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
        latency: &dyn LatencyModel,
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
        let lora_name = request.lora_request.as_ref().map(|l| l.lora_name.clone());
        let request_id = request.request_id;
        let client_index = request.client_index;
        let prompt_token_ids = request.prompt_token_ids.unwrap_or_default();
        let prompt_len = prompt_token_ids.len();

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
        let mut pacing = DecodePacing::for_prompt(prompt_len);
        // Prompt tokens served from the local prefix cache are not recomputed, so they
        // shorten the prefill (TTFT). The block pool measured the hit at admission.
        let first_delay = latency.first_token_delay(
            &mut rng,
            &FirstTokenCtx {
                num_prompt_tokens: prompt_len,
                num_cached_tokens: num_local_cached_tokens,
                do_remote_prefill: remote_prefill,
                num_running,
            },
        );
        let time_scale = if opt.time_scale > 0.0 {
            opt.time_scale
        } else {
            1.0
        };
        // Step alignment: when other requests are mid-step, a new prefill waits
        // for the in-flight step before its chunk runs, so the first token lands
        // roughly one inter-token gap later. The paced draw conditions that gap
        // on this batch's context/concurrency and pins the request's decode
        // donor in the process. An idle engine starts the prefill immediately.
        let step_wait = if num_running > 1 {
            latency.paced_inter_token_delay(&mut rng, num_running, &mut pacing)
        } else {
            Duration::ZERO
        };

        Ok(ActiveRequest {
            rng,
            pacing,
            request_id,
            client_index,
            prompt_len,
            prompt_token_ids,
            max_tokens,
            generated: 0,
            prefill_advertise,
            remote_prefill,
            num_local_cached_tokens,
            block_ids,
            lora_name,
            prefill_service: first_delay,
            step_surcharges: VecDeque::new(),
            next_at: Instant::now() + (first_delay + step_wait).div_f64(time_scale),
        })
    }

    /// The number of tokens this request should emit on the next step.
    fn chunk_len(&self, output_token_chunk_size: usize) -> usize {
        let remaining = self.max_tokens - self.generated;
        remaining.min(output_token_chunk_size)
    }

    /// Advance this request by one engine step, using externally-generated tokens.
    /// The caller is responsible for drawing tokens from the `TokenSource` (using
    /// `self.rng`) before calling this, so the rng draw order is preserved.
    fn step(&mut self, new_token_ids: Vec<u32>) -> EngineCoreOutput {
        self.generated += new_token_ids.len();

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
            // Cached prompt tokens are never recomputed, so they consume no
            // prefill budget; vLLM charges num_new_tokens (the uncached slice).
            let uncached = self.prompt_len.saturating_sub(self.num_local_cached_tokens);
            prefill_token_demand(uncached, long_prefill_threshold, max_batched_tokens)
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

/// A request that has been admitted (blocks pinned, batch slot reserved) but whose
/// remote KV pull is still in flight on a background thread. The pull result arrives
/// via `pull_completion_rx`; on completion `finish_pull` promotes it to an `ActiveRequest`.
struct PendingPull {
    request: Box<EngineCoreRequest>,
    block_ids: Vec<usize>,
    num_local_cached_tokens: usize,
    client_index: u32,
    prompt_len: usize,
    lora_name: Option<String>,
}

/// Result sent from a `spawn_blocking` pull task back to the engine loop.
type PullCompletion = (String, Result<u64>);

/// Internal state for one engine instance, owned by the engine loop task.
pub(crate) struct SimEngine {
    engine_index: u32,
    opt: Opt,
    latency: Box<dyn LatencyModel>,
    token_source: Box<dyn TokenSource>,
    scheduler: Box<dyn Scheduler>,
    /// Wrapped in `Arc<Mutex>` so `spawn_blocking` pull tasks can call `pull_prefilled` off
    /// the engine loop. The mutex is effectively uncontended: `advertise_prefilled` and
    /// `release` run inline on the engine task (which only does prefill or decode, never both
    /// on the same engine), and the blocking pull runs on a thread pool, so the two callers
    /// never overlap on the same engine instance.
    data_plane: Arc<StdMutex<Box<dyn KvDataPlane>>>,
    /// Prefix-cache + block-slot pool. Drives local cache hits, `kv_cache_usage`,
    /// `prefix_cache_stats`, and the KV-cache events the cache-aware router consumes.
    pool: BlockPool,
    /// Publisher for KV-cache events; `None` when events are disabled.
    events: Option<KvEventTx>,
    /// RNG for failure injection (Phase 5), seeded per engine for reproducibility.
    failure_rng: StdRng,
    /// Loaded LoRA adapters + the running-batch slot cap. Adapters arrive via `add_lora`
    /// utility calls; per-request usage drives the `running/waiting_lora_adapters` stats.
    loras: LoraRegistry,
    /// The running batch: requests being actively decoded. Capped at `max_num_seqs`.
    active_requests: BTreeMap<String, ActiveRequest>,
    /// Admitted-but-not-yet-running requests, in arrival order. Drained into `active_requests`
    /// as slots free up; its length is `vllm:num_requests_waiting`.
    waiting: VecDeque<Box<EngineCoreRequest>>,
    /// Requests whose blocks are pinned and batch slot reserved, but whose remote KV pull
    /// is still in flight. These count toward `running_capacity` and `scheduled_token_demand`
    /// to prevent over-admission while pulls are outstanding.
    pending_pulls: BTreeMap<String, PendingPull>,
    /// Sender half for pull completion results. Cloned into each `spawn_blocking` task.
    pull_completion_tx: mpsc::UnboundedSender<PullCompletion>,
    /// Receiver half; wrapped in Option so `take_internal_rx` can hand it to the loop once.
    pull_completion_rx: Option<mpsc::UnboundedReceiver<PullCompletion>>,
}

impl SimEngine {
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

    /// Handle a LoRA load/unload utility call, mutating the adapter registry. Returns the
    /// `bool`-carrying response the frontend awaits, or `None` if this isn't a LoRA method (so
    /// the caller falls back to the generic `utility_response`). The frontend encodes both
    /// methods' args as a one-element tuple: `(LoraRequest,)` for `add_lora`, `(lora_int_id,)`
    /// for `remove_lora`.
    fn lora_utility_outputs(
        &mut self,
        request: &EngineCoreUtilityRequest,
    ) -> Result<Option<EngineCoreOutputs>> {
        let ok = match request.method_name.as_str() {
            "add_lora" => {
                let (lora,): (LoraRequest,) = rmpv::ext::from_value(request.args.clone())
                    .map_err(|error| anyhow!("decoding add_lora args: {error}"))?;
                let name = lora.lora_name.clone();
                let id = lora.lora_int_id;
                let ok = self.loras.add(lora);
                info!(engine_index = self.engine_index, lora = %name, id, "add_lora");
                ok
            }
            "remove_lora" => {
                let (lora_int_id,): (u64,) = rmpv::ext::from_value(request.args.clone())
                    .map_err(|error| anyhow!("decoding remove_lora args: {error}"))?;
                let ok = self.loras.remove(lora_int_id);
                info!(
                    engine_index = self.engine_index,
                    id = lora_int_id,
                    ok,
                    "remove_lora"
                );
                ok
            }
            _ => return Ok(None),
        };
        Ok(Some(utility_result_outputs(
            self.engine_index,
            request.call_id,
            utility_envelope(ok)?,
        )))
    }

    /// Handle the `reset_prefix_cache` utility. Real vLLM refuses the reset when any blocks
    /// are pinned (requests running) and returns false. Waiting-queue requests hold no pins
    /// (pinning happens at admission), so they do not block a reset.
    fn reset_prefix_cache_outputs(
        &mut self,
        request: &EngineCoreUtilityRequest,
    ) -> Result<Option<EngineCoreOutputs>> {
        if request.method_name != "reset_prefix_cache" {
            return Ok(None);
        }
        let ok = if self.active_requests.is_empty() && self.pending_pulls.is_empty() {
            if let Some(event) = self.pool.reset()
                && let Some(events) = &self.events
            {
                events.publish(vec![event]);
            }
            true
        } else {
            warn!(
                engine_index = self.engine_index,
                num_running = self.active_requests.len(),
                num_pending_pulls = self.pending_pulls.len(),
                "refusing reset_prefix_cache: requests are still running or pulls pending"
            );
            false
        };
        Ok(Some(utility_result_outputs(
            self.engine_index,
            request.call_id,
            utility_envelope(ok)?,
        )))
    }

    /// Count running requests per LoRA adapter (from the batch) and waiting requests per
    /// adapter (from the queue) for the `running_lora_adapters`/`waiting_lora_adapters`
    /// scheduler stats. Base-model requests carry no adapter and are not counted.
    fn lora_counts(&self) -> (BTreeMap<String, u64>, BTreeMap<String, u64>) {
        let mut running = BTreeMap::new();
        for request in self.active_requests.values() {
            if let Some(name) = &request.lora_name {
                *running.entry(name.clone()).or_insert(0) += 1;
            }
        }
        // Pending pulls hold batch slots and count as running for LoRA accounting.
        for pending in self.pending_pulls.values() {
            if let Some(name) = &pending.lora_name {
                *running.entry(name.clone()).or_insert(0) += 1;
            }
        }
        let mut waiting = BTreeMap::new();
        for request in &self.waiting {
            if let Some(lora) = &request.lora_request {
                *waiting.entry(lora.lora_name.clone()).or_insert(0) += 1;
            }
        }
        (running, waiting)
    }

    /// Whether the LoRA slot cap lets this request join the running batch right now (see
    /// `LoraRegistry::admits`). The running batch's distinct adapters are read live.
    fn lora_admits(&self, request: &EngineCoreRequest) -> bool {
        let lora_name = request.lora_request.as_ref().map(|l| l.lora_name.as_str());
        self.loras.admits(
            lora_name,
            self.active_requests
                .values()
                .filter_map(|r| r.lora_name.as_deref())
                .chain(
                    self.pending_pulls
                        .values()
                        .filter_map(|p| p.lora_name.as_deref()),
                ),
        )
    }

    /// Drain one frontend request message.
    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        let mut outputs = Vec::new();

        match input {
            EngineInput::Request(request) => {
                let request_id = request.request_id.clone();
                let client_index = request.client_index;

                // Dedup against the running batch, pending pulls, and the waiting queue.
                if self.active_requests.contains_key(&request_id)
                    || self.pending_pulls.contains_key(&request_id)
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
                    // Release is cheap (a memset or no-op), lock inline.
                    if let Ok(mut plane) = self.data_plane.lock() {
                        plane.release(&request_id);
                    }
                    // A request is running, pending-pull, or waiting; abort whichever.
                    let client_index =
                        if let Some(request) = self.active_requests.remove(&request_id) {
                            self.pool.unpin(&request.block_ids);
                            self.token_source.on_request_finished(&request.request_id);
                            Some(request.client_index)
                        } else if let Some(pending) = self.pending_pulls.remove(&request_id) {
                            // Pull is in flight on a background thread; unpin blocks and let
                            // the orphaned task's completion be dropped in finish_pull.
                            self.pool.unpin(&pending.block_ids);
                            Some(pending.client_index)
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
                let client_index = request.client_index;
                // Stateful utilities (LoRA, prefix-cache reset) mutate engine state; try them
                // first, then fall back to the stateless generic responder.
                let outputs_msg = if let Some(out) = self.reset_prefix_cache_outputs(&request)? {
                    out
                } else if let Some(out) = self.lora_utility_outputs(&request)? {
                    out
                } else {
                    utility_response(self.engine_index, request)?
                };
                outputs.push(EngineOutput {
                    client_index,
                    outputs: outputs_msg,
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

    /// Move a request from the waiting queue into the running batch. For requests with a
    /// remote KV descriptor (`do_remote_prefill`), the pull is spawned on a blocking thread
    /// and the request is parked in `pending_pulls` until completion. The first-token clock
    /// starts when the pull finishes (in `finish_pull`), so real NIXL users should leave the
    /// `kv_cache_transfer` latency knobs at 0 to avoid double-counting the transfer time
    /// (the actual transfer wall time is the delay). Sim-timing users running the noop plane
    /// still get the modeled transfer delay from the latency knobs.
    ///
    /// Requests without remote KV go through single-phase admission: `ActiveRequest::new`
    /// is called inline and the request enters `active_requests` immediately. Returns an
    /// immediate-finish output if the request is invalid (e.g. `max_tokens == 0`).
    fn admit(&mut self, request: Box<EngineCoreRequest>) -> Option<EngineOutput> {
        let request_id = request.request_id.clone();
        let client_index = request.client_index;
        let remote = extract_kv_params(&request)
            .as_ref()
            .and_then(parse_remote_kv);
        let prompt_len = request
            .prompt_token_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or_default();

        // === BLOCK POOL: cache this prompt locally ===
        // Measure the local prefix hit, allocate slots for the new blocks, pin them all, and
        // emit BlockStored/BlockRemoved for the router. The slot ids are what the data plane
        // pages over NIXL and what we advertise as remote_block_ids.
        let lora_name = request.lora_request.as_ref().map(|l| l.lora_name.as_str());
        let prompt_slice = request.prompt_token_ids.as_deref().unwrap_or_default();
        let outcome =
            self.pool
                .cache_prompt(prompt_slice, lora_name, request.cache_salt.as_deref());
        if let Some(events) = &self.events {
            events.publish(outcome.events);
        }
        let num_local_cached = outcome.num_cached_tokens;
        let block_ids = outcome.block_ids;

        // === DATA PLANE: two-phase decode-side pull ===
        // A do_remote_prefill request carries the prefill engine's remote_* descriptor. Instead
        // of pulling inline (which blocks the engine loop for the entire TCP connect + transfer),
        // park the request and spawn the pull on a blocking thread. The completion arrives via
        // `pull_completion_rx` and `finish_pull` promotes it to an ActiveRequest.
        if let Some(remote) = remote {
            let lora_name_owned = request.lora_request.as_ref().map(|l| l.lora_name.clone());
            self.pending_pulls.insert(
                request_id.clone(),
                PendingPull {
                    request,
                    block_ids: block_ids.clone(),
                    num_local_cached_tokens: num_local_cached,
                    client_index,
                    prompt_len,
                    lora_name: lora_name_owned,
                },
            );

            let dp = Arc::clone(&self.data_plane);
            let tx = self.pull_completion_tx.clone();
            let rid = request_id.clone();
            let block_ids_for_pull = block_ids;
            // Use a plain OS thread rather than tokio::spawn_blocking so the pull works
            // both in the async engine loop and in sync #[test] contexts (which have no
            // tokio runtime). The thread does blocking IO (TCP connect + poll loop for
            // real NIXL), sends the result on the tokio mpsc (safe from any thread), and
            // exits. The thread is detached; its result is consumed via pull_completion_rx.
            if let Err(e) = std::thread::Builder::new()
                .name(format!("kv-pull-{}", rid))
                .spawn(move || {
                    let kv = RequestKv {
                        request_id: &rid,
                        num_tokens: prompt_len,
                        block_ids: &block_ids_for_pull,
                    };
                    let result = match dp.lock() {
                        Ok(mut plane) => plane.pull_prefilled(kv, &remote),
                        Err(poisoned) => {
                            // Mutex poisoned (a prior pull panicked). Treat as pull failure
                            // rather than propagating the panic; the engine keeps running.
                            let mut plane = poisoned.into_inner();
                            plane
                                .pull_prefilled(kv, &remote)
                                .map_err(|e| anyhow!("pull after mutex poison recovery: {e}"))
                        }
                    };
                    // If the receiver is dropped (engine shut down), this send failing is fine.
                    let _ = tx.send((rid, result));
                })
            {
                // Thread spawn failed (extremely rare, resource exhaustion). Send a failure
                // completion so finish_pull handles it on the next loop turn.
                warn!(request_id, %e, "failed to spawn pull thread; reporting pull failure");
                let _ = self
                    .pull_completion_tx
                    .send((request_id, Err(anyhow!("failed to spawn pull thread: {e}"))));
            }
            return None;
        }

        // Single-phase admission: no remote KV, admit directly.
        self.admit_direct(
            request_id,
            client_index,
            request,
            block_ids,
            num_local_cached,
        )
    }

    /// Complete single-phase admission by building the `ActiveRequest` and inserting it into
    /// the running batch. Also used by `finish_pull` after a two-phase pull completes.
    fn admit_direct(
        &mut self,
        request_id: String,
        client_index: u32,
        request: Box<EngineCoreRequest>,
        block_ids: Vec<usize>,
        num_local_cached: usize,
    ) -> Option<EngineOutput> {
        // Count this request among the running set for its own load-factor scaling.
        // Include pending pulls in the count since they occupy batch slots.
        let num_running = (self.active_requests.len() + self.pending_pulls.len()) as u64 + 1;
        match ActiveRequest::new(
            self.engine_index,
            request,
            &self.opt,
            &*self.latency,
            num_running,
            num_local_cached,
            block_ids.clone(),
        ) {
            Ok(active) => {
                // This admission runs its prefill as chunked steps in the
                // engine's single serial step stream, so every running
                // decode's next few gaps stretch by one chunk's service time
                // each (the prefill's total service spread over its chunks).
                // Remote-prefill requests (P/D decode side) burned their
                // prefill on another node and merely join the batch; full
                // cache hits run no chunk. Either way they block nothing.
                let blocks_engine = !active.remote_prefill
                    && active.prompt_len > num_local_cached
                    && !self.active_requests.is_empty();
                if blocks_engine {
                    let budget = (self.opt.max_num_batched_tokens as usize).max(1);
                    let uncached = active.prompt_len - num_local_cached;
                    let chunks = uncached.div_ceil(budget);
                    let per_chunk = active.prefill_service.div_f64(chunks as f64);
                    for running in self.active_requests.values_mut() {
                        running
                            .step_surcharges
                            .extend(std::iter::repeat_n(per_chunk, chunks));
                    }
                }
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

    /// Handle a completed pull from the background thread. Promotes the pending request to
    /// an `ActiveRequest` (starting its first-token clock now, after the transfer) and runs
    /// `schedule()` since batch budget may have changed. Returns any outputs produced.
    fn finish_pull(&mut self, request_id: String, result: Result<u64>) -> Vec<EngineOutput> {
        let Some(pending) = self.pending_pulls.remove(&request_id) else {
            // The request was aborted while the pull was in flight; its blocks are already
            // unpinned and its abort output already sent. Drop the orphaned result.
            debug!(
                request_id,
                "pull completed for unknown/aborted request; dropping"
            );
            return Vec::new();
        };

        match &result {
            Ok(bytes) => info!(request_id, bytes, "pulled remote KV before decode"),
            Err(error) => warn!(request_id, %error, "remote KV pull failed (admitting anyway)"),
        }

        let mut outputs = Vec::new();
        if let Some(out) = self.admit_direct(
            request_id,
            pending.client_index,
            pending.request,
            pending.block_ids,
            pending.num_local_cached_tokens,
        ) {
            outputs.push(out);
        }
        // Completing a pull may free budget if the request was invalid; try to admit more.
        outputs.extend(self.schedule());
        outputs
    }

    /// Current per-step token demand of the running batch plus pending pulls, charged against
    /// `max_num_batched_tokens`. Pending pulls count as prefilling requests (their prompt chunk)
    /// so the scheduler cannot over-admit while pulls are in flight.
    fn scheduled_token_demand(&self) -> usize {
        let threshold = self.opt.long_prefill_token_threshold as usize;
        let budget = self.opt.max_num_batched_tokens as usize;
        let active: usize = self
            .active_requests
            .values()
            .map(|request| request.token_demand(threshold, budget))
            .sum();
        let pending: usize = self
            .pending_pulls
            .values()
            .map(|p| prefill_token_demand(p.prompt_len, threshold, budget))
            .sum();
        active + pending
    }

    /// Index of the next waiting request to admit, delegating to the plugged `Scheduler`
    /// strategy and filtering through the LoRA slot cap. LoRA-blocked requests are skipped
    /// over (they stay queued and show up as `num_skipped_waiting_reqs`), matching vLLM
    /// rather than head-of-line blocking the whole queue on one stuck adapter.
    fn next_admissible_index(&self) -> Option<usize> {
        self.scheduler
            .next_admissible(&self.waiting, &|r| self.lora_admits(r))
    }

    /// Number of waiting requests the LoRA slot cap is currently blocking (vLLM's skipped
    /// waiting queue). Computed against the running batch's resident adapters.
    fn num_lora_skipped(&self) -> u64 {
        self.waiting.iter().filter(|r| !self.lora_admits(r)).count() as u64
    }

    /// Admit waiting requests into the running batch (in policy order) until the seq cap
    /// (`max_num_seqs`) or the per-step token budget (`max_num_batched_tokens`) is reached, or
    /// the queue empties. This mirrors vLLM's scheduler filling the batch under both bounds.
    /// Returns any immediate-finish outputs produced during admission.
    fn schedule(&mut self) -> Vec<EngineOutput> {
        let mut outputs = Vec::new();
        let budget = self.opt.max_num_batched_tokens as usize;

        // The `demand < budget` check admits the last request even when its prefill chunk
        // would overshoot. This matches vLLM's chunked-prefill semantics: vLLM admits that
        // request too and simply shrinks its chunk to the remaining budget. The admission
        // COUNT is identical; only per-step token accounting differs.
        while self.active_requests.len() + self.pending_pulls.len() < self.running_capacity()
            // Recomputed each admission: the freshly admitted request's demand
            // depends on its prefix-cache hit, which only admit() can measure.
            && self.scheduled_token_demand() < budget
        {
            // Next admissible request in policy order, skipping any the LoRA slot cap blocks.
            // The index comes from a scan of `waiting` under the same borrow, so remove()
            // only returns None on a buggy Scheduler impl; treat that as nothing admissible.
            let Some(request) = self
                .next_admissible_index()
                .and_then(|idx| self.waiting.remove(idx))
            else {
                break;
            };
            // Invalid requests finish immediately and never enter the batch;
            // admitted ones occupy budget (via scheduled_token_demand) until
            // their first token.
            if let Some(output) = self.admit(request) {
                outputs.push(output);
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

        // Split the token_source out of self so we can borrow active_requests mutably
        // while also calling the source. Swapped back after the loop.
        let mut token_source = std::mem::replace(
            &mut self.token_source,
            Box::new(RandomTokens { vocab_size: 0 }),
        );

        for request in self.active_requests.values_mut() {
            if request.next_at > now {
                continue;
            }
            let client_index = request.client_index;
            let is_first = request.generated == 0;

            // Token generation: draw from the request rng via the token source FIRST,
            // then draw the inter-token delay from the same rng. This preserves the
            // original rng draw order exactly.
            let chunk_len = request.chunk_len(self.opt.output_token_chunk_size);
            let ctx = TokenCtx {
                request_id: &request.request_id,
                prompt_token_ids: &request.prompt_token_ids,
                num_generated: request.generated,
            };
            let new_token_ids = token_source.next_tokens(&ctx, chunk_len, &mut request.rng);

            let mut output = request.step(new_token_ids);
            let request_id = request.request_id.clone();
            let finished = output.finished();

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
                let gap = self.latency.paced_inter_token_delay(
                    &mut request.rng,
                    num_running,
                    &mut request.pacing,
                );
                let surcharge = request
                    .step_surcharges
                    .pop_front()
                    .unwrap_or(Duration::ZERO);
                let time_scale = if self.opt.time_scale > 0.0 {
                    self.opt.time_scale
                } else {
                    1.0
                };
                request.next_at = now + (gap + surcharge).div_f64(time_scale);
            }

            let (outs, finished_set) = by_client
                .entry(client_index)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            if finished {
                finished_set.insert(request_id);
            }
            outs.push(std::mem::take(&mut output));
        }

        // Swap the token source back.
        self.token_source = token_source;

        for request_id in &finished_ids {
            if let Some(request) = self.active_requests.remove(request_id) {
                self.pool.unpin(&request.block_ids);
                self.token_source.on_request_finished(&request.request_id);
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
            // Advertise is a memset, lock inline (fast and uncontended, see data_plane doc).
            let adv_result = match self.data_plane.lock() {
                Ok(mut plane) => plane.advertise_prefilled(kv),
                Err(poisoned) => poisoned.into_inner().advertise_prefilled(kv),
            };
            match adv_result {
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
        let (running_lora_adapters, waiting_lora_adapters) = self.lora_counts();
        // Pending pulls count as running: the request has left the waiting queue, holds pinned
        // blocks, and occupies a batch slot. This matches vLLM's WAITING_FOR_REMOTE_KVS
        // accounting, which is grouped under num_running_reqs in the stats surface.
        SchedulerStats {
            num_running_reqs: (self.active_requests.len() + self.pending_pulls.len()) as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            num_skipped_waiting_reqs: self.num_lora_skipped(),
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
            running_lora_adapters,
            waiting_lora_adapters,
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

impl SimEngine {
    /// Build a fully-initialized engine. KV-event publishing is async (ZMQ bind), so
    /// construction is async too. The caller owns the loop; see `core::run_loop`.
    pub(crate) async fn new(
        engine_index: u32,
        opt: Opt,
        events: Option<KvEventTx>,
    ) -> Result<SimEngine> {
        let role = opt.pd_role;
        let cfg = NixlConfig {
            kv_block_bytes: opt.kv_block_bytes,
            tokens_per_block: opt.tokens_per_block,
            kv_cache_blocks: opt.kv_cache_size as usize,
            engine_id: opt.engine_id.clone(),
            side_channel_host: opt.side_channel_host.clone(),
            side_channel_port: opt.side_channel_port + engine_index,
        };
        let latency: Box<dyn LatencyModel> = opt.build_latency()?;
        let token_source: Box<dyn TokenSource> = Box::new(RandomTokens {
            vocab_size: opt.vocab_size,
        });
        let scheduler: Box<dyn Scheduler> = match opt.scheduling_policy {
            SchedulingPolicy::Fcfs => Box::new(sched::Fcfs),
            SchedulingPolicy::Priority => Box::new(sched::Priority),
        };
        let opt_seed = opt.seed;
        let max_loras = opt.max_loras as usize;
        let pool = BlockPool::new(
            opt.tokens_per_block,
            opt.kv_cache_size as usize,
            opt.kv_cache_none_seed,
        );
        let (pull_completion_tx, pull_completion_rx) = mpsc::unbounded_channel();
        Ok(SimEngine {
            engine_index,
            opt,
            latency,
            token_source,
            scheduler,
            data_plane: Arc::new(StdMutex::new(make_data_plane(role, cfg))),
            pool,
            events,
            failure_rng: StdRng::seed_from_u64(
                opt_seed ^ (engine_index as u64).wrapping_mul(0x9e3779b9),
            ),
            loras: LoraRegistry::new(max_loras),
            active_requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            pending_pulls: BTreeMap::new(),
            pull_completion_tx,
            pull_completion_rx: Some(pull_completion_rx),
        })
    }
}

impl EngineCore for SimEngine {
    type Internal = PullCompletion;

    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        SimEngine::handle_input(self, input)
    }

    fn take_internal_rx(&mut self) -> mpsc::UnboundedReceiver<Self::Internal> {
        match self.pull_completion_rx.take() {
            Some(rx) => rx,
            None => {
                warn!("take_internal_rx called more than once; returning a dummy channel");
                let (_tx, rx) = mpsc::unbounded_channel();
                rx
            }
        }
    }

    fn on_internal(&mut self, (request_id, result): Self::Internal) -> Vec<EngineOutput> {
        self.finish_pull(request_id, result)
    }

    fn earliest_deadline(&self) -> Option<Instant> {
        SimEngine::earliest_deadline(self)
    }

    fn step(&mut self) -> Vec<EngineOutput> {
        SimEngine::step(self)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use clap::Parser as _;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};

    use super::*;
    use crate::dataplane::{NixlConfig, PdRole, make_data_plane};
    use crate::engine_core::{EngineInput, EngineOutput};

    fn test_opt() -> Opt {
        // clap fills every field with its declared default (all latency knobs = 0 / instant).
        Opt::parse_from(["inference-sim"])
    }

    /// Build a test engine with a pre-taken internal rx for sync test usage. Returns the
    /// engine and the pull completion receiver (tests that need it can recv directly).
    fn test_engine(opt: Opt) -> (SimEngine, mpsc::UnboundedReceiver<PullCompletion>) {
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
        let latency: Box<dyn crate::latency::LatencyModel> = opt.build_latency().unwrap();
        let token_source: Box<dyn crate::tokens::TokenSource> =
            Box::new(crate::tokens::RandomTokens {
                vocab_size: opt.vocab_size,
            });
        let scheduler: Box<dyn crate::sched::Scheduler> = match opt.scheduling_policy {
            SchedulingPolicy::Fcfs => Box::new(crate::sched::Fcfs),
            SchedulingPolicy::Priority => Box::new(crate::sched::Priority),
        };
        let (pull_completion_tx, pull_completion_rx) = mpsc::unbounded_channel();
        let mut engine = SimEngine {
            engine_index: 0,
            latency,
            token_source,
            scheduler,
            data_plane: Arc::new(StdMutex::new(make_data_plane(PdRole::Both, cfg))),
            pool,
            events: None,
            failure_rng: StdRng::seed_from_u64(opt.seed),
            loras: LoraRegistry::new(opt.max_loras as usize),
            active_requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            pending_pulls: BTreeMap::new(),
            pull_completion_tx,
            pull_completion_rx: Some(pull_completion_rx),
            opt,
        };
        let rx = engine.pull_completion_rx.take().unwrap();
        (engine, rx)
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

    /// Apply pull completions until no pulls are pending. Every spawned pull thread sends
    /// exactly one completion and the engine holds a sender (the channel can't close), so
    /// blocking on the channel is deterministic, no sleeps needed. A no-op for requests
    /// without remote KV. Stray completions for already-aborted requests are consumed along
    /// the way (finish_pull drops them).
    fn settle_pulls(
        engine: &mut SimEngine,
        rx: &mut mpsc::UnboundedReceiver<PullCompletion>,
    ) -> Vec<EngineOutput> {
        let mut outputs = Vec::new();
        while !engine.pending_pulls.is_empty() {
            let (request_id, result) = rx
                .blocking_recv()
                .expect("engine holds a completion sender");
            outputs.extend(engine.finish_pull(request_id, result));
        }
        outputs
    }

    fn add(
        engine: &mut SimEngine,
        rx: &mut mpsc::UnboundedReceiver<PullCompletion>,
        req: EngineCoreRequest,
    ) {
        engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input");
        settle_pulls(engine, rx);
    }

    /// Drain steps until the engine is idle, returning the flat output list. Safe only when
    /// the latency model is instant (deadlines never in the future), as in these tests.
    fn drain(
        engine: &mut SimEngine,
        rx: &mut mpsc::UnboundedReceiver<PullCompletion>,
    ) -> Vec<EngineCoreOutput> {
        let mut all = Vec::new();
        while !engine.active_requests.is_empty() || !engine.pending_pulls.is_empty() {
            for out in settle_pulls(engine, rx) {
                all.extend(out.outputs.outputs);
            }
            // settle_pulls emptied pending_pulls, so the batch is non-empty whenever the
            // loop condition held: an empty batch here means a stall.
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
        let (mut engine, mut rx) = test_engine(test_opt());
        add(&mut engine, &mut rx, request("r1", 4, 3));
        // Instant model: the first token is due immediately.
        assert!(engine.earliest_deadline().unwrap() <= Instant::now());
        let outputs = drain(&mut engine, &mut rx);
        let tokens: usize = outputs.iter().map(|o| o.new_token_ids.len()).sum();
        assert_eq!(tokens, 3);
        assert!(outputs.last().unwrap().finished());
    }

    #[test]
    fn ttft_delays_the_first_token() {
        let mut opt = test_opt();
        opt.time_to_first_token = 10_000; // 10s, no std-dev -> exact, comfortably in the future
        let (mut engine, mut rx) = test_engine(opt);

        let before = Instant::now();
        add(&mut engine, &mut rx, request("r1", 4, 2));
        let deadline = engine.earliest_deadline().expect("a deadline");
        assert!(deadline >= before + Duration::from_millis(9_000));

        // The token is not due yet, so a step right now produces nothing and keeps the request.
        assert!(engine.step().is_empty());
        assert_eq!(engine.active_requests.len(), 1);
    }

    #[test]
    fn prefill_stats_only_on_first_output() {
        let (mut engine, mut rx) = test_engine(test_opt());
        add(&mut engine, &mut rx, request("r1", 7, 4));
        let outputs = drain(&mut engine, &mut rx);

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
        let (mut engine, mut rx) = test_engine(opt);

        // One request, 32 prompt tokens -> ceil(32/16) = 2 blocks while running.
        add(&mut engine, &mut rx, request("r1", 32, 5));
        let stats = engine.scheduler_stats();
        assert_eq!(stats.num_running_reqs, 1);
        assert_eq!(stats.num_waiting_reqs, 0);
        assert!(
            (stats.kv_cache_usage - 0.02).abs() < 1e-9,
            "got {}",
            stats.kv_cache_usage
        );

        // Drain to completion; the engine is empty so usage is back to zero.
        let _ = drain(&mut engine, &mut rx);
        let idle = engine.scheduler_stats();
        assert_eq!(idle.num_running_reqs, 0);
        assert_eq!(idle.kv_cache_usage, 0.0);
    }

    #[test]
    fn final_batch_reports_zero_running() {
        let (mut engine, mut rx) = test_engine(test_opt());
        add(&mut engine, &mut rx, request("r1", 4, 1)); // single output token, finishes in one step
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
        let (mut engine, mut rx) = test_engine(test_opt());
        let mut req = request("r1", 9, 2);
        let mut extra = HashMap::new();
        extra.insert(
            "kv_transfer_params".to_string(),
            serde_json::json!({ "do_remote_prefill": true }),
        );
        req.sampling_params.as_mut().unwrap().extra_args = Some(extra);
        add(&mut engine, &mut rx, req);

        let outputs = drain(&mut engine, &mut rx);
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
        let (mut engine, mut rx) = test_engine(opt);

        // First request is cold: both blocks are computed, none cached.
        add(&mut engine, &mut rx, request("r1", 8, 1));
        let first = drain(&mut engine, &mut rx);
        let s1 = first
            .iter()
            .find_map(|o| o.prefill_stats.as_ref())
            .expect("prefill stats");
        assert_eq!(s1.num_local_cached_tokens, 0);
        assert_eq!(s1.num_computed_tokens, 8);

        // r1's blocks stay cached after it finishes. The identical prompt now fully hits.
        add(&mut engine, &mut rx, request("r2", 8, 1));
        let second = drain(&mut engine, &mut rx);
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
    fn cache_salt_isolates_otherwise_identical_prompts() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4; // 8-token prompt = 2 full blocks
        let (mut engine, mut rx) = test_engine(opt);

        let salted = |id: &str, salt: &str| EngineCoreRequest {
            cache_salt: Some(salt.to_string()),
            ..request(id, 8, 1)
        };

        // r1 warms the cache under salt "a".
        add(&mut engine, &mut rx, salted("r1", "a"));
        let _ = drain(&mut engine, &mut rx);

        // Same prompt under a different salt must recompute, not hit r1's blocks.
        add(&mut engine, &mut rx, salted("r2", "b"));
        let s2 = drain(&mut engine, &mut rx)
            .iter()
            .find_map(|o| o.prefill_stats.as_ref().cloned())
            .expect("prefill stats");
        assert_eq!(
            s2.num_local_cached_tokens, 0,
            "different salt -> no prefix hit"
        );
        assert_eq!(s2.num_computed_tokens, 8);

        // Same prompt under the same salt "a" hits fully.
        add(&mut engine, &mut rx, salted("r3", "a"));
        let s3 = drain(&mut engine, &mut rx)
            .iter()
            .find_map(|o| o.prefill_stats.as_ref().cloned())
            .expect("prefill stats");
        assert_eq!(s3.num_local_cached_tokens, 8, "same salt -> full hit");
    }

    #[test]
    fn lora_isolates_prefix_cache_across_adapters() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4; // 8-token prompt = 2 full blocks
        let (mut engine, mut rx) = test_engine(opt);

        // request() builds an all-zero 8-token prompt; only the adapter differs. max_tokens 1
        // so each finishes in one step.
        let with_lora = |id: &str, name: &str| EngineCoreRequest {
            lora_request: Some(LoraRequest::new(
                name.to_string(),
                1,
                format!("/loras/{name}"),
                false,
                false,
            )),
            ..request(id, 8, 1)
        };

        // r1 (adapter A) warms the cache; a different adapter must recompute, not hit A's blocks.
        add(&mut engine, &mut rx, with_lora("r1", "A"));
        let _ = drain(&mut engine, &mut rx);
        add(&mut engine, &mut rx, with_lora("r2", "B"));
        let s2 = drain(&mut engine, &mut rx)
            .iter()
            .find_map(|o| o.prefill_stats.as_ref().cloned())
            .expect("prefill stats");
        assert_eq!(
            s2.num_local_cached_tokens, 0,
            "different adapter -> no prefix hit"
        );

        // Same adapter A, same prompt -> full hit.
        add(&mut engine, &mut rx, with_lora("r3", "A"));
        let s3 = drain(&mut engine, &mut rx)
            .iter()
            .find_map(|o| o.prefill_stats.as_ref().cloned())
            .expect("prefill stats");
        assert_eq!(s3.num_local_cached_tokens, 8, "same adapter -> full hit");
    }

    #[test]
    fn scheduler_stats_carry_prefix_cache_counters() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4;
        let (mut engine, mut rx) = test_engine(opt);

        // Cold request: its single step reports 2 queries and 0 hits (delta since last drain).
        add(&mut engine, &mut rx, request("r1", 8, 1));
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
        let (mut engine, mut rx) = test_engine(opt);

        // prompt 8 + max_tokens 5 = 13 > 10: context-length error, never queued.
        let out = submit(&mut engine, &mut rx, request("big", 8, 5));
        assert_eq!(finish_reason(&out[0]), Some(EngineCoreFinishReason::Length));
        assert!(engine.active_requests.is_empty());
        assert!(engine.waiting.is_empty());
    }

    #[test]
    fn max_model_len_admits_a_fitting_request() {
        let mut opt = test_opt();
        opt.max_model_len = 100;
        let (mut engine, mut rx) = test_engine(opt);
        submit(&mut engine, &mut rx, request("ok", 8, 5)); // 13 <= 100
        assert_eq!(engine.active_requests.len(), 1);
    }

    #[test]
    fn failure_injection_at_rate_one_fails_every_request() {
        let mut opt = test_opt();
        opt.failure_injection_rate = 1.0;
        opt.failure_types = vec![crate::FailureType::Error];
        let (mut engine, mut rx) = test_engine(opt);

        let out = submit(&mut engine, &mut rx, request("doomed", 4, 5));
        assert_eq!(finish_reason(&out[0]), Some(EngineCoreFinishReason::Error));
        assert!(engine.active_requests.is_empty());
    }

    #[test]
    fn failure_injection_at_rate_zero_never_fails() {
        let opt = test_opt(); // rate defaults to 0.0
        let (mut engine, mut rx) = test_engine(opt);
        let out = submit(&mut engine, &mut rx, request("safe", 4, 5));
        assert!(
            out.iter()
                .all(|o| finish_reason(o) != Some(EngineCoreFinishReason::Error))
        );
        assert_eq!(engine.active_requests.len(), 1);
    }

    fn submit(
        engine: &mut SimEngine,
        rx: &mut mpsc::UnboundedReceiver<PullCompletion>,
        req: EngineCoreRequest,
    ) -> Vec<EngineOutput> {
        let mut out = engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input");
        out.extend(settle_pulls(engine, rx));
        out
    }

    fn abort(
        engine: &mut SimEngine,
        rx: &mut mpsc::UnboundedReceiver<PullCompletion>,
        ids: &[&str],
    ) -> Vec<EngineOutput> {
        let ids = ids.iter().map(|s| s.to_string()).collect();
        let mut out = engine
            .handle_input(EngineInput::Abort(ids))
            .expect("handle_input");
        // Aborting may free batch slots and admit queued remote-KV requests; settle their pulls.
        out.extend(settle_pulls(engine, rx));
        out
    }

    /// The finish reason on a single-output engine batch (used to inspect rejections/aborts).
    fn finish_reason(out: &EngineOutput) -> Option<EngineCoreFinishReason> {
        out.outputs.outputs.first().and_then(|o| o.finish_reason)
    }

    #[test]
    fn batch_capped_at_max_num_seqs_rest_wait() {
        let mut opt = test_opt();
        opt.max_num_seqs = 2;
        let (mut engine, mut rx) = test_engine(opt);

        // Five long requests, batch holds 2; the other three wait.
        for i in 0..5 {
            submit(&mut engine, &mut rx, request(&format!("r{i}"), 4, 50));
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
        let (mut engine, mut rx) = test_engine(opt);

        // Three single-token requests through a batch of 1: each step finishes one and admits
        // the next, in arrival order.
        for i in 0..3 {
            submit(&mut engine, &mut rx, request(&format!("r{i}"), 4, 1));
        }
        assert_eq!(engine.active_requests.len(), 1);
        assert_eq!(engine.waiting.len(), 2);

        let finished_order: Vec<String> = drain(&mut engine, &mut rx)
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
        let (mut engine, mut rx) = test_engine(opt);

        submit(&mut engine, &mut rx, request("running", 4, 50));
        submit(&mut engine, &mut rx, request("queued", 4, 50));
        assert_eq!(engine.waiting.len(), 1);

        let out = abort(&mut engine, &mut rx, &["queued"]);
        assert_eq!(finish_reason(&out[0]), Some(EngineCoreFinishReason::Abort));
        assert!(engine.waiting.is_empty());
        assert!(engine.active_requests.contains_key("running"));
    }

    #[test]
    fn aborting_a_running_request_admits_a_waiting_one() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        let (mut engine, mut rx) = test_engine(opt);

        submit(&mut engine, &mut rx, request("running", 4, 50));
        submit(&mut engine, &mut rx, request("queued", 4, 50));
        assert!(engine.active_requests.contains_key("running"));
        assert_eq!(engine.waiting.len(), 1);

        // Freeing the only batch slot pulls the queued request into the batch.
        abort(&mut engine, &mut rx, &["running"]);
        assert!(engine.active_requests.contains_key("queued"));
        assert!(engine.waiting.is_empty());
    }

    #[test]
    fn unbounded_queue_never_rejects() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1; // vLLM never rejects on queue length; neither do we
        let (mut engine, mut rx) = test_engine(opt);

        for i in 0..10 {
            let out = submit(&mut engine, &mut rx, request(&format!("r{i}"), 4, 50));
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
        let (mut engine, mut rx) = test_engine(opt);

        // Each prefilling request demands half its 10 prompt tokens (chunk
        // packing). Budget 20 admits four (demand 0 -> 5 -> 10 -> 15 -> 20),
        // then 20 < 20 is false so the rest wait, despite 96 free seq slots.
        for i in 0..5 {
            submit(&mut engine, &mut rx, request(&format!("r{i}"), 10, 50));
        }
        assert_eq!(engine.active_requests.len(), 4);
        assert_eq!(engine.waiting.len(), 1);
        assert_eq!(engine.scheduler_stats().num_waiting_reqs, 1);
    }

    #[test]
    fn budget_frees_as_prefill_becomes_decode() {
        let mut opt = test_opt();
        opt.max_num_seqs = 100;
        opt.max_num_batched_tokens = 20;
        let (mut engine, mut rx) = test_engine(opt);

        for i in 0..5 {
            submit(&mut engine, &mut rx, request(&format!("r{i}"), 10, 50));
        }
        assert_eq!(engine.active_requests.len(), 4);

        // One step: the four prefilling requests emit their first token (instant model) and
        // become decoders (demand 1 each), freeing budget to admit the last from the queue.
        engine.step();
        assert!(engine.active_requests.len() > 4);
        assert!(engine.waiting.is_empty());
    }

    /// Build a request bound to a LoRA adapter (as the frontend would after the model name
    /// resolved to a loaded adapter).
    fn lora_request(id: &str, lora_name: &str, lora_int_id: u64) -> EngineCoreRequest {
        EngineCoreRequest {
            lora_request: Some(LoraRequest::new(
                lora_name.to_string(),
                lora_int_id,
                format!("/loras/{lora_name}"),
                false,
                false,
            )),
            ..request(id, 4, 50)
        }
    }

    /// Send a utility call and decode its typed result, the way the frontend's client does.
    fn call_utility<T: serde::de::DeserializeOwned, A: serde::Serialize + std::fmt::Debug>(
        engine: &mut SimEngine,
        method: &str,
        args: A,
    ) -> T {
        let request = EngineCoreUtilityRequest::new(0, 1, method, args).expect("build utility");
        let mut out = engine
            .handle_input(EngineInput::Utility(request))
            .expect("handle_input");
        out.remove(0)
            .outputs
            .utility_output
            .expect("utility output")
            .into_typed_result::<T>(method)
            .expect("typed result")
    }

    #[test]
    fn add_and_remove_lora_utilities_report_bool() {
        let (mut engine, _rx) = test_engine(test_opt());
        let lora = LoraRequest::new(
            "adapterA".to_string(),
            7,
            "/loras/a".to_string(),
            false,
            false,
        );
        assert!(call_utility::<bool, _>(&mut engine, "add_lora", (lora,)));
        // First remove finds it; second reports it gone.
        assert!(call_utility::<bool, _>(&mut engine, "remove_lora", (7u64,)));
        assert!(!call_utility::<bool, _>(
            &mut engine,
            "remove_lora",
            (7u64,)
        ));
    }

    #[test]
    fn running_request_counts_into_lora_adapters() {
        let (mut engine, mut rx) = test_engine(test_opt());
        add(&mut engine, &mut rx, lora_request("r1", "adapterA", 1));
        let stats = engine.scheduler_stats();
        assert_eq!(stats.running_lora_adapters.get("adapterA"), Some(&1));
        assert!(stats.waiting_lora_adapters.is_empty());
        // A base-model request adds nothing to the adapter maps.
        add(&mut engine, &mut rx, request("base", 4, 50));
        let stats = engine.scheduler_stats();
        assert_eq!(stats.running_lora_adapters.get("adapterA"), Some(&1));
        assert_eq!(stats.running_lora_adapters.len(), 1);
    }

    #[test]
    fn waiting_requests_count_into_waiting_lora_adapters() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1; // one slot, so the rest queue behind it
        let (mut engine, mut rx) = test_engine(opt);

        add(&mut engine, &mut rx, lora_request("run", "adapterA", 1));
        add(&mut engine, &mut rx, lora_request("wait1", "adapterB", 2));
        add(&mut engine, &mut rx, lora_request("wait2", "adapterB", 2));
        let stats = engine.scheduler_stats();
        assert_eq!(stats.running_lora_adapters.get("adapterA"), Some(&1));
        assert_eq!(
            stats.waiting_lora_adapters.get("adapterB"),
            Some(&2),
            "both queued requests for adapterB counted"
        );
    }

    #[test]
    fn max_loras_blocks_a_new_adapter_until_a_slot_frees() {
        let mut opt = test_opt();
        opt.max_loras = 1; // only one distinct adapter may run at a time
        opt.max_num_seqs = 8; // seq slots are not the binding limit here
        let (mut engine, mut rx) = test_engine(opt);

        // adapterA takes the single LoRA slot; adapterB can't be admitted and waits, even
        // though seq slots are free.
        add(&mut engine, &mut rx, lora_request("a", "adapterA", 1));
        add(&mut engine, &mut rx, lora_request("b", "adapterB", 2));
        assert_eq!(engine.active_requests.len(), 1);
        assert!(engine.active_requests.contains_key("a"));
        assert_eq!(engine.waiting.len(), 1);

        // The blocked adapterB request surfaces as skipped, not just waiting.
        assert_eq!(engine.scheduler_stats().num_skipped_waiting_reqs, 1);

        // A second adapterA request shares the resident slot, so it's admitted even though it
        // arrived behind the blocked adapterB request (skip-and-continue, not head-of-line).
        add(&mut engine, &mut rx, lora_request("a2", "adapterA", 1));
        assert!(
            engine.active_requests.contains_key("a2"),
            "same adapter needs no new slot, skips past the blocked one"
        );
        assert!(
            engine.waiting.iter().any(|r| r.request_id == "b"),
            "adapterB still waits"
        );
    }

    #[test]
    fn max_loras_zero_never_blocks_on_adapter_diversity() {
        let mut opt = test_opt();
        opt.max_loras = 0; // cap disabled
        opt.max_num_seqs = 8;
        let (mut engine, mut rx) = test_engine(opt);

        add(&mut engine, &mut rx, lora_request("a", "adapterA", 1));
        add(&mut engine, &mut rx, lora_request("b", "adapterB", 2));
        add(&mut engine, &mut rx, lora_request("c", "adapterC", 3));
        assert_eq!(engine.active_requests.len(), 3, "all distinct adapters run");
        assert!(engine.waiting.is_empty());
    }

    #[test]
    fn reset_prefix_cache_refused_while_requests_running() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4; // 8-token prompt = 2 full blocks
        let (mut engine, mut rx) = test_engine(opt);

        // Submit a long-running request so the engine is busy.
        submit(&mut engine, &mut rx, request("busy", 8, 100));
        assert_eq!(engine.active_requests.len(), 1);

        // Attempting to reset while busy must return false.
        let ok: bool = call_utility(&mut engine, "reset_prefix_cache", ());
        assert!(!ok, "reset must be refused while requests are running");

        // The cache must survive: an identical prompt should still get a full prefix hit.
        // (The busy request still holds pins, but a second request sharing the same prompt
        // also gets hits on the already-cached blocks.)
        submit(&mut engine, &mut rx, request("r2", 8, 1));
        let r2_stats = drain(&mut engine, &mut rx)
            .iter()
            .find(|o| o.request_id == "r2")
            .and_then(|o| o.prefill_stats.as_ref().cloned())
            .expect("prefill stats for r2");
        assert_eq!(
            r2_stats.num_local_cached_tokens, 8,
            "cache survived the refused reset"
        );
    }

    #[test]
    fn reset_prefix_cache_succeeds_when_idle() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4;
        let (mut engine, mut rx) = test_engine(opt);

        // Warm the cache, then drain so the engine is idle.
        add(&mut engine, &mut rx, request("warm", 8, 1));
        let _ = drain(&mut engine, &mut rx);
        assert!(engine.active_requests.is_empty());

        // Reset should succeed and return true.
        let ok: bool = call_utility(&mut engine, "reset_prefix_cache", ());
        assert!(ok, "reset must succeed when no requests are running");

        // The same prompt should now be a complete miss (cache was cleared).
        add(&mut engine, &mut rx, request("cold", 8, 1));
        let cold_stats = drain(&mut engine, &mut rx)
            .iter()
            .find_map(|o| o.prefill_stats.as_ref().cloned())
            .expect("prefill stats");
        assert_eq!(
            cold_stats.num_local_cached_tokens, 0,
            "cache was cleared by the reset"
        );
    }

    /// Build a do_remote_prefill request (decode side pulling from a remote prefill).
    fn remote_prefill_request(id: &str, prompt_len: usize, max_tokens: u32) -> EngineCoreRequest {
        let mut req = request(id, prompt_len, max_tokens);
        let mut extra = HashMap::new();
        extra.insert(
            "kv_transfer_params".to_string(),
            serde_json::json!({
                "do_remote_prefill": true,
                "remote_engine_id": "prefill-0",
                "remote_host": "127.0.0.1",
                "remote_port": 9999,
                "remote_block_ids": [0, 1],
                "remote_request_id": "prefill-req-1"
            }),
        );
        req.sampling_params.as_mut().unwrap().extra_args = Some(extra);
        req
    }

    #[test]
    fn pending_pull_abort_unpins_blocks_and_emits_abort() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4; // 8-token prompt = 2 blocks
        let (mut engine, mut rx) = test_engine(opt);

        // Submit a do_remote_prefill request. Don't drain completions yet, so it
        // stays in pending_pulls (the noop pull thread may or may not have finished,
        // but either way, the abort must be safe).
        let req = remote_prefill_request("rpull", 8, 5);
        engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input");

        // The request sits in pending_pulls (it has a remote KV descriptor) and its blocks
        // are pinned. We deliberately do NOT apply the completion before aborting.
        assert!(engine.pending_pulls.contains_key("rpull"));
        assert!(
            engine.pool.usage() > 0.0,
            "pending-pull request pins its blocks"
        );

        // Abort the request while it is pending (or already completed but not drained).
        let abort_out = engine
            .handle_input(EngineInput::Abort(vec!["rpull".to_string()]))
            .expect("handle_input");

        // The abort output must carry the abort finish reason.
        let has_abort = abort_out.iter().any(|o| {
            o.outputs
                .outputs
                .iter()
                .any(|out| out.finish_reason == Some(EngineCoreFinishReason::Abort))
        });
        assert!(
            has_abort,
            "abort output must be emitted for pending-pull request"
        );

        // The request must not be in pending_pulls or active_requests.
        assert!(!engine.pending_pulls.contains_key("rpull"));
        assert!(!engine.active_requests.contains_key("rpull"));

        // Blocks must be unpinned: pool usage should be back to 0 (blocks are cached
        // but unpinned, so usage is 0).
        assert_eq!(
            engine.pool.usage(),
            0.0,
            "blocks must be unpinned after abort"
        );

        // The pull thread still sends its completion; receive it deterministically and
        // verify finish_pull drops the orphan (request already gone) without outputs.
        let (orphan_id, orphan_result) = rx
            .blocking_recv()
            .expect("pull thread always sends a completion");
        assert_eq!(orphan_id, "rpull");
        let orphan_outputs = engine.finish_pull(orphan_id, orphan_result);
        assert!(
            orphan_outputs.is_empty(),
            "orphaned pull completion must not produce outputs for aborted request"
        );

        // Verify the same prompt can be submitted again (blocks are reusable).
        add(&mut engine, &mut rx, request("fresh", 8, 2));
        assert!(engine.active_requests.contains_key("fresh"));
        let outputs = drain(&mut engine, &mut rx);
        assert!(
            outputs.iter().any(|o| o.finished()),
            "fresh request completes"
        );
    }

    #[test]
    fn reset_prefix_cache_refused_while_pending_pulls_exist() {
        let mut opt = test_opt();
        opt.tokens_per_block = 4;
        let (mut engine, mut rx) = test_engine(opt);

        // Submit a do_remote_prefill request, don't drain completions.
        let req = remote_prefill_request("rpull", 8, 5);
        engine
            .handle_input(EngineInput::Request(Box::new(req)))
            .expect("handle_input");

        // No active_requests yet, but the pending pull holds pins, so reset must be refused.
        // (The completion may already sit on the channel; it is not applied until settled.)
        assert!(!engine.pending_pulls.is_empty());
        let ok: bool = call_utility(&mut engine, "reset_prefix_cache", ());
        assert!(
            !ok,
            "reset must be refused while pending pulls or active requests exist"
        );

        // Once the pull settles and the request runs to completion, reset succeeds.
        settle_pulls(&mut engine, &mut rx);
        let _ = drain(&mut engine, &mut rx);
        let ok: bool = call_utility(&mut engine, "reset_prefix_cache", ());
        assert!(ok, "reset succeeds once nothing holds pins");
    }

    #[test]
    fn echo_tokens_at_engine_level() {
        let opt = test_opt();
        let (mut engine, mut rx) = test_engine(opt);
        engine.token_source = Box::new(crate::tokens::EchoTokens);

        let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let req = EngineCoreRequest {
            request_id: "echo".to_string(),
            prompt_token_ids: Some(prompt.clone()),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 8,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        };
        add(&mut engine, &mut rx, req);
        let outputs = drain(&mut engine, &mut rx);
        let all_tokens: Vec<u32> = outputs.into_iter().flat_map(|o| o.new_token_ids).collect();
        assert_eq!(
            all_tokens, prompt,
            "engine with EchoTokens echoes prompt ids exactly"
        );
    }

    #[test]
    fn shortest_prompt_first_completes_shortest_first() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        let (mut engine, mut rx) = test_engine(opt);
        engine.scheduler = Box::new(crate::sched::ShortestPromptFirst);

        // Submit three requests with decreasing prompt lengths. With max_num_seqs=1 and
        // FCFS the first submitted would run first, but ShortestPromptFirst should pick
        // the shortest prompt from the waiting queue once the blocker finishes.
        let blocker = EngineCoreRequest {
            request_id: "blocker".to_string(),
            prompt_token_ids: Some(vec![0; 4]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 1,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        };
        add(&mut engine, &mut rx, blocker);

        // Queue three requests: 100, 50, 10 prompt tokens. All produce 1 output token.
        for (id, plen) in [("long", 100), ("mid", 50), ("short", 10)] {
            let req = EngineCoreRequest {
                request_id: id.to_string(),
                prompt_token_ids: Some(vec![0u32; plen]),
                sampling_params: Some(EngineCoreSamplingParams {
                    max_tokens: 1,
                    ..EngineCoreSamplingParams::for_test()
                }),
                ..Default::default()
            };
            add(&mut engine, &mut rx, req);
        }
        assert_eq!(engine.waiting.len(), 3);

        let finished_order: Vec<String> = drain(&mut engine, &mut rx)
            .into_iter()
            .filter(|o| o.finished())
            .map(|o| o.request_id)
            .collect();
        // blocker finishes first (already running), then shortest-first from queue.
        assert_eq!(finished_order, vec!["blocker", "short", "mid", "long"]);
    }

    #[test]
    fn priority_policy_admits_lowest_priority_value_first() {
        let mut opt = test_opt();
        opt.max_num_seqs = 1;
        opt.scheduling_policy = SchedulingPolicy::Priority;
        let (mut engine, mut rx) = test_engine(opt);

        let with_priority = |id: &str, p: i32| {
            let mut req = request(id, 4, 50);
            req.priority = p;
            req
        };

        submit(&mut engine, &mut rx, with_priority("blocker", 0)); // admitted immediately
        submit(&mut engine, &mut rx, with_priority("p10", 10));
        submit(&mut engine, &mut rx, with_priority("p1", 1));
        submit(&mut engine, &mut rx, with_priority("p5", 5));
        assert_eq!(engine.waiting.len(), 3);

        // Each freed slot admits the smallest remaining priority value, not arrival order.
        abort(&mut engine, &mut rx, &["blocker"]);
        assert!(engine.active_requests.contains_key("p1"));
        abort(&mut engine, &mut rx, &["p1"]);
        assert!(engine.active_requests.contains_key("p5"));
        abort(&mut engine, &mut rx, &["p5"]);
        assert!(engine.active_requests.contains_key("p10"));
    }
}
