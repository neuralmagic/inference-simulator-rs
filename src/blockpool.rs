//! The unified KV-cache block pool: prefix-cache accounting, physical block-slot
//! allocation, and KV-cache event generation, all from one place.
//!
//! This mirrors vLLM's `v1/core/block_pool.py`, which couples the same three concerns:
//! it tracks which content blocks are cached, hands out physical slot ids, measures how
//! much of an incoming prompt is already cached (the prefix hit), and emits the
//! `BlockStored` / `BlockRemoved` / `AllBlocksCleared` events that the llm-d cache-aware
//! router consumes over ZMQ.
//!
//! Why one struct for all three: the physical slot id IS the unit the data plane pages
//! over NIXL (`addr = pool_base + block_id * block_bytes`), the unit the router indexes
//! (`remote_block_ids`), and the unit eviction frees. Splitting them would mean keeping
//! three id spaces in sync; vLLM doesn't, and neither do we.
//!
//! ## Hashing vs the router
//!
//! The router (`llm-d-kv-cache`) does NOT trust our `block_hashes`: it re-hashes the
//! `token_ids` itself (FNV-64a over canonical CBOR) to build its prefix tree. So our
//! engine-side hash only has to be *stable per content* so that a later `BlockRemoved`
//! resolves back to the block a `BlockStored` introduced, and so a child block's
//! `parent_block_hash` equals the hash we already emitted for its parent. A chained FNV
//! gives us exactly that, cheaply, with no need to reproduce vLLM's (pickle-based,
//! per-process-random-seeded, deliberately non-reproducible) sha256 scheme.
//!
//! ```text
//!   prompt tokens ──chunk by block_size──▶ [blk0][blk1][blk2] (partial tail dropped)
//!         hash:  h0 = H(NONE, blk0)
//!                h1 = H(h0,   blk1)
//!                h2 = H(h1,   blk2)
//!   prefix hit = longest leading run already in `cached`; the rest are stored.
//! ```

use std::collections::HashMap;

/// A stable, opaque, engine-side block hash. The router re-derives its own keys from the
/// token ids, so this only needs to be a deterministic function of the block content (its
/// tokens and its parent hash) so eviction and parent chaining stay consistent.
pub type BlockHash = u64;

/// FNV-1a 64-bit offset basis, used to seed the no-parent ("NONE") hash.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hash a block's tokens chained onto its parent hash (or the NONE seed for the first block),
/// then its `extra_keys`, FNV-1a style. Deterministic across processes given the same
/// `none_seed`. The extra keys (the request's LoRA name, cache salt, etc.) are mixed after the
/// tokens exactly like vLLM's `hash_block_tokens(parent, tokens, extra_keys)`, and are
/// length-delimited so adjacent keys can't alias. Empty `extra_keys` reproduces the original
/// token-only hash, so unsalted base-model prompts keep their existing hashes.
fn hash_block(
    none_seed: u64,
    parent: Option<BlockHash>,
    tokens: &[u32],
    extra_keys: &[&str],
) -> BlockHash {
    let mut h = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    };
    mix(&parent.unwrap_or(none_seed).to_le_bytes());
    for &t in tokens {
        mix(&t.to_le_bytes());
    }
    for key in extra_keys {
        mix(&(key.len() as u64).to_le_bytes());
        mix(key.as_bytes());
    }
    h
}

/// A KV-cache event, wire-compatible (after msgpack encoding in [`crate::kvevents`]) with
/// vLLM's `BlockStored` / `BlockRemoved` / `AllBlocksCleared`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheEvent {
    /// New full blocks were inserted into the cache. `token_ids` is the flat concatenation
    /// of every newly stored block's tokens (the router re-chunks it by its own block
    /// size), and `parent_hash` is the hash of the last already-cached block, or `None`.
    /// `lora_name` is the adapter these blocks belong to (vLLM's `BlockStored.lora_name`),
    /// `None` for the base model; the router namespaces its prefix index by it.
    Stored {
        block_hashes: Vec<BlockHash>,
        parent_hash: Option<BlockHash>,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_name: Option<String>,
    },
    /// Blocks were evicted from the cache.
    Removed { block_hashes: Vec<BlockHash> },
    /// The whole prefix cache was reset (`reset_prefix_cache`).
    AllCleared,
}

/// One cached block's bookkeeping: its physical slot id, how many live requests hold it
/// (pinned, so it cannot be evicted), and an LRU timestamp.
#[derive(Debug)]
struct Slot {
    block_id: usize,
    refcnt: u32,
    last_used: u64,
}

/// The outcome of caching a request's prompt blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheOutcome {
    /// Tokens served from the local prefix cache (the prefix hit), `hit_blocks * block_size`.
    pub num_cached_tokens: usize,
    /// Physical slot ids for every full block of the prompt, in order (hits reuse existing
    /// slots, misses get freshly allocated ones). These are the `remote_block_ids` the data
    /// plane pages over NIXL.
    pub block_ids: Vec<usize>,
    /// Events to publish (one `Stored` for the newly cached run, plus any `Removed` from
    /// evictions made to fit it).
    pub events: Vec<KvCacheEvent>,
}

/// Prefix-cache + block-slot pool. Not thread-safe by design: it is owned by the single
/// engine task, exactly like vLLM's `BlockPool` lives inside the scheduler.
#[derive(Debug)]
pub struct BlockPool {
    block_size: usize,
    capacity: usize,
    none_seed: BlockHash,
    /// content hash -> slot. The authoritative "what is cached" map.
    cached: HashMap<BlockHash, Slot>,
    /// Physical slot ids not currently backing a cached block.
    free_slots: Vec<usize>,
    /// Monotonic LRU clock.
    tick: u64,
    /// Cumulative prefix-cache counters since the last snapshot, for `prefix_cache_stats`.
    queries: u64,
    hits: u64,
    requests: u64,
}

impl BlockPool {
    /// Build a pool of `capacity` blocks of `block_size` tokens each. `none_seed` chains the
    /// first block of every sequence; pin it to a fixed value (not random) so hashes are
    /// reproducible across restarts and across prefill/decode peers.
    pub fn new(block_size: usize, capacity: usize, none_seed: BlockHash) -> Self {
        let block_size = block_size.max(1);
        let capacity = capacity.max(1);
        // Hand out low ids first (pop from the back), so block ids read naturally in logs.
        let free_slots = (0..capacity).rev().collect();
        Self {
            block_size,
            capacity,
            none_seed,
            cached: HashMap::new(),
            free_slots,
            tick: 0,
            queries: 0,
            hits: 0,
            requests: 0,
        }
    }

    /// Number of blocks currently cached (occupied slots), including unreferenced ones kept
    /// around for prefix hits until evicted.
    pub fn used_blocks(&self) -> usize {
        self.cached.len()
    }

    /// Number of blocks currently referenced (pinned) by a live request. These are the
    /// blocks vLLM counts as "in use"; unreferenced cached blocks sit in the evictable pool.
    pub fn referenced_blocks(&self) -> usize {
        self.cached.values().filter(|slot| slot.refcnt > 0).count()
    }

    /// `kv_cache_usage` in `[0, 1]`: referenced blocks over capacity, matching vLLM, where
    /// cached-but-unreferenced blocks count as free (they can be evicted on demand).
    pub fn usage(&self) -> f64 {
        self.referenced_blocks() as f64 / self.capacity as f64
    }

    /// Allocate a free slot, evicting the least-recently-used *unpinned* cached block if the
    /// pool is full. Returns the slot id and an optional `Removed` event for the eviction, or
    /// `None` if every slot is pinned (the pool is over-subscribed by live requests).
    fn allocate_slot(&mut self) -> Option<(usize, Option<KvCacheEvent>)> {
        if let Some(id) = self.free_slots.pop() {
            return Some((id, None));
        }
        // Full: evict the LRU unpinned block.
        let victim = self
            .cached
            .iter()
            .filter(|(_, slot)| slot.refcnt == 0)
            .min_by_key(|(_, slot)| slot.last_used)
            .map(|(&hash, _)| hash)?;
        let slot = self.cached.remove(&victim).expect("victim was just found");
        Some((
            slot.block_id,
            Some(KvCacheEvent::Removed {
                block_hashes: vec![victim],
            }),
        ))
    }

    /// Cache a request's prompt and report the prefix hit. Chunks `tokens` into full blocks
    /// (a partial tail is dropped, matching vLLM, which only hashes full blocks), measures
    /// the longest leading run already cached, allocates+stores the remainder, and pins
    /// every block this request uses (call [`BlockPool::unpin`] when the request ends).
    ///
    /// `lora_name` and `cache_salt` (from the request) isolate this prompt's prefix: identical
    /// tokens under a different adapter or salt cache to disjoint blocks. Matching vLLM, the
    /// LoRA name is mixed into every block's hash while the cache salt only affects the first
    /// block (it isolates the whole chain anyway, through parent chaining).
    pub fn cache_prompt(
        &mut self,
        tokens: &[u32],
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> CacheOutcome {
        let cache_salt = cache_salt.filter(|s| !s.is_empty());
        let n_blocks = tokens.len() / self.block_size;
        self.requests += 1;
        self.queries += n_blocks as u64;

        let mut block_ids = Vec::with_capacity(n_blocks);
        let mut events = Vec::new();
        let mut parent: Option<BlockHash> = None;
        let mut in_prefix = true;
        let mut num_cached_blocks = 0usize;

        // The contiguous run of newly stored blocks, accumulated into one BlockStored.
        let mut new_hashes: Vec<BlockHash> = Vec::new();
        let mut new_tokens: Vec<u32> = Vec::new();
        let mut stored_parent: Option<BlockHash> = None;

        for i in 0..n_blocks {
            let block_toks = &tokens[i * self.block_size..(i + 1) * self.block_size];
            // `parent` holds the previous block's hash (None for the first block); capture it
            // before we overwrite it, so the first miss can record it as the stored run's parent.
            let parent_before = parent;
            // vLLM's extra-key order: LoRA name on every block, cache salt on the first only.
            let mut extra: [&str; 2] = ["", ""];
            let mut n_extra = 0;
            if let Some(name) = lora_name {
                extra[n_extra] = name;
                n_extra += 1;
            }
            if i == 0
                && let Some(salt) = cache_salt
            {
                extra[n_extra] = salt;
                n_extra += 1;
            }
            let hash = hash_block(self.none_seed, parent, block_toks, &extra[..n_extra]);
            parent = Some(hash);

            if let Some(slot) = self.cached.get_mut(&hash) {
                // Cache hit. A hit only counts toward the prefix while the run is unbroken.
                slot.refcnt += 1;
                slot.last_used = {
                    self.tick += 1;
                    self.tick
                };
                block_ids.push(slot.block_id);
                if in_prefix {
                    num_cached_blocks += 1;
                }
            } else {
                // First miss breaks the prefix run; everything after is freshly stored. The
                // stored run's parent is the last cached block's hash (or None at i == 0).
                if in_prefix {
                    in_prefix = false;
                    stored_parent = parent_before;
                }
                let Some((block_id, removed)) = self.allocate_slot() else {
                    // Pool fully pinned: cannot store this block. Stop; the caller still gets
                    // the prefix it found and the slots allocated so far.
                    break;
                };
                if let Some(removed) = removed {
                    events.push(removed);
                }
                self.cached.insert(
                    hash,
                    Slot {
                        block_id,
                        refcnt: 1,
                        last_used: {
                            self.tick += 1;
                            self.tick
                        },
                    },
                );
                block_ids.push(block_id);
                new_hashes.push(hash);
                new_tokens.extend_from_slice(block_toks);
            }
        }

        self.hits += num_cached_blocks as u64;

        if !new_hashes.is_empty() {
            events.push(KvCacheEvent::Stored {
                block_hashes: new_hashes,
                parent_hash: stored_parent,
                token_ids: new_tokens,
                block_size: self.block_size,
                lora_name: lora_name.map(str::to_string),
            });
        }

        CacheOutcome {
            num_cached_tokens: num_cached_blocks * self.block_size,
            block_ids,
            events,
        }
    }

    /// Release the pins a request held on its blocks, letting them be evicted later. Pass the
    /// `block_ids` returned by [`BlockPool::cache_prompt`].
    pub fn unpin(&mut self, block_ids: &[usize]) {
        let pinned: std::collections::HashSet<usize> = block_ids.iter().copied().collect();
        for slot in self.cached.values_mut() {
            if pinned.contains(&slot.block_id) && slot.refcnt > 0 {
                slot.refcnt -= 1;
            }
        }
    }

    /// Reset the whole prefix cache (the `reset_prefix_cache` utility). Returns an
    /// `AllBlocksCleared` event if anything was cached.
    pub fn reset(&mut self) -> Option<KvCacheEvent> {
        if self.cached.is_empty() {
            return None;
        }
        self.cached.clear();
        self.free_slots = (0..self.capacity).rev().collect();
        Some(KvCacheEvent::AllCleared)
    }

    /// Snapshot and clear the prefix-cache counters for one `prefix_cache_stats` report.
    pub fn take_stats(&mut self) -> PrefixStatsSnapshot {
        let snap = PrefixStatsSnapshot {
            requests: self.requests,
            queries: self.queries,
            hits: self.hits,
        };
        self.requests = 0;
        self.queries = 0;
        self.hits = 0;
        snap
    }
}

/// A drained snapshot of prefix-cache counters, mapped into the protocol's
/// `PrefixCacheStats` by the engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixStatsSnapshot {
    pub requests: u64,
    pub queries: u64,
    pub hits: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 12345;

    fn pool(block_size: usize, capacity: usize) -> BlockPool {
        BlockPool::new(block_size, capacity, SEED)
    }

    /// Tokens 0..n, so prompts share real prefixes when they share leading tokens.
    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn first_prompt_is_all_misses_and_emits_one_stored() {
        let mut p = pool(4, 16);
        let out = p.cache_prompt(&toks(12), None, None); // 3 full blocks
        assert_eq!(out.num_cached_tokens, 0, "cold cache: no prefix hit");
        assert_eq!(out.block_ids, vec![0, 1, 2]);
        assert_eq!(out.events.len(), 1);
        match &out.events[0] {
            KvCacheEvent::Stored {
                block_hashes,
                parent_hash,
                token_ids,
                block_size,
                lora_name,
            } => {
                assert_eq!(block_hashes.len(), 3);
                assert_eq!(*parent_hash, None, "first block has no parent");
                assert_eq!(token_ids.len(), 12);
                assert_eq!(*block_size, 4);
                assert_eq!(*lora_name, None, "base-model prompt carries no adapter");
            }
            other => panic!("expected Stored, got {other:?}"),
        }
    }

    #[test]
    fn partial_tail_block_is_dropped() {
        let mut p = pool(4, 16);
        let out = p.cache_prompt(&toks(10), None, None); // 2 full blocks + 2 leftover tokens
        assert_eq!(out.block_ids.len(), 2);
        match &out.events[0] {
            KvCacheEvent::Stored { token_ids, .. } => assert_eq!(token_ids.len(), 8),
            other => panic!("expected Stored, got {other:?}"),
        }
    }

    #[test]
    fn shared_prefix_is_a_hit_and_only_new_blocks_are_stored() {
        let mut p = pool(4, 16);
        let first = p.cache_prompt(&toks(8), None, None); // blocks [0,1]
        p.unpin(&first.block_ids);

        // Same 8-token prefix, then 4 new tokens -> block 2 is new, blocks 0,1 are hits.
        let mut longer = toks(8);
        longer.extend([100, 101, 102, 103]);
        let out = p.cache_prompt(&longer, None, None);

        assert_eq!(out.num_cached_tokens, 8, "two 4-token blocks hit");
        assert_eq!(
            out.block_ids,
            vec![0, 1, 2],
            "hits reuse slots, miss gets a new one"
        );
        let stored: Vec<_> = out
            .events
            .iter()
            .filter(|e| matches!(e, KvCacheEvent::Stored { .. }))
            .collect();
        assert_eq!(stored.len(), 1);
        match stored[0] {
            KvCacheEvent::Stored {
                block_hashes,
                parent_hash,
                token_ids,
                ..
            } => {
                assert_eq!(block_hashes.len(), 1, "only the one new block is stored");
                assert!(
                    parent_hash.is_some(),
                    "new block chains onto the cached prefix"
                );
                assert_eq!(token_ids, &vec![100, 101, 102, 103]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn parent_hash_of_new_run_matches_emitted_hash_of_prefix_block() {
        // The router resolves parent_hash against hashes from earlier BlockStored events,
        // so a child's parent_hash must equal the parent block's own emitted block_hash.
        let mut p = pool(4, 16);
        let first = p.cache_prompt(&toks(8), None, None);
        let first_hashes = match &first.events[0] {
            KvCacheEvent::Stored { block_hashes, .. } => block_hashes.clone(),
            _ => unreachable!(),
        };
        p.unpin(&first.block_ids);

        let mut longer = toks(8);
        longer.extend([100, 101, 102, 103]);
        let out = p.cache_prompt(&longer, None, None);
        let stored_parent = out.events.iter().find_map(|e| match e {
            KvCacheEvent::Stored { parent_hash, .. } => *parent_hash,
            _ => None,
        });
        assert_eq!(
            stored_parent,
            Some(first_hashes[1]),
            "new run's parent is the second (last cached) prefix block's hash"
        );
    }

    #[test]
    fn eviction_of_unpinned_block_emits_removed() {
        let mut p = pool(4, 2); // room for 2 blocks only
        let a = p.cache_prompt(&toks(8), None, None); // fills both slots with blocks [0,1]
        p.unpin(&a.block_ids); // unpinned -> evictable

        // A fresh 1-block prompt must evict an LRU block to fit.
        let out = p.cache_prompt(&[900, 901, 902, 903], None, None);
        let removed: Vec<_> = out
            .events
            .iter()
            .filter(|e| matches!(e, KvCacheEvent::Removed { .. }))
            .collect();
        assert_eq!(removed.len(), 1, "one eviction to make room");
    }

    #[test]
    fn pinned_blocks_are_not_evicted() {
        let mut p = pool(4, 2);
        let _a = p.cache_prompt(&toks(8), None, None); // pinned, both slots
        // Do NOT unpin. A new block cannot be stored (both slots pinned).
        let out = p.cache_prompt(&[900, 901, 902, 903], None, None);
        assert!(
            out.block_ids.is_empty(),
            "no slot available, nothing stored or paged"
        );
        assert!(
            !out.events
                .iter()
                .any(|e| matches!(e, KvCacheEvent::Stored { .. })),
            "must not store into a pinned-full pool"
        );
    }

    #[test]
    fn reset_clears_and_emits_all_cleared_once() {
        let mut p = pool(4, 8);
        let a = p.cache_prompt(&toks(8), None, None);
        p.unpin(&a.block_ids);
        assert_eq!(p.used_blocks(), 2);

        assert_eq!(p.reset(), Some(KvCacheEvent::AllCleared));
        assert_eq!(p.used_blocks(), 0);
        assert_eq!(p.reset(), None, "second reset on an empty cache is a no-op");

        // Slots are reusable after reset.
        let out = p.cache_prompt(&toks(4), None, None);
        assert_eq!(out.block_ids, vec![0]);
    }

    #[test]
    fn stats_accumulate_then_clear_on_take() {
        let mut p = pool(4, 16);
        let a = p.cache_prompt(&toks(8), None, None); // 2 queries, 0 hits
        p.unpin(&a.block_ids);
        let _b = p.cache_prompt(&toks(8), None, None); // 2 queries, 2 hits

        let snap = p.take_stats();
        assert_eq!(snap.requests, 2);
        assert_eq!(snap.queries, 4);
        assert_eq!(snap.hits, 2);

        let empty = p.take_stats();
        assert_eq!(
            empty,
            PrefixStatsSnapshot::default(),
            "counters reset after take"
        );
    }

    #[test]
    fn usage_counts_referenced_not_cached_blocks() {
        let mut p = pool(4, 10);
        let a = p.cache_prompt(&toks(8), None, None); // 2 blocks, pinned by this request
        assert_eq!(p.referenced_blocks(), 2);
        assert!((p.usage() - 0.2).abs() < 1e-9, "2 pinned / 10 capacity");

        p.unpin(&a.block_ids);
        assert_eq!(p.used_blocks(), 2, "blocks stay cached for prefix hits");
        assert_eq!(p.referenced_blocks(), 0, "but are no longer referenced");
        assert_eq!(p.usage(), 0.0, "usage drops once nothing references them");
    }

    #[test]
    fn different_cache_salts_do_not_share_a_prefix() {
        let mut p = pool(4, 16);
        // Same prompt under salt "a" then salt "b": the second must NOT hit the first's blocks.
        let first = p.cache_prompt(&toks(8), None, Some("a"));
        p.unpin(&first.block_ids);
        let second = p.cache_prompt(&toks(8), None, Some("b"));
        assert_eq!(
            second.num_cached_tokens, 0,
            "different salt -> no prefix reuse"
        );
        assert_ne!(
            first.block_ids, second.block_ids,
            "salted prompts land in disjoint slots"
        );
    }

    #[test]
    fn same_cache_salt_shares_a_prefix() {
        let mut p = pool(4, 16);
        let first = p.cache_prompt(&toks(8), None, Some("tenant-x"));
        p.unpin(&first.block_ids);
        let second = p.cache_prompt(&toks(8), None, Some("tenant-x"));
        assert_eq!(
            second.num_cached_tokens, 8,
            "same salt + tokens -> full hit"
        );
        assert_eq!(first.block_ids, second.block_ids, "reuses the same slots");
    }

    #[test]
    fn salted_prompt_does_not_hit_unsalted_one() {
        let mut p = pool(4, 16);
        let plain = p.cache_prompt(&toks(8), None, None);
        p.unpin(&plain.block_ids);
        let salted = p.cache_prompt(&toks(8), None, Some("s"));
        assert_eq!(
            salted.num_cached_tokens, 0,
            "a salt isolates from the unsalted cache too"
        );
    }

    #[test]
    fn empty_salt_is_treated_as_no_salt() {
        let mut p = pool(4, 16);
        let plain = p.cache_prompt(&toks(8), None, None);
        p.unpin(&plain.block_ids);
        // An empty-string salt must not fragment the cache away from the unsalted entry.
        let empty = p.cache_prompt(&toks(8), None, Some(""));
        assert_eq!(
            empty.num_cached_tokens, 8,
            "empty salt == no salt -> full hit"
        );
    }

    #[test]
    fn different_loras_do_not_share_a_prefix() {
        let mut p = pool(4, 16);
        // Same prompt under adapter "a" then "b": no cross-adapter reuse (vLLM mixes the
        // adapter name into every block hash).
        let a = p.cache_prompt(&toks(8), Some("a"), None);
        p.unpin(&a.block_ids);
        let b = p.cache_prompt(&toks(8), Some("b"), None);
        assert_eq!(
            b.num_cached_tokens, 0,
            "different adapter -> no prefix reuse"
        );
        assert_ne!(a.block_ids, b.block_ids, "adapters land in disjoint slots");
    }

    #[test]
    fn same_lora_shares_a_prefix() {
        let mut p = pool(4, 16);
        let first = p.cache_prompt(&toks(8), Some("a"), None);
        p.unpin(&first.block_ids);
        let second = p.cache_prompt(&toks(8), Some("a"), None);
        assert_eq!(
            second.num_cached_tokens, 8,
            "same adapter + tokens -> full hit"
        );
    }

    #[test]
    fn lora_prompt_does_not_hit_base_model_prefix() {
        let mut p = pool(4, 16);
        // The base model warms the cache; an adapter with the same tokens must not reuse it,
        // and vice versa. This is the fragmentation that costs cross-adapter cache hits.
        let base = p.cache_prompt(&toks(8), None, None);
        p.unpin(&base.block_ids);
        let lora = p.cache_prompt(&toks(8), Some("a"), None);
        assert_eq!(
            lora.num_cached_tokens, 0,
            "adapter cannot reuse base-model blocks"
        );
    }

    #[test]
    fn stored_event_carries_lora_name() {
        let mut p = pool(4, 16);
        let out = p.cache_prompt(&toks(8), Some("adapterA"), None);
        match &out.events[0] {
            KvCacheEvent::Stored { lora_name, .. } => {
                assert_eq!(lora_name.as_deref(), Some("adapterA"));
            }
            other => panic!("expected Stored, got {other:?}"),
        }
    }

    #[test]
    fn lora_and_salt_compose() {
        let mut p = pool(4, 16);
        // Same adapter, different salts -> still disjoint (both keys mix in).
        let x = p.cache_prompt(&toks(8), Some("a"), Some("t1"));
        p.unpin(&x.block_ids);
        let y = p.cache_prompt(&toks(8), Some("a"), Some("t2"));
        assert_eq!(
            y.num_cached_tokens, 0,
            "salt still isolates within one adapter"
        );
        p.unpin(&y.block_ids);
        // Same adapter + same salt -> full hit.
        let z = p.cache_prompt(&toks(8), Some("a"), Some("t1"));
        assert_eq!(z.num_cached_tokens, 8, "same adapter + same salt -> hit");
    }

    #[test]
    fn hashes_are_deterministic_across_pools() {
        let mut p1 = pool(4, 16);
        let mut p2 = pool(4, 16);
        let h1 = match &p1.cache_prompt(&toks(8), None, None).events[0] {
            KvCacheEvent::Stored { block_hashes, .. } => block_hashes.clone(),
            _ => unreachable!(),
        };
        let h2 = match &p2.cache_prompt(&toks(8), None, None).events[0] {
            KvCacheEvent::Stored { block_hashes, .. } => block_hashes.clone(),
            _ => unreachable!(),
        };
        assert_eq!(
            h1, h2,
            "same seed + tokens -> same hashes (stable for the router)"
        );
    }
}
