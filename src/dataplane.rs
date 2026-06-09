//! The KV-block data plane: the connector boundary where prefill/decode moves bytes.
//!
//! In real vLLM the produce/consume of `kv_transfer_params` and the NIXL transfer live
//! in the NixlConnector inside the engine/worker. Our mock engine *is* the engine, so
//! this module plays that connector role, wire-compatibly with the llm-d routing
//! sidecar and a real vLLM peer:
//!
//!   - PREFILL registers one KV pool with NIXL and serves a [`PoolDescriptor`] (its NIXL
//!     agent metadata + pool base address) over a small TCP metadata side channel. It
//!     advertises how to reach it as [`RemoteKv`] (`remote_engine_id`/`remote_host`/
//!     `remote_port`/`remote_block_ids`/`remote_request_id`), which the engine wraps into the
//!     real `kv_transfer_params` dict the sidecar relays. No mock-specific fields ride there.
//!   - DECODE connects to that side channel (host:port), loads the peer's agent metadata via
//!     `load_remote_md`, then posts a paged NIXL READ (one descriptor per block at
//!     `pool_base + block_id*block_bytes`) and verifies the per-request pattern.
//!
//! ```text
//!   prefill engine                                    decode engine
//!   ┌──────────────────────┐  kv_transfer_params      ┌──────────────────────┐
//!   │ register KV pool      │  {remote_host,port,      │ connect side channel  │
//!   │ TCP side channel :port│   engine_id,block_ids,  ─┼▶ load_remote_md,      │
//!   │  -> PoolDescriptor ───┼─  request_id}            │  paged NIXL READ,     │
//!   │  {agent_md,pool_base} ◀┼─────────────────────────┼─ verify(pattern)      │
//!   └──────────────────────┘     (descriptor fetch)    └──────────────────────┘
//! ```
//!
//! The pool base address travels over the side channel (vLLM ships it as
//! `kv_caches_base_addr` in its own `NixlAgentMetadata`), not in `kv_transfer_params`, and the
//! verify pattern is derived from `request_id` on both sides, so the params dict stays
//! byte-faithful to what a real vLLM engine produces.
//!
//! The default build ships [`NoopDataPlane`] (control plane only: produces/consumes the
//! real dict but moves no bytes), so the sidecar contract is exercisable with no NIXL.
//! The real transfer lives behind the `nixl` feature.

/// The role this engine plays in a disaggregated prefill/decode deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PdRole {
    /// Registers KV and advertises it for a remote puller.
    Prefill,
    /// Pulls remote KV before generating tokens.
    Decode,
    /// Monolithic: no handoff, behaves like a normal single engine.
    Both,
}

/// Sizing + identity knobs for the data plane.
#[derive(Debug, Clone)]
pub struct NixlConfig {
    /// Bytes per KV block (one paged slot in the registered pool).
    pub kv_block_bytes: usize,
    /// Prompt tokens that map to one KV block.
    pub tokens_per_block: usize,
    /// Total KV-cache capacity in blocks. The prefill registers one contiguous pool of
    /// `kv_cache_blocks * kv_block_bytes`, so block `i` lives at `pool_base + i*kv_block_bytes`.
    pub kv_cache_blocks: usize,
    /// This engine's id, advertised as `remote_engine_id` (a real decode peer matches
    /// it against the NIXL agent metadata's engine id).
    pub engine_id: String,
    /// Host a decode peer connects to for the NIXL metadata side channel.
    pub side_channel_host: String,
    /// Port the NIXL metadata side channel listens on.
    pub side_channel_port: u32,
}

/// How a decode peer reaches a prefilled request's KV. Every field here is a wire-faithful
/// `remote_*` field of vLLM's `kv_transfer_params` (no mock extensions): the decode learns
/// the pool base address out of band over the metadata side channel ([`PoolDescriptor`]), and
/// the verify pattern is derived from `request_id` on both sides, so nothing mock-specific
/// rides in the params dict.
#[derive(Debug, Clone)]
pub struct RemoteKv {
    pub engine_id: String,
    pub host: String,
    pub port: u32,
    /// Physical slot ids of this request's blocks in the prefill's KV pool. The decode reads
    /// each one at `pool_base + block_id * block_bytes` (pool base from the side channel).
    pub block_ids: Vec<i64>,
    /// The prefill's request id (`remote_request_id`). Both sides derive the per-request
    /// verify pattern from it, so it never has to be transmitted separately.
    pub request_id: String,
}

/// The prefill's KV-pool metadata, served over the side channel and consumed by a decode peer
/// to address the pool. This is the mock's minimal stand-in for vLLM's `NixlAgentMetadata`:
/// it carries the NIXL agent metadata bytes (`get_local_md`, loaded via `load_remote_md`) plus
/// the pool base address vLLM ships as `kv_caches_base_addr`. Unlike vLLM's versioned v4
/// schema (compatibility hashes, heartbeats), it is deliberately tiny and unversioned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolDescriptor {
    pub engine_id: String,
    /// Serialized NIXL agent metadata (`Agent::get_local_md`), loaded by the peer.
    pub nixl_agent_md: Vec<u8>,
    /// Base address of the prefill's single registered KV pool.
    pub pool_base: u64,
    /// Bytes per block (paged slot stride) in that pool.
    pub block_bytes: u64,
    /// Total blocks in the pool.
    pub num_blocks: u64,
}

impl PoolDescriptor {
    /// Encode to self-describing msgpack (array form), the side-channel wire format.
    pub fn encode(&self) -> anyhow::Result<Vec<u8>> {
        use rmpv::Value;
        let value = Value::Array(vec![
            Value::from(self.engine_id.as_str()),
            Value::Binary(self.nixl_agent_md.clone()),
            Value::from(self.pool_base),
            Value::from(self.block_bytes),
            Value::from(self.num_blocks),
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &value)?;
        Ok(buf)
    }

    /// Decode from the msgpack array produced by [`PoolDescriptor::encode`].
    pub fn decode(bytes: &[u8]) -> anyhow::Result<Self> {
        use anyhow::{Context as _, bail};
        use rmpv::Value;
        let mut cursor = std::io::Cursor::new(bytes);
        let value = rmpv::decode::read_value(&mut cursor).context("decode pool descriptor")?;
        let Value::Array(fields) = value else {
            bail!("pool descriptor is not an array");
        };
        if fields.len() < 5 {
            bail!("pool descriptor has {} fields, need 5", fields.len());
        }
        let engine_id = fields[0]
            .as_str()
            .context("pool descriptor engine_id")?
            .to_string();
        let nixl_agent_md = match &fields[1] {
            Value::Binary(b) => b.clone(),
            other => bail!("pool descriptor nixl_agent_md is not binary: {other:?}"),
        };
        let pool_base = fields[2].as_u64().context("pool descriptor pool_base")?;
        let block_bytes = fields[3].as_u64().context("pool descriptor block_bytes")?;
        let num_blocks = fields[4].as_u64().context("pool descriptor num_blocks")?;
        Ok(Self {
            engine_id,
            nixl_agent_md,
            pool_base,
            block_bytes,
            num_blocks,
        })
    }
}

/// Stable per-request fill byte, so a decode pull can verify it got the right bytes. Derived
/// identically on both sides from the (prefill) request id, so it never has to be transmitted.
pub fn pattern_for(request_id: &str) -> u8 {
    request_id
        .bytes()
        .fold(0xa5u8, |acc, b| acc.wrapping_add(b))
        | 1
}

/// The minimal view of a request the data plane needs, decoupled from the wire
/// `EngineCoreRequest` so the protocol crate can evolve without touching this trait.
#[derive(Debug, Clone, Copy)]
pub struct RequestKv<'a> {
    pub request_id: &'a str,
    /// Number of prompt tokens; informational (sizing comes from `block_ids`).
    pub num_tokens: usize,
    /// Physical KV-pool slot ids this request occupies, assigned by the block pool. These are
    /// the paged blocks the prefill advertises and the decode reads/lands.
    pub block_ids: &'a [usize],
}

/// The connector boundary: where simulated KV cache bytes move between engines.
pub trait KvDataPlane: Send {
    /// Prefill side: register/stage this request's KV and return how a decode peer
    /// reaches it (becomes the `remote_*` fields of `kv_transfer_params`).
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> anyhow::Result<RemoteKv>;

    /// Decode side: pull the remote KV described by `remote` before generation.
    /// Returns the number of bytes moved.
    fn pull_prefilled(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> anyhow::Result<u64>;

    /// Release any resources staged for a request (prefill dropping its KV buffer).
    fn release(&mut self, _request_id: &str) {}
}

/// Control-plane-only data plane: produces/consumes the real `kv_transfer_params`
/// addressing but moves zero bytes. This is what the default (no-NIXL) binary uses, so
/// the routing-sidecar contract is fully exercisable without libnixl/UCX.
pub struct NoopDataPlane {
    cfg: NixlConfig,
}

impl NoopDataPlane {
    pub fn new(cfg: NixlConfig) -> Self {
        Self { cfg }
    }
}

impl KvDataPlane for NoopDataPlane {
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> anyhow::Result<RemoteKv> {
        Ok(RemoteKv {
            engine_id: self.cfg.engine_id.clone(),
            host: self.cfg.side_channel_host.clone(),
            port: self.cfg.side_channel_port,
            block_ids: kv.block_ids.iter().map(|&id| id as i64).collect(),
            request_id: kv.request_id.to_string(),
        })
    }

    fn pull_prefilled(&mut self, _kv: RequestKv<'_>, _remote: &RemoteKv) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// Build the data plane for a given role. Without the `nixl` feature there is only the
/// no-op plane. With `nixl`, prefill/decode roles get a real NIXL-backed plane; if NIXL
/// init fails (no libnixl/UCX) we degrade to the no-op plane rather than crash, so the
/// same binary still runs as a pure protocol emulator.
pub fn make_data_plane(role: PdRole, cfg: NixlConfig) -> Box<dyn KvDataPlane> {
    #[cfg(feature = "nixl")]
    {
        if !matches!(role, PdRole::Both) {
            match nixl::NixlDataPlane::new(role, cfg.clone()) {
                Ok(plane) => return Box::new(plane),
                Err(error) => {
                    tracing::warn!(%error, "NIXL init failed; using no-op data plane");
                }
            }
        }
    }
    let _ = role;
    Box::new(NoopDataPlane::new(cfg))
}

/// Whether the NIXL bindings are the no-op stubs (true when built without real libnixl,
/// e.g. via `--features nixl-stub`, or when the `nixl` feature is off). Tests use this
/// to skip the real transfer on machines that cannot move bytes.
pub fn nixl_is_stub() -> bool {
    #[cfg(feature = "nixl")]
    {
        nixl_sys::is_stub()
    }
    #[cfg(not(feature = "nixl"))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_descriptor_round_trips() {
        let descriptor = PoolDescriptor {
            engine_id: "mock-prefill-0".to_string(),
            nixl_agent_md: vec![0xde, 0xad, 0xbe, 0xef, 0x00, 0x42],
            pool_base: 0x7f00_1234_5000,
            block_bytes: 4096,
            num_blocks: 16,
        };
        let bytes = descriptor.encode().expect("encode");
        let decoded = PoolDescriptor::decode(&bytes).expect("decode");
        assert_eq!(decoded, descriptor);
    }

    #[test]
    fn pool_descriptor_decode_rejects_truncated() {
        assert!(
            PoolDescriptor::decode(&[0x90]).is_err(),
            "empty array -> too few fields"
        );
        assert!(
            PoolDescriptor::decode(&[0xff, 0xff]).is_err(),
            "garbage -> error"
        );
    }

    #[test]
    fn pattern_is_stable_nonzero_and_request_specific() {
        // Both sides derive the verify pattern from the request id, so it must be a pure,
        // stable function of the id, and never zero (zero would alias an unwritten slot).
        assert_eq!(pattern_for("req-abc"), pattern_for("req-abc"));
        assert_ne!(pattern_for("req-abc"), 0);
        assert_ne!(pattern_for("req-abc"), pattern_for("req-xyz"));
    }
}

/// The real NIXL-backed KV data plane, paged over a single registered KV pool.
#[cfg(feature = "nixl")]
mod nixl {
    use std::time::{Duration, Instant};

    use anyhow::{Result, anyhow, bail};
    use nixl_sys::{
        Agent, AgentConfig, Backend, MemType, MemoryRegion as _, NixlError, NixlRegistration as _,
        OptArgs, SystemStorage, XferDescList, XferOp, is_stub,
    };
    use tracing::debug;

    use super::{KvDataPlane, NixlConfig, PdRole, RemoteKv, RequestKv};

    /// How long to poll a NIXL transfer before giving up.
    const XFER_TIMEOUT: Duration = Duration::from_secs(5);

    /// Map a `NixlError` into an `anyhow::Error` (it is not `std::error::Error`).
    fn ne(error: NixlError) -> anyhow::Error {
        anyhow!("nixl: {error:?}")
    }

    use std::collections::HashMap;
    use std::io::{Read as _, Write as _};
    use std::net::{TcpListener, TcpStream};

    use super::{PoolDescriptor, pattern_for};

    /// A remote prefill peer the decode has loaded, cached by engine id so we only fetch its
    /// metadata + `load_remote_md` once.
    struct LoadedPeer {
        /// NIXL agent name returned by `load_remote_md` (used to address transfers).
        agent_name: String,
        pool_base: u64,
        block_bytes: u64,
    }

    pub struct NixlDataPlane {
        role: PdRole,
        cfg: NixlConfig,
        /// NIXL agent name == engine id, so a peer's `remote_engine_id` is the agent name.
        agent: Agent,
        backend: Backend,
        /// The single registered KV pool: `kv_cache_blocks * kv_block_bytes` bytes. Block `i`
        /// lives at `base + i * kv_block_bytes`. Both roles register one (the prefill serves
        /// reads from it; the decode lands pulled blocks into it). Its base address is captured
        /// into the served `PoolDescriptor` at init; peers learn it over the side channel.
        pool: SystemStorage,
        /// Decode side: remote prefill peers loaded so far, keyed by engine id.
        peers: HashMap<String, LoadedPeer>,
    }

    impl NixlDataPlane {
        pub fn new(role: PdRole, cfg: NixlConfig) -> Result<Self> {
            // We exchange agent metadata ourselves over the side channel ([`PoolDescriptor`]
            // via `get_local_md`/`load_remote_md`), so NIXL's own listener thread is off; we
            // keep the prog thread, which advances UCX for both initiated and serviced reads.
            let acfg = AgentConfig {
                enable_prog_thread: true,
                enable_listen_thread: false,
                listen_port: cfg.side_channel_port as i32,
                ..AgentConfig::default()
            };
            let agent = Agent::new_configured(&cfg.engine_id, &acfg).map_err(ne)?;
            let (_, params) = agent.get_plugin_params("UCX").map_err(ne)?;
            let backend = agent.create_backend("UCX", &params).map_err(ne)?;

            // Register the whole KV pool up front, exactly once. This is the paged store the
            // block-pool slot ids index into.
            let pool_bytes = cfg.kv_cache_blocks.max(1) * cfg.kv_block_bytes;
            let mut pool = SystemStorage::new(pool_bytes).map_err(ne)?;
            if !is_stub() {
                let mut opt = OptArgs::new().map_err(ne)?;
                opt.add_backend(&backend).map_err(ne)?;
                pool.register(&agent, Some(&opt)).map_err(ne)?;
                // SAFETY: the pool's heap buffer address is stable for the agent's lifetime.
                let pool_base = unsafe { pool.as_ptr() as u64 };

                // The prefill is the metadata producer: serve our PoolDescriptor (agent md +
                // pool base) over the side channel so decode peers can address + connect to us.
                if matches!(role, PdRole::Prefill) {
                    let descriptor = PoolDescriptor {
                        engine_id: cfg.engine_id.clone(),
                        nixl_agent_md: agent.get_local_md().map_err(ne)?,
                        pool_base,
                        block_bytes: cfg.kv_block_bytes as u64,
                        num_blocks: cfg.kv_cache_blocks as u64,
                    };
                    serve_descriptor(cfg.side_channel_port, descriptor)?;
                }
            }

            debug!(
                engine_id = cfg.engine_id,
                ?role,
                side_channel_port = cfg.side_channel_port,
                pool_blocks = cfg.kv_cache_blocks,
                pool_bytes,
                "NIXL data plane ready (UCX backend, paged pool, side-channel metadata)"
            );
            Ok(Self {
                role,
                cfg,
                agent,
                backend,
                pool,
                peers: HashMap::new(),
            })
        }

        /// Fresh opt args carrying our backend (required for register and transfer calls).
        fn opt_args(&self) -> Result<OptArgs> {
            let mut opt = OptArgs::new().map_err(ne)?;
            opt.add_backend(&self.backend).map_err(ne)?;
            Ok(opt)
        }

        /// Fill a block slot in our pool with `pattern`, so a decode pull can verify the
        /// right bytes moved. `SystemStorage` only exposes a whole-buffer `memset`, so we
        /// write the sub-range through the pool's raw pointer.
        fn fill_slot(&mut self, block_id: usize, pattern: u8) {
            let offset = block_id * self.cfg.kv_block_bytes;
            let end = offset + self.cfg.kv_block_bytes;
            // SAFETY: `pool` owns a `Vec<u8>` of `kv_cache_blocks * kv_block_bytes`; callers
            // only pass in-range slot ids, so `[offset, end)` is within that allocation, and
            // we hold `&mut self` so no other reference aliases it.
            unsafe {
                let base = self.pool.as_ptr() as *mut u8;
                std::slice::from_raw_parts_mut(base.add(offset), end - offset).fill(pattern);
            }
        }

        /// Load a remote prefill peer (fetch its PoolDescriptor over the side channel + register
        /// its agent metadata), caching it by engine id so repeated pulls reuse the connection.
        fn ensure_peer(&mut self, engine_id: &str, host: &str, port: u32) -> Result<&LoadedPeer> {
            if !self.peers.contains_key(engine_id) {
                let descriptor = fetch_descriptor(host, port)?;
                let agent_name = self
                    .agent
                    .load_remote_md(&descriptor.nixl_agent_md)
                    .map_err(ne)?;
                debug!(
                    engine_id,
                    agent_name,
                    pool_base = descriptor.pool_base,
                    "loaded remote prefill peer over side channel"
                );
                self.peers.insert(
                    engine_id.to_string(),
                    LoadedPeer {
                        agent_name,
                        pool_base: descriptor.pool_base,
                        block_bytes: descriptor.block_bytes,
                    },
                );
            }
            Ok(self.peers.get(engine_id).expect("peer just inserted"))
        }

        fn do_advertise(&mut self, kv: RequestKv<'_>) -> Result<RemoteKv> {
            let pattern = pattern_for(kv.request_id);
            for &block_id in kv.block_ids {
                self.fill_slot(block_id, pattern);
            }
            debug!(
                request_id = kv.request_id,
                blocks = kv.block_ids.len(),
                "advertised paged KV"
            );
            Ok(RemoteKv {
                engine_id: self.cfg.engine_id.clone(),
                host: self.cfg.side_channel_host.clone(),
                port: self.cfg.side_channel_port,
                block_ids: kv.block_ids.iter().map(|&id| id as i64).collect(),
                request_id: kv.request_id.to_string(),
            })
        }

        /// Pull a remote prefill's KV: load the peer's metadata over the side channel (once),
        /// then post a single multi-descriptor NIXL READ, one descriptor per advertised block,
        /// gathering each remote slot (`pool_base + id*block_bytes`) into a contiguous local
        /// landing buffer, and verify the per-request pattern. Decode and prefill are distinct
        /// agents.
        fn do_pull(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> Result<u64> {
            let n_blocks = remote.block_ids.len();
            if n_blocks == 0 {
                return Ok(0);
            }
            // Both sides derive the verify pattern from the prefill's request id; nothing about
            // the bytes' identity travels on the wire.
            let pattern = pattern_for(&remote.request_id);

            let (peer_agent, pool_base, block_bytes) = {
                let peer = self.ensure_peer(&remote.engine_id, &remote.host, remote.port)?;
                (
                    peer.agent_name.clone(),
                    peer.pool_base,
                    peer.block_bytes as usize,
                )
            };
            if block_bytes == 0 || pool_base == 0 {
                return Ok(0);
            }
            let total = n_blocks * block_bytes;

            // One remote descriptor per block, at its paged address in the prefill's pool.
            let mut remote_descs = XferDescList::new(MemType::Dram).map_err(ne)?;
            for &block_id in &remote.block_ids {
                let addr = pool_base as usize + block_id as usize * block_bytes;
                remote_descs.add_desc(addr, block_bytes, 0);
            }

            // Land the bytes in a fresh, registered contiguous buffer, one local descriptor
            // per block so the desc lists line up (matching count + sizes).
            let opt = self.opt_args()?;
            let mut dst = SystemStorage::new(total).map_err(ne)?;
            dst.register(&self.agent, Some(&opt)).map_err(ne)?;
            // SAFETY: stable heap address of the Vec backing the storage.
            let dst_base = unsafe { dst.as_ptr() as usize };

            let mut local = XferDescList::new(MemType::Dram).map_err(ne)?;
            for j in 0..n_blocks {
                local.add_desc(dst_base + j * block_bytes, block_bytes, 0);
            }

            let req = self
                .agent
                .create_xfer_req(XferOp::Read, &local, &remote_descs, &peer_agent, Some(&opt))
                .map_err(ne)?;
            self.agent.post_xfer_req(&req, Some(&opt)).map_err(ne)?;

            let start = Instant::now();
            loop {
                if self.agent.get_xfer_status(&req).map_err(ne)?.is_success() {
                    break;
                }
                if start.elapsed() > XFER_TIMEOUT {
                    bail!(
                        "NIXL READ from {} timed out after {XFER_TIMEOUT:?}",
                        remote.engine_id
                    );
                }
                std::thread::sleep(Duration::from_micros(200));
            }

            // Verify each landed block carries the expected pattern, and copy it into our own
            // pool slot so a later request can prefix-hit it locally.
            let landed = dst.as_slice();
            for (j, &local_id) in kv.block_ids.iter().take(n_blocks).enumerate() {
                let got = landed.get(j * block_bytes).copied();
                if got != Some(pattern) {
                    bail!("KV verify failed for block {j}: expected 0x{pattern:02x}, got {got:?}");
                }
                self.fill_slot(local_id, pattern);
            }
            debug!(
                request_id = kv.request_id,
                bytes = total,
                blocks = n_blocks,
                engine_id = remote.engine_id,
                "pulled + verified paged KV over NIXL"
            );
            Ok(total as u64)
        }
    }

    impl KvDataPlane for NixlDataPlane {
        fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Result<RemoteKv> {
            // Under stub the addressing (control plane) is still produced; only the transfer is
            // a no-op (the pool isn't registered and no descriptor server runs).
            self.do_advertise(kv)
        }

        fn pull_prefilled(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> Result<u64> {
            if is_stub() {
                return Ok(0);
            }
            self.do_pull(kv, remote)
        }

        fn release(&mut self, request_id: &str) {
            // Block lifetime is the block pool's job now; the registered KV pool persists for
            // the agent's lifetime, so there is nothing per-request to free here.
            debug!(request_id, role = ?self.role, "release (paged pool persists)");
        }
    }

    /// Length-prefix (u32 big-endian) + msgpack: the side-channel framing for one descriptor.
    fn write_framed(stream: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
        stream.write_all(&(payload.len() as u32).to_be_bytes())?;
        stream.write_all(payload)?;
        stream.flush()
    }

    fn read_framed(stream: &mut TcpStream) -> Result<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        stream.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Spawn a tiny blocking TCP server that answers every connection with the (constant)
    /// encoded `PoolDescriptor`. This is the prefill's metadata side channel; a daemon thread
    /// for the agent's lifetime (the mock pod runs one engine), mirroring NIXL's own listener.
    fn serve_descriptor(port: u32, descriptor: PoolDescriptor) -> Result<()> {
        let payload = descriptor.encode()?;
        let listener = TcpListener::bind(("0.0.0.0", port as u16))
            .map_err(|e| anyhow!("bind metadata side channel on :{port}: {e}"))?;
        std::thread::Builder::new()
            .name(format!("kv-meta-{port}"))
            .spawn(move || {
                for stream in listener.incoming() {
                    match stream {
                        Ok(mut stream) => {
                            if let Err(e) = write_framed(&mut stream, &payload) {
                                debug!(%e, "metadata side channel write failed");
                            }
                        }
                        Err(e) => debug!(%e, "metadata side channel accept failed"),
                    }
                }
            })
            .map_err(|e| anyhow!("spawn metadata side channel thread: {e}"))?;
        Ok(())
    }

    /// Fetch and decode a prefill peer's `PoolDescriptor` from its side channel.
    fn fetch_descriptor(host: &str, port: u32) -> Result<PoolDescriptor> {
        let mut stream = TcpStream::connect((host, port as u16))
            .map_err(|e| anyhow!("connect metadata side channel {host}:{port}: {e}"))?;
        let payload = read_framed(&mut stream)?;
        PoolDescriptor::decode(&payload)
    }
}
