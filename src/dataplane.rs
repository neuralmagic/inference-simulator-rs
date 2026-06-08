//! The KV-block data plane: the connector boundary where prefill/decode moves bytes.
//!
//! In real vLLM the produce/consume of `kv_transfer_params` and the NIXL transfer live
//! in the NixlConnector inside the engine/worker. Our mock engine *is* the engine, so
//! this module plays that connector role, wire-compatibly with the llm-d routing
//! sidecar and a real vLLM peer:
//!
//!   - PREFILL registers a (fake) KV region with NIXL and advertises how to reach it
//!     as [`RemoteKv`] (`remote_engine_id` / `remote_host` / `remote_port` /
//!     `remote_block_ids`). The engine wraps that into the real `kv_transfer_params`
//!     dict the sidecar relays.
//!   - DECODE receives that addressing, fetches the prefill agent's NIXL metadata over
//!     a ZMQ side channel, and posts a NIXL READ to pull the blocks before generating.
//!
//! ```text
//!   prefill engine                            decode engine
//!   ┌────────────────────┐   kv_transfer_     ┌────────────────────┐
//!   │ register KV w/ NIXL │   params dict      │ fetch md (ZMQ),     │
//!   │ serve md (ZMQ)      │   {remote_host,    │ add_remote_agent,   │
//!   │ advertise RemoteKv ─┼─  port,engine_id, ─┼▶ NIXL READ, verify  │
//!   └────────────────────┘    block_ids}       └────────────────────┘
//! ```
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
    /// Bytes per fabricated KV block.
    pub kv_block_bytes: usize,
    /// Prompt tokens that map to one KV block.
    pub tokens_per_block: usize,
    /// This engine's id, advertised as `remote_engine_id` (a real decode peer matches
    /// it against the NIXL agent metadata's engine id).
    pub engine_id: String,
    /// Host a decode peer connects to for the NIXL metadata side channel.
    pub side_channel_host: String,
    /// Port the NIXL metadata side channel listens on.
    pub side_channel_port: u32,
}

/// How a decode peer reaches a prefilled request's KV. These are exactly the
/// `remote_*` fields of vLLM's `kv_transfer_params`.
#[derive(Debug, Clone)]
pub struct RemoteKv {
    pub engine_id: String,
    pub host: String,
    pub port: u32,
    /// Logical KV block ids on the prefill side to read.
    pub block_ids: Vec<i64>,
}

/// The minimal view of a request the data plane needs, decoupled from the wire
/// `EngineCoreRequest` so the protocol crate can evolve without touching this trait.
#[derive(Debug, Clone, Copy)]
pub struct RequestKv<'a> {
    pub request_id: &'a str,
    /// Number of prompt tokens; drives how many KV blocks we fabricate.
    pub num_tokens: usize,
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

    fn block_ids(&self, num_tokens: usize) -> Vec<i64> {
        let n = num_tokens.div_ceil(self.cfg.tokens_per_block).max(1);
        (0..n as i64).collect()
    }
}

impl KvDataPlane for NoopDataPlane {
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> anyhow::Result<RemoteKv> {
        Ok(RemoteKv {
            engine_id: self.cfg.engine_id.clone(),
            host: self.cfg.side_channel_host.clone(),
            port: self.cfg.side_channel_port,
            block_ids: self.block_ids(kv.num_tokens),
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

/// The real NIXL-backed KV data plane.
#[cfg(feature = "nixl")]
mod nixl {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use anyhow::{Result, anyhow, bail};
    use nixl_sys::{
        Agent, Backend, MemType, MemoryRegion as _, NixlError, NixlRegistration as _, OptArgs,
        SystemStorage, XferDescList, XferOp, is_stub,
    };
    use tracing::{debug, warn};

    use super::{KvDataPlane, NixlConfig, PdRole, RemoteKv, RequestKv};

    /// How long to poll a NIXL transfer before giving up.
    const XFER_TIMEOUT: Duration = Duration::from_secs(5);

    /// Map a `NixlError` into an `anyhow::Error` (it is not `std::error::Error`).
    fn ne(error: NixlError) -> anyhow::Error {
        anyhow!("nixl: {error:?}")
    }

    /// A registered KV buffer awaiting a pull, plus what a peer needs to read it.
    struct StagedKv {
        /// Liveness anchor: keeps the registered buffer alive at `addr` until release.
        #[allow(dead_code)]
        storage: SystemStorage,
        addr: u64,
        len: u64,
        pattern: u8,
    }

    pub struct NixlDataPlane {
        role: PdRole,
        cfg: NixlConfig,
        /// NIXL agent name == engine id, so a peer's `remote_engine_id` is the agent name.
        agent: Agent,
        backend: Backend,
        /// Prefill side: registered KV buffers awaiting a pull, keyed by request id.
        staged: HashMap<String, StagedKv>,
    }

    impl NixlDataPlane {
        pub fn new(role: PdRole, cfg: NixlConfig) -> Result<Self> {
            let agent = Agent::new(&cfg.engine_id).map_err(ne)?;
            let (_, params) = agent.get_plugin_params("UCX").map_err(ne)?;
            let backend = agent.create_backend("UCX", &params).map_err(ne)?;
            debug!(
                engine_id = cfg.engine_id,
                ?role,
                "NIXL data plane ready (UCX backend)"
            );
            Ok(Self {
                role,
                cfg,
                agent,
                backend,
                staged: HashMap::new(),
            })
        }

        /// Fresh opt args carrying our backend (required for register and transfer calls).
        fn opt_args(&self) -> Result<OptArgs> {
            let mut opt = OptArgs::new().map_err(ne)?;
            opt.add_backend(&self.backend).map_err(ne)?;
            Ok(opt)
        }

        fn do_advertise(&mut self, kv: RequestKv<'_>) -> Result<RemoteKv> {
            let n_blocks = kv.num_tokens.div_ceil(self.cfg.tokens_per_block).max(1);
            let total = n_blocks * self.cfg.kv_block_bytes;

            let mut storage = SystemStorage::new(total).map_err(ne)?;
            let pattern = pattern_for(kv.request_id);
            storage.memset(pattern);
            let opt = self.opt_args()?;
            storage.register(&self.agent, Some(&opt)).map_err(ne)?;
            // SAFETY: the buffer is a heap `Vec<u8>`; its address is stable across the
            // move into `staged` (moving the Vec header does not reallocate).
            let addr = unsafe { storage.as_ptr() as u64 };

            self.staged.insert(
                kv.request_id.to_string(),
                StagedKv {
                    storage,
                    addr,
                    len: total as u64,
                    pattern,
                },
            );
            debug!(
                request_id = kv.request_id,
                blocks = n_blocks,
                bytes = total,
                "advertised KV"
            );

            Ok(RemoteKv {
                engine_id: self.cfg.engine_id.clone(),
                host: self.cfg.side_channel_host.clone(),
                port: self.cfg.side_channel_port,
                block_ids: (0..n_blocks as i64).collect(),
            })
        }

        /// Real NIXL READ from a staged source buffer into a fresh landing buffer, using
        /// `remote_name` as the NIXL remote agent. Verifies the pattern. This is the
        /// transfer mechanic; the in-process path addresses our own agent, the cross-pod
        /// path (TODO: ZMQ side channel) will address the peer's.
        fn read_staged(&self, src: &StagedKv, remote_name: &str) -> Result<u64> {
            let opt = self.opt_args()?;
            let mut dst = SystemStorage::new(src.len as usize).map_err(ne)?;
            dst.register(&self.agent, Some(&opt)).map_err(ne)?;
            // SAFETY: same stable-heap-address reasoning as the prefill side.
            let dst_base = unsafe { dst.as_ptr() as usize };

            let mut local = XferDescList::new(MemType::Dram).map_err(ne)?;
            let mut remote = XferDescList::new(MemType::Dram).map_err(ne)?;
            local.add_desc(dst_base, src.len as usize, 0);
            remote.add_desc(src.addr as usize, src.len as usize, 0);

            let req = self
                .agent
                .create_xfer_req(XferOp::Read, &local, &remote, remote_name, Some(&opt))
                .map_err(ne)?;
            self.agent.post_xfer_req(&req, Some(&opt)).map_err(ne)?;

            let start = Instant::now();
            loop {
                if self.agent.get_xfer_status(&req).map_err(ne)?.is_success() {
                    break;
                }
                if start.elapsed() > XFER_TIMEOUT {
                    bail!("NIXL READ from {remote_name} timed out after {XFER_TIMEOUT:?}");
                }
                std::thread::sleep(Duration::from_micros(200));
            }

            let got = dst.as_slice().first().copied();
            if got != Some(src.pattern) {
                bail!(
                    "KV verify failed: expected 0x{:02x}, got {got:?}",
                    src.pattern
                );
            }
            Ok(src.len)
        }

        fn do_pull(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> Result<u64> {
            // In-process path: the prefill and decode roles are the same agent (e.g. the
            // loopback test), so the source buffer is staged locally and the remote agent
            // is ourselves. Address self by loading our own metadata as a remote.
            if remote.engine_id == self.cfg.engine_id
                && let Some(src) = self.staged.remove(kv.request_id)
            {
                let md = self.agent.get_local_md().map_err(ne)?;
                let self_name = self.agent.load_remote_md(&md).map_err(ne)?;
                let bytes = self.read_staged(&src, &self_name)?;
                debug!(
                    request_id = kv.request_id,
                    bytes, "pulled + verified KV (in-process)"
                );
                return Ok(bytes);
            }

            // TODO(increment-2): cross-pod pull. Connect a ZMQ REQ socket to
            // `remote.host:remote.port`, send `msgpack((b"get_meta_msg", rank))`, decode the
            // NixlHandshakePayload -> NixlAgentMetadata, `load_remote_md(agent_metadata)`,
            // build remote descs from base_addr + block_id*block_len, then `read_staged`.
            warn!(
                request_id = kv.request_id,
                engine_id = remote.engine_id,
                host = remote.host,
                port = remote.port,
                "cross-pod NIXL pull not wired yet (needs the ZMQ side channel); skipping transfer"
            );
            Ok(0)
        }
    }

    impl KvDataPlane for NixlDataPlane {
        fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Result<RemoteKv> {
            // Even under stub we can return addressing (control plane); only the transfer
            // is a no-op. But stub register/get_local_md may error, so short-circuit.
            if is_stub() {
                return Ok(RemoteKv {
                    engine_id: self.cfg.engine_id.clone(),
                    host: self.cfg.side_channel_host.clone(),
                    port: self.cfg.side_channel_port,
                    block_ids: vec![0],
                });
            }
            self.do_advertise(kv)
        }

        fn pull_prefilled(&mut self, kv: RequestKv<'_>, remote: &RemoteKv) -> Result<u64> {
            if is_stub() {
                return Ok(0);
            }
            self.do_pull(kv, remote)
        }

        fn release(&mut self, request_id: &str) {
            if self.staged.remove(request_id).is_some() {
                debug!(request_id, role = ?self.role, "released staged KV");
            }
        }
    }

    /// Stable per-request fill byte, so a decode pull can verify it got the right bytes.
    fn pattern_for(request_id: &str) -> u8 {
        request_id
            .bytes()
            .fold(0xa5u8, |acc, b| acc.wrapping_add(b))
            | 1
    }
}
