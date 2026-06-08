//! The KV-block data plane: the boundary where simulated prefill/decode actually
//! moves bytes between engines.
//!
//! Today's `llm-d-inference-sim` fakes P/D purely in the control plane: it changes
//! the latency model and tags a finish reason, but no KV cache bytes ever move.
//! This module is where that changes. A prefill engine registers fake KV-block
//! buffers and advertises them; a decode engine pulls those bytes over NIXL
//! (UCX/DRAM, or real RDMA NICs) before it "decodes". No GPU required.
//!
//! ```text
//!   prefill engine                         decode engine
//!   ┌────────────────────┐                 ┌────────────────────┐
//!   │ fabricate fake KV   │  kv_transfer_   │ load remote md,     │
//!   │ register w/ NIXL    │  params (JSON): │ post NIXL READ,     │
//!   │ advertise descs ────┼─ md + descs ───▶│ poll, verify, decode│
//!   └────────────────────┘                 └────────────────────┘
//! ```
//!
//! The default build ships [`NoopDataPlane`] (control-plane only, byte-for-byte the
//! current sim behavior). The real transfer lives behind the `nixl` feature.

use serde_json::Value;

/// The role this engine plays in a disaggregated prefill/decode deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PdRole {
    /// Computes (simulated) KV and hands it off. Advertises blocks for a remote puller.
    Prefill,
    /// Pulls remote KV before generating tokens.
    Decode,
    /// Monolithic: no handoff, behaves like a normal single engine.
    Both,
}

/// Sizing knobs for the fabricated KV cache. Real KV block bytes depend on model
/// dims; for the sim we fabricate a plausible size so the transfer has real volume.
#[derive(Debug, Clone, Copy)]
pub struct NixlConfig {
    /// Bytes per fabricated KV block.
    pub kv_block_bytes: usize,
    /// Prompt tokens that map to one KV block.
    pub tokens_per_block: usize,
}

/// The minimal view of a request the data plane needs, decoupled from the wire
/// `EngineCoreRequest` so the protocol crate can evolve without touching this trait.
#[derive(Debug, Clone, Copy)]
pub struct RequestKv<'a> {
    pub request_id: &'a str,
    /// Number of prompt tokens; drives how many KV blocks we fabricate.
    pub num_tokens: usize,
}

/// Where simulated KV cache bytes actually move between a prefill and a decode engine.
pub trait KvDataPlane: Send {
    /// Called on a prefill engine once a request's simulated KV cache is "ready".
    /// Returns the `kv_transfer_params` to embed in the engine output so the decode
    /// engine can locate and pull these blocks. `None` advertises nothing.
    fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Option<Value>;

    /// Called on a decode engine before generation when a request carries
    /// `kv_transfer_params`. Pulls the remote KV blocks. Returns bytes moved.
    fn pull_prefilled(&mut self, kv: RequestKv<'_>, params: &Value) -> anyhow::Result<u64>;

    /// Release any resources staged for a request (e.g. a prefill engine dropping
    /// the KV buffer once it is no longer needed). No-op by default.
    fn release(&mut self, _request_id: &str) {}
}

/// Control-plane-only data plane: models P/D handoff but moves zero bytes. This is
/// exactly what the simulator does today, kept as the default so the protocol spike
/// (bird one) needs no NIXL/UCX install.
pub struct NoopDataPlane;

impl KvDataPlane for NoopDataPlane {
    fn advertise_prefilled(&mut self, _kv: RequestKv<'_>) -> Option<Value> {
        None
    }

    fn pull_prefilled(&mut self, _kv: RequestKv<'_>, _params: &Value) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// Build the data plane for a given role. Without the `nixl` feature there is only
/// the no-op plane, so the default binary is a faithful protocol emulator and nothing
/// more. With `nixl`, prefill/decode roles get a real NIXL-backed plane (bird two);
/// if NIXL init fails (no libnixl, no UCX) we degrade to the no-op plane rather than
/// crash, so the same binary still runs as a pure protocol emulator.
pub fn make_data_plane(role: PdRole, cfg: NixlConfig) -> Box<dyn KvDataPlane> {
    #[cfg(feature = "nixl")]
    {
        if !matches!(role, PdRole::Both) {
            match nixl::NixlDataPlane::new(role, cfg) {
                Ok(plane) => return Box::new(plane),
                Err(error) => {
                    tracing::warn!(%error, "NIXL init failed; using no-op data plane");
                }
            }
        }
    }
    let _ = (role, cfg);
    Box::new(NoopDataPlane)
}

/// Whether the NIXL bindings are the no-op stubs (true when built without real
/// libnixl, e.g. via `--features nixl-stub`, or when the `nixl` feature is off).
/// Tests use this to skip the real transfer on machines that cannot move bytes.
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
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use nixl_sys::{
        Agent, Backend, MemType, MemoryRegion as _, NixlError, NixlRegistration as _, OptArgs,
        SystemStorage, XferDescList, XferOp, is_stub,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use tracing::{debug, warn};

    use super::{KvDataPlane, NixlConfig, PdRole, RequestKv};

    /// How long to poll a NIXL transfer before giving up.
    const XFER_TIMEOUT: Duration = Duration::from_secs(5);

    /// One advertised KV block: an addressable, registered region on the prefill engine.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct KvBlockDesc {
        addr: u64,
        len: u64,
        dev_id: u64,
    }

    /// The `kv_transfer_params` payload a prefill engine advertises and a decode engine
    /// pulls against. This is the sim-to-sim handshake.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct KvHandoff {
        /// NIXL agent name of the prefill engine (the remote for the decode pull).
        engine: String,
        /// Base64 of the prefill agent's NIXL metadata blob (`get_local_md`).
        agent_md: String,
        /// Registered KV block regions to read from.
        blocks: Vec<KvBlockDesc>,
        /// Byte the prefill buffer was filled with, so the decode side can verify the pull.
        pattern: u8,
    }

    /// Map a `NixlError` into an `anyhow::Error` (it is not `std::error::Error`).
    fn ne(error: NixlError) -> anyhow::Error {
        anyhow!("nixl: {error:?}")
    }

    pub struct NixlDataPlane {
        role: PdRole,
        cfg: NixlConfig,
        name: String,
        agent: Agent,
        backend: Backend,
        /// Prefill side: registered KV buffers awaiting a pull, keyed by request id.
        /// Held alive (and registered) until `release`d or the process exits.
        staged: HashMap<String, SystemStorage>,
    }

    impl NixlDataPlane {
        pub fn new(role: PdRole, cfg: NixlConfig) -> Result<Self> {
            // One engine plays one role per process, so role + pid is a unique agent name
            // that a peer process can address.
            let name = format!("mock-{}-{}", role_tag(role), std::process::id());
            let agent = Agent::new(&name).map_err(ne)?;
            let (_, params) = agent.get_plugin_params("UCX").map_err(ne)?;
            let backend = agent.create_backend("UCX", &params).map_err(ne)?;
            debug!(name, ?role, "NIXL data plane ready (UCX backend)");
            Ok(Self {
                role,
                cfg,
                name,
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

        fn do_advertise(&mut self, kv: RequestKv<'_>) -> Result<Value> {
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
            let md = self.agent.get_local_md().map_err(ne)?;

            let handoff = KvHandoff {
                engine: self.name.clone(),
                agent_md: BASE64.encode(&md),
                blocks: vec![KvBlockDesc {
                    addr,
                    len: total as u64,
                    dev_id: 0,
                }],
                pattern,
            };

            self.staged.insert(kv.request_id.to_string(), storage);
            debug!(
                request_id = kv.request_id,
                blocks = n_blocks,
                bytes = total,
                "advertised KV"
            );
            Ok(serde_json::to_value(handoff)?)
        }

        fn do_pull(&mut self, kv: RequestKv<'_>, params: &Value) -> Result<u64> {
            let handoff: KvHandoff = serde_json::from_value(params.clone())?;
            let remote_md = BASE64.decode(&handoff.agent_md)?;
            let remote_name = self.agent.load_remote_md(&remote_md).map_err(ne)?;

            let total: u64 = handoff.blocks.iter().map(|block| block.len).sum();
            let mut dst = SystemStorage::new(total as usize).map_err(ne)?;
            let opt = self.opt_args()?;
            dst.register(&self.agent, Some(&opt)).map_err(ne)?;

            // Local landing descriptors (where bytes arrive) and remote source
            // descriptors (the prefill engine's registered blocks).
            let mut local = XferDescList::new(MemType::Dram).map_err(ne)?;
            let mut remote = XferDescList::new(MemType::Dram).map_err(ne)?;
            // SAFETY: same stable-heap-address reasoning as the prefill side.
            let dst_base = unsafe { dst.as_ptr() as usize };
            let mut offset = 0usize;
            for block in &handoff.blocks {
                remote.add_desc(block.addr as usize, block.len as usize, block.dev_id);
                local.add_desc(dst_base + offset, block.len as usize, 0);
                offset += block.len as usize;
            }

            let req = self
                .agent
                .create_xfer_req(XferOp::Read, &local, &remote, &remote_name, Some(&opt))
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

            // The pull is only meaningful if the bytes actually arrived; verifying the
            // pattern folds correctness into the data path.
            let got = dst.as_slice().first().copied();
            if got != Some(handoff.pattern) {
                bail!(
                    "KV verify failed: expected 0x{:02x}, got {got:?}",
                    handoff.pattern
                );
            }
            debug!(request_id = kv.request_id, bytes = total, %remote_name, "pulled + verified KV");
            Ok(total)
        }
    }

    impl KvDataPlane for NixlDataPlane {
        fn advertise_prefilled(&mut self, kv: RequestKv<'_>) -> Option<Value> {
            if is_stub() {
                return None; // stub bindings cannot move bytes
            }
            match self.do_advertise(kv) {
                Ok(value) => Some(value),
                Err(error) => {
                    warn!(request_id = kv.request_id, %error, "NIXL advertise failed");
                    None
                }
            }
        }

        fn pull_prefilled(&mut self, kv: RequestKv<'_>, params: &Value) -> anyhow::Result<u64> {
            if is_stub() {
                return Ok(0);
            }
            self.do_pull(kv, params)
        }

        fn release(&mut self, request_id: &str) {
            // Dropping the storage releases the buffer; on a short-lived sim process
            // any backing NIXL registration is reclaimed at exit.
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

    fn role_tag(role: PdRole) -> &'static str {
        match role {
            PdRole::Prefill => "prefill",
            PdRole::Decode => "decode",
            PdRole::Both => "both",
        }
    }
}
