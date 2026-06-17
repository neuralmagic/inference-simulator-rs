//! Sim-owned handshake-harness types.
//!
//! vLLM's `vllm_engine_core_client::mock_engine` module provides these at the
//! head rev, but it was added after vLLM 0.22 and is absent on older lines. We
//! never depended on its *behavior* (the engine-side handshake is reimplemented
//! in [`crate::frontend_connect`]); we only used its data structs, one constant,
//! and a default dtype. Owning them here drops the dependency on `mock_engine`
//! entirely, so the sim builds against revs that predate the module. The structs
//! mirror the head crate's definitions field-for-field, and the constant/dtype
//! match its `DEFAULT_MOCK_MAX_MODEL_LEN` / `default_ready_response().dtype`.

use vllm_engine_core_client::protocol::ModelDtype;
use vllm_engine_core_client::protocol::handshake::HandshakeInitMessage;
use zeromq::{DealerSocket, PushSocket, SubSocket};

/// Default max model length advertised when the caller doesn't set one.
pub const DEFAULT_MOCK_MAX_MODEL_LEN: u64 = 1024 * 1024;

/// Default dtype advertised in the registration ready response.
pub fn default_dtype() -> ModelDtype {
    ModelDtype::Float32
}

/// Coordinator-side sockets for one mock engine when coordinator mode is on.
pub struct MockCoordinatorSockets {
    /// Receives coordinator broadcasts such as `START_DP_WAVE`.
    pub input_sub: SubSocket,
    /// Sends coordinator-only `EngineCoreOutputs` back to the frontend.
    pub output_push: PushSocket,
}

/// One mock engine's data-plane connection to one frontend client.
pub struct MockEngineDataSockets {
    /// Receives frontend requests.
    pub dealer: DealerSocket,
    /// Publishes request outputs back to the frontend.
    pub push: PushSocket,
}

/// Frontend-facing sockets owned by one mock engine.
pub struct MockEngineSockets {
    /// The INIT message the frontend sent during the handshake.
    pub init: HandshakeInitMessage,
    /// Data sockets for all frontend clients, in client-index order.
    pub data_sockets: Vec<MockEngineDataSockets>,
    /// Coordinator sockets, when coordinator mode is enabled.
    pub coordinator: Option<MockCoordinatorSockets>,
}
