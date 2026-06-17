//! Engine-core protocol glue shared between the simulator and the recording
//! tap: the frontend handshake (with a caller-supplied registration payload),
//! `kv_transfer_params` extraction, and conversions between the wire
//! finish-reason enum and the vllm-free trace schema.

pub mod frontend_connect;
pub mod kvparams;
pub mod mock_engine;
pub mod wire;
