//! Engine-core recording tap: a transparent ZMQ proxy between a real vLLM
//! frontend and a real engine-core that records per-request timing into a JSONL
//! trace (plus an optional step-stats sidecar stream).

pub mod tap;
