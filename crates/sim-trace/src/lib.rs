//! Trace tooling for the inference simulator: the JSONL trace schema, the
//! response-timing latency models, and the guidellm benchmark converter.
//!
//! This crate is deliberately free of the `vllm-engine-core-client` git
//! dependency so trace and calibration work can build and test fast, without
//! compiling the engine-core protocol stack. Anything that touches the wire
//! protocol (finish-reason conversions, step-stats snapshots) lives in the
//! engine crates instead.

pub mod config_hash;
pub mod latency;
pub mod perfetto;
pub mod trace;
pub mod trace_convert;
