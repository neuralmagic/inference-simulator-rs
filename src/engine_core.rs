//! The EngineCore trait boundary and the generic engine loop.
//!
//! `EngineCore` is the contract any engine implementation must satisfy to plug into the
//! ZMQ IO loop. `SimEngine` (in `engine.rs`) is the only implementation today, but the
//! trait lets us swap in alternative backends without touching the loop or transport.

use std::time::Duration;

use anyhow::{Result, anyhow};
use serde_tuple::Deserialize_tuple;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use vllm_engine_core_client::protocol::utility::UtilityCallId;
use vllm_engine_core_client::protocol::{EngineCoreOutputs, EngineCoreRequest};

/// The engine-core utility request (`add_lora` / `remove_lora` /
/// `reset_prefix_cache`).
///
/// We decode the wire tuple `(client_index, call_id, method_name, args)`
/// ourselves rather than via the crate's `EngineCoreUtilityRequest`: that type
/// only derives `Deserialize` on 0.23+ (the crate was client-only on 0.22), and
/// the mock engine is the decoding side. Field order and the tuple encoding
/// match the crate exactly (`OpaqueValue` is `rmpv::Value`), so it reads the
/// same bytes on every supported line.
#[derive(Debug, Clone, Deserialize_tuple)]
pub(crate) struct UtilityRequestSpec {
    pub client_index: u32,
    pub call_id: UtilityCallId,
    pub method_name: String,
    pub args: rmpv::Value,
}

/// Message sent from the IO loop to the engine task to drive the engine loop.
pub(crate) enum EngineInput {
    Request(Box<EngineCoreRequest>),
    Abort(Vec<String>),
    Utility(UtilityRequestSpec),
    StartDpWave,
}

/// Message sent from the engine task to the IO loop for one engine output batch.
pub(crate) struct EngineOutput {
    pub client_index: u32,
    pub outputs: EngineCoreOutputs,
}

/// The contract an engine implementation must satisfy to plug into the generic loop.
pub(crate) trait EngineCore: Send {
    /// Engine-internal completion events (e.g. finished KV pulls). Engines without
    /// internal events use () and hand back a channel that never fires.
    type Internal: Send + 'static;
    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>>;
    /// The loop owns the receiver (taken once at startup) so select! borrows stay clean.
    fn take_internal_rx(&mut self) -> mpsc::UnboundedReceiver<Self::Internal>;
    fn on_internal(&mut self, event: Self::Internal) -> Vec<EngineOutput>;
    fn earliest_deadline(&self) -> Option<Instant>;
    fn step(&mut self) -> Vec<EngineOutput>;
    /// Requests the engine still owes outputs for (running, queued, or parked).
    /// Drives the graceful-shutdown drain: the loop exits once this hits zero.
    fn num_unfinished_requests(&self) -> usize;
    /// Finish every unfinished request immediately with an Abort output (vLLM's
    /// `shutdown_timeout == 0` path and the drain-deadline fallback).
    fn abort_all_requests(&mut self) -> Vec<EngineOutput>;
    /// Refuse a request that arrived during shutdown with an immediate Abort output,
    /// mirroring vLLM's `_reject_add_in_shutdown`.
    fn reject_request(&self, request: Box<EngineCoreRequest>) -> EngineOutput;
}

/// Run the main loop for one engine, receiving `EngineInput` and sending `EngineOutput`
/// until shutdown completes. The engine must already be constructed; construction
/// stays with the caller.
///
/// Shutdown mirrors vLLM's engine core (`EngineCoreProc._handle_shutdown`): when
/// `shutdown` is cancelled, a zero `shutdown_timeout` aborts every unfinished request
/// immediately (Abort outputs to the frontend); a nonzero timeout drains in-flight
/// requests while rejecting new ones, aborting whatever remains at the deadline. vLLM
/// leaves deadline enforcement to its parent process; we run standalone, so the loop
/// enforces it itself. A second signal during the drain is ignored, as in vLLM.
///
/// Any `EngineCore` impl reuses this loop unchanged: the loop owns the select! over
/// inputs, internal events, and deadline-driven steps, so custom engines only need to
/// implement the trait, not duplicate the concurrency harness.
pub(crate) async fn run_loop<E: EngineCore>(
    mut engine: E,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::UnboundedSender<EngineOutput>,
    shutdown: CancellationToken,
    shutdown_timeout: Duration,
) -> Result<()> {
    let mut internal_rx = engine.take_internal_rx();
    // `None` while running; `Some(deadline)` once a drain has started.
    let mut drain_deadline: Option<Instant> = None;

    loop {
        if drain_deadline.is_some() && engine.num_unfinished_requests() == 0 {
            info!("shutdown: request processing complete");
            break;
        }
        let next_deadline = engine.earliest_deadline();
        let outputs = tokio::select! {
            biased;
            _ = shutdown.cancelled(), if drain_deadline.is_none() => {
                let unfinished = engine.num_unfinished_requests();
                if shutdown_timeout.is_zero() {
                    if unfinished > 0 {
                        info!(unfinished, "shutdown: aborting in-flight requests");
                    }
                    let outputs = engine.abort_all_requests();
                    for output in outputs {
                        output_tx
                            .send(output)
                            .map_err(|_| anyhow!("engine IO task shut down"))?;
                    }
                    break;
                }
                info!(
                    unfinished,
                    timeout_s = shutdown_timeout.as_secs(),
                    "shutdown: draining in-flight requests"
                );
                drain_deadline = Some(Instant::now() + shutdown_timeout);
                continue;
            }

            _ = async { sleep_until(drain_deadline.unwrap_or_else(Instant::now)).await },
                if drain_deadline.is_some() =>
            {
                warn!(
                    unfinished = engine.num_unfinished_requests(),
                    "shutdown: drain timed out; aborting remaining requests"
                );
                let outputs = engine.abort_all_requests();
                for output in outputs {
                    output_tx
                        .send(output)
                        .map_err(|_| anyhow!("engine IO task shut down"))?;
                }
                break;
            }

            input = input_rx.recv() => {
                let input = input.ok_or_else(|| anyhow!("engine input channel closed"))?;
                match input {
                    EngineInput::Request(request) if drain_deadline.is_some() => {
                        vec![engine.reject_request(request)]
                    }
                    input => engine.handle_input(input)?,
                }
            }

            Some(event) = internal_rx.recv() => {
                engine.on_internal(event)
            }

            _ = async { sleep_until(next_deadline.unwrap_or_else(Instant::now)).await },
                if next_deadline.is_some() =>
            {
                engine.step()
            }
        };

        // The send must never await: the step loop IS the engine clock, and
        // backpressure from a lagging consumer was measured stalling it for
        // 35-170ms bursts (16% of emissions late). A real engine fires its
        // outputs into the socket without pacing on the reader.
        for output in outputs {
            output_tx
                .send(output)
                .map_err(|_| anyhow!("engine IO task shut down"))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;
    use vllm_engine_core_client::protocol::{
        EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
        EngineCoreSamplingParams,
    };

    use crate::engine_core::{EngineCore, EngineInput, EngineOutput, run_loop};

    /// A trivially simple engine that responds to every request with one fixed token
    /// and an immediate Length finish. No scheduling, no rng, no latency, Internal = ().
    /// Proves the `run_loop` harness is fully reusable by a from-scratch engine.
    struct ConstantEngine {
        token_id: u32,
        internal_rx: Option<mpsc::UnboundedReceiver<()>>,
    }

    impl ConstantEngine {
        fn new(token_id: u32) -> Self {
            let (_tx, rx) = mpsc::unbounded_channel();
            ConstantEngine {
                token_id,
                internal_rx: Some(rx),
            }
        }
    }

    impl EngineCore for ConstantEngine {
        type Internal = ();

        fn handle_input(&mut self, input: EngineInput) -> anyhow::Result<Vec<EngineOutput>> {
            match input {
                EngineInput::Request(req) => {
                    let request_id = req.request_id.clone();
                    let client_index = req.client_index;
                    let output = EngineCoreOutput {
                        request_id: request_id.clone(),
                        new_token_ids: vec![self.token_id],
                        finish_reason: Some(EngineCoreFinishReason::Length),
                        ..Default::default()
                    };
                    Ok(vec![EngineOutput {
                        client_index,
                        outputs: EngineCoreOutputs {
                            engine_index: 0,
                            outputs: vec![output],
                            finished_requests: Some(BTreeSet::from([request_id])),
                            ..Default::default()
                        },
                    }])
                }
                _ => Ok(Vec::new()),
            }
        }

        fn take_internal_rx(&mut self) -> mpsc::UnboundedReceiver<()> {
            self.internal_rx.take().unwrap_or_else(|| {
                let (_tx, rx) = mpsc::unbounded_channel();
                rx
            })
        }

        fn on_internal(&mut self, _event: ()) -> Vec<EngineOutput> {
            Vec::new()
        }

        fn earliest_deadline(&self) -> Option<tokio::time::Instant> {
            None
        }

        fn step(&mut self) -> Vec<EngineOutput> {
            Vec::new()
        }

        fn num_unfinished_requests(&self) -> usize {
            0
        }

        fn abort_all_requests(&mut self) -> Vec<EngineOutput> {
            Vec::new()
        }

        fn reject_request(&self, request: Box<EngineCoreRequest>) -> EngineOutput {
            abort_output(request)
        }
    }

    /// Build the Abort output a rejected request gets, shared by the test engines.
    fn abort_output(request: Box<EngineCoreRequest>) -> EngineOutput {
        let output = EngineCoreOutput {
            request_id: request.request_id.clone(),
            finish_reason: Some(EngineCoreFinishReason::Abort),
            ..Default::default()
        };
        EngineOutput {
            client_index: request.client_index,
            outputs: EngineCoreOutputs {
                engine_index: 0,
                outputs: vec![output],
                finished_requests: Some(BTreeSet::from([request.request_id])),
                ..Default::default()
            },
        }
    }

    /// An engine whose single in-flight request finishes at a fixed deadline, for
    /// exercising the graceful-shutdown drain paths of `run_loop`.
    struct DelayedEngine {
        pending: Option<(String, u32, tokio::time::Instant)>,
        internal_rx: Option<mpsc::UnboundedReceiver<()>>,
    }

    impl DelayedEngine {
        fn new(request_id: &str, due: tokio::time::Instant) -> Self {
            let (_tx, rx) = mpsc::unbounded_channel();
            DelayedEngine {
                pending: Some((request_id.to_string(), 0, due)),
                internal_rx: Some(rx),
            }
        }

        fn finish_output(request_id: String, client_index: u32) -> EngineOutput {
            let output = EngineCoreOutput {
                request_id: request_id.clone(),
                new_token_ids: vec![7],
                finish_reason: Some(EngineCoreFinishReason::Length),
                ..Default::default()
            };
            EngineOutput {
                client_index,
                outputs: EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![output],
                    finished_requests: Some(BTreeSet::from([request_id])),
                    ..Default::default()
                },
            }
        }
    }

    impl EngineCore for DelayedEngine {
        type Internal = ();

        fn handle_input(&mut self, _input: EngineInput) -> anyhow::Result<Vec<EngineOutput>> {
            Ok(Vec::new())
        }

        fn take_internal_rx(&mut self) -> mpsc::UnboundedReceiver<()> {
            self.internal_rx.take().unwrap_or_else(|| {
                let (_tx, rx) = mpsc::unbounded_channel();
                rx
            })
        }

        fn on_internal(&mut self, _event: ()) -> Vec<EngineOutput> {
            Vec::new()
        }

        fn earliest_deadline(&self) -> Option<tokio::time::Instant> {
            self.pending.as_ref().map(|&(_, _, due)| due)
        }

        fn step(&mut self) -> Vec<EngineOutput> {
            let now = tokio::time::Instant::now();
            match self.pending.take() {
                Some((id, client, due)) if due <= now => {
                    vec![Self::finish_output(id, client)]
                }
                pending => {
                    self.pending = pending;
                    Vec::new()
                }
            }
        }

        fn num_unfinished_requests(&self) -> usize {
            usize::from(self.pending.is_some())
        }

        fn abort_all_requests(&mut self) -> Vec<EngineOutput> {
            let Some((id, client, _)) = self.pending.take() else {
                return Vec::new();
            };
            let output = EngineCoreOutput {
                request_id: id.clone(),
                finish_reason: Some(EngineCoreFinishReason::Abort),
                ..Default::default()
            };
            vec![EngineOutput {
                client_index: client,
                outputs: EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![output],
                    finished_requests: Some(BTreeSet::from([id])),
                    ..Default::default()
                },
            }]
        }

        fn reject_request(&self, request: Box<EngineCoreRequest>) -> EngineOutput {
            abort_output(request)
        }
    }

    #[tokio::test]
    async fn constant_engine_through_run_loop() {
        let engine = ConstantEngine::new(42);
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();

        let shutdown_clone = shutdown.clone();
        let loop_handle = tokio::spawn(run_loop(
            engine,
            input_rx,
            output_tx,
            shutdown_clone,
            std::time::Duration::ZERO,
        ));

        // Send two requests through the loop and verify each gets the constant response.
        for id in ["req-a", "req-b"] {
            let req = EngineCoreRequest {
                request_id: id.to_string(),
                prompt_token_ids: Some(vec![1, 2, 3]),
                sampling_params: Some(EngineCoreSamplingParams {
                    max_tokens: 1,
                    ..EngineCoreSamplingParams::for_test()
                }),
                ..Default::default()
            };
            input_tx
                .send(EngineInput::Request(Box::new(req)))
                .expect("send");

            let out = tokio::time::timeout(std::time::Duration::from_secs(2), output_rx.recv())
                .await
                .expect("timeout")
                .expect("recv");

            assert_eq!(out.outputs.outputs.len(), 1);
            let o = &out.outputs.outputs[0];
            assert_eq!(o.request_id, id);
            assert_eq!(o.new_token_ids, vec![42]);
            assert_eq!(o.finish_reason, Some(EngineCoreFinishReason::Length));
        }

        shutdown.cancel();
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), loop_handle)
            .await
            .expect("loop should exit on shutdown")
            .expect("join");
        assert!(result.is_ok());
    }

    /// With `shutdown_timeout == 0` (vLLM's default), cancellation aborts the in-flight
    /// request immediately: the client gets an Abort output and the loop exits.
    #[tokio::test]
    async fn shutdown_abort_mode_aborts_in_flight() {
        let due = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let engine = DelayedEngine::new("slow-1", due);
        let (_input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();

        let loop_handle = tokio::spawn(run_loop(
            engine,
            input_rx,
            output_tx,
            shutdown.clone(),
            std::time::Duration::ZERO,
        ));

        shutdown.cancel();

        let out = tokio::time::timeout(std::time::Duration::from_secs(2), output_rx.recv())
            .await
            .expect("abort output timed out")
            .expect("abort output");
        assert_eq!(out.outputs.outputs[0].request_id, "slow-1");
        assert_eq!(
            out.outputs.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), loop_handle)
            .await
            .expect("loop should exit after abort")
            .expect("join");
        assert!(result.is_ok());
    }

    /// With a nonzero timeout, the loop drains: the in-flight request finishes
    /// naturally, a request arriving mid-drain is rejected with Abort, and the loop
    /// exits once nothing is unfinished.
    #[tokio::test]
    async fn shutdown_drain_finishes_in_flight_and_rejects_new() {
        let due = tokio::time::Instant::now() + std::time::Duration::from_millis(200);
        let engine = DelayedEngine::new("drain-1", due);
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();

        let loop_handle = tokio::spawn(run_loop(
            engine,
            input_rx,
            output_tx,
            shutdown.clone(),
            std::time::Duration::from_secs(10),
        ));

        shutdown.cancel();

        // A request arriving during the drain is rejected immediately.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let late = EngineCoreRequest {
            request_id: "late-1".to_string(),
            ..Default::default()
        };
        input_tx
            .send(EngineInput::Request(Box::new(late)))
            .expect("send late request");

        let rejected = tokio::time::timeout(std::time::Duration::from_secs(2), output_rx.recv())
            .await
            .expect("rejection timed out")
            .expect("rejection output");
        assert_eq!(rejected.outputs.outputs[0].request_id, "late-1");
        assert_eq!(
            rejected.outputs.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );

        // The in-flight request still finishes with its natural reason.
        let finished = tokio::time::timeout(std::time::Duration::from_secs(2), output_rx.recv())
            .await
            .expect("drain output timed out")
            .expect("drain output");
        assert_eq!(finished.outputs.outputs[0].request_id, "drain-1");
        assert_eq!(
            finished.outputs.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), loop_handle)
            .await
            .expect("loop should exit after drain")
            .expect("join");
        assert!(result.is_ok());
    }

    /// A drain whose deadline expires aborts whatever is still unfinished.
    #[tokio::test]
    async fn shutdown_drain_deadline_aborts_remaining() {
        let due = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let engine = DelayedEngine::new("stuck-1", due);
        let (_input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();

        let loop_handle = tokio::spawn(run_loop(
            engine,
            input_rx,
            output_tx,
            shutdown.clone(),
            std::time::Duration::from_millis(200),
        ));

        shutdown.cancel();

        let out = tokio::time::timeout(std::time::Duration::from_secs(2), output_rx.recv())
            .await
            .expect("deadline abort timed out")
            .expect("deadline abort output");
        assert_eq!(out.outputs.outputs[0].request_id, "stuck-1");
        assert_eq!(
            out.outputs.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Abort)
        );

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), loop_handle)
            .await
            .expect("loop should exit after deadline abort")
            .expect("join");
        assert!(result.is_ok());
    }
}
