//! The EngineCore trait boundary and the generic engine loop.
//!
//! `EngineCore` is the contract any engine implementation must satisfy to plug into the
//! ZMQ IO loop. `SimEngine` (in `engine.rs`) is the only implementation today, but the
//! trait lets us swap in alternative backends without touching the loop or transport.

use anyhow::{Result, anyhow};
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::utility::EngineCoreUtilityRequest;
use vllm_engine_core_client::protocol::{EngineCoreOutputs, EngineCoreRequest};

/// Message sent from the IO loop to the engine task to drive the engine loop.
pub(crate) enum EngineInput {
    Request(Box<EngineCoreRequest>),
    Abort(Vec<String>),
    Utility(EngineCoreUtilityRequest),
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
}

/// Run the main loop for one engine, receiving `EngineInput` and sending `EngineOutput`
/// until `shutdown` is cancelled. The engine must already be constructed; construction
/// stays with the caller.
///
/// Any `EngineCore` impl reuses this loop unchanged: the loop owns the select! over
/// inputs, internal events, and deadline-driven steps, so custom engines only need to
/// implement the trait, not duplicate the concurrency harness.
pub(crate) async fn run_loop<E: EngineCore>(
    mut engine: E,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::Sender<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut internal_rx = engine.take_internal_rx();

    loop {
        let next_deadline = engine.earliest_deadline();
        let outputs = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,

            input = input_rx.recv() => {
                let input = input.ok_or_else(|| anyhow!("engine input channel closed"))?;
                engine.handle_input(input)?
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

        for output in outputs {
            output_tx
                .send(output)
                .await
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
    }

    #[tokio::test]
    async fn constant_engine_through_run_loop() {
        let engine = ConstantEngine::new(42);
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, mut output_rx) = mpsc::channel(16);
        let shutdown = CancellationToken::new();

        let shutdown_clone = shutdown.clone();
        let loop_handle = tokio::spawn(run_loop(engine, input_rx, output_tx, shutdown_clone));

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
}
