//! ZMQ IO loop: decode frontend frames into `EngineInput`, push `EngineOutput` back.
//! Same shape as vLLM's in-tree `vllm-mock-engine` IO loop, built entirely on the
//! public `vllm-engine-core-client` protocol API.

use anyhow::{Context as _, Result, anyhow, bail};
use futures::{Stream, StreamExt as _, stream};
use sim_protocol::mock_engine::MockEngineDataSockets;
use sim_protocol::wire::request_type_from_frame;
use tokio::sync::mpsc;
use tracing::warn;
use vllm_engine_core_client::protocol::{
    EngineCoreRequest, EngineCoreRequestType, decode_msgpack, encode_msgpack,
};
use zeromq::{DealerSocket, PushSocket, SocketRecv as _, SocketSend as _, ZmqMessage};

use crate::engine_core::{EngineInput, EngineOutput, UtilityRequestSpec};

/// Send one engine output batch to the client over the appropriate push socket.
async fn send_engine_outputs_to_client(
    push_sockets: &mut [PushSocket],
    EngineOutput {
        client_index,
        outputs,
    }: EngineOutput,
) -> Result<()> {
    let socket = match push_sockets.get_mut(client_index as usize) {
        Some(s) => s,
        None => {
            warn!(
                client_index,
                socket_count = push_sockets.len(),
                "client_index out of range, dropping output batch"
            );
            return Ok(());
        }
    };
    let message = ZmqMessage::from(encode_msgpack(&outputs)?);
    socket.send(message).await?;
    Ok(())
}

/// Continuously receive and decode messages from one dealer socket into `EngineInput`.
fn dealer_input_stream(dealer: DealerSocket) -> impl Stream<Item = Result<EngineInput>> {
    stream::unfold(dealer, |mut dealer| async {
        let input = loop {
            let message = match dealer.recv().await.context("dealer recv failed") {
                Ok(message) => message,
                Err(err) => break Err(err),
            };
            match decode_request(message) {
                Ok(input) => break Ok(input),
                Err(err) => warn!(%err, "failed to decode engine request; ignoring"),
            }
        };
        Some((input, dealer))
    })
}

/// Decode a `ZmqMessage` into an `EngineInput`.
fn decode_request(message: ZmqMessage) -> Result<EngineInput> {
    let frames = message.into_vec();
    if frames.is_empty() {
        bail!("empty engine request message");
    }
    if frames.len() != 2 {
        bail!("invalid frame count for engine request: {}", frames.len());
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = request_type_from_frame(request_type_frame) else {
        bail!("unknown engine request type: {:?}", request_type_frame);
    };

    let input = match request_type {
        EngineCoreRequestType::Add => {
            let request: Box<EngineCoreRequest> = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Request(request)
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Abort(request_ids)
        }
        EngineCoreRequestType::Utility => {
            let request: UtilityRequestSpec = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Utility(request)
        }
        EngineCoreRequestType::StartDpWave => EngineInput::StartDpWave,
    };

    Ok(input)
}

/// Run the IO loop: dealer frames -> `input_tx`, `output_rx` -> push sockets.
///
/// The loop has no shutdown signal of its own: it must keep delivering outputs while
/// the engine drains in-flight requests, so it exits only when the engine loop is done
/// (the output channel closes, or an input send finds the engine gone), after flushing
/// every queued output - the final tokens and abort notices of a graceful shutdown.
pub(crate) async fn run_io_loop(
    data_sockets: Vec<MockEngineDataSockets>,
    input_tx: mpsc::UnboundedSender<EngineInput>,
    mut output_rx: mpsc::UnboundedReceiver<EngineOutput>,
) -> Result<()> {
    let (dealers, mut push_sockets): (Vec<_>, Vec<_>) = data_sockets
        .into_iter()
        .map(|sockets| (sockets.dealer, sockets.push))
        .unzip();
    let mut input_streams =
        stream::select_all(dealers.into_iter().map(dealer_input_stream).map(Box::pin));

    loop {
        tokio::select! {
            biased;
            output = output_rx.recv() => {
                let Some(output) = output else {
                    // Engine loop exited and every queued output is flushed.
                    return Ok(());
                };
                send_engine_outputs_to_client(&mut push_sockets, output).await?;
            }

            input = input_streams.next() => {
                let input = input.ok_or_else(|| anyhow!("engine input streams closed"))??;
                if input_tx.send(input).is_err() {
                    // Engine loop exited between our select arms; flush and leave.
                    while let Some(output) = output_rx.recv().await {
                        send_engine_outputs_to_client(&mut push_sockets, output).await?;
                    }
                    return Ok(());
                }
            }
        }
    }
}
