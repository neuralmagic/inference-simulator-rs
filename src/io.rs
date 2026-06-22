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
    if frames.len() < 2 {
        bail!(
            "engine request needs at least 2 frames, got {}",
            frames.len()
        );
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = request_type_from_frame(request_type_frame) else {
        bail!("unknown engine request type: {:?}", request_type_frame);
    };

    let input = match request_type {
        EngineCoreRequestType::Add => {
            // The Python frontend's MsgpackEncoder splits large tensor payloads
            // (multimodal pixels, prompt embeds) out of the primary msgpack frame
            // into trailing aux frames, so a multimodal Add arrives as >2 frames.
            // The request decoder records those payloads as aux-frame indices, and
            // replay serves trace tokens regardless of prompt content, so decode
            // frame[1] and drop the aux frames (frames[2..]).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `Add` engine request message: type frame, the msgpack request,
    /// then `aux_frames` trailing aux frames (what the Python frontend appends
    /// for split tensor payloads).
    fn add_message(req: &EngineCoreRequest, aux_frames: usize) -> ZmqMessage {
        let mut msg = ZmqMessage::from(vec![0u8]); // 0 == Add
        msg.push_back(encode_msgpack(req).expect("encode request").into());
        for i in 0..aux_frames {
            msg.push_back(vec![0xABu8; i + 1].into());
        }
        msg
    }

    #[test]
    fn decodes_text_add_with_exactly_two_frames() {
        let req = EngineCoreRequest {
            request_id: "req-text".to_string(),
            ..Default::default()
        };
        let EngineInput::Request(decoded) =
            decode_request(add_message(&req, 0)).expect("decode 2-frame add")
        else {
            panic!("expected EngineInput::Request");
        };
        assert_eq!(decoded.request_id, "req-text");
    }

    #[test]
    fn decodes_multimodal_add_ignoring_trailing_aux_frames() {
        // A multimodal Add arrives as >2 frames (frame[1] msgpack + aux tensor
        // frames). Decode must read frame[1] and ignore the aux frames.
        let req = EngineCoreRequest {
            request_id: "req-mm".to_string(),
            ..Default::default()
        };
        let EngineInput::Request(decoded) =
            decode_request(add_message(&req, 3)).expect("decode 5-frame multimodal add")
        else {
            panic!("expected EngineInput::Request");
        };
        assert_eq!(decoded.request_id, "req-mm");
    }

    #[test]
    fn rejects_messages_with_fewer_than_two_frames() {
        assert!(decode_request(ZmqMessage::from(vec![0u8])).is_err());
    }
}
