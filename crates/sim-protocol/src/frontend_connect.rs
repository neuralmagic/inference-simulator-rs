//! Frontend handshake with a caller-supplied registration payload.
//!
//! Mirrors `vllm_engine_core_client::mock_engine::connect_to_frontend`, with
//! one difference: the engine-ready registration payload (the bytes sent on
//! input-socket registration) is raw, supplied by the caller, instead of an
//! encoded `EngineCoreReadyResponse`. The crate's struct lags the python
//! schema (it is missing `block_size`, which python's
//! `EngineCoreReadyResponse` requires), so re-encoding through it silently
//! drops fields and the python frontend rejects the registration. With raw
//! payloads, the tap relays the real engine's response verbatim (immune to
//! any future schema drift) and the sim emits its own complete
//! [`SimReadyResponse`].

use std::path::Path;
use std::time::Duration;

use crate::mock_engine::{MockCoordinatorSockets, MockEngineDataSockets, MockEngineSockets};
use anyhow::{Context as _, Result, anyhow, bail};
use serde::Serialize;
use tokio::time::timeout;
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::protocol::handshake::{HandshakeInitMessage, ReadyMessage};
use vllm_engine_core_client::protocol::{ModelDtype, decode_msgpack, encode_msgpack};
use zeromq::prelude::{Socket as _, SocketRecv as _, SocketSend as _};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, SubSocket, ZmqMessage};

/// The engine-ready registration response, matching python vLLM's
/// `EngineCoreReadyResponse` field-for-field (msgpack map encoding, so all six
/// keys must be present). This is the sim-owned superset of the crate's
/// struct, which lags the schema.
#[derive(Debug, Clone, Serialize)]
pub struct SimReadyResponse {
    pub max_model_len: u64,
    pub num_gpu_blocks: u64,
    /// Tokens per KV block. Required by the python frontend; absent from the
    /// crate's `EngineCoreReadyResponse`.
    pub block_size: u64,
    pub dp_stats_address: Option<String>,
    pub dtype: ModelDtype,
    pub vllm_version: String,
}

impl SimReadyResponse {
    pub fn encode(&self) -> Result<Vec<u8>> {
        encode_msgpack(self).map_err(|e| anyhow!("encoding ready response: {e}"))
    }
}

/// Wait for an IPC endpoint path to appear before attempting to connect.
/// TCP endpoints pass through immediately.
async fn wait_for_ipc_endpoint(endpoint: &str, connect_timeout: Duration) -> Result<()> {
    let Some(socket_path) = endpoint.strip_prefix("ipc://") else {
        return Ok(());
    };
    timeout(connect_timeout, async {
        while !Path::new(socket_path).exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .with_context(|| format!("waiting for IPC endpoint {endpoint}"))
}

fn status_message(status: &str, local: bool, headless: bool) -> ReadyMessage {
    ReadyMessage {
        status: Some(status.to_string()),
        local: Some(local),
        headless: Some(headless),
        parallel_config_hash: None,
    }
}

/// Join a frontend-owned handshake endpoint and open engine data sockets,
/// registering with `ready_response_payload` verbatim.
pub async fn connect_to_frontend_raw(
    frontend_handshake: &str,
    engine_id: EngineId,
    local: bool,
    headless: bool,
    ready_response_payload: &[u8],
    connect_timeout: Duration,
) -> Result<MockEngineSockets> {
    wait_for_ipc_endpoint(frontend_handshake, connect_timeout).await?;

    let peer_identity = PeerIdentity::try_from(engine_id.clone())
        .map_err(|e| anyhow!("invalid engine identity {:?}: {e}", engine_id.to_vec()))?;
    let mut options = SocketOptions::default();
    options.peer_identity(peer_identity.clone());
    let mut handshake = DealerSocket::with_options(options);
    handshake
        .connect(frontend_handshake)
        .await
        .with_context(|| format!("connecting to frontend handshake {frontend_handshake}"))?;
    handshake
        .send(ZmqMessage::from(
            encode_msgpack(&status_message("HELLO", local, headless))
                .map_err(|e| anyhow!("encoding HELLO: {e}"))?,
        ))
        .await
        .context("sending HELLO to frontend")?;

    let init_frames = handshake
        .recv()
        .await
        .context("receiving INIT from frontend")?
        .into_vec();
    if init_frames.len() != 1 {
        bail!(
            "expected one INIT frame from frontend, got {}",
            init_frames.len()
        );
    }
    let init: HandshakeInitMessage =
        decode_msgpack(init_frames[0].as_ref()).map_err(|e| anyhow!("decoding INIT: {e}"))?;

    if init.addresses.inputs.is_empty() {
        bail!("frontend INIT did not include an input address");
    }
    if init.addresses.inputs.len() != init.addresses.outputs.len() {
        bail!(
            "frontend INIT input/output address count mismatch: {} inputs, {} outputs",
            init.addresses.inputs.len(),
            init.addresses.outputs.len()
        );
    }

    let mut data_sockets = Vec::with_capacity(init.addresses.inputs.len());
    for (input_address, output_address) in init
        .addresses
        .inputs
        .iter()
        .zip(init.addresses.outputs.iter())
    {
        wait_for_ipc_endpoint(input_address, connect_timeout).await?;
        wait_for_ipc_endpoint(output_address, connect_timeout).await?;

        let mut input_options = SocketOptions::default();
        input_options.peer_identity(peer_identity.clone());
        let mut dealer = DealerSocket::with_options(input_options);
        dealer
            .connect(input_address)
            .await
            .with_context(|| format!("connecting input dealer to {input_address}"))?;
        dealer
            .send(ZmqMessage::from(ready_response_payload.to_vec()))
            .await
            .context("registering on frontend input socket")?;

        let mut push = PushSocket::new();
        push.connect(output_address)
            .await
            .with_context(|| format!("connecting output push to {output_address}"))?;

        data_sockets.push(MockEngineDataSockets { dealer, push });
    }

    let coordinator = match (
        init.addresses.coordinator_input.as_deref(),
        init.addresses.coordinator_output.as_deref(),
    ) {
        (Some(coordinator_input), Some(coordinator_output)) => {
            let mut input_sub = SubSocket::new();
            input_sub
                .connect(coordinator_input)
                .await
                .context("connecting coordinator sub")?;
            input_sub
                .subscribe("")
                .await
                .context("subscribing coordinator sub")?;

            let mut output_push = PushSocket::new();
            output_push
                .connect(coordinator_output)
                .await
                .context("connecting coordinator push")?;

            let ready = input_sub
                .recv()
                .await
                .context("receiving coordinator READY")?
                .into_vec();
            if ready.len() != 1 || ready[0].as_ref() != b"READY" {
                bail!("expected coordinator READY marker, got {ready:?}");
            }

            Some(MockCoordinatorSockets {
                input_sub,
                output_push,
            })
        }
        (None, None) => None,
        _ => bail!("coordinator handshake addresses must be both present or both absent"),
    };

    handshake
        .send(ZmqMessage::from(
            encode_msgpack(&status_message("READY", local, headless))
                .map_err(|e| anyhow!("encoding READY: {e}"))?,
        ))
        .await
        .context("sending READY to frontend")?;

    Ok(MockEngineSockets {
        init,
        data_sockets,
        coordinator,
    })
}

#[cfg(test)]
mod tests {
    use crate::frontend_connect::SimReadyResponse;
    use vllm_engine_core_client::protocol::ModelDtype;

    /// The python frontend decodes the registration payload into a msgspec
    /// dataclass with six required fields; every key must be present in the
    /// msgpack map.
    #[test]
    fn sim_ready_response_carries_all_python_required_fields() {
        let payload = SimReadyResponse {
            max_model_len: 32768,
            num_gpu_blocks: 1000,
            block_size: 16,
            dp_stats_address: None,
            dtype: ModelDtype::Float32,
            vllm_version: "test".to_string(),
        }
        .encode()
        .expect("encode");
        let decoded = rmpv::decode::read_value(&mut &payload[..]).expect("decode");
        let map = decoded.as_map().expect("map-keyed encoding");
        let keys: Vec<&str> = map.iter().filter_map(|(k, _)| k.as_str()).collect();
        for required in [
            "max_model_len",
            "num_gpu_blocks",
            "block_size",
            "dp_stats_address",
            "dtype",
            "vllm_version",
        ] {
            assert!(keys.contains(&required), "missing field {required}");
        }
    }
}
