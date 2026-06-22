# NIXL data plane

The NIXL path is optional. It needs `libnixl` and UCX on Linux, using either RDMA
NICs or shared-memory transports. On a development machine without that runtime,
typecheck the code path against stubs:

```bash
cargo check --features nixl-stub
```

On Linux with NIXL installed, split a prefill and a decode engine:

```bash
# prefill
cargo run --features nixl -- play --pd-role prefill \
  --engine-id mock-prefill --side-channel-host 127.0.0.1 --side-channel-port 5600 ...

# decode
cargo run --features nixl -- play --pd-role decode \
  --engine-id mock-decode --side-channel-port 5601 ...
```

The transfer path uses `remote_host` and `remote_port` from `kv_transfer_params` to
fetch the prefill's `PoolDescriptor` over a TCP metadata side channel, then issues
NIXL READs for the advertised block ids. Decode receives those `remote_*` fields per
request; the prefill address is not a decode CLI argument.

`tests/nixl_loopback.rs` validates the byte-transfer path with distinct prefill and
decode agents in one process. Kubernetes deployment validation is separate.
