# NIXL data plane

The NIXL path needs `libnixl` + UCX installed (Linux; RDMA NICs or shared memory).
On a box without it, typecheck against stubs:

```bash
cargo check --features nixl-stub
```

On Linux with NIXL installed, split a prefill and a decode engine:

```bash
# prefill
cargo run --features nixl -- --pd-role prefill \
  --engine-id mock-prefill --side-channel-host 127.0.0.1 --side-channel-port 5600 ...

# decode
cargo run --features nixl -- --pd-role decode \
  --engine-id mock-decode --side-channel-port 5601 ...
```

The transfer path uses `remote_host`/`remote_port` from `kv_transfer_params` to fetch
the prefill's `PoolDescriptor` over TCP, then issues NIXL READs for the advertised
block ids. Decode receives those `remote_*` fields per request; the prefill address is
not a decode CLI argument. The loopback test validates the byte-transfer path;
Kubernetes deployment validation is separate.
