# Verifying release artifacts

Every GitHub Release tarball ships with four supply-chain artifacts:

- `*.sha256` — a plain checksum, no tooling required (`shasum -a 256 -c <file>.sha256`).
- `*.cdx.json` — a [CycloneDX](https://cyclonedx.org/) SBOM of the build's dependency graph.
- `*.sig` + `*.pem` — a [cosign](https://docs.sigstore.dev/) keyless signature and its
  Fulcio certificate, for offline verification.
- a [SLSA build provenance](https://slsa.dev/) attestation recorded in GitHub, binding the
  tarball's digest to the workflow run that produced it.

Verify provenance (proves it was built by this repo's release workflow):

```bash
gh attestation verify vllm-vcr-vllm0.23-x86_64-unknown-linux-musl.tar.gz \
  --repo neuralmagic/vllm-vcr
```

Verify the cosign signature without GitHub:

```bash
cosign verify-blob \
  --certificate vllm-vcr-vllm0.23-x86_64-unknown-linux-musl.tar.gz.pem \
  --signature  vllm-vcr-vllm0.23-x86_64-unknown-linux-musl.tar.gz.sig \
  --certificate-identity-regexp '^https://github.com/neuralmagic/vllm-vcr/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  vllm-vcr-vllm0.23-x86_64-unknown-linux-musl.tar.gz
```
