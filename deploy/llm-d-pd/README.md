# GPU-free llm-d P/D deployment

A self-contained copy of the llm-d v0.7.0 prefill/decode deployment, repointed
at the `vllm-vcr` image. It stands up the llm-d control plane
(Router/EPP, InferencePool, routing sidecar) in front of GPU-free model servers
that run the vLLM Rust frontend over this simulator. The image includes libnixl
and UCX for CPU NIXL transfers. No GPUs or model weights are required.

```
client ─▶ Router/EPP ─▶ decode pod ─ routing-sidecar(nixlv2) ─┬─▶ prefill pod (vllm-rs + mock-engine, NIXL :5600)
                                                               └─▶ decode  pod (vllm-rs + mock-engine) ── NIXL READ ─┘
```

What's vendored (from the llm-d guide, adapted to CPU + our image):

- `router/` — the Router base + pd-disaggregation EPP/InferencePool values.
- `modelserver/` — prefill + decode Deployments (our image, no GPU, small model) + the
real `llm-d-routing-sidecar` on decode.
- `helmfile.yaml` — installs the Router chart + applies the model servers.

## Prerequisites

- Build & push the image (from the repo root):
  ```bash
  podman build -t ghcr.io/neuralmagic/vllm-vcr:dev .
  podman push ghcr.io/neuralmagic/vllm-vcr:dev
  ```
  If you use a custom tag, update `modelserver/kustomization.yaml` before applying.
- A cluster with the llm-d control-plane deps:
  ```bash
  kubectl config use-context <your-cluster>
  ```
- Install the Gateway API Inference Extension CRDs and create the namespace:
  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.5.0/v1-manifests.yaml
  kubectl create namespace llm-d-pd-mock
  ```

## Deploy

```bash
helmfile -f deploy/llm-d-pd/helmfile.yaml apply
kubectl -n llm-d-pd-mock get pods   # mock-pd-prefill (x2), mock-pd-decode (x1), EPP
```

## Test

```bash
IP=$(kubectl get service pd-disaggregation-epp -n llm-d-pd-mock -o jsonpath='{.spec.clusterIP}')
kubectl run curl-debug --rm -it --image=cfmanteiga/alpine-bash-curl-jq --env="IP=$IP" -- \
  curl -sS -X POST http://${IP}/v1/completions -H 'Content-Type: application/json' -d '{
    "model": "Qwen/Qwen3-0.6B", "prompt": "How are you today?", "max_tokens": 16 }' | jq
```

Check data-plane logs:

```bash
kubectl -n llm-d-pd-mock logs -l llm-d.ai/role=prefill -c modelserver | grep "advertised KV"
kubectl -n llm-d-pd-mock logs -l llm-d.ai/role=decode  -c modelserver | grep "pulled remote KV"
```

## Cleanup

```bash
helmfile -f deploy/llm-d-pd/helmfile.yaml destroy
kubectl delete namespace llm-d-pd-mock
```

