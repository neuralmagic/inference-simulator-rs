#!/usr/bin/env bash
#
# DiffusionGemma image-in / text-out e2e smoke: prove a client gets coherent
# content back through the full frontend -> tap -> engine path with an image in
# the request. Run AFTER the rig is ready (just dg-up / kubectl rollout status).
#
# Note: the PYTHON frontend fetches image_url server-side, so the frontend pod
# needs egress to the image host (istio egress is slow but works for plain
# HTTP). Swap to an inline data: URI if egress is blocked.
#
# max_tokens is set explicitly on purpose: we pinned the engine to eb28452,
# which predates #45417's "unset HF default max_new_tokens" fix.
set -euo pipefail

NS="${NS:-your-namespace}"
DEPLOY="${DEPLOY:-trace-capture-diffusiongemma}"
MODEL="${MODEL:-RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic}"
IMAGE_URL="${IMAGE_URL:-https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg}"

echo "==> port-forwarding $DEPLOY :8000"
kubectl -n "$NS" port-forward "deploy/$DEPLOY" 8000:8000 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

echo "==> /v1/models"
curl -sf http://127.0.0.1:8000/v1/models | python3 -m json.tool

echo "==> chat/completions with an image (non-streaming, eyeball the content)"
curl -sf http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'
{
  "model": "$MODEL",
  "max_tokens": 128,
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "Describe this image in one sentence."},
      {"type": "image_url", "image_url": {"url": "$IMAGE_URL"}}
    ]}
  ]
}
EOF

echo
echo "==> if that printed a coherent description of the image, the e2e path works."
echo "    Trace is accumulating at /trace/trace.jsonl on the tap container."
