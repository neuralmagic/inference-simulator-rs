#!/usr/bin/env bash
#
# gemma-4 image-in / text-out e2e smoke: prove a client gets coherent content back
# through the full frontend -> tap -> engine path with an INLINE image (no egress).
#
# The frontend fetches remote image_url server-side, which 403s through istio
# egress, so we embed a solid-red PNG as a base64 data: URI built inline (pure
# zlib, no PIL). Expect a 200 + a caption that mentions red.
#
# max_tokens is set explicitly on purpose: the engine is pinned to 16e91176, which
# predates #45417's "unset HF default max_new_tokens" fix.
set -euo pipefail

NS="${NS:-your-namespace}"
DEPLOY="${DEPLOY:-trace-capture-gemma4}"
MODEL="${MODEL:-RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic}"
PROMPT="${PROMPT:-What is the dominant color of this image? Answer in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-64}"

# Build a 64x64 solid-red PNG, base64 data URI, with stdlib zlib only.
DATA_URI="$(python3 - <<'PY'
import zlib, struct, base64
W = H = 64
def chunk(tag, data):
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
raw = b"".join(b"\x00" + b"\xff\x00\x00" * W for _ in range(H))  # filter byte + RGB red per row
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw, 9))
       + chunk(b"IEND", b""))
print("data:image/png;base64," + base64.b64encode(png).decode())
PY
)"

echo "==> port-forwarding $DEPLOY :8000"
kubectl -n "$NS" port-forward "deploy/$DEPLOY" 8000:8000 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

echo "==> /v1/models"
curl -sf http://127.0.0.1:8000/v1/models | python3 -m json.tool

echo "==> chat/completions with an inline image (non-streaming, eyeball the content)"
curl -sf http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'
{
  "model": "$MODEL",
  "max_tokens": $MAX_TOKENS,
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "$PROMPT"},
      {"type": "image_url", "image_url": {"url": "$DATA_URI"}}
    ]}
  ]
}
EOF

echo
echo "==> if that printed a coherent answer mentioning red, the e2e path works."
echo "    Trace is accumulating at /trace/trace.jsonl on the tap container."
