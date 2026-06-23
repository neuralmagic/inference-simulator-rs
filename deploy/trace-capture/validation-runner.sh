#!/bin/bash
# In-pod load driver for Kueue conformance Jobs (generated from models.toml).
# Runs PHASES against localhost:8000; idles until trace is fetched (2h timeout).
set -euo pipefail

pip install --quiet httpx

echo "==> waiting for frontend on :8000 (engine startup includes the weight download)"
python - <<'EOF'
import time
import urllib.request

while True:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
        break
    except Exception:
        time.sleep(5)
EOF

loadgen() {
    python /scripts/loadgen.py --url http://127.0.0.1:8000 --model "${MODEL:-Qwen/Qwen3-8B}" "$@"
}

mark() {
    wc -l </trace/trace.jsonl | tr -d ' ' >"/trace/marker-$1" 2>/dev/null || echo 0 >"/trace/marker-$1"
}

for phase in $PHASES; do
    case "$phase" in
    sweep)
        # ~1.54 wire tokens per synthetic word: spans ~0.8k-12.3k tokens across
        # the latency model's prompt buckets, at idle and loaded concurrency.
        for words in ${SWEEP_WORDS:-512 1000 1500 3000 5500 8000}; do
            for conc in ${SWEEP_CONC:-1 8}; do
                secs=75
                [ "$conc" = 1 ] && secs=45
                echo "==> sweep prompt=$words words c$conc (${secs}s)"
                loadgen --pattern constant --concurrency "$conc" --duration "$secs" \
                    --prompt-tokens "$words" --output-tokens 128 \
                    --out "/trace/sweep-p$words-c$conc.json"
            done
        done
        ;;
    multiturn)
        echo "==> multiturn 0.4 sessions/s x 5 turns, ~10k-token shared prefix (240s)"
        loadgen --pattern multiturn --rate "${MT_RATE:-0.4}" --turns 5 \
            --prefix-tokens 6500 --prompt-tokens 128 --output-tokens 128 \
            --duration 240 --seed "${MT_SEED:-7}" \
            --out /trace/multiturn-loadgen.json
        ;;
    *)
        echo "unknown phase: $phase" >&2
        exit 1
        ;;
    esac
    mark "$phase"
done

touch /trace/loadgen-done
echo "==> capture done; waiting for fetch (kubectl exec ... touch /trace/fetched), max 2h"
for _ in $(seq 1440); do
    [ -f /trace/fetched ] && break
    sleep 5
done
echo "==> exiting"
