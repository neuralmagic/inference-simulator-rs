#!/usr/bin/env python3
"""Minimal streaming load generator for tap captures.

Drives an OpenAI-compatible /v1/completions endpoint at fixed concurrency with
synthetic fixed-size prompts, measuring client-side TTFT and mean ITL per
request (the same mean-only view guidellm reports, computed the same way:
(last_token_time - first_token_time) / (n - 1)).

The server-side truth comes from the tap; this exists so the same run yields
both views without depending on guidellm's scheduler.

Usage:
  uv run --with httpx loadgen.py --url http://127.0.0.1:8000 \
      --model Qwen/Qwen3-8B --concurrency 16 --duration 120 \
      --prompt-tokens 512 --output-tokens 128 --out c16.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

import httpx

# ~1 token/word for common BPE vocabs; the tap records the true prompt_tokens
# from the wire, so approximate sizing here is fine.
WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


def make_prompt(rng: random.Random, n_tokens: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


async def one_request(
    client: httpx.AsyncClient, args: argparse.Namespace, rng: random.Random, run_start: float
) -> dict | None:
    prompt = make_prompt(rng, args.prompt_tokens)
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "ignore_eos": True,
        "stream": True,
    }
    start = time.perf_counter()
    stamps: list[float] = []
    try:
        async with client.stream("POST", f"{args.url}/v1/completions", json=body) as r:
            if r.status_code != 200:
                return {"error": f"http {r.status_code}"}
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                stamps.append(time.perf_counter())
    except httpx.HTTPError as e:
        return {"error": str(e)}
    if not stamps:
        return {"error": "no tokens"}
    first, last = stamps[0], stamps[-1]
    chunks = len(stamps)
    itl_ms = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    return {
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": chunks,
        "ttft_ms": (first - start) * 1000.0,
        "itl_mean_ms": ((last - first) / (chunks - 1)) * 1000.0 if chunks > 1 else None,
        "itl_ms": itl_ms,
        "concurrency": args.concurrency,
        "arrival_ms": (start - run_start) * 1000.0,
    }


async def worker(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    deadline: float,
    results: list,
    seed: int,
    run_start: float,
) -> None:
    rng = random.Random(seed)
    while time.perf_counter() < deadline:
        res = await one_request(client, args, rng, run_start)
        if res is not None:
            results.append(res)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--trace-out",
        help="also append records in the inference-sim trace JSONL schema "
        "(client-side measurements; writes a meta line if the file is new)",
    )
    args = p.parse_args()

    results: list[dict] = []
    run_start = time.perf_counter()
    deadline = run_start + args.duration
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        await asyncio.gather(
            *(
                worker(client, args, deadline, results, seed, run_start)
                for seed in range(args.concurrency)
            )
        )

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": ok, "errors": errs}, f, indent=1)

    if args.trace_out:
        new_file = not os.path.exists(args.trace_out) or os.path.getsize(args.trace_out) == 0
        with open(args.trace_out, "a") as f:
            if new_file:
                f.write(json.dumps({"meta": {"model": args.model, "source": "loadgen-client"}}) + "\n")
            for r in ok:
                rec = {
                    "prompt_tokens": r["prompt_tokens"],
                    "cached_tokens": 0,
                    "output_tokens": r["output_tokens"],
                    "ttft_ms": r["ttft_ms"],
                    "itl_ms": r["itl_ms"],
                    "concurrency": r["concurrency"],
                    # Relative to this invocation's start; appended runs restart at 0.
                    "arrival_ms": r["arrival_ms"],
                }
                f.write(json.dumps(rec) + "\n")
    itls = sorted(r["itl_mean_ms"] for r in ok if r["itl_mean_ms"] is not None)
    ttfts = sorted(r["ttft_ms"] for r in ok)

    def pct(v: list[float], q: float) -> float:
        return v[min(int(q * len(v)), len(v) - 1)] if v else 0.0

    print(
        f"done: {len(ok)} ok, {len(errs)} errors | "
        f"ttft p50/p99 {pct(ttfts, 0.5):.0f}/{pct(ttfts, 0.99):.0f} ms | "
        f"itl-mean p50/p99 {pct(itls, 0.5):.1f}/{pct(itls, 0.99):.1f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
