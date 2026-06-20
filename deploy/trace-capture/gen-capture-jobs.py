#!/usr/bin/env python3
"""Generate conformance-capture Kueue Jobs from models.toml.

Emits Kubernetes manifests as JSON (a superset of YAML, so `kubectl apply -f -` takes
it) using only the stdlib. One Job per selected [[capture]] target; the scenario drives
the engine and tap flags so the config_hash and the engine agree on prefix-cache and
spec-decode. Replaces the hand-maintained validation-jobs.yaml.

Usage:
  gen-capture-jobs.py --list                       # names + scenarios
  gen-capture-jobs.py <name> [<name> ...]          # selected targets
  gen-capture-jobs.py --all                        # every target
  gen-capture-jobs.py qwen3-8b | kubectl apply -f -

The pod is the same sidecar stack as before (engine + tap + frontend, loadgen as the
main container); only the per-target knobs change.
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

MANIFEST = Path(__file__).with_name("models.toml")

# Scalar keys a [[capture]] may inherit from [defaults] (or override).
INHERITED = (
    "namespace queue service_account gpu gpu_memory_utilization tp block_size "
    "max_num_seqs max_model_len enforce_eager engine_cpu_request engine_cpu_limit "
    "engine_memory_request engine_memory_limit model_cache_size"
).split()


def spec_config_json(descriptor: str) -> str:
    """Map a speculative descriptor (config_hash input) to vLLM's --speculative-config."""
    m = re.fullmatch(r"ngram-k(\d+)", descriptor)
    if m:
        return json.dumps(
            {
                "method": "ngram",
                "num_speculative_tokens": int(m.group(1)),
                "prompt_lookup_max": 4,
                "prompt_lookup_min": 2,
            }
        )
    sys.exit(f"unknown speculative descriptor {descriptor!r} (extend spec_config_json)")


def engine_args(c: dict) -> list[str]:
    args = [
        c["model"],
        "--headless",
        "--data-parallel-address=127.0.0.1",
        "--data-parallel-rpc-port=5580",
        "--data-parallel-size=1",
        f"--tensor-parallel-size={c['tp']}",
        f"--gpu-memory-utilization={c['gpu_memory_utilization']}",
        f"--max-model-len={c['max_model_len']}",
        f"--max-num-seqs={c['max_num_seqs']}",
    ]
    if c.get("enforce_eager", True):
        args.append("--enforce-eager")
    scenario = c["scenario"]
    if scenario == "nocache":
        args.append("--no-enable-prefix-caching")
    elif scenario == "specdecode":
        args += ["--speculative-config", spec_config_json(c["speculative"])]
    return args


def canonical_engine_config(c: dict) -> str:
    """The deployed behavioral engine flags, canonical (sorted key=value, ';'-joined).

    This is the config_hash's `engine_config` digest input (config_hash.rs v3). Only
    behavioral knobs go in (no transport/addressing); the sort makes it order-stable.
    """
    fields = {
        "model": c["model"],
        "tensor_parallel": c["tp"],
        "gpu_memory_utilization": c["gpu_memory_utilization"],
        "max_model_len": c["max_model_len"],
        "max_num_seqs": c["max_num_seqs"],
        "block_size": c["block_size"],
        "enforce_eager": bool(c.get("enforce_eager", True)),
        "enable_prefix_caching": c["scenario"] != "nocache",
        "speculative": c.get("speculative", "none") if c["scenario"] == "specdecode" else "none",
    }

    def fmt(v: object) -> str:
        return "true" if v is True else "false" if v is False else str(v)

    return ";".join(f"{k}={fmt(fields[k])}" for k in sorted(fields))


def tap_args(c: dict) -> list[str]:
    # --model/--gpu/--tp/--block-size/--max-num-seqs are recorded in the trace meta for
    # readability; the config_hash itself is gpu + vllm_tag + the engine_config digest.
    return [
        "--frontend-handshake=tcp://127.0.0.1:5570",
        "--engine-handshake=tcp://127.0.0.1:5580",
        "--input-address=tcp://127.0.0.1:29560",
        "--output-address=tcp://127.0.0.1:29561",
        "--trace-out=/trace/trace.jsonl",
        f"--model={c['model']}",
        f"--gpu={c['gpu']}",
        f"--tp={c['tp']}",
        f"--block-size={c['block_size']}",
        f"--vllm-version={c['vllm_tag']}",
        f"--max-num-seqs={c['max_num_seqs']}",
        f"--engine-config={canonical_engine_config(c)}",
    ]


def env(pairs: dict) -> list[dict]:
    return [{"name": k, "value": v} for k, v in pairs.items()]


def build_job(c: dict, lines: dict) -> dict:
    tag = c["vllm_tag"]
    if tag not in lines:
        sys.exit(f"capture {c['name']!r} targets unknown line {tag!r} (add [lines.{tag!r}])")
    line = lines[tag]
    label_model = c["model"].split("/")[-1]  # k8s label can't hold the '/'
    sidecar = {"restartPolicy": "Always"}
    # The tap/frontend image is a floating per-line tag (:vllm<line>); a node may
    # cache a stale build under it, so force a re-pull. The engine is digest-pinned
    # (immutable), so it keeps the default IfNotPresent.
    floating = {**sidecar, "imagePullPolicy": "Always"}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"trace-{c['name']}",
            "namespace": c["namespace"],
            "labels": {
                "kueue.x-k8s.io/queue-name": c["queue"],
                "llm-d.ai/guide": "trace-capture",
            },
        },
        "spec": {
            "suspend": True,  # Kueue unsuspends on admission
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {
                        "llm-d.ai/guide": "trace-capture",
                        "llm-d.ai/model": label_model,
                        "llm-d.ai/scenario": c["scenario"],
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": c["service_account"],
                    "initContainers": [
                        {
                            "name": "engine",
                            **sidecar,
                            "image": line["engine_image"],
                            "command": ["vllm", "serve"],
                            "args": engine_args(c),
                            "env": env(
                                {
                                    "HF_HUB_CACHE": "/models",
                                    "HF_HUB_DISABLE_XET": "1",
                                    "VLLM_RPC_TIMEOUT": "7200000",
                                }
                            ),
                            "resources": {
                                "requests": {
                                    "cpu": c["engine_cpu_request"],
                                    "memory": c["engine_memory_request"],
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": c["engine_cpu_limit"],
                                    "memory": c["engine_memory_limit"],
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "volumeMounts": [
                                {"mountPath": "/dev/shm", "name": "shm"},
                                {"mountPath": "/.cache", "name": "torch-compile-cache"},
                                {"mountPath": "/.triton", "name": "triton-cache"},
                                {"mountPath": "/.config", "name": "vllm-config"},
                                {"mountPath": "/models", "name": "model-cache"},
                            ],
                        },
                        {
                            "name": "tap",
                            **floating,
                            "image": line["tap_image"],
                            "command": ["/usr/local/bin/vllm-vcr", "record"],
                            "args": tap_args(c),
                            "env": env({"RUST_LOG": "info"}),
                            "resources": {
                                "requests": {"cpu": "2", "memory": "1Gi"},
                                "limits": {"cpu": "4", "memory": "2Gi"},
                            },
                            "volumeMounts": [{"mountPath": "/trace", "name": "trace"}],
                        },
                        {
                            "name": "frontend",
                            **floating,
                            "image": line["tap_image"],
                            "command": ["/usr/local/bin/vllm-rs", "serve"],
                            "args": [
                                c["model"],
                                "--data-parallel-size=1",
                                "--data-parallel-size-local=0",
                                "--handshake-port=5570",
                                "--host=0.0.0.0",
                                "--port=8000",
                            ],
                            "env": env({"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}),
                            "resources": {
                                "requests": {"cpu": "4", "memory": "8Gi"},
                                "limits": {"cpu": "8", "memory": "16Gi"},
                            },
                        },
                    ],
                    "containers": [
                        {
                            "name": "loadgen",
                            "image": "python:3.12-slim",
                            "command": ["bash", "/scripts/runner.sh"],
                            "env": env({"PHASES": c["phases"], "MODEL": c["model"]}),
                            "resources": {
                                "requests": {"cpu": "2", "memory": "2Gi"},
                                "limits": {"cpu": "4", "memory": "4Gi"},
                            },
                            "volumeMounts": [
                                {"mountPath": "/scripts", "name": "scripts"},
                                {"mountPath": "/trace", "name": "trace"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
                        {"name": "torch-compile-cache", "emptyDir": {}},
                        {"name": "triton-cache", "emptyDir": {}},
                        {"name": "vllm-config", "emptyDir": {}},
                        {"name": "model-cache", "emptyDir": {"sizeLimit": c["model_cache_size"]}},
                        {"name": "trace", "emptyDir": {}},
                        {"name": "scripts", "configMap": {"name": "validation-scripts"}},
                    ],
                },
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate conformance capture Jobs.")
    ap.add_argument("names", nargs="*", help="capture target names to emit")
    ap.add_argument("--all", action="store_true", help="emit every capture target")
    ap.add_argument("--list", action="store_true", help="list target names + scenarios")
    args = ap.parse_args()

    with open(MANIFEST, "rb") as f:
        m = tomllib.load(f)
    defaults = m.get("defaults", {})
    lines = m.get("lines", {})
    captures = m.get("capture", [])
    by_name = {c["name"]: c for c in captures}

    if args.list:
        for c in captures:
            print(f"{c['name']:24} {c['scenario']:11} {c['vllm_tag']:9} {c['model']}")
        return

    if args.all:
        selected = captures
    elif args.names:
        missing = [n for n in args.names if n not in by_name]
        if missing:
            sys.exit(f"unknown capture target(s): {', '.join(missing)} (try --list)")
        selected = [by_name[n] for n in args.names]
    else:
        ap.error("give capture name(s), --all, or --list")

    jobs = [build_job({**defaults, **c}, lines) for c in selected]
    out = jobs[0] if len(jobs) == 1 else {"apiVersion": "v1", "kind": "List", "items": jobs}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
