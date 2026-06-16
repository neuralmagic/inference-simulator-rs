#!/usr/bin/env python3
"""Emit GROUND-TRUTH vLLM wire bytes for a multimodal EngineCoreRequest.

Runs inside the vLLM image (CPU only, no GPU, no engine, no model weights) so it
is immune to the disaggregated-rig flakiness and the gpu-pruner. It builds a
realistic image request, runs vLLM's REAL `MsgpackEncoder`, and prints each ZMQ
frame as hex. The primary msgpack frame (frame 0) is exactly what the tap's
`decode_msgpack::<EngineCoreRequest>` must decode; the rest are aux tensor
frames. Feed the hex into the Rust decoder to settle whether the serde model
actually round-trips a real multimodal request (large-tensor aux-frame path),
not just the synthetic inline fixture the crate's python_compat test covers.

Usage (in pod): VLLM_ALLOW_INSECURE_SERIALIZATION=1 python3 mm_encode_groundtruth.py
"""

import dataclasses
import sys

import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalBatchedField,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackEncoder


def build_elem(key: str, data, field) -> MultiModalFieldElem:
    """Construct a MultiModalFieldElem regardless of its exact field set across
    vLLM revs, by inspecting the dataclass at runtime."""
    names = {f.name for f in dataclasses.fields(MultiModalFieldElem)}
    kwargs = {"data": data, "field": field}
    if "modality" in names:
        kwargs["modality"] = "image"
    if "key" in names:
        kwargs["key"] = key
    return MultiModalFieldElem(**kwargs)


def dump(label: str, request: EngineCoreRequest) -> None:
    enc = MsgpackEncoder()
    frames = enc.encode(request)
    frames = [bytes(f) for f in frames]
    print(f"### {label}")
    print(f"frames={len(frames)} sizes={[len(f) for f in frames]}")
    print(f"BLOB0_HEX {frames[0].hex()}")
    for i, f in enumerate(frames[1:], start=1):
        # aux frames are raw tensor bytes; print only a short prefix + len.
        print(f"AUX{i}_LEN {len(f)} prefix={f[:16].hex()}")
    print()


def make_request(rid: str, tensor: torch.Tensor) -> EngineCoreRequest:
    # A flat field with one slice covering the whole tensor (typical for
    # pixel_values), plus the placeholder range the prompt reserves for it.
    flat = MultiModalFlatField(slices=[slice(0, tensor.shape[0])], dim=0)
    elem = build_elem("pixel_values", tensor, flat)
    # second field: a small batched int-ish tensor (e.g. num_patches metadata)
    small = torch.tensor([tensor.shape[0]], dtype=torch.int64)
    elem2 = build_elem("image_grid_thw", small, MultiModalBatchedField())
    # MultiModalKwargsItem is a UserDict keyed by the model kwarg name.
    item = MultiModalKwargsItem({"pixel_values": elem, "image_grid_thw": elem2})

    spec = MultiModalFeatureSpec(
        data=item,
        modality="image",
        identifier="encoder-cache-key",
        mm_position=PlaceholderRange(offset=4, length=256),
        mm_hash="processor-hash",
    )
    return EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=[2, 108] + [262144] * 256 + [108],  # text + image placeholders
        mm_features=[spec],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def main() -> int:
    # Large tensor (> 256 B threshold) -> rides as an AUX frame (aux-index int in
    # the blob). This is the path the crate's compat test never exercises.
    big = torch.zeros((1024, 1176), dtype=torch.bfloat16)  # ~2.3 MB
    dump("large-tensor-aux-frame", make_request("req-mm-large", big))

    # Small tensor (< 256 B) -> inlined as Ext(3) in the blob, single frame.
    small = torch.zeros((2, 2), dtype=torch.float32)  # 16 B
    dump("small-tensor-inline", make_request("req-mm-small", small))

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
