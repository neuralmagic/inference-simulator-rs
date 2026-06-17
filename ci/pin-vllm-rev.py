#!/usr/bin/env python3
"""Pin Cargo.toml's vllm-engine-core-client to a compat.toml line, for the CI matrix.

Cargo rejects patching a git dependency to a different rev of the SAME source, so a
per-line build can't use `--config 'patch...rev=...'` (the approach in an earlier
draft of docs/versioning.md). Instead this swaps the rev directly in
[workspace.dependencies]. The optional fork [patch] (a DIFFERENT source, e.g. the
wseaton/vllm fork carrying vllm-project/vllm#45848) is rewritten to the line's
patch_repo/patch_rev, or removed when the line has no fork (it then builds against
protocol_rev upstream).

After running this the rev no longer matches Cargo.lock, so build/test must NOT pass
--locked on matrix legs.

Usage: ci/pin-vllm-rev.py <line>
"""

import re
import sys
import tomllib

VLLM_GIT = "https://github.com/vllm-project/vllm.git"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: pin-vllm-rev.py <line>")
    line = sys.argv[1]

    with open("compat.toml", "rb") as f:
        lines = tomllib.load(f).get("vllm", [])
    entry = next((v for v in lines if v["line"] == line), None)
    if entry is None:
        sys.exit(f"no compat.toml entry for line {line}")

    with open("Cargo.toml", encoding="utf-8") as f:
        text = f.read()

    # 1. Swap the base dependency rev (the vllm-project/vllm declaration, never the
    #    fork line inside [patch]).
    dep_re = re.compile(
        r'(vllm-engine-core-client = \{ git = "' + re.escape(VLLM_GIT) + r'", rev = ")[0-9a-f]+("\s*\})'
    )
    text, n = dep_re.subn(r"\g<1>" + entry["protocol_rev"] + r"\g<2>", text)
    if n != 1:
        sys.exit(f"expected exactly one base dependency line, rewrote {n}")

    # 2. Set the fork [patch] block to this line's fork, or strip it entirely.
    #    The committed Cargo.toml may carry NO [patch] block (the default line can
    #    build against upstream), so we can't assume one exists to rewrite. Strip
    #    any existing block, then append a fresh one when this line has a fork.
    #    Stripping first keeps the script idempotent across local re-runs.
    patch_re = re.compile(
        r'\n*\[patch\."' + re.escape(VLLM_GIT) + r'"\]\n'
        r"vllm-engine-core-client = \{[^}]*\}\n"
    )
    text = patch_re.sub("", text)
    if entry.get("patch_rev"):
        block = (
            f'[patch."{VLLM_GIT}"]\n'
            f'vllm-engine-core-client = {{ git = "{entry["patch_repo"]}", '
            f'rev = "{entry["patch_rev"]}" }}\n'
        )
        text = text.rstrip("\n") + "\n\n" + block

    with open("Cargo.toml", "w", encoding="utf-8") as f:
        f.write(text)
    print(
        f"pinned vllm-engine-core-client: line={line} rev={entry['protocol_rev']} "
        f"patch={entry.get('patch_rev') or 'none'}"
    )


if __name__ == "__main__":
    main()
