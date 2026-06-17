//! Stamp the target vLLM version into the binary at build time, and emit the
//! per-line capability cfgs the engine needs to compile against older lines.
//!
//! The build matrix builds one image per vLLM line; each build must know which
//! line it speaks so the handshake guard can reject mismatched peers and the
//! ready-response can advertise the right `vllm_version`. A CI matrix build
//! sets `VLLM_TARGET_VERSION` directly (the line it is building); otherwise we
//! stamp the `default = true` line from `compat.toml`.
//!
//! Capability cfgs: where the protocol crate's API diverges across lines, the
//! engine gates on a discrete capability (e.g. `vllm_lora_typed`) rather than a
//! version number. This build script maps the target line to those cfgs. They
//! only reach THIS (root) crate, which is where the gated code lives.

use std::path::PathBuf;

fn main() {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let compat_path = PathBuf::from(&manifest_dir).join("compat.toml");

    println!("cargo:rerun-if-changed={}", compat_path.display());
    println!("cargo:rerun-if-env-changed=VLLM_TARGET_VERSION");

    // CI override wins: a matrix build pins the line it is building.
    let target = match std::env::var("VLLM_TARGET_VERSION") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            let manifest = sim_compat::CompatManifest::load(&compat_path)
                .unwrap_or_else(|e| panic!("loading {}: {e}", compat_path.display()));
            let line = manifest
                .default_line()
                .unwrap_or_else(|e| panic!("compat.toml default line: {e}"));
            line.tag.clone()
        }
    };

    println!("cargo:rustc-env=VLLM_TARGET_VERSION={target}");

    // Declare every capability cfg so unknown values don't warn (Rust check-cfg).
    println!("cargo::rustc-check-cfg=cfg(vllm_lora_typed)");

    // `vllm_lora_typed`: the protocol crate exposes a typed `protocol::lora`
    // module and `EngineCoreRequest.lora_request: Option<LoraRequest>`. Absent
    // on 0.22 (lora is opaque rmpv there); present on 0.23+.
    if line_at_least(&target, 0, 23) {
        println!("cargo::rustc-cfg=vllm_lora_typed");
    }
}

/// Whether the target tag's `major.minor` line is >= `(major, minor)`.
///
/// A tag with no parseable `major.minor` (e.g. `"nightly"`, which tracks vLLM
/// main) is treated as the **newest** line, so every capability cfg turns on:
/// main always carries the latest protocol surface.
fn line_at_least(tag: &str, major: u32, minor: u32) -> bool {
    let Some(line) = sim_compat::minor_line(tag) else {
        return true;
    };
    let mut parts = line.split('.');
    let parsed = (|| {
        let maj: u32 = parts.next()?.parse().ok()?;
        let min: u32 = parts.next()?.parse().ok()?;
        Some((maj, min))
    })();
    match parsed {
        Some((maj, min)) => (maj, min) >= (major, minor),
        None => true,
    }
}
