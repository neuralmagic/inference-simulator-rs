//! The `vllm-vcr completions <shell>` subcommand emits a usable script for every
//! supported shell, naming the real subcommands. Runs the actual binary (no mocks).

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_vllm-vcr");

/// Every shell clap_complete supports should produce a non-trivial script that
/// references the binary and its subcommands.
#[test]
fn completions_generate_for_all_shells() {
    for shell in ["bash", "zsh", "fish", "powershell", "elvish"] {
        let out = Command::new(BIN)
            .args(["completions", shell])
            .output()
            .unwrap_or_else(|e| panic!("failed to run `vllm-vcr completions {shell}`: {e}"));

        assert!(
            out.status.success(),
            "completions {shell} exited non-zero: {}",
            String::from_utf8_lossy(&out.stderr)
        );

        let script = String::from_utf8(out.stdout).expect("completion script is UTF-8");
        assert!(
            script.len() > 100,
            "completions {shell} produced a suspiciously short script ({} bytes)",
            script.len()
        );
        assert!(
            script.contains("vllm-vcr"),
            "completions {shell} never names the binary"
        );
        for sub in ["record", "play", "inspect"] {
            assert!(
                script.contains(sub),
                "completions {shell} is missing the `{sub}` subcommand"
            );
        }
    }
}

/// An unknown shell is rejected with a non-zero exit and the valid choices, so a
/// typo fails loudly instead of emitting a broken script.
#[test]
fn completions_reject_unknown_shell() {
    let out = Command::new(BIN)
        .args(["completions", "notashell"])
        .output()
        .expect("failed to run binary");

    assert!(!out.status.success(), "unknown shell should exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("possible values") || stderr.contains("invalid value"),
        "error should list the valid shells, got: {stderr}"
    );
}
