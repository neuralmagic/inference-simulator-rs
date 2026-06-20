//! Repo automation, run via `cargo xtask <cmd>`. Reuses `sim-compat` so
//! `compat.toml` is parsed in exactly one place (these used to be inline Python
//! in the workflows, re-parsing the manifest each time).

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sim_compat::{CompatManifest, GoldenManifest};
use toml_edit::{DocumentMut, Item, Table, Value};

const COMPAT_TOML: &str = "compat.toml";
const MANIFEST_TOML: &str = "conformance/manifest.toml";
const CARGO_TOML: &str = "Cargo.toml";
const VLLM_GIT: &str = "https://github.com/vllm-project/vllm.git";

/// Native per-arch release runners (no cross-compile).
const PLATFORMS: &[Platform] = &[
    Platform {
        runner: "ubuntu-latest",
        target: "x86_64-unknown-linux-musl",
        os: "linux",
    },
    Platform {
        runner: "ubuntu-24.04-arm",
        target: "aarch64-unknown-linux-musl",
        os: "linux",
    },
    Platform {
        runner: "macos-14",
        target: "aarch64-apple-darwin",
        os: "macos",
    },
];

#[derive(Parser)]
#[command(name = "xtask", about = "vllm-vcr repo automation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Emit the CI build matrix (one row per compat.toml line) as compact JSON.
    CiMatrix,
    /// Emit the Docker image build matrix as compact JSON.
    DockerMatrix,
    /// Emit the release line x platform matrix as compact JSON.
    ReleaseMatrix,
    /// Print the sim semver from `[workspace.package].version`.
    Simver,
    /// Print `TAG=…; FREPO=…; FREF=…` shell assignments for a line's vllm-rs
    /// frontend build args (`eval` it). FREPO/FREF point at the fork when the
    /// line has one, else upstream at protocol_rev.
    FrontendArgs {
        /// compat.toml line, e.g. "0.23" or "nightly".
        line: String,
    },
    /// Pin `Cargo.toml`'s vllm-engine-core-client to a compat.toml line's rev
    /// (and fork `[patch]`), or to an explicit `--rev` override.
    PinVllm {
        /// compat.toml line, e.g. "0.23" or "nightly".
        line: String,
        /// Override the line's `protocol_rev` with this sha (the nightly canary
        /// pins to the live upstream HEAD).
        #[arg(long)]
        rev: Option<String>,
    },
}

#[derive(Serialize, Clone, Copy)]
struct Platform {
    runner: &'static str,
    target: &'static str,
    os: &'static str,
}

#[derive(Serialize)]
struct CiRow {
    line: String,
    tag: String,
    protocol_rev: String,
    fidelity_validated: bool,
    default: bool,
    has_goldens: bool,
}

#[derive(Serialize)]
struct DockerRow {
    line: String,
    tag: String,
    protocol_rev: String,
    patch_repo: String,
    patch_rev: String,
    fidelity_validated: bool,
    default: bool,
}

#[derive(Serialize)]
struct ReleaseRow {
    line: String,
    tag: String,
    protocol_rev: String,
    default: bool,
    runner: &'static str,
    target: &'static str,
    os: &'static str,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::CiMatrix => ci_matrix(),
        Command::DockerMatrix => docker_matrix(),
        Command::ReleaseMatrix => release_matrix(),
        Command::Simver => simver(),
        Command::FrontendArgs { line } => frontend_args(&line),
        Command::PinVllm { line, rev } => pin_vllm(&line, rev.as_deref()),
    }
}

fn ci_matrix() -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let golden_lines = golden_lines()?;
    let rows: Vec<CiRow> = compat
        .lines
        .iter()
        .map(|v| CiRow {
            line: v.line.clone(),
            tag: v.tag.clone(),
            protocol_rev: v.protocol_rev.clone(),
            fidelity_validated: v.fidelity_validated,
            default: v.default,
            has_goldens: golden_lines.contains(v.line.as_str()),
        })
        .collect();
    print_json(&rows)
}

fn docker_matrix() -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let rows: Vec<DockerRow> = compat
        .lines
        .iter()
        .map(|v| DockerRow {
            line: v.line.clone(),
            tag: v.tag.clone(),
            protocol_rev: v.protocol_rev.clone(),
            patch_repo: v.patch_repo.clone().unwrap_or_default(),
            patch_rev: v.patch_rev.clone().unwrap_or_default(),
            fidelity_validated: v.fidelity_validated,
            default: v.default,
        })
        .collect();
    print_json(&rows)
}

fn release_matrix() -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let rows: Vec<ReleaseRow> = compat
        .lines
        .iter()
        .flat_map(|v| {
            PLATFORMS.iter().map(move |p| ReleaseRow {
                line: v.line.clone(),
                tag: v.tag.clone(),
                protocol_rev: v.protocol_rev.clone(),
                default: v.default,
                runner: p.runner,
                target: p.target,
                os: p.os,
            })
        })
        .collect();
    print_json(&rows)
}

fn simver() -> Result<()> {
    let doc = read_cargo_toml()?;
    let version = doc["workspace"]["package"]["version"]
        .as_str()
        .context("[workspace.package].version is not a string")?;
    println!("{version}");
    Ok(())
}

fn frontend_args(line: &str) -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let entry = compat
        .lines
        .iter()
        .find(|v| v.line == line)
        .with_context(|| format!("no compat.toml entry for line {line}"))?;
    let repo = entry.patch_repo.as_deref().unwrap_or(VLLM_GIT);
    let reference = entry.patch_rev.as_deref().unwrap_or(&entry.protocol_rev);
    println!("TAG={}; FREPO={repo}; FREF={reference}", entry.tag);
    Ok(())
}

fn pin_vllm(line: &str, rev_override: Option<&str>) -> Result<()> {
    let compat = CompatManifest::load(COMPAT_TOML)?;
    let entry = compat
        .lines
        .iter()
        .find(|v| v.line == line)
        .with_context(|| format!("no compat.toml entry for line {line}"))?;
    let rev = rev_override.unwrap_or(&entry.protocol_rev);

    let mut doc = read_cargo_toml()?;
    apply_pin(
        &mut doc,
        rev,
        entry.patch_repo.as_deref(),
        entry.patch_rev.as_deref(),
    )?;

    std::fs::write(CARGO_TOML, doc.to_string()).context("writing Cargo.toml")?;
    eprintln!(
        "pinned vllm-engine-core-client: line={line} rev={rev} patch={}",
        entry.patch_rev.as_deref().unwrap_or("none")
    );
    Ok(())
}

/// Set the base dependency rev and the fork `[patch]` block in `doc`. Stripping
/// any existing patch first keeps re-runs idempotent.
fn apply_pin(
    doc: &mut DocumentMut,
    rev: &str,
    patch_repo: Option<&str>,
    patch_rev: Option<&str>,
) -> Result<()> {
    // Base dependency rev in [workspace.dependencies] (never the fork [patch]).
    let dep = doc["workspace"]["dependencies"]["vllm-engine-core-client"]
        .as_inline_table_mut()
        .context("workspace.dependencies.vllm-engine-core-client must be an inline table")?;
    dep.insert("rev", rev.into());

    if let Some(patch) = doc.get_mut("patch").and_then(Item::as_table_mut) {
        if let Some(src) = patch.get_mut(VLLM_GIT).and_then(Item::as_table_mut) {
            src.remove("vllm-engine-core-client");
            if src.is_empty() {
                patch.remove(VLLM_GIT);
            }
        }
        if patch.is_empty() {
            doc.remove("patch");
        }
    }

    if let Some(patch_rev) = patch_rev {
        let patch_repo = patch_repo.context("patch_rev without patch_repo in compat.toml")?;
        let mut fork = toml_edit::InlineTable::new();
        fork.insert("git", patch_repo.into());
        fork.insert("rev", patch_rev.into());

        let patch = doc
            .entry("patch")
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .context("[patch] is not a table")?;
        patch.set_implicit(true); // emit only [patch."url"], not a bare [patch]
        let src = patch
            .entry(VLLM_GIT)
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .context("[patch.<vllm>] is not a table")?;
        src.insert("vllm-engine-core-client", Item::Value(Value::from(fork)));
    }
    Ok(())
}

/// Lines with at least one golden registered (drives the conformance fetch leg).
fn golden_lines() -> Result<BTreeSet<String>> {
    if !Path::new(MANIFEST_TOML).exists() {
        return Ok(BTreeSet::new());
    }
    let manifest = GoldenManifest::load(MANIFEST_TOML)?;
    Ok(manifest.goldens.into_iter().map(|g| g.line).collect())
}

fn read_cargo_toml() -> Result<DocumentMut> {
    std::fs::read_to_string(CARGO_TOML)
        .context("reading Cargo.toml")?
        .parse::<DocumentMut>()
        .context("parsing Cargo.toml")
}

fn print_json<T: Serialize>(rows: &[T]) -> Result<()> {
    println!("{}", serde_json::to_string(rows)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "\
[workspace.dependencies]
vllm-engine-core-client = { git = \"https://github.com/vllm-project/vllm.git\", rev = \"oldrev\" }
anyhow = \"1\"
";

    fn pin(toml: &str, rev: &str, repo: Option<&str>, patch_rev: Option<&str>) -> String {
        let mut doc: DocumentMut = toml.parse().unwrap();
        apply_pin(&mut doc, rev, repo, patch_rev).unwrap();
        doc.to_string()
    }

    #[test]
    fn pins_rev_with_no_fork() {
        let out = pin(BASE, "newrev", None, None);
        assert!(out.contains(r#"rev = "newrev""#));
        assert!(!out.contains("oldrev"));
        assert!(
            !out.contains("[patch"),
            "no fork means no patch block: {out}"
        );
    }

    #[test]
    fn adds_fork_patch_block() {
        let out = pin(
            BASE,
            "newrev",
            Some("https://github.com/wseaton/vllm.git"),
            Some("forkrev"),
        );
        assert!(out.contains(r#"[patch."https://github.com/vllm-project/vllm.git"]"#));
        assert!(out.contains(r#"git = "https://github.com/wseaton/vllm.git""#));
        assert!(out.contains(r#"rev = "forkrev""#));
        // No bare [patch] header, only the quoted-source table.
        assert!(!out.contains("\n[patch]\n"));
    }

    #[test]
    fn re_pinning_to_no_fork_strips_the_patch() {
        let forked = pin(
            BASE,
            "r1",
            Some("https://github.com/wseaton/vllm.git"),
            Some("fork1"),
        );
        let stripped = pin(&forked, "r2", None, None);
        assert!(!stripped.contains("[patch"), "stale patch left: {stripped}");
        assert!(stripped.contains(r#"rev = "r2""#));
    }

    #[test]
    fn is_idempotent() {
        let once = pin(
            BASE,
            "r",
            Some("https://github.com/wseaton/vllm.git"),
            Some("f"),
        );
        let twice = pin(
            &once,
            "r",
            Some("https://github.com/wseaton/vllm.git"),
            Some("f"),
        );
        assert_eq!(once, twice);
    }
}
