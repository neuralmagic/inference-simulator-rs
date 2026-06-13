# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot calibration sample dumps produced by `inference-sim-trace calibrate --dump-samples`.

Terminology (matches the README): "captured" curves are tap recordings of a
real engine; "modeled" curves are the simulator's trace-fitted statistical
model. Captured timings are never played back verbatim.

Produces, depending on flags:
  replay-fidelity.png       captured vs modeled vs knob model: survival + Q-Q
  mean-vs-pertoken.png      per-token ITL vs per-request mean ITL, both captured
  <--compare-out>           survival overlay of labeled traces (default sim-comparison.png)
  <--cache-effect-out>      per-turn-cohort TTFT survival (default multiturn-cache-effect.png)

Usage:
  uv run scripts/plot_calibration.py \
      --samples /tmp/calib-samples.json \
      --trace /tmp/trace.jsonl \
      --out-dir docs/images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# House palette: captured = near-black, modeled = Red Hat red, reference
# models = brand blue, extras for ad-hoc --compare invocations.
C_CAP = "#151515"
C_MOD = "#ee0000"
C_REF = "#0066cc"
C_MUT = "#8a8d90"
EXTRA = ["#3e8635", "#8476d1"]

# Legacy short keys still map to the canonical display labels, so existing
# invocations (and the cache-effect join keys) keep working.
DISPLAY = {
    "real": "captured (ground-truth run)",
    "source": "captured (ground-truth run)",
    "replay": "modeled (schedule replay)",
    "knobs": "knob model (best fit)",
    "nocache": "modeled, prefix cache off (what-if)",
}

matplotlib.rcParams.update(
    {
        "font.family": ["Red Hat Text", "Helvetica Neue", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": "#444444",
        "axes.edgecolor": "#b0b0b0",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d8d8d8",
        "grid.linewidth": 0.6,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 170,
    }
)


def display_label(label: str) -> str:
    return DISPLAY.get(label, label)


def style_axes(ax: Axes) -> None:
    ax.grid(which="major", color="#d8d8d8", linewidth=0.6)
    ax.grid(which="minor", color="#efefef", linewidth=0.5)
    ax.set_axisbelow(True)


def title(ax: Axes, text: str) -> None:
    ax.set_title(text, fontfamily=["Red Hat Display", "Helvetica Neue", "Arial"], color="#151515")


def survival(ax: Axes, data: np.ndarray, label: str, color: str, lw: float, ls: str = "-") -> None:
    x = np.sort(data)
    y = np.maximum(1.0 - np.arange(1, len(x) + 1) / len(x), 1.0 / len(x))
    ax.step(x, y, where="post", label=label, color=color, lw=lw, ls=ls, solid_capstyle="round")


def qq(ax: Axes, src: np.ndarray, rep: np.ndarray, color: str) -> None:
    qs = np.linspace(0.001, 0.999, 400)
    ax.plot(np.quantile(src, qs), np.quantile(rep, qs), ".", ms=3.5, color=color, alpha=0.65)
    lim = [
        min(src.min(), rep.min()) * 0.9,
        max(np.quantile(src, 0.999), np.quantile(rep, 0.999)) * 1.1,
    ]
    ax.plot(lim, lim, "--", color="#b0b0b0", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)


def fidelity_figure(dump: dict, out: Path) -> None:
    """Captured vs modeled vs knob model, ITL only. TTFT is intentionally
    absent: since the step-model rework, loaded TTFT comes from engine
    mechanics (queueing, chunk interference), not a sampled distribution, so
    its marginal can only be checked wire-level by the arrival-replay gates.
    The per-token ITL distribution is still a direct in-sample fit check."""
    src_i = np.array(dump["source"]["itl_ms"])
    rep_i = np.array(dump["replay"]["itl_ms"])
    knb_i = np.array(dump["knobfit"]["itl_ms"])

    fig, (ax, ax_q) = plt.subplots(1, 2, figsize=(13, 5))

    survival(ax, src_i, "captured (real engine, per-token)", C_CAP, 2.7)
    survival(ax, rep_i, "modeled (trace fit)", C_MOD, 1.7)
    survival(ax, knb_i, "knob model (best fit)", C_REF, 1.5, "--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inter-token latency (ms)")
    ax.set_ylabel("P(ITL > x)")
    title(ax, "Inter-token latency survival")
    ax.legend()
    style_axes(ax)

    qq(ax_q, src_i, rep_i, C_MOD)
    ax_q.set_xlabel("captured ITL quantiles (ms)")
    ax_q.set_ylabel("modeled ITL quantiles (ms)")
    title(ax_q, "Q-Q: ITL, captured vs modeled")
    style_axes(ax_q)

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def mean_vs_pertoken_figure(dump: dict, trace_path: Path, out: Path) -> None:
    src_i = np.array(dump["source"]["itl_ms"])
    per_req_means = []
    with open(trace_path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            itls = r.get("itl_ms")
            if itls:
                per_req_means.append(sum(itls) / len(itls))
    if not per_req_means:
        print(f"skipping {out}: trace has no per-token itl_ms arrays")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    survival(ax, src_i, "per-token ITL (captured)", C_CAP, 2.7)
    survival(ax, np.array(per_req_means), "per-request mean ITL (client-report view)", C_REF, 2.0)
    # Sub-ms outliers (chunked finish gaps) would drag the log axis into empty
    # space; anchor the left edge near the bulk of the distribution instead.
    left = float(np.quantile(src_i, 0.01)) * 0.5
    p999 = float(np.quantile(src_i, 0.999))
    ax.axvline(p999, color=C_MOD, ls=":", lw=1)
    ax.annotate(
        f"per-token p99.9 = {p999:.1f} ms",
        xy=(p999, 1.5e-3),
        xytext=(left * 3.0, 1e-4),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": C_MOD},
        color=C_MOD,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left, src_i.max() * 1.5)
    ax.set_xlabel("inter-token latency (ms)")
    ax.set_ylabel("P(ITL > x)")
    title(ax, "Both curves captured; averaging hides the tail")
    ax.legend(loc="lower left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def load_trace_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Pool TTFT and per-token ITL samples from a trace JSONL."""
    ttfts: list[float] = []
    itls: list[float] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            ttfts.append(r["ttft_ms"])
            if r.get("itl_ms"):
                itls.extend(r["itl_ms"])
            elif r.get("itl_summary"):
                itls.extend([r["itl_summary"]["mean_ms"]] * r["itl_summary"]["count"])
    return np.array(ttfts), np.array(itls)


def comparison_figure(traces: list[tuple[str, Path]], out: Path) -> None:
    """Survival curves for several traces on shared axes: one labeled trace per curve.

    The first entry is drawn as the captured reference (near-black, heavy);
    later entries cycle through the modeled palette."""
    palette = [C_CAP, C_MOD, C_REF, *EXTRA]
    fig, (ax_itl, ax_ttft) = plt.subplots(1, 2, figsize=(13, 5))

    for (label, path), color in zip(traces, palette):
        ttfts, itls = load_trace_samples(path)
        lw = 2.7 if color == C_CAP else 1.8
        survival(ax_itl, itls, display_label(label), color, lw)
        survival(ax_ttft, ttfts, display_label(label), color, lw)

    ax_itl.set_xscale("log")
    ax_itl.set_yscale("log")
    ax_itl.set_xlabel("inter-token latency (ms)")
    ax_itl.set_ylabel("P(ITL > x)")
    title(ax_itl, "Inter-token latency survival")
    ax_itl.legend()
    style_axes(ax_itl)

    ax_ttft.set_yscale("log")
    ax_ttft.set_xlabel("TTFT (ms)")
    ax_ttft.set_ylabel("P(TTFT > x)")
    title(ax_ttft, "TTFT survival")
    ax_ttft.legend()
    style_axes(ax_ttft)

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def load_records(path: Path) -> list[dict]:
    """All non-meta records of a trace JSONL, as dicts."""
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            records.append(r)
    return records


def turn_depths(records: list[dict]) -> list[int]:
    """Session-turn depth per record (1 = session root), inferred from chained
    block_hashes the same way the replay harness infers sessions: a record's
    parent is the most recent earlier record whose full hash chain is a proper
    prefix of its own chain."""
    by_last_hash: dict[int, int] = {}
    depths: list[int] = []
    for i, r in enumerate(records):
        chain = r.get("block_hashes") or []
        depth = 1
        for k in range(len(chain) - 1, 0, -1):
            p = by_last_hash.get(chain[k - 1])
            if p is not None and len(records[p].get("block_hashes") or []) == k:
                depth = depths[p] + 1
                break
        if chain:
            by_last_hash[chain[-1]] = i
        depths.append(depth)
    return depths


def cache_effect_figure(traces: list[tuple[str, Path]], out: Path) -> None:
    """Prove the prefix-cache effect is reproduced shape-wise, not just in the
    pooled marginal: cohort requests by session-turn depth (turn 1 hits only
    the shared prefix; deeper turns hit their session's growing context) and
    overlay captured vs modeled TTFT survival per cohort. Compensating errors
    between cohorts would show here and not in the pooled curve. An optional
    nocache=... trace adds the cache-off what-if for magnitude."""
    by_label = dict(traces)
    src = load_records(by_label["real"])
    src.sort(key=lambda r: r["arrival_ms"])
    depths = turn_depths(src)

    def by_arrival(label: str) -> dict[float, dict]:
        return {round(r["arrival_ms"], 3): r for r in load_records(by_label[label])}

    rep = by_arrival("replay")
    cold = by_arrival("nocache") if "nocache" in by_label else {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cohorts = [
        ("Turn 1 (shared-prefix hit)", lambda d: d == 1, axes[0]),
        ("Turns 2+ (growing-context hit)", lambda d: d >= 2, axes[1]),
    ]
    for cohort_title, pred, ax in cohorts:
        keys = [round(r["arrival_ms"], 3) for r, d in zip(src, depths) if pred(d)]
        real = np.array([r["ttft_ms"] for r, d in zip(src, depths) if pred(d)])
        replay = np.array([rep[k]["ttft_ms"] for k in keys if k in rep])
        survival(ax, real, display_label("real"), C_CAP, 2.7)
        survival(ax, replay, display_label("replay"), C_MOD, 1.8)
        if cold:
            nocache = np.array([cold[k]["ttft_ms"] for k in keys if k in cold])
            survival(ax, nocache, display_label("nocache"), C_REF, 1.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("P(TTFT > x)")
        title(ax, cohort_title)
        ax.legend()
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def load_chunk_tokens(path: Path) -> np.ndarray:
    """Pool per-decode-chunk token counts from a trace JSONL. `itl_tokens` is
    parallel to `itl_ms` (the chunk that closed each gap); records that omit it
    decoded one token per gap, so they contribute that many 1s. The first
    (prefill) chunk has no gap and is excluded, so captured and replayed traces
    compare apples to apples."""
    sizes: list[int] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            toks = r.get("itl_tokens")
            if toks:
                sizes.extend(int(t) for t in toks)
            elif r.get("itl_ms"):
                sizes.extend([1] * len(r["itl_ms"]))
    return np.array(sizes)


def load_accept_per_pos(path: Path) -> np.ndarray:
    """Sum `num_accepted_tokens_per_pos` across a step-stats sidecar JSONL,
    giving total accepted draft tokens per draft position (position 0 = the
    first speculated token). Length is the configured speculative budget K."""
    total: np.ndarray | None = None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            spec = (rec.get("scheduler") or {}).get("spec_decoding_stats")
            if not spec:
                continue
            per_pos = spec.get("num_accepted_tokens_per_pos") or []
            if not per_pos:
                continue
            arr = np.array(per_pos, dtype=float)
            if total is None:
                total = np.zeros(len(arr))
            if len(arr) > len(total):
                grown = np.zeros(len(arr))
                grown[: len(total)] = total
                total = grown
            total[: len(arr)] += arr
    return total if total is not None else np.array([])


def median_gap_by_chunk_size(path: Path, k_max: int) -> np.ndarray:
    """Median per-chunk gap grouped by how many tokens that chunk delivered.
    Speculative decoding verifies all K drafts in one target forward pass, so
    the step time barely moves with the accepted count - this is the joint
    `(gap, tokens)` structure the replay must reproduce, the part a flattened
    gap/N replay destroys. NaN for sizes the trace never produced."""
    buckets: dict[int, list[float]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                continue
            gaps = r.get("itl_ms")
            toks = r.get("itl_tokens")
            if not gaps:
                continue
            if not toks:
                toks = [1] * len(gaps)
            for g, t in zip(gaps, toks):
                buckets.setdefault(int(t), []).append(g)
    return np.array(
        [float(np.median(buckets[k])) if buckets.get(k) else np.nan for k in range(1, k_max + 1)]
    )


def chunk_pmf(sizes: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Probability mass over chunk sizes 1..k_max."""
    xs = np.arange(1, k_max + 1)
    counts = np.array([(sizes == k).sum() for k in xs], dtype=float)
    total = counts.sum()
    return xs, (counts / total if total else counts)


def spec_fidelity_figure(
    traces: dict[str, Path], steps: dict[str, Path], out: Path
) -> None:
    """Prove multi-token-step (speculative decoding / diffusion) replay: the
    modeled stream reproduces the capture's per-chunk burst sizes (Deltas) and
    per-chunk pacing, and the scheduler-stats integration round-trips the
    per-position acceptance. A flattened replay would collapse panel A to all
    1s and pace panel B ~Kx too fast."""
    cap_sizes = load_chunk_tokens(traces["real"])
    rep_sizes = load_chunk_tokens(traces["replay"])

    have_c = bool(steps)
    ncols = 3 if have_c else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))

    # Panel A: per-chunk token-count distribution (the burst structure).
    ax_a = axes[0]
    k_max = int(max(cap_sizes.max(initial=1), rep_sizes.max(initial=1)))
    xs, cap_pmf = chunk_pmf(cap_sizes, k_max)
    _, rep_pmf = chunk_pmf(rep_sizes, k_max)
    w = 0.4
    ax_a.bar(xs - w / 2, cap_pmf, w, label="captured (real engine)", color=C_CAP)
    ax_a.bar(xs + w / 2, rep_pmf, w, label="replayed (verbatim)", color=C_MOD)
    ax_a.set_xticks(xs)
    ax_a.set_xlabel("tokens delivered per decode step")
    ax_a.set_ylabel("fraction of decode steps")
    title(ax_a, "Multi-token output (Deltas) per step")
    ax_a.legend()
    style_axes(ax_a)

    # Panel B: median step time vs tokens delivered. The spec-decode signature
    # is a flat line (one forward pass verifies all K drafts), and the replay
    # reproduces it; the dashed curve is what a gap/N-flattened replay would do.
    ax_b = axes[1]
    cap_gap = median_gap_by_chunk_size(traces["real"], k_max)
    rep_gap = median_gap_by_chunk_size(traces["replay"], k_max)
    ax_b.bar(xs - w / 2, np.nan_to_num(cap_gap), w, label="captured (real engine)", color=C_CAP)
    ax_b.bar(xs + w / 2, np.nan_to_num(rep_gap), w, label="replayed (verbatim)", color=C_MOD)
    step = float(cap_gap[~np.isnan(cap_gap)][0]) if np.any(~np.isnan(cap_gap)) else 1.0
    ax_b.plot(
        xs, step / xs, "o--", color=C_MUT, lw=1.5, ms=5,
        label="gap/N (flattened replay)",
    )
    ax_b.set_xticks(xs)
    ax_b.set_ylim(0, max(np.nanmax(cap_gap), step) * 1.45)
    ax_b.set_xlabel("tokens delivered per decode step")
    ax_b.set_ylabel("median step time (ms)")
    title(ax_b, "Step time is flat in burst size")
    ax_b.legend(loc="upper center", ncol=2)
    style_axes(ax_b)

    # Panel C: per-position acceptance from the SchedulerStats sidecar (the
    # scheduler-stats integration, end to end).
    if have_c:
        ax_c = axes[2]
        cap_pos = load_accept_per_pos(steps["real"])
        pos = np.arange(1, len(cap_pos) + 1)
        if "replay" in steps:
            rep_pos = load_accept_per_pos(steps["replay"])
            n = max(len(cap_pos), len(rep_pos))
            pos = np.arange(1, n + 1)
            cap_y = np.zeros(n)
            cap_y[: len(cap_pos)] = cap_pos
            rep_y = np.zeros(n)
            rep_y[: len(rep_pos)] = rep_pos
            ax_c.bar(pos - w / 2, cap_y, w, label="captured (real engine)", color=C_CAP)
            ax_c.bar(pos + w / 2, rep_y, w, label="replayed (sim-emitted)", color=C_MOD)
        else:
            ax_c.bar(pos, cap_pos, 0.6, label="captured (real engine)", color=C_CAP)
        ax_c.set_xticks(pos)
        ax_c.set_xlabel("draft position")
        ax_c.set_ylabel("accepted draft tokens")
        title(ax_c, "Speculation acceptance (SchedulerStats)")
        ax_c.legend()
        style_axes(ax_c)

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def parse_labeled_trace(spec: str) -> tuple[str, Path]:
    label, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    return label, Path(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, help="calibrate --dump-samples output")
    p.add_argument("--trace", type=Path, help="the source trace JSONL")
    p.add_argument(
        "--compare",
        type=parse_labeled_trace,
        action="append",
        metavar="LABEL=PATH",
        help="repeatable; plot survival curves of several traces on shared axes "
        "(first entry is drawn as the captured reference; the short keys "
        "real/replay/knobs/nocache expand to canonical display labels)",
    )
    p.add_argument(
        "--compare-out",
        default="sim-comparison.png",
        help="output filename for the --compare figure",
    )
    p.add_argument(
        "--cache-effect",
        type=parse_labeled_trace,
        action="append",
        metavar="LABEL=PATH",
        help="repeatable; per-turn-cohort TTFT survival from real=trace.jsonl, "
        "replay=measured.jsonl, and optional nocache=measured.jsonl (traces "
        "are joined on arrival_ms; real needs block_hashes)",
    )
    p.add_argument(
        "--cache-effect-out",
        default="multiturn-cache-effect.png",
        help="output filename for the --cache-effect figure",
    )
    p.add_argument(
        "--spec-fidelity",
        type=parse_labeled_trace,
        action="append",
        metavar="LABEL=PATH",
        help="repeatable; multi-token-step (spec-decode/diffusion) replay proof "
        "from real=tap-trace.jsonl and replay=dump.jsonl (both need itl_tokens "
        "for the burst panel)",
    )
    p.add_argument(
        "--spec-steps",
        type=parse_labeled_trace,
        action="append",
        metavar="LABEL=PATH",
        help="repeatable; SchedulerStats sidecars for the acceptance panel, "
        "real=step-stats.jsonl and optional replay=step-stats.jsonl",
    )
    p.add_argument(
        "--spec-fidelity-out",
        default="spec-decode-fidelity.png",
        help="output filename for the --spec-fidelity figure",
    )
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.samples and args.trace:
        dump = json.load(open(args.samples))
        fidelity_figure(dump, args.out_dir / "replay-fidelity.png")
        mean_vs_pertoken_figure(dump, args.trace, args.out_dir / "mean-vs-pertoken.png")
    if args.compare:
        comparison_figure(args.compare, args.out_dir / args.compare_out)
    if args.cache_effect:
        labels = {label for label, _ in args.cache_effect}
        if not {"real", "replay"} <= labels:
            p.error("--cache-effect needs at least real=... and replay=...")
        cache_effect_figure(args.cache_effect, args.out_dir / args.cache_effect_out)
    if args.spec_fidelity:
        traces = dict(args.spec_fidelity)
        if not {"real", "replay"} <= set(traces):
            p.error("--spec-fidelity needs at least real=... and replay=...")
        steps = dict(args.spec_steps or [])
        spec_fidelity_figure(traces, steps, args.out_dir / args.spec_fidelity_out)
    if (
        not (args.samples and args.trace)
        and not args.compare
        and not args.cache_effect
        and not args.spec_fidelity
    ):
        p.error(
            "nothing to do: pass --samples/--trace, --compare, --cache-effect, "
            "and/or --spec-fidelity"
        )


if __name__ == "__main__":
    main()
