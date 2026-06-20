//! `vllm-vcr inspect`: convert benchmark reports to the trace format, summarize
//! existing traces, render Perfetto views, and run calibration comparisons.

use std::fs;
use std::io::{self, BufWriter, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Subcommand;
use sim_s3::TraceUri;
use vllm_vcr::calibrate::{self, PromptReplay, SessionPacing, SimDriver};
use vllm_vcr::perfetto::{Overlays, PerfettoOptions, write_perfetto};
use vllm_vcr::step_stats::{read_step_stats, step_spans, step_stats_counters};
use vllm_vcr::trace::{TraceWriter, open_trace_reader, read_trace_file};
use vllm_vcr::trace_convert::{
    ConvertOptions, convert_guidellm, summarize_trace, write_conversion, write_summary,
};

/// Output format for calibration reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum ReportFormat {
    /// Human-readable text report.
    #[default]
    Text,
    /// Machine-readable JSON.
    Json,
}

/// Magnitude profile for the synthesized demo trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum DemoProfile {
    /// Realistic magnitudes (TTFT ~50-400ms, ITL ~10-60ms).
    #[default]
    Realistic,
    /// Fast/small magnitudes suitable for e2e testing (TTFT ~15-40ms, ITL ~3-10ms).
    Fast,
}

/// Which e2e harness runs: closed-loop sampled batches or open-loop arrival replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum E2eHarness {
    /// Sample N requests and drive them closed-loop.
    #[default]
    Sampled,
    /// Replay the trace's recorded arrival schedule open-loop (real time).
    /// Requires records with arrival_ms; runtime equals the schedule's span.
    ReplayArrivals,
}

#[derive(Subcommand)]
pub(crate) enum InspectCommand {
    /// Convert a guidellm benchmark report (JSON) to a vllm-vcr trace (JSONL).
    FromGuidellm {
        /// Path to the guidellm report JSON file.
        input: PathBuf,
        /// Output JSONL trace file path.
        #[arg(short, long)]
        output: PathBuf,
        /// Override the model name in trace metadata.
        #[arg(long)]
        model: Option<String>,
        /// GPU identifier for trace metadata.
        #[arg(long)]
        gpu: Option<String>,
        /// Tensor-parallel degree for trace metadata.
        #[arg(long)]
        tp: Option<u32>,
    },
    /// Print summary statistics from an existing trace file.
    Summarize {
        /// Path or `s3://bucket/key` URI to the JSONL trace file.
        input: TraceUri,
    },
    /// Convert a trace into the Chrome Trace Event Format (drop the output on
    /// <https://ui.perfetto.dev> to view request spans over a concurrency
    /// counter). Reads `.gz` traces transparently.
    Perfetto {
        /// Path or `s3://bucket/key` URI to the JSONL trace file (`.gz` ok).
        input: TraceUri,
        /// Output JSON path or `s3://bucket/key` URI. Defaults to stdout when
        /// omitted (or no stdout dump when `--open` is set).
        #[arg(short, long)]
        output: Option<TraceUri>,
        /// Step-stats sidecar (`--step-stats-out` from the tap, `.gz` ok, path
        /// or `s3://` URI) to overlay as batch-level counter tracks: scheduler
        /// queue depths, KV-cache usage, and spec-decode acceptance rate.
        #[arg(long)]
        step_stats: Option<TraceUri>,
        /// Override the process-row label (defaults to the trace's model).
        #[arg(long)]
        name: Option<String>,
        /// Give every request its own track instead of packing them into
        /// reusable lanes (peak-concurrency rows). Handy for small traces.
        #[arg(long)]
        track_per_request: bool,
        /// Serve the trace over localhost and open it in the Perfetto UI. Blocks
        /// (the UI fetches from this process) until Ctrl-C.
        #[arg(long)]
        open: bool,
        /// Port for `--open`'s local server. Defaults to 0, letting the OS
        /// pick a free ephemeral port (the actual URL is printed and opened),
        /// so concurrent or just-restarted servers never collide. Pin it only
        /// when you want a stable URL.
        #[arg(long, default_value_t = 0)]
        port: u16,
    },
    /// Synthesize a heavy-tailed demo trace for calibration testing.
    GenDemo {
        /// Output JSONL trace file path.
        #[arg(short, long)]
        output: PathBuf,
        /// Number of records to generate.
        #[arg(long, default_value_t = 200)]
        records: usize,
        /// RNG seed for deterministic output.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Magnitude profile; bare `--fast` selects the fast one.
        #[arg(
            long = "fast",
            value_enum,
            default_value_t = DemoProfile::Realistic,
            default_missing_value = "fast",
            num_args = 0..=1
        )]
        fast: DemoProfile,
    },
    /// Model-level calibration: compare source trace quantiles against TraceLatency
    /// replay and a best-fit KnobLatency. No transport, exact and fast.
    Calibrate {
        /// Path to the JSONL trace file.
        trace: PathBuf,
        /// Total number of samples to draw (divided across records).
        #[arg(long, default_value_t = 100000)]
        samples: usize,
        /// RNG seed for deterministic output.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Maximum allowed relative error for replay vs source (PASS/FAIL threshold).
        #[arg(long, default_value_t = 0.10)]
        tolerance: f64,
        /// Report format; bare `--json` selects JSON.
        #[arg(
            long = "json",
            value_enum,
            default_value_t = ReportFormat::Text,
            default_missing_value = "json",
            num_args = 0..=1
        )]
        json: ReportFormat,
        /// Also write the pooled source/replay/knob-fit sample arrays to this path
        /// as JSON, for external plotting.
        #[arg(long)]
        dump_samples: Option<PathBuf>,
    },
    /// Wire-level calibration: spin the real simulator in-process and measure
    /// client-side TTFT/ITL against source trace quantiles.
    ///
    /// Runtime: with the demo trace's default magnitudes (~15-100ms TTFT, ~3-30ms ITL)
    /// this finishes in under 60s at N=60. Increase --requests for tighter quantiles
    /// at the cost of wall time.
    CalibrateE2e {
        /// Path to the JSONL trace file.
        trace: PathBuf,
        /// Number of requests to send through the simulator
        /// (default: 60 sampled; with --replay-arrivals, the whole schedule).
        #[arg(long)]
        requests: Option<usize>,
        /// RNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Latency model driving the sim; bare `--knob-fit` selects knob-fit.
        #[arg(
            long = "knob-fit",
            value_enum,
            default_value_t = SimDriver::TraceReplay,
            default_missing_value = "knob-fit",
            num_args = 0..=1
        )]
        knob_fit: SimDriver,
        /// Maximum allowed relative error (looser default for transport jitter).
        #[arg(long, default_value_t = 0.25)]
        tolerance: f64,
        /// Which harness runs; bare `--replay-arrivals` selects arrival replay.
        #[arg(
            long = "replay-arrivals",
            value_enum,
            default_value_t = E2eHarness::Sampled,
            default_missing_value = "replay-arrivals",
            num_args = 0..=1
        )]
        replay_arrivals: E2eHarness,
        /// With --replay-arrivals: build the sim's latency model from this
        /// trace instead of the replayed one, validating against an arrival
        /// process the model was not fitted on.
        #[arg(long, requires = "replay_arrivals")]
        latency_trace: Option<PathBuf>,
        /// With --replay-arrivals: also write the client-side measurements as
        /// a trace JSONL, for plotting against the source (e.g.
        /// scripts/plot_calibration.py --compare).
        #[arg(long, requires = "replay_arrivals")]
        dump_trace: Option<PathBuf>,
        /// With --replay-arrivals: arrival pacing; bare `--replay-sessions`
        /// chains each multiturn session closed-loop (turn N+1 fires when
        /// turn N completes plus the recorded think gap). Sessions are
        /// inferred from block-hash chains.
        #[arg(
            long = "replay-sessions",
            value_enum,
            default_value_t = SessionPacing::OpenLoop,
            default_missing_value = "chained",
            num_args = 0..=1,
            requires = "replay_arrivals"
        )]
        replay_sessions: SessionPacing,
        /// With --replay-arrivals: prompt reconstruction; bare `--cold-prompts`
        /// replays every prompt as unique tokens (cache-off what-if).
        #[arg(
            long = "cold-prompts",
            value_enum,
            default_value_t = PromptReplay::SharedPrefixes,
            default_missing_value = "cold",
            num_args = 0..=1,
            requires = "replay_arrivals"
        )]
        cold_prompts: PromptReplay,
        /// With --replay-arrivals: time compression (sim delays divided by
        /// this, measurements re-multiplied). Faster inner loops, slightly
        /// noisier quantiles; use 1.0 for final validation.
        #[arg(long, default_value_t = 1.0, requires = "replay_arrivals")]
        time_scale: f64,
        /// With --replay-arrivals: extra flag tokens for the in-process sim,
        /// repeated per token (e.g. --sim-arg=--kv-cache-size --sim-arg=8192).
        /// Must mirror the capture engine's scheduler/cache config.
        #[arg(
            long = "sim-arg",
            requires = "replay_arrivals",
            allow_hyphen_values = true
        )]
        sim_args: Vec<String>,
        /// Report format; bare `--json` selects JSON.
        #[arg(
            long = "json",
            value_enum,
            default_value_t = ReportFormat::Text,
            default_missing_value = "json",
            num_args = 0..=1
        )]
        json: ReportFormat,
    },
}

/// Local path for an input URI; fetches s3:// to scratch on a short-lived runtime.
fn materialize_input(uri: &TraceUri) -> Result<PathBuf> {
    if let Some(path) = uri.local_path() {
        return Ok(path.to_path_buf());
    }
    block_on(uri.materialize(&std::env::temp_dir()))
}

/// Write rendered bytes to an output URI: local files directly; s3:// staged to
/// scratch and uploaded.
fn write_output(bytes: &[u8], uri: &TraceUri) -> Result<()> {
    let local = uri.write_path(&std::env::temp_dir());
    fs::write(&local, bytes).with_context(|| format!("writing {}", local.display()))?;
    if uri.is_remote() {
        block_on(uri.upload(&local))?;
    }
    Ok(())
}

/// Run one async S3 op to completion on a fresh single-thread runtime.
fn block_on<T>(fut: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime for S3 I/O")?
        .block_on(fut)
}

pub(crate) fn run(command: InspectCommand) -> Result<ExitCode> {
    match command {
        InspectCommand::FromGuidellm {
            input,
            output,
            model,
            gpu,
            tp,
        } => {
            let report_json = fs::read_to_string(&input)
                .with_context(|| format!("reading {}", input.display()))?;

            let opts = ConvertOptions { model, gpu, tp };
            let (meta, records) = convert_guidellm(&report_json, &opts)?;

            let mut writer = TraceWriter::create(&output)?;
            write_conversion(&mut writer, &meta, &records)?;
            writer.finish()?;

            eprintln!("wrote {} records to {}", records.len(), output.display());
        }
        InspectCommand::Summarize { input } => {
            let input = materialize_input(&input)?;
            let reader = open_trace_reader(&input)?;
            let (meta, stats) = summarize_trace(reader)?;

            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            write_summary(&mut writer, &meta, &stats)?;
        }
        InspectCommand::Perfetto {
            input,
            output,
            step_stats,
            name,
            track_per_request,
            open,
            port,
        } => {
            let input = materialize_input(&input)?;
            let (meta, records) = read_trace_file(&input)?;
            let opts = PerfettoOptions {
                process_name: name,
                track_per_request,
            };

            // Optional sidecar overlays: batch-level counter tracks plus the
            // step-centric step-span track (true per-step prefill/decode view).
            let overlays = match step_stats {
                Some(uri) => {
                    let path = materialize_input(&uri)?;
                    let reader = open_trace_reader(&path)?;
                    let snapshots = read_step_stats(reader)?;
                    Overlays {
                        counters: step_stats_counters(&snapshots),
                        steps: step_spans(&snapshots),
                    }
                }
                None => Overlays::default(),
            };

            // Render once into memory, then route: save, serve, and/or dump.
            let mut bytes = Vec::new();
            let summary = write_perfetto(&mut bytes, &meta, &records, &overlays, &opts)?;

            if let Some(uri) = &output {
                write_output(&bytes, uri)?;
                eprintln!("wrote {} events to {uri}", summary.events);
            }
            if summary.dropped_requests > 0 {
                eprintln!(
                    "note: {} of {} records had no arrival_ms and were dropped (cannot place on a timeline)",
                    summary.dropped_requests,
                    summary.placed_requests + summary.dropped_requests,
                );
            }

            if open {
                serve_perfetto(&bytes, port)?;
            } else if output.is_none() {
                io::stdout()
                    .write_all(&bytes)
                    .context("writing perfetto trace to stdout")?;
            }
        }
        InspectCommand::GenDemo {
            output,
            records,
            seed,
            fast,
        } => {
            let (meta, recs) = match fast {
                DemoProfile::Fast => calibrate::gen_demo_fast(records, seed),
                DemoProfile::Realistic => calibrate::gen_demo(records, seed),
            };
            calibrate::write_demo_trace(&output, &meta, &recs)?;
            eprintln!("wrote {} records to {}", recs.len(), output.display());
        }
        InspectCommand::Calibrate {
            trace,
            samples,
            seed,
            tolerance,
            json,
            dump_samples,
        } => {
            let report = calibrate::calibrate_from_file(&trace, samples, seed, tolerance)?;

            if let Some(dump_path) = dump_samples {
                let (_meta, records) = read_trace_file(&trace)?;
                let dump = calibrate::dump_samples(&records, samples, seed)?;
                let out = fs::File::create(&dump_path)
                    .with_context(|| format!("creating {}", dump_path.display()))?;
                serde_json::to_writer(BufWriter::new(out), &dump)
                    .context("serializing sample dump")?;
                eprintln!("wrote sample dump to {}", dump_path.display());
            }

            write_report_as(json, &report)?;

            if !report.verdict.replay_pass {
                return Ok(ExitCode::FAILURE);
            }
        }
        InspectCommand::CalibrateE2e {
            trace,
            requests,
            seed,
            knob_fit,
            tolerance,
            replay_arrivals,
            latency_trace,
            dump_trace,
            replay_sessions,
            cold_prompts,
            time_scale,
            sim_args,
            json,
        } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("building tokio runtime for calibrate-e2e")?;

            let report = if replay_arrivals == E2eHarness::ReplayArrivals {
                let cfg = calibrate::ReplayArrivalsConfig {
                    trace_path: &trace,
                    latency_trace: latency_trace.as_deref(),
                    max_requests: requests,
                    tolerance,
                    driver: knob_fit,
                    ipc_tag: seed.to_string(),
                    extra_sim_args: sim_args,
                    pacing: replay_sessions,
                    prompts: cold_prompts,
                    time_scale,
                };
                let outcome = runtime.block_on(calibrate::replay_arrivals(&cfg))?;
                eprintln!(
                    "replay-arrivals: {}/{} requests completed in {:.1}s (max send lag {:.1}ms)",
                    outcome.requests_completed,
                    outcome.requests_replayed,
                    outcome.wall_time_s,
                    outcome.max_send_lag_ms,
                );
                if let Some(dump_path) = dump_trace {
                    let meta = vllm_vcr::trace::TraceMeta {
                        source: Some("replay-arrivals".to_string()),
                        ..Default::default()
                    };
                    let mut writer = TraceWriter::create(&dump_path)?;
                    vllm_vcr::trace::write_trace(&mut writer, &meta, &outcome.measured)?;
                    writer.finish()?;
                    eprintln!(
                        "wrote {} measured records to {}",
                        outcome.measured.len(),
                        dump_path.display()
                    );
                }
                outcome.report
            } else {
                runtime.block_on(calibrate_e2e_impl(
                    &trace,
                    requests.unwrap_or(60),
                    seed,
                    knob_fit,
                    tolerance,
                ))?
            };

            write_report_as(json, &report)?;

            if !report.verdict.replay_pass {
                return Ok(ExitCode::FAILURE);
            }
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// Serve the rendered trace bytes over localhost (CORS-open so the hosted
/// Perfetto UI can fetch them) and open the UI deep-linked to it. The UI pulls
/// the file from this process, so we stay up serving every request until the
/// user interrupts. Single in-memory body, served verbatim on any path.
fn serve_perfetto(trace_json: &[u8], port: u16) -> Result<()> {
    use std::io::Read as _;
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};

    use socket2::{Domain, Protocol, Socket, Type};

    // SO_REUSEADDR before bind: each served connection lingers in TIME_WAIT on
    // this port for ~2*MSL after Ctrl-C, and a plain bind would then fail with
    // EADDRINUSE on the next run. The flag lets us rebind over those.
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .context("creating perfetto serve socket")?;
    socket
        .set_reuse_address(true)
        .context("setting SO_REUSEADDR on perfetto serve socket")?;
    socket
        .bind(&addr.into())
        .with_context(|| format!("binding 127.0.0.1:{port} (is the port already in use?)"))?;
    socket
        .listen(128)
        .with_context(|| format!("listening on 127.0.0.1:{port}"))?;
    let listener = TcpListener::from(socket);
    // Resolve the actual port: with the default (0) the OS assigned an ephemeral
    // one, so read it back rather than echoing the 0.
    let bound_port = listener
        .local_addr()
        .context("reading perfetto serve address")?
        .port();
    let trace_url = format!("http://127.0.0.1:{bound_port}/trace.json");
    let ui_url = format!("https://ui.perfetto.dev/#!/?url={trace_url}");

    open_browser(&ui_url);
    eprintln!("serving {} bytes at {trace_url}", trace_json.len());
    eprintln!("opening {ui_url}");
    eprintln!("Ctrl-C when you're done viewing.");

    // CORS headers on every response: Perfetto fetches cross-origin from
    // ui.perfetto.dev, so the body must be allowed for any origin.
    let header = format!(
        "HTTP/1.1 200 OK\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        trace_json.len()
    );
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(e) => {
                tracing::warn!("perfetto serve: accept failed: {e}");
                continue;
            }
        };
        // Drain the request line/headers so the client's write completes before
        // we reply; we serve the same body regardless of what was asked for.
        let mut scratch = [0u8; 2048];
        let _ = stream.read(&mut scratch);
        if let Err(e) = stream
            .write_all(header.as_bytes())
            .and_then(|()| stream.write_all(trace_json))
        {
            tracing::warn!("perfetto serve: write failed: {e}");
        }
    }
    Ok(())
}

/// Open a URL in the platform browser, best-effort. A failure here is not fatal:
/// the server is already up, so we just print the URL for the user to paste.
fn open_browser(url: &str) {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    if let Err(e) = std::process::Command::new(opener).arg(url).spawn() {
        eprintln!("could not launch browser via `{opener}` ({e}); open this URL yourself:\n{url}");
    }
}

/// Print a calibration report to stdout in the requested format.
fn write_report_as(format: ReportFormat, report: &calibrate::CalibrationReport) -> Result<()> {
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    match format {
        ReportFormat::Json => {
            serde_json::to_writer_pretty(&mut writer, report)
                .context("serializing report to JSON")?;
            writeln!(writer)?;
        }
        ReportFormat::Text => calibrate::write_report(&mut writer, report)?,
    }
    Ok(())
}

/// Wire-level calibration implementation. Spins the real simulator in-process
/// and measures client-side TTFT/ITL.
///
/// Concurrency semantics caveat: TraceLatency samples by the engine's live
/// num_running, which the workload shapes only approximately. Bucket-level
/// comparison is the honest granularity.
async fn calibrate_e2e_impl(
    trace_path: &std::path::Path,
    num_requests: usize,
    seed: u64,
    driver: SimDriver,
    tolerance: f64,
) -> Result<calibrate::CalibrationReport> {
    use std::collections::HashMap;
    use std::time::Instant;

    use futures::StreamExt;
    use rand::Rng;
    use rand::SeedableRng as _;
    use tokio_util::sync::CancellationToken;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
    use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
    use vllm_vcr::latency::{NUM_CONCURRENCY_BUCKETS, concurrency_bucket};
    use vllm_vcr::trace::TraceRecord;
    use vllm_vcr::{Opt, run};

    let (_meta, all_records) = read_trace_file(trace_path)?;

    if all_records.is_empty() {
        anyhow::bail!("trace has no records");
    }

    // Sample N records from the trace (seeded)
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut selected: Vec<&TraceRecord> = Vec::with_capacity(num_requests);
    for _ in 0..num_requests {
        let idx = rng.random_range(0..all_records.len());
        selected.push(&all_records[idx]);
    }

    // Build simulator flags
    let addr = format!("ipc:///tmp/inf-sim-cal-{}-{}.ipc", std::process::id(), seed);
    let trace_path_str = trace_path.to_string_lossy().to_string();

    let mut args: Vec<String> = vec![
        "play".to_string(),
        "--handshake-address".to_string(),
        addr.clone(),
        "--max-num-seqs".to_string(),
        "64".to_string(),
    ];

    if driver == SimDriver::KnobFit {
        let knob = calibrate::fit_knob_from_trace(&all_records);
        args.extend([
            "--time-to-first-token".to_string(),
            knob.time_to_first_token.to_string(),
            "--time-to-first-token-std-dev".to_string(),
            knob.time_to_first_token_std_dev.to_string(),
            "--inter-token-latency".to_string(),
            knob.inter_token_latency.to_string(),
            "--inter-token-latency-std-dev".to_string(),
            knob.inter_token_latency_std_dev.to_string(),
        ]);
    } else {
        args.extend(["--latency-trace".to_string(), trace_path_str]);
    }

    let opt = Opt::parse_from(&args);
    let token = CancellationToken::new();
    let sim_token = token.clone();
    let sim_opt = opt.clone();

    tokio::spawn(async move {
        let _ = run(sim_opt, sim_token).await;
    });

    // Connect client
    let config = EngineCoreClientConfig::new_single(&addr);
    let client = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        EngineCoreClient::connect(config),
    )
    .await
    .map_err(|_| anyhow::anyhow!("client connect timed out"))?
    .context("client connect failed")?;

    let wall_start = Instant::now();

    let mut measured_ttfts: Vec<Vec<f64>> = vec![Vec::new(); NUM_CONCURRENCY_BUCKETS];
    let mut measured_itls: Vec<Vec<f64>> = vec![Vec::new(); NUM_CONCURRENCY_BUCKETS];

    // Process requests in batches by concurrency level for honest measurement.
    let mut by_concurrency: HashMap<u64, Vec<(usize, &TraceRecord)>> = HashMap::new();
    for (i, rec) in selected.iter().enumerate() {
        by_concurrency
            .entry(rec.concurrency)
            .or_default()
            .push((i, rec));
    }

    let mut req_count = 0usize;

    for (concurrency, group) in &by_concurrency {
        let conc = (*concurrency as usize).max(1);
        for chunk in group.chunks(conc) {
            let mut handles = Vec::new();

            for (i, rec) in chunk {
                let max_tokens = rec.output_tokens.min(32) as u32;
                let prompt_len = rec.prompt_tokens;
                let request_id = format!("cal-{i}");

                let request = EngineCoreRequest {
                    request_id: request_id.clone(),
                    prompt_token_ids: Some(calibrate::synthetic_prompt(*i, prompt_len)),
                    sampling_params: Some(EngineCoreSamplingParams {
                        max_tokens,
                        ..EngineCoreSamplingParams::for_test()
                    }),
                    ..Default::default()
                };

                let stream = client.call(request).await.context("call failed")?;
                let cb = concurrency_bucket(rec.concurrency);
                handles.push((cb, stream, Instant::now()));
            }

            for (cb, mut stream, call_start) in handles {
                let mut first_token_time: Option<Instant> = None;
                let mut token_times: Vec<Instant> = Vec::new();

                let timeout_dur = std::time::Duration::from_secs(30);
                let result = tokio::time::timeout(timeout_dur, async {
                    while let Some(item) = stream.next().await {
                        let output = item.context("stream item error")?;
                        let now = Instant::now();
                        if !output.new_token_ids.is_empty() {
                            if first_token_time.is_none() {
                                first_token_time = Some(now);
                            }
                            token_times.push(now);
                        }
                        if output.finish_reason.is_some() {
                            break;
                        }
                    }
                    Ok::<(), anyhow::Error>(())
                })
                .await;

                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::warn!("request stream error: {e:#}");
                        continue;
                    }
                    Err(_) => {
                        tracing::warn!("request timed out");
                        continue;
                    }
                }

                if let Some(first) = first_token_time {
                    let ttft_ms = (first - call_start).as_secs_f64() * 1000.0;
                    measured_ttfts[cb].push(ttft_ms);
                }

                for window in token_times.windows(2) {
                    let gap_ms = (window[1] - window[0]).as_secs_f64() * 1000.0;
                    measured_itls[cb].push(gap_ms);
                }

                req_count += 1;
            }
        }
    }

    let wall_time = wall_start.elapsed();
    token.cancel();

    // Build measured quantiles
    let (measured_buckets, pooled_ttft, pooled_itl) =
        calibrate::buckets_from_measured(measured_ttfts, measured_itls);

    let replay_stats =
        calibrate::quantiles_from_buckets(&measured_buckets, &pooled_ttft, &pooled_itl);

    // Source quantiles from trace
    let source_buckets = calibrate::source_samples_by_bucket(&all_records);
    let (source_pooled_ttft, source_pooled_itl) = calibrate::pool_samples(&all_records);
    let source_stats =
        calibrate::quantiles_from_buckets(&source_buckets, &source_pooled_ttft, &source_pooled_itl);

    let src_pooled = source_stats.last();
    let rep_pooled = replay_stats.last();

    let zero_q = calibrate::Quantiles {
        p50: 0.0,
        p90: 0.0,
        p99: 0.0,
    };
    let (src_ttft_q, src_itl_q) = src_pooled
        .map(|b| (b.ttft, b.itl))
        .unwrap_or((zero_q, zero_q));
    let (rep_ttft_q, rep_itl_q) = rep_pooled
        .map(|b| (b.ttft, b.itl))
        .unwrap_or((zero_q, zero_q));

    let replay_ttft_err = src_ttft_q.max_relative_error(&rep_ttft_q);
    let replay_itl_err = src_itl_q.max_relative_error(&rep_itl_q);
    let replay_pass = replay_ttft_err <= tolerance && replay_itl_err <= tolerance;

    eprintln!(
        "calibrate-e2e: {} requests in {:.1}s (p99 with small N is indicative only)",
        req_count,
        wall_time.as_secs_f64()
    );

    let verdict = calibrate::Verdict {
        source_ttft_tail_ratio: src_ttft_q.tail_ratio(),
        source_itl_tail_ratio: src_itl_q.tail_ratio(),
        replay_ttft_tail_ratio: rep_ttft_q.tail_ratio(),
        replay_itl_tail_ratio: rep_itl_q.tail_ratio(),
        // The e2e harness measures one model per run; there is no separate knob-fit
        // sample set to report (run again with --knob-fit to measure that model).
        knobfit_ttft_tail_ratio: None,
        knobfit_itl_tail_ratio: None,
        replay_ttft_max_error: replay_ttft_err,
        replay_itl_max_error: replay_itl_err,
        replay_pass,
        knobfit_tail_capped: false,
        tolerance,
    };

    Ok(calibrate::CalibrationReport {
        measured_label: match driver {
            SimDriver::KnobFit => "knobfit",
            SimDriver::TraceReplay => "replay",
        }
        .to_string(),
        source: source_stats,
        replay: replay_stats,
        knobfit: None,
        // The e2e harness measures client-side streams; per-request totals are a
        // model-level metric.
        request_total: None,
        verdict,
    })
}
