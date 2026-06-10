//! CLI for converting benchmark reports to the inference-sim trace format,
//! summarizing existing traces, and running calibration comparisons.

use std::fs;
use std::io::{self, BufReader, BufWriter, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use inference_simulator_rs::calibrate;
use inference_simulator_rs::trace_convert::{
    ConvertOptions, convert_guidellm, summarize_trace, write_conversion, write_summary,
};

#[derive(Parser)]
#[command(
    name = "inference-sim-trace",
    about = "Convert benchmark reports to inference-sim trace format, summarize traces, and run calibration."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a guidellm benchmark report (JSON) to an inference-sim trace (JSONL).
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
        /// Path to the JSONL trace file.
        input: PathBuf,
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
        /// Generate a fast/small-magnitude trace suitable for e2e testing
        /// (TTFT ~15-40ms, ITL ~3-10ms).
        #[arg(long)]
        fast: bool,
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
        /// Emit JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
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
        /// Number of requests to send through the simulator.
        #[arg(long, default_value_t = 60)]
        requests: usize,
        /// RNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Use knob-fit latency instead of trace replay.
        #[arg(long)]
        knob_fit: bool,
        /// Maximum allowed relative error (looser default for transport jitter).
        #[arg(long, default_value_t = 0.25)]
        tolerance: f64,
        /// Emit JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();

    match cli.command {
        Command::FromGuidellm {
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

            let file = fs::File::create(&output)
                .with_context(|| format!("creating {}", output.display()))?;
            let mut writer = BufWriter::new(file);
            write_conversion(&mut writer, &meta, &records)?;

            eprintln!("wrote {} records to {}", records.len(), output.display());
        }
        Command::Summarize { input } => {
            let file =
                fs::File::open(&input).with_context(|| format!("opening {}", input.display()))?;
            let reader = BufReader::new(file);
            let (meta, stats) = summarize_trace(reader)?;

            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            write_summary(&mut writer, &meta, &stats)?;
        }
        Command::GenDemo {
            output,
            records,
            seed,
            fast,
        } => {
            let (meta, recs) = if fast {
                calibrate::gen_demo_fast(records, seed)
            } else {
                calibrate::gen_demo(records, seed)
            };
            calibrate::write_demo_trace(&output, &meta, &recs)?;
            eprintln!("wrote {} records to {}", recs.len(), output.display());
        }
        Command::Calibrate {
            trace,
            samples,
            seed,
            tolerance,
            json,
            dump_samples,
        } => {
            let report = calibrate::calibrate_from_file(&trace, samples, seed, tolerance)?;

            if let Some(dump_path) = dump_samples {
                let file = fs::File::open(&trace)
                    .with_context(|| format!("opening {}", trace.display()))?;
                let (_meta, records) =
                    inference_simulator_rs::trace::read_trace(BufReader::new(file))?;
                let dump = calibrate::dump_samples(&records, samples, seed)?;
                let out = fs::File::create(&dump_path)
                    .with_context(|| format!("creating {}", dump_path.display()))?;
                serde_json::to_writer(BufWriter::new(out), &dump)
                    .context("serializing sample dump")?;
                eprintln!("wrote sample dump to {}", dump_path.display());
            }

            if json {
                let stdout = io::stdout();
                let mut writer = BufWriter::new(stdout.lock());
                serde_json::to_writer_pretty(&mut writer, &report)
                    .context("serializing report to JSON")?;
                writeln!(writer)?;
            } else {
                let stdout = io::stdout();
                let mut writer = BufWriter::new(stdout.lock());
                calibrate::write_report(&mut writer, &report)?;
            }

            if !report.verdict.replay_pass {
                return Ok(ExitCode::FAILURE);
            }
        }
        Command::CalibrateE2e {
            trace,
            requests,
            seed,
            knob_fit,
            tolerance,
            json,
        } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("building tokio runtime for calibrate-e2e")?;

            let report = runtime.block_on(calibrate_e2e_impl(
                &trace, requests, seed, knob_fit, tolerance,
            ))?;

            if json {
                let stdout = io::stdout();
                let mut writer = BufWriter::new(stdout.lock());
                serde_json::to_writer_pretty(&mut writer, &report)
                    .context("serializing report to JSON")?;
                writeln!(writer)?;
            } else {
                let stdout = io::stdout();
                let mut writer = BufWriter::new(stdout.lock());
                calibrate::write_report(&mut writer, &report)?;
            }

            if !report.verdict.replay_pass {
                return Ok(ExitCode::FAILURE);
            }
        }
    }

    Ok(ExitCode::SUCCESS)
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
    use_knob_fit: bool,
    tolerance: f64,
) -> Result<calibrate::CalibrationReport> {
    use std::collections::HashMap;
    use std::time::Instant;

    use clap::Parser as _;
    use futures::StreamExt;
    use inference_simulator_rs::latency::{
        CONCURRENCY_RANGES, NUM_CONCURRENCY_BUCKETS, concurrency_bucket,
    };
    use inference_simulator_rs::trace::{TraceRecord, read_trace};
    use inference_simulator_rs::{Opt, run};
    use rand::Rng;
    use rand::SeedableRng as _;
    use tokio_util::sync::CancellationToken;
    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
    use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

    let file = std::fs::File::open(trace_path)
        .with_context(|| format!("opening trace: {}", trace_path.display()))?;
    let reader = std::io::BufReader::new(file);
    let (_meta, all_records) =
        read_trace(reader).with_context(|| format!("parsing trace: {}", trace_path.display()))?;

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
        "inference-sim".to_string(),
        "--handshake-address".to_string(),
        addr.clone(),
        "--max-num-seqs".to_string(),
        "64".to_string(),
    ];

    if use_knob_fit {
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
                    prompt_token_ids: Some(vec![42u32; prompt_len]),
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
    let mut measured_buckets = Vec::new();
    let mut pooled_ttft = Vec::new();
    let mut pooled_itl = Vec::new();

    for (i, (ttfts, itls)) in measured_ttfts
        .iter_mut()
        .zip(measured_itls.iter_mut())
        .enumerate()
    {
        if ttfts.is_empty() {
            continue;
        }
        pooled_ttft.extend(ttfts.iter().copied());
        pooled_itl.extend(itls.iter().copied());

        let (lo, hi) = CONCURRENCY_RANGES[i];
        let label = if hi == u64::MAX {
            format!("{lo}+")
        } else if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}-{hi}")
        };
        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        measured_buckets.push((label, ttfts.clone(), itls.clone()));
    }

    pooled_ttft.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    pooled_itl.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        measured_label: if use_knob_fit { "knobfit" } else { "replay" }.to_string(),
        source: source_stats,
        replay: replay_stats,
        knobfit: None,
        // The e2e harness measures client-side streams; per-request totals are a
        // model-level metric.
        request_total: None,
        verdict,
    })
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(io::stderr)
        .init();

    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
