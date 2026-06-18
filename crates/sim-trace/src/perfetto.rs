//! Convert a JSONL trace into the Chrome Trace Event Format that
//! <https://ui.perfetto.dev> ingests. Times are emitted in microseconds (the
//! format's unit), i.e. the trace's milliseconds times 1000. Records without an
//! `arrival_ms` cannot be placed on a timeline and are dropped.

use std::io::Write;

use anyhow::{Context as _, Result};
use serde::Serialize;
use serde_json::json;

use crate::trace::{TraceMeta, TraceRecord};

const PID: u64 = 1;
/// Separate process so the UI groups the step track apart from the request shapes.
const STEP_PID: u64 = 2;

/// One Chrome Trace Event. Phase `X` = duration (`ts`+`dur`), `C` = counter, `M` = metadata.
#[derive(Debug, Serialize)]
struct TraceEvent {
    #[serde(rename = "ph")]
    phase: &'static str,
    pid: u64,
    tid: u64,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cat: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    /// Reserved Perfetto color-palette name; arbitrary hex is ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    cname: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<serde_json::Value>,
}

/// Perfetto reserved color names (not arbitrary hex): warm = prefill, green = decode.
const COLOR_PREFILL: &str = "rail_animation";
const COLOR_DECODE: &str = "thread_state_running";
const COLOR_MIXED: &str = "rail_load";

#[derive(Debug, Serialize)]
struct PerfettoDoc {
    #[serde(rename = "traceEvents")]
    trace_events: Vec<TraceEvent>,
    /// Axis label only; the timestamps themselves are microseconds.
    #[serde(rename = "displayTimeUnit")]
    display_time_unit: &'static str,
}

#[derive(Debug, Clone, Default)]
pub struct PerfettoOptions {
    /// Process-row label; defaults to the trace's model.
    pub process_name: Option<String>,
    /// One track per request instead of packing into reusable lanes.
    pub track_per_request: bool,
}

/// A counter track to overlay on the timeline. `(ts_ms, value)` samples on the
/// same capture-start clock as `arrival_ms`. Kept protocol-free (the sidecar
/// reader hands these in) so this crate avoids the engine-core stats type.
#[derive(Debug, Clone)]
pub struct CounterSeries {
    pub name: String,
    pub samples: Vec<(f64, f64)>,
}

/// One executed scheduler step on the step-centric track. `[start_ms, end_ms]`
/// is its wall-clock interval; protocol-free so the sidecar reader fills it.
#[derive(Debug, Clone)]
pub struct StepSpan {
    pub start_ms: f64,
    pub end_ms: f64,
    pub running: u32,
    pub waiting: u32,
    /// Requests that ran prefill this step (0 ⇒ a pure decode step).
    pub prefill_requests: u32,
    pub prefill_tokens: u32,
    pub kv_cache_usage: f64,
    /// `(accepted, draft)` tokens when spec decoding ran this step.
    pub spec: Option<(u32, u32)>,
}

/// Sidecar-derived overlays: counter tracks plus the step-centric track.
#[derive(Debug, Clone, Default)]
pub struct Overlays {
    pub counters: Vec<CounterSeries>,
    pub steps: Vec<StepSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfettoSummary {
    pub placed_requests: usize,
    pub dropped_requests: usize,
    pub events: usize,
}

/// Milliseconds to microseconds (the Chrome format's time unit).
fn us(ms: f64) -> f64 {
    ms * 1000.0
}

/// Tokens delivered at TTFT. `itl_tokens` holds every *later* chunk's size, so
/// the first chunk owns the remainder (spec-decode/diffusion bursts >1 token);
/// without `itl_tokens` it's a single autoregressive token.
fn first_chunk_tokens(record: &TraceRecord) -> usize {
    match &record.itl_tokens {
        Some(tokens) => {
            let later: usize = tokens.iter().map(|&t| t as usize).sum();
            record.output_tokens.saturating_sub(later)
        }
        None => record.output_tokens.min(1),
    }
}

/// Time the request occupied the timeline: prefill plus every decode gap (or the
/// ITL summary's mean*count when there's no per-gap array).
fn span_duration_ms(record: &TraceRecord) -> f64 {
    let decode_ms = match &record.itl_ms {
        Some(gaps) => gaps.iter().sum(),
        None => record
            .itl_summary
            .as_ref()
            .map(|s| s.mean_ms * s.count as f64)
            .unwrap_or(0.0),
    };
    record.ttft_ms + decode_ms
}

/// The `output_token_ids` covering a chunk, when recorded (`tap --record-tokens`);
/// `None` otherwise so spans stay lean.
fn token_slice(record: &TraceRecord, start: usize, len: usize) -> Option<Vec<u32>> {
    let ids = record.output_token_ids.as_ref()?;
    let end = (start + len).min(ids.len());
    ids.get(start..end).map(<[u32]>::to_vec)
}

/// Emit a request's prefill + per-gap decode spans on track `tid`, appending its
/// per-gap `running`/`prefill` counter samples (absolute time, merged later).
/// Every span carries `index` since a lane may be shared across requests.
fn emit_request(
    events: &mut Vec<TraceEvent>,
    running: &mut Vec<(f64, f64)>,
    prefill: &mut Vec<(f64, f64)>,
    record: &TraceRecord,
    tid: u64,
    index: usize,
) {
    let arrival = record.arrival_ms.unwrap_or(0.0);

    // Prefill: arrival -> first token.
    let first_tokens = first_chunk_tokens(record);
    let mut prefill_args = json!({
        "req": index,
        "prompt_tokens": record.prompt_tokens,
        "cached_tokens": record.cached_tokens,
        "output_tokens": record.output_tokens,
        "ttft_ms": record.ttft_ms,
        "first_chunk_tokens": first_tokens,
    });
    if let Some(reason) = record.finish_reason {
        prefill_args["finish_reason"] = json!(reason);
    }
    if let Some(ids) = token_slice(record, 0, first_tokens) {
        prefill_args["token_ids"] = json!(ids);
    }
    events.push(TraceEvent {
        phase: "X",
        pid: PID,
        tid,
        name: format!("req{index} prefill"),
        cat: Some("prefill"),
        ts: Some(us(arrival)),
        dur: Some(us(record.ttft_ms)),
        cname: Some(COLOR_PREFILL),
        args: Some(prefill_args),
    });

    // Decode: one span per gap, walking the cumulative clock from first token.
    let Some(gaps) = record.itl_ms.as_ref() else {
        return;
    };
    let ctx = record.itl_ctx.as_ref();
    let mut clock = arrival + record.ttft_ms;
    let mut token_cursor = first_tokens;
    for (i, &gap_ms) in gaps.iter().enumerate() {
        let chunk_tokens = record
            .itl_tokens
            .as_ref()
            .and_then(|t| t.get(i))
            .map(|&t| t as usize)
            .unwrap_or(1);

        let mut args = json!({ "req": index, "gap_ms": gap_ms, "tokens": chunk_tokens });
        if let Some(ctx) = ctx {
            if let Some(&n) = ctx.num_running.get(i) {
                args["num_running"] = json!(n);
                running.push((clock + gap_ms, f64::from(n)));
            }
            if let Some(&p) = ctx.prefill_tokens.get(i) {
                args["prefill_tokens"] = json!(p);
                prefill.push((clock + gap_ms, f64::from(p)));
            }
        }
        if let Some(ids) = token_slice(record, token_cursor, chunk_tokens) {
            args["token_ids"] = json!(ids);
        }

        let name = if chunk_tokens == 1 {
            "decode".to_string()
        } else {
            format!("decode x{chunk_tokens}")
        };
        events.push(TraceEvent {
            phase: "X",
            pid: PID,
            tid,
            name,
            cat: Some("decode"),
            ts: Some(us(clock)),
            dur: Some(us(gap_ms)),
            cname: Some(COLOR_DECODE),
            args: Some(args),
        });

        clock += gap_ms;
        token_cursor += chunk_tokens;
    }
}

/// Greedy interval partitioning: assign each request (in arrival order) to the
/// lowest-index free lane, opening a new one only when none is free, so the lane
/// count is the peak concurrency. A lane frees the instant its request ends
/// (`end <= arrival`), matching the tie rule in [`active_request_counter_refs`]
/// (exits before entries), so the lane count is exactly the counter's peak.
fn assign_lanes(records: &[&TraceRecord]) -> (Vec<usize>, usize) {
    let mut lane_end: Vec<f64> = Vec::new();
    let mut lane_of = Vec::with_capacity(records.len());
    for record in records {
        let arrival = record.arrival_ms.unwrap_or(0.0);
        let end = arrival + span_duration_ms(record);
        let free = lane_end.iter().position(|&e| e <= arrival);
        let lane = match free {
            Some(i) => {
                lane_end[i] = end;
                i
            }
            None => {
                lane_end.push(end);
                lane_end.len() - 1
            }
        };
        lane_of.push(lane);
    }
    let lanes = lane_end.len();
    (lane_of, lanes)
}

/// Sweep request spans into an `active_requests` step function: +1 at each
/// arrival, -1 at each completion, sampled at every change.
fn active_request_counter_refs(records: &[&TraceRecord]) -> Vec<(f64, f64)> {
    let mut deltas: Vec<(f64, i64)> = Vec::with_capacity(records.len() * 2);
    for record in records {
        let arrival = record.arrival_ms.unwrap_or(0.0);
        deltas.push((arrival, 1));
        deltas.push((arrival + span_duration_ms(record), -1));
    }
    // Exits before entries at a tie, so a hand-off isn't a phantom +1 spike.
    deltas.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut samples = Vec::new();
    let mut active: i64 = 0;
    for (ms, delta) in deltas {
        active += delta;
        samples.push((ms, active.max(0) as f64));
    }
    samples
}

/// Append a counter track: one `C` event per `(ms, value)` sample, sorted by time.
fn emit_counter(events: &mut Vec<TraceEvent>, name: &str, mut samples: Vec<(f64, f64)>) {
    if samples.is_empty() {
        return;
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (ms, value) in samples {
        events.push(TraceEvent {
            phase: "C",
            pid: PID,
            tid: 0,
            name: name.to_string(),
            cat: None,
            ts: Some(us(ms)),
            dur: None,
            cname: None,
            args: Some(json!({ name: value })),
        });
    }
}

/// Emit the step-centric track: one span per scheduler step, back to back on a
/// single row under a dedicated process. Steps are sequential so they never
/// overlap, unlike the per-request shapes (which reconstruct latency envelopes).
fn emit_step_track(events: &mut Vec<TraceEvent>, steps: &[StepSpan]) {
    if steps.is_empty() {
        return;
    }
    events.push(TraceEvent {
        phase: "M",
        pid: STEP_PID,
        tid: 0,
        name: "process_name".to_string(),
        cat: None,
        ts: None,
        dur: None,
        cname: None,
        args: Some(json!({ "name": "scheduler steps" })),
    });
    events.push(TraceEvent {
        phase: "M",
        pid: STEP_PID,
        tid: 1,
        name: "thread_name".to_string(),
        cat: None,
        ts: None,
        dur: None,
        cname: None,
        args: Some(json!({ "name": "steps" })),
    });

    for step in steps {
        let dur_ms = (step.end_ms - step.start_ms).max(0.0);
        let prefilling = step.prefill_requests > 0 || step.prefill_tokens > 0;
        let (name, cat, cname) = if prefilling && step.running > 0 {
            (
                format!(
                    "prefill+decode B{} (+{}r {}t)",
                    step.running, step.prefill_requests, step.prefill_tokens
                ),
                "step_prefill",
                COLOR_MIXED,
            )
        } else if prefilling {
            (
                format!(
                    "prefill (+{}r {}t)",
                    step.prefill_requests, step.prefill_tokens
                ),
                "step_prefill",
                COLOR_PREFILL,
            )
        } else {
            (
                format!("decode B{}", step.running),
                "step_decode",
                COLOR_DECODE,
            )
        };

        let mut args = json!({
            "running": step.running,
            "waiting": step.waiting,
            "prefill_requests": step.prefill_requests,
            "prefill_tokens": step.prefill_tokens,
            "kv_cache_usage": step.kv_cache_usage,
            "step_ms": dur_ms,
        });
        if let Some((accepted, draft)) = step.spec {
            args["accepted"] = json!(accepted);
            args["draft"] = json!(draft);
        }
        events.push(TraceEvent {
            phase: "X",
            pid: STEP_PID,
            tid: 1,
            name,
            cat: Some(cat),
            ts: Some(us(step.start_ms)),
            dur: Some(us(dur_ms)),
            cname: Some(cname),
            args: Some(args),
        });
    }
}

/// Convert a trace to a Chrome/Perfetto JSON document and write it to `writer`.
pub fn write_perfetto(
    writer: &mut impl Write,
    meta: &TraceMeta,
    records: &[TraceRecord],
    overlays: &Overlays,
    opts: &PerfettoOptions,
) -> Result<PerfettoSummary> {
    // Arrival order, dropping records that cannot be placed (no arrival time).
    let mut placed: Vec<&TraceRecord> = records.iter().filter(|r| r.arrival_ms.is_some()).collect();
    placed.sort_by(|a, b| {
        a.arrival_ms
            .partial_cmp(&b.arrival_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let dropped_requests = records.len() - placed.len();

    let mut events = Vec::new();

    let process_name = opts.process_name.clone().unwrap_or_else(|| {
        meta.model
            .clone()
            .map(|m| format!("inference trace: {m}"))
            .unwrap_or_else(|| "inference trace".to_string())
    });
    events.push(TraceEvent {
        phase: "M",
        pid: PID,
        tid: 0,
        name: "process_name".to_string(),
        cat: None,
        ts: None,
        dur: None,
        cname: None,
        args: Some(json!({ "name": process_name })),
    });

    // Lane packing (default) reuses a track once its request finishes; tid 0
    // carries the process/counter tracks, so request tracks start at 1.
    let (track_of, track_count) = if opts.track_per_request {
        ((0..placed.len()).collect::<Vec<_>>(), placed.len())
    } else {
        assign_lanes(&placed)
    };

    let track_labels: Vec<String> = if opts.track_per_request {
        placed
            .iter()
            .enumerate()
            .map(|(i, r)| format!("req {i} (p{} o{})", r.prompt_tokens, r.output_tokens))
            .collect()
    } else {
        (0..track_count)
            .map(|lane| format!("lane {lane}"))
            .collect()
    };
    for (track, label) in track_labels.into_iter().enumerate() {
        events.push(TraceEvent {
            phase: "M",
            pid: PID,
            tid: track as u64 + 1,
            name: "thread_name".to_string(),
            cat: None,
            ts: None,
            dur: None,
            cname: None,
            args: Some(json!({ "name": label })),
        });
    }

    let mut running = Vec::new();
    let mut prefill = Vec::new();
    for (index, record) in placed.iter().enumerate() {
        let tid = track_of[index] as u64 + 1;
        emit_request(&mut events, &mut running, &mut prefill, record, tid, index);
    }

    emit_counter(
        &mut events,
        "active_requests",
        active_request_counter_refs(&placed),
    );
    emit_counter(&mut events, "engine_running", running);
    emit_counter(&mut events, "prefill_tokens", prefill);

    for series in &overlays.counters {
        emit_counter(&mut events, &series.name, series.samples.clone());
    }
    emit_step_track(&mut events, &overlays.steps);

    let summary = PerfettoSummary {
        placed_requests: placed.len(),
        dropped_requests,
        events: events.len(),
    };

    let doc = PerfettoDoc {
        trace_events: events,
        display_time_unit: "ms",
    };
    serde_json::to_writer(&mut *writer, &doc).context("serializing perfetto trace")?;
    writeln!(writer).context("writing perfetto trace newline")?;

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::trace::{ItlContext, ItlSummary, TraceFinishReason};

    /// Convert and parse back into JSON for inspection.
    fn convert(records: &[TraceRecord]) -> (PerfettoSummary, Value) {
        convert_overlays(records, &Overlays::default())
    }

    /// Convert with sidecar overlays (counters and/or step spans) injected.
    fn convert_overlays(records: &[TraceRecord], overlays: &Overlays) -> (PerfettoSummary, Value) {
        let mut buf = Vec::new();
        let summary = write_perfetto(
            &mut buf,
            &TraceMeta::default(),
            records,
            overlays,
            &PerfettoOptions::default(),
        )
        .unwrap();
        let doc: Value = serde_json::from_slice(&buf).unwrap();
        (summary, doc)
    }

    fn events(doc: &Value) -> &Vec<Value> {
        doc["traceEvents"].as_array().unwrap()
    }

    /// All complete (`ph:"X"`) events with the given name.
    fn spans<'a>(doc: &'a Value, name: &str) -> Vec<&'a Value> {
        events(doc)
            .iter()
            .filter(|e| e["ph"] == "X" && e["name"] == name)
            .collect()
    }

    /// Prefill spans, matched by category (the name carries the req index).
    fn prefill_spans(doc: &Value) -> Vec<&Value> {
        events(doc)
            .iter()
            .filter(|e| e["ph"] == "X" && e["cat"] == "prefill")
            .collect()
    }

    /// thread_name labels in tid order (the track rows).
    fn track_names(doc: &Value) -> Vec<String> {
        let mut named: Vec<(u64, String)> = events(doc)
            .iter()
            .filter(|e| e["name"] == "thread_name")
            .map(|e| {
                (
                    e["tid"].as_u64().unwrap(),
                    e["args"]["name"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        named.sort_by_key(|(tid, _)| *tid);
        named.into_iter().map(|(_, n)| n).collect()
    }

    /// All counter (`ph:"C"`) samples for a series, as (ts, value).
    fn counter(doc: &Value, name: &str) -> Vec<(f64, f64)> {
        events(doc)
            .iter()
            .filter(|e| e["ph"] == "C" && e["name"] == name)
            .map(|e| (e["ts"].as_f64().unwrap(), e["args"][name].as_f64().unwrap()))
            .collect()
    }

    #[test]
    fn prefill_and_decode_land_on_the_microsecond_timeline() {
        let record = TraceRecord {
            prompt_tokens: 100,
            output_tokens: 3,
            ttft_ms: 50.0,
            itl_ms: Some(vec![10.0, 12.0]),
            arrival_ms: Some(1000.0),
            ..Default::default()
        };
        let (summary, doc) = convert(&[record]);
        assert_eq!(summary.placed_requests, 1);
        assert_eq!(summary.dropped_requests, 0);
        assert_eq!(doc["displayTimeUnit"], "ms");

        // Prefill: arrival 1000ms -> ts 1_000_000us, dur 50ms -> 50_000us.
        let prefill = prefill_spans(&doc);
        assert_eq!(prefill.len(), 1);
        assert_eq!(prefill[0]["ts"].as_f64().unwrap(), 1_000_000.0);
        assert_eq!(prefill[0]["dur"].as_f64().unwrap(), 50_000.0);

        // Prefill and decode get distinct phase colors so they don't blend.
        assert_eq!(prefill[0]["cname"], "rail_animation");

        // Two decode gaps, walking the clock from first token at 1050ms.
        let decode = spans(&doc, "decode");
        assert_eq!(decode.len(), 2);
        assert_eq!(decode[0]["cname"], "thread_state_running");
        assert_eq!(decode[0]["ts"].as_f64().unwrap(), 1_050_000.0);
        assert_eq!(decode[0]["dur"].as_f64().unwrap(), 10_000.0);
        assert_eq!(decode[1]["ts"].as_f64().unwrap(), 1_060_000.0);
        assert_eq!(decode[1]["dur"].as_f64().unwrap(), 12_000.0);
    }

    #[test]
    fn records_without_arrival_are_dropped() {
        let placed = TraceRecord {
            prompt_tokens: 10,
            output_tokens: 1,
            ttft_ms: 5.0,
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let orphan = TraceRecord {
            arrival_ms: None,
            ..placed.clone()
        };
        let (summary, doc) = convert(&[placed, orphan]);
        assert_eq!(summary.placed_requests, 1);
        assert_eq!(summary.dropped_requests, 1);
        // Only one request track (one prefill span).
        assert_eq!(prefill_spans(&doc).len(), 1);
    }

    #[test]
    fn overlapping_requests_drive_active_counter_to_two() {
        // Two requests, second arrives while the first is still decoding.
        let a = TraceRecord {
            output_tokens: 2,
            ttft_ms: 10.0,
            itl_ms: Some(vec![100.0]),
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let b = TraceRecord {
            output_tokens: 1,
            ttft_ms: 10.0,
            arrival_ms: Some(20.0),
            ..Default::default()
        };
        let (_, doc) = convert(&[a, b]);
        let samples = counter(&doc, "active_requests");
        let peak = samples.iter().map(|&(_, v)| v).fold(0.0_f64, f64::max);
        assert_eq!(peak, 2.0, "the two requests overlap, so depth hits 2");
        // Ends back at zero once both finish.
        assert_eq!(samples.last().unwrap().1, 0.0);
    }

    /// Build N requests on a fixed pitch with a fixed footprint, so the overlap
    /// (and thus the lane count) is controllable: a request occupies
    /// `ttft + itl` ms starting at `i * pitch`.
    fn staggered(n: usize, pitch_ms: f64, ttft_ms: f64, decode_ms: f64) -> Vec<TraceRecord> {
        (0..n)
            .map(|i| TraceRecord {
                output_tokens: 2,
                ttft_ms,
                itl_ms: Some(vec![decode_ms]),
                arrival_ms: Some(i as f64 * pitch_ms),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn lane_packing_collapses_rows_to_peak_concurrency() {
        // 10 requests, each lives 100ms, arriving every 25ms => up to 4 overlap.
        let records = staggered(10, 25.0, 40.0, 60.0);
        let (_, doc) = convert(&records);

        let lanes = track_names(&doc);
        let peak = counter(&doc, "active_requests")
            .iter()
            .map(|&(_, v)| v)
            .fold(0.0_f64, f64::max) as usize;
        // Far fewer rows than requests, and exactly the peak concurrency.
        assert!(lanes.len() < 10, "should pack below the request count");
        assert_eq!(lanes.len(), peak, "lane count must equal peak concurrency");
        assert!(lanes.iter().all(|n| n.starts_with("lane ")));
        // Every request still emitted its prefill, just packed onto the lanes.
        assert_eq!(prefill_spans(&doc).len(), 10);
    }

    #[test]
    fn packed_lanes_never_have_overlapping_spans() {
        let records = staggered(20, 15.0, 30.0, 50.0);
        let (_, doc) = convert(&records);
        // Group complete spans by track and assert each track is non-overlapping.
        let mut by_tid: std::collections::HashMap<u64, Vec<(f64, f64)>> = Default::default();
        for e in events(&doc).iter().filter(|e| e["ph"] == "X") {
            by_tid
                .entry(e["tid"].as_u64().unwrap())
                .or_default()
                .push((e["ts"].as_f64().unwrap(), e["dur"].as_f64().unwrap()));
        }
        for spans in by_tid.values_mut() {
            spans.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for w in spans.windows(2) {
                assert!(
                    w[0].0 + w[0].1 <= w[1].0 + 1e-6,
                    "packed lane has overlapping spans: {:?} then {:?}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn track_per_request_keeps_one_labelled_row_each() {
        let records = staggered(5, 5.0, 30.0, 50.0); // heavy overlap
        let mut buf = Vec::new();
        write_perfetto(
            &mut buf,
            &TraceMeta::default(),
            &records,
            &Overlays::default(),
            &PerfettoOptions {
                track_per_request: true,
                ..Default::default()
            },
        )
        .unwrap();
        let doc: Value = serde_json::from_slice(&buf).unwrap();
        let names = track_names(&doc);
        assert_eq!(names.len(), 5, "one row per request");
        assert!(names.iter().all(|n| n.starts_with("req ")));
    }

    #[test]
    fn itl_ctx_becomes_engine_counters_and_span_args() {
        let record = TraceRecord {
            prompt_tokens: 800,
            output_tokens: 3,
            ttft_ms: 20.0,
            itl_ms: Some(vec![5.0, 6.0]),
            itl_ctx: Some(ItlContext {
                num_running: vec![4, 4],
                prefill_tokens: vec![0, 800],
            }),
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let (_, doc) = convert(&[record]);

        let running = counter(&doc, "engine_running");
        assert_eq!(
            running.iter().map(|&(_, v)| v).collect::<Vec<_>>(),
            vec![4.0, 4.0]
        );
        let prefill = counter(&doc, "prefill_tokens");
        assert_eq!(
            prefill.iter().map(|&(_, v)| v).collect::<Vec<_>>(),
            vec![0.0, 800.0]
        );

        // The prefill-interfered second gap carries it on the span too.
        let decode = spans(&doc, "decode");
        assert_eq!(decode[1]["args"]["prefill_tokens"], 800);
        assert_eq!(decode[1]["args"]["num_running"], 4);
    }

    #[test]
    fn spec_decode_chunks_name_and_split_tokens() {
        // 1 first token + chunks of 4 and 1 = 6 output tokens over two gaps.
        let record = TraceRecord {
            output_tokens: 6,
            ttft_ms: 10.0,
            itl_ms: Some(vec![5.0, 6.0]),
            itl_tokens: Some(vec![4, 1]),
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let (_, doc) = convert(&[record]);
        // First chunk owns the remainder: 6 - (4+1) = 1.
        assert_eq!(prefill_spans(&doc)[0]["args"]["first_chunk_tokens"], 1);
        // The 4-token chunk is named, the 1-token chunk is plain "decode".
        assert_eq!(spans(&doc, "decode x4").len(), 1);
        assert_eq!(spans(&doc, "decode").len(), 1);
    }

    #[test]
    fn recorded_token_ids_annotate_spans_for_detokenize_mode() {
        let record = TraceRecord {
            output_tokens: 3,
            ttft_ms: 10.0,
            itl_ms: Some(vec![5.0, 6.0]),
            output_token_ids: Some(vec![7, 8, 9]),
            finish_reason: Some(TraceFinishReason::Stop),
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let (_, doc) = convert(&[record]);
        // First token id rides the prefill span, the rest map to their gaps.
        assert_eq!(prefill_spans(&doc)[0]["args"]["token_ids"], json!([7]));
        assert_eq!(prefill_spans(&doc)[0]["args"]["finish_reason"], "stop");
        let decode = spans(&doc, "decode");
        assert_eq!(decode[0]["args"]["token_ids"], json!([8]));
        assert_eq!(decode[1]["args"]["token_ids"], json!([9]));
    }

    #[test]
    fn injected_counters_become_their_own_tracks() {
        let record = TraceRecord {
            output_tokens: 1,
            ttft_ms: 10.0,
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        // Step-stats-style series the tap pulls from the sidecar.
        let extra = vec![
            CounterSeries {
                name: "kv_cache_usage".to_string(),
                samples: vec![(0.0, 0.1), (5.0, 0.4)],
            },
            CounterSeries {
                name: "waiting_reqs".to_string(),
                samples: vec![(2.0, 3.0)],
            },
        ];
        let overlays = Overlays {
            counters: extra,
            steps: Vec::new(),
        };
        let (_, doc) = convert_overlays(&[record], &overlays);
        assert_eq!(
            counter(&doc, "kv_cache_usage"),
            vec![(0.0, 0.1), (us(5.0), 0.4)]
        );
        assert_eq!(counter(&doc, "waiting_reqs"), vec![(us(2.0), 3.0)]);
    }

    #[test]
    fn step_track_renders_prefill_and_decode_steps() {
        let record = TraceRecord {
            output_tokens: 1,
            ttft_ms: 10.0,
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let overlays = Overlays {
            counters: Vec::new(),
            steps: vec![
                // A pure decode step over [0,12]ms with batch 16.
                StepSpan {
                    start_ms: 0.0,
                    end_ms: 12.0,
                    running: 16,
                    waiting: 0,
                    prefill_requests: 0,
                    prefill_tokens: 0,
                    kv_cache_usage: 0.1,
                    spec: Some((40, 60)),
                },
                // A prefill+decode step over [12,51]ms (wider) with 2 prefills.
                StepSpan {
                    start_ms: 12.0,
                    end_ms: 51.0,
                    running: 14,
                    waiting: 1,
                    prefill_requests: 2,
                    prefill_tokens: 1600,
                    kv_cache_usage: 0.12,
                    spec: None,
                },
            ],
        };
        let (_, doc) = convert_overlays(&[record], &overlays);

        // The step track is its own process, distinct from the request lanes.
        let step_events: Vec<&Value> = events(&doc)
            .iter()
            .filter(|e| e["pid"] == 2 && e["ph"] == "X")
            .collect();
        assert_eq!(step_events.len(), 2);
        // Decode step: narrow, named by batch.
        assert_eq!(step_events[0]["name"], "decode B16");
        assert_eq!(step_events[0]["ts"].as_f64().unwrap(), 0.0);
        assert_eq!(step_events[0]["dur"].as_f64().unwrap(), 12_000.0);
        assert_eq!(step_events[0]["args"]["accepted"], 40);
        assert_eq!(step_events[0]["cname"], "thread_state_running");
        // Prefill step: wider, tagged prefill+decode with request/token counts.
        assert_eq!(step_events[1]["name"], "prefill+decode B14 (+2r 1600t)");
        assert_eq!(step_events[1]["dur"].as_f64().unwrap(), 39_000.0);
        assert_eq!(step_events[1]["args"]["prefill_tokens"], 1600);
        // A mixed step gets the distinct "mixed" shade.
        assert_eq!(step_events[1]["cname"], "rail_load");
    }

    #[test]
    fn summary_only_records_get_a_prefill_but_no_decode_spans() {
        // No per-gap array: the request still places (prefill + a span_duration
        // derived from the summary), just without decode slices.
        let record = TraceRecord {
            output_tokens: 10,
            ttft_ms: 30.0,
            itl_summary: Some(ItlSummary {
                mean_ms: 15.0,
                count: 9,
            }),
            arrival_ms: Some(0.0),
            ..Default::default()
        };
        let (summary, doc) = convert(&[record]);
        assert_eq!(summary.placed_requests, 1);
        assert_eq!(prefill_spans(&doc).len(), 1);
        assert_eq!(spans(&doc, "decode").len(), 0);
        // The span still occupies its full modeled width in the active counter:
        // 30 + 15*9 = 165ms.
        let samples = counter(&doc, "active_requests");
        assert_eq!(samples[0], (0.0, 1.0));
        assert_eq!(samples.last().unwrap().0, us(165.0));
    }
}
