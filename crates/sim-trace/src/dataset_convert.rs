//! Parse HuggingFace dataset files (JSON/JSONL/CSV/Parquet) into structured data.
//!
//! This module provides utilities to load datasets and extract prompt/response pairs.
//! Used by `HFDatasetTokens` to drive token generation from actual dataset text.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

/// Parse a dataset file (JSON/JSONL/CSV/Parquet) into a vector of JSON values.
pub fn parse_dataset(path: &Path) -> Result<Vec<Value>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "json" | "jsonl" => parse_json_dataset(path),
        "csv" => parse_csv_dataset(path),
        "parquet" => parse_parquet_dataset(path),
        other => bail!(
            "unsupported dataset extension {other:?} in {}; expected json, jsonl, csv, or parquet",
            path.display()
        ),
    }
}

/// Extract prompt text from a dataset row.
pub fn extract_prompt(row: &Value) -> Result<String> {
    let obj = row
        .as_object()
        .context("dataset row must be a JSON object")?;

    if let Some(text) = first_string_field(
        obj,
        &[
            "prompt",
            "instruction",
            "question",
            "query",
            "input_text",
            "context",
        ],
    ) {
        let input = first_string_field(obj, &["input", "inputs"]).unwrap_or_default();
        if input.is_empty() {
            return Ok(text);
        }
        return Ok(format!("{text}\n{input}"));
    }

    if let Some(messages) = obj.get("messages").and_then(Value::as_array) {
        return Ok(format_chat_messages(messages, true));
    }

    if let Some(text) = first_string_field(obj, &["text", "content"]) {
        return Ok(text);
    }

    bail!("dataset row has no recognizable prompt field");
}

/// Extract response text from a dataset row.
pub fn extract_response(row: &Value) -> Result<String> {
    let obj = row
        .as_object()
        .context("dataset row must be a JSON object")?;

    if let Some(text) = first_string_field(
        obj,
        &[
            "output",
            "response",
            "completion",
            "answer",
            "target",
            "label",
        ],
    ) {
        return Ok(text);
    }

    if let Some(messages) = obj.get("messages").and_then(Value::as_array) {
        let assistant = format_chat_messages(messages, false);
        if !assistant.is_empty() {
            return Ok(assistant);
        }
    }

    bail!("dataset row has no recognizable response field");
}

fn parse_json_dataset(path: &Path) -> Result<Vec<Value>> {
    let content = std::fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;
    if content.is_empty() {
        return Ok(Vec::new());
    }

    // JSONL: one object per line.
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"))
        || looks_like_jsonl(&content)
    {
        let mut rows = Vec::new();
        for (line_num, line) in BufReader::new(content.as_slice()).lines().enumerate() {
            let line = line.with_context(|| format!("reading line {}", line_num + 1))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            rows.push(
                serde_json::from_str(trimmed)
                    .with_context(|| format!("line {}: invalid JSON", line_num + 1))?,
            );
        }
        return Ok(rows);
    }

    let value: Value =
        serde_json::from_slice(&content).context("parsing JSON dataset")?;
    rows_from_json_value(value)
}

fn looks_like_jsonl(content: &[u8]) -> bool {
    let text = String::from_utf8_lossy(content);
    let mut non_empty = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let Some(first) = non_empty.next() else {
        return false;
    };
    first.starts_with('{') && non_empty.next().is_some()
}

fn rows_from_json_value(value: Value) -> Result<Vec<Value>> {
    match value {
        Value::Array(rows) => Ok(rows),
        Value::Object(map) => {
            for key in ["train", "validation", "test", "data"] {
                if let Some(Value::Array(rows)) = map.get(key) {
                    return Ok(rows.clone());
                }
            }
            if let Some((_, Value::Array(rows))) = map.into_iter().find(|(_, v)| v.is_array()) {
                return Ok(rows);
            }
            bail!("JSON dataset object has no array split to convert");
        }
        _ => bail!("JSON dataset must be an array or split object"),
    }
}

fn parse_csv_dataset(path: &Path) -> Result<Vec<Value>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("opening CSV {}", path.display()))?;
    let headers = reader
        .headers()
        .context("reading CSV headers")?
        .clone();
    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result.context("reading CSV row")?;
        let mut obj = serde_json::Map::new();
        for (header, field) in headers.iter().zip(record.iter()) {
            obj.insert(header.to_string(), Value::String(field.to_string()));
        }
        rows.push(Value::Object(obj));
    }
    Ok(rows)
}

fn parse_parquet_dataset(path: &Path) -> Result<Vec<Value>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("opening parquet dataset")?;
    let reader = builder.build().context("building parquet reader")?;

    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.context("reading parquet batch")?;
        let schema = batch.schema();
        for row_idx in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                if let Some(text) = parquet_cell_as_string(col, row_idx) {
                    obj.insert(field.name().clone(), Value::String(text));
                }
            }
            rows.push(Value::Object(obj));
        }
    }
    Ok(rows)
}

fn parquet_cell_as_string(col: &dyn Array, row_idx: usize) -> Option<String> {
    if col.is_null(row_idx) {
        return None;
    }
    match col.data_type() {
        arrow::datatypes::DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>()?;
            Some(arr.value(row_idx).to_string())
        }
        arrow::datatypes::DataType::LargeUtf8 => {
            use arrow::array::LargeStringArray;
            let arr = col.as_any().downcast_ref::<LargeStringArray>()?;
            Some(arr.value(row_idx).to_string())
        }
        _ => Some(format!("{:?}", col.slice(row_idx, 1))),
    }
}

fn format_chat_messages(messages: &[Value], user_only: bool) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let Some(obj) = msg.as_object() else {
            continue;
        };
        let role = obj
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let content = obj
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if content.is_empty() {
            continue;
        }
        match (user_only, role.as_str()) {
            (true, "user" | "system") => parts.push(content.to_string()),
            (false, "assistant") => parts.push(content.to_string()),
            _ => {}
        }
    }
    parts.join("\n")
}

fn first_string_field(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| obj.get(*key).and_then(Value::as_str).map(str::to_string))
}

/// Whether `--replay-tokens` should load this path as an in-memory HuggingFace
/// dataset rather than a captured trace JSONL.
pub fn is_dataset_file(path: &Path) -> Result<bool> {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        return Ok(false);
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" | "parquet" => return Ok(true),
        "json" | "jsonl" => {}
        _ => return Ok(false),
    }

    let content = std::fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;
    if content.is_empty() {
        return Ok(false);
    }

    if ext == "jsonl" || looks_like_jsonl(&content) {
        for line in BufReader::new(content.as_slice()).lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed)
                .with_context(|| format!("parsing first line of {}", path.display()))?;
            return Ok(!looks_like_trace_record(&value) && !value.get("meta").is_some());
        }
        return Ok(false);
    }

    let value: Value =
        serde_json::from_slice(&content).context("parsing JSON file")?;
    if value.get("meta").is_some() || value.get("benchmarks").is_some() {
        return Ok(false);
    }
    Ok(!looks_like_trace_json(&value))
}

fn looks_like_trace_json(value: &Value) -> bool {
    match value {
        Value::Array(rows) => rows.first().is_some_and(looks_like_trace_record),
        Value::Object(map) => {
            for key in ["train", "validation", "test", "data"] {
                if let Some(Value::Array(rows)) = map.get(key) {
                    return rows.first().is_some_and(looks_like_trace_record);
                }
            }
            false
        }
        _ => false,
    }
}

fn looks_like_trace_record(value: &Value) -> bool {
    value.get("ttft_ms").and_then(Value::as_f64).is_some()
        && value.get("prompt_tokens").is_some()
        && value.get("output_tokens").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_array() {
        let dir = std::env::temp_dir().join(format!(
            "sim-trace-dataset-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("test.json");
        std::fs::write(
            &input,
            r#"[
              {"instruction":"Say hi","input":"","output":"Hello there"},
              {"instruction":"Count","input":"to three","output":"one two three"}
            ]"#,
        )
        .unwrap();

        let rows = parse_dataset(&input).unwrap();
        assert_eq!(rows.len(), 2);

        let prompt1 = extract_prompt(&rows[0]).unwrap();
        assert_eq!(prompt1, "Say hi");
        let response1 = extract_response(&rows[0]).unwrap();
        assert_eq!(response1, "Hello there");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn is_dataset_file_detects_instruction_json() {
        let dir = std::env::temp_dir().join(format!(
            "sim-trace-dataset-detect-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("data.json");
        std::fs::write(
            &input,
            r#"[{"instruction":"Say hi","output":"Hello there"}]"#,
        )
        .unwrap();
        assert!(is_dataset_file(&input).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn is_dataset_file_rejects_trace_shape_jsonl() {
        let dir = std::env::temp_dir().join(format!(
            "sim-trace-trace-detect-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("trace.jsonl");
        std::fs::write(
            &input,
            r#"{"prompt_tokens":10,"output_tokens":5,"ttft_ms":10.0,"concurrency":1}"#,
        )
        .unwrap();
        assert!(!is_dataset_file(&input).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }
}
