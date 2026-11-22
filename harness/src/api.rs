use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Duration;
use tokio::sync::watch;

use crate::config::EffortLevel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            Self::Parts(_) => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::Text(s) => Some(s),
            Self::Parts(_) => None,
        }
    }

    /// Estimate token cost accurately (avoiding treating raw image base64 bytes as text tokens).
    pub fn estimated_tokens(&self) -> usize {
        match self {
            Self::Text(s) => s.len() / 3,
            Self::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => text.len() / 3,
                    ContentPart::ImageUrl { .. } => 768,
                })
                .sum(),
        }
    }

    /// Concise human-readable text rendering for LLM summary compaction (never dumps base64).
    pub fn summary_text(&self, max_chars: usize) -> String {
        match self {
            Self::Text(s) => {
                let mut out: String = s.chars().take(max_chars).collect();
                if s.chars().count() > max_chars {
                    out.push_str(" ...[truncated]");
                }
                out
            }
            Self::Parts(parts) => {
                let mut out = String::new();
                for p in parts {
                    match p {
                        ContentPart::ImageUrl { .. } => out.push_str("[image] "),
                        ContentPart::Text { text } => out.push_str(text),
                    }
                }
                let mut trimmed: String = out.chars().take(max_chars).collect();
                if out.chars().count() > max_chars {
                    trimmed.push_str(" ...[truncated]");
                }
                trimmed
            }
        }
    }

    /// Replace embedded base64 images with compact text placeholders during compaction.
    pub fn strip_images(&mut self) -> usize {
        if let Self::Parts(parts) = self {
            let mut removed = 0usize;
            for p in parts.iter_mut() {
                if matches!(p, ContentPart::ImageUrl { .. }) {
                    *p = ContentPart::Text {
                        text: "[earlier image omitted during compaction]".to_string(),
                    };
                    removed += 1;
                }
            }
            return removed;
        }
        0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(content.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Text(content.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user_multimodal(text: String, image_data_urls: Vec<String>) -> Self {
        if image_data_urls.is_empty() {
            return Self::user(text);
        }
        let mut parts: Vec<ContentPart> = image_data_urls
            .into_iter()
            .map(|url| ContentPart::ImageUrl {
                image_url: ImageUrl { url },
            })
            .collect();
        if !text.trim().is_empty() {
            parts.push(ContentPart::Text { text });
        }
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(parts)),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(
        content: String,
        reasoning: String,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        let content_opt = if content.is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(MessageContent::Text(content))
        };
        let reasoning_opt = if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        };
        let tc_opt = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };
        Self {
            role: "assistant".to_string(),
            content: content_opt,
            reasoning_content: reasoning_opt,
            tool_calls: tc_opt,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timings {
    #[serde(default)]
    pub prompt_n: u64,
    #[serde(default)]
    pub prompt_per_second: f64,
    #[serde(default)]
    pub predicted_n: u64,
    #[serde(default)]
    pub predicted_per_second: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct LoraAdapter {
    pub id: i32,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct ServerProps {
    pub model_alias: Option<String>,
    pub n_ctx: Option<usize>,
    pub has_vision: bool,
}

#[derive(Debug, Clone)]
pub struct ChatRequestOptions {
    pub model: String,
    pub effort: EffortLevel,
    pub lora_scale: Option<f32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub max_tokens: i32,
}

#[derive(Debug, Clone)]
pub enum StreamDelta {
    Reasoning(String),
    Content(String),
    Timings(Timings),
}

#[derive(Debug, Clone, Default)]
pub struct StreamOutcome {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub timings: Option<Timings>,
    pub usage: Option<Usage>,
    pub cancelled: bool,
}

#[derive(Debug, Clone)]
pub struct ApiClient {
    base_url: String,
    api_key: String,
    http: Client,
    control_http: Client,
}

impl ApiClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to initialize HTTP client")?;
        let control_http = Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(10))
            .build()
            .context("Failed to initialize control HTTP client")?;
        Ok(Self {
            base_url,
            api_key: api_key.into(),
            http,
            control_http,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.api_key)
        }
    }

    pub async fn health(&self) -> Result<String> {
        let url = format!("{}/health", self.base_url);
        let resp = self
            .apply_auth(self.control_http.get(&url))
            .send()
            .await
            .with_context(|| format!("Failed to reach llama-server at {url}"))?;
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if status.is_success() {
            Ok(body
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("ok")
                .to_string())
        } else {
            let msg = extract_error_msg(&body).unwrap_or_else(|| format!("HTTP {status}"));
            bail!("{msg}")
        }
    }

    pub async fn props(&self) -> Result<ServerProps> {
        let url = format!("{}/props", self.base_url);
        let resp = self
            .apply_auth(self.control_http.get(&url))
            .send()
            .await?;
        let status = resp.status();
        let body: Value = resp.json().await?;
        if !status.is_success() {
            bail!("GET /props failed: HTTP {status}");
        }
        let model_alias = body
            .get("model_alias")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(ToString::to_string);
        let n_ctx = body
            .pointer("/default_generation_settings/n_ctx")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let has_vision = body
            .pointer("/modalities/vision")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(ServerProps {
            model_alias,
            n_ctx,
            has_vision,
        })
    }

    pub async fn get_lora_adapters(&self) -> Result<Vec<LoraAdapter>> {
        let url = format!("{}/lora-adapters", self.base_url);
        let resp = self
            .apply_auth(self.control_http.get(&url))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            bail!("GET /lora-adapters failed: HTTP {status}");
        }
        let adapters: Vec<LoraAdapter> = resp.json().await?;
        Ok(adapters)
    }

    pub async fn set_lora_scale(&self, scale: f32, adapter_id: i32) -> Result<Vec<LoraAdapter>> {
        let url = format!("{}/lora-adapters", self.base_url);
        let payload = json!([{ "id": adapter_id, "scale": scale }]);
        let resp = self
            .apply_auth(self.control_http.post(&url))
            .json(&payload)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body: Value = resp.json().await.unwrap_or(Value::Null);
            let msg = extract_error_msg(&body).unwrap_or_else(|| format!("HTTP {status}"));
            bail!("POST /lora-adapters failed: {msg}");
        }
        self.get_lora_adapters().await
    }

    pub async fn stream_chat_completion<F>(
        &self,
        messages: &[Message],
        tools: &[Value],
        opts: &ChatRequestOptions,
        mut cancel_rx: watch::Receiver<bool>,
        mut on_delta: F,
    ) -> Result<StreamOutcome>
    where
        F: FnMut(StreamDelta),
    {
        let mut body = json!({
            "model": opts.model,
            "messages": messages,
            "stream": true,
            "timings_per_token": true,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            "top_k": opts.top_k,
            "min_p": opts.min_p,
        });

        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = json!("auto");
        }
        if opts.effort != EffortLevel::Default {
            body["reasoning_effort"] = json!(opts.effort.as_str());
        }
        if let Some(scale) = opts.lora_scale {
            body["lora"] = json!([{ "id": 0, "scale": scale }]);
        }
        if opts.max_tokens > 0 {
            body["max_tokens"] = json!(opts.max_tokens);
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        let req = self
            .apply_auth(self.http.post(&url))
            .header("Accept", "text/event-stream")
            .json(&body);

        let resp = tokio::select! {
            _ = wait_cancelled(&mut cancel_rx) => {
                return Ok(StreamOutcome { cancelled: true, ..Default::default() });
            }
            res = req.send() => {
                res.with_context(|| format!("Failed to POST {url}"))?
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let raw = resp.text().await.unwrap_or_default();
            let parsed: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            let msg = extract_error_msg(&parsed).unwrap_or_else(|| {
                if raw.is_empty() {
                    format!("HTTP {status}")
                } else {
                    raw
                }
            });
            return Err(anyhow!("Server returned HTTP {status}: {msg}"));
        }

        let mut outcome = StreamOutcome::default();
        let mut tool_acc: BTreeMap<usize, PartialToolCall> = BTreeMap::new();
        let mut byte_stream = resp.bytes_stream();
        let mut line_buf: Vec<u8> = Vec::with_capacity(4096);

        loop {
            let next_chunk = tokio::select! {
                _ = wait_cancelled(&mut cancel_rx) => {
                    outcome.cancelled = true;
                    break;
                }
                chunk = byte_stream.next() => chunk,
            };

            let Some(chunk_res) = next_chunk else {
                break;
            };
            let chunk = chunk_res.context("Stream read error from llama-server")?;
            line_buf.extend_from_slice(&chunk);

            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let raw_line: Vec<u8> = line_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&raw_line).trim().to_string();
                if process_sse_line(&line, &mut outcome, &mut tool_acc, &mut on_delta)? {
                    outcome.tool_calls = finalize_tool_calls(tool_acc);
                    return Ok(outcome);
                }
            }
        }

        if !line_buf.is_empty() {
            let line = String::from_utf8_lossy(&line_buf).trim().to_string();
            let _ = process_sse_line(&line, &mut outcome, &mut tool_acc, &mut on_delta);
        }

        outcome.tool_calls = finalize_tool_calls(tool_acc);
        Ok(outcome)
    }
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn finalize_tool_calls(acc: BTreeMap<usize, PartialToolCall>) -> Vec<ToolCall> {
    acc.into_iter()
        .filter(|(_, p)| !p.name.is_empty())
        .map(|(idx, p)| ToolCall {
            id: if p.id.is_empty() {
                format!("call_{idx}")
            } else {
                p.id
            },
            call_type: "function".to_string(),
            function: ToolFunctionCall {
                name: p.name,
                arguments: if p.arguments.trim().is_empty() {
                    "{}".to_string()
                } else {
                    p.arguments
                },
            },
        })
        .collect()
}

/// Returns `Ok(true)` when `[DONE]` is encountered.
fn process_sse_line<F>(
    line: &str,
    outcome: &mut StreamOutcome,
    tool_acc: &mut BTreeMap<usize, PartialToolCall>,
    on_delta: &mut F,
) -> Result<bool>
where
    F: FnMut(StreamDelta),
{
    if line.is_empty() || line.starts_with(':') {
        return Ok(false);
    }
    let Some(payload) = line.strip_prefix("data:") else {
        return Ok(false);
    };
    let payload = payload.trim();
    if payload == "[DONE]" {
        return Ok(true);
    }

    let chunk: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(_) => return Ok(false),
    };

    if chunk.get("error").is_some() {
        let msg = extract_error_msg(&chunk).unwrap_or_else(|| "Unknown stream error".to_string());
        return Err(anyhow!(msg));
    }

    if let Some(t_val) = chunk.get("timings") {
        if let Ok(t) = serde_json::from_value::<Timings>(t_val.clone()) {
            outcome.timings = Some(t.clone());
            on_delta(StreamDelta::Timings(t));
        }
    }

    if let Some(u_val) = chunk.get("usage") {
        if let Ok(u) = serde_json::from_value::<Usage>(u_val.clone()) {
            outcome.usage = Some(u);
        }
    }

    let Some(choice) = chunk
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    else {
        return Ok(false);
    };

    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
        outcome.finish_reason = Some(fr.to_string());
    }

    let Some(delta) = choice.get("delta") else {
        return Ok(false);
    };

    if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
        if !r.is_empty() {
            outcome.reasoning.push_str(r);
            on_delta(StreamDelta::Reasoning(r.to_string()));
        }
    }

    if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
        if !c.is_empty() {
            outcome.content.push_str(c);
            on_delta(StreamDelta::Content(c.to_string()));
        }
    }

    if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tcs {
            let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let entry = tool_acc.entry(idx).or_default();
            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                if !id.is_empty() {
                    entry.id = id.to_string();
                }
            }
            if let Some(func) = tc.get("function") {
                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        entry.name.push_str(name);
                    }
                }
                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                    if !args.is_empty() {
                        entry.arguments.push_str(args);
                    }
                }
            }
        }
    }

    Ok(false)
}

async fn wait_cancelled(rx: &mut watch::Receiver<bool>) {
    if *rx.borrow() {
        return;
    }
    while rx.changed().await.is_ok() {
        if *rx.borrow() {
            return;
        }
    }
    futures_util::future::pending::<()>().await;
}

fn extract_error_msg(val: &Value) -> Option<String> {
    if let Some(msg) = val.pointer("/error/message").and_then(|v| v.as_str()) {
        return Some(msg.to_string());
    }
    if let Some(msg) = val.get("error").and_then(|v| v.as_str()) {
        return Some(msg.to_string());
    }
    if let Some(msg) = val.get("message").and_then(|v| v.as_str()) {
        return Some(msg.to_string());
    }
    None
}

pub fn is_context_overflow_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_ascii_lowercase();
    msg.contains("exceeds the available context size")
        || msg.contains("exceeds context")
        || msg.contains("context length exceeded")
        || msg.contains("prompt is too long")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_streamed_reasoning_content_and_tool_calls() {
        let lines = [
            r#"data: {"choices":[{"delta":{"reasoning_content":"Checking files."}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Let me inspect "}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Cargo.toml\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
        ];

        let mut outcome = StreamOutcome::default();
        let mut acc = BTreeMap::new();
        let mut deltas = Vec::new();
        let mut done = false;
        for line in lines {
            done = process_sse_line(line, &mut outcome, &mut acc, &mut |d| deltas.push(d)).unwrap();
        }
        assert!(done);
        outcome.tool_calls = finalize_tool_calls(acc);
        assert_eq!(outcome.reasoning, "Checking files.");
        assert_eq!(outcome.content, "Let me inspect ");
        assert_eq!(outcome.tool_calls.len(), 1);
        assert_eq!(outcome.tool_calls[0].id, "call_1");
        assert_eq!(outcome.tool_calls[0].function.name, "read_file");
        assert_eq!(
            outcome.tool_calls[0].function.arguments,
            r#"{"path":"Cargo.toml"}"#
        );
    }
}
