use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::api::{Message, Timings};
use crate::tools::ToolKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolStatus {
    PendingApproval,
    Running,
    Success,
    Error,
    Denied,
}

#[derive(Debug, Clone)]
pub enum TranscriptEntry {
    User {
        text: String,
    },
    Assistant {
        reasoning: String,
        content: String,
        timings: Option<Timings>,
        streaming: bool,
    },
    Tool {
        id: String,
        name: String,
        #[allow(dead_code)]
        kind: ToolKind,
        summary: String,
        diff: Option<String>,
        output: String,
        status: ToolStatus,
    },
    SystemNote {
        text: String,
        is_error: bool,
    },
}

#[derive(Debug, Default)]
pub struct Session {
    pub transcript: Vec<TranscriptEntry>,
    pub messages: Vec<Message>,
    pub last_prompt_tokens: u64,
    pub messages_at_last_usage: usize,
    pub last_gen_tps: f64,
}

/// Conservative token estimate for a single message (chars / 3 + framing overhead).
pub fn estimate_message_tokens(msg: &Message) -> usize {
    let chars = msg.content.as_deref().map_or(0, str::len)
        + msg.reasoning_content.as_deref().map_or(0, str::len)
        + msg.tool_calls.as_ref().map_or(0, |tcs| {
            tcs.iter()
                .map(|t| t.function.name.len() + t.function.arguments.len() + 24)
                .sum()
        });
    (chars / 3) + 12
}

/// Conservative total token estimate including system prompt + tool definitions overhead.
pub fn estimate_request_tokens(messages: &[Message], overhead_tokens: usize) -> usize {
    let msg_tokens: usize = messages.iter().map(estimate_message_tokens).sum();
    overhead_tokens + msg_tokens
}

/// Find safe split indices where a new user turn starts (never splitting an assistant
/// `tool_calls` message away from its following `tool` result messages).
pub fn safe_split_index(messages: &[Message], keep_recent_min: usize) -> Option<usize> {
    if messages.len() <= keep_recent_min {
        return None;
    }
    let target = messages.len().saturating_sub(keep_recent_min);

    // Walk backward from target to find a `user` message boundary.
    for idx in (1..=target).rev() {
        if messages[idx].role == "user" {
            return Some(idx);
        }
    }
    // Otherwise walk forward to the first non-tool message after target.
    for idx in target..messages.len() {
        if messages[idx].role == "user" {
            return Some(idx);
        }
    }
    None
}

/// Stage 1 lightweight compaction: strip older preserved `reasoning_content` and trim bulky
/// older tool outputs while keeping recent messages verbatim.
pub fn compact_messages_light(messages: &mut [Message], keep_recent: usize) -> (usize, usize) {
    let mut trimmed = 0usize;
    let mut freed = 0usize;

    // Strip reasoning_content from all assistant messages except the most recent 2 messages.
    let reasoning_cutoff = messages.len().saturating_sub(2);
    for msg in &mut messages[..reasoning_cutoff] {
        if msg.role == "assistant" {
            if let Some(r) = msg.reasoning_content.take() {
                if !r.is_empty() {
                    freed += r.len();
                    trimmed += 1;
                }
            }
        }
    }

    // Trim bulky tool outputs older than `keep_recent`.
    let tool_cutoff = messages.len().saturating_sub(keep_recent);
    for msg in &mut messages[..tool_cutoff] {
        if msg.role == "tool" {
            if let Some(content) = &mut msg.content {
                if content.len() > 600 {
                    let before = content.len();
                    let head: String = content.chars().take(300).collect();
                    *content = format!("{head}\n... [compacted {before} chars]");
                    freed += before.saturating_sub(content.len());
                    trimmed += 1;
                }
            }
        }
    }

    (trimmed, freed)
}

/// Emergency deterministic compaction: strips ALL reasoning traces, trims ALL non-final tool
/// outputs aggressively, and drops oldest turn groups until under `target_tokens`.
pub fn compact_messages_emergency(
    messages: &mut Vec<Message>,
    target_tokens: usize,
    overhead_tokens: usize,
) -> (usize, usize) {
    let before_tokens = estimate_request_tokens(messages, overhead_tokens);
    let mut dropped_msgs = 0usize;

    // 1. Strip all reasoning traces and shrink older tool outputs.
    for msg in messages.iter_mut() {
        if msg.role == "assistant" {
            msg.reasoning_content = None;
        }
    }
    let last_idx = messages.len().saturating_sub(1);
    for (idx, msg) in messages.iter_mut().enumerate() {
        if idx < last_idx && msg.role == "tool" {
            if let Some(c) = &mut msg.content {
                if c.len() > 300 {
                    let len = c.len();
                    let preview: String = c.chars().take(180).collect();
                    *c = format!("{preview}\n... [compacted {len} chars]");
                }
            }
        }
    }

    // 2. Drop oldest complete turn chunks while still over target_tokens.
    while estimate_request_tokens(messages, overhead_tokens) > target_tokens && messages.len() > 2 {
        // Find the next user message after index 0 so we drop a full turn cleanly.
        let next_user = (1..messages.len()).find(|&i| messages[i].role == "user");
        match next_user {
            Some(cut) if cut < messages.len() => {
                messages.drain(0..cut);
                dropped_msgs += cut;
            }
            _ => {
                // Single massive turn: trim the largest non-final message content.
                if let Some((idx, _)) = messages
                    .iter()
                    .enumerate()
                    .take(messages.len().saturating_sub(1))
                    .max_by_key(|(_, m)| estimate_message_tokens(m))
                {
                    if let Some(c) = &mut messages[idx].content {
                        if c.len() > 400 {
                            let half = c.chars().count() / 2;
                            let kept: String = c.chars().take(half).collect();
                            *c = format!("{kept}\n... [truncated for context limit]");
                            continue;
                        }
                    }
                }
                break;
            }
        }
    }

    if dropped_msgs > 0 {
        let note = Message::user(format!(
            "[Context auto-compacted: dropped {dropped_msgs} earlier message(s) to fit context window.]"
        ));
        // Only prepend notice if the first remaining message is not already a compaction notice.
        let already_noted = messages
            .first()
            .and_then(|m| m.content.as_deref())
            .map(|c| c.starts_with("[Context auto-compacted") || c.starts_with("[Compacted conversation"))
            .unwrap_or(false);
        if !already_noted {
            messages.insert(0, note);
        }
    }

    let after_tokens = estimate_request_tokens(messages, overhead_tokens);
    (dropped_msgs, before_tokens.saturating_sub(after_tokens))
}

/// Format a slice of messages into compact readable text for LLM summarization.
pub fn format_messages_for_summary(messages: &[Message], max_chars: usize) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = msg.role.to_uppercase();
        let mut piece = String::new();
        if let Some(name) = &msg.name {
            piece.push_str(&format!("[{role}:{name}] "));
        } else {
            piece.push_str(&format!("[{role}] "));
        }
        if let Some(content) = &msg.content {
            let trimmed: String = content.chars().take(1200).collect();
            piece.push_str(&trimmed);
            if content.chars().count() > 1200 {
                piece.push_str(" ...[truncated]");
            }
        }
        if let Some(tcs) = &msg.tool_calls {
            for tc in tcs {
                let args_preview: String = tc.function.arguments.chars().take(240).collect();
                piece.push_str(&format!("\n  -> call {}({})", tc.function.name, args_preview));
            }
        }
        piece.push('\n');
        if out.len() + piece.len() > max_chars {
            out.push_str("... [earlier excerpt truncated]\n");
            break;
        }
        out.push_str(&piece);
    }
    out
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.transcript.clear();
        self.messages.clear();
        self.last_prompt_tokens = 0;
        self.messages_at_last_usage = 0;
    }

    pub fn push_note(&mut self, text: impl Into<String>, is_error: bool) {
        self.transcript.push(TranscriptEntry::SystemNote {
            text: text.into(),
            is_error,
        });
    }

    pub fn record_usage(&mut self, total_tokens: u64, msg_count: usize) {
        if total_tokens > 0 {
            self.last_prompt_tokens = total_tokens;
            self.messages_at_last_usage = msg_count;
        }
    }

    pub fn estimated_tokens(&self) -> usize {
        // ~1200 tokens covers the system prompt + 6 tool JSON schemas.
        let est_all = estimate_request_tokens(&self.messages, 1200);
        if self.last_prompt_tokens == 0 {
            return est_all;
        }
        let delta = if self.messages.len() > self.messages_at_last_usage {
            self.messages[self.messages_at_last_usage..]
                .iter()
                .map(estimate_message_tokens)
                .sum()
        } else {
            0
        };
        (self.last_prompt_tokens as usize + delta).max(est_all)
    }

    /// Trim bulky older tool outputs and reasoning traces to free up context window space.
    /// Returns `(trimmed_messages, freed_chars)`.
    pub fn compact(&mut self) -> (usize, usize) {
        let res = compact_messages_light(&mut self.messages, 6);
        if res.1 > 0 {
            self.last_prompt_tokens = 0;
            self.messages_at_last_usage = 0;
        }
        res
    }

    pub fn export_markdown(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut out = String::from("# Inkling Code Session\n\n");
        for entry in &self.transcript {
            match entry {
                TranscriptEntry::User { text } => {
                    out.push_str("## User\n\n");
                    out.push_str(text);
                    out.push_str("\n\n");
                }
                TranscriptEntry::Assistant {
                    reasoning,
                    content,
                    ..
                } => {
                    out.push_str("## Assistant\n\n");
                    if !reasoning.trim().is_empty() {
                        out.push_str("<details><summary>Thinking</summary>\n\n");
                        out.push_str(reasoning);
                        out.push_str("\n\n</details>\n\n");
                    }
                    if !content.trim().is_empty() {
                        out.push_str(content);
                        out.push_str("\n\n");
                    }
                }
                TranscriptEntry::Tool {
                    name,
                    summary,
                    diff,
                    output,
                    status,
                    ..
                } => {
                    out.push_str(&format!("### Tool: `{name}` — {summary} ({status:?})\n\n"));
                    if let Some(d) = diff {
                        out.push_str("```diff\n");
                        out.push_str(d);
                        out.push_str("\n```\n\n");
                    }
                    if !output.trim().is_empty() {
                        out.push_str("```text\n");
                        out.push_str(output);
                        out.push_str("\n```\n\n");
                    }
                }
                TranscriptEntry::SystemNote { text, .. } => {
                    out.push_str(&format!("> {text}\n\n"));
                }
            }
        }
        fs::write(path, out).with_context(|| format!("Failed to save {}", path.display()))
    }

    pub fn export_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(&self.messages)?;
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("Failed to save {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ToolCall, ToolFunctionCall};

    #[test]
    fn compact_trims_old_tool_outputs_and_reasoning() {
        let mut sess = Session::new();
        sess.messages.push(Message::user("first"));
        sess.messages.push(Message::assistant(
            "checking".to_string(),
            "long thinking ".repeat(100),
            vec![],
        ));
        sess.messages
            .push(Message::tool_result("c1", "read_file", "x".repeat(2000)));
        for i in 0..6 {
            sess.messages.push(Message::user(format!("msg {i}")));
        }
        let (trimmed, freed) = sess.compact();
        assert_eq!(trimmed, 2);
        assert!(freed > 2500);
        assert!(sess.messages[1].reasoning_content.is_none());
        assert!(sess.messages[2]
            .content
            .as_deref()
            .unwrap()
            .contains("[compacted"));
    }

    #[test]
    fn safe_split_never_breaks_tool_call_pairs() {
        let tc = ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: ToolFunctionCall {
                name: "read_file".into(),
                arguments: "{}".into(),
            },
        };
        let msgs = vec![
            Message::user("turn 1"),
            Message::assistant(String::new(), String::new(), vec![tc.clone()]),
            Message::tool_result("c1", "read_file", "data"),
            Message::assistant("done 1".into(), String::new(), vec![]),
            Message::user("turn 2"),
            Message::assistant(String::new(), String::new(), vec![tc]),
            Message::tool_result("c1", "read_file", "data 2"),
        ];
        let split = safe_split_index(&msgs, 3).unwrap();
        assert_eq!(split, 4);
        assert_eq!(msgs[split].role, "user");
    }

    #[test]
    fn emergency_compaction_fits_budget() {
        let mut msgs = Vec::new();
        for i in 0..10 {
            msgs.push(Message::user(format!("turn {i}: {}", "a".repeat(3000))));
            msgs.push(Message::assistant(
                format!("reply {i}: {}", "b".repeat(3000)),
                "thought ".repeat(500),
                vec![],
            ));
        }
        let (_dropped, freed) = compact_messages_emergency(&mut msgs, 4000, 1000);
        assert!(freed > 0);
        assert!(estimate_request_tokens(&msgs, 1000) <= 4500);
    }
}
