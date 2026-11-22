use anyhow::{anyhow, bail, Context, Result};
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    ReadOnly,
    Write,
    Shell,
}

#[derive(Debug, Clone)]
pub struct PreparedTool {
    pub name: String,
    pub kind: ToolKind,
    pub summary: String,
    pub preview_diff: Option<String>,
    pub args: Value,
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub ok: bool,
    pub output: String,
    pub diff: Option<String>,
}

pub fn tool_definitions(plan_mode: bool) -> Vec<Value> {
    let mut defs = vec![
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file with line numbers. Supports optional 1-based start_line and end_line.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to workspace root or absolute path." },
                        "start_line": { "type": "integer", "description": "1-based starting line number (default: 1)." },
                        "end_line": { "type": "integer", "description": "1-based inclusive ending line number (default: start_line + 299)." }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories inside the workspace (respects .gitignore).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Subdirectory path relative to workspace (default: '.')." },
                        "max_depth": { "type": "integer", "description": "Maximum recursion depth (default: 4)." }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "grep_search",
                "description": "Search for a regex or literal text pattern across workspace files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regex pattern to search for." },
                        "path": { "type": "string", "description": "Optional file or subdirectory to narrow search." },
                        "case_insensitive": { "type": "boolean", "description": "Ignore case when matching (default: false)." }
                    },
                    "required": ["pattern"]
                }
            }
        }),
    ];

    if !plan_mode {
        defs.push(json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace an exact substring inside an existing file. Fails if old_text is missing or matches multiple places unless replace_all is true.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to file to edit." },
                        "old_text": { "type": "string", "description": "Exact existing text to replace (including indentation/newlines)." },
                        "new_text": { "type": "string", "description": "Replacement text." },
                        "replace_all": { "type": "boolean", "description": "Replace every occurrence instead of requiring a unique match (default: false)." }
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        }));
        defs.push(json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a new file or completely overwrite an existing file with the provided content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to create or overwrite." },
                        "content": { "type": "string", "description": "Full file contents." }
                    },
                    "required": ["path", "content"]
                }
            }
        }));
        defs.push(json!({
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute a shell command inside the workspace and return exit code, stdout, and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Shell command to run via /bin/zsh -lc." },
                        "cwd": { "type": "string", "description": "Optional working directory relative to workspace." },
                        "timeout_secs": { "type": "integer", "description": "Timeout in seconds (default: 60, max: 300)." }
                    },
                    "required": ["command"]
                }
            }
        }));
    }

    defs
}

pub fn resolve_path(workspace: &Path, raw: &str) -> PathBuf {
    let p = PathBuf::from(raw);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

pub fn display_rel(workspace: &Path, path: &Path) -> String {
    path.strip_prefix(workspace)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

pub fn compute_unified_diff(old: &str, new: &str, label: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = format!("--- a/{label}\n+++ b/{label}\n");
    let mut has_changes = false;

    for group in diff.grouped_ops(3) {
        has_changes = true;
        let first = &group[0];
        let last = &group[group.len() - 1];
        let old_start = first.old_range().start + 1;
        let old_len = last.old_range().end - first.old_range().start;
        let new_start = first.new_range().start + 1;
        let new_len = last.new_range().end - first.new_range().start;
        out.push_str(&format!(
            "@@ -{old_start},{old_len} +{new_start},{new_len} @@\n"
        ));

        for op in group {
            for change in diff.iter_changes(&op) {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                out.push_str(sign);
                out.push_str(change.value());
                if !change.value().ends_with('\n') {
                    out.push('\n');
                }
            }
        }
    }

    if !has_changes {
        return format!("(no changes to {label})");
    }
    truncate_lines(&out, 140)
}

pub fn prepare_tool(workspace: &Path, name: &str, raw_args: &str) -> Result<PreparedTool> {
    let args: Value = serde_json::from_str(raw_args)
        .with_context(|| format!("Invalid JSON arguments for tool '{name}': {raw_args}"))?;

    match name {
        "read_file" => {
            let path = required_str(&args, "path")?;
            let start = args.get("start_line").and_then(|v| v.as_u64());
            let end = args.get("end_line").and_then(|v| v.as_u64());
            let range = match (start, end) {
                (Some(s), Some(e)) => format!(" (L{s}-L{e})"),
                (Some(s), None) => format!(" (from L{s})"),
                _ => String::new(),
            };
            Ok(PreparedTool {
                name: name.to_string(),
                kind: ToolKind::ReadOnly,
                summary: format!("{path}{range}"),
                preview_diff: None,
                args,
            })
        }
        "list_files" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            Ok(PreparedTool {
                name: name.to_string(),
                kind: ToolKind::ReadOnly,
                summary: path.to_string(),
                preview_diff: None,
                args,
            })
        }
        "grep_search" => {
            let pattern = required_str(&args, "pattern")?;
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            Ok(PreparedTool {
                name: name.to_string(),
                kind: ToolKind::ReadOnly,
                summary: format!("/{pattern}/ in {path}"),
                preview_diff: None,
                args,
            })
        }
        "edit_file" => {
            let path_str = required_str(&args, "path")?;
            let old_text = required_str(&args, "old_text")?;
            let new_text = required_str(&args, "new_text")?;
            let replace_all = args
                .get("replace_all")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let full_path = resolve_path(workspace, path_str);
            let label = display_rel(workspace, &full_path);

            let preview = match fs::read_to_string(&full_path) {
                Ok(existing) => {
                    let count = existing.matches(old_text).count();
                    if count == 0 {
                        Some(format!("(preview warning: old_text not found in {label})"))
                    } else if count > 1 && !replace_all {
                        Some(format!(
                            "(preview warning: old_text matched {count} times in {label}; set replace_all=true or add context)"
                        ))
                    } else {
                        let updated = if replace_all {
                            existing.replace(old_text, new_text)
                        } else {
                            existing.replacen(old_text, new_text, 1)
                        };
                        Some(compute_unified_diff(&existing, &updated, &label))
                    }
                }
                Err(e) => Some(format!("(preview error reading {label}: {e})")),
            };

            Ok(PreparedTool {
                name: name.to_string(),
                kind: ToolKind::Write,
                summary: label,
                preview_diff: preview,
                args,
            })
        }
        "write_file" => {
            let path_str = required_str(&args, "path")?;
            let content = required_str(&args, "content")?;
            let full_path = resolve_path(workspace, path_str);
            let label = display_rel(workspace, &full_path);
            let old = fs::read_to_string(&full_path).unwrap_or_default();
            let diff = compute_unified_diff(&old, content, &label);
            let summary = if full_path.exists() {
                format!("{label} (overwrite, {} bytes)", content.len())
            } else {
                format!("{label} (new file, {} bytes)", content.len())
            };
            Ok(PreparedTool {
                name: name.to_string(),
                kind: ToolKind::Write,
                summary,
                preview_diff: Some(diff),
                args,
            })
        }
        "run_command" => {
            let command = required_str(&args, "command")?;
            let first_line = command.lines().next().unwrap_or(command);
            let summary = if first_line.len() > 96 {
                format!("{}…", &first_line[..95])
            } else {
                first_line.to_string()
            };
            Ok(PreparedTool {
                name: name.to_string(),
                kind: ToolKind::Shell,
                summary,
                preview_diff: None,
                args,
            })
        }
        other => bail!("Unknown tool '{other}'"),
    }
}

pub async fn execute_tool(
    workspace: &Path,
    prepared: &PreparedTool,
    cancel_rx: watch::Receiver<bool>,
) -> ToolOutput {
    let res = match prepared.name.as_str() {
        "read_file" => exec_read_file(workspace, &prepared.args),
        "list_files" => exec_list_files(workspace, &prepared.args),
        "grep_search" => exec_grep_search(workspace, &prepared.args).await,
        "edit_file" => exec_edit_file(workspace, &prepared.args),
        "write_file" => exec_write_file(workspace, &prepared.args),
        "run_command" => exec_run_command(workspace, &prepared.args, cancel_rx).await,
        other => Err(anyhow!("Unknown tool '{other}'")),
    };

    match res {
        Ok(out) => out,
        Err(err) => ToolOutput {
            ok: false,
            output: format!("Error: {err}"),
            diff: None,
        },
    }
}

fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing required string argument '{key}'"))
}

fn exec_read_file(workspace: &Path, args: &Value) -> Result<ToolOutput> {
    let path_str = required_str(args, "path")?;
    let full = resolve_path(workspace, path_str);
    let content = fs::read_to_string(&full)
        .with_context(|| format!("Failed to read {}", full.display()))?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start_line = args
        .get("start_line")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let default_end = (start_line + 299).min(total.max(1));
    let end_line = args
        .get("end_line")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).min(start_line + 399))
        .unwrap_or(default_end)
        .min(total);

    if total == 0 {
        return Ok(ToolOutput {
            ok: true,
            output: format!("{} is empty (0 lines).", display_rel(workspace, &full)),
            diff: None,
        });
    }
    if start_line > total {
        return Ok(ToolOutput {
            ok: true,
            output: format!(
                "{} has only {total} lines (requested start_line={start_line}).",
                display_rel(workspace, &full)
            ),
            diff: None,
        });
    }

    let mut out = format!(
        "File: {} (showing lines {start_line}-{end_line} of {total})\n",
        display_rel(workspace, &full)
    );
    let mut byte_budget = 32_000usize;
    let mut last_shown = start_line - 1;
    for idx in (start_line - 1)..end_line {
        let formatted = format!("{:>4} | {}\n", idx + 1, lines[idx]);
        if formatted.len() > byte_budget {
            out.push_str(&format!(
                "... [truncated at byte limit; continue from start_line={}]\n",
                idx + 1
            ));
            break;
        }
        byte_budget -= formatted.len();
        out.push_str(&formatted);
        last_shown = idx + 1;
    }
    if last_shown < total && byte_budget > 0 {
        out.push_str(&format!(
            "... ({} more lines; read with start_line={})\n",
            total - last_shown,
            last_shown + 1
        ));
    }

    Ok(ToolOutput {
        ok: true,
        output: out,
        diff: None,
    })
}

fn exec_list_files(workspace: &Path, args: &Value) -> Result<ToolOutput> {
    let sub = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let max_depth = args
        .get("max_depth")
        .and_then(|v| v.as_u64())
        .unwrap_or(4)
        .clamp(1, 12) as usize;
    let root = resolve_path(workspace, sub);
    if !root.exists() {
        bail!("Path does not exist: {}", root.display());
    }

    let mut walker = WalkBuilder::new(&root);
    walker
        .max_depth(Some(max_depth))
        .hidden(false)
        .git_ignore(true)
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(name.as_ref(), ".git" | "node_modules" | "target" | ".inkling" | "__pycache__")
        });

    let mut items = Vec::new();
    let mut truncated = false;
    for entry in walker.build().flatten() {
        if entry.path() == root {
            continue;
        }
        let rel = display_rel(workspace, entry.path());
        let suffix = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            "/"
        } else {
            ""
        };
        items.push(format!("{rel}{suffix}"));
        if items.len() >= 250 {
            truncated = true;
            break;
        }
    }
    items.sort();

    let mut out = if items.is_empty() {
        "(no matching entries)".to_string()
    } else {
        items.join("\n")
    };
    if truncated {
        out.push_str("\n... [truncated at 250 entries; pass a narrower path or smaller max_depth]");
    }

    Ok(ToolOutput {
        ok: true,
        output: out,
        diff: None,
    })
}

async fn exec_grep_search(workspace: &Path, args: &Value) -> Result<ToolOutput> {
    let pattern = required_str(args, "pattern")?;
    let sub = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let case_insensitive = args
        .get("case_insensitive")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let target = resolve_path(workspace, sub);
    if !target.exists() {
        bail!("Search path does not exist: {}", target.display());
    }

    // Prefer ripgrep when available for maximum speed.
    let mut rg = Command::new("rg");
    rg.args([
        "--no-heading",
        "--line-number",
        "--color=never",
        "--max-count=20",
        "--glob=!.git/**",
        "--glob=!target/**",
        "--glob=!node_modules/**",
        "--glob=!.inkling/**",
    ]);
    if case_insensitive {
        rg.arg("-i");
    }
    rg.arg("--").arg(pattern).arg(&target);
    rg.current_dir(workspace);

    if let Ok(output) = rg.output().await {
        if output.status.success() || output.status.code() == Some(1) {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.lines().take(120).collect();
            let text = if lines.is_empty() {
                format!("No matches for /{pattern}/ in {sub}")
            } else {
                let mut s = lines.join("\n");
                if stdout.lines().count() > 120 {
                    s.push_str("\n... [truncated at 120 matches]");
                }
                s
            };
            return Ok(ToolOutput {
                ok: true,
                output: text,
                diff: None,
            });
        }
    }

    // Pure-Rust fallback using ignore + regex.
    let re = RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .with_context(|| format!("Invalid regex pattern: {pattern}"))?;

    let mut matches = Vec::new();
    let walker = WalkBuilder::new(&target)
        .git_ignore(true)
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(name.as_ref(), ".git" | "node_modules" | "target" | ".inkling")
        })
        .build();

    'outer: for entry in walker.flatten() {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let Ok(content) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let rel = display_rel(workspace, entry.path());
        for (idx, line) in content.lines().enumerate() {
            if re.is_match(line) {
                matches.push(format!("{rel}:{}: {}", idx + 1, line));
                if matches.len() >= 120 {
                    matches.push("... [truncated at 120 matches]".to_string());
                    break 'outer;
                }
            }
        }
    }

    let text = if matches.is_empty() {
        format!("No matches for /{pattern}/ in {sub}")
    } else {
        matches.join("\n")
    };
    Ok(ToolOutput {
        ok: true,
        output: text,
        diff: None,
    })
}

fn exec_edit_file(workspace: &Path, args: &Value) -> Result<ToolOutput> {
    let path_str = required_str(args, "path")?;
    let old_text = required_str(args, "old_text")?;
    let new_text = required_str(args, "new_text")?;
    let replace_all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if old_text.is_empty() {
        bail!("old_text must not be empty");
    }

    let full = resolve_path(workspace, path_str);
    let label = display_rel(workspace, &full);
    let existing =
        fs::read_to_string(&full).with_context(|| format!("Failed to read {label}"))?;

    let count = existing.matches(old_text).count();
    if count == 0 {
        bail!("old_text was not found in {label}. Read the file again to match exact text/whitespace.");
    }
    if count > 1 && !replace_all {
        bail!("old_text matched {count} locations in {label}. Provide more surrounding context or pass replace_all=true.");
    }

    let updated = if replace_all {
        existing.replace(old_text, new_text)
    } else {
        existing.replacen(old_text, new_text, 1)
    };

    fs::write(&full, &updated).with_context(|| format!("Failed to write {label}"))?;
    let diff = compute_unified_diff(&existing, &updated, &label);
    let replaced = if replace_all { count } else { 1 };
    Ok(ToolOutput {
        ok: true,
        output: format!("Edited {label} ({replaced} replacement{}).", if replaced == 1 { "" } else { "s" }),
        diff: Some(diff),
    })
}

fn exec_write_file(workspace: &Path, args: &Value) -> Result<ToolOutput> {
    let path_str = required_str(args, "path")?;
    let content = required_str(args, "content")?;
    let full = resolve_path(workspace, path_str);
    let label = display_rel(workspace, &full);

    let old = fs::read_to_string(&full).unwrap_or_default();
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create parent dirs for {label}"))?;
    }
    fs::write(&full, content).with_context(|| format!("Failed to write {label}"))?;
    let diff = compute_unified_diff(&old, content, &label);

    Ok(ToolOutput {
        ok: true,
        output: format!("Wrote {label} ({} bytes, {} lines).", content.len(), content.lines().count()),
        diff: Some(diff),
    })
}

async fn exec_run_command(
    workspace: &Path,
    args: &Value,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<ToolOutput> {
    let command = required_str(args, "command")?;
    let cwd = args
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(|s| resolve_path(workspace, s))
        .unwrap_or_else(|| workspace.to_path_buf());
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(60)
        .clamp(1, 300);

    let shell = if Path::new("/bin/zsh").exists() {
        "/bin/zsh"
    } else {
        "/bin/sh"
    };

    let mut cmd = Command::new(shell);
    cmd.args(["-lc", command])
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = cmd
        .spawn()
        .with_context(|| format!("Failed to spawn shell command: {command}"))?;

    let wait_fut = child.wait_with_output();
    let timeout = Duration::from_secs(timeout_secs);

    let result = tokio::select! {
        _ = wait_cancelled(&mut cancel_rx) => {
            return Ok(ToolOutput {
                ok: false,
                output: "Command cancelled by user.".to_string(),
                diff: None,
            });
        }
        res = tokio::time::timeout(timeout, wait_fut) => res,
    };

    let output = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => bail!("Command execution failed: {e}"),
        Err(_) => {
            return Ok(ToolOutput {
                ok: false,
                output: format!("Command timed out after {timeout_secs}s: {command}"),
                diff: None,
            });
        }
    };

    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut combined = format!("Exit code: {code}\n");
    if !stdout.trim().is_empty() {
        combined.push_str(&format!("--- stdout ---\n{}\n", truncate_middle(&stdout, 14_000)));
    }
    if !stderr.trim().is_empty() {
        combined.push_str(&format!("--- stderr ---\n{}\n", truncate_middle(&stderr, 8_000)));
    }
    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        combined.push_str("(no output)");
    }

    Ok(ToolOutput {
        ok: output.status.success(),
        output: combined.trim_end().to_string(),
        diff: None,
    })
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

fn truncate_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return text.to_string();
    }
    let head = &lines[..max_lines];
    format!(
        "{}\n... [{} more lines truncated]",
        head.join("\n"),
        lines.len() - max_lines
    )
}

fn truncate_middle(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.trim_end().to_string();
    }
    let half = max_bytes / 2;
    let start = floor_char_boundary(text, half);
    let end = ceil_char_boundary(text, text.len().saturating_sub(half));
    let omitted = end.saturating_sub(start);
    format!(
        "{}\n... [omitted ~{omitted} bytes] ...\n{}",
        &text[..start],
        &text[end..].trim_end()
    )
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn edit_write_read_and_diff_work() {
        let dir = std::env::temp_dir().join(format!("inkling-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let (_tx, rx) = watch::channel(false);

        let prep_w = prepare_tool(
            &dir,
            "write_file",
            r#"{"path":"hello.txt","content":"line 1\nline 2\nline 3\n"}"#,
        )
        .unwrap();
        assert_eq!(prep_w.kind, ToolKind::Write);
        assert!(prep_w.preview_diff.as_deref().unwrap().contains("+line 2"));
        let res_w = execute_tool(&dir, &prep_w, rx.clone()).await;
        assert!(res_w.ok);

        let prep_e = prepare_tool(
            &dir,
            "edit_file",
            r#"{"path":"hello.txt","old_text":"line 2","new_text":"line TWO"}"#,
        )
        .unwrap();
        assert!(prep_e.preview_diff.as_deref().unwrap().contains("-line 2"));
        assert!(prep_e.preview_diff.as_deref().unwrap().contains("+line TWO"));
        let res_e = execute_tool(&dir, &prep_e, rx.clone()).await;
        assert!(res_e.ok);

        let prep_r = prepare_tool(&dir, "read_file", r#"{"path":"hello.txt"}"#).unwrap();
        let res_r = execute_tool(&dir, &prep_r, rx).await;
        assert!(res_r.ok);
        assert!(res_r.output.contains("2 | line TWO"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn plan_mode_hides_mutating_tools() {
        let plan_tools = tool_definitions(true);
        let names: Vec<&str> = plan_tools
            .iter()
            .filter_map(|v| v.pointer("/function/name").and_then(|s| s.as_str()))
            .collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"grep_search"));
        assert!(!names.contains(&"write_file"));
        assert!(!names.contains(&"edit_file"));
        assert!(!names.contains(&"run_command"));
    }
}
