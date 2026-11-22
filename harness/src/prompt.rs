use crate::config::PermissionMode;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Default)]
pub struct WorkspaceInfo {
    pub display_name: String,
    pub git_branch: Option<String>,
    pub git_dirty: bool,
    pub top_entries: Vec<String>,
}

pub fn inspect_workspace(workspace: &Path) -> WorkspaceInfo {
    let display_name = workspace
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| workspace.display().to_string());

    let git_branch = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(workspace)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    let git_dirty = if git_branch.is_some() {
        Command::new("git")
            .args(["status", "--porcelain", "--untracked-files=no"])
            .current_dir(workspace)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(false)
    } else {
        false
    };

    let mut top_entries = Vec::new();
    if let Ok(entries) = std::fs::read_dir(workspace) {
        let mut names: Vec<String> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with('.') {
                    return None;
                }
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                Some(if is_dir { format!("{name}/") } else { name })
            })
            .collect();
        names.sort();
        names.truncate(30);
        top_entries = names;
    }

    WorkspaceInfo {
        display_name,
        git_branch,
        git_dirty,
        top_entries,
    }
}

pub fn build_system_prompt(
    workspace: &Path,
    info: &WorkspaceInfo,
    mode: PermissionMode,
    extra: Option<&str>,
) -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    let git_line = match &info.git_branch {
        Some(branch) => format!(
            "- Git branch: `{branch}` ({})\n",
            if info.git_dirty {
                "has modified tracked files"
            } else {
                "clean working tree"
            }
        ),
        None => "- Git: not a git repository\n".to_string(),
    };

    let entries_line = if info.top_entries.is_empty() {
        String::new()
    } else {
        format!("- Top-level entries: {}\n", info.top_entries.join(", "))
    };

    let mode_guidance = match mode {
        PermissionMode::Plan => {
            "Current mode: PLAN (read-only). Only inspect, analyze, and propose concrete implementation plans. Do not attempt to modify files or run mutating commands."
        }
        PermissionMode::Ask => {
            "Current mode: ASK. Read/search tools execute automatically; file modifications and shell commands require user confirmation."
        }
        PermissionMode::AutoEdit => {
            "Current mode: AUTO-EDIT. Read/search and file edit tools execute automatically; shell commands require user confirmation."
        }
        PermissionMode::Yolo => {
            "Current mode: YOLO. All tools execute automatically without prompting."
        }
    };

    let mut prompt = format!(
        "You are an expert software engineering agent operating inside a local command-line development harness.\n\n\
        Environment:\n\
        - Workspace: `{workspace}`\n\
        - Platform: {os} ({arch})\n\
        {git_line}\
        {entries_line}\
        - {mode_guidance}\n\n\
        Working guidelines:\n\
        1. Inspect the codebase before making changes: use `list_files`, `grep_search`, and `read_file` to understand structure and conventions.\n\
        2. Prefer targeted `edit_file` changes over rewriting entire files with `write_file`.\n\
        3. After making code changes, verify them by running relevant tests, linters, or builds via `run_command` (unless in PLAN mode).\n\
        4. Keep responses direct, concise, and technical. Reference exact file paths and symbols you read or changed.\n\
        5. Never fabricate file contents, command outputs, or test results — rely on actual tool observations.\n\
        6. When calling tools, pass valid JSON arguments matching each tool's schema.",
        workspace = workspace.display(),
    );

    if let Some(custom) = extra {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            prompt.push_str("\n\nAdditional user instructions:\n");
            prompt.push_str(trimmed);
        }
    }

    prompt
}
