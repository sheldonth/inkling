use anyhow::Result;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::agent::{
    compact_history_with_summary, run_agent_turn, AgentEvent, ApprovalDecision, ApprovalRequest,
    SharedControl,
};
use crate::api::{ApiClient, Message};
use crate::config::{
    format_lora_status, parse_lora_scale, EffortLevel, HarnessConfig, PermissionMode,
};
use crate::images::{
    encode_image_data_url, extract_dropped_images, format_size, parse_image_path_token,
    StagedImage, IMAGE_EXTENSIONS,
};
use crate::prompt::{inspect_workspace, WorkspaceInfo};
use crate::session::{Session, ToolStatus, TranscriptEntry};

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const SLASH_HINTS: &[(&str, &str)] = &[
    ("/help", "show commands and keybinds"),
    ("/image", "attach image file (or drag & drop directly)"),
    ("/images", "list or clear staged images"),
    ("/mode", "set mode: ask | auto-edit | yolo | plan"),
    ("/lora", "set LoRA: on | off | 0.6 | inherit | global <val>"),
    ("/effort", "set reasoning effort: none..max"),
    ("/thinking", "toggle thinking blocks (Ctrl+T)"),
    ("/tools", "toggle expanded tool output (Ctrl+O)"),
    ("/diff", "show git diff in transcript"),
    ("/compact", "summarize & compact older conversation turns"),
    ("/clear", "reset conversation history"),
    ("/save", "export session to .md or .json"),
    ("/status", "inspect server, LoRA, vision, and context"),
    ("/exit", "quit inkling code"),
];

struct TerminalGuard {
    kitty_keyboard: bool,
}

impl TerminalGuard {
    fn enter() -> Result<(Self, Terminal<CrosstermBackend<Stdout>>)> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )?;

        // Enable Kitty keyboard disambiguation when supported so Shift+Enter / Cmd+Enter
        // are reported distinctly (skipped cleanly on older terminals/tmux without support).
        let kitty_keyboard = supports_keyboard_enhancement().unwrap_or(false);
        if kitty_keyboard {
            let _ = execute!(
                stdout,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            );
        }

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok((Self { kitty_keyboard }, terminal))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        if self.kitty_keyboard {
            let _ = execute!(stdout, PopKeyboardEnhancementFlags);
        }
        let _ = execute!(
            stdout,
            DisableBracketedPaste,
            DisableMouseCapture,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
    }
}

/// A single visually wrapped row inside the multiline composer.
#[derive(Debug, Clone)]
struct VisualRow {
    /// Inclusive starting byte offset in `Composer::text`.
    start: usize,
    /// Exclusive ending byte offset in `Composer::text` (excludes trailing `\n`).
    end: usize,
    /// Rendered slice text.
    text: String,
    /// Whether this visual row ended due to a hard `\n` (as opposed to soft wrap).
    hard_break: bool,
}

#[derive(Default)]
struct Composer {
    text: String,
    cursor: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    draft_backup: String,
    scroll: u16,
    last_inner_width: usize,
}

impl Composer {
    fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.history_idx = None;
    }

    fn insert_str(&mut self, s: &str) {
        let cleaned = s.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &cleaned);
        self.cursor += cleaned.len();
        self.history_idx = None;
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = prev_char_boundary(&self.text, self.cursor);
        self.text.drain(prev..self.cursor);
        self.cursor = prev;
        self.history_idx = None;
    }

    fn delete(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        let next = next_char_boundary(&self.text, self.cursor);
        self.text.drain(self.cursor..next);
        self.history_idx = None;
    }

    fn delete_prev_word(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.text[..self.cursor];
        let trimmed_len = before.trim_end_matches(|c: char| c.is_whitespace()).len();
        let mut start = trimmed_len;
        for (idx, ch) in before[..trimmed_len].char_indices().rev() {
            if ch.is_whitespace() {
                start = idx + ch.len_utf8();
                break;
            }
            start = idx;
        }
        self.text.drain(start..self.cursor);
        self.cursor = start;
        self.history_idx = None;
    }

    fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = prev_char_boundary(&self.text, self.cursor);
        }
    }

    fn move_right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor = next_char_boundary(&self.text, self.cursor);
        }
    }

    fn move_line_start(&mut self) {
        let before = &self.text[..self.cursor];
        self.cursor = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    }

    fn move_line_end(&mut self) {
        let after = &self.text[self.cursor..];
        if let Some(rel) = after.find('\n') {
            self.cursor += rel;
        } else {
            self.cursor = self.text.len();
        }
    }

    fn kill_to_end(&mut self) {
        let after = &self.text[self.cursor..];
        let end = after
            .find('\n')
            .map(|r| self.cursor + r)
            .unwrap_or(self.text.len());
        if end > self.cursor {
            self.text.drain(self.cursor..end);
        }
    }

    fn clear_line(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.scroll = 0;
        self.history_idx = None;
    }

    /// If the character immediately before the cursor is `\`, replace it with `\n`
    /// (universal backslash line-continuation fallback for tmux/terminals without Shift+Enter).
    fn consume_backslash_continuation(&mut self) -> bool {
        if self.cursor > 0 && self.text.as_bytes()[self.cursor - 1] == b'\\' {
            self.text.replace_range((self.cursor - 1)..self.cursor, "\n");
            self.history_idx = None;
            return true;
        }
        false
    }

    /// Compute exact soft-wrapped visual rows for `width` columns.
    fn visual_rows(&self, width: usize) -> Vec<VisualRow> {
        let max_w = width.max(1);
        let mut rows: Vec<VisualRow> = Vec::new();
        let mut row_start = 0usize;
        let mut row_w = 0usize;

        for (idx, ch) in self.text.char_indices() {
            if ch == '\n' {
                rows.push(VisualRow {
                    start: row_start,
                    end: idx,
                    text: self.text[row_start..idx].to_string(),
                    hard_break: true,
                });
                row_start = idx + 1;
                row_w = 0;
                continue;
            }
            let cw = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
            if row_w > 0 && row_w + cw > max_w {
                rows.push(VisualRow {
                    start: row_start,
                    end: idx,
                    text: self.text[row_start..idx].to_string(),
                    hard_break: false,
                });
                row_start = idx;
                row_w = cw;
            } else {
                row_w += cw;
            }
        }

        // Always push the final row (including empty trailing row after `\n` or empty text).
        rows.push(VisualRow {
            start: row_start,
            end: self.text.len(),
            text: self.text[row_start..].to_string(),
            hard_break: true,
        });

        rows
    }

    /// Return `(visual_row_index, visual_col)` for `self.cursor`.
    fn visual_cursor(&self, rows: &[VisualRow]) -> (usize, usize) {
        for (idx, row) in rows.iter().enumerate() {
            let is_last = idx + 1 == rows.len();
            let in_row = if row.hard_break || is_last {
                self.cursor >= row.start && self.cursor <= row.end
            } else {
                // On soft-wrapped rows, a cursor at `row.end` belongs to the start of the NEXT row.
                self.cursor >= row.start && self.cursor < row.end
            };
            if in_row {
                let col = UnicodeWidthStr::width(&self.text[row.start..self.cursor]);
                return (idx, col);
            }
        }
        let last = rows.len().saturating_sub(1);
        let col = rows
            .get(last)
            .map(|r| UnicodeWidthStr::width(r.text.as_str()))
            .unwrap_or(0);
        (last, col)
    }

    fn move_up_or_history(&mut self) {
        let width = self.last_inner_width.max(20);
        let rows = self.visual_rows(width);
        let (vrow, vcol) = self.visual_cursor(&rows);
        if vrow == 0 {
            if self.history.is_empty() {
                return;
            }
            let next_idx = match self.history_idx {
                None => {
                    self.draft_backup = self.text.clone();
                    self.history.len() - 1
                }
                Some(0) => 0,
                Some(i) => i - 1,
            };
            self.history_idx = Some(next_idx);
            self.text = self.history[next_idx].clone();
            self.cursor = self.text.len();
        } else {
            self.set_visual_pos(&rows, vrow - 1, vcol);
        }
    }

    fn move_down_or_history(&mut self) {
        let width = self.last_inner_width.max(20);
        let rows = self.visual_rows(width);
        let (vrow, vcol) = self.visual_cursor(&rows);
        if vrow + 1 >= rows.len() {
            let Some(idx) = self.history_idx else {
                return;
            };
            if idx + 1 >= self.history.len() {
                self.history_idx = None;
                self.text = std::mem::take(&mut self.draft_backup);
                self.cursor = self.text.len();
            } else {
                self.history_idx = Some(idx + 1);
                self.text = self.history[idx + 1].clone();
                self.cursor = self.text.len();
            }
        } else {
            self.set_visual_pos(&rows, vrow + 1, vcol);
        }
    }

    fn set_visual_pos(&mut self, rows: &[VisualRow], target_row: usize, target_col: usize) {
        let Some(row) = rows.get(target_row) else {
            return;
        };
        let slice = &self.text[row.start..row.end];
        let mut acc = 0usize;
        let mut offset = slice.len();
        for (idx, ch) in slice.char_indices() {
            let w = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
            if acc + w > target_col {
                offset = idx;
                break;
            }
            acc += w;
        }
        // On a soft-wrapped row, stay strictly before `row.end` so cursor doesn't snap to next row.
        if !row.hard_break && offset == slice.len() && !slice.is_empty() {
            offset = prev_char_boundary(slice, offset);
        }
        self.cursor = row.start + offset;
    }

    fn take_submitted(&mut self) -> String {
        let out = self.text.trim().to_string();
        if !out.is_empty() && self.history.last().map(String::as_str) != Some(out.as_str()) {
            self.history.push(out.clone());
        }
        self.text.clear();
        self.cursor = 0;
        self.scroll = 0;
        self.history_idx = None;
        self.draft_backup.clear();
        out
    }
}

/// Returns true only for slash commands like `/help` or `/lora off`, NOT filesystem paths like
/// `/Users/...` or `/var/folders/...` pasted or dragged into the prompt.
fn is_slash_command_input(raw: &str) -> bool {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with('/') || trimmed.contains('\n') {
        return false;
    }
    let head = trimmed.split_whitespace().next().unwrap_or("");
    // A second '/' inside the command token means it's a path (e.g. /var/folders/...).
    !head[1..].contains('/')
}

fn prev_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.saturating_sub(1);
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn next_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = (idx + 1).min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

struct App {
    api: ApiClient,
    cfg: HarnessConfig,
    workspace_info: WorkspaceInfo,
    control: Arc<Mutex<SharedControl>>,
    session: Session,
    composer: Composer,
    staged_images: Vec<StagedImage>,
    has_vision: bool,
    server_lora_scale: Option<f32>,
    busy: bool,
    cancel_tx: Option<watch::Sender<bool>>,
    pending_approval: Option<ApprovalRequest>,
    approval_scroll: u16,
    scroll_offset: u16,
    follow_bottom: bool,
    last_max_scroll: u16,
    spinner_tick: usize,
    should_quit: bool,
}

impl App {
    fn new(
        api: ApiClient,
        cfg: HarnessConfig,
        server_lora_scale: Option<f32>,
        has_vision: bool,
        initial_images: Vec<PathBuf>,
    ) -> Self {
        let workspace_info = inspect_workspace(&cfg.workspace);
        let control = Arc::new(Mutex::new(SharedControl::new(cfg.mode)));
        let mut session = Session::new();
        let vision_note = if has_vision {
            "Vision: on (drag & drop images)"
        } else {
            "Vision: off (add *mmproj*.gguf to enable images)"
        };
        session.push_note(
            format!(
                "Ready in {} • Mode: {} • {} • Shift+Tab cycles mode • /help for commands",
                cfg.workspace.display(),
                cfg.mode.label(),
                vision_note
            ),
            false,
        );
        let mut staged_images = Vec::new();
        for p in initial_images {
            match StagedImage::from_path(p) {
                Ok(img) => staged_images.push(img),
                Err(e) => session.push_note(e.to_string(), true),
            }
        }
        Self {
            api,
            cfg,
            workspace_info,
            control,
            session,
            composer: Composer::default(),
            staged_images,
            has_vision,
            server_lora_scale,
            busy: false,
            cancel_tx: None,
            pending_approval: None,
            approval_scroll: 0,
            scroll_offset: 0,
            follow_bottom: true,
            last_max_scroll: 0,
            spinner_tick: 0,
            should_quit: false,
        }
    }

    fn stage_image_paths(&mut self, paths: Vec<PathBuf>) -> usize {
        let mut added = 0usize;
        for path in paths {
            if self.staged_images.iter().any(|s| s.path == path) {
                continue;
            }
            match StagedImage::from_path(path) {
                Ok(img) => {
                    self.staged_images.push(img);
                    added += 1;
                }
                Err(e) => self.session.push_note(e.to_string(), true),
            }
        }
        added
    }

    async fn check_vision_ready(&mut self) -> bool {
        if self.has_vision {
            return true;
        }
        if let Ok(props) = self.api.props().await {
            self.has_vision = props.has_vision;
        }
        if !self.has_vision {
            self.session.push_note(
                "Vision is not enabled on llama-server (no `mmproj` projector loaded).\n\
                 Drop Inkling-2-Mini's `*mmproj*.gguf` into the repo root (or set `paths.mmproj` in inkling.toml) and run `inkling restart`.",
                true,
            );
            return false;
        }
        true
    }

    fn effective_lora_scale(&self) -> Option<f32> {
        self.cfg.lora_scale.or(self.server_lora_scale)
    }

    fn current_mode(&self) -> PermissionMode {
        self.control.lock().unwrap().mode
    }

    fn set_mode(&mut self, mode: PermissionMode) {
        self.control.lock().unwrap().mode = mode;
        self.cfg.mode = mode;
    }

    fn drop_empty_streaming_assistant(&mut self) {
        let should_pop = matches!(
            self.session.transcript.last(),
            Some(TranscriptEntry::Assistant {
                reasoning,
                content,
                streaming: true,
                ..
            }) if reasoning.is_empty() && content.is_empty()
        );
        if should_pop {
            self.session.transcript.pop();
        } else if let Some(TranscriptEntry::Assistant { streaming, .. }) =
            self.session.transcript.last_mut()
        {
            *streaming = false;
        }
    }

    async fn start_user_turn(
        &mut self,
        prompt: String,
        agent_tx: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let images_to_send = self.staged_images.clone();
        if !images_to_send.is_empty() && !self.check_vision_ready().await {
            return;
        }

        let mut encoded_urls = Vec::with_capacity(images_to_send.len());
        let mut image_names = Vec::with_capacity(images_to_send.len());
        for img in &images_to_send {
            match encode_image_data_url(&img.path) {
                Ok((url, _)) => {
                    encoded_urls.push(url);
                    image_names.push(img.name.clone());
                }
                Err(e) => {
                    self.session.push_note(format!("Failed to load image '{}': {e}", img.name), true);
                    return;
                }
            }
        }
        self.staged_images.clear();

        self.session.transcript.push(TranscriptEntry::User {
            text: prompt.clone(),
            images: image_names,
        });
        let user_msg = if encoded_urls.is_empty() {
            Message::user(prompt)
        } else {
            Message::user_multimodal(prompt, encoded_urls)
        };
        self.session.messages.push(user_msg);
        self.follow_bottom = true;
        self.busy = true;

        let (cancel_tx, cancel_rx) = watch::channel(false);
        self.cancel_tx = Some(cancel_tx);

        let api = self.api.clone();
        let cfg = self.cfg.clone();
        let control = self.control.clone();
        let history = self.session.messages.clone();
        let tx = agent_tx.clone();

        tokio::spawn(async move {
            run_agent_turn(api, cfg, control, history, cancel_rx, tx).await;
        });
    }

    fn cancel_turn(&mut self) {
        if let Some(tx) = &self.cancel_tx {
            let _ = tx.send(true);
        }
        if let Some(req) = self.pending_approval.take() {
            let _ = req
                .responder
                .send(ApprovalDecision::Deny(Some("Cancelled by user".to_string())));
        }
    }

    fn handle_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::AssistantStart => {
                self.session.transcript.push(TranscriptEntry::Assistant {
                    reasoning: String::new(),
                    content: String::new(),
                    timings: None,
                    streaming: true,
                });
            }
            AgentEvent::AssistantAbort => {
                self.drop_empty_streaming_assistant();
            }
            AgentEvent::HistoryCompacted { messages, note } => {
                self.session.messages = messages;
                self.session.last_prompt_tokens = 0;
                self.session.messages_at_last_usage = 0;
                self.session.push_note(note, false);
            }
            AgentEvent::ReasoningDelta(chunk) => {
                if let Some(TranscriptEntry::Assistant { reasoning, .. }) =
                    self.session.transcript.last_mut()
                {
                    reasoning.push_str(&chunk);
                }
            }
            AgentEvent::ContentDelta(chunk) => {
                if let Some(TranscriptEntry::Assistant { content, .. }) =
                    self.session.transcript.last_mut()
                {
                    content.push_str(&chunk);
                }
            }
            AgentEvent::LiveTimings(t) => {
                if t.predicted_per_second > 0.0 {
                    self.session.last_gen_tps = t.predicted_per_second;
                }
                if let Some(TranscriptEntry::Assistant { timings, .. }) =
                    self.session.transcript.last_mut()
                {
                    *timings = Some(t);
                }
            }
            AgentEvent::AssistantDone { timings, usage } => {
                if let Some(u) = usage {
                    if u.total_tokens > 0 {
                        let msg_len = self.session.messages.len();
                        self.session.record_usage(u.total_tokens, msg_len);
                    }
                }
                if let Some(t) = &timings {
                    if t.predicted_per_second > 0.0 {
                        self.session.last_gen_tps = t.predicted_per_second;
                    }
                }
                if let Some(TranscriptEntry::Assistant {
                    timings: entry_t,
                    streaming,
                    ..
                }) = self.session.transcript.last_mut()
                {
                    *streaming = false;
                    if timings.is_some() {
                        *entry_t = timings;
                    }
                }
            }
            AgentEvent::ToolPlanned {
                id,
                name,
                kind,
                summary,
                diff,
            } => {
                self.session.transcript.push(TranscriptEntry::Tool {
                    id,
                    name,
                    kind,
                    summary,
                    diff,
                    output: String::new(),
                    status: ToolStatus::PendingApproval,
                });
            }
            AgentEvent::ApprovalRequired(req) => {
                self.approval_scroll = 0;
                self.pending_approval = Some(req);
            }
            AgentEvent::ToolRunning { id } => {
                if let Some(TranscriptEntry::Tool { status, .. }) = self
                    .session
                    .transcript
                    .iter_mut()
                    .rev()
                    .find(|e| matches!(e, TranscriptEntry::Tool { id: tid, .. } if tid == &id))
                {
                    *status = ToolStatus::Running;
                }
            }
            AgentEvent::ToolFinished {
                id,
                ok,
                output,
                diff,
            } => {
                if let Some(TranscriptEntry::Tool {
                    status,
                    output: entry_out,
                    diff: entry_diff,
                    ..
                }) = self
                    .session
                    .transcript
                    .iter_mut()
                    .rev()
                    .find(|e| matches!(e, TranscriptEntry::Tool { id: tid, .. } if tid == &id))
                {
                    *status = if ok {
                        ToolStatus::Success
                    } else if output.starts_with("User denied") || output.starts_with("Denied:") {
                        ToolStatus::Denied
                    } else {
                        ToolStatus::Error
                    };
                    *entry_out = output;
                    if diff.is_some() {
                        *entry_diff = diff;
                    }
                }
            }
            AgentEvent::Note { text, is_error } => {
                self.session.push_note(text, is_error);
            }
            AgentEvent::TurnFinished {
                messages,
                cancelled,
            } => {
                self.drop_empty_streaming_assistant();
                self.session.messages = messages;
                self.busy = false;
                self.cancel_tx = None;
                self.pending_approval = None;
                if cancelled {
                    self.session.push_note("Turn cancelled.", false);
                }
                self.workspace_info = inspect_workspace(&self.cfg.workspace);
            }
        }
    }

    async fn handle_slash_command(&mut self, raw: &str) {
        let mut parts = raw.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
        let arg = parts.next().unwrap_or("").trim();

        match cmd.as_str() {
            "/exit" | "/quit" | "/q" => {
                self.should_quit = true;
            }
            "/help" => {
                self.session.push_note(
                    "Commands:\n\
                     • /image <path>                              Attach an image (or drag & drop directly)\n\
                     • /images [clear]                            List or clear staged images (or Ctrl+X)\n\
                     • /mode [ask|auto-edit|yolo|plan]            Switch tool permission mode (or Shift+Tab)\n\
                     • /lora [on|off|base|0.6|inherit|global <v>] Live abliteration LoRA scale\n\
                     • /effort [default|none|minimal|low|medium|high|xhigh|max]\n\
                     • /thinking                                  Toggle thinking display (Ctrl+T)\n\
                     • /tools                                     Expand/collapse tool output (Ctrl+O)\n\
                     • /diff                                      Show current workspace `git diff`\n\
                     • /compact [fast]                            Summarize & compact older turns (auto runs at ~78% ctx)\n\
                     • /clear                                     Clear conversation history\n\
                     • /save <path.md|path.json>                  Export transcript\n\
                     • /status                                    Inspect server, vision & session state\n\
                     • /exit                                      Quit\n\n\
                     Keybinds:\n\
                     • Enter: send  |  Shift+Enter / Alt+Enter / Ctrl+J / \\+Enter: newline\n\
                     • Shift+Tab: cycle permission mode  |  Ctrl+T: thinking  |  Ctrl+O: tool output  |  Ctrl+X: clear staged images\n\
                     • PgUp/PgDn / Mouse wheel / Ctrl+U/D: scroll  |  Ctrl+End: jump to bottom\n\
                     • Ctrl+C: cancel active turn (or clear input / exit when idle)",
                    false,
                );
            }
            "/image" => {
                if arg.is_empty() {
                    self.session.push_note("Usage: /image <path-to-image>", true);
                } else if let Some(path) = parse_image_path_token(arg, &self.cfg.workspace) {
                    let added = self.stage_image_paths(vec![path]);
                    if added > 0 {
                        if let Some(last) = self.staged_images.last() {
                            self.session.push_note(
                                format!(
                                    "Staged image: {} ({}) [{} staged]",
                                    last.name,
                                    format_size(last.size_bytes),
                                    self.staged_images.len()
                                ),
                                false,
                            );
                        }
                        if !self.has_vision {
                            let _ = self.check_vision_ready().await;
                        }
                    }
                } else {
                    self.session.push_note(
                        format!(
                            "Not a valid existing image file: {arg}\nSupported formats: {}",
                            IMAGE_EXTENSIONS.join(", ")
                        ),
                        true,
                    );
                }
            }
            "/images" => {
                if arg.eq_ignore_ascii_case("clear") {
                    self.staged_images.clear();
                    self.session.push_note("Cleared staged images.", false);
                } else if self.staged_images.is_empty() {
                    self.session.push_note("No staged images.", false);
                } else {
                    let list: Vec<String> = self
                        .staged_images
                        .iter()
                        .enumerate()
                        .map(|(i, img)| {
                            format!(
                                "{}. {} ({}) — {}",
                                i + 1,
                                img.name,
                                format_size(img.size_bytes),
                                img.path.display()
                            )
                        })
                        .collect();
                    self.session.push_note(list.join("\n"), false);
                }
            }
            "/mode" => {
                if arg.is_empty() {
                    let m = self.current_mode();
                    self.session.push_note(
                        format!("Current mode: {} ({})", m.label(), m.description()),
                        false,
                    );
                } else {
                    match arg.parse::<PermissionMode>() {
                        Ok(mode) => {
                            self.set_mode(mode);
                            self.session.push_note(
                                format!("Mode -> {} ({})", mode.label(), mode.description()),
                                false,
                            );
                        }
                        Err(e) => self.session.push_note(e.to_string(), true),
                    }
                }
            }
            "/lora" => {
                if arg.is_empty() {
                    let eff = self.effective_lora_scale();
                    let src = if self.cfg.lora_scale.is_some() {
                        "session override"
                    } else {
                        "server global"
                    };
                    self.session.push_note(
                        format!("LoRA: {} ({src})", format_lora_status(eff)),
                        false,
                    );
                } else if arg.eq_ignore_ascii_case("inherit") {
                    self.cfg.lora_scale = None;
                    if let Ok(adapters) = self.api.get_lora_adapters().await {
                        self.server_lora_scale = adapters.first().map(|a| a.scale);
                    }
                    self.session.push_note(
                        format!(
                            "LoRA set to inherit server global ({}).",
                            format_lora_status(self.server_lora_scale)
                        ),
                        false,
                    );
                } else if let Some(rest) = arg.strip_prefix("global ") {
                    match parse_lora_scale(rest) {
                        Ok(scale) => match self.api.set_lora_scale(scale, 0).await {
                            Ok(adapters) => {
                                let actual = adapters.first().map(|a| a.scale).unwrap_or(scale);
                                self.server_lora_scale = Some(actual);
                                self.cfg.lora_scale = None;
                                self.session.push_note(
                                    format!(
                                        "Server global LoRA scale -> {}",
                                        format_lora_status(Some(actual))
                                    ),
                                    false,
                                );
                            }
                            Err(e) => self.session.push_note(e.to_string(), true),
                        },
                        Err(e) => self.session.push_note(e.to_string(), true),
                    }
                } else {
                    match parse_lora_scale(arg) {
                        Ok(scale) => {
                            self.cfg.lora_scale = Some(scale);
                            self.session.push_note(
                                format!(
                                    "Session LoRA override -> {}",
                                    format_lora_status(Some(scale))
                                ),
                                false,
                            );
                        }
                        Err(e) => self.session.push_note(e.to_string(), true),
                    }
                }
            }
            "/effort" => {
                if arg.is_empty() {
                    self.session.push_note(
                        format!("Reasoning effort: {}", self.cfg.effort),
                        false,
                    );
                } else {
                    match arg.parse::<EffortLevel>() {
                        Ok(eff) => {
                            self.cfg.effort = eff;
                            self.session
                                .push_note(format!("Reasoning effort -> {eff}"), false);
                        }
                        Err(e) => self.session.push_note(e.to_string(), true),
                    }
                }
            }
            "/thinking" => {
                self.cfg.show_thinking = !self.cfg.show_thinking;
                self.session.push_note(
                    format!(
                        "Thinking display: {}",
                        if self.cfg.show_thinking {
                            "expanded"
                        } else {
                            "collapsed"
                        }
                    ),
                    false,
                );
            }
            "/tools" => {
                self.cfg.expand_tools = !self.cfg.expand_tools;
                self.session.push_note(
                    format!(
                        "Tool output display: {}",
                        if self.cfg.expand_tools {
                            "expanded"
                        } else {
                            "compact"
                        }
                    ),
                    false,
                );
            }
            "/diff" => {
                let output = Command::new("git")
                    .args(["diff", "--stat", "--patch"])
                    .current_dir(&self.cfg.workspace)
                    .output();
                match output {
                    Ok(o) if o.status.success() => {
                        let text = String::from_utf8_lossy(&o.stdout).trim().to_string();
                        if text.is_empty() {
                            self.session
                                .push_note("Working tree has no unstaged git diff.", false);
                        } else {
                            self.session.transcript.push(TranscriptEntry::Tool {
                                id: "git-diff".to_string(),
                                name: "git diff".to_string(),
                                kind: crate::tools::ToolKind::ReadOnly,
                                summary: "workspace changes".to_string(),
                                diff: Some(text),
                                output: String::new(),
                                status: ToolStatus::Success,
                            });
                        }
                    }
                    Ok(o) => self.session.push_note(
                        format!(
                            "git diff failed: {}",
                            String::from_utf8_lossy(&o.stderr).trim()
                        ),
                        true,
                    ),
                    Err(e) => self.session.push_note(format!("git diff error: {e}"), true),
                }
            }
            "/compact" => {
                if arg.eq_ignore_ascii_case("fast") {
                    let (trimmed, freed) = self.session.compact();
                    self.session.push_note(
                        format!("Fast-compacted {trimmed} older items (~{freed} chars freed)."),
                        false,
                    );
                } else {
                    let (_tx, rx) = watch::channel(false);
                    if let Some(note) = compact_history_with_summary(
                        &self.api,
                        &self.cfg,
                        &mut self.session.messages,
                        1200,
                        rx,
                        true,
                    )
                    .await
                    {
                        self.session.last_prompt_tokens = 0;
                        self.session.messages_at_last_usage = 0;
                        self.session.push_note(note, false);
                    } else {
                        self.session
                            .push_note("Conversation is already compact.", false);
                    }
                }
            }
            "/clear" => {
                self.session.clear();
                self.staged_images.clear();
                self.scroll_offset = 0;
                self.follow_bottom = true;
                self.session.push_note("Conversation cleared.", false);
            }
            "/save" => {
                if arg.is_empty() {
                    self.session
                        .push_note("Usage: /save <file.md|file.json>", true);
                } else {
                    let path = PathBuf::from(arg);
                    let res = if path.extension().and_then(|e| e.to_str()) == Some("json") {
                        self.session.export_json(&path)
                    } else {
                        self.session.export_markdown(&path)
                    };
                    match res {
                        Ok(()) => self
                            .session
                            .push_note(format!("Saved session to {}", path.display()), false),
                        Err(e) => self.session.push_note(e.to_string(), true),
                    }
                }
            }
            "/status" => {
                let health = self
                    .api
                    .health()
                    .await
                    .unwrap_or_else(|e| format!("unreachable ({e})"));
                if let Ok(adapters) = self.api.get_lora_adapters().await {
                    self.server_lora_scale = adapters.first().map(|a| a.scale);
                }
                if let Ok(props) = self.api.props().await {
                    self.has_vision = props.has_vision;
                }
                let m = self.current_mode();
                let allowed: Vec<String> = self
                    .control
                    .lock()
                    .unwrap()
                    .always_allowed_tools
                    .iter()
                    .cloned()
                    .collect();
                self.session.push_note(
                    format!(
                        "Server:    {} ({health})\n\
                         Model:     {}\n\
                         Vision:    {}\n\
                         Workspace: {}\n\
                         Mode:      {} ({})\n\
                         LoRA:      {}\n\
                         Effort:    {}\n\
                         Context:   ~{} / {} tokens (auto-compacts at ~78%)\n\
                         Images:    {} staged\n\
                         Allowed:   {}",
                        self.api.base_url(),
                        self.cfg.model,
                        if self.has_vision { "enabled" } else { "disabled (no mmproj)" },
                        self.cfg.workspace.display(),
                        m.label(),
                        m.description(),
                        format_lora_status(self.effective_lora_scale()),
                        self.cfg.effort,
                        self.session.estimated_tokens(),
                        self.cfg.ctx_size_hint,
                        self.staged_images.len(),
                        if allowed.is_empty() {
                            "(none)".to_string()
                        } else {
                            allowed.join(", ")
                        }
                    ),
                    false,
                );
            }
            other => {
                self.session.push_note(
                    format!("Unknown command '{other}'. Type /help for available commands."),
                    true,
                );
            }
        }
        self.follow_bottom = true;
    }

    async fn handle_key(
        &mut self,
        key: KeyEvent,
        agent_tx: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        if key.kind == KeyEventKind::Release {
            return;
        }

        // Approval modal takes keyboard focus while active.
        if self.pending_approval.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    if let Some(req) = self.pending_approval.take() {
                        let _ = req.responder.send(ApprovalDecision::AllowOnce);
                    }
                }
                KeyCode::Char('a') | KeyCode::Char('A') => {
                    if let Some(req) = self.pending_approval.take() {
                        let _ = req.responder.send(ApprovalDecision::AllowToolForSession);
                    }
                }
                KeyCode::Char('e') | KeyCode::Char('E') => {
                    if let Some(req) = self.pending_approval.take() {
                        self.set_mode(PermissionMode::AutoEdit);
                        let _ = req.responder.send(ApprovalDecision::SwitchAutoEdit);
                    }
                }
                KeyCode::Char('!') => {
                    if let Some(req) = self.pending_approval.take() {
                        self.set_mode(PermissionMode::Yolo);
                        let _ = req.responder.send(ApprovalDecision::SwitchYolo);
                    }
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    if let Some(req) = self.pending_approval.take() {
                        let _ = req.responder.send(ApprovalDecision::Deny(None));
                    }
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cancel_turn();
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.approval_scroll = self.approval_scroll.saturating_add(1);
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.approval_scroll = self.approval_scroll.saturating_sub(1);
                }
                KeyCode::PageDown => {
                    self.approval_scroll = self.approval_scroll.saturating_add(10);
                }
                KeyCode::PageUp => {
                    self.approval_scroll = self.approval_scroll.saturating_sub(10);
                }
                _ => {}
            }
            return;
        }

        // Global shortcuts
        match (key.code, key.modifiers) {
            (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => {
                if self.busy {
                    self.cancel_turn();
                } else if !self.composer.text.is_empty() {
                    self.composer.clear_line();
                } else {
                    self.should_quit = true;
                }
                return;
            }
            (KeyCode::Char('d'), m) if m.contains(KeyModifiers::CONTROL) => {
                if self.composer.text.is_empty() && !self.busy {
                    self.should_quit = true;
                } else {
                    self.scroll_down(12);
                }
                return;
            }
            (KeyCode::BackTab, _) => {
                let next = self.current_mode().cycle();
                self.set_mode(next);
                self.session.push_note(
                    format!("Mode -> {} ({})", next.label(), next.description()),
                    false,
                );
                return;
            }
            (KeyCode::Char('t'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.cfg.show_thinking = !self.cfg.show_thinking;
                return;
            }
            (KeyCode::Char('o'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.cfg.expand_tools = !self.cfg.expand_tools;
                return;
            }
            (KeyCode::Char('x'), m) if m.contains(KeyModifiers::CONTROL) => {
                if !self.staged_images.is_empty() {
                    self.staged_images.clear();
                    self.session.push_note("Cleared staged images.", false);
                }
                return;
            }
            (KeyCode::PageUp, _) => {
                self.scroll_up(14);
                return;
            }
            (KeyCode::PageDown, _) => {
                self.scroll_down(14);
                return;
            }
            (KeyCode::Char('u'), m) if m.contains(KeyModifiers::CONTROL) => {
                if self.composer.text.is_empty() {
                    self.scroll_up(12);
                } else {
                    self.composer.clear_line();
                }
                return;
            }
            (KeyCode::End, m) if m.contains(KeyModifiers::CONTROL) => {
                self.follow_bottom = true;
                self.scroll_offset = self.last_max_scroll;
                return;
            }
            (KeyCode::Esc, _) if self.busy => {
                self.cancel_turn();
                return;
            }
            _ => {}
        }

        // Composer input handling
        match (key.code, key.modifiers) {
            // Any modifier + Enter inserts a newline (Shift+Enter, Alt+Enter, Ctrl+Enter, Cmd+Enter, Cmd+Shift+Enter).
            (KeyCode::Enter, m) if !m.is_empty() => {
                self.composer.insert_char('\n');
            }
            (KeyCode::Char('j'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.composer.insert_char('\n');
            }
            (KeyCode::Enter, _) => {
                // Universal backslash continuation (`\` + Enter -> newline), great for tmux/SSH.
                if self.composer.consume_backslash_continuation() {
                    return;
                }
                let submitted = self.composer.take_submitted();
                if submitted.is_empty() {
                    if !self.staged_images.is_empty() && !self.busy {
                        self.start_user_turn(String::new(), agent_tx).await;
                    }
                    return;
                }
                if is_slash_command_input(&submitted) {
                    if self.busy
                        && !matches!(
                            submitted.as_str(),
                            "/exit" | "/quit" | "/thinking" | "/tools" | "/status"
                        )
                    {
                        self.session.push_note(
                            "Wait for the current turn to finish (or press Ctrl+C) before running that command.",
                            false,
                        );
                    } else {
                        self.handle_slash_command(&submitted).await;
                    }
                    return;
                }

                // Extract any drag-and-dropped image paths from the submitted text.
                let (cleaned_text, dropped_imgs) =
                    extract_dropped_images(&submitted, &self.cfg.workspace);
                if !dropped_imgs.is_empty() {
                    self.stage_image_paths(dropped_imgs);
                    if !self.has_vision {
                        let _ = self.check_vision_ready().await;
                    }
                    // If ONLY image path(s) were dropped, stage them cleanly so the user can type a question.
                    if cleaned_text.is_empty() {
                        return;
                    }
                }

                if self.busy {
                    self.session.push_note(
                        "Agent is busy — wait for completion or press Ctrl+C / Esc to cancel.",
                        false,
                    );
                } else {
                    self.start_user_turn(cleaned_text, agent_tx).await;
                }
            }
            (KeyCode::Backspace, m) if m.contains(KeyModifiers::ALT) => {
                self.composer.delete_prev_word();
            }
            (KeyCode::Char('w'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.composer.delete_prev_word();
            }
            (KeyCode::Backspace, _) => self.composer.backspace(),
            (KeyCode::Delete, _) => self.composer.delete(),
            (KeyCode::Left, _) => self.composer.move_left(),
            (KeyCode::Right, _) => self.composer.move_right(),
            (KeyCode::Up, _) => self.composer.move_up_or_history(),
            (KeyCode::Down, _) => self.composer.move_down_or_history(),
            (KeyCode::Home, _) => self.composer.move_line_start(),
            (KeyCode::End, _) => self.composer.move_line_end(),
            (KeyCode::Char('a'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.composer.move_line_start();
            }
            (KeyCode::Char('e'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.composer.move_line_end();
            }
            (KeyCode::Char('k'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.composer.kill_to_end();
            }
            (KeyCode::Char(c), m)
                if !m.contains(KeyModifiers::CONTROL) && !m.contains(KeyModifiers::ALT) =>
            {
                self.composer.insert_char(c);
            }
            _ => {}
        }
    }

    fn scroll_up(&mut self, lines: u16) {
        self.follow_bottom = false;
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    fn scroll_down(&mut self, lines: u16) {
        self.scroll_offset = self
            .scroll_offset
            .saturating_add(lines)
            .min(self.last_max_scroll);
        if self.scroll_offset >= self.last_max_scroll {
            self.follow_bottom = true;
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let composer_inner_w = area.width.saturating_sub(2).max(1) as usize;
        self.composer.last_inner_width = composer_inner_w;

        let visual_rows = self.composer.visual_rows(composer_inner_w);
        let max_composer_rows = (area.height / 3).clamp(4, 12);
        let composer_visible_rows = (visual_rows.len() as u16).clamp(1, max_composer_rows);

        let show_slash_hint = is_slash_command_input(&self.composer.text);
        let hint_height = if show_slash_hint { 1 } else { 0 };
        let image_bar_height = if self.staged_images.is_empty() { 0 } else { 1 };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),                         // header
                Constraint::Min(5),                            // transcript
                Constraint::Length(image_bar_height),          // staged images strip
                Constraint::Length(hint_height),               // slash autocomplete hint
                Constraint::Length(composer_visible_rows + 2), // auto-expanding composer
                Constraint::Length(1),                         // footer
            ])
            .split(area);

        self.render_header(frame, chunks[0]);
        self.render_transcript(frame, chunks[1]);
        if image_bar_height > 0 {
            self.render_staged_images_bar(frame, chunks[2]);
        }
        if show_slash_hint {
            self.render_slash_hint(frame, chunks[3]);
        }
        self.render_composer(frame, chunks[4], &visual_rows);
        self.render_footer(frame, chunks[5]);

        if self.pending_approval.is_some() {
            self.render_approval_modal(frame, area);
        }
    }

    fn render_staged_images_bar(&self, frame: &mut Frame, area: Rect) {
        let mut spans: Vec<Span<'static>> = vec![Span::styled(
            " 🖼 Images: ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        for (idx, img) in self.staged_images.iter().enumerate() {
            if idx > 0 {
                spans.push(Span::raw("  "));
            }
            spans.push(Span::styled(
                format!(" {} ({}) ", img.name, format_size(img.size_bytes)),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        spans.push(Span::styled(
            "   (Enter to send • Ctrl+X to clear)",
            Style::default().fg(Color::DarkGray),
        ));
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let branch = match &self.workspace_info.git_branch {
            Some(b) if self.workspace_info.git_dirty => format!(" ({b}*)"),
            Some(b) => format!(" ({b})"),
            None => String::new(),
        };
        let lora_str = format_lora_status(self.effective_lora_scale());
        let lora_color = match self.effective_lora_scale() {
            Some(v) if (v - 1.0).abs() < 1e-5 => Color::Magenta,
            Some(v) if v.abs() < 1e-5 => Color::Gray,
            _ => Color::Yellow,
        };

        let line = Line::from(vec![
            Span::styled(
                " ◆ INKLING CODE ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {}{}", self.workspace_info.display_name, branch),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
            Span::styled(&self.cfg.model, Style::default().fg(Color::Cyan)),
            Span::styled("  │  LoRA: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                lora_str,
                Style::default()
                    .fg(lora_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  │  effort: ", Style::default().fg(Color::DarkGray)),
            Span::styled(self.cfg.effort.as_str(), Style::default().fg(Color::Yellow)),
            Span::styled("  │  vision: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if self.has_vision { "on" } else { "off" },
                Style::default().fg(if self.has_vision { Color::Green } else { Color::DarkGray }),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(line).style(Style::default().bg(Color::Rgb(24, 28, 36))),
            area,
        );
    }

    fn render_transcript(&mut self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line<'static>> = Vec::new();
        let spinner = SPINNER_FRAMES[self.spinner_tick % SPINNER_FRAMES.len()];

        for entry in &self.session.transcript {
            match entry {
                TranscriptEntry::User { text, images } => {
                    lines.push(Line::from(""));
                    if !images.is_empty() {
                        let mut img_spans = vec![Span::styled(
                            "❯ ",
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        )];
                        for (idx, name) in images.iter().enumerate() {
                            if idx > 0 {
                                img_spans.push(Span::raw(" "));
                            }
                            img_spans.push(Span::styled(
                                format!(" 🖼 {name} "),
                                Style::default()
                                    .fg(Color::Black)
                                    .bg(Color::Cyan),
                            ));
                        }
                        lines.push(Line::from(img_spans));
                    }
                    for (idx, l) in text.lines().enumerate() {
                        let prefix = if idx == 0 && images.is_empty() { "❯ " } else { "  " };
                        lines.push(Line::from(vec![
                            Span::styled(
                                prefix,
                                Style::default()
                                    .fg(Color::Green)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                l.to_string(),
                                Style::default()
                                    .fg(Color::White)
                                    .add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
                TranscriptEntry::Assistant {
                    reasoning,
                    content,
                    timings,
                    streaming,
                } => {
                    if !reasoning.is_empty() {
                        if self.cfg.show_thinking {
                            let label = if *streaming && content.is_empty() {
                                format!("┃ {spinner} thinking")
                            } else {
                                "┃ thinking".to_string()
                            };
                            lines.push(Line::from(Span::styled(
                                label,
                                Style::default()
                                    .fg(Color::DarkGray)
                                    .add_modifier(Modifier::ITALIC),
                            )));
                            for rline in reasoning.lines() {
                                lines.push(Line::from(vec![
                                    Span::styled("┃ ", Style::default().fg(Color::DarkGray)),
                                    Span::styled(
                                        rline.to_string(),
                                        Style::default()
                                            .fg(Color::Gray)
                                            .add_modifier(Modifier::DIM | Modifier::ITALIC),
                                    ),
                                ]));
                            }
                            lines.push(Line::from(""));
                        } else {
                            let count = reasoning.lines().count().max(1);
                            let tag = if *streaming && content.is_empty() {
                                format!("▸ {spinner} Thinking ({count} lines) · Ctrl+T to expand")
                            } else {
                                format!("▸ Thinking ({count} lines) · Ctrl+T to expand")
                            };
                            lines.push(Line::from(Span::styled(
                                tag,
                                Style::default()
                                    .fg(Color::DarkGray)
                                    .add_modifier(Modifier::ITALIC),
                            )));
                        }
                    }

                    if !content.is_empty() {
                        render_markdown_lines(content, &mut lines);
                    } else if *streaming && reasoning.is_empty() {
                        lines.push(Line::from(Span::styled(
                            format!("{spinner} waiting for model..."),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }

                    if !*streaming {
                        if let Some(t) = timings {
                            if t.predicted_n > 0 {
                                lines.push(Line::from(Span::styled(
                                    format!(
                                        "  {} prompt tok @ {:.1} t/s · {} gen tok @ {:.1} t/s",
                                        t.prompt_n,
                                        t.prompt_per_second,
                                        t.predicted_n,
                                        t.predicted_per_second
                                    ),
                                    Style::default().fg(Color::DarkGray),
                                )));
                            }
                        }
                        lines.push(Line::from(""));
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
                    let (icon, icon_color) = match status {
                        ToolStatus::PendingApproval => ("?", Color::Yellow),
                        ToolStatus::Running => (spinner, Color::Cyan),
                        ToolStatus::Success => ("✓", Color::Green),
                        ToolStatus::Error => ("✗", Color::Red),
                        ToolStatus::Denied => ("⊘", Color::Yellow),
                    };

                    lines.push(Line::from(vec![
                        Span::styled("╭─ ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{icon} "),
                            Style::default()
                                .fg(icon_color)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            name.clone(),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("  {summary}"), Style::default().fg(Color::Gray)),
                    ]));

                    if let Some(d) = diff {
                        let max_diff = if self.cfg.expand_tools { 120 } else { 24 };
                        let diff_lines: Vec<&str> = d.lines().collect();
                        for dl in diff_lines.iter().take(max_diff) {
                            let style = if dl.starts_with('+') && !dl.starts_with("+++") {
                                Style::default().fg(Color::Green)
                            } else if dl.starts_with('-') && !dl.starts_with("---") {
                                Style::default().fg(Color::Red)
                            } else if dl.starts_with("@@") {
                                Style::default().fg(Color::Cyan)
                            } else {
                                Style::default().fg(Color::DarkGray)
                            };
                            lines.push(Line::from(vec![
                                Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                                Span::styled((*dl).to_string(), style),
                            ]));
                        }
                        if diff_lines.len() > max_diff {
                            lines.push(Line::from(Span::styled(
                                format!(
                                    "│ … {} more diff lines (Ctrl+O to expand)",
                                    diff_lines.len() - max_diff
                                ),
                                Style::default().fg(Color::DarkGray),
                            )));
                        }
                    }

                    if !output.is_empty() {
                        let max_out = if self.cfg.expand_tools { 80 } else { 8 };
                        let out_lines: Vec<&str> = output.lines().collect();
                        let out_color = if *status == ToolStatus::Error {
                            Color::LightRed
                        } else {
                            Color::Gray
                        };
                        for ol in out_lines.iter().take(max_out) {
                            lines.push(Line::from(vec![
                                Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                                Span::styled((*ol).to_string(), Style::default().fg(out_color)),
                            ]));
                        }
                        if out_lines.len() > max_out {
                            lines.push(Line::from(Span::styled(
                                format!(
                                    "│ … {} more output lines (Ctrl+O to expand)",
                                    out_lines.len() - max_out
                                ),
                                Style::default().fg(Color::DarkGray),
                            )));
                        }
                    }

                    lines.push(Line::from(Span::styled(
                        "╰─",
                        Style::default().fg(Color::DarkGray),
                    )));
                }
                TranscriptEntry::SystemNote { text, is_error } => {
                    let color = if *is_error {
                        Color::LightRed
                    } else {
                        Color::DarkGray
                    };
                    for l in text.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("  {l}"),
                            Style::default().fg(color),
                        )));
                    }
                    lines.push(Line::from(""));
                }
            }
        }

        let inner_width = area.width.saturating_sub(2).max(1);
        let total_wrapped = wrapped_line_count(&lines, inner_width as usize) as u16;
        let para = Paragraph::new(lines).wrap(Wrap { trim: false });
        let viewport_h = area.height.max(1);
        let max_scroll = total_wrapped.saturating_sub(viewport_h);
        self.last_max_scroll = max_scroll;
        if self.follow_bottom {
            self.scroll_offset = max_scroll;
        } else {
            self.scroll_offset = self.scroll_offset.min(max_scroll);
        }

        let scrolled = para.scroll((self.scroll_offset, 0));
        let padded = Rect {
            x: area.x + 1,
            y: area.y,
            width: inner_width,
            height: area.height,
        };
        frame.render_widget(scrolled, padded);

        if !self.follow_bottom && self.scroll_offset < max_scroll && area.width > 28 {
            let badge = " ↓ scrolled up (Ctrl+End) ";
            let bx = area.x + area.width.saturating_sub(badge.len() as u16 + 1);
            let by = area.y + area.height.saturating_sub(1);
            let badge_rect = Rect::new(bx, by, badge.len() as u16, 1);
            frame.render_widget(Clear, badge_rect);
            frame.render_widget(
                Paragraph::new(Span::styled(
                    badge,
                    Style::default().fg(Color::Black).bg(Color::Yellow),
                )),
                badge_rect,
            );
        }
    }

    fn render_slash_hint(&self, frame: &mut Frame, area: Rect) {
        let prefix = self.composer.text.trim();
        let matches: Vec<String> = SLASH_HINTS
            .iter()
            .filter(|(c, _)| c.starts_with(prefix))
            .take(4)
            .map(|(c, d)| format!("{c} ({d})"))
            .collect();
        let text = if matches.is_empty() {
            " No matching slash command — type /help".to_string()
        } else {
            format!(" {}", matches.join("  •  "))
        };
        frame.render_widget(
            Paragraph::new(Span::styled(text, Style::default().fg(Color::DarkGray))),
            area,
        );
    }

    fn render_composer(&mut self, frame: &mut Frame, area: Rect, visual_rows: &[VisualRow]) {
        let border_color = if self.busy { Color::Yellow } else { Color::Cyan };
        let visible_h = area.height.saturating_sub(2).max(1) as usize;
        let (cursor_row, cursor_col) = self.composer.visual_cursor(visual_rows);

        // Keep cursor row inside visible composer window when text exceeds max visible rows.
        let mut scroll = self.composer.scroll as usize;
        if cursor_row < scroll {
            scroll = cursor_row;
        } else if cursor_row >= scroll + visible_h {
            scroll = cursor_row + 1 - visible_h;
        }
        let max_scroll = visual_rows.len().saturating_sub(visible_h);
        scroll = scroll.min(max_scroll);
        self.composer.scroll = scroll as u16;

        let overflow_tag = if visual_rows.len() > visible_h {
            format!(" [{}/{}] ", cursor_row + 1, visual_rows.len())
        } else {
            String::new()
        };

        let title = if self.busy {
            let spinner = SPINNER_FRAMES[self.spinner_tick % SPINNER_FRAMES.len()];
            format!(" {spinner} Running — Esc/Ctrl+C to cancel {overflow_tag}")
        } else {
            format!(
                " Prompt (Enter send • Shift+Enter / \\+Enter / Ctrl+J newline • /help) {overflow_tag}"
            )
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(Span::styled(title, Style::default().fg(border_color)));

        let widget = if self.composer.text.is_empty() && !self.busy {
            Paragraph::new(Span::styled(
                "Ask Inkling to inspect, edit, or test code...",
                Style::default().fg(Color::DarkGray),
            ))
            .block(block)
        } else {
            let rendered_lines: Vec<Line<'_>> = visual_rows
                .iter()
                .map(|r| Line::from(r.text.as_str()))
                .collect();
            Paragraph::new(rendered_lines)
                .block(block)
                .scroll((self.composer.scroll, 0))
        };

        frame.render_widget(widget, area);

        if self.pending_approval.is_none() {
            let rel_row = cursor_row.saturating_sub(scroll);
            let cx = (area.x + 1 + cursor_col as u16).min(area.x + area.width.saturating_sub(2));
            let cy = (area.y + 1 + rel_row as u16).min(area.y + area.height.saturating_sub(2));
            frame.set_cursor_position(Position::new(cx, cy));
        }
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let mode = self.current_mode();
        let mode_bg = match mode {
            PermissionMode::Ask => Color::Blue,
            PermissionMode::AutoEdit => Color::Magenta,
            PermissionMode::Yolo => Color::Red,
            PermissionMode::Plan => Color::Green,
        };

        let est = self.session.estimated_tokens();
        let max_ctx = self.cfg.ctx_size_hint.max(1);
        let pct = (est * 100) / max_ctx;
        let ctx_color = if pct >= 85 {
            Color::Red
        } else if pct >= 65 {
            Color::Yellow
        } else {
            Color::Gray
        };

        let ctx_k = est as f64 / 1000.0;
        let max_k = max_ctx as f64 / 1000.0;
        let tps = if self.session.last_gen_tps > 0.0 {
            format!("  {:.1} t/s", self.session.last_gen_tps)
        } else {
            String::new()
        };

        let line = Line::from(vec![
            Span::styled(
                format!(" {} ", mode.label()),
                Style::default()
                    .fg(Color::Black)
                    .bg(mode_bg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ctx ~{ctx_k:.1}k/{max_k:.0}k ({pct}%){tps}"),
                Style::default().fg(ctx_color),
            ),
            Span::styled(
                "   │   Shift+Tab mode  Ctrl+T think  Ctrl+O tools  Ctrl+C cancel",
                Style::default().fg(Color::DarkGray),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(line).style(Style::default().bg(Color::Rgb(24, 28, 36))),
            area,
        );
    }

    fn render_approval_modal(&self, frame: &mut Frame, area: Rect) {
        let Some(req) = &self.pending_approval else {
            return;
        };

        let width = area.width.saturating_sub(8).clamp(48, 100);
        let height = area.height.saturating_sub(6).clamp(12, 28);
        let popup = Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        };

        frame.render_widget(Clear, popup);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(Span::styled(
                format!(" Approval Required: {} ", req.name),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);

        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(3),
                Constraint::Length(2),
            ])
            .split(inner);

        let header = Line::from(vec![
            Span::styled("Target: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                req.summary.clone(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);
        frame.render_widget(Paragraph::new(header), sections[0]);

        let mut detail_lines: Vec<Line<'static>> = Vec::new();
        if let Some(cmd) = &req.command_detail {
            detail_lines.push(Line::from(Span::styled(
                "Command:",
                Style::default().fg(Color::DarkGray),
            )));
            for l in cmd.lines() {
                detail_lines.push(Line::from(Span::styled(
                    format!("  $ {l}"),
                    Style::default().fg(Color::Cyan),
                )));
            }
        }
        if let Some(diff) = &req.diff {
            for dl in diff.lines() {
                let style = if dl.starts_with('+') && !dl.starts_with("+++") {
                    Style::default().fg(Color::Green)
                } else if dl.starts_with('-') && !dl.starts_with("---") {
                    Style::default().fg(Color::Red)
                } else if dl.starts_with("@@") {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::Gray)
                };
                detail_lines.push(Line::from(Span::styled(dl.to_string(), style)));
            }
        }
        if detail_lines.is_empty() {
            detail_lines.push(Line::from(Span::styled(
                "(no preview)",
                Style::default().fg(Color::DarkGray),
            )));
        }

        frame.render_widget(
            Paragraph::new(detail_lines)
                .wrap(Wrap { trim: false })
                .scroll((self.approval_scroll, 0)),
            sections[1],
        );

        let actions = Line::from(vec![
            Span::styled(
                " [y/Enter] Allow ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!(" [a] Always allow {} ", req.name),
                Style::default().fg(Color::Black).bg(Color::Cyan),
            ),
            Span::raw("  "),
            Span::styled(
                " [e] Auto-Edit ",
                Style::default().fg(Color::Black).bg(Color::Magenta),
            ),
            Span::raw("  "),
            Span::styled(
                " [n/Esc] Deny ",
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);
        frame.render_widget(Paragraph::new(actions), sections[2]);
    }
}

fn wrapped_line_count(lines: &[Line<'_>], max_width: usize) -> usize {
    let w = max_width.max(1);
    lines
        .iter()
        .map(|line| {
            let line_w: usize = line
                .spans
                .iter()
                .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                .sum();
            if line_w == 0 {
                1
            } else {
                line_w.div_ceil(w)
            }
        })
        .sum()
}

fn render_markdown_lines(content: &str, out: &mut Vec<Line<'static>>) {
    let mut in_code_block = false;
    for raw in content.lines() {
        let trimmed = raw.trim_start();
        if let Some(fence_rest) = trimmed.strip_prefix("```") {
            in_code_block = !in_code_block;
            if in_code_block {
                let lang = fence_rest.trim();
                let header = if lang.is_empty() {
                    "┌─ code".to_string()
                } else {
                    format!("┌─ {lang}")
                };
                out.push(Line::from(Span::styled(
                    header,
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                out.push(Line::from(Span::styled(
                    "└─",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            continue;
        }

        if in_code_block {
            out.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    raw.to_string(),
                    Style::default().fg(Color::Rgb(190, 230, 255)),
                ),
            ]));
            continue;
        }

        if trimmed.starts_with("### ") || trimmed.starts_with("## ") || trimmed.starts_with("# ") {
            let heading = trimmed.trim_start_matches('#').trim();
            out.push(Line::from(Span::styled(
                heading.to_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        out.push(Line::from(parse_inline_spans(raw)));
    }
}

fn parse_inline_spans(line: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0usize;
    let mut buf = String::new();

    while i < chars.len() {
        if chars[i] == '`' {
            if let Some(end) = chars[i + 1..].iter().position(|&c| c == '`') {
                if !buf.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut buf)));
                }
                let code: String = chars[i + 1..i + 1 + end].iter().collect();
                spans.push(Span::styled(
                    code,
                    Style::default()
                        .fg(Color::Yellow)
                        .bg(Color::Rgb(35, 38, 46)),
                ));
                i = i + 2 + end;
                continue;
            }
        }
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = (i + 2..chars.len().saturating_sub(1))
                .find(|&j| chars[j] == '*' && chars[j + 1] == '*')
            {
                if !buf.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut buf)));
                }
                let bold: String = chars[i + 2..end].iter().collect();
                spans.push(Span::styled(
                    bold,
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                i = end + 2;
                continue;
            }
        }
        buf.push(chars[i]);
        i += 1;
    }

    if !buf.is_empty() || spans.is_empty() {
        spans.push(Span::raw(buf));
    }
    spans
}

pub async fn run_tui(
    api: ApiClient,
    cfg: HarnessConfig,
    server_lora_scale: Option<f32>,
    has_vision: bool,
    initial_images: Vec<PathBuf>,
    initial_prompt: Option<String>,
) -> Result<()> {
    let (_guard, mut terminal) = TerminalGuard::enter()?;

    let mut app = App::new(api, cfg, server_lora_scale, has_vision, initial_images);
    let (agent_tx, mut agent_rx) = mpsc::unbounded_channel();

    if let Some(prompt) = initial_prompt {
        let (cleaned, dropped) = extract_dropped_images(&prompt, &app.cfg.workspace);
        if !dropped.is_empty() {
            app.stage_image_paths(dropped);
        }
        if !cleaned.trim().is_empty() || !app.staged_images.is_empty() {
            app.start_user_turn(cleaned, &agent_tx).await;
        }
    }

    let mut term_events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(90));

    loop {
        terminal.draw(|f| app.render(f))?;

        if app.should_quit {
            app.cancel_turn();
            break;
        }

        tokio::select! {
            Some(ev) = agent_rx.recv() => {
                app.handle_agent_event(ev);
                while let Ok(more) = agent_rx.try_recv() {
                    app.handle_agent_event(more);
                }
            }
            Some(Ok(term_ev)) = term_events.next() => {
                match term_ev {
                    Event::Key(key) => app.handle_key(key, &agent_tx).await,
                    Event::Paste(pasted) => {
                        if app.pending_approval.is_none() {
                            // Drag-and-drop in macOS terminals arrives via bracketed paste:
                            // immediately turn dropped image paths into clean staged pills!
                            let (remaining, dropped) =
                                extract_dropped_images(&pasted, &app.cfg.workspace);
                            if !dropped.is_empty() {
                                app.stage_image_paths(dropped);
                                if !app.has_vision {
                                    let _ = app.check_vision_ready().await;
                                }
                                if !remaining.is_empty() {
                                    app.composer.insert_str(&remaining);
                                }
                            } else {
                                app.composer.insert_str(&pasted);
                            }
                        }
                    }
                    Event::Mouse(mouse) => match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            if app.pending_approval.is_some() {
                                app.approval_scroll = app.approval_scroll.saturating_sub(3);
                            } else {
                                app.scroll_up(3);
                            }
                        }
                        MouseEventKind::ScrollDown => {
                            if app.pending_approval.is_some() {
                                app.approval_scroll = app.approval_scroll.saturating_add(3);
                            } else {
                                app.scroll_down(3);
                            }
                        }
                        _ => {}
                    },
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }
            _ = tick.tick() => {
                if app.busy {
                    app.spinner_tick = app.spinner_tick.wrapping_add(1);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_soft_wraps_and_tracks_cursor() {
        let mut c = Composer::default();
        c.insert_str("abcdefghij"); // 10 chars
        let rows = c.visual_rows(4);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].text, "abcd");
        assert_eq!(rows[1].text, "efgh");
        assert_eq!(rows[2].text, "ij");
        assert_eq!(c.visual_cursor(&rows), (2, 2));

        // Moving cursor to boundary byte 4 places it at start of row 1.
        c.cursor = 4;
        assert_eq!(c.visual_cursor(&rows), (1, 0));

        // Backslash continuation converts `\` + Enter into `\n`.
        c.cursor = c.text.len();
        c.insert_char('\\');
        assert!(c.consume_backslash_continuation());
        assert!(c.text.ends_with('\n'));
    }

    #[test]
    fn slash_command_ignores_filesystem_paths() {
        assert!(is_slash_command_input("/help"));
        assert!(is_slash_command_input("/lora off"));
        assert!(is_slash_command_input("/save /tmp/out.md"));
        assert!(!is_slash_command_input("/var/folders/th/Screenshot.png"));
        assert!(!is_slash_command_input("/Users/s/Development/local_llm"));
        assert!(!is_slash_command_input("hello /help"));
    }
}
