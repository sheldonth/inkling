use anyhow::Result;
use std::collections::HashSet;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot, watch};

use crate::api::{
    is_context_overflow_error, ApiClient, ChatRequestOptions, Message, StreamDelta, Timings, Usage,
};
use crate::config::{EffortLevel, HarnessConfig, PermissionMode};
use crate::prompt::{build_system_prompt, inspect_workspace};
use crate::session::{
    compact_messages_emergency, compact_messages_light, estimate_request_tokens,
    format_messages_for_summary, safe_split_index,
};
use crate::tools::{execute_tool, prepare_tool, tool_definitions, ToolKind};

#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    AllowOnce,
    AllowToolForSession,
    SwitchAutoEdit,
    SwitchYolo,
    Deny(Option<String>),
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct ApprovalRequest {
    pub id: String,
    pub name: String,
    pub kind: ToolKind,
    pub summary: String,
    pub diff: Option<String>,
    pub command_detail: Option<String>,
    pub responder: oneshot::Sender<ApprovalDecision>,
}

#[derive(Debug)]
pub enum AgentEvent {
    AssistantStart,
    ReasoningDelta(String),
    ContentDelta(String),
    LiveTimings(Timings),
    AssistantDone {
        timings: Option<Timings>,
        usage: Option<Usage>,
    },
    AssistantAbort,
    HistoryCompacted {
        messages: Vec<Message>,
        note: String,
    },
    ToolPlanned {
        id: String,
        name: String,
        kind: ToolKind,
        summary: String,
        diff: Option<String>,
    },
    ApprovalRequired(ApprovalRequest),
    ToolRunning {
        id: String,
    },
    ToolFinished {
        id: String,
        ok: bool,
        output: String,
        diff: Option<String>,
    },
    Note {
        text: String,
        is_error: bool,
    },
    TurnFinished {
        messages: Vec<Message>,
        cancelled: bool,
    },
}

#[derive(Debug, Clone)]
pub struct SharedControl {
    pub mode: PermissionMode,
    pub always_allowed_tools: HashSet<String>,
}

impl SharedControl {
    pub fn new(mode: PermissionMode) -> Self {
        Self {
            mode,
            always_allowed_tools: HashSet::new(),
        }
    }
}

/// Ask the model to summarize the older half of `history` into a concise handoff message.
/// Falls back to deterministic emergency compaction if summarization cannot run or fails.
pub async fn compact_history_with_summary(
    api: &ApiClient,
    cfg: &HarnessConfig,
    history: &mut Vec<Message>,
    overhead_tokens: usize,
    cancel_rx: watch::Receiver<bool>,
    force_summary: bool,
) -> Option<String> {
    let ctx_limit = cfg.ctx_size_hint.max(4096);
    let before = estimate_request_tokens(history, overhead_tokens);

    // Stage 1: lossless/lightweight trim (strip old thinking traces + trim bulky old tool outputs).
    let (trimmed, freed_chars) = compact_messages_light(history, 6);
    let after_light = estimate_request_tokens(history, overhead_tokens);

    let trigger_summary = force_summary || after_light >= (ctx_limit * 72 / 100);
    if !trigger_summary {
        if freed_chars > 0 {
            return Some(format!(
                "Auto-compacted context ({trimmed} old thinking/tool blocks trimmed): ~{:.1}k → ~{:.1}k / {:.0}k tok",
                before as f64 / 1000.0,
                after_light as f64 / 1000.0,
                ctx_limit as f64 / 1000.0
            ));
        }
        return None;
    }

    // Stage 2: summarize older complete turn prefix with the local model.
    if let Some(split) = safe_split_index(history, 4) {
        let older = &history[..split];
        let summary_budget_chars = ((ctx_limit / 2) * 3).clamp(8_000, 48_000);
        let excerpt = format_messages_for_summary(older, summary_budget_chars);
        let summary_msgs = vec![
            Message::system(
                "You compress an ongoing software-engineering conversation so work can continue seamlessly. \
                 Produce a concise, structured technical summary covering: (1) user's current goal & constraints, \
                 (2) files inspected or edited (with exact paths), (3) key findings/errors/decisions, and \
                 (4) pending next steps. Be terse and factual; omit fluff.",
            ),
            Message::user(format!("Summarize this earlier conversation:\n\n{excerpt}")),
        ];

        let opts = ChatRequestOptions {
            model: cfg.model.clone(),
            effort: EffortLevel::Low,
            lora_scale: cfg.lora_scale,
            temperature: 0.3,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            max_tokens: 1024,
        };

        if let Ok(outcome) = api
            .stream_chat_completion(&summary_msgs, &[], &opts, cancel_rx, |_| {})
            .await
        {
            let summary_text = outcome.content.trim().to_string();
            if !outcome.cancelled && !summary_text.is_empty() {
                let kept: Vec<Message> = history[split..].to_vec();
                history.clear();
                history.push(Message::user(format!(
                    "[Compacted conversation summary covering {split} earlier messages]\n\n{summary_text}"
                )));
                history.extend(kept);

                let after_sum = estimate_request_tokens(history, overhead_tokens);
                if after_sum < (ctx_limit * 78 / 100) {
                    return Some(format!(
                        "Auto-compacted {split} earlier messages into summary: ~{:.1}k → ~{:.1}k / {:.0}k tok",
                        before as f64 / 1000.0,
                        after_sum as f64 / 1000.0,
                        ctx_limit as f64 / 1000.0
                    ));
                }
            }
        }
    }

    // Stage 3: deterministic emergency trim so we ALWAYS fit safely under budget.
    let target = (ctx_limit * 55 / 100).max(2048);
    let (dropped, _) = compact_messages_emergency(history, target, overhead_tokens);
    let after_emerg = estimate_request_tokens(history, overhead_tokens);
    if after_emerg < before || dropped > 0 || freed_chars > 0 {
        Some(format!(
            "Auto-compacted context (dropped {dropped} old msgs): ~{:.1}k → ~{:.1}k / {:.0}k tok",
            before as f64 / 1000.0,
            after_emerg as f64 / 1000.0,
            ctx_limit as f64 / 1000.0
        ))
    } else {
        None
    }
}

pub async fn run_agent_turn(
    api: ApiClient,
    cfg: HarnessConfig,
    control: Arc<Mutex<SharedControl>>,
    mut history: Vec<Message>,
    cancel_rx: watch::Receiver<bool>,
    tx: mpsc::UnboundedSender<AgentEvent>,
) {
    let workspace_info = inspect_workspace(&cfg.workspace);
    let mut cancelled = false;

    for step in 0..cfg.max_steps {
        if *cancel_rx.borrow() {
            cancelled = true;
            break;
        }

        let current_mode = control.lock().unwrap().mode;
        let system_prompt = build_system_prompt(
            &cfg.workspace,
            &workspace_info,
            current_mode,
            cfg.system_prompt_extra.as_deref(),
        );
        let tools = tool_definitions(current_mode == PermissionMode::Plan);
        let tools_chars: usize = tools.iter().map(|t| t.to_string().len()).sum();
        let overhead_tokens = (system_prompt.len() + tools_chars) / 3 + 256;

        // Proactive auto-compaction when approaching context limit (78% of ctx_size).
        let ctx_limit = cfg.ctx_size_hint.max(4096);
        let est = estimate_request_tokens(&history, overhead_tokens);
        if est >= (ctx_limit * 78 / 100) {
            if let Some(note) = compact_history_with_summary(
                &api,
                &cfg,
                &mut history,
                overhead_tokens,
                cancel_rx.clone(),
                false,
            )
            .await
            {
                let _ = tx.send(AgentEvent::HistoryCompacted {
                    messages: history.clone(),
                    note,
                });
            }
        }

        let opts = ChatRequestOptions {
            model: cfg.model.clone(),
            effort: cfg.effort,
            lora_scale: cfg.lora_scale,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            top_k: cfg.top_k,
            min_p: cfg.min_p,
            max_tokens: cfg.max_tokens,
        };

        // Execute request with up to 2 automatic overflow-compaction retries if llama-server
        // reports HTTP 400 context size exceeded.
        let mut outcome_opt = None;
        for attempt in 0..3 {
            let mut req_messages = Vec::with_capacity(history.len() + 1);
            req_messages.push(Message::system(system_prompt.clone()));
            req_messages.extend(history.iter().cloned());

            let _ = tx.send(AgentEvent::AssistantStart);

            let tx_delta = tx.clone();
            match api
                .stream_chat_completion(
                    &req_messages,
                    &tools,
                    &opts,
                    cancel_rx.clone(),
                    move |delta| match delta {
                        StreamDelta::Reasoning(s) => {
                            let _ = tx_delta.send(AgentEvent::ReasoningDelta(s));
                        }
                        StreamDelta::Content(s) => {
                            let _ = tx_delta.send(AgentEvent::ContentDelta(s));
                        }
                        StreamDelta::Timings(t) => {
                            let _ = tx_delta.send(AgentEvent::LiveTimings(t));
                        }
                    },
                )
                .await
            {
                Ok(o) => {
                    outcome_opt = Some(o);
                    break;
                }
                Err(err) => {
                    let _ = tx.send(AgentEvent::AssistantAbort);
                    if is_context_overflow_error(&err) && attempt < 2 {
                        let target_pct = if attempt == 0 { 55 } else { 35 };
                        let target = (ctx_limit * target_pct / 100).max(1500);
                        let (dropped, _) =
                            compact_messages_emergency(&mut history, target, overhead_tokens);
                        let after = estimate_request_tokens(&history, overhead_tokens);
                        let _ = tx.send(AgentEvent::HistoryCompacted {
                            messages: history.clone(),
                            note: format!(
                                "Hit server context limit; auto-compacted (dropped {dropped} msgs → ~{:.1}k tok) and retrying…",
                                after as f64 / 1000.0
                            ),
                        });
                        continue;
                    }
                    let _ = tx.send(AgentEvent::Note {
                        text: format!("Model request failed: {err}"),
                        is_error: true,
                    });
                    break;
                }
            }
        }

        let Some(outcome) = outcome_opt else {
            break;
        };

        let _ = tx.send(AgentEvent::AssistantDone {
            timings: outcome.timings.clone(),
            usage: outcome.usage.clone(),
        });

        if outcome.cancelled {
            if !outcome.content.is_empty() || !outcome.reasoning.is_empty() {
                history.push(Message::assistant(
                    outcome.content,
                    outcome.reasoning,
                    Vec::new(),
                ));
            }
            cancelled = true;
            break;
        }

        let tool_calls = outcome.tool_calls.clone();
        history.push(Message::assistant(
            outcome.content,
            outcome.reasoning,
            tool_calls.clone(),
        ));

        if tool_calls.is_empty() {
            break;
        }

        for tc in tool_calls {
            if *cancel_rx.borrow() {
                cancelled = true;
                break;
            }

            let prepared =
                match prepare_tool(&cfg.workspace, &tc.function.name, &tc.function.arguments) {
                    Ok(p) => p,
                    Err(err) => {
                        let msg = format!("Tool argument error: {err}");
                        let _ = tx.send(AgentEvent::ToolPlanned {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            kind: ToolKind::ReadOnly,
                            summary: "(invalid arguments)".to_string(),
                            diff: None,
                        });
                        let _ = tx.send(AgentEvent::ToolFinished {
                            id: tc.id.clone(),
                            ok: false,
                            output: msg.clone(),
                            diff: None,
                        });
                        history.push(Message::tool_result(&tc.id, &tc.function.name, msg));
                        continue;
                    }
                };

            let _ = tx.send(AgentEvent::ToolPlanned {
                id: tc.id.clone(),
                name: prepared.name.clone(),
                kind: prepared.kind,
                summary: prepared.summary.clone(),
                diff: prepared.preview_diff.clone(),
            });

            // Permission gate
            let (mode_now, already_allowed) = {
                let guard = control.lock().unwrap();
                (
                    guard.mode,
                    guard.always_allowed_tools.contains(&prepared.name),
                )
            };

            if mode_now == PermissionMode::Plan && prepared.kind != ToolKind::ReadOnly {
                let denial = format!(
                    "Denied: '{}' is blocked in PLAN (read-only) mode. Ask the user or wait until they switch modes.",
                    prepared.name
                );
                let _ = tx.send(AgentEvent::ToolFinished {
                    id: tc.id.clone(),
                    ok: false,
                    output: denial.clone(),
                    diff: None,
                });
                history.push(Message::tool_result(&tc.id, &prepared.name, denial));
                continue;
            }

            let needs_approval = !already_allowed
                && match (mode_now, prepared.kind) {
                    (_, ToolKind::ReadOnly) => false,
                    (PermissionMode::Yolo, _) => false,
                    (PermissionMode::AutoEdit, ToolKind::Write) => false,
                    (PermissionMode::AutoEdit, ToolKind::Shell) => true,
                    (PermissionMode::Ask, ToolKind::Write | ToolKind::Shell) => true,
                    (PermissionMode::Plan, _) => true,
                };

            if needs_approval {
                let (resp_tx, resp_rx) = oneshot::channel();
                let command_detail = if prepared.name == "run_command" {
                    prepared
                        .args
                        .get("command")
                        .and_then(|v| v.as_str())
                        .map(ToString::to_string)
                } else {
                    None
                };

                let _ = tx.send(AgentEvent::ApprovalRequired(ApprovalRequest {
                    id: tc.id.clone(),
                    name: prepared.name.clone(),
                    kind: prepared.kind,
                    summary: prepared.summary.clone(),
                    diff: prepared.preview_diff.clone(),
                    command_detail,
                    responder: resp_tx,
                }));

                let mut cancel_wait = cancel_rx.clone();
                let decision = tokio::select! {
                    _ = async {
                        if *cancel_wait.borrow() { return; }
                        while cancel_wait.changed().await.is_ok() {
                            if *cancel_wait.borrow() { return; }
                        }
                        futures_util::future::pending::<()>().await;
                    } => {
                        cancelled = true;
                        ApprovalDecision::Deny(Some("Cancelled by user.".to_string()))
                    }
                    res = resp_rx => {
                        res.unwrap_or(ApprovalDecision::Deny(Some("Approval channel closed.".to_string())))
                    }
                };

                match decision {
                    ApprovalDecision::AllowOnce => {}
                    ApprovalDecision::AllowToolForSession => {
                        control
                            .lock()
                            .unwrap()
                            .always_allowed_tools
                            .insert(prepared.name.clone());
                    }
                    ApprovalDecision::SwitchAutoEdit => {
                        control.lock().unwrap().mode = PermissionMode::AutoEdit;
                    }
                    ApprovalDecision::SwitchYolo => {
                        control.lock().unwrap().mode = PermissionMode::Yolo;
                    }
                    ApprovalDecision::Deny(reason) => {
                        let msg = match reason {
                            Some(r) if !r.trim().is_empty() => {
                                format!("User denied execution of '{}': {r}", prepared.name)
                            }
                            _ => format!("User denied execution of '{}'.", prepared.name),
                        };
                        let _ = tx.send(AgentEvent::ToolFinished {
                            id: tc.id.clone(),
                            ok: false,
                            output: msg.clone(),
                            diff: None,
                        });
                        history.push(Message::tool_result(&tc.id, &prepared.name, msg));
                        if cancelled {
                            break;
                        }
                        continue;
                    }
                }
            }

            let _ = tx.send(AgentEvent::ToolRunning { id: tc.id.clone() });
            let out = execute_tool(&cfg.workspace, &prepared, cancel_rx.clone()).await;

            let _ = tx.send(AgentEvent::ToolFinished {
                id: tc.id.clone(),
                ok: out.ok,
                output: out.output.clone(),
                diff: out.diff.clone(),
            });

            let mut tool_msg = out.output;
            if let Some(d) = out.diff {
                tool_msg.push_str("\n\nDiff:\n");
                tool_msg.push_str(&d);
            }
            history.push(Message::tool_result(&tc.id, &prepared.name, tool_msg));
        }

        if cancelled {
            break;
        }

        if step + 1 == cfg.max_steps {
            let _ = tx.send(AgentEvent::Note {
                text: format!(
                    "Reached max agent tool steps ({}) for this turn.",
                    cfg.max_steps
                ),
                is_error: false,
            });
        }
    }

    let _ = tx.send(AgentEvent::TurnFinished {
        messages: history,
        cancelled,
    });
}

/// Non-TUI headless execution (`inkling code --exec "prompt"`) for scripting & verification.
pub async fn run_headless(api: ApiClient, cfg: HarnessConfig, prompt: String) -> Result<i32> {
    let control = Arc::new(Mutex::new(SharedControl::new(cfg.mode)));
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let (tx, mut rx) = mpsc::unbounded_channel();

    let history = vec![Message::user(prompt)];
    let show_thinking = cfg.show_thinking;
    let handle = tokio::spawn(run_agent_turn(api, cfg, control, history, cancel_rx, tx));

    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = cancel_tx.send(true);
        }
    });

    let mut in_thinking = false;
    let mut exit_code = 0;

    while let Some(ev) = rx.recv().await {
        match ev {
            AgentEvent::AssistantStart => {}
            AgentEvent::AssistantAbort => {
                if in_thinking {
                    eprintln!("\n[/thinking]\x1b[0m");
                    in_thinking = false;
                }
            }
            AgentEvent::HistoryCompacted { note, .. } => {
                eprintln!("\x1b[33m[{note}]\x1b[0m");
            }
            AgentEvent::ReasoningDelta(chunk) => {
                if show_thinking {
                    if !in_thinking {
                        eprint!("\x1b[2m[thinking]\n");
                        in_thinking = true;
                    }
                    eprint!("{chunk}");
                    let _ = std::io::stderr().flush();
                }
            }
            AgentEvent::ContentDelta(chunk) => {
                if in_thinking {
                    eprintln!("\n[/thinking]\x1b[0m");
                    in_thinking = false;
                }
                print!("{chunk}");
                let _ = std::io::stdout().flush();
            }
            AgentEvent::LiveTimings(_) => {}
            AgentEvent::AssistantDone { .. } => {
                if in_thinking {
                    eprintln!("\n[/thinking]\x1b[0m");
                    in_thinking = false;
                }
                println!();
            }
            AgentEvent::ToolPlanned { name, summary, .. } => {
                eprintln!("\x1b[36m[tool] {name}: {summary}\x1b[0m");
            }
            AgentEvent::ApprovalRequired(req) => {
                eprint!(
                    "\x1b[33mApprove {} ({})? [y/N]: \x1b[0m",
                    req.name, req.summary
                );
                let _ = std::io::stderr().flush();
                let mut answer = String::new();
                let decision = if std::io::stdin().read_line(&mut answer).is_ok()
                    && matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                {
                    ApprovalDecision::AllowOnce
                } else {
                    ApprovalDecision::Deny(None)
                };
                let _ = req.responder.send(decision);
            }
            AgentEvent::ToolRunning { .. } => {}
            AgentEvent::ToolFinished { ok, output, .. } => {
                let first = output.lines().next().unwrap_or("");
                if ok {
                    eprintln!("\x1b[32m  ✓ {first}\x1b[0m");
                } else {
                    eprintln!("\x1b[31m  ✗ {first}\x1b[0m");
                }
            }
            AgentEvent::Note { text, is_error } => {
                if is_error {
                    eprintln!("\x1b[31mError: {text}\x1b[0m");
                    exit_code = 1;
                } else {
                    eprintln!("\x1b[2m{text}\x1b[0m");
                }
            }
            AgentEvent::TurnFinished { cancelled, .. } => {
                if cancelled {
                    exit_code = 130;
                }
                break;
            }
        }
    }

    let _ = handle.await;
    Ok(exit_code)
}
