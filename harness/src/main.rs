mod agent;
mod api;
mod config;
mod prompt;
mod session;
mod tools;
mod tui;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use crate::api::ApiClient;
use crate::config::{parse_lora_scale, EffortLevel, HarnessConfig, PermissionMode};

#[derive(Debug, Parser)]
#[command(
    name = "inkling-code",
    version,
    about = "Fast native Rust TUI coding agent harness for Inkling-2-Mini (abliterated)"
)]
struct Cli {
    /// Base URL of the running llama-server
    #[arg(long)]
    server_url: Option<String>,

    /// Optional Bearer API key for llama-server
    #[arg(long)]
    api_key: Option<String>,

    /// Model alias exposed by llama-server
    #[arg(long)]
    model: Option<String>,

    /// Workspace directory to operate in (defaults to current working directory)
    #[arg(short = 'w', long)]
    workspace: Option<PathBuf>,

    /// Tool permission mode: ask, auto-edit, yolo, plan
    #[arg(short = 'm', long, default_value = "ask")]
    mode: String,

    /// Reasoning effort: default, none, minimal, low, medium, high, xhigh, max
    #[arg(short = 'e', long)]
    effort: Option<String>,

    /// Per-request LoRA scale override (float or on/off/base). Omit to inherit server global.
    #[arg(long)]
    lora_scale: Option<String>,

    /// Force base model (LoRA scale 0.0) for this session
    #[arg(long)]
    base: bool,

    /// Sampling temperature
    #[arg(long, default_value_t = 1.0)]
    temp: f32,

    /// Top-p sampling
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Top-k sampling
    #[arg(long, default_value_t = 40)]
    top_k: i32,

    /// Min-p sampling
    #[arg(long, default_value_t = 0.05)]
    min_p: f32,

    /// Max tokens per completion (-1 = unlimited)
    #[arg(short = 'n', long, default_value_t = -1, allow_hyphen_values = true)]
    max_tokens: i32,

    /// Start with thinking blocks collapsed
    #[arg(long)]
    hide_thinking: bool,

    /// Extra system instructions appended to the agent prompt
    #[arg(short = 's', long)]
    system: Option<String>,

    /// Maximum tool-loop steps per user turn
    #[arg(long, default_value_t = 30)]
    max_steps: usize,

    /// Run a single prompt non-interactively without TUI (great for scripting/CI)
    #[arg(long)]
    exec: Option<String>,

    /// Optional initial prompt to send immediately in TUI mode
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let workspace = match cli.workspace {
        Some(p) => p,
        None => std::env::current_dir().context("Failed to determine current directory")?,
    };
    let workspace = workspace
        .canonicalize()
        .with_context(|| format!("Workspace path does not exist: {}", workspace.display()))?;

    let server_url = cli
        .server_url
        .or_else(|| std::env::var("INKLING_SERVER_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:9931".to_string());
    let api_key = cli
        .api_key
        .or_else(|| std::env::var("INKLING_API_KEY").ok())
        .unwrap_or_default();
    let default_model = cli
        .model
        .or_else(|| std::env::var("INKLING_ALIAS").ok())
        .unwrap_or_else(|| "inkling-abliterated".to_string());
    let effort_raw = cli
        .effort
        .or_else(|| std::env::var("INKLING_EFFORT").ok())
        .unwrap_or_else(|| "medium".to_string());

    let mode: PermissionMode = cli.mode.parse()?;
    let effort: EffortLevel = effort_raw.parse()?;
    let lora_scale = if cli.base {
        Some(0.0)
    } else if let Some(raw) = &cli.lora_scale {
        Some(parse_lora_scale(raw)?)
    } else {
        None
    };

    let api = ApiClient::new(&server_url, &api_key)?;

    // Quick startup check & metadata probe
    api.health().await.with_context(|| {
        format!(
            "Could not reach llama-server at {server_url}. Start it with `inkling start`."
        )
    })?;

    let props = api.props().await.ok();
    let server_lora_scale = api
        .get_lora_adapters()
        .await
        .ok()
        .and_then(|v| v.first().map(|a| a.scale));

    let model = props
        .as_ref()
        .and_then(|p| p.model_alias.clone())
        .unwrap_or(default_model);
    let ctx_size_hint = props.as_ref().and_then(|p| p.n_ctx).unwrap_or(32768);

    let cfg = HarnessConfig {
        server_url,
        api_key,
        model,
        workspace,
        mode,
        effort,
        lora_scale,
        temperature: cli.temp,
        top_p: cli.top_p,
        top_k: cli.top_k,
        min_p: cli.min_p,
        max_tokens: cli.max_tokens,
        show_thinking: !cli.hide_thinking,
        expand_tools: false,
        system_prompt_extra: cli.system,
        max_steps: cli.max_steps,
        ctx_size_hint,
    };

    if let Some(exec_prompt) = cli.exec {
        let rc = agent::run_headless(api, cfg, exec_prompt).await?;
        std::process::exit(rc);
    }

    let initial_prompt = if cli.prompt.is_empty() {
        None
    } else {
        Some(cli.prompt.join(" "))
    };

    tui::run_tui(api, cfg, server_lora_scale, initial_prompt).await
}
