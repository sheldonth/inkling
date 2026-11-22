use anyhow::{bail, Result};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Ask,
    AutoEdit,
    Yolo,
    Plan,
}

impl PermissionMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::Ask => Self::AutoEdit,
            Self::AutoEdit => Self::Yolo,
            Self::Yolo => Self::Plan,
            Self::Plan => Self::Ask,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ask => "ASK",
            Self::AutoEdit => "AUTO-EDIT",
            Self::Yolo => "YOLO",
            Self::Plan => "PLAN",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Ask => "auto-read, confirm writes & shell",
            Self::AutoEdit => "auto-read & edits, confirm shell",
            Self::Yolo => "auto-approve all tools",
            Self::Plan => "read-only analysis & planning",
        }
    }
}

impl fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for PermissionMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ask" | "confirm" | "default" => Ok(Self::Ask),
            "auto-edit" | "autoedit" | "edit" => Ok(Self::AutoEdit),
            "yolo" | "auto" | "full-auto" | "allow-all" => Ok(Self::Yolo),
            "plan" | "readonly" | "read-only" => Ok(Self::Plan),
            other => bail!(
                "Unknown permission mode '{other}'; expected one of: ask, auto-edit, yolo, plan"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortLevel {
    Default,
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl EffortLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl fmt::Display for EffortLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EffortLevel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "none" => Ok(Self::None),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            other => bail!(
                "Unknown reasoning effort '{other}'; expected one of: default, none, minimal, low, medium, high, xhigh, max"
            ),
        }
    }
}

pub fn parse_lora_scale(raw: &str) -> Result<f32> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "on" | "abliterated" | "uncensored" | "enable" | "enabled" => Ok(1.0),
        "off" | "base" | "disable" | "disabled" | "none" => Ok(0.0),
        other => other.parse::<f32>().map_err(|_| {
            anyhow::anyhow!(
                "Invalid LoRA scale '{raw}'; expected a number (e.g. 1.0, 0.6, 0.0) or on/off/base"
            )
        }),
    }
}

pub fn format_lora_status(scale: Option<f32>) -> String {
    match scale {
        None => "none".to_string(),
        Some(v) if v.abs() < 1e-5 => "0.0 (base)".to_string(),
        Some(v) if (v - 1.0).abs() < 1e-5 => "1.0 (abliterated)".to_string(),
        Some(v) => format!("{v:.2} (custom)"),
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HarnessConfig {
    pub server_url: String,
    pub api_key: String,
    pub model: String,
    pub workspace: PathBuf,
    pub mode: PermissionMode,
    pub effort: EffortLevel,
    /// Per-request LoRA override; `None` inherits the server's active global LoRA scale.
    pub lora_scale: Option<f32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub max_tokens: i32,
    pub show_thinking: bool,
    pub expand_tools: bool,
    pub system_prompt_extra: Option<String>,
    pub max_steps: usize,
    pub ctx_size_hint: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_cycles_in_order() {
        assert_eq!(PermissionMode::Ask.cycle(), PermissionMode::AutoEdit);
        assert_eq!(PermissionMode::AutoEdit.cycle(), PermissionMode::Yolo);
        assert_eq!(PermissionMode::Yolo.cycle(), PermissionMode::Plan);
        assert_eq!(PermissionMode::Plan.cycle(), PermissionMode::Ask);
    }

    #[test]
    fn parse_lora_aliases() {
        assert_eq!(parse_lora_scale("on").unwrap(), 1.0);
        assert_eq!(parse_lora_scale("abliterated").unwrap(), 1.0);
        assert_eq!(parse_lora_scale("off").unwrap(), 0.0);
        assert_eq!(parse_lora_scale("base").unwrap(), 0.0);
        assert!((parse_lora_scale("0.65").unwrap() - 0.65).abs() < 1e-6);
        assert!(parse_lora_scale("wat").is_err());
    }

    #[test]
    fn parse_efforts() {
        assert_eq!("medium".parse::<EffortLevel>().unwrap(), EffortLevel::Medium);
        assert_eq!("xhigh".parse::<EffortLevel>().unwrap(), EffortLevel::XHigh);
        assert!("bogus".parse::<EffortLevel>().is_err());
    }
}
