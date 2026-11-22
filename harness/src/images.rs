use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "tga", "webp", "heic", "heif", "tif", "tiff",
];

const SIPS_CONVERT_EXTENSIONS: &[&str] = &["heic", "heif", "webp", "tif", "tiff"];

#[derive(Debug, Clone)]
pub struct StagedImage {
    pub path: PathBuf,
    pub name: String,
    pub size_bytes: u64,
}

impl StagedImage {
    pub fn from_path(path: PathBuf) -> Result<Self> {
        let meta = fs::metadata(&path)
            .with_context(|| format!("Image file not found: {}", path.display()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        Ok(Self {
            path,
            name,
            size_bytes: meta.len(),
        })
    }
}

pub fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{bytes} B")
    }
}

/// Pure-Rust RFC 4648 base64 encoder (zero extra dependencies).
pub fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn unescape_shell_path(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn expand_tilde(path_str: &str) -> PathBuf {
    if let Some(rest) = path_str.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path_str)
}

fn scan_shell_tokens(text: &str) -> Vec<(usize, usize, &str)> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut tokens = Vec::new();
    let mut i = 0usize;

    while i < n {
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        if bytes[i] == b'\'' || bytes[i] == b'"' {
            let q = bytes[i];
            if let Some(rel_end) = bytes[i + 1..].iter().position(|&b| b == q) {
                i = i + 1 + rel_end + 1;
                tokens.push((start, i, &text[start..i]));
                continue;
            }
        }
        while i < n && !bytes[i].is_ascii_whitespace() {
            if bytes[i] == b'\\' && i + 1 < n {
                i += 2;
            } else {
                i += 1;
            }
        }
        tokens.push((start, i, &text[start..i]));
    }
    tokens
}

pub fn parse_image_path_token(raw_token: &str, workspace: &Path) -> Option<PathBuf> {
    let token = raw_token.trim();
    if token.is_empty() {
        return None;
    }

    let mut candidate = if token.len() >= 2
        && ((token.starts_with('\'') && token.ends_with('\''))
            || (token.starts_with('"') && token.ends_with('"')))
    {
        token[1..token.len() - 1].to_string()
    } else {
        unescape_shell_path(token)
    };

    if let Some(rest) = candidate.strip_prefix("file://") {
        let stripped = rest.strip_prefix("localhost").unwrap_or(rest);
        candidate = percent_decode(stripped);
    }

    if candidate.is_empty() {
        return None;
    }

    let mut path = expand_tilde(&candidate);
    if !path.is_absolute() {
        let ws_candidate = workspace.join(&path);
        if ws_candidate.is_file() {
            path = ws_candidate;
        }
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())?;
    if !IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        return None;
    }

    if path.is_file() {
        Some(path.canonicalize().unwrap_or(path))
    } else {
        None
    }
}

/// Extract existing drag-and-dropped image paths from raw input, returning `(remaining_text, paths)`.
pub fn extract_dropped_images(raw_text: &str, workspace: &Path) -> (String, Vec<PathBuf>) {
    let tokens = scan_shell_tokens(raw_text);
    let mut images = Vec::new();
    let mut remove_spans = Vec::new();

    for (start, end, tok) in tokens {
        if let Some(img) = parse_image_path_token(tok, workspace) {
            images.push(img);
            remove_spans.push((start, end));
        }
    }

    if remove_spans.is_empty() {
        return (raw_text.to_string(), images);
    }

    let mut out = String::with_capacity(raw_text.len());
    let mut cursor = 0usize;
    for (start, end) in remove_spans {
        out.push_str(&raw_text[cursor..start]);
        cursor = end;
    }
    out.push_str(&raw_text[cursor..]);

    // Collapse double spaces left behind where paths were stripped, preserving newlines.
    let cleaned_lines: Vec<String> = out
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let remaining = cleaned_lines.join("\n").trim().to_string();
    (remaining, images)
}

fn mime_for_ext(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "webp" => "image/webp",
        "tif" | "tiff" => "image/tiff",
        "heic" | "heif" => "image/heic",
        _ => "image/png",
    }
}

/// Read an image into a `data:<mime>;base64,...` URL (auto-converting HEIC/WebP/TIFF via macOS `sips`).
pub fn encode_image_data_url(path: &Path) -> Result<(String, usize)> {
    if !path.is_file() {
        bail!("Image file not found: {}", path.display());
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "png".to_string());

    let (bytes, mime) = if cfg!(target_os = "macos")
        && SIPS_CONVERT_EXTENSIONS.contains(&ext.as_str())
        && Path::new("/usr/bin/sips").is_file()
    {
        let tmp_out = std::env::temp_dir().join(format!(
            "inkling-img-{}-{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let status = Command::new("/usr/bin/sips")
            .args(["-s", "format", "png"])
            .arg(path)
            .arg("--out")
            .arg(&tmp_out)
            .output();
        match status {
            Ok(out) if out.status.success() && tmp_out.is_file() => {
                let data = fs::read(&tmp_out)?;
                let _ = fs::remove_file(&tmp_out);
                (data, "image/png")
            }
            _ => {
                let _ = fs::remove_file(&tmp_out);
                (fs::read(path)?, mime_for_ext(&ext))
            }
        }
    } else {
        (
            fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?,
            mime_for_ext(&ext),
        )
    };

    let size = bytes.len();
    let encoded = base64_encode(&bytes);
    Ok((format!("data:{mime};base64,{encoded}"), size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn extracts_macos_escaped_and_quoted_dropped_images() {
        let dir = std::env::temp_dir().join(format!("inkling-img-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let p1 = dir.join("Screenshot 2026-11-21 at 8.00 PM.png");
        let p2 = dir.join("diagram.jpg");
        fs::write(&p1, b"\x89PNG\r\n\x1a\nfake").unwrap();
        fs::write(&p2, b"\xff\xd8\xff\xe0fake").unwrap();

        let escaped = p1.display().to_string().replace(' ', r"\ ");
        let (rem, imgs) = extract_dropped_images(&escaped, &dir);
        assert_eq!(rem, "");
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0], p1.canonicalize().unwrap());

        let mixed = format!("don't ignore {escaped} or '{}' please", p2.display());
        let (rem2, imgs2) = extract_dropped_images(&mixed, &dir);
        assert_eq!(rem2, "don't ignore or please");
        assert_eq!(imgs2.len(), 2);

        let uri = format!("file://{}", p2.display());
        assert_eq!(
            parse_image_path_token(&uri, &dir),
            Some(p2.canonicalize().unwrap())
        );

        let (data_url, sz) = encode_image_data_url(&p1).unwrap();
        assert!(data_url.starts_with("data:image/png;base64,"));
        assert_eq!(sz, 12);

        let _ = fs::remove_dir_all(&dir);
    }
}
