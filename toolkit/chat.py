"""Interactive REPL, one-shot/piped prompts, and direct llama-cli runner."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None  # type: ignore[assignment]

from .client import ChatResult, ServerClient, ServerClientError
from .config import (
    AppConfig,
    EFFORT_LEVELS,
    ConfigError,
    ensure_executable,
    ensure_file_exists,
    format_lora_status,
    validate_effort,
)
from .server import ServerError, ensure_server_running, get_server_status


DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _use_color(stream: Any) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _style(text: str, code: str, stream: Any) -> str:
    if not _use_color(stream):
        return text
    return f"{code}{text}{RESET}"


def _is_libedit() -> bool:
    if readline is None:
        return False
    backend = getattr(readline, "backend", "")
    if backend == "editline":
        return True
    return "libedit" in (readline.__doc__ or "").lower()


def _readline_prompt(label: str, code: str = GREEN + BOLD, stream: Any = None) -> str:
    """Format a prompt string safely for Python's readline/libedit `input()`."""
    target = sys.stdout if stream is None else stream
    if not _use_color(target):
        return label
    # macOS Python uses libedit, whose screen-refresh logic miscalculates column width
    # and corrupts Up-arrow history recall (leaving stray `[0m` artifacts) whenever ANSI
    # escape sequences appear inside the `input()` prompt string.
    if _is_libedit():
        return label
    # GNU readline requires non-printing escapes to be wrapped in \001...\002.
    return f"\001{code}\002{label}\001{RESET}\002"


def format_stats(result: ChatResult) -> str:
    t = result.timings or {}
    parts: list[str] = []
    if "prompt_n" in t and "prompt_per_second" in t:
        parts.append(f"prompt: {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} t/s")
    if "predicted_n" in t and "predicted_per_second" in t:
        parts.append(f"gen: {t['predicted_n']} tok @ {t['predicted_per_second']:.1f} t/s")
    if not parts and result.usage:
        u = result.usage
        parts.append(
            f"tokens: prompt={u.get('prompt_tokens', '?')} completion={u.get('completion_tokens', '?')}"
        )
    return " | ".join(parts)


class StreamPrinter:
    """Renders streamed thinking + final content cleanly for TTYs and pipes."""

    def __init__(
        self,
        *,
        show_thinking: bool,
        stdout: Any = None,
        stderr: Any = None,
        piped: bool | None = None,
    ) -> None:
        self.stdout = sys.stdout if stdout is None else stdout
        self.stderr = sys.stderr if stderr is None else stderr
        self.piped = (not getattr(self.stdout, "isatty", lambda: False)()) if piped is None else piped
        self.show_thinking = show_thinking
        self._in_thinking = False
        self._saw_thinking = False
        self._saw_content = False

    def on_reasoning(self, chunk: str) -> None:
        if not self.show_thinking or not chunk:
            return
        # When stdout is piped, send thinking to stderr so stdout only gets the final answer.
        target = self.stderr if self.piped else self.stdout
        if not self._in_thinking:
            self._in_thinking = True
            self._saw_thinking = True
            header = _style("[thinking]\n", DIM, target)
            target.write(header)
        target.write(_style(chunk, DIM, target))
        target.flush()

    def on_content(self, chunk: str) -> None:
        if not chunk:
            return
        if self._in_thinking:
            target = self.stderr if self.piped else self.stdout
            target.write(_style("\n[/thinking]\n\n", DIM, target))
            target.flush()
            self._in_thinking = False
        self._saw_content = True
        self.stdout.write(chunk)
        self.stdout.flush()

    def finish(self) -> None:
        if self._in_thinking:
            target = self.stderr if self.piped else self.stdout
            target.write(_style("\n[/thinking]\n", DIM, target))
            target.flush()
            self._in_thinking = False
        if self._saw_content:
            self.stdout.write("\n")
            self.stdout.flush()


IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tga",
        ".webp",
        ".heic",
        ".heif",
        ".tif",
        ".tiff",
    }
)
SIPS_CONVERT_EXTENSIONS: frozenset[str] = frozenset(
    {".heic", ".heif", ".webp", ".tif", ".tiff"}
)
_SHELL_UNESCAPE_RE = re.compile(r"\\(.)")


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} B"


def _scan_shell_tokens(text: str) -> list[tuple[int, int, str]]:
    """Scan whitespace-delimited tokens while respecting quotes and backslash-escaped spaces."""
    tokens: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        if text[i] in ("'", '"'):
            quote = text[i]
            end_quote = text.find(quote, i + 1)
            if end_quote != -1:
                i = end_quote + 1
                tokens.append((start, i, text[start:i]))
                continue
        while i < n and not text[i].isspace():
            if text[i] == "\\" and i + 1 < n:
                i += 2
            else:
                i += 1
        tokens.append((start, i, text[start:i]))
    return tokens


def parse_image_path_token(raw_token: str) -> Path | None:
    """Resolve a terminal-pasted / drag-and-dropped token to an existing image Path if valid."""
    token = raw_token.strip()
    if not token:
        return None

    # Strip matching outer quotes if present.
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        candidate = token[1:-1]
    else:
        # Unescape macOS terminal drag-and-drop backslash escapes (e.g. `\ `, `\(`, `\'`).
        candidate = _SHELL_UNESCAPE_RE.sub(r"\1", token)

    if candidate.startswith("file://"):
        parsed = urlparse(candidate)
        if parsed.scheme == "file":
            candidate = unquote(parsed.path)

    if not candidate:
        return None

    path = Path(candidate).expanduser()
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        if path.is_file():
            return path.resolve()
    except OSError:
        return None
    return None


def extract_dropped_images(raw_text: str) -> tuple[str, list[Path]]:
    """Extract existing drag-and-dropped image paths from user input, returning (remaining_text, images)."""
    tokens = _scan_shell_tokens(raw_text)
    images: list[Path] = []
    remove_spans: list[tuple[int, int]] = []

    for start, end, tok in tokens:
        img = parse_image_path_token(tok)
        if img is not None:
            images.append(img)
            remove_spans.append((start, end))

    if not remove_spans:
        return raw_text, []

    pieces: list[str] = []
    cursor = 0
    for start, end in remove_spans:
        pieces.append(raw_text[cursor:start])
        cursor = end
    pieces.append(raw_text[cursor:])
    remaining = re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()
    return remaining, images


def encode_image_data_url(path: Path) -> tuple[str, int]:
    """Read an image file (auto-converting HEIC/WebP/TIFF via macOS sips if needed) into a data URL."""
    if not path.is_file():
        raise ConfigError(f"Image file not found: {path}")

    ext = path.suffix.lower()
    raw_bytes: bytes
    mime: str

    if ext in SIPS_CONVERT_EXTENSIONS and sys.platform == "darwin" and Path("/usr/bin/sips").is_file():
        with tempfile.TemporaryDirectory() as tmp:
            out_png = Path(tmp) / "converted.png"
            proc = subprocess.run(
                ["/usr/bin/sips", "-s", "format", "png", str(path), "--out", str(out_png)],
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0 and out_png.is_file():
                raw_bytes = out_png.read_bytes()
                mime = "image/png"
            else:
                raw_bytes = path.read_bytes()
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
    else:
        raw_bytes = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}", len(raw_bytes)


def build_user_message_content(text: str, images: list[Path]) -> str | list[dict[str, Any]]:
    """Build OpenAI chat content: plain string for text-only, or multimodal array with image_url parts."""
    if not images:
        return text

    parts: list[dict[str, Any]] = []
    for img_path in images:
        data_url, _ = encode_image_data_url(img_path)
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    if text.strip():
        parts.append({"type": "text", "text": text})
    return parts


def _server_supports_vision(client: ServerClient, props: dict[str, Any] | None = None) -> bool:
    data = props
    if data is None:
        try:
            data = client.props(timeout=3.0)
        except ServerClientError:
            return False
    modalities = data.get("modalities")
    if isinstance(modalities, dict):
        return bool(modalities.get("vision", False))
    return False


def _vision_missing_message(cfg: AppConfig) -> str:
    if cfg.paths.mmproj is not None and cfg.paths.mmproj.is_file():
        return (
            "The running llama-server does not have vision enabled yet, even though an mmproj file is configured "
            f"({cfg.paths.mmproj.name}).\n"
            "Restart the server to load it: `inkling restart`"
        )
    return (
        "Image input requires Inkling-2-Mini's vision projector (`mmproj`) GGUF, which is not loaded on llama-server.\n"
        f"Drop an `*mmproj*.gguf` file into {cfg.repo_root} (or set `paths.mmproj` in inkling.toml), "
        "then run `inkling restart`."
    )


def compose_user_prompt(
    prompt_text: str,
    *,
    stdin_text: str | None = None,
    files: list[Path] | None = None,
) -> str:
    """Combine attached files, piped stdin, and prompt text into one clean user turn."""
    sections: list[str] = []
    for file_path in files or []:
        if not file_path.is_file():
            raise ConfigError(f"Input file not found: {file_path}")
        content = file_path.read_text(encoding="utf-8", errors="replace")
        sections.append(f"--- File: {file_path} ---\n{content}")

    if stdin_text is not None and stdin_text.strip():
        sections.append(stdin_text.rstrip("\n"))

    if prompt_text.strip():
        sections.append(prompt_text.strip())

    return "\n\n".join(sections)


def _current_server_lora_scale(client: ServerClient) -> float | None:
    try:
        adapters = client.get_lora_adapters(timeout=3.0)
    except ServerClientError:
        return None
    if adapters:
        try:
            return float(adapters[0].get("scale", 0.0))
        except (TypeError, ValueError):
            return None
    return None


def run_ask(
    cfg: AppConfig,
    *,
    prompt: str = "",
    files: list[Path] | None = None,
    images: list[Path] | None = None,
    system: str | None = None,
    effort: str | None = None,
    lora_scale: float | None = None,
    base_only: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    stream: bool = True,
    show_thinking: bool | None = None,
    show_stats: bool | None = None,
    json_output: bool = False,
    auto_start: bool | None = None,
    direct: bool = False,
) -> int:
    """Run a single-turn completion (supports positional prompt, -f files, -i images, and stdin piping)."""
    stdin_text: str | None = None
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()

    cleaned_prompt, dropped_images = extract_dropped_images(prompt)
    all_images: list[Path] = list(images or [])
    for img in dropped_images:
        if img not in all_images:
            all_images.append(img)

    full_prompt = compose_user_prompt(cleaned_prompt, stdin_text=stdin_text, files=files)
    if not full_prompt and not all_images:
        raise ConfigError("No prompt provided. Pass a prompt argument, -f FILE, -i IMAGE, or pipe text via stdin.")

    if direct:
        return run_direct(
            cfg,
            prompt=full_prompt or "Describe this image.",
            images=all_images,
            system=system,
            effort=effort,
            lora_scale=lora_scale,
            base_only=base_only,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            seed=seed,
            single_turn=True,
            show_timings=bool(show_stats),
        )

    status = ensure_server_running(cfg, auto_start=auto_start)
    client = ServerClient(status.base_url, api_key=cfg.server.api_key)

    if all_images and not _server_supports_vision(client, status.props):
        raise ServerError(_vision_missing_message(cfg))

    messages: list[dict[str, Any]] = []
    use_system = cfg.generation.system_prompt if system is None else system
    if use_system:
        messages.append({"role": "system", "content": use_system})
    messages.append({"role": "user", "content": build_user_message_content(full_prompt, all_images)})

    use_effort = validate_effort(effort) if effort is not None else cfg.generation.reasoning_effort
    effective_lora = 0.0 if base_only else lora_scale
    piped = not sys.stdout.isatty()
    if show_thinking is None:
        # In pipes, omit thinking by default so stdout/stderr stay clean for scripting.
        use_thinking = False if piped else cfg.generation.show_thinking
    else:
        use_thinking = show_thinking

    use_stats = cfg.generation.show_stats if show_stats is None else show_stats

    if json_output:
        result = client.chat_completion(
            messages,
            model=cfg.server.alias,
            stream=False,
            reasoning_effort=use_effort,
            lora_scale=effective_lora,
            temperature=cfg.generation.temperature if temperature is None else temperature,
            top_p=cfg.generation.top_p if top_p is None else top_p,
            top_k=cfg.generation.top_k if top_k is None else top_k,
            min_p=cfg.generation.min_p if min_p is None else min_p,
            max_tokens=cfg.generation.max_tokens if max_tokens is None else max_tokens,
            seed=seed,
        )
        print(
            json.dumps(
                {
                    "content": result.content,
                    "reasoning": result.reasoning,
                    "finish_reason": result.finish_reason,
                    "timings": result.timings,
                    "usage": result.usage,
                },
                indent=2,
            )
        )
        return 0

    printer = StreamPrinter(show_thinking=use_thinking, piped=piped)
    result = client.chat_completion(
        messages,
        model=cfg.server.alias,
        stream=stream,
        reasoning_effort=use_effort,
        lora_scale=effective_lora,
        temperature=cfg.generation.temperature if temperature is None else temperature,
        top_p=cfg.generation.top_p if top_p is None else top_p,
        top_k=cfg.generation.top_k if top_k is None else top_k,
        min_p=cfg.generation.min_p if min_p is None else min_p,
        max_tokens=cfg.generation.max_tokens if max_tokens is None else max_tokens,
        seed=seed,
        on_reasoning=printer.on_reasoning,
        on_content=printer.on_content,
    )
    printer.finish()

    if use_stats:
        stats_str = format_stats(result)
        if stats_str:
            sys.stderr.write(_style(f"[{stats_str}]\n", DIM, sys.stderr))
            sys.stderr.flush()

    return 130 if result.interrupted else 0


def run_chat(
    cfg: AppConfig,
    *,
    system: str | None = None,
    images: list[Path] | None = None,
    effort: str | None = None,
    lora_scale: float | None = None,
    base_only: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    max_tokens: int | None = None,
    show_thinking: bool | None = None,
    show_stats: bool | None = None,
    auto_start: bool | None = None,
    native: bool = False,
    initial_prompt: str | None = None,
) -> int:
    """Interactive multi-turn chat against the persistent llama-server."""
    status = ensure_server_running(cfg, auto_start=auto_start)

    if native:
        return run_direct(
            cfg,
            attach=True,
            attach_url=status.base_url,
            system=system,
            images=images,
            prompt=initial_prompt,
        )

    client = ServerClient(status.base_url, api_key=cfg.server.api_key)
    session_system = cfg.generation.system_prompt if system is None else system
    session_effort = validate_effort(effort) if effort is not None else cfg.generation.reasoning_effort
    session_lora: float | None = 0.0 if base_only else lora_scale
    session_temp = cfg.generation.temperature if temperature is None else temperature
    session_top_p = cfg.generation.top_p if top_p is None else top_p
    session_top_k = cfg.generation.top_k if top_k is None else top_k
    session_min_p = cfg.generation.min_p if min_p is None else min_p
    session_max_tokens = cfg.generation.max_tokens if max_tokens is None else max_tokens
    session_thinking = cfg.generation.show_thinking if show_thinking is None else show_thinking
    session_stats = cfg.generation.show_stats if show_stats is None else show_stats

    history_file = cfg.history_file
    if readline is not None:
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            if history_file.is_file():
                readline.read_history_file(str(history_file))
            readline.set_history_length(500)
        except OSError:
            pass

    server_lora = _current_server_lora_scale(client)
    active_scale = server_lora if session_lora is None else session_lora
    has_vision = _server_supports_vision(client, status.props)
    vision_tag = "on" if has_vision else "off (no mmproj)"
    print(
        _style("Inkling-2-Mini", BOLD + CYAN, sys.stdout)
        + f"  server={status.base_url}  lora={format_lora_status(active_scale)}  "
        + f"effort={session_effort}  vision={vision_tag}"
    )
    print(
        _style(
            "Drag & drop image files into the prompt (or /image <path>) • /help for commands • \"\"\" multiline • Ctrl+C/Ctrl+D exit.\n",
            DIM,
            sys.stdout,
        )
    )

    messages: list[dict[str, Any]] = []
    staged_files: list[Path] = []
    staged_images: list[Path] = []

    for img_path in images or []:
        if not img_path.is_file():
            raise ConfigError(f"Image file not found: {img_path}")
        staged_images.append(img_path.resolve())
        print(f"Staged image: {img_path.name} ({_format_size(img_path.stat().st_size)})")
    if staged_images:
        print()

    def build_messages() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if session_system:
            out.append({"role": "system", "content": session_system})
        out.extend(messages)
        return out

    def check_vision_ready() -> bool:
        nonlocal has_vision
        if has_vision:
            return True
        # Re-check /props once in case the user restarted llama-server with mmproj.
        has_vision = _server_supports_vision(client)
        if not has_vision:
            print(_style(f"\n{_vision_missing_message(cfg)}\n", YELLOW, sys.stderr), file=sys.stderr)
            return False
        return True

    def send_user_turn(text: str) -> None:
        nonlocal staged_files, staged_images
        if staged_images and not check_vision_ready():
            return
        user_text = compose_user_prompt(text, files=staged_files)
        try:
            content = build_user_message_content(user_text, staged_images)
        except ConfigError as exc:
            print(_style(f"{exc}\n", YELLOW, sys.stderr), file=sys.stderr)
            return
        staged_files.clear()
        staged_images.clear()
        messages.append({"role": "user", "content": content})
        send_turn()

    def send_turn() -> None:
        printer = StreamPrinter(show_thinking=session_thinking, piped=False)
        try:
            result = client.chat_completion(
                build_messages(),
                model=cfg.server.alias,
                stream=True,
                reasoning_effort=session_effort,
                lora_scale=session_lora,
                temperature=session_temp,
                top_p=session_top_p,
                top_k=session_top_k,
                min_p=session_min_p,
                max_tokens=session_max_tokens,
                on_reasoning=printer.on_reasoning,
                on_content=printer.on_content,
            )
        except ServerClientError as exc:
            print(_style(f"\nError: {exc}\n", YELLOW, sys.stderr), file=sys.stderr)
            return

        printer.finish()
        if result.interrupted:
            print(_style("[interrupted]", YELLOW, sys.stderr), file=sys.stderr)
        if result.content or result.reasoning:
            msg: dict[str, Any] = {"role": "assistant", "content": result.content}
            if result.reasoning:
                msg["reasoning_content"] = result.reasoning
            messages.append(msg)
        if session_stats:
            stats_str = format_stats(result)
            if stats_str:
                print(_style(f"[{stats_str}]", DIM, sys.stderr), file=sys.stderr)
        print()

    if initial_prompt:
        cleaned_init, dropped_init = extract_dropped_images(initial_prompt)
        for img in dropped_init:
            staged_images.append(img)
        send_user_turn(cleaned_init)

    try:
        while True:
            try:
                badge = f"[{len(staged_images)} img] " if staged_images else ""
                prompt_label = _readline_prompt(f"{badge}you> ", GREEN + BOLD, sys.stdout)
                line = input(prompt_label)
            except (EOFError, KeyboardInterrupt):
                print()
                break

            stripped = line.strip()
            if not stripped:
                if staged_images or staged_files:
                    send_user_turn("")
                continue

            if stripped in ("\"\"\"", "/multiline"):
                print(_style("Enter multiline text; finish with \"\"\" on its own line:", DIM, sys.stdout))
                buf: list[str] = []
                while True:
                    try:
                        ml = input("... ")
                    except (EOFError, KeyboardInterrupt):
                        break
                    if ml.strip() == "\"\"\"":
                        break
                    buf.append(ml)
                raw_ml = "\n".join(buf).strip()
                cleaned_ml, dropped_ml = extract_dropped_images(raw_ml)
                for img in dropped_ml:
                    staged_images.append(img)
                    print(f"Attached image: {img.name} ({_format_size(img.stat().st_size)})")
                if not cleaned_ml and not staged_files and not staged_images:
                    continue
                send_user_turn(cleaned_ml)
                continue

            # Check for drag-and-dropped image path(s) BEFORE slash-command dispatch,
            # because macOS absolute paths start with `/`.
            cleaned_line, dropped_imgs = extract_dropped_images(line)
            if dropped_imgs:
                for img in dropped_imgs:
                    staged_images.append(img)
                    print(
                        _style(
                            f"Attached image: {img.name} ({_format_size(img.stat().st_size)})",
                            CYAN,
                            sys.stdout,
                        )
                    )
                if not has_vision:
                    has_vision = _server_supports_vision(client)
                    if not has_vision:
                        print(_style(f"{_vision_missing_message(cfg)}\n", YELLOW, sys.stderr), file=sys.stderr)
                if not cleaned_line.strip():
                    print(
                        _style(
                            f"  [{len(staged_images)} image(s) staged — type your question, drop more images, or press Enter to send]\n",
                            DIM,
                            sys.stdout,
                        )
                    )
                    continue
                send_user_turn(cleaned_line)
                continue

            first_tok = stripped.split(maxsplit=1)[0]
            if stripped.startswith("/") and "/" not in first_tok[1:]:
                parts = stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/exit", "/quit", "/q"):
                    break
                if cmd == "/help":
                    print(
                        "Commands:\n"
                        "  /image <path>               Attach an image (or drag & drop directly into prompt)\n"
                        "  /images [clear]             List or clear staged images\n"
                        "  /lora [scale|on|off|base|inherit|global <scale>]  Adjust abliteration LoRA scale\n"
                        "  /effort [default|none|minimal|low|medium|high|xhigh|max]\n"
                        "  /thinking [on|off]          Toggle reasoning output display\n"
                        "  /stats [on|off]             Toggle tokens/sec stats after each turn\n"
                        "  /temp [float]               Show or set sampling temperature\n"
                        "  /max-tokens [int]           Show or set max tokens (-1 = unlimited)\n"
                        "  /system [text|clear]        Show, set, or clear system prompt\n"
                        "  /read <file>                Attach text file to next message\n"
                        "  /multiline (or \"\"\")        Enter multiline input\n"
                        "  /regen                      Regenerate last assistant reply\n"
                        "  /undo                       Remove last user+assistant exchange\n"
                        "  /clear                      Reset conversation history\n"
                        "  /save <file.md|file.json>   Export transcript\n"
                        "  /status                     Show session and server status\n"
                        "  /exit                       Quit chat\n"
                    )
                    continue
                if cmd == "/image":
                    if not arg:
                        print(_style("Usage: /image <path-to-image>\n", YELLOW, sys.stdout))
                        continue
                    img = parse_image_path_token(arg)
                    if img is None:
                        print(
                            _style(
                                f"Not a valid existing image file: {arg}\n"
                                f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}\n",
                                YELLOW,
                                sys.stdout,
                            )
                        )
                        continue
                    staged_images.append(img)
                    print(
                        _style(
                            f"Staged image: {img.name} ({_format_size(img.stat().st_size)}) [{len(staged_images)} staged]\n",
                            CYAN,
                            sys.stdout,
                        )
                    )
                    if not has_vision:
                        has_vision = _server_supports_vision(client)
                        if not has_vision:
                            print(_style(f"{_vision_missing_message(cfg)}\n", YELLOW, sys.stderr), file=sys.stderr)
                    continue
                if cmd == "/images":
                    if arg.lower() == "clear":
                        staged_images.clear()
                        print(_style("Cleared staged images.\n", DIM, sys.stdout))
                    elif not staged_images:
                        print("No staged images.\n")
                    else:
                        for idx, img in enumerate(staged_images, 1):
                            print(f"  {idx}. {img} ({_format_size(img.stat().st_size)})")
                        print()
                    continue
                if cmd == "/clear":
                    messages.clear()
                    staged_files.clear()
                    staged_images.clear()
                    print(_style("Conversation cleared.\n", DIM, sys.stdout))
                    continue
                if cmd == "/undo":
                    if messages and messages[-1]["role"] == "assistant":
                        messages.pop()
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()
                    print(_style("Removed last turn.\n", DIM, sys.stdout))
                    continue
                if cmd == "/regen":
                    if not messages:
                        print(_style("Nothing to regenerate.\n", YELLOW, sys.stdout))
                        continue
                    if messages[-1]["role"] == "assistant":
                        messages.pop()
                    if not messages or messages[-1]["role"] != "user":
                        print(_style("No user turn to regenerate.\n", YELLOW, sys.stdout))
                        continue
                    send_turn()
                    continue
                if cmd == "/effort":
                    if not arg:
                        print(f"Reasoning effort: {session_effort} (choices: {', '.join(EFFORT_LEVELS)})\n")
                    else:
                        try:
                            session_effort = validate_effort(arg)
                            print(f"Reasoning effort set to {session_effort}.\n")
                        except ConfigError as exc:
                            print(_style(f"{exc}\n", YELLOW, sys.stdout))
                    continue
                if cmd == "/lora":
                    if not arg:
                        srv_scale = _current_server_lora_scale(client)
                        mode = (
                            f"server default ({format_lora_status(srv_scale)})"
                            if session_lora is None
                            else f"session override ({format_lora_status(session_lora)})"
                        )
                        print(f"LoRA: {mode}\n")
                    elif arg.lower() == "inherit":
                        session_lora = None
                        print("LoRA set to inherit server default.\n")
                    elif arg.lower().startswith("global "):
                        val_str = arg.split(maxsplit=1)[1].strip()
                        try:
                            scale_val = parse_lora_scale_arg(val_str)
                            client.set_lora_scale(scale_val)
                            session_lora = None
                            print(f"Server global LoRA scale set to {format_lora_status(scale_val)}.\n")
                        except (ConfigError, ServerClientError) as exc:
                            print(_style(f"{exc}\n", YELLOW, sys.stdout))
                    else:
                        try:
                            session_lora = parse_lora_scale_arg(arg)
                            print(f"Session LoRA scale set to {format_lora_status(session_lora)}.\n")
                        except ConfigError as exc:
                            print(_style(f"{exc}\n", YELLOW, sys.stdout))
                    continue
                if cmd == "/thinking":
                    if not arg:
                        session_thinking = not session_thinking
                    else:
                        session_thinking = arg.lower() in ("on", "true", "1", "yes")
                    print(f"Show thinking: {'on' if session_thinking else 'off'}\n")
                    continue
                if cmd == "/stats":
                    if not arg:
                        session_stats = not session_stats
                    else:
                        session_stats = arg.lower() in ("on", "true", "1", "yes")
                    print(f"Show stats: {'on' if session_stats else 'off'}\n")
                    continue
                if cmd == "/temp":
                    if not arg:
                        print(f"Temperature: {session_temp:g}\n")
                    else:
                        try:
                            session_temp = float(arg)
                            print(f"Temperature set to {session_temp:g}.\n")
                        except ValueError:
                            print(_style(f"Invalid temperature: {arg}\n", YELLOW, sys.stdout))
                    continue
                if cmd == "/max-tokens":
                    if not arg:
                        print(f"Max tokens: {session_max_tokens}\n")
                    else:
                        try:
                            session_max_tokens = int(arg)
                            print(f"Max tokens set to {session_max_tokens}.\n")
                        except ValueError:
                            print(_style(f"Invalid integer: {arg}\n", YELLOW, sys.stdout))
                    continue
                if cmd == "/system":
                    if not arg:
                        print(f"System prompt: {session_system or '(none)'}\n")
                    elif arg.lower() == "clear":
                        session_system = ""
                        print("System prompt cleared.\n")
                    else:
                        session_system = arg
                        print("System prompt updated.\n")
                    continue
                if cmd == "/read":
                    if not arg:
                        print(_style("Usage: /read <file>\n", YELLOW, sys.stdout))
                        continue
                    path = Path(arg).expanduser()
                    if not path.is_file():
                        print(_style(f"File not found: {path}\n", YELLOW, sys.stdout))
                        continue
                    staged_files.append(path)
                    print(f"Staged {path} for next message.\n")
                    continue
                if cmd == "/save":
                    if not arg:
                        print(_style("Usage: /save <file.md|file.json>\n", YELLOW, sys.stdout))
                        continue
                    out_path = Path(arg).expanduser()
                    _save_transcript(out_path, build_messages())
                    print(f"Saved {len(messages)} messages to {out_path}\n")
                    continue
                if cmd == "/status":
                    srv_scale = _current_server_lora_scale(client)
                    eff_lora = srv_scale if session_lora is None else session_lora
                    print(
                        f"Server:   {status.base_url}\n"
                        f"Model:    {cfg.server.alias}\n"
                        f"Vision:   {'enabled' if _server_supports_vision(client) else 'disabled (no mmproj)'}\n"
                        f"LoRA:     {format_lora_status(eff_lora)}"
                        + (" (session override)" if session_lora is not None else " (server global)")
                        + f"\nEffort:   {session_effort}\n"
                        f"Temp:     {session_temp:g} | top_p={session_top_p:g} | min_p={session_min_p:g}\n"
                        f"Thinking: {'on' if session_thinking else 'off'} | Stats: {'on' if session_stats else 'off'}\n"
                        f"Turns:    {len(messages)} messages | Staged images: {len(staged_images)}\n"
                    )
                    continue

                print(_style(f"Unknown command: {cmd}. Type /help for options.\n", YELLOW, sys.stdout))
                continue

            send_user_turn(line)
    finally:
        if readline is not None:
            try:
                readline.write_history_file(str(history_file))
            except OSError:
                pass

    return 0


def _save_transcript(path: Path, messages: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(messages, indent=2) + "\n", encoding="utf-8")
        return

    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        lines.append(f"## {role}\n")
        if msg.get("reasoning_content"):
            lines.append(f"<details><summary>Thinking</summary>\n\n{msg['reasoning_content']}\n\n</details>\n")
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    lines.append("_[attached image]_\n")
                elif isinstance(part, dict) and part.get("type") == "text":
                    lines.append(f"{part.get('text', '')}\n")
        else:
            lines.append(f"{content}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_lora_scale_arg(value: str) -> float:
    lowered = value.strip().lower()
    if lowered in ("on", "abliterated", "uncensored", "enable", "enabled"):
        return 1.0
    if lowered in ("off", "base", "disable", "disabled", "none"):
        return 0.0
    try:
        return float(lowered)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid LoRA scale {value!r}; expected a number (e.g. 1.0, 0.5, 0.0) or on/off/base."
        ) from exc


def build_direct_command(
    cfg: AppConfig,
    *,
    attach: bool = False,
    attach_url: str | None = None,
    prompt: str | None = None,
    images: list[Path] | None = None,
    system: str | None = None,
    effort: str | None = None,
    lora_scale: float | None = None,
    base_only: bool = False,
    mmproj: Path | None = None,
    no_mmproj: bool = False,
    ctx_size: int | None = None,
    gpu_layers: str | None = None,
    flash_attn: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    single_turn: bool = False,
    show_timings: bool = True,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build a direct llama-cli invocation (standalone or attached to llama-server)."""
    cmd: list[str] = [str(cfg.paths.llama_cli)]

    if attach:
        url = attach_url or cfg.server_base_url
        cmd.extend(["--server-base", url])
    else:
        use_ctx = ctx_size if ctx_size is not None else cfg.direct.ctx_size
        use_ngl = gpu_layers if gpu_layers is not None else cfg.direct.gpu_layers
        use_fa = flash_attn if flash_attn is not None else cfg.direct.flash_attn
        use_effort = validate_effort(effort if effort is not None else cfg.generation.reasoning_effort)
        use_temp = temperature if temperature is not None else cfg.generation.temperature
        use_top_p = top_p if top_p is not None else cfg.generation.top_p
        use_top_k = top_k if top_k is not None else cfg.generation.top_k
        use_min_p = min_p if min_p is not None else cfg.generation.min_p
        use_max_tokens = max_tokens if max_tokens is not None else cfg.generation.max_tokens

        cmd.extend(
            [
                "--model",
                str(cfg.paths.model),
                "--ctx-size",
                str(use_ctx),
                "--n-gpu-layers",
                str(use_ngl),
                "--flash-attn",
                str(use_fa),
                "--jinja",
                "--temp",
                f"{use_temp:g}",
                "--top-p",
                f"{use_top_p:g}",
                "--top-k",
                str(use_top_k),
                "--min-p",
                f"{use_min_p:g}",
            ]
        )

        if use_max_tokens is not None and use_max_tokens > 0:
            cmd.extend(["--n-predict", str(use_max_tokens)])
        if seed is not None and seed >= 0:
            cmd.extend(["--seed", str(seed)])

        if use_effort != "default":
            cmd.extend(["--reasoning-effort", use_effort])

        effective_scale = 0.0 if base_only else (cfg.lora.scale if lora_scale is None else float(lora_scale))
        if abs(effective_scale) >= 1e-6:
            cmd.extend(["--lora-scaled", f"{cfg.paths.lora}:{effective_scale:g}"])

        if not no_mmproj:
            use_mmproj = mmproj if mmproj is not None else cfg.paths.mmproj
            if use_mmproj is not None:
                cmd.extend(["--mmproj", str(use_mmproj)])

    use_system = cfg.generation.system_prompt if system is None else system
    if use_system:
        cmd.extend(["--system-prompt", use_system])

    if images:
        cmd.extend(["--image", ",".join(str(p) for p in images)])

    if prompt:
        cmd.extend(["--prompt", prompt])
    if single_turn:
        cmd.append("--single-turn")
    if not show_timings:
        cmd.append("--no-show-timings")

    if not attach and cfg.direct.extra_args:
        cmd.extend(cfg.direct.extra_args)
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_direct(
    cfg: AppConfig,
    *,
    attach: bool = False,
    attach_url: str | None = None,
    allow_duplicate: bool = False,
    prompt: str | None = None,
    images: list[Path] | None = None,
    system: str | None = None,
    effort: str | None = None,
    lora_scale: float | None = None,
    base_only: bool = False,
    mmproj: Path | None = None,
    no_mmproj: bool = False,
    ctx_size: int | None = None,
    gpu_layers: str | None = None,
    flash_attn: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    single_turn: bool = False,
    show_timings: bool = True,
    extra_args: list[str] | None = None,
) -> int:
    """Execute llama-cli directly (either standalone or attached to the background server)."""
    ensure_executable(cfg.paths.llama_cli, "llama-cli binary")

    if not attach:
        ensure_file_exists(cfg.paths.model, "Model GGUF")
        effective_scale = 0.0 if base_only else (cfg.lora.scale if lora_scale is None else float(lora_scale))
        if abs(effective_scale) >= 1e-6:
            ensure_file_exists(cfg.paths.lora, "LoRA adapter GGUF")
        effective_mmproj = None if no_mmproj else (mmproj if mmproj is not None else cfg.paths.mmproj)
        if effective_mmproj is not None:
            ensure_file_exists(effective_mmproj, "Vision projector (mmproj) GGUF")

        if not allow_duplicate:
            status = get_server_status(cfg, timeout=1.0)
            if status.running:
                raise ServerError(
                    f"Background llama-server is already running at {status.base_url} "
                    f"(holding the ~28 GB model in RAM).\n"
                    "Use `inkling chat`, `inkling direct --attach`, or stop the server with `inkling stop`.\n"
                    "Pass `--allow-duplicate` if you intentionally want to load a second copy."
                )

    cmd = build_direct_command(
        cfg,
        attach=attach,
        attach_url=attach_url,
        prompt=prompt,
        images=images,
        system=system,
        effort=effort,
        lora_scale=lora_scale,
        base_only=base_only,
        mmproj=mmproj,
        no_mmproj=no_mmproj,
        ctx_size=ctx_size,
        gpu_layers=gpu_layers,
        flash_attn=flash_attn,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        seed=seed,
        single_turn=single_turn,
        show_timings=show_timings,
        extra_args=extra_args,
    )

    proc = subprocess.run(cmd, check=False)
    return proc.returncode
