"""Command-line interface for the Inkling-2-Mini abliterated local LLM toolkit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any
import webbrowser

from . import __version__
from .chat import parse_lora_scale_arg, run_ask, run_chat, run_direct
from .client import ServerClient, ServerClientError
from .config import (
    AppConfig,
    ConfigError,
    EFFORT_LEVELS,
    format_lora_status,
    load_config,
)
from .server import (
    ServerError,
    ServerStatus,
    ensure_server_running,
    get_server_status,
    read_log_tail,
    start_server,
    stop_server,
)


def _format_uptime(started_at: float | None) -> str:
    if not started_at:
        return "unknown"
    seconds = max(0, int(time.time() - started_at))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_status(status: ServerStatus, cfg: AppConfig, *, as_json: bool = False) -> None:
    lora_scale: float | None = None
    lora_path: str | None = None
    if status.lora_adapters:
        try:
            lora_scale = float(status.lora_adapters[0].get("scale", 0.0))
        except (TypeError, ValueError):
            lora_scale = None
        lora_path = status.lora_adapters[0].get("path")

    if as_json:
        payload = {
            "running": status.running,
            "healthy": status.healthy,
            "managed": status.managed,
            "pid": status.pid,
            "base_url": status.base_url,
            "api_url": status.api_url,
            "health_detail": status.health_detail,
            "started_at": status.started_at,
            "log_file": str(status.log_file) if status.log_file else None,
            "lora_scale": lora_scale,
            "lora_path": lora_path,
            "props": status.props,
        }
        print(json.dumps(payload, indent=2))
        return

    state_str = "running (healthy)" if status.healthy else ("starting" if status.running else "stopped")
    print(f"Status:   {state_str}")
    if status.running:
        pid_str = f"{status.pid} (managed)" if status.managed and status.pid else "external"
        print(f"PID:      {pid_str}")
        if status.started_at:
            print(f"Uptime:   {_format_uptime(status.started_at)}")
        print(f"Web UI:   {status.base_url}")
        print(f"API Base: {status.api_url}")
        alias = (status.props or {}).get("model_alias") or cfg.server.alias
        print(f"Model:    {alias}")
        if status.props:
            n_ctx = (status.props.get("default_generation_settings") or {}).get("n_ctx")
            if n_ctx:
                print(f"Context:  {n_ctx} tokens")
            modalities = status.props.get("modalities") or {}
            vision_str = "enabled" if modalities.get("vision") else "disabled (no mmproj)"
            print(f"Vision:   {vision_str}")
        print(f"LoRA:     {format_lora_status(lora_scale)}")
        if status.log_file:
            print(f"Logs:     {status.log_file}")
    else:
        print(f"Target:   {status.base_url}")
        print(f"Model:    {cfg.paths.model.name}")
        print(f"LoRA:     {cfg.paths.lora.name} (default scale {cfg.lora.scale:g})")


def _open_browser(url: str) -> None:
    if sys.platform == "darwin":
        proc = subprocess.run(["open", url], check=False)
        if proc.returncode == 0:
            return
    webbrowser.open(url)


def _cmd_start(cfg: AppConfig, args: argparse.Namespace) -> int:
    lora_scale = parse_lora_scale_arg(args.lora_scale) if args.lora_scale is not None else None
    extra = list(args.extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    mmproj = Path(args.mmproj).expanduser().resolve() if args.mmproj else None
    result = start_server(
        cfg,
        foreground=args.foreground,
        wait=not args.no_wait,
        host=args.host,
        port=args.port,
        alias=args.alias,
        ctx_size=args.ctx_size,
        gpu_layers=args.gpu_layers,
        flash_attn=args.flash_attn,
        parallel=args.parallel,
        webui=False if args.no_webui else None,
        lora_scale=lora_scale,
        base_only=args.base,
        no_lora=args.no_lora,
        mmproj=mmproj,
        no_mmproj=args.no_mmproj,
        reasoning_effort=args.effort,
        temperature=args.temp,
        extra_args=extra,
    )
    if isinstance(result, int):
        return result
    _print_status(result, cfg)
    if args.open_ui and result.healthy:
        _open_browser(result.base_url)
    return 0


def _cmd_stop(cfg: AppConfig, _args: argparse.Namespace) -> int:
    stopped = stop_server(cfg)
    if stopped:
        print("Stopped background llama-server.")
    else:
        print("No managed llama-server was running.")
    return 0


def _cmd_restart(cfg: AppConfig, args: argparse.Namespace) -> int:
    stop_server(cfg)
    return _cmd_start(cfg, args)


def _cmd_status(cfg: AppConfig, args: argparse.Namespace) -> int:
    status = get_server_status(cfg)
    _print_status(status, cfg, as_json=args.json)
    return 0 if status.healthy else 1


def _cmd_logs(cfg: AppConfig, args: argparse.Namespace) -> int:
    if not cfg.log_file.is_file():
        print(f"No server log file found at {cfg.log_file}")
        return 1
    if args.follow:
        proc = subprocess.run(["tail", "-n", str(args.lines), "-f", str(cfg.log_file)], check=False)
        return proc.returncode
    output = read_log_tail(cfg.log_file, lines=args.lines)
    if output:
        print(output)
    return 0


def _cmd_ui(cfg: AppConfig, args: argparse.Namespace) -> int:
    auto_start = False if args.no_autostart else None
    status = ensure_server_running(cfg, auto_start=auto_start)
    print(f"Web UI:   {status.base_url}")
    print(f"API Base: {status.api_url}")
    if not args.no_open:
        _open_browser(status.base_url)
    return 0


def _cmd_env(cfg: AppConfig, args: argparse.Namespace) -> int:
    status = get_server_status(cfg)
    api_url = status.api_url if status.running else cfg.api_base_url
    alias = (status.props or {}).get("model_alias") if status.props else cfg.server.alias
    api_key = cfg.server.api_key or "local"

    if args.json:
        print(
            json.dumps(
                {
                    "OPENAI_BASE_URL": api_url,
                    "OPENAI_API_KEY": api_key,
                    "OPENAI_MODEL": alias,
                    "INKLING_SERVER_URL": status.base_url if status.running else cfg.server_base_url,
                },
                indent=2,
            )
        )
        return 0

    print(f"export OPENAI_BASE_URL={shlex.quote(api_url)}")
    print(f"export OPENAI_API_KEY={shlex.quote(api_key)}")
    print(f"export OPENAI_MODEL={shlex.quote(alias)}")
    return 0


def _cmd_lora(cfg: AppConfig, args: argparse.Namespace) -> int:
    status = get_server_status(cfg)
    if not status.healthy:
        raise ServerError(
            f"Server is not running at {cfg.server_base_url}. Start it with `inkling start` first."
        )
    client = ServerClient(status.base_url, api_key=cfg.server.api_key)

    if args.value is None:
        adapters = client.get_lora_adapters()
        if not adapters:
            print("No LoRA adapters loaded on server.")
            return 0
        for item in adapters:
            scale = float(item.get("scale", 0.0))
            print(f"[{item.get('id', 0)}] {item.get('path', '')}  scale={format_lora_status(scale)}")
        return 0

    scale = parse_lora_scale_arg(args.value)
    adapters = client.set_lora_scale(scale, adapter_id=args.adapter_id)
    if adapters:
        active = float(adapters[0].get("scale", scale))
        print(f"Updated live LoRA scale -> {format_lora_status(active)}")
    else:
        print(f"Updated live LoRA scale -> {format_lora_status(scale)}")
    return 0


def _read_system_prompt(args: argparse.Namespace) -> str | None:
    if getattr(args, "system_file", None):
        path = Path(args.system_file).expanduser()
        if not path.is_file():
            raise ConfigError(f"System prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    return args.system


def _cmd_ask(cfg: AppConfig, args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    files = [Path(p).expanduser() for p in (args.file or [])]
    images = [Path(p).expanduser() for p in (args.image or [])]
    lora_scale = parse_lora_scale_arg(args.lora_scale) if args.lora_scale is not None else None
    show_thinking = True if args.show_thinking else (False if args.hide_thinking else None)
    auto_start = True if args.autostart else (False if args.no_autostart else None)

    return run_ask(
        cfg,
        prompt=prompt,
        files=files,
        images=images,
        system=_read_system_prompt(args),
        effort=args.effort,
        lora_scale=lora_scale,
        base_only=args.base,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stream=not args.no_stream,
        show_thinking=show_thinking,
        show_stats=True if args.stats else None,
        json_output=args.json,
        auto_start=auto_start,
        direct=args.direct,
    )


def _cmd_chat(cfg: AppConfig, args: argparse.Namespace) -> int:
    lora_scale = parse_lora_scale_arg(args.lora_scale) if args.lora_scale is not None else None
    show_thinking = True if args.show_thinking else (False if args.hide_thinking else None)
    auto_start = True if args.autostart else (False if args.no_autostart else None)
    images = [Path(p).expanduser() for p in (args.image or [])]
    initial_prompt = " ".join(args.prompt).strip() if args.prompt else None

    return run_chat(
        cfg,
        system=_read_system_prompt(args),
        images=images,
        effort=args.effort,
        lora_scale=lora_scale,
        base_only=args.base,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        show_thinking=show_thinking,
        show_stats=True if args.stats else None,
        auto_start=auto_start,
        native=args.native,
        initial_prompt=initial_prompt,
    )


def _cmd_direct(cfg: AppConfig, args: argparse.Namespace) -> int:
    lora_scale = parse_lora_scale_arg(args.lora_scale) if args.lora_scale is not None else None
    mmproj = Path(args.mmproj).expanduser().resolve() if args.mmproj else None
    images = [Path(p).expanduser() for p in (args.image or [])]
    extra = list(args.extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    return run_direct(
        cfg,
        attach=args.attach,
        allow_duplicate=args.allow_duplicate,
        prompt=args.prompt,
        images=images,
        system=_read_system_prompt(args),
        effort=args.effort,
        lora_scale=lora_scale,
        base_only=args.base,
        mmproj=mmproj,
        no_mmproj=args.no_mmproj,
        ctx_size=args.ctx_size,
        gpu_layers=args.gpu_layers,
        flash_attn=args.flash_attn,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        single_turn=args.single_turn,
        show_timings=not args.no_timings,
        extra_args=extra,
    )


def _build_harness(cfg: AppConfig) -> None:
    manifest = cfg.paths.harness_dir / "Cargo.toml"
    if not manifest.is_file():
        raise ConfigError(f"Rust harness manifest not found: {manifest}")
    sys.stderr.write("Building Rust TUI harness (`inkling-code`)...\n")
    sys.stderr.flush()
    proc = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(manifest)],
        check=False,
    )
    if proc.returncode != 0:
        raise ConfigError(f"Failed to build Rust harness (cargo exited with {proc.returncode}).")


def build_code_command(cfg: AppConfig, server_url: str, args: argparse.Namespace) -> list[str]:
    mode = args.mode
    if args.yolo:
        mode = "yolo"
    elif args.auto_edit:
        mode = "auto-edit"
    elif args.plan:
        mode = "plan"

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd().resolve()
    effort = args.effort if args.effort is not None else cfg.generation.reasoning_effort

    cmd: list[str] = [
        str(cfg.paths.harness_bin),
        "--server-url",
        server_url,
        "--model",
        cfg.server.alias,
        "--workspace",
        str(workspace),
        "--mode",
        mode,
        "--effort",
        effort,
        "--temp",
        f"{cfg.generation.temperature if args.temp is None else args.temp:g}",
        "--top-p",
        f"{cfg.generation.top_p if args.top_p is None else args.top_p:g}",
        "--top-k",
        str(cfg.generation.top_k if args.top_k is None else args.top_k),
        "--min-p",
        f"{cfg.generation.min_p if args.min_p is None else args.min_p:g}",
        "--max-steps",
        str(args.max_steps),
    ]

    max_tokens = cfg.generation.max_tokens if args.max_tokens is None else args.max_tokens
    if max_tokens > 0:
        cmd.extend(["--max-tokens", str(max_tokens)])

    if cfg.server.api_key:
        cmd.extend(["--api-key", cfg.server.api_key])
    if args.base:
        cmd.append("--base")
    elif args.lora_scale is not None:
        cmd.extend(["--lora-scale", str(parse_lora_scale_arg(args.lora_scale))])
    if args.hide_thinking or (not args.show_thinking and not cfg.generation.show_thinking):
        cmd.append("--hide-thinking")

    sys_prompt = _read_system_prompt(args)
    if sys_prompt is None and cfg.generation.system_prompt:
        sys_prompt = cfg.generation.system_prompt
    if sys_prompt:
        cmd.extend(["--system", sys_prompt])

    if args.exec_prompt:
        cmd.extend(["--exec", args.exec_prompt])
    if args.prompt:
        cmd.extend(args.prompt)

    return cmd


def _cmd_code(cfg: AppConfig, args: argparse.Namespace) -> int:
    if args.rebuild or not cfg.paths.harness_bin.is_file():
        _build_harness(cfg)

    auto_start = True if args.autostart else (False if args.no_autostart else None)
    status = ensure_server_running(cfg, auto_start=auto_start)
    cmd = build_code_command(cfg, status.base_url, args)
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _cmd_config(cfg: AppConfig, args: argparse.Namespace) -> int:
    info: dict[str, Any] = {
        "repo_root": str(cfg.repo_root),
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "paths": {
            "model": {"path": str(cfg.paths.model), "exists": cfg.paths.model.is_file()},
            "lora": {"path": str(cfg.paths.lora), "exists": cfg.paths.lora.is_file()},
            "mmproj": {
                "path": str(cfg.paths.mmproj) if cfg.paths.mmproj else None,
                "exists": bool(cfg.paths.mmproj and cfg.paths.mmproj.is_file()),
            },
            "llama_cli": {"path": str(cfg.paths.llama_cli), "exists": cfg.paths.llama_cli.is_file()},
            "llama_server": {"path": str(cfg.paths.llama_server), "exists": cfg.paths.llama_server.is_file()},
            "harness_bin": {"path": str(cfg.paths.harness_bin), "exists": cfg.paths.harness_bin.is_file()},
            "state_dir": str(cfg.paths.state_dir),
        },
        "lora": {"scale": cfg.lora.scale},
        "server": {
            "host": cfg.server.host,
            "port": cfg.server.port,
            "base_url": cfg.server_base_url,
            "api_url": cfg.api_base_url,
            "alias": cfg.server.alias,
            "ctx_size": cfg.server.ctx_size,
            "gpu_layers": cfg.server.gpu_layers,
            "flash_attn": cfg.server.flash_attn,
            "parallel": cfg.server.parallel,
            "cache_ram": cfg.server.cache_ram,
            "webui": cfg.server.webui,
            "auto_start": cfg.server.auto_start,
        },
        "generation": {
            "reasoning_effort": cfg.generation.reasoning_effort,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "top_k": cfg.generation.top_k,
            "min_p": cfg.generation.min_p,
            "max_tokens": cfg.generation.max_tokens,
            "show_thinking": cfg.generation.show_thinking,
        },
    }
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"Config:       {cfg.config_path or '(defaults)'}")
    for key in ("model", "lora", "llama_cli", "llama_server", "harness_bin"):
        entry = info["paths"][key]
        mark = "ok" if entry["exists"] else "MISSING"
        print(f"{key:13s} [{mark}] {entry['path']}")
    mm_entry = info["paths"]["mmproj"]
    if mm_entry["path"]:
        mark = "ok" if mm_entry["exists"] else "MISSING"
        print(f"{'mmproj':13s} [{mark}] {mm_entry['path']}")
    else:
        print(f"{'mmproj':13s} [none] (optional — add *mmproj*.gguf to enable image vision)")
    print(f"Server URL:   {cfg.server_base_url} (API: {cfg.api_base_url}, alias: {cfg.server.alias})")
    print(f"LoRA scale:   {format_lora_status(cfg.lora.scale)}")
    print(
        f"Defaults:     ctx={cfg.server.ctx_size} effort={cfg.generation.reasoning_effort} "
        f"temp={cfg.generation.temperature:g} auto_start={cfg.server.auto_start}"
    )
    return 0


def _cmd_install(cfg: AppConfig, args: argparse.Namespace) -> int:
    source = (cfg.repo_root / "inkling").resolve()
    if not source.is_file():
        raise ConfigError(f"Entrypoint not found: {source}")

    target_dir = Path(args.dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    link_path = target_dir / args.name

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == source:
            print(f"Already installed: {link_path} -> {source}")
            return 0
        if not args.force:
            raise ConfigError(f"Refusing to overwrite existing {link_path}; pass --force to replace.")
        link_path.unlink()

    link_path.symlink_to(source)
    print(f"Installed symlink: {link_path} -> {source}")

    path_entries = [Path(p).expanduser().resolve() for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if target_dir not in path_entries:
        print(f"\nNote: {target_dir} is not currently on your PATH. Add it to ~/.zshrc:")
        print(f'  export PATH="{target_dir}:$PATH"')
    return 0


def _cmd_uninstall(cfg: AppConfig, args: argparse.Namespace) -> int:
    source = (cfg.repo_root / "inkling").resolve()
    link_path = Path(args.dir).expanduser().resolve() / args.name
    if not link_path.exists() and not link_path.is_symlink():
        print(f"Not installed at {link_path}")
        return 0
    if not link_path.is_symlink() or link_path.resolve() != source:
        raise ConfigError(f"{link_path} does not point to {source}; refusing to remove.")
    link_path.unlink()
    print(f"Removed symlink: {link_path}")
    return 0


def _add_server_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-f", "--foreground", action="store_true", help="Run llama-server in foreground")
    parser.add_argument("--no-wait", action="store_true", help="Return immediately without waiting for model load")
    parser.add_argument("--open", "--ui", dest="open_ui", action="store_true", help="Open Web UI once ready")
    parser.add_argument("--host", help="Bind host (default from inkling.toml)")
    parser.add_argument("--port", type=int, help="Bind port (default from inkling.toml)")
    parser.add_argument("--alias", help="OpenAI API model alias")
    parser.add_argument("-c", "--ctx-size", type=int, help="Context window size (0 = auto-fit)")
    parser.add_argument("-ngl", "--gpu-layers", help="GPU layers ('all', 'auto', or integer)")
    parser.add_argument("-fa", "--flash-attn", choices=("on", "off", "auto"), help="Flash Attention mode")
    parser.add_argument("-np", "--parallel", type=int, help="Number of parallel server slots")
    parser.add_argument("--no-webui", action="store_true", help="Disable built-in Llama Chat Web UI")
    parser.add_argument("--lora-scale", help="Initial LoRA scale (float or on/off/base)")
    parser.add_argument("--base", action="store_true", help="Start with LoRA loaded at scale 0.0 (base model)")
    parser.add_argument("--no-lora", action="store_true", help="Do not load the LoRA adapter at all")
    parser.add_argument("--mmproj", help="Path to vision projector (mmproj) GGUF for image inputs")
    parser.add_argument("--no-mmproj", action="store_true", help="Disable auto-detected mmproj vision projector")
    parser.add_argument("-e", "--effort", choices=EFFORT_LEVELS, help="Default reasoning effort")
    parser.add_argument("--temp", type=float, help="Default sampling temperature")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to llama-server")


def _add_generation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-s", "--system", help="System prompt text")
    parser.add_argument("--system-file", help="Read system prompt from file")
    parser.add_argument("-e", "--effort", choices=EFFORT_LEVELS, help="Reasoning effort level")
    parser.add_argument("--lora-scale", help="Per-request LoRA scale override (float or on/off/base)")
    parser.add_argument("--base", action="store_true", help="Run with base model (LoRA scale 0.0)")
    parser.add_argument("--temp", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, help="Top-k sampling")
    parser.add_argument("--min-p", type=float, help="Min-p sampling")
    parser.add_argument("-n", "--max-tokens", type=int, help="Max tokens to generate")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--show-thinking", action="store_true", help="Show model thinking output")
    think_group.add_argument("--hide-thinking", "--no-thinking", dest="hide_thinking", action="store_true", help="Hide model thinking output")
    parser.add_argument("--stats", action="store_true", help="Print tokens/sec stats to stderr")
    auto_group = parser.add_mutually_exclusive_group()
    auto_group.add_argument("--autostart", action="store_true", help="Auto-start server if stopped")
    auto_group.add_argument("--no-autostart", action="store_true", help="Fail instead of auto-starting server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inkling",
        description="CLI, persistent server, Web UI, and llama-cli toolkit for Inkling-2-Mini (abliterated).",
    )
    parser.add_argument("--version", action="version", version=f"inkling {__version__}")
    parser.add_argument("--config", help="Path to custom inkling.toml")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_start = sub.add_parser("start", aliases=["serve", "up"], help="Start persistent background llama-server")
    _add_server_flags(p_start)
    p_start.set_defaults(func=_cmd_start)

    p_stop = sub.add_parser("stop", aliases=["down"], help="Stop background llama-server")
    p_stop.set_defaults(func=_cmd_stop)

    p_restart = sub.add_parser("restart", help="Restart background llama-server")
    _add_server_flags(p_restart)
    p_restart.set_defaults(func=_cmd_restart)

    p_status = sub.add_parser("status", help="Show server health, URLs, model info, and active LoRA scale")
    p_status.add_argument("--json", action="store_true", help="Output JSON")
    p_status.set_defaults(func=_cmd_status)

    p_logs = sub.add_parser("logs", help="View background llama-server logs")
    p_logs.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to display (default: 50)")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output live")
    p_logs.set_defaults(func=_cmd_logs)

    p_ui = sub.add_parser("ui", aliases=["web"], help="Ensure server is running and open the built-in Web UI")
    p_ui.add_argument("--no-open", action="store_true", help="Print URL without opening browser")
    p_ui.add_argument("--no-autostart", action="store_true", help="Do not auto-start server if stopped")
    p_ui.set_defaults(func=_cmd_ui)

    p_env = sub.add_parser("env", help="Print OPENAI_* environment exports for external clients/SDKs")
    p_env.add_argument("--json", action="store_true", help="Output JSON")
    p_env.set_defaults(func=_cmd_env)

    p_lora = sub.add_parser("lora", help="Inspect or dynamically adjust the abliteration LoRA scale on the running server")
    p_lora.add_argument("value", nargs="?", help="New scale: float (e.g. 1.0, 0.5, 0.0) or on/off/base")
    p_lora.add_argument("--id", dest="adapter_id", type=int, default=0, help="LoRA adapter index (default: 0)")
    p_lora.set_defaults(func=_cmd_lora)

    p_ask = sub.add_parser("ask", aliases=["p", "prompt"], help="Fast one-shot prompt (supports stdin piping, -f files, & -i images)")
    _add_generation_flags(p_ask)
    p_ask.add_argument("-f", "--file", action="append", help="Attach a text file to the prompt (repeatable)")
    p_ask.add_argument("-i", "--image", action="append", help="Attach an image file (repeatable)")
    p_ask.add_argument("--seed", type=int, help="RNG seed")
    p_ask.add_argument("--no-stream", action="store_true", help="Wait for complete response instead of streaming")
    p_ask.add_argument("--json", action="store_true", help="Output structured JSON response")
    p_ask.add_argument("--direct", action="store_true", help="Run standalone llama-cli directly instead of using server")
    p_ask.add_argument("prompt", nargs="*", help="Prompt text (optional if piping stdin or passing -f/-i)")
    p_ask.set_defaults(func=_cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive streaming chat against the persistent server (supports drag-and-drop images)")
    _add_generation_flags(p_chat)
    p_chat.add_argument("-i", "--image", action="append", help="Pre-stage an image file (repeatable)")
    p_chat.add_argument("--native", action="store_true", help="Attach native llama-cli TUI to the running server")
    p_chat.add_argument("prompt", nargs="*", help="Optional initial message")
    p_chat.set_defaults(func=_cmd_chat)

    p_code = sub.add_parser("code", aliases=["agent", "tui"], help="Launch the Rust TUI coding agent harness")
    _add_generation_flags(p_code)
    p_code.add_argument("-w", "--workspace", help="Workspace directory (default: current working directory)")
    p_code.add_argument("-m", "--mode", choices=("ask", "auto-edit", "yolo", "plan"), default="ask", help="Permission mode (default: ask)")
    mode_short = p_code.add_mutually_exclusive_group()
    mode_short.add_argument("--yolo", action="store_true", help="Shortcut for --mode yolo")
    mode_short.add_argument("--auto-edit", action="store_true", help="Shortcut for --mode auto-edit")
    mode_short.add_argument("--plan", action="store_true", help="Shortcut for --mode plan (read-only)")
    p_code.add_argument("--max-steps", type=int, default=30, help="Max tool-calling steps per turn (default: 30)")
    p_code.add_argument("--exec", dest="exec_prompt", help="Run a single headless agent turn without TUI")
    p_code.add_argument("--rebuild", action="store_true", help="Force rebuild of the Rust harness binary")
    p_code.add_argument("prompt", nargs="*", help="Optional initial prompt for the TUI")
    p_code.set_defaults(func=_cmd_code)

    p_direct = sub.add_parser("direct", aliases=["cli", "standalone"], help="Direct llama-cli wrapper (no background server)")
    p_direct.add_argument("--attach", action="store_true", help="Connect native llama-cli to the running server (--server-base)")
    p_direct.add_argument("--allow-duplicate", action="store_true", help="Allow loading a second model copy even if server is running")
    p_direct.add_argument("-p", "--prompt", help="Initial prompt")
    p_direct.add_argument("-s", "--system", help="System prompt text")
    p_direct.add_argument("--system-file", help="Read system prompt from file")
    p_direct.add_argument("-e", "--effort", choices=EFFORT_LEVELS, help="Reasoning effort level")
    p_direct.add_argument("--lora-scale", help="LoRA scale (float or on/off/base)")
    p_direct.add_argument("--base", action="store_true", help="Run base model without LoRA")
    p_direct.add_argument("--mmproj", help="Path to vision projector (mmproj) GGUF")
    p_direct.add_argument("--no-mmproj", action="store_true", help="Disable auto-detected mmproj vision projector")
    p_direct.add_argument("-i", "--image", action="append", help="Attach an image file (repeatable)")
    p_direct.add_argument("-c", "--ctx-size", type=int, help="Context window size")
    p_direct.add_argument("-ngl", "--gpu-layers", help="GPU layers ('all', 'auto', or integer)")
    p_direct.add_argument("-fa", "--flash-attn", choices=("on", "off", "auto"), help="Flash Attention mode")
    p_direct.add_argument("--temp", type=float, help="Sampling temperature")
    p_direct.add_argument("--top-p", type=float, help="Top-p sampling")
    p_direct.add_argument("--top-k", type=int, help="Top-k sampling")
    p_direct.add_argument("--min-p", type=float, help="Min-p sampling")
    p_direct.add_argument("-n", "--max-tokens", type=int, help="Max tokens to generate")
    p_direct.add_argument("--seed", type=int, help="RNG seed")
    p_direct.add_argument("-1", "--single-turn", action="store_true", help="Run a single turn and exit")
    p_direct.add_argument("--no-timings", action="store_true", help="Hide llama-cli timing footer")
    p_direct.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments forwarded to llama-cli (prefix with --)")
    p_direct.set_defaults(func=_cmd_direct)

    p_config = sub.add_parser("config", help="Show resolved configuration and verify model/binary paths")
    p_config.add_argument("--json", action="store_true", help="Output JSON")
    p_config.set_defaults(func=_cmd_config)

    p_install = sub.add_parser("install", help="Symlink `inkling` onto your PATH (default: ~/.local/bin)")
    p_install.add_argument("--dir", default="~/.local/bin", help="Target directory for symlink (default: ~/.local/bin)")
    p_install.add_argument("--name", default="inkling", help="Command name (default: inkling)")
    p_install.add_argument("--force", action="store_true", help="Overwrite existing target")
    p_install.set_defaults(func=_cmd_install)

    p_uninstall = sub.add_parser("uninstall", help="Remove the installed PATH symlink")
    p_uninstall.add_argument("--dir", default="~/.local/bin", help="Target directory (default: ~/.local/bin)")
    p_uninstall.add_argument("--name", default="inkling", help="Command name (default: inkling)")
    p_uninstall.set_defaults(func=_cmd_uninstall)

    return parser


def _print_overview(cfg: AppConfig) -> None:
    status = get_server_status(cfg, timeout=1.0)
    _print_status(status, cfg)
    print(
        "\nQuick commands:\n"
        "  inkling code                    Launch Rust TUI coding agent in current directory\n"
        "  inkling start                   Start persistent server (keeps 28GB model hot)\n"
        "  inkling chat                    Interactive chat against running server\n"
        "  inkling ask \"...\"               Fast one-shot prompt (also accepts piped stdin)\n"
        "  inkling ui                      Open built-in Web UI in your browser\n"
        "  inkling lora [on|off|0.75]      Toggle or scale abliteration LoRA live\n"
        "  inkling direct                  Run llama-cli directly (no background server)\n"
        "  inkling install                 Symlink `inkling` into ~/.local/bin\n"
        "\nRun `inkling --help` or `inkling <command> --help` for full options."
    )


def main(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(root, config_path=args.config)
        if not args.command:
            _print_overview(cfg)
            return 0
        return int(args.func(cfg, args))
    except (ConfigError, ServerError, ServerClientError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
