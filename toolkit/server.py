"""Persistent background llama-server lifecycle and live status management."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from .client import ServerClient, ServerClientError
from .config import (
    AppConfig,
    ensure_executable,
    ensure_file_exists,
    validate_effort,
)


class ServerError(RuntimeError):
    """Raised when starting, stopping, or querying llama-server fails."""


@dataclass
class ServerStatus:
    running: bool
    healthy: bool
    managed: bool
    pid: int | None
    base_url: str
    api_url: str
    health_detail: str
    started_at: float | None = None
    log_file: Path | None = None
    props: dict[str, Any] | None = None
    lora_adapters: list[dict[str, Any]] | None = None
    state: dict[str, Any] | None = None


def build_server_command(
    cfg: AppConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    alias: str | None = None,
    ctx_size: int | None = None,
    gpu_layers: str | None = None,
    flash_attn: str | None = None,
    parallel: int | None = None,
    cache_ram: int | None = None,
    webui: bool | None = None,
    lora_scale: float | None = None,
    base_only: bool = False,
    no_lora: bool = False,
    mmproj: Path | None = None,
    no_mmproj: bool = False,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the llama-server command line with dynamic LoRA support."""
    use_host = host if host is not None else cfg.server.host
    use_port = port if port is not None else cfg.server.port
    use_alias = alias if alias is not None else cfg.server.alias
    use_ctx = ctx_size if ctx_size is not None else cfg.server.ctx_size
    use_ngl = gpu_layers if gpu_layers is not None else cfg.server.gpu_layers
    use_fa = flash_attn if flash_attn is not None else cfg.server.flash_attn
    use_parallel = parallel if parallel is not None else cfg.server.parallel
    use_cache_ram = cache_ram if cache_ram is not None else cfg.server.cache_ram
    use_webui = cfg.server.webui if webui is None else webui

    use_effort = validate_effort(
        reasoning_effort if reasoning_effort is not None else cfg.generation.reasoning_effort
    )
    use_temp = temperature if temperature is not None else cfg.generation.temperature
    use_top_p = top_p if top_p is not None else cfg.generation.top_p
    use_top_k = top_k if top_k is not None else cfg.generation.top_k
    use_min_p = min_p if min_p is not None else cfg.generation.min_p

    cmd: list[str] = [
        str(cfg.paths.llama_server),
        "--model",
        str(cfg.paths.model),
        "--host",
        str(use_host),
        "--port",
        str(use_port),
        "--alias",
        str(use_alias),
        "--ctx-size",
        str(use_ctx),
        "--n-gpu-layers",
        str(use_ngl),
        "--flash-attn",
        str(use_fa),
        "--parallel",
        str(use_parallel),
        "--cache-ram",
        str(use_cache_ram),
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

    if use_effort != "default":
        cmd.extend(["--reasoning-effort", use_effort])

    if not no_lora:
        effective_scale = 0.0 if base_only else (cfg.lora.scale if lora_scale is None else float(lora_scale))
        if abs(effective_scale) < 1e-6:
            # Preload the LoRA unapplied so `inkling lora on` or per-request scaling works instantly.
            cmd.extend(["--lora", str(cfg.paths.lora), "--lora-init-without-apply"])
        else:
            cmd.extend(["--lora-scaled", f"{cfg.paths.lora}:{effective_scale:g}"])

    if not no_mmproj:
        use_mmproj = mmproj if mmproj is not None else cfg.paths.mmproj
        if use_mmproj is not None:
            cmd.extend(["--mmproj", str(use_mmproj)])

    if cfg.server.api_key:
        cmd.extend(["--api-key", cfg.server.api_key])

    cmd.append("--webui" if use_webui else "--no-webui")

    if cfg.server.extra_args:
        cmd.extend(cfg.server.extra_args)
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def _read_state(state_file: Path) -> dict[str, Any] | None:
    if not state_file.is_file():
        return None
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_state(state_file: Path, payload: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(state_file)


def _remove_state(state_file: Path) -> None:
    try:
        state_file.unlink()
    except FileNotFoundError:
        pass


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_llama_server_pid(pid: int) -> bool:
    """Confirm PID is alive and still corresponds to a llama-server process."""
    if not is_pid_alive(pid):
        return False
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except OSError:
        return True
    cmdline = proc.stdout.strip()
    if not cmdline:
        return False
    return "llama-server" in cmdline


def read_log_tail(log_file: Path, lines: int = 40) -> str:
    if not log_file.is_file():
        return ""
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def get_server_status(cfg: AppConfig, timeout: float = 2.0) -> ServerStatus:
    """Inspect managed PID state plus live HTTP health, props, and LoRA adapters."""
    state = _read_state(cfg.state_file)
    pid: int | None = None
    managed = False
    started_at: float | None = None
    base_url = cfg.server_base_url

    if state and isinstance(state.get("pid"), int):
        candidate_pid = state["pid"]
        if is_llama_server_pid(candidate_pid):
            pid = candidate_pid
            managed = True
            started_at = float(state["started_at"]) if "started_at" in state else None
            if state.get("base_url"):
                base_url = str(state["base_url"])
        else:
            # Stale PID file from an earlier crashed/killed process.
            _remove_state(cfg.state_file)
            state = None

    client = ServerClient(base_url, api_key=cfg.server.api_key)
    healthy, health_detail = client.health(timeout=timeout)

    props: dict[str, Any] | None = None
    loras: list[dict[str, Any]] | None = None
    if healthy:
        try:
            props = client.props(timeout=timeout)
        except ServerClientError:
            pass
        try:
            loras = client.get_lora_adapters(timeout=timeout)
        except ServerClientError:
            pass

    running = managed or healthy or ("loading" in health_detail.lower())
    return ServerStatus(
        running=running,
        healthy=healthy,
        managed=managed,
        pid=pid,
        base_url=base_url,
        api_url=f"{base_url}/v1",
        health_detail=health_detail,
        started_at=started_at,
        log_file=cfg.log_file if cfg.log_file.is_file() else None,
        props=props,
        lora_adapters=loras,
        state=state,
    )


def wait_for_ready(
    cfg: AppConfig,
    *,
    base_url: str | None = None,
    pid: int | None = None,
    timeout: int | None = None,
    quiet: bool = False,
) -> ServerStatus:
    """Poll /health until llama-server finishes loading the model or exits."""
    url = base_url or cfg.server_base_url
    max_wait = cfg.server.startup_timeout if timeout is None else timeout
    client = ServerClient(url, api_key=cfg.server.api_key)
    deadline = time.monotonic() + max_wait
    start = time.monotonic()
    last_note = ""

    while time.monotonic() < deadline:
        if pid is not None and not is_pid_alive(pid):
            tail = read_log_tail(cfg.log_file, lines=30)
            detail = f"\n\nLast lines from {cfg.log_file}:\n{tail}" if tail else ""
            raise ServerError(f"llama-server exited unexpectedly during startup.{detail}")

        healthy, detail = client.health(timeout=1.5)
        if healthy:
            if not quiet and sys.stderr.isatty():
                elapsed = time.monotonic() - start
                sys.stderr.write(f"\rServer ready in {elapsed:.1f}s ({url})\033[K\n")
                sys.stderr.flush()
            return get_server_status(cfg)

        if not quiet and sys.stderr.isatty():
            elapsed = time.monotonic() - start
            note = "loading model..." if "loading" in detail.lower() else "starting..."
            msg = f"\rWaiting for llama-server ({note} {elapsed:.0f}s)\033[K"
            if msg != last_note:
                sys.stderr.write(msg)
                sys.stderr.flush()
                last_note = msg

        time.sleep(0.5)

    tail = read_log_tail(cfg.log_file, lines=25)
    detail = f"\n\nLast lines from {cfg.log_file}:\n{tail}" if tail else ""
    raise ServerError(f"Timed out after {max_wait}s waiting for llama-server at {url}.{detail}")


def start_server(
    cfg: AppConfig,
    *,
    foreground: bool = False,
    wait: bool = True,
    quiet: bool = False,
    host: str | None = None,
    port: int | None = None,
    alias: str | None = None,
    ctx_size: int | None = None,
    gpu_layers: str | None = None,
    flash_attn: str | None = None,
    parallel: int | None = None,
    cache_ram: int | None = None,
    webui: bool | None = None,
    lora_scale: float | None = None,
    base_only: bool = False,
    no_lora: bool = False,
    mmproj: Path | None = None,
    no_mmproj: bool = False,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    extra_args: list[str] | None = None,
) -> ServerStatus | int:
    """Start llama-server in the background (default) or foreground."""
    ensure_executable(cfg.paths.llama_server, "llama-server binary")
    ensure_file_exists(cfg.paths.model, "Model GGUF")
    if not no_lora:
        ensure_file_exists(cfg.paths.lora, "LoRA adapter GGUF")
    effective_mmproj = None if no_mmproj else (mmproj if mmproj is not None else cfg.paths.mmproj)
    if effective_mmproj is not None:
        ensure_file_exists(effective_mmproj, "Vision projector (mmproj) GGUF")

    use_host = host if host is not None else cfg.server.host
    use_port = port if port is not None else cfg.server.port
    connect_host = "127.0.0.1" if use_host in ("0.0.0.0", "::", "") else use_host
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    base_url = f"http://{connect_host}:{use_port}"

    existing = get_server_status(cfg)
    if existing.running:
        if existing.managed and existing.base_url != base_url and not foreground:
            raise ServerError(
                f"A managed llama-server is already running at {existing.base_url} (PID {existing.pid}).\n"
                "Run `inkling stop` before starting on a different host/port, or pass `--foreground`."
            )
        if existing.base_url == base_url:
            if foreground:
                raise ServerError(
                    f"llama-server is already running at {base_url}"
                    + (f" (PID {existing.pid})" if existing.pid else "")
                    + ". Run `inkling stop` first."
                )
            if existing.healthy:
                return existing
            if wait:
                return wait_for_ready(cfg, base_url=base_url, pid=existing.pid, quiet=quiet)
            return existing

    cmd = build_server_command(
        cfg,
        host=host,
        port=port,
        alias=alias,
        ctx_size=ctx_size,
        gpu_layers=gpu_layers,
        flash_attn=flash_attn,
        parallel=parallel,
        cache_ram=cache_ram,
        webui=webui,
        lora_scale=lora_scale,
        base_only=base_only,
        no_lora=no_lora,
        mmproj=mmproj,
        no_mmproj=no_mmproj,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        extra_args=extra_args,
    )

    if foreground:
        if not quiet:
            sys.stderr.write(f"Starting llama-server in foreground at {base_url}\n")
            sys.stderr.flush()
        proc = subprocess.run(cmd, check=False)
        return proc.returncode

    cfg.paths.state_dir.mkdir(parents=True, exist_ok=True)
    log_fh = cfg.log_file.open("wb")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    effective_scale = (
        None
        if no_lora
        else (0.0 if base_only else (cfg.lora.scale if lora_scale is None else float(lora_scale)))
    )
    _write_state(
        cfg.state_file,
        {
            "pid": proc.pid,
            "host": use_host,
            "port": use_port,
            "base_url": base_url,
            "alias": alias or cfg.server.alias,
            "model": str(cfg.paths.model),
            "lora": None if no_lora else str(cfg.paths.lora),
            "mmproj": str(effective_mmproj) if effective_mmproj is not None else None,
            "initial_lora_scale": effective_scale,
            "started_at": time.time(),
            "log_file": str(cfg.log_file),
            "command": cmd,
        },
    )

    if wait:
        try:
            return wait_for_ready(cfg, base_url=base_url, pid=proc.pid, quiet=quiet)
        except Exception:
            if not is_pid_alive(proc.pid):
                _remove_state(cfg.state_file)
            raise

    return get_server_status(cfg)


def stop_server(cfg: AppConfig, timeout: float = 10.0) -> bool:
    """Stop the managed background llama-server. Returns True if a server was stopped."""
    state = _read_state(cfg.state_file)
    if not state or not isinstance(state.get("pid"), int):
        status = get_server_status(cfg)
        if status.running and not status.managed:
            raise ServerError(
                f"A server is responding at {status.base_url}, but it was not started by `inkling start`."
            )
        return False

    pid = state["pid"]
    if not is_llama_server_pid(pid):
        _remove_state(cfg.state_file)
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_state(cfg.state_file)
        return False
    except PermissionError as exc:
        raise ServerError(f"Permission denied stopping llama-server (PID {pid}): {exc}") from exc

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            _remove_state(cfg.state_file)
            return True
        time.sleep(0.15)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise ServerError(f"Permission denied force-stopping llama-server (PID {pid}): {exc}") from exc
    time.sleep(0.2)
    _remove_state(cfg.state_file)
    return True


def ensure_server_running(
    cfg: AppConfig,
    *,
    auto_start: bool | None = None,
    quiet: bool = False,
) -> ServerStatus:
    """Ensure llama-server is reachable and ready, auto-starting if configured."""
    status = get_server_status(cfg)
    if status.healthy:
        return status

    if status.running and not status.healthy:
        return wait_for_ready(cfg, base_url=status.base_url, pid=status.pid, quiet=quiet)

    allow_auto = cfg.server.auto_start if auto_start is None else auto_start
    if not allow_auto:
        raise ServerError(
            f"llama-server is not running at {cfg.server_base_url}.\n"
            "Start it with `inkling start`, pass `--autostart`, or run standalone with `inkling direct`."
        )

    if not quiet:
        sys.stderr.write("No running server found; starting background llama-server...\n")
        sys.stderr.flush()
    result = start_server(cfg, foreground=False, wait=True, quiet=quiet)
    assert isinstance(result, ServerStatus)
    return result
