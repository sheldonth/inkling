"""Configuration loading, path resolution, and environment overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any


EFFORT_LEVELS: tuple[str, ...] = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class ConfigError(RuntimeError):
    """Raised when configuration or required files are invalid."""


@dataclass
class PathsConfig:
    model: Path
    lora: Path
    mmproj: Path | None
    llama_cli: Path
    llama_server: Path
    harness_dir: Path
    harness_bin: Path
    state_dir: Path


@dataclass
class LoraConfig:
    scale: float = 1.0


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9931
    alias: str = "inkling-abliterated"
    ctx_size: int = 32768
    gpu_layers: str = "all"
    flash_attn: str = "auto"
    parallel: int = 1
    cache_ram: int = 8192
    webui: bool = True
    auto_start: bool = True
    startup_timeout: int = 180
    api_key: str = ""
    extra_args: list[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    reasoning_effort: str = "medium"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    max_tokens: int = -1
    system_prompt: str = ""
    show_thinking: bool = True
    show_stats: bool = False


@dataclass
class DirectConfig:
    ctx_size: int = 32768
    gpu_layers: str = "all"
    flash_attn: str = "auto"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    repo_root: Path
    config_path: Path | None
    paths: PathsConfig
    lora: LoraConfig
    server: ServerConfig
    generation: GenerationConfig
    direct: DirectConfig

    @property
    def client_host(self) -> str:
        """Return a connectable localhost address even when bound to 0.0.0.0 or ::."""
        if self.server.host in ("0.0.0.0", "::", ""):
            return "127.0.0.1"
        return self.server.host

    @property
    def server_base_url(self) -> str:
        host = self.client_host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.server.port}"

    @property
    def api_base_url(self) -> str:
        return f"{self.server_base_url}/v1"

    @property
    def state_file(self) -> Path:
        return self.paths.state_dir / "server.json"

    @property
    def log_file(self) -> Path:
        return self.paths.state_dir / "server.log"

    @property
    def history_file(self) -> Path:
        return self.paths.state_dir / "history"


def _resolve_path(raw: str | Path, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _detect_mmproj(base_dir: Path) -> Path | None:
    """Auto-detect an mmproj GGUF file in the repository root if present."""
    if not base_dir.is_dir():
        return None
    candidates = sorted(
        p for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower()
    )
    return candidates[0].resolve() if candidates else None


def _resolve_optional_path(raw: str | Path | None, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return _resolve_path(text, base_dir)


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"Invalid boolean value: {raw!r}")


def validate_effort(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in EFFORT_LEVELS:
        allowed = ", ".join(EFFORT_LEVELS)
        raise ConfigError(f"Unknown reasoning effort {value!r}; expected one of: {allowed}")
    return normalized


def load_config(
    repo_root: Path,
    config_path: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> AppConfig:
    """Load config from TOML file (if present) and apply environment overrides."""
    environ = os.environ if env is None else env
    repo_root = Path(repo_root).resolve()

    explicit_cfg = config_path or environ.get("INKLING_CONFIG")
    if explicit_cfg:
        cfg_file = Path(explicit_cfg).expanduser().resolve()
        if not cfg_file.is_file():
            raise ConfigError(f"Config file not found: {cfg_file}")
    else:
        default_cfg = repo_root / "inkling.toml"
        cfg_file = default_cfg if default_cfg.is_file() else None

    raw: dict[str, Any] = {}
    if cfg_file is not None:
        try:
            with cfg_file.open("rb") as fh:
                raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Failed to parse {cfg_file}: {exc}") from exc

    base_dir = cfg_file.parent if cfg_file is not None else repo_root

    paths_tbl = raw.get("paths", {})
    lora_tbl = raw.get("lora", {})
    server_tbl = raw.get("server", {})
    gen_tbl = raw.get("generation", {})
    direct_tbl = raw.get("direct", {})

    mmproj_raw = environ.get("INKLING_MMPROJ")
    if mmproj_raw is None:
        mmproj_raw = paths_tbl.get("mmproj", "")
    mmproj_path = _resolve_optional_path(mmproj_raw, base_dir)
    if mmproj_path is None:
        mmproj_path = _detect_mmproj(base_dir)

    paths = PathsConfig(
        model=_resolve_path(
            environ.get("INKLING_MODEL") or paths_tbl.get("model", "Inkling-2-Mini-Q8_0.gguf"),
            base_dir,
        ),
        lora=_resolve_path(
            environ.get("INKLING_LORA") or paths_tbl.get("lora", "Inkling-2-Mini-Uncensored-LoRA.gguf"),
            base_dir,
        ),
        mmproj=mmproj_path,
        llama_cli=_resolve_path(
            environ.get("INKLING_LLAMA_CLI") or paths_tbl.get("llama_cli", "llama.cpp/build/bin/llama-cli"),
            base_dir,
        ),
        llama_server=_resolve_path(
            environ.get("INKLING_LLAMA_SERVER") or paths_tbl.get("llama_server", "llama.cpp/build/bin/llama-server"),
            base_dir,
        ),
        harness_dir=_resolve_path(
            environ.get("INKLING_HARNESS_DIR") or paths_tbl.get("harness_dir", "harness"),
            base_dir,
        ),
        harness_bin=_resolve_path(
            environ.get("INKLING_HARNESS_BIN") or paths_tbl.get("harness_bin", "harness/target/release/inkling-code"),
            base_dir,
        ),
        state_dir=_resolve_path(
            environ.get("INKLING_STATE_DIR") or paths_tbl.get("state_dir", ".inkling"),
            base_dir,
        ),
    )

    lora_scale_raw = environ.get("INKLING_LORA_SCALE")
    lora_scale = float(lora_scale_raw) if lora_scale_raw is not None else float(lora_tbl.get("scale", 1.0))
    lora = LoraConfig(scale=lora_scale)

    host = environ.get("INKLING_HOST") or str(server_tbl.get("host", "127.0.0.1"))
    port = int(environ["INKLING_PORT"]) if "INKLING_PORT" in environ else int(server_tbl.get("port", 9931))
    ctx_size = (
        int(environ["INKLING_CTX_SIZE"])
        if "INKLING_CTX_SIZE" in environ
        else int(server_tbl.get("ctx_size", 32768))
    )
    auto_start = (
        _parse_bool(environ["INKLING_AUTO_START"])
        if "INKLING_AUTO_START" in environ
        else bool(server_tbl.get("auto_start", True))
    )
    server = ServerConfig(
        host=host,
        port=port,
        alias=environ.get("INKLING_ALIAS") or str(server_tbl.get("alias", "inkling-abliterated")),
        ctx_size=ctx_size,
        gpu_layers=str(server_tbl.get("gpu_layers", "all")),
        flash_attn=str(server_tbl.get("flash_attn", "auto")),
        parallel=int(server_tbl.get("parallel", 1)),
        cache_ram=int(server_tbl.get("cache_ram", 8192)),
        webui=bool(server_tbl.get("webui", True)),
        auto_start=auto_start,
        startup_timeout=int(server_tbl.get("startup_timeout", 180)),
        api_key=environ.get("INKLING_API_KEY") or str(server_tbl.get("api_key", "")),
        extra_args=[str(x) for x in server_tbl.get("extra_args", [])],
    )

    effort_raw = environ.get("INKLING_EFFORT") or str(gen_tbl.get("reasoning_effort", "medium"))
    generation = GenerationConfig(
        reasoning_effort=validate_effort(effort_raw),
        temperature=float(gen_tbl.get("temperature", 1.0)),
        top_p=float(gen_tbl.get("top_p", 0.95)),
        top_k=int(gen_tbl.get("top_k", 40)),
        min_p=float(gen_tbl.get("min_p", 0.05)),
        max_tokens=int(gen_tbl.get("max_tokens", -1)),
        system_prompt=str(gen_tbl.get("system_prompt", "")),
        show_thinking=bool(gen_tbl.get("show_thinking", True)),
        show_stats=bool(gen_tbl.get("show_stats", False)),
    )

    direct = DirectConfig(
        ctx_size=int(direct_tbl.get("ctx_size", server.ctx_size)),
        gpu_layers=str(direct_tbl.get("gpu_layers", server.gpu_layers)),
        flash_attn=str(direct_tbl.get("flash_attn", server.flash_attn)),
        extra_args=[str(x) for x in direct_tbl.get("extra_args", [])],
    )

    return AppConfig(
        repo_root=repo_root,
        config_path=cfg_file,
        paths=paths,
        lora=lora,
        server=server,
        generation=generation,
        direct=direct,
    )


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigError(f"{label} not found: {path}")


def ensure_executable(path: Path, label: str) -> None:
    ensure_file_exists(path, label)
    if not os.access(path, os.X_OK):
        raise ConfigError(f"{label} is not executable: {path}")


def format_lora_status(scale: float | None) -> str:
    if scale is None:
        return "none"
    if abs(scale) < 1e-6:
        return "0.0 (base)"
    if abs(scale - 1.0) < 1e-6:
        return "1.0 (abliterated)"
    return f"{scale:g} (custom)"
