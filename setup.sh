#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

INSTALL_PATH=0
REBUILD=0
SKIP_LLAMA=0
SKIP_RUST=0
NO_BREW=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [OPTIONS]

Bootstrap dependencies, build llama.cpp (llama-server & llama-cli), and build
the Rust TUI coding harness (inkling-code) for Inkling-2-Mini + abliterated LoRA.

Options:
  --install       Symlink `inkling` into ~/.local/bin after building
  --rebuild       Force rebuild of llama.cpp and the Rust harness
  --skip-llama    Skip building llama.cpp binaries
  --skip-rust     Skip building the Rust TUI harness
  --no-brew       Do not auto-install missing tools with Homebrew
  -h, --help      Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)    INSTALL_PATH=1; shift ;;
    --rebuild)    REBUILD=1; shift ;;
    --skip-llama) SKIP_LLAMA=1; shift ;;
    --skip-rust)  SKIP_RUST=1; shift ;;
    --no-brew)    NO_BREW=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

ensure_brew_pkg() {
  local cmd="$1"
  local pkg="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$NO_BREW" -eq 0 && "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    log "Installing $pkg via Homebrew..."
    brew install "$pkg"
  else
    err "Missing required tool '$cmd' (install '$pkg' first, or run without --no-brew on macOS)."
    exit 1
  fi
}

log "Checking prerequisites..."

ensure_brew_pkg git git
ensure_brew_pkg python3 python

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  err "Python 3.11+ is required (for stdlib tomllib). Found: $(python3 --version)"
  exit 1
fi

if [[ "$SKIP_LLAMA" -eq 0 ]]; then
  ensure_brew_pkg cmake cmake
fi

if [[ "$SKIP_RUST" -eq 0 ]]; then
  ensure_brew_pkg cargo rust
fi

if ! command -v rg >/dev/null 2>&1; then
  if [[ "$NO_BREW" -eq 0 && "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    log "Installing ripgrep (optional, speeds up grep_search)..."
    brew install ripgrep || warn "Failed to install ripgrep; harness will use its built-in Rust search fallback."
  else
    warn "ripgrep (rg) not found; harness will use its built-in Rust search fallback."
  fi
fi

# 1. Initialize llama.cpp submodule if needed
if [[ "$SKIP_LLAMA" -eq 0 ]]; then
  if [[ ! -f "$REPO_ROOT/llama.cpp/CMakeLists.txt" ]]; then
    log "Initializing llama.cpp git submodule..."
    git submodule update --init --recursive llama.cpp
  fi

  LLAMA_SERVER="$REPO_ROOT/llama.cpp/build/bin/llama-server"
  LLAMA_CLI="$REPO_ROOT/llama.cpp/build/bin/llama-cli"

  if [[ "$REBUILD" -eq 1 || ! -x "$LLAMA_SERVER" || ! -x "$LLAMA_CLI" ]]; then
    log "Building llama-server and llama-cli..."
    CMAKE_ARGS=(
      -S "$REPO_ROOT/llama.cpp"
      -B "$REPO_ROOT/llama.cpp/build"
      -DCMAKE_BUILD_TYPE=Release
    )
    if [[ "$(uname -s)" == "Darwin" ]]; then
      CMAKE_ARGS+=(-DGGML_METAL=ON)
    fi
    cmake "${CMAKE_ARGS[@]}"
    NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
    cmake --build "$REPO_ROOT/llama.cpp/build" --target llama-server llama-cli -j "$NCPU"
  else
    log "llama.cpp binaries already built (pass --rebuild to recompile)."
  fi
fi

# 2. Build the Rust TUI coding harness
if [[ "$SKIP_RUST" -eq 0 ]]; then
  HARNESS_BIN="$REPO_ROOT/harness/target/release/inkling-code"
  if [[ "$REBUILD" -eq 1 || ! -x "$HARNESS_BIN" ]]; then
    log "Building Rust TUI coding harness (harness/target/release/inkling-code)..."
    cargo build --release --manifest-path "$REPO_ROOT/harness/Cargo.toml"
  else
    log "Rust TUI harness already built (pass --rebuild to recompile)."
  fi
fi

chmod +x "$REPO_ROOT/inkling" "$REPO_ROOT/setup.sh"

# 3. Optional PATH install
if [[ "$INSTALL_PATH" -eq 1 ]]; then
  log "Installing symlink into ~/.local/bin/inkling..."
  "$REPO_ROOT/inkling" install
fi

# 4. Verify configuration and model/LoRA presence
log "Checking resolved configuration..."
"$REPO_ROOT/inkling" config

MODEL_FILE="$REPO_ROOT/Inkling-2-Mini-Q8_0.gguf"
LORA_FILE="$REPO_ROOT/Inkling-2-Mini-Uncensored-LoRA.gguf"

if [[ ! -f "$MODEL_FILE" || ! -f "$LORA_FILE" ]]; then
  echo ""
  warn "Model or LoRA GGUF file is missing (GGUF weights are excluded from git)."
  echo "Place your files in $REPO_ROOT (or update paths in inkling.toml):"
  echo "  - Inkling-2-Mini-Q8_0.gguf"
  echo "  - Inkling-2-Mini-Uncensored-LoRA.gguf"
fi

echo ""
log "Setup complete! Next steps:"
echo "  ./inkling start         # Start persistent llama-server with abliterated LoRA"
echo "  ./inkling code          # Launch the Rust TUI coding agent in the current directory"
echo "  ./inkling ui            # Open the built-in browser Web UI"
if [[ "$INSTALL_PATH" -eq 0 ]]; then
  echo "  ./inkling install       # Optional: symlink 'inkling' into ~/.local/bin"
fi
