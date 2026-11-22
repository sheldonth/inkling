# Inkling-2-Mini (Abliterated) Local LLM Harness & Toolkit

A lightweight, self-contained **CLI toolkit**, **persistent local server manager**, **Web UI / OpenAI API launcher**, and **native Rust TUI coding agent** (`inkling code`) built specifically for running **Inkling-2-Mini** with an **abliterated LoRA** applied dynamically on top of it via [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

- **Dynamic LoRA at runtime (`--lora-scaled`)**: Keep a single base GGUF and apply the 43 MB abliteration LoRA dynamically — toggle between abliterated (`1.0`), base (`0.0`), or custom scales live without reloading the 28 GB model or duplicating weights on disk.
- **Persistent hot server**: Keep the model warm in memory so CLI prompts, interactive chats, the coding agent TUI, browser Web UI, and OpenAI-compatible SDKs/tools respond immediately.
- **Fast native Rust TUI coding harness (`inkling code`)**: Codex / Claude-Code-style terminal dev agent built with `ratatui` + `tokio`, wired directly to Inkling-2-Mini's native tool calling, with mode-cycling approvals, colored diffs, live LoRA/effort controls, and automatic context compaction.
- **Zero Python dependencies**: The `inkling` CLI uses only the Python 3.11+ standard library.

---

## Setup

> **Note on model weights:** Large `.gguf` files are excluded from git. Bring your own `Inkling-2-Mini` base GGUF and abliteration LoRA GGUF (and optionally adjust paths in [`inkling.toml`](./inkling.toml)).

```bash
# 1. Clone including the llama.cpp submodule
git clone --recurse-submodules <your-repo-url>
cd local_llm

# 2. Place your GGUF files in the repo root (or edit paths in inkling.toml):
#    - Inkling-2-Mini-Q8_0.gguf
#    - Inkling-2-Mini-Uncensored-LoRA.gguf

# 3. Install dependencies (via Homebrew on macOS) and build llama.cpp + the Rust TUI harness
./setup.sh --install
```

`setup.sh` checks/installs `cmake`, `rust`, `python3` (3.11+), and `ripgrep`, initializes the `llama.cpp` submodule if needed, compiles `llama-server` and `llama-cli` (with Metal on Apple Silicon), compiles `harness/target/release/inkling-code`, and (with `--install`) symlinks `inkling` into `~/.local/bin/inkling` so you can run it from any repository.

---

## Quick Start

```bash
# 1. Start the persistent background server (loads the ~28GB model + abliterated LoRA once)
./inkling start

# 2. Launch the Rust TUI coding agent in any project directory
./inkling code

# 3. Interactive streaming chat (connects instantly to the hot server)
./inkling chat

# 4. Fast one-shot prompts & shell piping
./inkling ask "Summarize the pros and cons of SQLite vs DuckDB"
git diff | ./inkling ask "Review this diff for bugs"
./inkling ask -f README.md "Turn this into 3 bullet points"

# 5. Open the built-in browser Web UI (Llama Chat)
./inkling ui

# 6. Symlink `inkling` onto ~/.local/bin (if you didn't pass --install to setup.sh)
./inkling install
```

> **Auto-start:** By default (`auto_start = true` in `inkling.toml`), running `inkling code`, `inkling chat`, `inkling ask`, or `inkling ui` while the server is stopped will automatically start the background server first. Pass `--no-autostart` to disable that for a single command.

---

## Coding Agent TUI (`inkling code`)

Built in Rust (`ratatui` + `crossterm` + `tokio` + `reqwest`) under [`harness/`](./harness/), `inkling code` provides a full-screen terminal development agent powered natively by Inkling-2-Mini's tool calling.

```bash
# Run in your current working directory (after `inkling install`, works from any project)
inkling code
inkling code -w ~/my-project          # Explicit workspace
inkling code --auto-edit              # Auto-approve reads & file edits, confirm shell
inkling code --yolo                   # Auto-approve all tools
inkling code --plan                   # Read-only analysis & planning
inkling code --base                   # Session override: base model (LoRA scale 0.0)
inkling code --exec "Run tests and summarize failures" --yolo   # Headless non-TUI run
inkling code --rebuild                # Rebuild the Rust release binary
```

### Built-in Tools

| Tool | Class | Capabilities |
|---|---|---|
| `read_file` | Read-only | Line-numbered file reading with `start_line` / `end_line` |
| `list_files` | Read-only | `.gitignore`-aware workspace listing |
| `grep_search` | Read-only | Fast `rg`-backed regex search (pure-Rust fallback) |
| `edit_file` | Write | Exact substring edit with live colored unified diff preview |
| `write_file` | Write | File creation/overwrite with diff preview |
| `run_command` | Shell | Execute shell commands in workspace with timeout & output capture |

### Mode Cycling, Keybinds & Auto-Compaction

Press **`Shift+Tab`** anytime to cycle permission modes:
- **`ASK`** *(default)* — auto-run read/search tools; interactive approval modal (`[y/Enter]` Allow, `[a]` Always allow tool for session, `[e]` Switch to Auto-Edit, `[!]` Yolo, `[n/Esc]` Deny) with colored diff preview before writes or shell commands.
- **`AUTO-EDIT`** — auto-approve reads and file edits; confirm shell commands.
- **`YOLO`** — auto-approve all tools.
- **`PLAN`** — read-only tools only.

Keybinds & slash commands inside `inkling code`:
- **`Enter`**: send | **`Shift+Enter` / `Alt+Enter` / `Ctrl+J` / `\`+Enter**: newline *(for native `Shift+Enter` inside `tmux`, add `set -s extended-keys on` to `~/.tmux.conf`)*
- **`Ctrl+C` / `Esc`**: cancel active turn | **`Ctrl+T`**: expand/collapse thinking blocks | **`Ctrl+O`**: expand/collapse tool outputs | **`Ctrl+X`**: clear staged images
- **`PgUp` / `PgDn` / Mouse wheel**: scroll transcript | **`Ctrl+End`**: jump to bottom
- **Drag-and-drop images**: Drag image files directly onto the TUI (immediately turns into clean staged pills above the prompt without cluttering the text box; auto-converts `.heic`/`.webp`/`.tiff` on macOS via `sips`; requires an `*mmproj*.gguf` loaded on `llama-server`).
- **Auto-compaction**: Automatically strips old thinking/tool outputs & old embedded images, summarizes older turns at ~78% context usage, and recovers/retries automatically if the server ever hits its context limit.
- Slash commands: `/image <path>`, `/images [clear]`, `/mode`, `/lora [on|off|0.6|inherit|global <val>]`, `/effort [none..max]`, `/diff`, `/compact [fast]`, `/clear`, `/save <file.md|file.json>`, `/status`, `/help`, `/exit`

---

## Core CLI & Server Workflows

### 1. Persistent Local Server (`start` / `stop` / `restart` / `status` / `logs`)

```bash
./inkling start                    # Background server with LoRA scale 1.0 (abliterated)
./inkling start --open             # Start and automatically open the browser Web UI
./inkling start --base             # Start with LoRA preloaded at scale 0.0 (toggleable live)
./inkling start -c 65536 -e high   # Override context size and default reasoning effort
./inkling start -f                 # Run in foreground with live llama-server output
./inkling status                   # Show PID, uptime, URLs, context, and live LoRA scale
./inkling logs -f                  # Tail background server logs
./inkling stop                     # Gracefully stop the background server
```

### 2. Dynamic LoRA Control (No Model Reload)

```bash
./inkling lora                     # Inspect active LoRA adapter and scale
./inkling lora off                 # Switch server globally to 0.0 (base Inkling-2-Mini)
./inkling lora on                  # Switch server globally to 1.0 (abliterated)
./inkling lora 0.6                 # Custom partial LoRA strength

# Per-request overrides without changing the server's global scale:
./inkling ask --base "Your prompt"
./inkling ask --lora-scale 0.75 "Your prompt"
```

### 3. Fast One-Shot, Piping, Chat & Drag-and-Drop Images (`ask` / `chat`)

When stdout is piped or redirected, thinking output is omitted from stdout so shell pipelines only receive the final answer.

```bash
./inkling ask "Explain Raft consensus in 5 sentences"
./inkling ask -e none "Give me a bash one-liner to find files > 100MB"
./inkling ask -e high --stats "Prove why sqrt(2) is irrational"
./inkling ask --json "Return a short greeting"
cat error.log | ./inkling ask "What caused this crash?"
./inkling ask -i screenshot.png "What does this error say?"
./inkling chat                     # Streaming REPL with readline history, drag-and-drop images, & slash commands
./inkling chat --native            # Attach native llama-cli TUI to the running server
```

**Image / Vision support in `inkling chat` (and `inkling ask`):**
- Drop Inkling-2-Mini's vision projector (`*mmproj*.gguf`) into the repo root (auto-detected automatically, or set `paths.mmproj` in `inkling.toml` / pass `--mmproj FILE`), then run `./inkling restart`.
- Inside `inkling chat`, **drag and drop any image file** (`.png`, `.jpg`, `.gif`, `.bmp`, `.webp`, `.heic`, `.tiff` — macOS automatically converts `.heic`/`.webp`/`.tiff` via `sips`) straight into the `you> ` prompt:
  - Drop image(s) by themselves and hit Enter to stage them (`[1 img] you> `), then type your question (or press Enter again to send immediately).
  - Or drop an image inline alongside your question on the same line.
  - Use `/image <path>` or `/images [clear]` to manage staged images manually.

### 4. Web UI & OpenAI-Compatible API (`ui` / `env`)

`llama-server` serves both the embedded **Llama Chat** Web UI and an OpenAI-compatible API at `http://127.0.0.1:9931/v1`:

```bash
./inkling ui
eval "$(./inkling env)"
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9931/v1", api_key="local")
resp = client.chat.completions.create(
    model="inkling-abliterated",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"reasoning_effort": "medium"},
)
print(resp.choices[0].message.content)
```

### 5. Direct `llama-cli` Wrapper (`direct`, aliases `cli`, `standalone`)

Run `llama-cli` directly with tuned defaults (no persistent background server):

```bash
./inkling direct                           # Standalone interactive llama-cli (abliterated)
./inkling direct --base                    # Standalone base model (no LoRA)
./inkling direct --lora-scale 0.8 -e low   # Custom LoRA scale and reasoning effort
./inkling direct -1 -p "Hello"             # Single turn and exit
./inkling direct --attach                  # Connect native llama-cli to running server
./inkling direct -- --seed 42              # Forward extra flags directly to llama-cli
```

---

## Configuration & Repository Layout

```text
.
├── inkling             # Executable Python entrypoint (symlink-safe for PATH installs)
├── inkling.toml        # Default model/LoRA paths, server settings, and sampling params
├── setup.sh            # Dependency check/installer + llama.cpp & Rust harness builder
├── toolkit/            # Zero-dependency Python CLI, server manager, and SSE client
├── harness/            # Native Rust TUI coding agent (`inkling-code`)
├── tests/              # Python unit test suite
└── llama.cpp/          # Git submodule (ggml-org/llama.cpp)
```

All defaults in [`inkling.toml`](./inkling.toml) can be overridden via environment variables (`INKLING_HOST`, `INKLING_PORT`, `INKLING_LORA_SCALE`, `INKLING_EFFORT`, `INKLING_CTX_SIZE`, `INKLING_AUTO_START`, `INKLING_API_KEY`, `INKLING_CONFIG`). Runtime state (`server.json`, `server.log`, REPL `history`) is kept inside `.inkling/` (gitignored).

### Running Tests

```bash
cargo test --manifest-path harness/Cargo.toml
python3 -m unittest discover -s tests -v
```
