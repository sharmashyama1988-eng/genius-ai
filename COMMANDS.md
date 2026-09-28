# ⚡ Genius AI — Complete Command Reference

> All commands start with `/` — type them directly in the chat terminal.

---

## 🤖 Agentic Mode (Claude Code-Level)

| Command | Description |
|---|---|
| `/agent <task>` | Fully autonomous coding — creates files, runs code, fixes errors automatically |
| `/agent <task>` | Example: `/agent create a snake game in Python using pygame` |
| `/index` | Scan entire codebase — build symbol map, detect imports & entry points |
| `/index <symbol>` | Find any class/function across all files — shows file name + line number |

**How `/agent` works:**
1. Scans your project folder automatically
2. Plans which files to create/edit
3. Writes production-quality code
4. Runs it to verify
5. If error → auto-fixes and runs again
6. Reports what was done

---

## ⚙️ Protocol Commands (`protocol.txt`)

| Command | Description |
|---|---|
| `/run` | List all registered commands from `protocol.txt` |
| `/run dev` | Start Genius AI |
| `/run test` | Run all tests (`pytest`) |
| `/run build` | Build project |
| `/run lint` | Run linter (`flake8`) |
| `/run format` | Auto-format code (`black`) |
| `/run install` | Install dependencies |
| `/run git-sync` | Stage + commit + push everything |

> **`protocol.txt`** is in your project root. Add any custom command:
> ```ini
> [mycommand]
> command = python my_script.py
> description = What this does
> ```

---

## 🔀 Git Operations

| Command | Description |
|---|---|
| `/git` | Full git status report |
| `/git status` | Branch, staged files, ahead/behind remote |
| `/git commit [message]` | Stage all changes + commit |
| `/git push` | Push to origin |
| `/git pull` | Pull from origin |
| `/git branch` | List all local + remote branches |
| `/git branch <name>` | Create new branch and switch to it |
| `/git log [n]` | Show last N commits (default 10) |
| `/git diff` | Show unstaged changes |
| `/git pr [base]` | Generate Pull Request summary vs base branch |
| `/git conflicts` | Show all merge conflict sections |
| `/git init` | Initialize a new git repository |

---

## 🧠 Model & Context Window

### Switch Provider

| Command | Context Window | Notes |
|---|---|---|
| `/model` | — | Show all providers + context sizes |
| `/model local:qwen-0.5b` | 32K | Ultra-light, no GPU, 2GB RAM |
| `/model local:qwen-1.5b` | 128K | Balanced, 4GB RAM |
| `/model local:qwen-3b` | 128K | Strong reasoning, 6GB RAM |
| `/model local:qwen-7b` | **1M** | Recommended, 14GB RAM |
| `/model local:qwen-coder-7b` | **1M** | Best for `/agent` tasks |
| `/model local:qwen-14b` | **1M** | High quality, 28GB RAM |
| `/model claude` | 200K | Claude API (needs `ANTHROPIC_API_KEY`) |
| `/model gemini` | **1M–2M** | Gemini API (needs `GEMINI_API_KEY`) |
| `/model ollama` | varies | Local Ollama |

### Context Window Presets

| Command | Active Tokens | Best For |
|---|---|---|
| `/context small` | 8K | Quick chat |
| `/context medium` | 32K | Default — everyday use |
| `/context large` | 64K | Complex code tasks |
| `/context xl` | 128K | Claude-3 level |
| `/context gemini` | 500K | Large codebases |
| `/context ultra` | 1M | Entire projects |
| `/context infinite` | ∞ Unlimited | Full archive mode |

> `/context` — shows current window stats (active tokens, archive chunks, compression ratio)

---

## 🔍 Research & Search

| Command | Description |
|---|---|
| `/search <query>` | Perplexity-style — Wikipedia + Web + numbered citations |
| `/on` | Research mode **ON** — search every query |
| `/off` | Research mode **OFF** — fast direct mode (no web) |
| `/auto` | Research mode **AUTO** — smart routing (default) |
| `/research` | Show current research mode |
| `/research [auto\|on\|off]` | Change research mode |

---

## 📁 File Operations

| Command | Description |
|---|---|
| `/project <path>` | Change active workspace folder |
| `/files` | List files in active project |
| `/read <file>` | Read and display file contents |
| `/create <file>` | Create a new file (you provide content) |
| `/view <image>` | Inspect image — resolution, format, metadata |

---

## 💻 Code & Math

| Command | Description |
|---|---|
| `/code <problem>` | Generate production-ready code or algorithm |
| `/calc <expression>` | Solve math — equations, formulas, series (AST-based) |
| `/exec <command>` | Run shell command (with safety gate confirmation) |

---

## 🎨 Display & Session

| Command | Description |
|---|---|
| `/think` | Toggle xThinking stream display on/off |
| `/chatview` | Switch between card view and stream view |
| `/chatview html` | Export chat as HTML file |
| `/lang <code>` | Set language: `hi`, `en`, `hi-Latn`, `es`, `fr`, `auto` |
| `/stats` | Session stats — model, memory, turns, timings |
| `/export md` | Export conversation as Markdown |
| `/export json` | Export conversation as JSON |
| `/sessions` | List all past conversation sessions |
| `/new` | Start a fresh conversation session |
| `/clear` | Clear screen + reset working memory |
| `/help` | Show commands list |
| `/exit` | Exit Genius AI |

---

## 🚀 Quick Start Examples

```bash
# Create a full project autonomously
/agent create a REST API in Python using FastAPI with SQLite database

# Game development
/agent build a snake game in Python using pygame

# Add tests automatically
/agent write unit tests for all functions in src/main.py

# Refactor code
/agent refactor the database module to use async/await throughout

# Git workflow
/git commit added new feature
/git push

# Run project commands
/run test
/run build
/run dev

# Switch to high-context model for large codebases
/model local:qwen-7b
/context ultra

# Search with citations
/search how does Python asyncio event loop work

# Index codebase and find a symbol
/index
/index UserManager
```

---

## 🔑 Environment Variables (`.env`)

```env
# Required for Claude API (/model claude)
ANTHROPIC_API_KEY=your_key_here

# Required for Gemini API (/model gemini) — 1M context
GEMINI_API_KEY=your_key_here

# Optional: default model provider on startup
GENIUS_DEFAULT_PROVIDER=claude
```

---

## 📋 `protocol.txt` Format

Create/edit `protocol.txt` in your project root:

```ini
[run]
command = python main.py
description = Run the app

[test]
command = pytest tests/ -v
description = Run all tests

[build]
command = npm run build
description = Build for production

[deploy]
command = git push heroku main
description = Deploy to Heroku
```

Then use: `/run test`, `/run build`, `/run deploy`

---

*Genius AI v2.0 — Built with ❤️ | [GitHub](https://github.com/sharmashyama1988-eng/genius-ai)*
