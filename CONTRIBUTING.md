# Contributing to Genius AI

Thank you for your interest in contributing to **Genius AI**! We welcome contributions from developers, researchers, and open-source enthusiasts around the world.

Genius AI is an enterprise-grade autonomous reasoning engine powered by the **Cognitive State Graph (CSG v2.0)** and **xThinking Engine**, engineered for lightning-fast sub-second responses, 100% free model orchestration, and low-resource hardware execution.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Architectural Invariants](#architectural-invariants)
- [Development Setup](#development-setup)
- [Coding Standards & Style Guide](#coding-standards--style-guide)
- [Testing & Benchmarking](#testing--benchmarking)
- [Pull Request Workflow](#pull-request-workflow)
- [Commit Message Conventions](#commit-message-conventions)
- [Reporting Bugs & Security Issues](#reporting-bugs--security-issues)

---

## Code of Conduct

This project and everyone participating in it is governed by the [Genius AI Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior following our reporting procedures.

---

## Architectural Invariants

Every contribution **must** adhere strictly to the following core architectural invariants:

1. **Strictly Free-Tier LLM Hierarchy (Zero Cost Mandate)**
   - All default providers and model cascades must operate exclusively on 100% free models (e.g. `:free` models on OpenRouter, Google AI Studio free tier, or local Ollama instances).
   - Never introduce dependencies on mandatory paid API keys or commercial paywalls.

2. **Strict Resource Efficiency (< 350MB RSS Standard)**
   - The operational steady-state Resident Set Size (RSS) memory must remain strictly **under 350MB** during conversational execution.
   - Zero memory leaks: Proactively trigger generational garbage collection (`gc.collect()`) and OS-level memory trim (`EmptyWorkingSet` on Windows, `malloc_trim` on Linux) after intensive turns.
   - All SQLite databases must operate under WAL mode (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`) with bounded memory caches.

3. **Sub-Second TTFT & Fast-Path Latency Mandate**
   - High-speed conversational turns (greetings, confirmations, smalltalk) must be evaluated via deterministic fast-paths returning responses in **< 3ms**.
   - Remote HTTP requests must use persistent keep-alive client pooling (`HttpClientPool`) to eliminate per-request TCP/TLS handshake overhead.
   - Streaming token emissions to CLI must buffer micro-chunks with a 16ms (60 FPS) flush deadline to prevent Rich console repaint bottlenecking.

4. **Verified Epistemic Grounding**
   - When operating in research mode, statements must satisfy the grounding threshold ($S_{\text{ground}} \ge 0.62$) against retrieved citations (Wikipedia, DuckDuckGo, FTS5 episodic storage).
   - Hallucination pruning must drop claims that fail textual containment or attribution tests.

---

## Development Setup

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Git**: 2.30+
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 12+

### 1. Fork & Clone the Repository
```bash
git clone https://github.com/<your-username>/genius-ai.git
cd genius-ai
```

### 2. Set Up Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and supply at least one free API key:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
OPENROUTER_API_KEY=sk-or-v1-...   # Free tier access to Nemotron & Gemma
GEMINI_API_KEY=AIzaSy...          # Optional Google AI Studio free tier
```

### 5. Verify Installation
Launch the interactive terminal interface:
```bash
python run_genius.py
```

---

## Coding Standards & Style Guide

- **Type Annotations**: Enforce strict typing on all function signatures and return types using `typing` and `from __future__ import annotations`.
- **Async Concurrency**: All I/O-bound operations (HTTP requests, database reads, file crawling) must be asynchronous using `asyncio` and `httpx`.
- **Formatting**:
  - Maximum line length: 120 characters.
  - Follow PEP 8 guidelines.
  - Use clear, descriptive variable names (no single-letter variables except loop counters `i, j`).
- **Error Handling**: Graceful degradation everywhere. Never let an uncaught exception crash the interactive loop; capture errors, log them, and fall back to auxiliary providers or edge local heuristics.
- **Documentation**: All public classes, methods, and functions must have descriptive Google-style docstrings.

---

## Testing & Benchmarking

Before opening a Pull Request, always verify that the codebase compiles cleanly and passes all test suites:

### 1. Bytecode Compilation Check
```bash
python -m py_compile src/**/*.py
```

### 2. Unit & Integration Tests
```bash
pytest tests/ -v
```

### 3. Latency & Resource Benchmark
Run the automated benchmark suite to verify sub-second TTFT and < 350MB RSS memory:
```bash
python tests/benchmark_runtime.py
```

---

## Pull Request Workflow

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```
2. **Make Atomic Commits**: Keep commits focused and logically grouped.
3. **Run Self-Checks**: Ensure all tests and benchmarks pass locally.
4. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**: Submit your PR against the `main` branch of `sharmashyama1988-eng/genius-ai`. Fill out the provided [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(scope)`: A new feature or capability
- `fix(scope)`: A bug fix
- `perf(scope)`: A performance improvement (latency reduction, memory savings)
- `refactor(scope)`: Code changes that neither fix a bug nor add a feature
- `docs(scope)`: Documentation updates or additions
- `test(scope)`: Adding or updating tests
- `ci(scope)`: Changes to CI/CD workflows or build configurations

### Example:
```
feat(runtime): integrate ultra-fast connection pool and fast-path short-circuit
perf(memory): add Windows EmptyWorkingSet and generational GC memory trimmer
fix(search): prevent asyncio event loop blocking during HTML parsing
```

---

## Reporting Bugs & Security Issues

- **Bug Reports**: Open an issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml).
- **Security Vulnerabilities**: For security concerns or vulnerability reports, please review our [Security Policy](SECURITY.md) and report privately rather than opening public issues.
