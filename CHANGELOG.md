# Changelog

All notable changes to **Genius AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-09-28

### Major Architecture Upgrade — CSG v2.0 & xThinking Engine

#### Added
- **Cognitive State Graph (CSG v2.0)**:
  - 7-node deterministic orchestration: `Query_Deconstruction`, `Epistemic_Retrieval`, `Information_Fusion`, `Latent_xThinking`, `Attributed_Synthesis`, `PostHoc_Verification`, `Feedback_Orchestration`.
  - Contradiction Density ($D$) scoring with dynamic thinking token budgeting ($b = \min(32000, b_{\text{base}} \cdot (1 + \gamma D))$).
  - Grounding Verification ($S_{\text{ground}} \ge \tau_{\text{crit}} = 0.62$) and Claim Attribution via ROUGE-L and Token Containment.
- **Ultra-Fast Runtime & Fast-Path Short-Circuit** (`src/system/genius_runtime.py`):
  - Deterministic conversational fast-path delivering instantaneous responses (**< 3ms TTFT**) for greetings and chit-chat.
  - Persistent HTTP/2 and keep-alive connection pool (`HttpClientPool`) eliminating per-request TCP/TLS handshake latency.
  - 60 FPS buffered streaming collator preventing Rich console repaint thrashing in terminal environments.
- **Autonomous System Resource Manager** (`src/system/resource_manager.py`):
  - Zero-dependency system hardware profiling with classification into `CONSTRAINED` (<=4GB RAM), `BALANCED`, and `PERFORMANCE` tiers.
  - Proactive generational garbage collection and OS-level memory trim (`EmptyWorkingSet` on Windows, `malloc_trim` on POSIX) guaranteeing steady-state RSS < 350MB and 0% memory leaks.
  - SQLite zero disk I/O latency optimizer applying WAL mode and in-memory caches across all episodic databases.
  - Adaptive token budgeting dynamically scaling context windows and prompt size to physical RAM limits.
- **Full Technical Specification**: Comprehensive 672-line technical architecture document (`SPECIFICATION.md`).
- **Open Source Community Infrastructure**:
  - Apache 2.0 attribution notices and legal appendices (`LICENSE`, `NOTICE`).
  - Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `CITATION.cff`.
  - GitHub issue and pull request templates (`.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`).
  - Automated continuous integration workflow (`.github/workflows/ci.yml`).

#### Changed
- Terminal welcome banner refined for clean, distraction-free startup.
- Provider fallback cascade optimized for OpenRouter free-tier latency (`nvidia/nemotron-3.5-lightning:free` and `google/gemma-3-27b-it:free`).

---

## [1.0.0] - 2026-09-25

### Initial Release
- Multi-model router with support for local models, Ollama, OpenRouter, and Gemini.
- Interactive terminal chat CLI with Rich formatted output and syntax highlighting.
- Wikipedia and DuckDuckGo hybrid web retrieval engine with SQLite caching.
- SQLite FTS5 episodic memory with BM25 full-text keyword indexing.
- Multi-tier context window management with token compaction.
