# Security Policy for Genius AI

Genius AI takes the security and safety of our autonomous agent system, reasoning engine, and community very seriously. We appreciate the responsible disclosure of vulnerabilities.

---

## Supported Versions

Only the latest active minor release receives active security patches.

| Version | Supported          | Status                     |
| ------- | ------------------ | -------------------------- |
| 2.0.x   | :white_check_mark: | Active (Current CSG Core)  |
| 1.x     | :x:                | Deprecated                 |

---

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities via public GitHub issues.**

If you believe you have discovered a vulnerability in Genius AI, please report it privately:

1. **Private Vulnerability Reporting**: Use the [GitHub Security Advisory](https://github.com/sharmashyama1988-eng/genius-ai/security/advisories/new) feature on the repository.
2. **Direct Email**: Alternatively, email the maintainer directly at:
   - `sharma.shyama1988@gmail.com`
   - Include `[SECURITY] Genius AI Vulnerability Report` in the subject line.

### Information to Include in Your Report
To help us triage and resolve the issue quickly, please provide:
- A description of the vulnerability, its potential impact, and severity.
- Step-by-step instructions or minimal reproducible proof-of-concept (PoC) code.
- Environment details: Operating System, Python version, installed dependencies.
- Any suggested mitigations or patches if available.

### What to Expect
- **Initial Response**: We will acknowledge receipt of your report within 48 hours.
- **Triage & Assessment**: We will assess the severity and keep you informed of our progress.
- **Resolution**: Once patched, a security release will be published, and you will be credited in the release notes (unless you prefer anonymity).

---

## Security Best Practices for Users

1. **API Key Safeguards**:
   - Never commit your `.env` file or hardcode API keys into tracked git files.
   - The `.gitignore` in this repository is strictly configured to prevent `.env`, `.env.local`, and SQLite credential stores from leaking.
2. **System Command Execution (`/exec`, `/run`)**:
   - Genius AI possesses tools to execute local terminal commands and file modifications.
   - Always run the agent in a designated workspace directory. Exercise discretion when executing untrusted scripts or shell commands.
3. **Local Database Isolation**:
   - Episodic memory (`genius_memory.db`) and web caches are stored locally on your machine. Ensure proper OS file permissions on your project root.
