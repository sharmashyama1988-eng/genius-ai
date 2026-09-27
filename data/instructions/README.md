# 🎨 Genius AI — User Customization & Instruction Hub

Welcome to the **Custom Instructions Directory** for Genius AI!

This folder allows you to customize the personality, tone, behavior, reasoning depth, and operational rules of Genius AI **without writing a single line of Python code**.

---

## 🚀 How It Works

1. Any `.md` (Markdown) file you place in this folder (`data/instructions/`) is **automatically loaded by Genius AI** when reasoning begins.
2. The contents are merged seamlessly into the agent's core cognitive context.
3. You can add, edit, or delete files at any time — changes take effect immediately!

---

## 📁 Suggested Files You Can Create

| File Name | Purpose | Example Use Case |
|---|---|---|
| `custom_persona.md` | Personality & Persona | Make Genius speak like a senior principal architect, a friendly tutor, or a concise terminal assistant. |
| `coding_guidelines.md` | Coding Standards | Enforce PEP 8, TypeScript strict mode, clean architecture, or specific unit testing frameworks. |
| `project_rules.md` | Domain-Specific Knowledge | Add internal project conventions, database schemas, or API endpoint guidelines. |
| `output_style.md` | Output Preferences | Set bullet-point preferences, language tone (Hinglish/English), or formatting constraints. |

---

## 💡 Quick Example

Create a file named `my_preferences.md`:

```markdown
# My Custom Preferences

- Always prioritize code readability and modular architecture.
- For Hindi / Hinglish queries, maintain a natural, friendly, and respectful conversational tone.
- When explaining complex algorithms, provide step-by-step intuition first before showing the final code.
```

That's it! Genius AI will now follow your instructions across every session.
