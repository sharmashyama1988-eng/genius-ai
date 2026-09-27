"""Universal Text Sanitizer & Plain Text Formatter for Genius.

Converts raw LaTeX math, raw markdown syntax, and unrendered formatting tokens
into clean, human-readable, website-style plain text with proper Unicode math.
"""

from __future__ import annotations

import re


class TextSanitizer:
    """Transforms raw markdown and LaTeX into clean, website-like plain text."""

    # Map of ASCII/LaTeX superscripts to Unicode
    _SUPERSCRIPT_MAP = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "n": "ⁿ", "i": "ⁱ", "k": "ᵏ", "x": "ˣ", "y": "ʸ",
    }

    # Map of ASCII subscripts to Unicode
    _SUBSCRIPT_MAP = {
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
        "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
        "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
        "v": "ᵥ", "x": "ₓ",
    }

    @classmethod
    def clean_for_display(cls, text: str) -> str:
        """Sanitizes text removing raw markdown and LaTeX syntax, returning clean readable text."""
        if not text:
            return ""

        # Protect code blocks ```...``` from markdown/math sanitization
        code_blocks: list[str] = []
        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"@@@CODEBLOCK{len(code_blocks)-1}@@@"

        res = re.sub(r"```[\s\S]*?```", _save_code, text)

        # 1. LaTeX math font wrappers: \mathbf{...}, \mathit{...}, \mathrm{...}, \text{...}
        for _ in range(3):
            res = re.sub(r"\\(?:mathbf|mathit|mathrm|text|bm|textbf)\{([^}]*)\}", r"\1", res)

        # 2. Fractions: \frac{a}{b} -> (a) / (b)
        for _ in range(3):
            res = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1) / (\2)", res)

        # 3. Square roots: \sqrt{x} -> √(x)
        for _ in range(3):
            res = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", res)
        res = re.sub(r"\\sqrt\s*([a-zA-Z0-9]+)", r"√\1", res)

        # 4. Standard LaTeX math symbols
        symbol_replacements = [
            (r"\\mathcal\{O\}", "O"),
            (r"\\mathcal\{([A-Za-z])\}", r"\1"),
            (r"[ \t]*\\cdot[ \t]*", " · "),
            (r"[ \t]*\\times[ \t]*", " × "),
            (r"[ \t]*\\div[ \t]*", " ÷ "),
            (r"[ \t]*\\pm[ \t]*", " ± "),
            (r"[ \t]*\\mp[ \t]*", " ∓ "),
            (r"[ \t]*\\approx[ \t]*", " ≈ "),
            (r"[ \t]*\\(?:neq|ne)[ \t]*", " ≠ "),
            (r"[ \t]*\\(?:leq|le)[ \t]*", " ≤ "),
            (r"[ \t]*\\(?:geq|ge)[ \t]*", " ≥ "),
            (r"\\infty", "∞"),
            (r"\\pi", "π"),
            (r"\\theta", "θ"),
            (r"\\Delta|\\delta", "Δ"),
            (r"\\sum(?:_\{[^}]*\})?(?:\^\{[^}]*\})?", "∑"),
            (r"\\int(?:_\{[^}]*\})?(?:\^\{[^}]*\})?", "∫"),
            (r"\\left\(", "("),
            (r"\\right\)", ")"),
            (r"\\left\[", "["),
            (r"\\right\]", "]"),
            (r"\\left\\{", "{"),
            (r"\\right\\}", "}"),
            (r"\\quad|\\qquad", "   "),
            (r"\\circ", "°"),
        ]
        for pattern, repl in symbol_replacements:
            res = re.sub(pattern, repl, res)

        # 5. Convert superscripts: ^2 -> ², ^3 -> ³, ^{n-1} -> ⁿ⁻¹
        def _replace_super(m: re.Match) -> str:
            raw = m.group(1) or m.group(2)
            converted = "".join(cls._SUPERSCRIPT_MAP.get(c, c) for c in raw)
            return converted

        res = re.sub(r"\^\{([0-9a-zA-Z\+\-]+)\}", _replace_super, res)
        res = re.sub(r"\^([0-9a-zA-Z\+\-])", _replace_super, res)

        # 6. Convert subscripts: _n -> ₙ, _{k-1} -> ₖ₋₁
        def _replace_sub(m: re.Match) -> str:
            raw = m.group(1) or m.group(2)
            converted = "".join(cls._SUBSCRIPT_MAP.get(c, c) for c in raw)
            return converted

        res = re.sub(r"_\{([0-9a-zA-Z\+\-]+)\}", _replace_sub, res)
        res = re.sub(r"_([0-9a-zA-Z\+\-])", _replace_sub, res)

        # 7. Strip math enclosing markers: $$, $, \(, \)
        res = re.sub(r"\$\$", "", res)
        res = re.sub(r"\\\((.*?)\\\)", r"\1", res)
        res = re.sub(r"\\\[(.*?)\\\]", r"\1", res)
        # Strip inline $math$ if present
        res = re.sub(r"(?<!\\)\$(.*?)\$", r"\1", res)

        # 8. Strip markdown headers: ### Title -> Title
        lines = []
        for line in res.splitlines():
            # Check for markdown headings
            header_match = re.match(r"^[ \t]*#{1,6}\s*(.*)$", line)
            if header_match:
                cleaned_line = header_match.group(1).strip()
                lines.append(cleaned_line)
            else:
                lines.append(line)
        res = "\n".join(lines)

        # 9. Strip bold / italic formatting tokens: **bold** -> bold, *italic* -> italic
        # But preserve bullet lists like "* item"
        # First handle bold: **text** or __text__
        res = re.sub(r"\*\*([^\*\n]+)\*\*", r"\1", res)
        res = re.sub(r"__([^_\n]+)__", r"\1", res)

        # 10. Normalize bullet points: lines starting with '* ' or '- ' become '• '
        bullet_lines = []
        for line in res.splitlines():
            bullet_match = re.match(r"^([ \t]*)[\*\-]\s+(.*)$", line)
            if bullet_match:
                indent = bullet_match.group(1)
                content = bullet_match.group(2)
                bullet_lines.append(f"{indent}• {content}")
            else:
                bullet_lines.append(line)
        res = "\n".join(bullet_lines)

        # 11. Normalize remaining orphaned single asterisks used for italics
        res = re.sub(r"(?<!\*)\*([a-zA-Z0-9\s,\.\-]{1,50})\*(?!\*)", r"\1", res)

        # 12. Normalize multiple spaces and excess blank lines (max 2 consecutive newlines)
        res = re.sub(r"\n{3,}", "\n\n", res)

        # 13. Restore code blocks
        for idx, block in enumerate(code_blocks):
            res = res.replace(f"@@@CODEBLOCK{idx}@@@", block)

        return res.strip()
