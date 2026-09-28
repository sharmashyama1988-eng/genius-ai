"""Multilingual manager providing seamless language adaptation and background switching."""

from __future__ import annotations

import logging
from typing import Dict, Optional
from .router import DetectedLanguage, LanguageProfile, LanguageRouter

logger = logging.getLogger(__name__)


class MultilingualManager:
    """Orchestrates dynamic multilingual detection and model configuration."""

    def __init__(self, default_lang: str = "en") -> None:
        self.router = LanguageRouter()
        self.default_lang = default_lang
        self.current_profile: LanguageProfile = self.router.profiles.get(default_lang, self.router.profiles["en"])

    def route_query(self, query: str) -> DetectedLanguage:
        """Analyzes query in real-time and updates active language profile."""
        detected = self.router.detect(query)
        self.current_profile = detected.profile
        logger.info(
            f"Dynamic Language Router: detected '{detected.name}' ({detected.code}) "
            f"with confidence {detected.confidence:.2f}"
        )
        return detected

    def load_user_instructions(self, instructions_dir: str | Path = "data/instructions") -> str:
        """Loads all custom markdown instructions and rules from data/instructions/."""
        from pathlib import Path
        p = Path(instructions_dir)
        if not p.exists() or not p.is_dir():
            return ""

        instruction_blocks = []
        for file in sorted(p.glob("*.md")):
            if file.name.lower() == "readme.md":
                continue
            try:
                content = file.read_text(encoding="utf-8").strip()
                if content:
                    instruction_blocks.append(f"### [Custom Instruction: {file.stem}]\n{content}")
            except Exception as e:
                logger.warning(f"Failed to read custom instruction {file}: {e}")

        if not instruction_blocks:
            return ""

        return (
            "=== USER CUSTOMIZATIONS & OPERATIONAL RULES ===\n"
            + "\n\n".join(instruction_blocks)
            + "\n================================================\n\n"
        )

    def get_system_prompt_for_language(
        self,
        detected: DetectedLanguage,
        formatted_context: str,
        exemplars: Optional[list] = None,
    ) -> str:
        """Constructs an optimized, native-sounding system prompt tailored to the detected language."""
        profile = detected.profile

        exemplar_section = ""
        if exemplars:
            blocks = []
            for idx, ex in enumerate(exemplars, 1):
                instr = getattr(ex, "instruction", "")
                out = getattr(ex, "output", "")
                src = getattr(ex, "source", "exemplar")
                if instr and out:
                    blocks.append(
                        f"--- EXEMPLAR {idx} [{src}] ---\n"
                        f"Query: {instr}\n"
                        f"Demonstrated Reasoning:\n{out[:4500]}\n"
                        f"-----------------------------"
                    )
            if blocks:
                exemplar_section = "\n=== COGNITIVE REASONING EXEMPLARS ===\n" + "\n".join(blocks) + "\n=====================================\n\n"

        user_customizations = self.load_user_instructions()

        prompt = (
            f"{profile.system_instruction}\n\n"
            f"=== VERIFIED FACTUAL KNOWLEDGE ===\n"
            f"{formatted_context}\n"
            f"==================================\n\n"
            f"{exemplar_section}"
            f"{user_customizations}"
            f"REASONING & THINKING DIRECTIVE:\n"
            f"1. You MUST first perform structured reasoning inside `<think>` and `</think>` tags.\n"
            f"   {profile.thinking_instruction}\n"
            f"2. After `</think>`, deliver your final answer as Genius in {profile.name}.\n"
            f"3. Ensure facts are cited with inline tags [1], [2] matching the source facts above.\n"
            f"4. Speaking style: Fluent, natural, highly intelligent, perfectly adapted to {profile.name}.\n"
            f"5. REAL-TIME LIVE KNOWLEDGE: You are equipped with a real-time web search engine. The live facts and market data extracted above are real and fresh. NEVER state 'I cannot browse the internet' or complain about cutoff dates. Always answer directly, authoritatively, and accurately using the live facts provided."
        )
        return prompt
