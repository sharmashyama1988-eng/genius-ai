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

    def get_system_prompt_for_language(
        self,
        detected: DetectedLanguage,
        formatted_context: str,
    ) -> str:
        """Constructs an optimized, native-sounding system prompt tailored to the detected language."""
        profile = detected.profile

        prompt = (
            f"{profile.system_instruction}\n\n"
            f"=== VERIFIED FACTUAL KNOWLEDGE (WIKIPEDIA) ===\n"
            f"{formatted_context}\n"
            f"===============================================\n\n"
            f"REASONING & THINKING DIRECTIVE:\n"
            f"1. You MUST first perform structured reasoning inside `<think>` and `</think>` tags.\n"
            f"   {profile.thinking_instruction}\n"
            f"2. After `</think>`, deliver your final answer as Genius in {profile.name}.\n"
            f"3. Ensure facts are cited with inline tags [1], [2] matching the source facts above.\n"
            f"4. Speaking style: Fluent, natural, highly intelligent, perfectly adapted to {profile.name}."
        )
        return prompt
