"""Multilingual dynamic routing and specialized language engine."""

from .router import LanguageRouter, LanguageProfile, DetectedLanguage
from .manager import MultilingualManager

__all__ = ["LanguageRouter", "LanguageProfile", "DetectedLanguage", "MultilingualManager"]
