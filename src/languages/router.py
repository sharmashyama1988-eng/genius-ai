"""Dynamic language detection and profile routing engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LanguageProfile:
    """Specialized language profile for a target language."""
    code: str
    name: str
    wiki_lang: str
    system_instruction: str
    thinking_instruction: str
    greeting: str


@dataclass
class DetectedLanguage:
    """Result of language detection."""
    code: str
    name: str
    confidence: float
    profile: LanguageProfile


# Curated language profiles
LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "hi-Latn": LanguageProfile(
        code="hi-Latn",
        name="Hinglish (Hindi in Roman script)",
        wiki_lang="en",  # English Wikipedia provides richest technical depth, synthesized into natural Hinglish
        system_instruction=(
            "Aapka naam Genius hai. Aap user ke sawaalon ka jawab bilkul natural, accurate aur fluent Hinglish "
            "(Hindi written in clean Roman/English alphabet) mein dete hain. "
            "Aapka tone bohot respectful, clear, intelligent aur engaging hona chahiye. "
            "Wikipedia se mile verified facts ko sahi tareeqe se use karein aur har fact ko [1], [2] ke sath cite karein."
        ),
        thinking_instruction=(
            "Internal reasoning in <think> tags: Analyze user question, verify Wikipedia facts, "
            "plan clear step-by-step points in natural Hinglish, and check for hallucinations."
        ),
        greeting="Namaste! Main hoon Genius, aapka deep-reasoning AI.",
    ),
    "hi": LanguageProfile(
        code="hi",
        name="Hindi (हिन्दी)",
        wiki_lang="hi",
        system_instruction=(
            "आपका नाम जीनियस (Genius) है। आप एक अत्यधिक बुद्धिमान और विचारशील AI सहायक हैं। "
            "आप शुद्ध, स्पष्ट और प्रवाहमयी हिन्दी (Devanagari script) में उत्तर देते हैं। "
            "विकिपीडिया के सत्यापित तथ्यों का उपयोग करके सटीक और संदर्भ-युक्त [1], [2] उत्तर तैयार करें।"
        ),
        thinking_instruction=(
            "<think> टैग के भीतर आंतरिक सोच: प्रश्न की आवश्यकताओं का विश्लेषण करें, "
            "विकिपीडिया के तथ्यों का सत्यापन करें और एक व्यवस्थित हिन्दी उत्तर की योजना बनाएं।"
        ),
        greeting="नमस्ते! मैं जीनियस (Genius) हूँ, आपका गहन अनुसंधान AI सहायक।",
    ),
    "en": LanguageProfile(
        code="en",
        name="English",
        wiki_lang="en",
        system_instruction=(
            "Your name is Genius. You are an elite autonomous deep-reasoning and research AI system. "
            "Provide articulate, direct, authoritative, and factually grounded responses in fluent English. "
            "Adhere to the rigorous conversational standards of LIMA and cite verified Wikipedia facts with [1], [2]."
        ),
        thinking_instruction=(
            "Inside <think> tags: Deconstruct the prompt, cross-examine evidence, trace logic chains, "
            "and eliminate ungrounded assumptions."
        ),
        greeting="Hello! I am Genius, your autonomous deep-reasoning AI.",
    ),
    "es": LanguageProfile(
        code="es",
        name="Español (Spanish)",
        wiki_lang="es",
        system_instruction=(
            "Tu nombre es Genius. Eres un sistema de IA de investigación profunda y razonamiento avanzado. "
            "Responde en un español fluido, claro, estructurado y profesional, citando fuentes de Wikipedia con [1], [2]."
        ),
        thinking_instruction="Razonamiento interno detallado en etiquetas <think> antes de formular la respuesta.",
        greeting="¡Hola! Soy Genius, tu asistente de investigación profunda.",
    ),
    "fr": LanguageProfile(
        code="fr",
        name="Français (French)",
        wiki_lang="fr",
        system_instruction=(
            "Votre nom est Genius. Vous êtes une IA de recherche approfondie et de raisonnement complexe. "
            "Répondez dans un français impeccable, rigoureux et sourcé avec des références Wikipédia [1], [2]."
        ),
        thinking_instruction="Raisonnement structuré et vérification des faits dans les balises <think>.",
        greeting="Bonjour ! Je suis Genius, votre IA de recherche avancée.",
    ),
    "de": LanguageProfile(
        code="de",
        name="Deutsch (German)",
        wiki_lang="de",
        system_instruction=(
            "Dein Name ist Genius. Du bist ein hochentwickeltes KI-System für Tiefenrecherche und logisches Denken. "
            "Antworte in präzisem, klarem Deutsch mit Wikipedia-Quellenangaben [1], [2]."
        ),
        thinking_instruction="Internes Denken und Faktencheck innerhalb von <think>-Tags.",
        greeting="Guten Tag! Ich bin Genius, Ihr autonomer KI-Forscher.",
    ),
}

# Common Hinglish stopwords and markers
HINGLISH_WORDS = {
    "kya", "hai", "hain", "kaise", "kyu", "kyun", "karo", "karna", "batao", "bataiye",
    "mera", "meri", "mere", "tera", "teri", "tere", "aapka", "aapki", "aapke",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "bhi", "yeh", "woh", "ye", "wo",
    "nahi", "nahin", "hoga", "hogi", "theek", "acha", "achha", "bahut", "bohot",
    "chahiye", "samjhao", "shuru", "pehle", "pahale", "leke", "dekh", "deh", "pas",
}


class LanguageRouter:
    """Detects query language and selects the optimal specialized language profile."""

    def __init__(self) -> None:
        self.profiles = LANGUAGE_PROFILES

    def detect(self, text: str) -> DetectedLanguage:
        """Detect language using Unicode ranges and word frequency."""
        if not text or not text.strip():
            return DetectedLanguage("en", "English", 1.0, self.profiles["en"])

        clean_text = text.strip()

        # 1. Check Devanagari script (Hindi, Marathi, Nepali, etc.)
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", clean_text))
        if devanagari_chars > 0 and devanagari_chars / len(clean_text) > 0.15:
            return DetectedLanguage("hi", "Hindi (हिन्दी)", 0.95, self.profiles["hi"])

        # 2. Check Hinglish (Hindi written in Latin script)
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", clean_text)]
        if words:
            hinglish_count = sum(1 for w in words if w in HINGLISH_WORDS)
            ratio = hinglish_count / len(words)
            if hinglish_count >= 2 or ratio >= 0.25:
                return DetectedLanguage("hi-Latn", "Hinglish (Hindi-English)", 0.90, self.profiles["hi-Latn"])

        # 3. Check Spanish markers
        spanish_words = {"que", "como", "cuando", "donde", "por", "para", "con", "una", "los", "las", "hola"}
        if sum(1 for w in words if w in spanish_words) >= 3:
            return DetectedLanguage("es", "Español", 0.85, self.profiles["es"])

        # 4. Check French markers
        french_words = {"que", "qui", "dans", "avec", "pour", "est", "une", "des", "les", "bonjour"}
        if sum(1 for w in words if w in french_words) >= 3:
            return DetectedLanguage("fr", "Français", 0.85, self.profiles["fr"])

        # 5. Check German markers
        german_words = {"und", "der", "die", "das", "mit", "fuer", "für", "ist", "nicht", "hallo"}
        if sum(1 for w in words if w in german_words) >= 3:
            return DetectedLanguage("de", "Deutsch", 0.85, self.profiles["de"])

        # Default to English profile
        return DetectedLanguage("en", "English", 0.80, self.profiles["en"])
