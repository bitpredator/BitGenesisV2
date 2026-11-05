"""Detect the supported response language from a user message."""

import re
import unicodedata
from typing import Literal

Language = Literal["en", "it"]

_WORD_PATTERN = re.compile(r"[a-z]+")
_ITALIAN_MARKERS = frozenset(
    {
        "anche",
        "allora",
        "aiuto",
        "aiutami",
        "aiutarmi",
        "avere",
        "buonasera",
        "buongiorno",
        "buon",
        "che",
        "chi",
        "ciao",
        "come",
        "con",
        "cosa",
        "della",
        "delle",
        "dello",
        "dove",
        "ehi",
        "essere",
        "fare",
        "faccio",
        "fammi",
        "grazie",
        "guida",
        "gli",
        "hanno",
        "il",
        "invece",
        "la",
        "lo",
        "al",
        "alla",
        "alle",
        "allo",
        "ai",
        "agli",
        "le",
        "mi",
        "ma",
        "molto",
        "nel",
        "nei",
        "nella",
        "nelle",
        "non",
        "per",
        "perche",
        "pomeriggio",
        "poi",
        "posso",
        "presentati",
        "potresti",
        "puoi",
        "piu",
        "si",
        "se",
        "di",
        "da",
        "del",
        "degli",
        "quale",
        "quali",
        "qual",
        "quando",
        "quanto",
        "questa",
        "questo",
        "salve",
        "scrivi",
        "sono",
        "spiega",
        "spiegami",
        "traduci",
        "una",
        "uno",
        "un",
        "vorrei",
        "voglio",
        "sto",
        "ora",
        "cercando",
        "capire",
        "devo",
        "dimmi",
        "mostra",
        "riassumi",
        "rendilo",
        "correggi",
        "crea",
        "aggiungi",
        "informale",
    }
)
_ENGLISH_MARKERS = frozenset(
    {
        "about",
        "after",
        "are",
        "because",
        "before",
        "can",
        "could",
        "does",
        "doing",
        "for",
        "create",
        "explain",
        "fix",
        "from",
        "goodbye",
        "hello",
        "help",
        "hey",
        "hi",
        "how",
        "is",
        "please",
        "summarize",
        "translate",
        "write",
        "should",
        "thanks",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
)


def detect_language(text: str) -> Language:
    """Detect Italian or English; default to English when unclear or unsupported."""
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    words = _WORD_PATTERN.findall(normalized)
    italian_score = sum(word in _ITALIAN_MARKERS for word in words)
    english_score = sum(word in _ENGLISH_MARKERS for word in words)
    return "it" if italian_score > english_score else "en"


def language_instruction(language: Language) -> str:
    """Return a per-turn instruction to keep replies in the user's language."""
    if language == "it":
        return "rispondi in italiano."
    return "reply in English."
