"""Shared instructions for BitGenesis conversation profiles."""

import re
import unicodedata
from typing import Literal

AgentName = Literal["chat", "coding"]

_INTRODUCTION_PATTERNS = (
    re.compile(r"\bpresentati\b"),
    re.compile(r"\bchi sei\b"),
    re.compile(r"\bparlami di te\b"),
    re.compile(r"\bcosa sai fare\b"),
    re.compile(r"\bche cosa sai fare\b"),
    re.compile(r"\bquali sono le tue capacita\b"),
    re.compile(r"\bpresentazione di bitgenesis\b"),
    re.compile(r"\bwho are you\b"),
    re.compile(r"\bintroduce yourself\b"),
    re.compile(r"\bwhat can you do\b"),
)
_GREETING_PATTERN = re.compile(
    r"(?:ciao|salve|buongiorno|buonasera|buon pomeriggio|ehi|hey|hello)"
    r"(?: bitgenesis)?[.!?… ]*"
)

AGENT_INSTRUCTIONS: dict[AgentName, str] = {
    "chat": (
        "Sei l'agente chat di BitGenesis. Aiuta l'utente con domande e "
        "conversazioni generali in modo chiaro e conciso."
    ),
    "coding": (
        "Sei l'agente coding di BitGenesis. Comprendi requisiti e codice "
        "esistente; chiedi chiarimenti se manca un requisito essenziale. "
        "Segui convenzioni e versioni del progetto; scegli soluzioni semplici "
        "e complete, limitando modifiche e dipendenze non necessarie. Preserva "
        "il comportamento, valida gli input e gestisci gli errori "
        "esplicitamente. Suggerisci test pertinenti per le modifiche. Non "
        "inventare API o risultati: dichiara ipotesi, limiti e test non "
        "eseguiti. Fornisci codice chiaro e utilizzabile, spiegando "
        "brevemente come provarlo."
    ),
}


def self_introduction(prompt: str, agent: AgentName) -> str | None:
    """Return a factual introduction for explicit identity and capability prompts."""
    normalized = unicodedata.normalize("NFKD", prompt.casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    if not any(pattern.search(normalized) for pattern in _INTRODUCTION_PATTERNS):
        return None

    if agent == "coding":
        return (
            "Sono BitGenesis, un assistente sperimentale in italiano che "
            "funziona in locale sul tuo computer. Nel profilo Coding posso "
            "provare ad aiutarti a comprendere, scrivere e correggere codice. "
            "Non eseguo programmi o test e le mie risposte possono essere "
            "imprecise: verifica il codice prima di usarlo."
        )
    return (
        "Sono BitGenesis, un assistente sperimentale in italiano che "
        "funziona in locale sul tuo computer. Posso conversare e provare ad "
        "aiutarti con nozioni di base, anche di Python, e con l'uso del "
        "progetto. Non navigo in Internet né eseguo codice; le mie risposte "
        "possono essere imprecise, quindi verifica le informazioni e prova "
        "il codice prima di usarlo."
    )


def simple_greeting(prompt: str, agent: AgentName) -> str | None:
    """Return a short deterministic response to a standalone greeting."""
    normalized = unicodedata.normalize("NFKD", prompt.casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).strip()
    if not _GREETING_PATTERN.fullmatch(normalized):
        return None

    if agent == "coding":
        return (
            "Ciao! Sono BitGenesis. Per una richiesta di programmazione, "
            "descrivimi l'obiettivo e, se serve, condividi il codice."
        )
    return "Ciao! Sono BitGenesis. Come posso aiutarti?"
