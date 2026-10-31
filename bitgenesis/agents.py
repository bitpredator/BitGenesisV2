"""Shared instructions for BitGenesis conversation profiles."""

import re
import unicodedata
from typing import Literal

from bitgenesis.language import Language, detect_language, language_instruction

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
    re.compile(r"\btell me about yourself\b"),
    re.compile(r"\bwhat can you do\b"),
    re.compile(r"\bwhat are your capabilities\b"),
)
_GREETING_PATTERN = re.compile(
    r"(?:ciao|salve|buongiorno|buonasera|buon pomeriggio|ehi|hey|hello|hi|"
    r"good morning|good afternoon|good evening)"
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
_CODING_INSTRUCTION_EN = (
    "Understand the requirements and existing code; ask for clarification if "
    "an essential requirement is missing. Follow the project's conventions "
    "and supported versions; choose simple, complete solutions and avoid "
    "unnecessary changes or dependencies. Preserve behavior, validate inputs, "
    "and handle errors explicitly. Suggest relevant tests for changes. Do not "
    "invent APIs or results: state assumptions, limitations, and tests not "
    "run. Provide clear, usable code and briefly explain how to try it."
)
_CHAT_INSTRUCTION_EN = (
    "You are BitGenesis's chat agent. Help the user with general questions "
    "and conversations clearly and concisely."
)


def agent_instructions(agent: AgentName, language: Language) -> str:
    """Return profile instructions in the selected response language."""
    if language == "en":
        return (
            _CODING_INSTRUCTION_EN
            if agent == "coding"
            else _CHAT_INSTRUCTION_EN
        )
    return AGENT_INSTRUCTIONS[agent]


def build_response_prompt(
    message: str,
    agent: AgentName,
    project_instructions: str,
    language: Language,
) -> str:
    """Build the same language-aware prompt for terminal and web chats."""
    sections = []
    if agent == "coding":
        heading = (
            "Coding agent instructions:"
            if language == "en"
            else "Istruzioni per l'agente coding:"
        )
        sections.append(f"{heading}\n{agent_instructions(agent, language)}")
    if project_instructions:
        heading = (
            "Permanent project instructions from Genesis.md:"
            if language == "en"
            else "Istruzioni permanenti del progetto da Genesis.md:"
        )
        sections.append(f"{heading}\n{project_instructions}")

    request_label = (
        "User request" if language == "en" else "Richiesta dell'utente"
    )
    sections.extend(
        (f"{request_label}: {message}", language_instruction(language))
    )
    return "\n\n".join(sections)


def local_user_rule(language: Language) -> tuple[str, str]:
    """Return a localized fixed-name instruction for local chat history."""
    if language == "en":
        return (
            "Rule for this local chat: address the user as User.",
            "In this local chat I will address the user as User.",
        )
    return (
        "Regola per questa chat locale: chiama l'utente Utente.",
        "In questa chat locale chiamerò l'utente Utente.",
    )


def self_introduction(
    prompt: str,
    agent: AgentName,
    language: Language | None = None,
) -> str | None:
    """Return a factual introduction for explicit identity and capability prompts."""
    normalized = unicodedata.normalize("NFKD", prompt.casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    if not any(pattern.search(normalized) for pattern in _INTRODUCTION_PATTERNS):
        return None

    selected_language = detect_language(prompt) if language is None else language
    if selected_language == "en":
        if agent == "coding":
            return (
                "I'm BitGenesis, an experimental assistant that runs locally on "
                "your computer. In Coding mode, I can try to help you understand, "
                "write, and fix code. I don't run programs or tests, and my "
                "answers may be inaccurate, so check the code before using it."
            )
        return (
            "I'm BitGenesis, an experimental assistant that runs locally on your "
            "computer. I can chat and try to help with basic topics, including "
            "Python and using this project. I don't browse the Internet or run "
            "code; my answers may be inaccurate, so verify information and test "
            "code before using it."
        )

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


def simple_greeting(
    prompt: str,
    agent: AgentName,
    language: Language | None = None,
) -> str | None:
    """Return a short deterministic response to a standalone greeting."""
    normalized = unicodedata.normalize("NFKD", prompt.casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).strip()
    if not _GREETING_PATTERN.fullmatch(normalized):
        return None

    selected_language = detect_language(prompt) if language is None else language
    if selected_language == "en":
        if agent == "coding":
            return (
                "Hello! I'm BitGenesis. For a programming request, describe "
                "your goal and share the code if needed."
            )
        return "Hello! I'm BitGenesis. How can I help?"

    if agent == "coding":
        return (
            "Ciao! Sono BitGenesis. Per una richiesta di programmazione, "
            "descrivimi l'obiettivo e, se serve, condividi il codice."
        )
    return "Ciao! Sono BitGenesis. Come posso aiutarti?"
