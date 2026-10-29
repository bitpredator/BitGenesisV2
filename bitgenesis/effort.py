"""Select chat response budgets from explicit or automatic effort levels."""

import re
from typing import Literal, TypeGuard

EffortLevel = Literal["auto", "minimo", "medio", "alto"]
GenerationEffort = Literal["minimo", "medio", "alto"]
EFFORT_LEVELS: tuple[EffortLevel, ...] = ("auto", "minimo", "medio", "alto")

_HIGH_EFFORT_CUES = {
    "analizza",
    "confronta",
    "crea",
    "debug",
    "differenze",
    "implementa",
    "piano",
    "perche",
    "perché",
    "progetta",
    "proponi",
    "riassumi",
    "scrivi",
    "spiega",
    "spiegami",
    "sintetizza",
    "valuta",
}


def is_effort_level(value: str) -> TypeGuard[EffortLevel]:
    return value in EFFORT_LEVELS


def choose_effort(
    prompt: str,
    requested_effort: EffortLevel,
) -> GenerationEffort:
    if requested_effort != "auto":
        return requested_effort
    words = re.findall(r"\w+", prompt.casefold())
    if len(words) > 14 or _HIGH_EFFORT_CUES.intersection(words):
        return "alto"
    if len(words) > 5 or "?" in prompt:
        return "medio"
    return "minimo"


def effort_token_budget(base_tokens: int, effort: GenerationEffort) -> int:
    if base_tokens == 0:
        return 0
    if effort == "minimo":
        return max(1, base_tokens // 4)
    if effort == "medio":
        return base_tokens
    if effort == "alto":
        return base_tokens * 2
    raise ValueError(f"Livello di impegno non valido: {effort}")
