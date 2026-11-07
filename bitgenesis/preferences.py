"""Detect and store explicit user work preferences."""

import re
from pathlib import Path

_TITLE = "# Preferenze di lavoro dell'utente"
_PREFERENCE_MARKER = re.compile(
    r"\b(?:preferisco|preferirei|mi trovo meglio|mi è più utile|"
    r"mi e piu utile|mi piace lavorare|per lavorare preferisco|"
    r"da ora in poi|d'ora in avanti|ricordati che|tieni a mente che|"
    r"i prefer|i would prefer|i'd prefer|i work best|"
    r"it helps me when|i like to work|from now on|"
    r"please remember that|keep in mind that)\b",
    re.IGNORECASE,
)
_WORK_CONTEXT = re.compile(
    r"\b(?:rispost\w*|answer\w*|brev\w*|concise|short|dettagli\w*|"
    r"detail\w*|passagg\w*|step\w*|codic\w*|code|lingu\w*|language|"
    r"spiegaz\w*|explan\w*|esemp\w*|example\w*|formatt\w*|format\w*|"
    r"approcci\w*|approach\w*|test\w*|comment\w*|python|terminal|"
    r"commit\w*|document\w*|lavor\w*|work|program\w*|coding|"
    r"modific\w*|minim\w*|minimal\w*|simple|semplic\w*|diff|"
    r"elenc\w*|list\w*|punt\w*|bullet\w*)\b",
    re.IGNORECASE,
)
_PERSISTENT_MARKER = re.compile(
    r"\b(?:da ora in poi|d'ora in avanti|ricordati che|tieni a mente che|"
    r"from now on|please remember that|keep in mind that)\b",
    re.IGNORECASE,
)
_MAX_PREFERENCE_LENGTH = 1000


def detect_work_preference(message: str) -> str | None:
    """Return a short, explicit work preference, not an inferred one."""
    text = " ".join(message.split())
    if (
        not text
        or len(text) > _MAX_PREFERENCE_LENGTH
        or not _PREFERENCE_MARKER.search(text)
        or not (
            _WORK_CONTEXT.search(text)
            or _PERSISTENT_MARKER.search(text)
        )
    ):
        return None
    return text


def read_preferences(path: Path) -> str:
    """Read saved preferences, returning an empty string when absent."""
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").removeprefix(_TITLE).strip()


def update_preferences(
    path: Path,
    text: str,
    *,
    replace: bool = False,
) -> bool:
    """Append a preference once or replace all saved preferences atomically."""
    content = " ".join(text.split())
    if not content:
        raise ValueError("Le preferenze non possono essere vuote.")
    if len(content) > _MAX_PREFERENCE_LENGTH:
        raise ValueError(
            f"Ogni preferenza deve essere lunga al massimo "
            f"{_MAX_PREFERENCE_LENGTH} caratteri."
        )

    path = Path(path)
    existing = read_preferences(path)
    preferences = [
        line.removeprefix("- ").strip()
        for line in existing.splitlines()
        if line.strip()
    ]
    if replace:
        preferences = [content]
        changed = existing != content
    else:
        known = {preference.casefold() for preference in preferences}
        changed = content.casefold() not in known
        if changed:
            preferences.append(content)
    if not changed:
        return False

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            f"{_TITLE}\n\n"
            + "\n".join(f"- {preference}" for preference in preferences)
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return True
