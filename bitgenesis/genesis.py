"""Read and update project-specific persistent instructions."""

from pathlib import Path

_TITLE = "# Istruzioni permanenti per BitGenesis"


def read_instructions(path: Path) -> str:
    """Read project instructions, returning an empty string when no file exists."""
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").removeprefix(_TITLE).strip()


def update_instructions(path: Path, text: str, *, replace: bool = False) -> None:
    """Append or replace project instructions using an atomic file update."""
    content = text.strip()
    if not content:
        raise ValueError("Le istruzioni non possono essere vuote.")

    path = Path(path)
    existing = read_instructions(path)
    instructions = content if replace or not existing else f"{existing}\n\n{content}"
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            f"{_TITLE}\n\n{instructions}\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
