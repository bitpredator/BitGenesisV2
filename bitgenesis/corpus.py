"""Load and split local text documents for model training."""

from dataclasses import dataclass
from pathlib import Path
import random


@dataclass(frozen=True)
class CorpusSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]


def load_corpus(directory: Path) -> list[str]:
    """Read UTF-8 .txt documents recursively in a stable order."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {directory}")

    files = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.as_posix(),
    )
    documents = [path.read_text(encoding="utf-8") for path in files]
    documents = [document for document in documents if document]
    if not documents:
        raise ValueError(f"No non-empty .txt documents found in corpus directory: {directory}")
    return documents


def split_corpus(
    documents: list[str] | tuple[str, ...],
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> CorpusSplit:
    """Split whole documents reproducibly, keeping each document in one split."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than or equal to 0 and less than 1")
    if not documents:
        raise ValueError("At least one corpus document is required")
    if validation_fraction == 0:
        return CorpusSplit(train=tuple(documents), validation=())
    if len(documents) < 2:
        raise ValueError("At least two documents are required for a validation split")

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    return CorpusSplit(
        train=tuple(shuffled[validation_count:]),
        validation=tuple(shuffled[:validation_count]),
    )
