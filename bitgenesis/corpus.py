"""Load and split local text documents for model training."""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CorpusRecord(Generic[T]):
    content: T
    path: Path
    line_number: int


@dataclass(frozen=True)
class CorpusSplit(Generic[T]):
    train: tuple[T, ...]
    validation: tuple[T, ...]


@dataclass(frozen=True)
class Conversation:
    turns: tuple[tuple[str, str], ...]


def load_corpus_records(directory: Path) -> list[CorpusRecord[str]]:
    """Read UTF-8 text documents and retain their source paths."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {directory}")

    files = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".txt"
        ),
        key=lambda path: path.as_posix(),
    )
    records = [
        CorpusRecord(content=path.read_text(encoding="utf-8"), path=path, line_number=1)
        for path in files
    ]
    records = [record for record in records if record.content]
    if not records:
        raise ValueError(
            f"No non-empty .txt documents found in corpus directory: {directory}"
        )
    return records


def load_corpus(directory: Path) -> list[str]:
    """Read UTF-8 .txt documents recursively in a stable order."""
    return [record.content for record in load_corpus_records(directory)]


def load_conversation_records(directory: Path) -> list[CorpusRecord[Conversation]]:
    """Load JSONL conversations and retain their source paths and line numbers."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Conversation directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Conversation path is not a directory: {directory}")

    files = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".jsonl"
        ),
        key=lambda path: path.as_posix(),
    )
    records: list[CorpusRecord[Conversation]] = []
    for path in files:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {error.msg}"
                    ) from error
                records.append(
                    CorpusRecord(
                        content=_parse_conversation(record, path, line_number),
                        path=path,
                        line_number=line_number,
                    )
                )

    if not records:
        raise ValueError(
            f"No non-empty conversations found in JSONL files in: {directory}"
        )
    return records


def load_conversations(directory: Path) -> list[Conversation]:
    """Load JSONL conversations with alternating user/assistant messages."""
    return [record.content for record in load_conversation_records(directory)]


def find_duplicate_groups(
    records: Sequence[CorpusRecord[T]],
) -> list[tuple[CorpusRecord[T], ...]]:
    """Return exact duplicate records, preserving each source location."""
    grouped: dict[Hashable, list[CorpusRecord[T]]] = {}
    for record in records:
        try:
            grouped.setdefault(record.content, []).append(record)
        except TypeError as error:
            raise TypeError("Corpus record content must be hashable") from error
    return [tuple(group) for group in grouped.values() if len(group) > 1]


def _parse_conversation(record: object, path: Path, line_number: int) -> Conversation:
    location = f"{path} at line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"Conversation record must be a JSON object ({location})")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError(
            f"Conversation messages must contain user/assistant pairs ({location})"
        )

    turns: list[tuple[str, str]] = []
    for index in range(0, len(messages), 2):
        user_message = messages[index]
        assistant_message = messages[index + 1]
        if (
            not isinstance(user_message, dict)
            or user_message.get("role") != "user"
            or not isinstance(user_message.get("content"), str)
            or not user_message["content"].strip()
        ):
            raise ValueError(
                f"Message {index + 1} must have role 'user' and non-empty content ({location})"
            )
        if (
            not isinstance(assistant_message, dict)
            or assistant_message.get("role") != "assistant"
            or not isinstance(assistant_message.get("content"), str)
            or not assistant_message["content"].strip()
        ):
            raise ValueError(
                f"Message {index + 2} must have role 'assistant' and non-empty content ({location})"
            )
        turns.append((user_message["content"], assistant_message["content"]))
    return Conversation(turns=tuple(turns))


def split_corpus(
    documents: list[T] | tuple[T, ...],
    validation_fraction: float = 0.1,
    seed: int = 42,
    *,
    group_key: Callable[[T], Hashable] | None = None,
) -> CorpusSplit[T]:
    """Split reproducibly, optionally keeping equal-key records together."""
    if not 0 <= validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be greater than or equal to 0 and less than 1"
        )
    if not documents:
        raise ValueError("At least one corpus document is required")
    if validation_fraction == 0:
        return CorpusSplit(train=tuple(documents), validation=())
    if group_key is None:
        groups = [[document] for document in documents]
    else:
        grouped: dict[Hashable, list[T]] = {}
        for document in documents:
            grouped.setdefault(group_key(document), []).append(document)
        groups = list(grouped.values())

    if len(groups) < 2:
        raise ValueError("At least two documents are required for a validation split")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    validation = tuple(
        document
        for group in shuffled[:validation_count]
        for document in group
    )
    train = tuple(
        document
        for group in shuffled[validation_count:]
        for document in group
    )
    return CorpusSplit(
        train=train,
        validation=validation,
    )
