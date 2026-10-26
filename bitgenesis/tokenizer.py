"""A deterministic byte-level BPE tokenizer implemented with the standard library."""

from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path

_FORMAT_VERSION = 1
_BYTE_VOCABULARY_SIZE = 256


class BPETokenizer:
    """Encode UTF-8 text with byte-pair merges learned from a local corpus."""

    def __init__(self, vocabulary: list[bytes], merges: list[tuple[bytes, bytes]]) -> None:
        if len(vocabulary) < _BYTE_VOCABULARY_SIZE:
            raise ValueError("Tokenizer vocabulary must contain all 256 byte values")
        if vocabulary[:_BYTE_VOCABULARY_SIZE] != [bytes([value]) for value in range(256)]:
            raise ValueError("Tokenizer vocabulary has an invalid byte-token prefix")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        if any(not left or not right or left not in vocabulary or right not in vocabulary for left, right in merges):
            raise ValueError("Tokenizer merges contain invalid tokens")
        if any(left + right not in vocabulary for left, right in merges):
            raise ValueError("Tokenizer vocabulary is missing a merged token")

        self._vocabulary = tuple(vocabulary)
        self._merges = tuple(merges)
        self._token_ids = {token: token_id for token_id, token in enumerate(vocabulary)}

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 1000) -> "BPETokenizer":
        """Learn the most frequent adjacent byte merges from text documents."""
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if vocab_size < _BYTE_VOCABULARY_SIZE:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        sequences = [
            [bytes([value]) for value in text.encode("utf-8")]
            for text in texts
            if text
        ]
        if not sequences:
            raise ValueError("At least one non-empty training text is required")

        vocabulary = [bytes([value]) for value in range(256)]
        vocabulary_set = set(vocabulary)
        merges: list[tuple[bytes, bytes]] = []

        while len(vocabulary) < vocab_size:
            pair_counts: Counter[tuple[bytes, bytes]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))

            eligible_pairs = [
                pair for pair in pair_counts
                if pair[0] + pair[1] not in vocabulary_set
            ]
            if not eligible_pairs:
                break

            pair = max(eligible_pairs, key=lambda item: (pair_counts[item], item))
            merged_token = pair[0] + pair[1]
            merges.append(pair)
            vocabulary.append(merged_token)
            vocabulary_set.add(merged_token)

            for index, sequence in enumerate(sequences):
                merged_sequence: list[bytes] = []
                cursor = 0
                while cursor < len(sequence):
                    if cursor + 1 < len(sequence) and sequence[cursor] == pair[0] and sequence[cursor + 1] == pair[1]:
                        merged_sequence.append(merged_token)
                        cursor += 2
                    else:
                        merged_sequence.append(sequence[cursor])
                        cursor += 1
                sequences[index] = merged_sequence

        return cls(vocabulary, merges)

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs; every UTF-8 input character is representable."""
        sequence = [bytes([value]) for value in text.encode("utf-8")]
        for left, right in self._merges:
            merged_token = left + right
            merged_sequence: list[bytes] = []
            cursor = 0
            while cursor < len(sequence):
                if cursor + 1 < len(sequence) and sequence[cursor] == left and sequence[cursor + 1] == right:
                    merged_sequence.append(merged_token)
                    cursor += 2
                else:
                    merged_sequence.append(sequence[cursor])
                    cursor += 1
            sequence = merged_sequence
        return [self._token_ids[token] for token in sequence]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back to text, rejecting invalid IDs or malformed UTF-8."""
        token_bytes = bytearray()
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Token ID is outside the vocabulary: {token_id}")
            token_bytes.extend(self._vocabulary[token_id])
        return token_bytes.decode("utf-8")

    def save(self, path: Path) -> None:
        """Save the tokenizer vocabulary and merge rules as UTF-8 JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "vocabulary": [token.hex() for token in self._vocabulary],
            "merges": [[left.hex(), right.hex()] for left, right in self._merges],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load a tokenizer saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != _FORMAT_VERSION:
            raise ValueError("Unsupported or invalid tokenizer format")

        vocabulary_data = payload.get("vocabulary")
        merges_data = payload.get("merges")
        if (
            not isinstance(vocabulary_data, list)
            or any(not isinstance(token, str) for token in vocabulary_data)
            or not isinstance(merges_data, list)
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(token, str) for token in pair)
                for pair in merges_data
            )
        ):
            raise ValueError("Tokenizer file contains invalid vocabulary or merges")

        try:
            vocabulary = [bytes.fromhex(token) for token in vocabulary_data]
            merges = [(bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in merges_data]
        except ValueError as error:
            raise ValueError("Tokenizer file contains invalid hexadecimal tokens") from error
        return cls(vocabulary, merges)
