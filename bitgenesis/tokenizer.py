"""A deterministic byte-level BPE tokenizer implemented with the standard library."""

from collections import Counter
from collections.abc import Iterable, Sequence
import codecs
import hashlib
import json
from pathlib import Path

from bitgenesis.corpus import Conversation

_FORMAT_VERSION = 2
_BYTE_VOCABULARY_SIZE = 256
_SPECIAL_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>")


class BPETokenizer:
    """Encode UTF-8 text with byte-pair merges learned from a local corpus."""

    def __init__(
        self, vocabulary: list[bytes], merges: list[tuple[bytes, bytes]]
    ) -> None:
        if len(vocabulary) < _BYTE_VOCABULARY_SIZE:
            raise ValueError("Tokenizer vocabulary must contain all 256 byte values")
        if vocabulary[:_BYTE_VOCABULARY_SIZE] != [
            bytes([value]) for value in range(256)
        ]:
            raise ValueError("Tokenizer vocabulary has an invalid byte-token prefix")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        if any(
            not left or not right or left not in vocabulary or right not in vocabulary
            for left, right in merges
        ):
            raise ValueError("Tokenizer merges contain invalid tokens")
        if any(left + right not in vocabulary for left, right in merges):
            raise ValueError("Tokenizer vocabulary is missing a merged token")

        self._vocabulary = tuple(vocabulary)
        self._merges = tuple(merges)
        self._token_ids = {token: token_id for token_id, token in enumerate(vocabulary)}
        self._special_token_ids = {
            token: len(vocabulary) + index
            for index, token in enumerate(_SPECIAL_TOKENS)
        }

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 1000) -> "BPETokenizer":
        """Learn the most frequent adjacent byte merges from text documents."""
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if vocab_size < _BYTE_VOCABULARY_SIZE:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        sequences = [
            [bytes([value]) for value in text.encode("utf-8")] for text in texts if text
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
                pair for pair in pair_counts if pair[0] + pair[1] not in vocabulary_set
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
                    if (
                        cursor + 1 < len(sequence)
                        and sequence[cursor] == pair[0]
                        and sequence[cursor + 1] == pair[1]
                    ):
                        merged_sequence.append(merged_token)
                        cursor += 2
                    else:
                        merged_sequence.append(sequence[cursor])
                        cursor += 1
                sequences[index] = merged_sequence

        return cls(vocabulary, merges)

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary) + len(_SPECIAL_TOKENS)

    @property
    def text_vocab_size(self) -> int:
        """Return the number of byte-level BPE tokens, excluding control tokens."""
        return len(self._vocabulary)

    def special_token_id(self, token: str) -> int:
        """Return the ID for a reserved conversation token."""
        try:
            return self._special_token_ids[token]
        except KeyError as error:
            raise ValueError(f"Unknown special token: {token}") from error

    @property
    def fingerprint(self) -> str:
        """Return a stable digest for checking model/tokenizer pairing."""
        digest = hashlib.sha256()
        for token in self._vocabulary:
            digest.update(len(token).to_bytes(4, byteorder="big"))
            digest.update(token)
        digest.update(b"\xff")
        for left, right in self._merges:
            for token in (left, right):
                digest.update(len(token).to_bytes(4, byteorder="big"))
                digest.update(token)
        for token in _SPECIAL_TOKENS:
            digest.update(token.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs; every UTF-8 input character is representable."""
        sequence = [bytes([value]) for value in text.encode("utf-8")]
        for left, right in self._merges:
            merged_token = left + right
            merged_sequence: list[bytes] = []
            cursor = 0
            while cursor < len(sequence):
                if (
                    cursor + 1 < len(sequence)
                    and sequence[cursor] == left
                    and sequence[cursor + 1] == right
                ):
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
        pieces: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Token ID is outside the vocabulary: {token_id}")
            if token_id < len(self._vocabulary):
                token_bytes.extend(self._vocabulary[token_id])
            else:
                pieces.append(token_bytes.decode("utf-8"))
                token_bytes.clear()
                pieces.append(_SPECIAL_TOKENS[token_id - len(self._vocabulary)])
        pieces.append(token_bytes.decode("utf-8"))
        return "".join(pieces)

    def encode_chat_prompt(
        self,
        user_message: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> list[int]:
        """Encode prior turns and a new user message for assistant generation."""
        if not user_message.strip():
            raise ValueError("User message must not be empty")
        encoded: list[int] = []
        for user_turn, assistant_turn in history:
            if not user_turn.strip() or not assistant_turn.strip():
                raise ValueError("Conversation history must contain non-empty messages")
            encoded.extend(
                (
                    self.special_token_id("<|user|>"),
                    *self.encode(user_turn),
                    self.special_token_id("<|assistant|>"),
                    *self.encode(assistant_turn),
                    self.special_token_id("<|end|>"),
                )
            )
        encoded.extend(
            (
                self.special_token_id("<|user|>"),
                *self.encode(user_message),
                self.special_token_id("<|assistant|>"),
            )
        )
        return encoded

    def encode_conversation(self, conversation: Conversation) -> list[int]:
        """Encode user/assistant turns from a :class:`Conversation`."""
        if not conversation.turns:
            raise ValueError(
                "Conversation must contain at least one user/assistant turn"
            )
        encoded: list[int] = []
        for turn in conversation.turns:
            if (
                not isinstance(turn, tuple)
                or len(turn) != 2
                or any(
                    not isinstance(message, str) or not message.strip()
                    for message in turn
                )
            ):
                raise ValueError(
                    "Conversation turns must contain non-empty user and assistant text"
                )
            user_message, assistant_message = turn
            encoded.extend(self.encode_chat_prompt(user_message))
            encoded.extend(self.encode(assistant_message))
            encoded.append(self.special_token_id("<|end|>"))
        return encoded

    def decode_chat_reply(self, token_ids: Iterable[int]) -> str:
        """Decode assistant tokens until the end-of-response marker."""
        reply_ids: list[int] = []
        end_token_id = self.special_token_id("<|end|>")
        for token_id in token_ids:
            if token_id == end_token_id:
                break
            if token_id >= len(self._vocabulary):
                raise ValueError("Assistant reply contains an unexpected special token")
            reply_ids.append(token_id)
        return self.decode(reply_ids)

    def valid_next_token_ids(
        self,
        token_ids: Iterable[int],
        require_complete: bool = False,
        max_remaining_tokens: int | None = None,
    ) -> list[int]:
        """Return tokens that keep a generated UTF-8 byte stream valid."""
        if max_remaining_tokens is not None and (
            isinstance(max_remaining_tokens, bool)
            or not isinstance(max_remaining_tokens, int)
            or max_remaining_tokens < 0
        ):
            raise ValueError("max_remaining_tokens must be a non-negative integer")

        decoder_type = codecs.getincrementaldecoder("utf-8")
        decoder = decoder_type(errors="strict")
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Token ID is outside the vocabulary: {token_id}")
            if token_id < len(self._vocabulary):
                decoder.decode(self._vocabulary[token_id], final=False)
            else:
                decoder.decode(b"", final=True)
                decoder = decoder_type(errors="strict")

        decoder_state = decoder.getstate()
        valid_ids = []
        for token_id in range(self.vocab_size):
            candidate_decoder = decoder_type(errors="strict")
            candidate_decoder.setstate(decoder_state)
            try:
                if token_id < len(self._vocabulary):
                    candidate_decoder.decode(self._vocabulary[token_id], final=False)
                else:
                    candidate_decoder.decode(b"", final=True)
                pending_bytes = candidate_decoder.getstate()[0]
                required_continuations = 0
                if pending_bytes:
                    lead_byte = pending_bytes[0]
                    sequence_length = (
                        2
                        if 0xC2 <= lead_byte <= 0xDF
                        else 3
                        if 0xE0 <= lead_byte <= 0xEF
                        else 4
                        if 0xF0 <= lead_byte <= 0xF4
                        else 0
                    )
                    required_continuations = sequence_length - len(pending_bytes)
                if require_complete and pending_bytes:
                    continue
                if (
                    max_remaining_tokens is not None
                    and required_continuations > max_remaining_tokens
                ):
                    continue
                if require_complete:
                    candidate_decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                continue
            valid_ids.append(token_id)
        return valid_ids

    def save(self, path: Path) -> None:
        """Save the tokenizer vocabulary and merge rules as UTF-8 JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "vocabulary": [token.hex() for token in self._vocabulary],
            "merges": [[left.hex(), right.hex()] for left, right in self._merges],
            "special_tokens": list(_SPECIAL_TOKENS),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load a tokenizer saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != _FORMAT_VERSION
        ):
            raise ValueError("Unsupported or invalid tokenizer format")

        vocabulary_data = payload.get("vocabulary")
        merges_data = payload.get("merges")
        if (
            payload.get("special_tokens") != list(_SPECIAL_TOKENS)
            or not isinstance(vocabulary_data, list)
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
            merges = [
                (bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in merges_data
            ]
        except ValueError as error:
            raise ValueError(
                "Tokenizer file contains invalid hexadecimal tokens"
            ) from error
        return cls(vocabulary, merges)
