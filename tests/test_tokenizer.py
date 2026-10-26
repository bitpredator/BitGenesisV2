import tempfile
import unittest
from pathlib import Path

from bitgenesis.tokenizer import BPETokenizer


class BPETokenizerTests(unittest.TestCase):
    def test_learns_merges_and_round_trips_italian_text(self) -> None:
        tokenizer = BPETokenizer.train(["ciao ciao", "perché ciao"], vocab_size=270)
        text = "Ciao, perché? 🙂"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertGreater(tokenizer.vocab_size, 256)

    def test_representable_bytes_cover_characters_not_seen_during_training(self) -> None:
        tokenizer = BPETokenizer.train(["ciao"], vocab_size=256)

        self.assertEqual(tokenizer.decode(tokenizer.encode("È una prova 🙂")), "È una prova 🙂")

    def test_valid_next_tokens_preserve_utf8_and_can_finish_partial_characters(self) -> None:
        tokenizer = BPETokenizer.train(["ciao"], vocab_size=256)
        continuation_ids = tokenizer.valid_next_token_ids([0xC3])
        final_continuation_ids = tokenizer.valid_next_token_ids(
            [0xC3],
            require_complete=True,
        )
        four_byte_lead = tokenizer.valid_next_token_ids(
            [],
            max_remaining_tokens=2,
        )
        enough_for_four_byte_character = tokenizer.valid_next_token_ids(
            [],
            max_remaining_tokens=3,
        )

        self.assertIn(0xA9, continuation_ids)
        self.assertNotIn(ord("A"), continuation_ids)
        self.assertIn(0xA9, final_continuation_ids)
        self.assertNotIn(0xF0, four_byte_lead)
        self.assertIn(0xF0, enough_for_four_byte_character)

    def test_training_is_deterministic(self) -> None:
        texts = ["abc abc", "abd abd"]
        first = BPETokenizer.train(texts, vocab_size=265)
        second = BPETokenizer.train(texts, vocab_size=265)

        self.assertEqual(first.encode("abc abd"), second.encode("abc abd"))
        self.assertEqual(first.vocab_size, second.vocab_size)

    def test_save_and_load_preserve_encoding(self) -> None:
        tokenizer = BPETokenizer.train(["ciao ciao", "salve"], vocab_size=270)
        text = "Ciao, BitGenesis!"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "models" / "tokenizer.json"
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)

        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)
        self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)

    def test_rejects_invalid_training_options_and_token_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 256"):
            BPETokenizer.train(["ciao"], vocab_size=255)
        with self.assertRaisesRegex(ValueError, "non-empty training text"):
            BPETokenizer.train(["", ""], vocab_size=256)

        tokenizer = BPETokenizer.train(["ciao"], vocab_size=256)
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            tokenizer.decode([256])


if __name__ == "__main__":
    unittest.main()
