import tempfile
import unittest
from pathlib import Path

from bitgenesis.corpus import load_corpus, split_corpus


class LoadCorpusTests(unittest.TestCase):
    def test_loads_utf8_text_recursively_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "second.txt").write_text("Ciao", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "first.TXT").write_text("È una prova.", encoding="utf-8")
            (root / "ignored.csv").write_text("not text", encoding="utf-8")
            (root / "empty.txt").write_text("", encoding="utf-8")

            self.assertEqual(load_corpus(root), ["È una prova.", "Ciao"])

    def test_rejects_missing_directory_and_empty_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                load_corpus(root / "missing")
            with self.assertRaisesRegex(ValueError, "No non-empty"):
                load_corpus(root)

    def test_invalid_utf8_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.txt"
            path.write_bytes(b"\xff")

            with self.assertRaises(UnicodeDecodeError):
                load_corpus(Path(temp_dir))


class SplitCorpusTests(unittest.TestCase):
    def test_split_is_reproducible_and_keeps_documents_disjoint(self) -> None:
        documents = [f"document {index}" for index in range(10)]

        first = split_corpus(documents, validation_fraction=0.2, seed=11)
        second = split_corpus(documents, validation_fraction=0.2, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(len(first.validation), 2)
        self.assertEqual(set(first.train).union(first.validation), set(documents))
        self.assertFalse(set(first.train).intersection(first.validation))

    def test_zero_validation_fraction_keeps_every_document_for_training(self) -> None:
        split = split_corpus(["solo documento"], validation_fraction=0)

        self.assertEqual(split.train, ("solo documento",))
        self.assertEqual(split.validation, ())

    def test_rejects_invalid_validation_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            split_corpus([])
        with self.assertRaisesRegex(ValueError, "At least two"):
            split_corpus(["solo documento"])
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            split_corpus(["a", "b"], validation_fraction=1)


if __name__ == "__main__":
    unittest.main()
