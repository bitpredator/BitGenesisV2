import tempfile
import unittest
from pathlib import Path

from bitgenesis.corpus import (
    find_duplicate_groups,
    load_conversation_records,
    load_conversations,
    load_corpus,
    load_corpus_records,
    split_corpus,
)


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

    def test_records_include_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("Testo", encoding="utf-8")

            records = load_corpus_records(Path(temp_dir))

        self.assertEqual(records[0].content, "Testo")
        self.assertEqual(records[0].path, path)
        self.assertEqual(records[0].line_number, 1)

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

    def test_keeps_duplicate_records_in_the_same_split(self) -> None:
        documents = [
            ("duplicato", "file-a.jsonl", 1),
            ("unico", "file-a.jsonl", 2),
            ("duplicato", "file-b.jsonl", 7),
            ("altro", "file-c.jsonl", 1),
        ]

        split = split_corpus(
            documents,
            validation_fraction=0.5,
            seed=11,
            group_key=lambda record: record[0],
        )

        train_content = {record[0] for record in split.train}
        validation_content = {record[0] for record in split.validation}
        self.assertFalse(train_content.intersection(validation_content))
        self.assertEqual(
            sum(record[0] == "duplicato" for record in split.train)
            + sum(record[0] == "duplicato" for record in split.validation),
            2,
        )

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


class LoadConversationsTests(unittest.TestCase):
    def test_programming_curriculum_is_valid_training_data(self) -> None:
        dataset_dir = (
            Path(__file__).resolve().parents[1] / "data" / "datasets"
        )

        records = load_conversation_records(dataset_dir)
        lessons = [
            record.content
            for record in records
            if record.path.name == "programmazione_python_basi.jsonl"
        ]
        lesson_text = "\n".join(
            message
            for conversation in lessons
            for turn in conversation.turns
            for message in turn
        )

        self.assertEqual(len(lessons), 11)
        self.assertIn('print("Ciao!")', lesson_text)
        self.assertIn('nome = "Ada"', lesson_text)
        self.assertIn("somma = a + b", lesson_text)
        self.assertIn("Vivo a Roma", lesson_text)

    def test_loads_jsonl_conversations_recursively_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "second.jsonl").write_text(
                '{"messages":[{"role":"user","content":"Ciao"},{"role":"assistant","content":"Salve"}]}\n',
                encoding="utf-8",
            )
            nested = root / "nested"
            nested.mkdir()
            (nested / "first.JSONL").write_text(
                '{"messages":[{"role":"user","content":"Come va?"},'
                '{"role":"assistant","content":"Bene."},'
                '{"role":"user","content":"Grazie"},{"role":"assistant","content":"Prego."}]}\n',
                encoding="utf-8",
            )
            (root / "ignored.txt").write_text("not jsonl", encoding="utf-8")

            conversations = load_conversations(root)

        self.assertEqual(
            conversations[0].turns,
            (("Come va?", "Bene."), ("Grazie", "Prego.")),
        )
        self.assertEqual(conversations[1].turns, (("Ciao", "Salve"),))

    def test_duplicate_groups_retain_file_and_line_locations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            duplicate = (
                '{"messages":[{"role":"user","content":"Ciao"},'
                '{"role":"assistant","content":"Salve"}]}\n'
            )
            first.write_text(duplicate + duplicate, encoding="utf-8")
            second.write_text(
                '{"messages":[{"role":"user","content":"Come va?"},'
                '{"role":"assistant","content":"Bene."}]}\n',
                encoding="utf-8",
            )

            records = load_conversation_records(root)
            duplicate_groups = find_duplicate_groups(records)

        self.assertEqual(len(duplicate_groups), 1)
        self.assertEqual(
            [(record.path, record.line_number) for record in duplicate_groups[0]],
            [(first, 1), (first, 2)],
        )

    def test_reports_invalid_json_and_invalid_message_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "invalid.jsonl"
            path.write_text("{broken}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_conversations(root)

            path.write_text(
                '{"messages":[{"role":"assistant","content":"Risposta"},'
                '{"role":"user","content":"Domanda"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "role 'user'"):
                load_conversations(root)

    def test_rejects_unpaired_or_empty_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "incomplete.jsonl"
            path.write_text(
                '{"messages":[{"role":"user","content":"Ciao"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pairs"):
                load_conversations(root)

            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No non-empty conversations"):
                load_conversations(root)


if __name__ == "__main__":
    unittest.main()
