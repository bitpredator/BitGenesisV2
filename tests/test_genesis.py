import tempfile
import unittest
from pathlib import Path

from bitgenesis.genesis import read_instructions, update_instructions


class GenesisInstructionsTests(unittest.TestCase):
    def test_reads_missing_file_as_empty_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Genesis.md"

            self.assertEqual(read_instructions(path), "")
            self.assertFalse(path.exists())

    def test_appends_and_replaces_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Genesis.md"
            update_instructions(path, "Usa Python 3.11.")
            update_instructions(path, "Scrivi test per le nuove funzioni.")

            self.assertEqual(
                read_instructions(path),
                "Usa Python 3.11.\n\nScrivi test per le nuove funzioni.",
            )
            update_instructions(path, "Rispondi in italiano.", replace=True)

            self.assertEqual(read_instructions(path), "Rispondi in italiano.")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# "))
            self.assertFalse(path.with_name(".Genesis.md.tmp").exists())

    def test_rejects_empty_instructions_without_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Genesis.md"

            with self.assertRaisesRegex(ValueError, "non possono essere vuote"):
                update_instructions(path, " \n ")

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
