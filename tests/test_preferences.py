import tempfile
import unittest
from pathlib import Path

from bitgenesis.preferences import (
    detect_work_preference,
    read_preferences,
    update_preferences,
)


class WorkPreferenceTests(unittest.TestCase):
    def test_detects_explicit_italian_and_english_work_preferences(self) -> None:
        self.assertEqual(
            detect_work_preference(
                "Preferisco risposte brevi e passaggi numerati."
            ),
            "Preferisco risposte brevi e passaggi numerati.",
        )
        self.assertEqual(
            detect_work_preference("Preferisco modifiche minime."),
            "Preferisco modifiche minime.",
        )
        self.assertEqual(
            detect_work_preference("From now on, use short answers."),
            "From now on, use short answers.",
        )

    def test_ignores_ambiguous_or_unrelated_messages(self) -> None:
        self.assertIsNone(detect_work_preference("Rispondi in modo conciso."))
        self.assertIsNone(detect_work_preference("Preferisco il caffè."))
        self.assertIsNone(detect_work_preference("Preferisco risposte concise." * 60))

    def test_appends_unique_preferences_and_replaces_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Preferenze.md"

            self.assertTrue(update_preferences(path, "  Risposte brevi  "))
            self.assertFalse(update_preferences(path, "risposte brevi"))
            self.assertTrue(update_preferences(path, "Passaggi numerati"))
            self.assertEqual(
                read_preferences(path),
                "- Risposte brevi\n- Passaggi numerati",
            )

            self.assertTrue(
                update_preferences(
                    path,
                    "Solo esempi essenziali",
                    replace=True,
                )
            )
            self.assertEqual(read_preferences(path), "- Solo esempi essenziali")
            self.assertFalse(path.with_name(".Preferenze.md.tmp").exists())

    def test_rejects_empty_preference_without_creating_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Preferenze.md"

            with self.assertRaisesRegex(ValueError, "non possono essere vuote"):
                update_preferences(path, " \n ")

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
