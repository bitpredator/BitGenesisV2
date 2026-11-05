import unittest

from bitgenesis.agents import (
    bitpredator_response,
    build_response_prompt,
    license_response,
    self_introduction,
    simple_greeting,
)
from bitgenesis.language import detect_language, language_instruction


class LanguageDetectionTests(unittest.TestCase):
    def test_detects_italian_and_english_messages(self) -> None:
        cases = (
            ("Come posso organizzare il mio lavoro?", "it"),
            ("Spiegami come funziona questo programma.", "it"),
            ("Potresti aiutarmi con questo?", "it"),
            ("Rendilo più informale.", "it"),
            ("What can I do to organize my work?", "en"),
            ("Ciao", "it"),
            ("Hello", "en"),
        )

        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(detect_language(message), expected)

    def test_defaults_to_english_when_language_is_unclear_or_unsupported(self) -> None:
        for message in ("12345", "Bonjour, comment allez-vous?", "def hello():"):
            with self.subTest(message=message):
                self.assertEqual(detect_language(message), "en")

    def test_instruction_matches_detected_language(self) -> None:
        self.assertIn("rispondi in italiano", language_instruction("it"))
        self.assertIn("reply in English", language_instruction("en"))

    def test_response_prompt_includes_the_detected_language(self) -> None:
        italian_prompt = build_response_prompt(
            "Spiegami il codice.",
            "coding",
            "",
            "it",
        )
        english_prompt = build_response_prompt(
            "Explain this code.",
            "coding",
            "",
            "en",
        )

        self.assertIn("Richiesta dell'utente: Spiegami il codice.", italian_prompt)
        self.assertIn("rispondi in italiano", italian_prompt)
        self.assertIn("User request: Explain this code.", english_prompt)
        self.assertIn("reply in English", english_prompt)
        self.assertIn("Coding agent instructions", english_prompt)

    def test_story_requests_include_corpus_guidance_in_both_languages(self) -> None:
        italian_prompt = build_response_prompt(
            "Raccontami una storia ambientata in montagna.",
            "chat",
            "",
            "it",
        )
        english_prompt = build_response_prompt(
            "Write a story set in the mountains.",
            "chat",
            "",
            "en",
        )
        ordinary_prompt = build_response_prompt(
            "Qual è la differenza tra una storia e un racconto?",
            "chat",
            "",
            "it",
        )

        self.assertIn("parole del corpus", italian_prompt)
        self.assertIn("corpus words", english_prompt)
        self.assertIn("Original story", english_prompt)
        self.assertNotIn("Storia originale:", ordinary_prompt)

    def test_greetings_and_introductions_use_the_detected_language(self) -> None:
        cases = (
            ("Hi", "Hello! I'm BitGenesis. How can I help?"),
            ("Ehi", "Ciao! Sono BitGenesis. Come posso aiutarti?"),
            ("Buon pomeriggio", "Ciao! Sono BitGenesis. Come posso aiutarti?"),
            ("Ciao", "Ciao! Sono BitGenesis. Come posso aiutarti?"),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(simple_greeting(message, "chat"), expected)
        self.assertIn(
            "I'm BitGenesis",
            self_introduction("Who are you?", "chat") or "",
        )
        self.assertIn(
            "Sono BitGenesis",
            self_introduction("Chi sei?", "chat") or "",
        )

    def test_license_questions_get_a_factual_response_in_both_languages(self) -> None:
        self.assertEqual(
            license_response("Con che licenza è distribuito BitGenesis?"),
            "BitGenesis è distribuito con licenza MIT. "
            "Consulta il file LICENSE per il testo completo.",
        )
        self.assertEqual(
            license_response("What license is BitGenesis distributed under?"),
            "BitGenesis is distributed under the MIT License. "
            "See the LICENSE file for the full text.",
        )
        self.assertIsNone(license_response("Come funziona BitGenesis?"))

    def test_bitpredator_profile_questions_get_a_factual_response(self) -> None:
        italian = bitpredator_response("Cosa sai dirmi su Bitpredator?")
        english = bitpredator_response("What do you know about Bitpredator?")

        self.assertIsNotNone(italian)
        self.assertIsNotNone(english)
        for response in (italian, english):
            self.assertIn("https://www.twitch.tv/bitpredator", response or "")
            self.assertIn("https://github.com/bitpredator", response or "")
        self.assertIn("creatore", italian or "")
        self.assertIn("Skull Network Italia", italian or "")
        self.assertIn("verificate", italian or "")
        self.assertIn("creator", english or "")
        self.assertIn("founder and administrator", english or "")
        self.assertIn("independently verified", english or "")
        self.assertIsNone(
            bitpredator_response("Cosa sai dirmi su BitGenesis?")
        )
        self.assertIsNone(
            bitpredator_response("Come funziona il nome Bitpredator?")
        )


if __name__ == "__main__":
    unittest.main()
