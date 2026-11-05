import json
import tempfile
import unittest
from pathlib import Path

from bitgenesis.retrieval import LocalKnowledgeBase
from bitgenesis.tokenizer import BPETokenizer


class LocalKnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.profile = (
            "Bitpredator è il creatore di BitGenesis. La presentazione "
            "descrive il suo interesse per informatica, Python e gaming."
        )
        (self.root / "profile.jsonl").write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Chi è Bitpredator?"},
                        {"role": "assistant", "content": self.profile},
                    ]
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.root / "notes.TXT").write_text(
            "Python permette di scrivere programmi.\n\n"
            "La biblioteca apre alle nove.",
            encoding="utf-8",
        )
        self.knowledge_base = LocalKnowledgeBase.from_directory(self.root)

    def test_search_ranks_matching_conversation_and_text_sources(self) -> None:
        matches = self.knowledge_base.search(
            "Quali sono gli interessi Python e informatici di Bitpredator?"
        )

        self.assertTrue(matches)
        self.assertIn("Python", matches[0].text)
        self.assertIn("informatica", matches[0].text)
        self.assertEqual(matches[0].source, "profile.jsonl:1#1")
        self.assertTrue(
            any(match.source == "notes.TXT:p1" for match in matches)
        )

    def test_search_returns_no_results_without_matching_terms(self) -> None:
        self.assertEqual(
            self.knowledge_base.search("meteo pianeta marte"),
            [],
        )

    def test_answers_strong_matches_with_a_cited_source_quote(self) -> None:
        answer = self.knowledge_base.answer_from_sources(
            "Quali interessi Python ha Bitpredator?",
            language="it",
        )

        self.assertIsNotNone(answer)
        self.assertIn("[profile.jsonl:1#1]", answer or "")
        self.assertIn("Python", answer or "")
        self.assertIsNone(
            self.knowledge_base.answer_from_sources(
                "Bitpredator?",
                language="it",
            )
        )
        self.assertIsNone(
            self.knowledge_base.answer_from_sources(
                "Dove si trova Marte oggi?",
                language="it",
            )
        )

    def test_prefers_the_top_ranked_relevant_passage_over_generic_overlap(self) -> None:
        knowledge_base = LocalKnowledgeBase(
            [
                (
                    "Bitpredator è il creatore di BitGenesis e ama "
                    "l'informatica, Python e il gaming.",
                    "profile.txt:p1",
                    "Chi è Bitpredator? Bitpredator è il creatore di "
                    "BitGenesis e ama l'informatica, Python e il gaming.",
                ),
                (
                    "Cita competenze di reverse engineering, OSINT, "
                    "sviluppo software, Python e sicurezza.",
                    "skills.jsonl:1#1",
                    "Di che cosa si occupa Bitpredator in ambito "
                    "informatico? Cita competenze di reverse engineering, "
                    "OSINT, sviluppo software, Python e sicurezza.",
                ),
            ]
        )

        answer = knowledge_base.answer_from_sources(
            "Quali sono le competenze informatiche di Bitpredator?",
            language="it",
        )

        self.assertIsNotNone(answer)
        self.assertIn("[skills.jsonl:1#1]", answer or "")
        self.assertIn("reverse engineering", answer or "")

    def test_augments_prompt_with_source_and_fits_context_window(self) -> None:
        tokenizer = BPETokenizer.train([self.profile], vocab_size=256)
        prompt = "Quali interessi ha Bitpredator?"
        history = [("Regola", "Chiama l'utente Utente.")]

        augmented = self.knowledge_base.augment_prompt(
            "Quali interessi ha Bitpredator?",
            prompt,
            history=history,
            tokenizer=tokenizer,
            context_length=2048,
            language="it",
        )

        self.assertIn("[profile.jsonl:1#1]", augmented)
        self.assertIn("Bitpredator è il creatore di BitGenesis.", augmented)
        self.assertLessEqual(
            len(tokenizer.encode_chat_prompt(augmented, history)),
            2048,
        )

    def test_keeps_original_prompt_when_no_passage_fits(self) -> None:
        tokenizer = BPETokenizer.train([self.profile], vocab_size=256)
        prompt = "Domanda Bitpredator"

        augmented = self.knowledge_base.augment_prompt(
            "Bitpredator",
            prompt,
            history=[],
            tokenizer=tokenizer,
            context_length=1,
            language="it",
        )

        self.assertEqual(augmented, prompt)

    def test_rejects_missing_corpus_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            LocalKnowledgeBase.from_directory(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
