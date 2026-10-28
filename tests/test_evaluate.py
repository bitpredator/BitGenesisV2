import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from bitgenesis.evaluate import (
    EvaluationCase,
    evaluate_cases,
    load_evaluation_cases,
)


class LoadEvaluationCasesTests(unittest.TestCase):
    def test_loads_fixed_evaluation_cases(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "evaluation"
            / "prompts.jsonl"
        )

        cases = load_evaluation_cases(path)

        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertTrue(all(case.criteria for case in cases))

    def test_rejects_duplicate_ids_and_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompts.jsonl"
            case = {"id": "same", "prompt": "Ciao?", "criteria": ["Saluta"]}
            path.write_text(
                json.dumps(case) + "\n" + json.dumps(case) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_evaluation_cases(path)

            path.write_text("{invalid}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_evaluation_cases(path)


class EvaluateCasesTests(unittest.TestCase):
    def test_records_responses_and_flags_empty_or_internal_markers(self) -> None:
        chat = Mock()
        chat.reply.side_effect = ("Risposta valida", "", "Risposta <|assistant|>")
        cases = [
            EvaluationCase(str(index), "Domanda?", ("Criterio",))
            for index in range(3)
        ]

        results = evaluate_cases(
            chat,
            cases,
            model="checkpoint.pt",
            seed=42,
            temperature=0.7,
            max_new_tokens=80,
        )

        self.assertEqual(
            [(result.non_empty, result.no_internal_markers) for result in results],
            [(True, True), (False, True), (True, False)],
        )
        self.assertTrue(all(result.seed == 42 for result in results))
        self.assertEqual(results[0].model, "checkpoint.pt")
        self.assertIsNone(results[0].criteria[0].passed)
        self.assertEqual(chat.reply.call_count, 3)


if __name__ == "__main__":
    unittest.main()
