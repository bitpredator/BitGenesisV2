import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from bitgenesis.evaluate import (
    EvaluationCase,
    benchmark_efforts,
    evaluate_cases,
    load_evaluation_cases,
)
from bitgenesis.effort import choose_effort


class LoadEvaluationCasesTests(unittest.TestCase):
    def test_loads_fixed_evaluation_cases(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "evaluation"
            / "prompts.jsonl"
        )

        cases = load_evaluation_cases(path)

        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertTrue(all(case.criteria for case in cases))
        self.assertIn("license", {case.id for case in cases})
        self.assertIn("italian-origin", {case.id for case in cases})

    def test_effort_benchmark_cases_cover_auto_levels(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "evaluation"
            / "effort_prompts.jsonl"
        )

        cases = load_evaluation_cases(path)

        self.assertEqual(len(cases), 4)
        self.assertEqual(
            [choose_effort(case.prompt, "auto") for case in cases],
            ["minimo", "medio", "alto", "alto"],
        )

    def test_loads_programming_basics_evaluation_cases(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "evaluation"
            / "programming_prompts.jsonl"
        )

        cases = load_evaluation_cases(path)

        self.assertEqual(len(cases), 4)
        self.assertIn("python-print", {case.id for case in cases})
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

    def test_benchmarks_all_effort_levels_and_records_tokens_and_duration(self) -> None:
        chat = Mock()
        chat.reply.return_value = "Ciao"
        chat.tokenizer.encode.return_value = [1, 2]
        case = EvaluationCase("greeting", "Ciao!", ("Saluta brevemente",))

        with (
            patch(
                "bitgenesis.evaluate.time.perf_counter",
                side_effect=(1.0, 1.25, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0),
            ),
            patch("bitgenesis.evaluate.torch.manual_seed") as seed,
        ):
            results = benchmark_efforts(
                chat,
                [case],
                model="checkpoint.pt",
                seed=42,
                temperature=0.7,
                max_new_tokens=16,
            )

        self.assertEqual(
            [result.requested_effort for result in results],
            ["auto", "minimo", "medio", "alto"],
        )
        self.assertEqual(
            [result.selected_effort for result in results],
            ["minimo", "minimo", "medio", "alto"],
        )
        self.assertEqual(
            [result.max_new_tokens for result in results],
            [4, 4, 16, 32],
        )
        self.assertEqual(
            [result.generation_seconds for result in results],
            [0.25, 0.5, 1.0, 2.0],
        )
        self.assertTrue(all(result.response_tokens == 2 for result in results))
        self.assertEqual([result.seed for result in results], [42] * 4)
        self.assertEqual(seed.call_args_list, [call(42)] * 4)
        self.assertEqual(chat.reply.call_count, 4)


if __name__ == "__main__":
    unittest.main()
