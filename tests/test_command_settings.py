import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bitgenesis.evaluate import EvaluationCase, benchmark_efforts, main as evaluate_main
from bitgenesis.generate import main as generate_main


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        models_dir=Path("local/models"),
        generation=SimpleNamespace(max_new_tokens=23, temperature=0.65),
        evaluation=SimpleNamespace(max_new_tokens=91, temperature=0.55),
    )


class GenerateCommandSettingsTests(unittest.TestCase):
    def test_uses_config_defaults_and_command_line_overrides(self) -> None:
        with (
            patch("bitgenesis.generate.load_config", return_value=make_config()),
            patch("bitgenesis.generate.generate_text", return_value="Risposta") as generate,
        ):
            generate_main(["--prompt", "Ciao"])
            self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 23)
            self.assertEqual(generate.call_args.kwargs["temperature"], 0.65)

            generate_main(
                [
                    "--prompt",
                    "Ciao",
                    "--max-new-tokens",
                    "42",
                    "--temperature",
                    "0.8",
                ]
            )

        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 42)
        self.assertEqual(generate.call_args.kwargs["temperature"], 0.8)


class EvaluateCommandSettingsTests(unittest.TestCase):
    def test_uses_config_defaults_and_command_line_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            cases_path = Path(temp_dir) / "cases.jsonl"
            with (
                patch("bitgenesis.evaluate.load_config", return_value=make_config()),
                patch(
                    "bitgenesis.evaluate.load_evaluation_cases",
                    return_value=[],
                ),
                patch("bitgenesis.evaluate.ConversationChat"),
                patch("bitgenesis.evaluate.evaluate_cases", return_value=[]) as evaluate,
            ):
                evaluate_main(
                    ["--cases", str(cases_path), "--output", str(output_path)]
                )
                self.assertEqual(evaluate.call_args.kwargs["max_new_tokens"], 91)
                self.assertEqual(evaluate.call_args.kwargs["temperature"], 0.55)

                evaluate_main(
                    [
                        "--cases",
                        str(cases_path),
                        "--output",
                        str(output_path),
                        "--max-new-tokens",
                        "44",
                        "--temperature",
                        "0.9",
                    ]
                )

            self.assertEqual(evaluate.call_args.kwargs["max_new_tokens"], 44)
            self.assertEqual(evaluate.call_args.kwargs["temperature"], 0.9)

    def test_compare_effort_dispatches_to_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "effort.jsonl"
            cases_path = Path(temp_dir) / "cases.jsonl"
            with (
                patch("bitgenesis.evaluate.load_config", return_value=make_config()),
                patch(
                    "bitgenesis.evaluate.load_evaluation_cases",
                    return_value=[
                        EvaluationCase("greeting", "Ciao!", ("Saluta brevemente",))
                    ],
                ),
                patch("bitgenesis.evaluate.ConversationChat") as chat_class,
                patch(
                    "bitgenesis.evaluate.benchmark_efforts",
                    wraps=benchmark_efforts,
                ) as benchmark,
                patch("bitgenesis.evaluate.evaluate_cases") as evaluate,
            ):
                chat_class.return_value.reply.return_value = "Ciao!"
                chat_class.return_value.tokenizer.encode.return_value = [1, 2]
                evaluate_main(
                    [
                        "--compare-effort",
                        "--cases",
                        str(cases_path),
                        "--output",
                        str(output_path),
                    ]
                )

        benchmark.assert_called_once()
        evaluate.assert_not_called()
        self.assertEqual(benchmark.call_args.kwargs["temperature"], 0.55)
        self.assertEqual(benchmark.call_args.kwargs["max_new_tokens"], 91)


if __name__ == "__main__":
    unittest.main()
