import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bitgenesis.config import load_config
from bitgenesis.train import main


class TrainCommandTests(unittest.TestCase):
    def test_refuses_metrics_output_that_would_overwrite_training_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "datasets"
            data_dir.mkdir()
            source_path = data_dir / "examples.jsonl"
            source_path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Ciao"},
                            {"role": "assistant", "content": "Salve"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "models"
            base_args = [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--data-format",
                "conversations",
            ]

            for metrics_path in (
                source_path,
                output_dir / "bitgenesis.pt",
                output_dir / "tokenizer.json",
            ):
                with self.subTest(metrics_path=metrics_path):
                    with self.assertRaises(SystemExit) as error:
                        main(
                            [
                                *base_args,
                                "--metrics-output",
                                str(metrics_path),
                            ]
                        )
                    self.assertEqual(error.exception.code, 2)

            self.assertTrue(source_path.read_text(encoding="utf-8").endswith("\n"))

    def test_saves_run_configuration_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "datasets"
            data_dir.mkdir()
            conversations = [
                {
                    "messages": [
                        {"role": "user", "content": "Ciao BitGenesis"},
                        {"role": "assistant", "content": "Salve"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Come stai oggi"},
                        {"role": "assistant", "content": "Sto bene"},
                    ]
                },
            ]
            (data_dir / "examples.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in conversations),
                encoding="utf-8",
            )
            config_path = root / "config.toml"
            config_path.write_text(
                '[app]\nlanguage = "it"\n\n[paths]\ndatasets = "datasets"\n'
                'models = "models"\n\n[generation]\nmax_new_tokens = 5\n'
                'temperature = 0.5\n\n[model]\nvocab_size = 256\n'
                'context_length = 2\nembedding_dim = 4\nheads = 1\n'
                'layers = 1\ndropout = 0.0\n',
                encoding="utf-8",
            )
            output_dir = root / "models"

            with patch(
                "bitgenesis.train.load_config",
                return_value=load_config(config_path),
            ):
                main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(output_dir),
                        "--data-format",
                        "conversations",
                        "--context-length",
                        "1",
                        "--embedding-dim",
                        "8",
                        "--heads",
                        "2",
                        "--steps",
                        "1",
                        "--batch-size",
                        "1",
                        "--eval-interval",
                        "1",
                        "--eval-batches",
                        "1",
                        "--generate-tokens",
                        "3",
                        "--temperature",
                        "0.9",
                        "--prompt",
                        "",
                    ]
                )

            report_path = output_dir / "training_metrics.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["format_version"], 1)
            self.assertEqual(
                report["configuration"]["data_format"], "conversations"
            )
            self.assertEqual(report["configuration"]["seed"], 42)
            self.assertEqual(report["configuration"]["training_examples"], 1)
            self.assertEqual(report["configuration"]["validation_examples"], 1)
            self.assertGreaterEqual(
                report["configuration"]["model"]["vocab_size"], 256
            )
            self.assertLessEqual(report["configuration"]["model"]["vocab_size"], 259)
            self.assertEqual(
                report["configuration"]["model"]["context_length"], 1
            )
            self.assertEqual(
                report["configuration"]["model"]["embedding_dim"], 8
            )
            self.assertEqual(report["configuration"]["model"]["num_heads"], 2)
            self.assertEqual(report["configuration"]["model"]["num_layers"], 1)
            self.assertEqual(report["configuration"]["model"]["dropout"], 0.0)
            self.assertEqual(report["configuration"]["generate_tokens"], 3)
            self.assertEqual(report["configuration"]["temperature"], 0.9)
            self.assertEqual(len(report["history"]), 1)
            self.assertEqual(report["history"][0]["step"], 1.0)
            self.assertTrue((output_dir / "bitgenesis.pt").is_file())
            self.assertTrue((output_dir / "tokenizer.json").is_file())


if __name__ == "__main__":
    unittest.main()
