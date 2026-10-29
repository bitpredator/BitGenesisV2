import tempfile
import unittest
from pathlib import Path

from bitgenesis.config import load_config


class LoadConfigTests(unittest.TestCase):
    def test_loads_default_configuration(self) -> None:
        config = load_config()

        self.assertEqual(config.language, "it")
        self.assertEqual(config.datasets_dir, Path("data/datasets").resolve())
        self.assertEqual(config.models_dir, Path("data/models").resolve())
        self.assertEqual(config.generation.max_new_tokens, 80)
        self.assertEqual(config.generation.temperature, 1.0)
        self.assertEqual(config.evaluation.max_new_tokens, 160)
        self.assertEqual(config.evaluation.temperature, 0.7)
        self.assertEqual(config.model.vocab_size, 1000)
        self.assertEqual(config.model.context_length, 128)
        self.assertEqual(config.model.embedding_dim, 64)
        self.assertEqual(config.model.heads, 4)
        self.assertEqual(config.model.layers, 2)
        self.assertEqual(config.model.dropout, 0.1)

    def test_resolves_custom_relative_paths_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom.toml"
            config_path.write_text(
                '[app]\nlanguage = "it"\n\n[paths]\ndatasets = "corpus"\nmodels = "weights"\n',
                encoding="utf-8",
            )

            config = load_config(config_path)

            base_dir = config_path.resolve().parent
            self.assertEqual(config.datasets_dir, base_dir / "corpus")
            self.assertEqual(config.models_dir, base_dir / "weights")
            self.assertEqual(config.model.context_length, 128)

    def test_loads_generation_and_model_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom.toml"
            config_path.write_text(
                '[app]\nlanguage = "it"\n\n[paths]\ndatasets = "data"\nmodels = "models"\n\n'
                '[generation]\nmax_new_tokens = 64\ntemperature = 0.8\n\n'
                '[evaluation]\nmax_new_tokens = 120\ntemperature = 0.6\n\n'
                '[model]\nvocab_size = 512\ncontext_length = 96\n'
                'embedding_dim = 48\nheads = 4\nlayers = 3\ndropout = 0.2\n',
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.generation.max_new_tokens, 64)
        self.assertEqual(config.generation.temperature, 0.8)
        self.assertEqual(config.evaluation.max_new_tokens, 120)
        self.assertEqual(config.evaluation.temperature, 0.6)
        self.assertEqual(config.model.vocab_size, 512)
        self.assertEqual(config.model.context_length, 96)
        self.assertEqual(config.model.embedding_dim, 48)
        self.assertEqual(config.model.heads, 4)
        self.assertEqual(config.model.layers, 3)
        self.assertEqual(config.model.dropout, 0.2)

    def test_rejects_invalid_generation_and_model_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.toml"
            base_config = (
                '[app]\nlanguage = "it"\n\n[paths]\ndatasets = "data"\nmodels = "models"\n'
            )
            invalid_settings = (
                ("[generation]\ntemperature = 0\n", "generation.temperature"),
                ("[generation]\nmax_new_tokens = -1\n", "generation.max_new_tokens"),
                ("[model]\nvocab_size = 255\n", "model.vocab_size"),
                (
                    "[model]\nembedding_dim = 10\nheads = 4\n",
                    "model.embedding_dim",
                ),
                ("[model]\ndropout = 1.0\n", "model.dropout"),
            )
            for settings, error_message in invalid_settings:
                with self.subTest(settings=settings):
                    config_path.write_text(
                        base_config + "\n" + settings,
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, error_message):
                        load_config(config_path)

    def test_rejects_empty_language(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.toml"
            config_path.write_text(
                '[app]\nlanguage = ""\n\n[paths]\ndatasets = "data"\nmodels = "models"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "app.language"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
