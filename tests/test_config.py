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

    def test_resolves_custom_relative_paths_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom.toml"
            config_path.write_text(
                '[app]\nlanguage = "it"\n\n[paths]\ndatasets = "corpus"\nmodels = "weights"\n',
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.datasets_dir, Path(temp_dir) / "corpus")
            self.assertEqual(config.models_dir, Path(temp_dir) / "weights")

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
