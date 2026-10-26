"""Load BitGenesis settings from a TOML configuration file."""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class Config:
    language: str
    datasets_dir: Path
    models_dir: Path


def _required_string(section: dict[str, Any], section_name: str, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration value '{section_name}.{key}' must be a non-empty string")
    return value


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration; relative data paths are based on the TOML file's directory."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.toml"
    else:
        config_path = Path(config_path)

    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)

    app = raw_config.get("app")
    paths = raw_config.get("paths")
    if not isinstance(app, dict):
        raise ValueError("Configuration section 'app' must be a table")
    if not isinstance(paths, dict):
        raise ValueError("Configuration section 'paths' must be a table")

    base_dir = config_path.resolve().parent
    return Config(
        language=_required_string(app, "app", "language"),
        datasets_dir=base_dir / _required_string(paths, "paths", "datasets"),
        models_dir=base_dir / _required_string(paths, "paths", "models"),
    )
