"""Load BitGenesis settings from a TOML configuration file."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 80
    temperature: float = 1.0


@dataclass(frozen=True)
class EvaluationSettings:
    max_new_tokens: int = 160
    temperature: float = 0.7


@dataclass(frozen=True)
class ModelSettings:
    vocab_size: int = 1000
    context_length: int = 128
    embedding_dim: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class Config:
    language: str
    datasets_dir: Path
    models_dir: Path
    generation: GenerationSettings
    evaluation: EvaluationSettings
    model: ModelSettings


def _required_string(section: dict[str, Any], section_name: str, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration value '{section_name}.{key}' must be a non-empty string")
    return value


def _optional_table(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw_config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a table")
    return value


def _positive_integer(
    section: dict[str, Any],
    section_name: str,
    key: str,
    default: int,
    *,
    allow_zero: bool = False,
) -> int:
    value = section.get(key, default)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(
            f"Configuration value '{section_name}.{key}' must be a {qualifier} integer"
        )
    return value


def _number(
    section: dict[str, Any],
    section_name: str,
    key: str,
    default: float,
) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Configuration value '{section_name}.{key}' must be a finite number"
        )
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(
            f"Configuration value '{section_name}.{key}' must be a finite number"
        ) from error
    if not math.isfinite(converted):
        raise ValueError(
            f"Configuration value '{section_name}.{key}' must be a finite number"
        )
    return converted


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
    generation = _optional_table(raw_config, "generation")
    evaluation = _optional_table(raw_config, "evaluation")
    model = _optional_table(raw_config, "model")

    generation_settings = GenerationSettings(
        max_new_tokens=_positive_integer(
            generation, "generation", "max_new_tokens", 80, allow_zero=True
        ),
        temperature=_number(generation, "generation", "temperature", 1.0),
    )
    evaluation_settings = EvaluationSettings(
        max_new_tokens=_positive_integer(
            evaluation, "evaluation", "max_new_tokens", 160
        ),
        temperature=_number(evaluation, "evaluation", "temperature", 0.7),
    )
    model_settings = ModelSettings(
        vocab_size=_positive_integer(model, "model", "vocab_size", 1000),
        context_length=_positive_integer(model, "model", "context_length", 128),
        embedding_dim=_positive_integer(model, "model", "embedding_dim", 64),
        heads=_positive_integer(model, "model", "heads", 4),
        layers=_positive_integer(model, "model", "layers", 2),
        dropout=_number(model, "model", "dropout", 0.1),
    )
    if model_settings.vocab_size < 256:
        raise ValueError(
            "Configuration value 'model.vocab_size' must be at least 256"
        )
    if model_settings.embedding_dim % model_settings.heads:
        raise ValueError(
            "Configuration value 'model.embedding_dim' must be divisible by "
            "'model.heads'"
        )
    if not 0 <= model_settings.dropout < 1:
        raise ValueError(
            "Configuration value 'model.dropout' must be greater than or equal "
            "to 0 and less than 1"
        )
    if generation_settings.temperature <= 0:
        raise ValueError(
            "Configuration value 'generation.temperature' must be greater than zero"
        )
    if evaluation_settings.temperature <= 0:
        raise ValueError(
            "Configuration value 'evaluation.temperature' must be greater than zero"
        )

    base_dir = config_path.resolve().parent
    return Config(
        language=_required_string(app, "app", "language"),
        datasets_dir=base_dir / _required_string(paths, "paths", "datasets"),
        models_dir=base_dir / _required_string(paths, "paths", "models"),
        generation=generation_settings,
        evaluation=evaluation_settings,
        model=model_settings,
    )
