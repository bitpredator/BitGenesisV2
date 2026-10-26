# BitGenesis

BitGenesis is an AI project in its early stages of development.

This repository is the starting point for building the project. Its features, technical design, and setup instructions will be documented here as they take shape.

## Initial goal

BitGenesis is planned with these goals:

- Start as a Python project compatible with Windows.
- Understand Italian input and respond in Italian.
- Develop and train its own model from scratch, running locally without externally hosted AI services.
- Eventually become a Windows application.

These are development goals; they have not been implemented yet.

## Configuration

The initial configuration is stored in [`config.toml`](config.toml). It sets Italian as the default language and defines local directories for datasets and model files.

The configuration loader requires Python 3.11 or newer and uses only the Python standard library:

```python
from bitgenesis.config import load_config

config = load_config()
print(config.language)
```

Relative paths in `config.toml` are resolved from the directory containing that file.
Local secrets, virtual environments, datasets, and trained model files are excluded by `.gitignore`.

## Status

Under development. There is no released version or documented installation process yet.

## License

BitGenesis is licensed under the MIT License. See [LICENSE](LICENSE) for details.
