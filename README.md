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

See [install.md](install.md) for Windows setup and verification instructions.

The configuration loader requires Python 3.11 or newer and uses only the Python standard library:

```python
from bitgenesis.config import load_config

config = load_config()
print(config.language)
```

Relative paths in `config.toml` are resolved from the directory containing that file.
Local secrets, virtual environments, datasets, and trained model files are excluded by `.gitignore`.

## Run the terminal interface

With Python 3.11 or newer, start BitGenesis from the repository root:

```console
python -m bitgenesis
```

The initial interface accepts Italian text and exits when you type `esci` (or
`exit`, `quit`, or `uscita`). It does not generate AI responses yet; it clearly
reports that the model has not been trained.

## Local text data and tokenizer

Place UTF-8 `.txt` files in `data/datasets/` (subdirectories are read too). Each
file is treated as one document; files are loaded in a stable path order. Invalid
UTF-8 files and a missing or empty corpus are reported as errors. Keep only
training material you have the right to use.

The corpus utilities support a reproducible document-level training/validation
split, keeping each document in one split. The standard-library BPE tokenizer
learns byte merges from training documents. Its 256 base tokens cover every
UTF-8 input, including characters absent from the training corpus. `vocab_size`
sets the maximum vocabulary size (minimum 256); the learned vocabulary can be
smaller when the corpus has too few distinct merges. Generation constrains
sampled tokens to valid UTF-8. The tokenizer can be saved locally under
`data/models/`, which is ignored by Git:

```python
from bitgenesis.config import load_config
from bitgenesis.corpus import load_corpus, split_corpus
from bitgenesis.tokenizer import BPETokenizer

config = load_config()
corpus = load_corpus(config.datasets_dir)
split = split_corpus(corpus, validation_fraction=0.1, seed=42)
tokenizer = BPETokenizer.train(split.train, vocab_size=1000)
tokenizer.save(config.models_dir / "tokenizer.json")
```

The validation documents are kept separate from tokenizer training. This step
also includes an experimental, small decoder-only Transformer that can be
trained locally from random initialization. See [install.md](install.md) for
the optional PyTorch setup and the commands for training and text generation.

## Status

Under development. The terminal interface is a basic placeholder, not an AI model.
Local corpus loading, splitting, and byte-level BPE tokenization are available;
experimental local Transformer training and text generation are available.
This small prototype is not a capable conversational assistant.

## License

BitGenesis is licensed under the MIT License. See [LICENSE](LICENSE) for details.
