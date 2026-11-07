# BitGenesis

BitGenesis is an AI project in its early stages of development.

This repository is the starting point for building the project. Its features, technical design, and setup instructions will be documented here as they take shape.

## Initial goal

BitGenesis is planned with these goals:

- Start as a Python project compatible with Windows.
- Support Italian and English input, replying in the detected language and
  using English when the language is unclear; responses remain experimental.
- Develop and train its own model from scratch, running locally without externally hosted AI services.
- Eventually become a Windows application.

These goals are under active development.

## Configuration

The initial configuration is stored in [`config.toml`](config.toml). It sets
Italian as the default language, local dataset and model directories, text
generation and evaluation defaults, and the model architecture used by
training. Command-line options override these defaults.

See [install.md](install.md) for Windows setup and verification instructions.
That guide also describes the currently known software and hardware
requirements; RAM and storage recommendations are pending reproducible
measurements.

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

The interface recognizes Italian and English messages and asks the model to
reply in the same language. If a message is too short or its language is not
recognized, it uses English. This is a lightweight language heuristic, so
verify the response language when needed. The chat exits when you type `esci`
(or `exit`, `quit`, or `uscita`). After you train a conversation model, it
loads the local checkpoint and tokenizer, uses previous turns in the current
session as context, and generates replies. Type `/compact` to preview a locally
generated summary of older turns; it replaces the in-memory history only after
you confirm. Choose generation effort with `/impegno auto|minimo|medio|alto`;
the `--effort` option sets the starting level. Without a trained conversation
model, the interface explains how to train one.

Choose between two profiles in the same chat with `/agente chat` or
`/agente coding`; `/agente` displays the active profile. Chat is the default.
The coding profile guides responses toward understanding, writing, fixing, and
testing code, while the chat profile is for general conversation. Both
profiles share the same local model, conversation history, project
`Genesis.md` instructions, and user `Preferenze.md`; they are not separately
trained models, and their capabilities depend on the training data.
For explicit requests to introduce itself or describe its capabilities, both
interfaces use a short factual response instead of the experimental model.
Standalone greetings such as `Ciao` also get a short, deterministic response;
other requests continue to use the trained model.
For programming requests, Coding follows the project's existing conventions,
prefers focused changes, asks for essential missing requirements, and is
expected to be explicit about assumptions and tests it has not run.

Project instructions can be stored in `Genesis.md` in the directory where
BitGenesis is started. Use `/genesis aggiungi <testo>` to append instructions,
`/genesis sostituisci <testo>` to replace them, and `/genesis mostra` to review
the current text. The file is read for each reply and included in the chat
context; BitGenesis changes it only through these explicit commands, not by
guessing from ordinary conversation. The experimental model may not reliably
follow the instructions.

BitGenesis saves explicit work preferences separately in `Preferenze.md` in
the directory where it was started. Statements such as “Preferisco risposte
brevi e passaggi numerati” are recognized and saved automatically; ambiguous
messages are not used to infer preferences. Use `/preferenze mostra`,
`/preferenze aggiungi <testo>`, and `/preferenze sostituisci <testo>` in the
terminal, or the Preferenze di lavoro panel in the web interface to review and
edit them. The file is ignored by Git because it may contain personal
preferences.

The chat can also build a local family tree from explicit relationship
statements in Italian or English. Each accepted statement updates
`AlberoGenealogico.json` and its generation-ordered
`AlberoGenealogico.svg`; the web interface displays the graphic as it changes.
Siblings and spouses are placed at the same generation and shown with separate connectors.
Use its form to add either relation or remove an incorrect link. In the terminal,
use `/albero mostra`, `/albero aggiungi <genitore> -> <figlio>`, or
`/albero rimuovi <genitore> -> <figlio>`; sibling pairs can be managed with
`/albero aggiungi fratelli <nome> -> <nome>` and
`/albero rimuovi fratelli <nome> -> <nome>`, and spouses with
`/albero aggiungi coniugi <nome> -> <nome>` and
`/albero rimuovi coniugi <nome> -> <nome>`. BitGenesis only records
relationships stated explicitly and rejects links that would create a cycle.
These personal files are ignored by Git.

In local mode BitGenesis addresses the person using it as `Utente` in Italian
and `User` in English; there is no local nickname setting. Choosing a personal
nickname is reserved for a future official server, where it can be associated
with a verified account.

## Use the local web interface

If you prefer not to use the interactive terminal, start the browser
interface from the repository folder:

```console
python -m bitgenesis --web
```

BitGenesis opens its local page in your browser. The interface lets you chat
with the local model, choose the chat or coding profile and generation effort,
manage `Genesis.md` and `Preferenze.md`, and start or monitor conversational
training from the included `.jsonl` data. Training runs on the computer and
may take time.
The web server listens only on `127.0.0.1`; it is not available to other
devices on the network. Keep the terminal window open while using the page;
press `Ctrl+C` there to stop the server. Use `--port 8766` to select another
local port if the default port is busy.

The chat and its history are kept in memory for the current server session.
Closing or restarting the server clears the conversation. Model files and
project instructions and user preferences remain on the computer.

To compare the effort levels on a small, fixed prompt set, train a conversation
model and run:

```console
python -m bitgenesis.evaluate --compare-effort --max-new-tokens 80
```

The benchmark currently covers these requests and Auto's expected selection:

| Case | Prompt | Auto level |
|---|---|---|
| Greeting | `Ciao!` | Minimum |
| Simple question | `Che cos'è BitGenesis?` | Medium |
| Explanation | `Spiegami in breve cos'è BitGenesis.` | High |
| Multi-step request | Compare local and online models, then suggest two improvements for BitGenesis. | High |

These expected levels test the current rule-based classifier; they are not
measured model-quality scores. The automated tests verify level selection,
token budgets, and report fields:

```console
python -m unittest discover -s tests -p "test_evaluate.py" -v
```

The JSONL report records each response, selected level, generated token count,
and generation time. Review response quality manually using the criteria in
the report; timing and length alone do not measure correctness.

## Bilingual conversation datasets

The five included conversation datasets have English counterparts ending in
`_inglese.jsonl`; the original Italian files are unchanged. By default,
conversation training loads all JSONL files under `data/datasets/`, so it
includes both languages. This adds bilingual training examples but does not
guarantee that the experimental model will consistently understand or answer
in either language. The configuration still specifies Italian as its default;
chat replies follow message-language detection and use English when unclear.
BitGenesis command-training datasets are grouped under
`data/datasets/commands/`.
The Italian and English conversation sets also include paired examples that
explain calculations, code fixes, testing, planning, and comparisons in steps.
The three Italian stories are also available as complete assistant responses
in `conversazione_italiano.jsonl`; they do not have English counterparts.
Story-writing requests receive a creative-writing instruction that asks the
model to make an original story from vocabulary and ideas learned from its
training stories, while respecting the requested details. The automatic effort
setting gives story requests a larger response budget. The model learns from a
snapshot of the data during training; after adding or changing examples, retrain
it with `python -m bitgenesis.train --data-format conversations --steps 1000`
before expecting the checkpoint to learn them.

## Learn Python programming

The included Italian and English conversation lessons cover introductory
Python and practical topics: variables and expressions, conditionals, loops,
lists, dictionaries, functions, type annotations, exceptions, files and JSON,
modules, virtual environments, classes, readable code, and `unittest`.
The examples target the project's Python 3.11+ baseline and are original
learning material, not a substitute for reviewing and running code. The
general conversation corpora also include examples for clarification,
concise explanations, faithful summaries, and acknowledging uncertainty.
These original examples broaden training coverage but do not make the small
from-scratch model a reliable general assistant. Review the expanded default
prompt set with `python -m bitgenesis.evaluate` after training. Train the
lessons together with the other included conversation examples:

```console
python -m bitgenesis.train --data-format conversations --steps 1000
```

Training uses CUDA automatically when a compatible GPU is available; force CPU
with `--device cpu`, or require CUDA with `--device cuda`. The PyTorch CUDA
allocator is capped at 3 GiB, leaving a 1 GiB margin toward a 4 GiB target
for the CUDA context and other overhead. This cap applies to allocations managed
by PyTorch, not other GPU processes or all driver memory, so total VRAM use can
still vary. If CUDA is unavailable, `auto` reports that it is using the CPU.

Training periodically saves a resumable checkpoint to
`data/models/bitgenesis_training.pt`; adjust the frequency with
`--checkpoint-interval`. It also saves `bitgenesis_best.pt` whenever the
measured validation loss improves and uses that best-validation checkpoint
for inference as `bitgenesis.pt`. The metrics report records the selected
step and validation loss. To continue up to a total of 2000 steps, run:

```console
python -m bitgenesis.train --data-format conversations --steps 2000 --resume
```

Resuming requires the same encoded training data and training/model settings.
The separate `bitgenesis.pt` remains the inference checkpoint. `--resume`
prefers the resumable checkpoint and falls back to `bitgenesis.pt`. Older
model-only checkpoints can be used as a starting point with `--resume-from`,
but resume with a fresh optimizer and step count.

After training, review programming explanations across the covered topics
with the separate prompt set:

```console
python -m bitgenesis.evaluate --cases data/evaluation/programming_prompts.jsonl --output data/models/programming_results.jsonl
```

This is a small experimental dataset, not evidence that the model can reliably
write or run programs. Review the generated explanations and code manually.

## Local text data and tokenizer

Place UTF-8 `.txt` files in `data/datasets/` (subdirectories are read too). Each
file is treated as one document; files are loaded in a stable path order. Invalid
UTF-8 files and a missing or empty corpus are reported as errors. Keep only
training material you have the right to use.

The Italian text corpus includes three original stories in
`data/datasets/it-stories/` (`IT-STORY-0001.txt` through `IT-STORY-0003.txt`).
Train on these and any other `.txt` documents in the dataset directory with the
default text format:

```console
python -m bitgenesis.train --data-format text --steps 1000
```

Text and conversation training are separate: `--data-format conversations`
continues to read only `.jsonl` files, while `--data-format text` reads only
`.txt` files.

The corpus utilities support a reproducible document-level training/validation
split, keeping each document in one split. The standard-library BPE tokenizer
learns byte merges from training documents. Its 256 base tokens cover every
UTF-8 input, including characters absent from the training corpus. `vocab_size`
sets the maximum number of byte-level tokens (minimum 256); the tokenizer also
reserves three conversation control tokens. The learned byte-level vocabulary
can be smaller when the corpus has too few distinct merges. Generation
constrains sampled tokens to valid UTF-8. The tokenizer can be saved locally
under `data/models/`, which is ignored by Git:

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
the optional PyTorch setup, structured conversation data format, and commands
for training and response generation. Conversational training calculates the
loss only for assistant response tokens and response-end markers; user messages
and role markers provide context without contributing to the loss. Plain-text
training continues to use all next-token targets. Training saves its run
configuration and sampled train/validation loss history to
`data/models/training_metrics.json`; use `--metrics-output` to choose another
path. These losses help track experiments but do not measure response quality.

## Status

Under development. Local corpus loading, splitting, byte-level BPE
tokenization, experimental Transformer training, and interactive chat with a
local conversation checkpoint are available. The small prototype is not a
capable or reliable conversational assistant.

## License

BitGenesis is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Bug reports, feature requests, and contribution proposals can be submitted
through the [GitHub issue templates](.github/ISSUE_TEMPLATE). See
[CONTRIBUTING.md](.github/CONTRIBUTING.md) for development and pull request
guidelines. For security concerns, see [Security.md](Security.md) and report
vulnerabilities privately rather than in a public issue.
