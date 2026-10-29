# Installation

BitGenesis currently runs on Windows with Python 3.11 or newer. Configuration,
corpus loading, and tokenization use only the Python standard library. Training
and generation require the optional PyTorch dependency.

## Set up the project

Open PowerShell in the repository folder and create a virtual environment:

```powershell
py -3 -m venv .venv
```

If the Python launcher is unavailable, use `python -m venv .venv` instead.
Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is restricted, you can run the environment's Python
directly without changing the execution policy:

```powershell
.\.venv\Scripts\python.exe -m bitgenesis
```

## Install training support

Training and generation need PyTorch. Activate the virtual environment, then
install the dependency:

```powershell
python -m pip install -r requirements-training.txt
```

Training uses the CPU by default, which works without a GPU. The model starts
with random weights; the result depends on the amount, quality, and permitted
use of your local text. Training a tiny example is not expected to produce
useful conversational behavior.

## Verify the installation

Run the test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Start the terminal interface:

```powershell
python -m bitgenesis
```

The terminal shows an ASCII welcome banner with the model-file status and
reminds you that responses are experimental. Type `esci` to exit. After
training a conversation model as described below,
`python -m bitgenesis` loads the local checkpoint and tokenizer and generates
responses during the session. It reports a clear message if either artifact is
missing. Previous user/assistant turns are included in subsequent prompts
during the session; conversation history is not saved after exit.

## Add local training data

Training data stays in `data\datasets\`, which is excluded from Git. Only use
material you have permission to use.

For conversational training, create UTF-8 `.jsonl` files. Each non-empty line
must contain a JSON object with a `messages` array made of alternating,
non-empty user and assistant messages, starting with the user and ending with
the assistant. A line may contain multiple turns:

```json
{"messages":[{"role":"user","content":"Ciao!"},{"role":"assistant","content":"Ciao, come posso aiutarti?"}]}
{"messages":[{"role":"user","content":"Quanto fa 2 + 2?"},{"role":"assistant","content":"Fa 4."},{"role":"user","content":"E 3 + 3?"},{"role":"assistant","content":"Fa 6."}]}
```

Each JSONL line is one conversation, and files are loaded recursively in stable
path order. The training pipeline splits whole conversations before learning
the tokenizer, so validation conversations are not used to train it. At least
two conversations are required for a non-empty validation split. Both splits
must contain more tokens than the configured context length (128 by default).

Plain UTF-8 `.txt` documents remain supported as an alternative. Select the
input format explicitly using `--data-format text` or
`--data-format conversations`.

## Train and generate text

Train with the JSONL conversation format from the repository root:

```powershell
python -m bitgenesis.train --data-format conversations --steps 1000
```

For plain-text `.txt` files instead, run:

```powershell
python -m bitgenesis.train --data-format text --steps 1000
```

Training data and model artifacts are saved locally in the ignored `data`
directory. The default model is intentionally small and CPU-oriented. Adjust
the context length, model dimensions, batch size, and step count for your
hardware and dataset. The training command prints periodic training and
validation losses and saves the run settings and loss history to
`data\models\training_metrics.json`. Use `--metrics-output` to choose a
different report path, and a different `--output-dir` for each run you want to
keep. The losses are sampled estimates for monitoring training, not a measure
of answer quality. The command also prints a short generated reply. The
tokenizer marks user and assistant roles and an end-of-response token;
generation stops at that marker or when the token limit is reached. For
conversational data, the loss is
calculated only on assistant response tokens and end-of-response markers;
user messages and role markers are context only. Plain-text training continues
to calculate loss on every next-token target.

Generate text from a saved checkpoint later:

```powershell
python -m bitgenesis.generate --prompt "Ciao, come stai?"
```

The generated output is an experiment, not evidence that the model understands
Italian or can reliably answer questions.

## Evaluate a conversation model

The fixed Italian evaluation prompts and manual review criteria are in
`data\evaluation\prompts.jsonl`. This file is separate from the training
directory and is not loaded by the training command. The prompts use different
wording from the synthetic training examples to check whether the model can
respond to the same facts without simply repeating a training question.

After training a conversation model, run the evaluation with a fixed seed:

```powershell
python -m bitgenesis.evaluate --seed 42 --output data\models\evaluation.jsonl
```

The report records each prompt, response, review criteria, generation settings,
checkpoint path, and checks for empty replies or internal conversation
markers. In each report row, fill in `passed` as `true` or `false` for every
criterion after reviewing the response. Passing the automatic checks alone
does not mean its factual answer is correct. Keep reports out of the training
directory.

To compare two checkpoints, keep their model and tokenizer files in separate
directories and run the same cases, seed, temperature, and token limit for
each. Give each run a different `--output` path and compare the saved reports.
