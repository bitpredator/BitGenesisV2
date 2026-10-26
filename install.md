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

The current interface acknowledges messages but does not generate AI responses.
Type `esci` to exit.

## Add local training text

Place UTF-8 `.txt` documents in `data\datasets\`. The directory and its
contents are local and excluded from Git. Add at least two non-empty documents
so BitGenesis can keep validation text separate from training text. Each split
must also contain more tokens than the configured context length (128 by
default).

## Train and generate text

Start a training run from the repository root:

```powershell
python -m bitgenesis.train --steps 1000
```

Training data and model artifacts are saved locally in the ignored `data`
directory. The default model is intentionally small and CPU-oriented. Adjust
the context length, model dimensions, batch size, and step count for your
hardware and dataset. The training command prints periodic training and
validation losses and a short generation sample.

Generate text from a saved checkpoint later:

```powershell
python -m bitgenesis.generate --prompt "Ciao, come stai?"
```

The generated output is an experiment, not evidence that the model understands
Italian or can reliably answer questions.
