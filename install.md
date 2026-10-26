# Installation

BitGenesis currently runs on Windows with Python 3.11 or newer. Its current
configuration, corpus, and tokenizer features use only the Python standard
library, so no third-party Python packages are required.

## Set up the project

Open PowerShell in the repository folder and create a virtual environment:

```powershell
py -3.11 -m venv .venv
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
contents are local and excluded from Git. Corpus loading, splitting, and
byte-level BPE tokenization are available, but language-model training is not
implemented yet. PyTorch and other third-party packages are therefore not
currently required. Installation steps for model-training dependencies will be
added when that functionality is implemented.
