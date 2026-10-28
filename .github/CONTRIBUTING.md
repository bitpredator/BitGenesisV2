# Contributing to BitGenesis

Thank you for your interest in contributing. BitGenesis is an early-stage
experimental project. For substantial changes, open an issue first to discuss
the goal and approach. For simple fixes, you can submit a pull request directly.

## Ways to contribute

Bug fixes, tests, documentation improvements, reproducible reports, feature
proposals, and code improvements are all welcome. For corpus, prompts, and
evaluation data, only contribute content that you have the right to use and
share.

## Set up the environment

The project requires Python 3.11 or later. From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

PyTorch is optional for tasks that do not involve training or generation. To
install it:

```powershell
python -m pip install -r requirements-training.txt
```

## Pull requests

- Keep changes focused and consistent with existing conventions.
- Add or update tests for changed behavior.
- Run `python -m unittest discover -s tests -v` and report the checks you ran
  in your pull request.
- Update documentation when commands, requirements, or user-visible behavior
  change.
- Explain the problem, solution, and any limitations.
- Do not include credentials, personal information, private corpora,
  checkpoints, or other locally generated files.

Pull requests are reviewed before being merged; submitting a proposal does not
guarantee that it will be accepted or implemented.

## License

The repository code is distributed under the MIT License. By contributing, you
submit changes that may be distributed under the project's license. Make sure
any content or dependency you add can be used and redistributed under its
applicable terms.
