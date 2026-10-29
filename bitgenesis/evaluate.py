"""Run a repeatable, human-reviewed evaluation of a local conversation model."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import torch

from bitgenesis.config import load_config
from bitgenesis.generate import ConversationChat

_DEFAULT_CASES = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "prompts.jsonl"
)
_INTERNAL_MARKERS = ("<|user|>", "<|assistant|>", "<|end|>")


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    prompt: str
    criteria: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCriterion:
    description: str
    passed: bool | None = None


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    prompt: str
    criteria: tuple[EvaluationCriterion, ...]
    response: str
    model: str
    seed: int
    temperature: float
    max_new_tokens: int
    non_empty: bool
    no_internal_markers: bool


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load fixed prompt cases, reporting malformed JSONL with its location."""
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid evaluation JSON in {path} at line {line_number}: "
                    f"{error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Evaluation case must be a JSON object ({path} at line "
                    f"{line_number})"
                )
            case_id = record.get("id")
            prompt = record.get("prompt")
            criteria = record.get("criteria")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(
                    f"Evaluation case needs a non-empty id ({path} at line "
                    f"{line_number})"
                )
            if case_id in seen_ids:
                raise ValueError(
                    f"Duplicate evaluation case id {case_id!r} ({path} at line "
                    f"{line_number})"
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Evaluation case {case_id!r} needs a non-empty prompt"
                )
            if (
                not isinstance(criteria, list)
                or not criteria
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in criteria
                )
            ):
                raise ValueError(
                    f"Evaluation case {case_id!r} needs non-empty string criteria"
                )
            seen_ids.add(case_id)
            cases.append(
                EvaluationCase(
                    id=case_id,
                    prompt=prompt,
                    criteria=tuple(criteria),
                )
            )
    if not cases:
        raise ValueError(f"No evaluation cases found in: {path}")
    return cases


def evaluate_cases(
    chat: ConversationChat,
    cases: list[EvaluationCase],
    *,
    model: str,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> list[EvaluationResult]:
    """Generate a response per case and record objective output checks."""
    results: list[EvaluationResult] = []
    for case in cases:
        response = chat.reply(
            case.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results.append(
            EvaluationResult(
                id=case.id,
                prompt=case.prompt,
                criteria=tuple(
                    EvaluationCriterion(description=item) for item in case.criteria
                ),
                response=response,
                model=model,
                seed=seed,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                non_empty=bool(response.strip()),
                no_internal_markers=not any(
                    marker in response for marker in _INTERNAL_MARKERS
                ),
            )
        )
    return results


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Evaluate a local conversation model with fixed Italian prompts."
    )
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument(
        "--model", type=Path, default=config.models_dir / "bitgenesis.pt"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=config.models_dir / "tokenizer.json"
    )
    parser.add_argument(
        "--output", type=Path, default=config.models_dir / "evaluation_results.jsonl"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.evaluation.temperature,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.evaluation.max_new_tokens,
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than zero")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.cases.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the evaluation cases")

    cases = load_evaluation_cases(args.cases)
    torch.manual_seed(args.seed)
    chat = ConversationChat(args.model, args.tokenizer)
    results = evaluate_cases(
        chat,
        cases,
        model=str(args.model),
        seed=args.seed,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    failures = sum(
        not result.non_empty or not result.no_internal_markers for result in results
    )
    print(f"Risultati salvati in: {args.output}")
    print(
        f"Casi: {len(results)}; risposte vuote o con marcatori interni: {failures}. "
        "Valuta manualmente ogni risposta secondo i criteri registrati."
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
