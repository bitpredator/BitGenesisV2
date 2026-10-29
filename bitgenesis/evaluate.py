"""Run a repeatable, human-reviewed evaluation of a local conversation model."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import torch

from bitgenesis.config import load_config
from bitgenesis.effort import (
    EFFORT_LEVELS,
    EffortLevel,
    GenerationEffort,
    choose_effort,
    effort_token_budget,
)
from bitgenesis.generate import ConversationChat

_DEFAULT_CASES = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "prompts.jsonl"
)
_EFFORT_CASES = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evaluation"
    / "effort_prompts.jsonl"
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


@dataclass(frozen=True)
class EffortBenchmarkResult:
    id: str
    prompt: str
    criteria: tuple[EvaluationCriterion, ...]
    requested_effort: EffortLevel
    selected_effort: GenerationEffort
    response: str
    response_tokens: int
    generation_seconds: float
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


def benchmark_efforts(
    chat: ConversationChat,
    cases: list[EvaluationCase],
    *,
    model: str,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> list[EffortBenchmarkResult]:
    """Compare all effort levels on fixed prompts with paired random seeds."""
    results: list[EffortBenchmarkResult] = []
    for case_index, case in enumerate(cases):
        case_seed = seed + case_index
        selected_auto_effort = choose_effort(case.prompt, "auto")
        for requested_effort in EFFORT_LEVELS:
            selected_effort = (
                selected_auto_effort
                if requested_effort == "auto"
                else requested_effort
            )
            token_budget = effort_token_budget(max_new_tokens, selected_effort)
            torch.manual_seed(case_seed)
            started_at = time.perf_counter()
            response = chat.reply(
                case.prompt,
                max_new_tokens=token_budget,
                temperature=temperature,
            )
            generation_seconds = time.perf_counter() - started_at
            results.append(
                EffortBenchmarkResult(
                    id=case.id,
                    prompt=case.prompt,
                    criteria=tuple(
                        EvaluationCriterion(description=item)
                        for item in case.criteria
                    ),
                    requested_effort=requested_effort,
                    selected_effort=selected_effort,
                    response=response,
                    response_tokens=len(chat.tokenizer.encode(response)),
                    generation_seconds=generation_seconds,
                    model=model,
                    seed=case_seed,
                    temperature=temperature,
                    max_new_tokens=token_budget,
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
    parser.add_argument(
        "--compare-effort",
        action="store_true",
        help="Compare auto, minimo, medio, and alto on a small fixed prompt set.",
    )
    parser.add_argument("--cases", type=Path)
    parser.add_argument(
        "--model", type=Path, default=config.models_dir / "bitgenesis.pt"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=config.models_dir / "tokenizer.json"
    )
    parser.add_argument("--output", type=Path)
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
    cases_path = args.cases or (
        _EFFORT_CASES if args.compare_effort else _DEFAULT_CASES
    )
    output_path = args.output or config.models_dir / (
        "effort_benchmark_results.jsonl"
        if args.compare_effort
        else "evaluation_results.jsonl"
    )
    if cases_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite the evaluation cases")

    cases = load_evaluation_cases(cases_path)
    if not args.compare_effort:
        torch.manual_seed(args.seed)
    chat = ConversationChat(args.model, args.tokenizer)
    if args.compare_effort:
        results = benchmark_efforts(
            chat,
            cases,
            model=str(args.model),
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        results = evaluate_cases(
            chat,
            cases,
            model=str(args.model),
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    failures = sum(
        not result.non_empty or not result.no_internal_markers for result in results
    )
    print(f"Risultati salvati in: {output_path}")
    if args.compare_effort:
        print("Livello scelto da Auto per ciascun prompt:")
        for case in cases:
            print(f"{case.id}: {choose_effort(case.prompt, 'auto')}")
        for effort in EFFORT_LEVELS:
            effort_results = [
                result
                for result in results
                if result.requested_effort == effort
            ]
            average_seconds = sum(
                result.generation_seconds for result in effort_results
            ) / len(effort_results)
            average_tokens = sum(
                result.response_tokens for result in effort_results
            ) / len(effort_results)
            print(
                f"{effort}: media {average_tokens:.1f} token risposta, "
                f"{average_seconds:.2f} s per risposta"
            )
    run_label = "Esecuzioni" if args.compare_effort else "Casi"
    print(
        f"{run_label}: {len(results)}; risposte vuote o con marcatori interni: "
        f"{failures}. Valuta manualmente ogni risposta secondo i criteri registrati."
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
