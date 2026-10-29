"""Run BitGenesis in an interactive terminal session."""

from collections.abc import Callable, Sequence
import argparse
import math
from pathlib import Path

from bitgenesis.config import load_config
from bitgenesis.effort import (
    EFFORT_LEVELS,
    EffortLevel,
    choose_effort,
    effort_token_budget,
    is_effort_level,
)
from bitgenesis.generate import (
    CompactionError,
    ContextLengthError,
    ConversationChat,
)

_EXIT_COMMANDS = {"esci", "exit", "quit", "uscita"}
_BANNER_WIDTH = 52


def _write_welcome(
    write_output: Callable[[str], None],
    model_files_found: bool,
) -> None:
    border = f"  +{'-' * _BANNER_WIDTH}+"
    status = (
        "File modello: presenti"
        if model_files_found
        else "File modello: non trovati"
    )
    write_output("")
    write_output(border)
    write_output(f"  |{'BITGENESIS'.center(_BANNER_WIDTH)}|")
    write_output(
        f"  |{'Assistente IA sperimentale in italiano'.center(_BANNER_WIDTH)}|"
    )
    write_output(border)
    write_output(f"  |{status.center(_BANNER_WIDTH)}|")
    write_output(
        f"  |{'Risposte sperimentali: possono essere imprecise.'.center(_BANNER_WIDTH)}|"
    )
    write_output(
        f"  |{'/compact riassume i turni precedenti.'.center(_BANNER_WIDTH)}|"
    )
    effort_hint = "/impegno auto|minimo|medio|alto cambia il livello."
    write_output(f"  |{effort_hint.center(_BANNER_WIDTH)}|")
    write_output(f"  |{'Digita esci per terminare.'.center(_BANNER_WIDTH)}|")
    write_output(border)
    write_output("")


def run_chat(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    effort: EffortLevel = "auto",
) -> None:
    """Run the Italian terminal chat using a saved local model when available."""
    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn

    model_files_found = (
        model_path is not None
        and tokenizer_path is not None
        and model_path.is_file()
        and tokenizer_path.is_file()
    )
    _write_welcome(write_output, model_files_found)
    history: list[tuple[str, str]] = []
    chat: ConversationChat | None = None
    current_effort = effort
    while True:
        try:
            message = read_input("  Tu > ")
        except (EOFError, KeyboardInterrupt):
            write_output("  BitGenesis > A presto!")
            return

        if message.strip().casefold() in _EXIT_COMMANDS:
            write_output("  BitGenesis > A presto!")
            return
        command = message.strip().casefold().split()
        if command and command[0] == "/impegno":
            if len(command) == 1:
                write_output(
                    f"  BitGenesis > Impegno attuale: {current_effort}. "
                    "Usa /impegno auto|minimo|medio|alto per cambiarlo."
                )
            elif len(command) == 2 and is_effort_level(command[1]):
                current_effort = command[1]
                write_output(f"  BitGenesis > Impegno impostato: {current_effort}.")
            else:
                write_output(
                    "  BitGenesis > Livello non valido. Usa "
                    "/impegno auto|minimo|medio|alto."
                )
            continue
        if message.strip().casefold() == "/compact":
            if not history:
                write_output(
                    "  BitGenesis > Non ci sono ancora scambi da compattare."
                )
                continue
            if (
                model_path is None
                or tokenizer_path is None
                or not model_path.is_file()
                or not tokenizer_path.is_file()
            ):
                write_output(
                    "  BitGenesis > Serve un modello conversazionale disponibile "
                    "per creare un riepilogo."
                )
                continue
            if chat is None:
                chat = ConversationChat(model_path, tokenizer_path)
            try:
                summary, compacted_history, kept_turns = chat.compact(history)
            except CompactionError as error:
                write_output(f"  BitGenesis > Compattazione non riuscita: {error}")
                continue

            write_output("  BitGenesis > Riepilogo proposto:")
            for line in summary.splitlines():
                write_output(f"    {line}")
            write_output(
                "  BitGenesis > Il riepilogo potrebbe omettere dettagli. "
                "Applicarlo? (s/n)"
            )
            try:
                confirmation = read_input("  Confermi > ").strip().casefold()
            except (EOFError, KeyboardInterrupt):
                confirmation = ""
            if confirmation in {"s", "si", "sì"}:
                summarized_turns = len(history) - kept_turns
                history[:] = compacted_history
                summarized_label = (
                    "1 scambio riassunto"
                    if summarized_turns == 1
                    else f"{summarized_turns} scambi riassunti"
                )
                write_output(
                    f"  BitGenesis > Conversazione compattata: "
                    f"{summarized_label}, "
                    f"{kept_turns} recenti mantenuti."
                )
            else:
                write_output(
                    "  BitGenesis > Compattazione annullata; cronologia invariata."
                )
            continue
        if message.strip():
            if model_path is None or tokenizer_path is None:
                write_output(
                    "  BitGenesis > Modello non disponibile. Addestrami prima con "
                    "'python -m bitgenesis.train --data-format conversations'."
                )
                continue
            if not model_path.is_file() or not tokenizer_path.is_file():
                write_output(
                    "  BitGenesis > Modello non disponibile. Addestrami prima con "
                    "'python -m bitgenesis.train --data-format conversations'."
                )
                continue
            if chat is None:
                chat = ConversationChat(model_path, tokenizer_path)
            selected_effort = choose_effort(message.strip(), current_effort)
            token_budget = effort_token_budget(max_new_tokens, selected_effort)
            effort_label = (
                f"{selected_effort} (automatico)"
                if current_effort == "auto"
                else selected_effort
            )
            write_output(f"  BitGenesis > Impegno: {effort_label}.")
            try:
                response = chat.reply(
                    message.strip(),
                    history,
                    max_new_tokens=token_budget,
                    temperature=temperature,
                )
            except ContextLengthError as error:
                write_output(f"  BitGenesis > {error}")
                continue
            write_output(f"  BitGenesis > {response or '[risposta vuota]'}")
            history.append((message.strip(), response))


def main(argv: Sequence[str] | None = None) -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Chat with BitGenesis in Italian.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.generation.max_new_tokens,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.generation.temperature,
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default="auto",
        help="Livello di impegno iniziale della chat (default: auto).",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must not be negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than zero")

    model_path = config.models_dir / "bitgenesis.pt"
    tokenizer_path = config.models_dir / "tokenizer.json"
    run_chat(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        effort=args.effort,
    )


if __name__ == "__main__":
    main()
