"""Run BitGenesis in an interactive terminal session."""

from collections.abc import Callable, Sequence
import argparse
import math
from pathlib import Path

from bitgenesis.agents import AGENT_INSTRUCTIONS, AgentName
from bitgenesis.config import load_config
from bitgenesis.effort import (
    EFFORT_LEVELS,
    EffortLevel,
    choose_effort,
    effort_token_budget,
    is_effort_level,
)
from bitgenesis.genesis import read_instructions, update_instructions
from bitgenesis.generate import (
    CompactionError,
    ContextLengthError,
    ConversationChat,
)

_EXIT_COMMANDS = {"esci", "exit", "quit", "uscita"}
_BANNER_WIDTH = 52
_LOCAL_USER_NAME = "Utente"


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
    write_output(
        f"  |{'/genesis gestisce istruzioni permanenti.'.center(_BANNER_WIDTH)}|"
    )
    write_output(f"  |{'/agente chat|coding seleziona il profilo.'.center(_BANNER_WIDTH)}|")
    effort_hint = "/impegno auto|minimo|medio|alto cambia il livello."
    write_output(f"  |{effort_hint.center(_BANNER_WIDTH)}|")
    if model_files_found:
        name_hint = "In locale ti chiamerò Utente."
        write_output(f"  |{name_hint.center(_BANNER_WIDTH)}|")
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
    genesis_path: Path | None = None,
) -> None:
    """Run the Italian terminal chat using a saved local model when available."""
    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    project_genesis_path = (
        Path.cwd() / "Genesis.md" if genesis_path is None else Path(genesis_path)
    )

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
    current_agent: AgentName = "chat"
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
        if command and command[0] == "/agente":
            if len(command) == 1:
                write_output(
                    f"  BitGenesis > Agente attuale: {current_agent}. "
                    "Usa /agente chat oppure /agente coding per cambiarlo."
                )
            elif len(command) == 2 and command[1] in ("chat", "coding"):
                current_agent = command[1]
                write_output(
                    f"  BitGenesis > Agente impostato: {current_agent}."
                )
            else:
                write_output(
                    "  BitGenesis > Agente non valido. Usa "
                    "/agente chat oppure /agente coding."
                )
            continue
        if command and command[0] == "/genesis":
            parts = message.strip().split(maxsplit=2)
            action = parts[1].casefold() if len(parts) > 1 else ""
            if action == "mostra" and len(parts) == 2:
                try:
                    instructions = read_instructions(project_genesis_path)
                except (OSError, UnicodeError) as error:
                    write_output(
                        f"  BitGenesis > Non riesco a leggere "
                        f"{project_genesis_path}: {error}"
                    )
                else:
                    if instructions:
                        write_output(
                            f"  BitGenesis > Istruzioni in "
                            f"{project_genesis_path}:\n{instructions}"
                        )
                    else:
                        write_output(
                            "  BitGenesis > Non ci sono istruzioni permanenti. "
                            "Usa /genesis aggiungi <testo>."
                        )
            elif action in {"aggiungi", "sostituisci"} and len(parts) == 3:
                try:
                    update_instructions(
                        project_genesis_path,
                        parts[2],
                        replace=action == "sostituisci",
                    )
                except (OSError, ValueError) as error:
                    write_output(
                        f"  BitGenesis > Non riesco ad aggiornare "
                        f"{project_genesis_path}: {error}"
                    )
                else:
                    verb = "aggiunte" if action == "aggiungi" else "aggiornate"
                    write_output(
                        f"  BitGenesis > Istruzioni {verb} in "
                        f"{project_genesis_path}."
                    )
            else:
                write_output(
                    "  BitGenesis > Usa /genesis mostra, "
                    "/genesis aggiungi <testo> oppure "
                    "/genesis sostituisci <testo>."
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
            chat_history = list(history)
            try:
                project_instructions = read_instructions(project_genesis_path)
            except (OSError, UnicodeError) as error:
                write_output(
                    f"  BitGenesis > Non riesco a leggere "
                    f"{project_genesis_path}: {error}"
                )
                continue
            prompt_sections = []
            if current_agent == "coding":
                prompt_sections.append(
                    "Istruzioni per l'agente coding:\n"
                    f"{AGENT_INSTRUCTIONS[current_agent]}"
                )
            if project_instructions:
                prompt_sections.append(
                    "Istruzioni permanenti del progetto da Genesis.md:\n"
                    f"{project_instructions}"
                )
            if prompt_sections:
                prompt = (
                    "\n\n".join(prompt_sections)
                    + f"\n\nRichiesta dell'utente: {message.strip()}"
                )
            else:
                prompt = message.strip()
            chat_history.append(
                (
                    f"Regola per questa chat locale: chiama l'utente "
                    f"{_LOCAL_USER_NAME}.",
                    f"In questa chat locale chiamerò l'utente {_LOCAL_USER_NAME}.",
                )
            )
            try:
                response = chat.reply(
                    prompt,
                    chat_history,
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
    parser.add_argument(
        "--web",
        action="store_true",
        help="Apri l'interfaccia web locale invece della chat nel terminale.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Porta dell'interfaccia web locale (default: 8765).",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must not be negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than zero")
    if args.web and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    model_path = config.models_dir / "bitgenesis.pt"
    tokenizer_path = config.models_dir / "tokenizer.json"
    if args.web:
        from bitgenesis.web import run_web_server

        run_web_server(
            config=config,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            effort=args.effort,
            port=args.port,
        )
        return
    run_chat(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        effort=args.effort,
    )


if __name__ == "__main__":
    main()
