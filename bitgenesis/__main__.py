"""Run BitGenesis in an interactive terminal session."""

from collections.abc import Callable
from pathlib import Path

from bitgenesis.config import load_config
from bitgenesis.generate import ConversationChat

_EXIT_COMMANDS = {"esci", "exit", "quit", "uscita"}


def run_chat(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
) -> None:
    """Run the Italian terminal chat using a saved local model when available."""
    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn

    write_output("BitGenesis: Ciao! Scrivimi in italiano. Digita 'esci' per terminare.")
    history: list[tuple[str, str]] = []
    chat: ConversationChat | None = None
    while True:
        try:
            message = read_input("Tu: ")
        except (EOFError, KeyboardInterrupt):
            write_output("BitGenesis: A presto!")
            return

        if message.strip().casefold() in _EXIT_COMMANDS:
            write_output("BitGenesis: A presto!")
            return
        if message.strip():
            if model_path is None or tokenizer_path is None:
                write_output(
                    "BitGenesis: Modello non disponibile. Addestrami prima con "
                    "'python -m bitgenesis.train --data-format conversations'."
                )
                continue
            if not model_path.is_file() or not tokenizer_path.is_file():
                write_output(
                    "BitGenesis: Modello non disponibile. Addestrami prima con "
                    "'python -m bitgenesis.train --data-format conversations'."
                )
                continue
            if chat is None:
                chat = ConversationChat(model_path, tokenizer_path)
            response = chat.reply(
                message.strip(),
                history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            write_output(f"BitGenesis: {response or '[risposta vuota]'}")
            history.append((message.strip(), response))


def main() -> None:
    config = load_config()
    model_path = config.models_dir / "bitgenesis.pt"
    tokenizer_path = config.models_dir / "tokenizer.json"
    run_chat(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )


if __name__ == "__main__":
    main()
