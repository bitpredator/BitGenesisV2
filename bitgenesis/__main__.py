"""Run BitGenesis in an interactive terminal session."""

from collections.abc import Callable

from bitgenesis.config import load_config

_EXIT_COMMANDS = {"esci", "exit", "quit", "uscita"}


def run_chat(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> None:
    """Run the Italian terminal interface without implying model capabilities."""
    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn

    write_output("BitGenesis: Ciao! Scrivimi in italiano. Digita 'esci' per terminare.")
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
            write_output(
                "BitGenesis: Ho ricevuto il tuo messaggio, ma non sono ancora "
                "addestrata a generare risposte."
            )


def main() -> None:
    load_config()
    run_chat()


if __name__ == "__main__":
    main()
