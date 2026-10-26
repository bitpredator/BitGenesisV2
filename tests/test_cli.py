import unittest
from unittest.mock import patch

from bitgenesis.__main__ import main, run_chat


class RunChatTests(unittest.TestCase):
    def test_receives_message_and_exits_in_italian(self) -> None:
        messages = iter(["Ciao BitGenesis", " esci "])
        output: list[str] = []

        run_chat(input_fn=lambda _prompt: next(messages), output_fn=output.append)

        self.assertEqual(
            output,
            [
                "BitGenesis: Ciao! Scrivimi in italiano. Digita 'esci' per terminare.",
                "BitGenesis: Ho ricevuto il tuo messaggio, ma non sono ancora "
                "addestrata a generare risposte.",
                "BitGenesis: A presto!",
            ],
        )

    def test_empty_messages_are_ignored_and_exit_commands_are_case_insensitive(self) -> None:
        messages = iter(["  ", "QUIT"])
        output: list[str] = []

        run_chat(input_fn=lambda _prompt: next(messages), output_fn=output.append)

        self.assertEqual(len(output), 2)
        self.assertEqual(output[-1], "BitGenesis: A presto!")

    def test_end_of_input_exits_cleanly(self) -> None:
        output: list[str] = []

        def end_input(_prompt: str) -> str:
            raise EOFError

        run_chat(input_fn=end_input, output_fn=output.append)

        self.assertEqual(output[-1], "BitGenesis: A presto!")

    def test_main_loads_configuration_before_starting_chat(self) -> None:
        with (
            patch("bitgenesis.__main__.load_config") as load_config,
            patch("bitgenesis.__main__.run_chat") as run_chat_mock,
        ):
            main()

        load_config.assert_called_once_with()
        run_chat_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
