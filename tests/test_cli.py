import unittest
from pathlib import Path
from unittest.mock import patch

from bitgenesis.__main__ import main, run_chat


class RunChatTests(unittest.TestCase):
    def test_reports_missing_model_and_exits_in_italian(self) -> None:
        messages = iter(["Ciao BitGenesis", " esci "])
        output: list[str] = []

        run_chat(input_fn=lambda _prompt: next(messages), output_fn=output.append)

        self.assertEqual(
            output,
            [
                "BitGenesis: Ciao! Scrivimi in italiano. Digita 'esci' per terminare.",
                "BitGenesis: Modello non disponibile. Addestrami prima con "
                "'python -m bitgenesis.train --data-format conversations'.",
                "BitGenesis: A presto!",
            ],
        )

    def test_generates_replies_using_local_model_paths(self) -> None:
        messages = iter([" Ciao! ", "esci"])
        output: list[str] = []
        model_path = Path("models/bitgenesis.pt")
        tokenizer_path = Path("models/tokenizer.json")
        history_snapshots: list[list[tuple[str, str]]] = []

        with (
            patch.object(Path, "is_file", return_value=True),
            patch("bitgenesis.__main__.ConversationChat") as chat_class,
        ):
            chat = chat_class.return_value
            chat.reply.return_value = "Salve!"
            chat.reply.side_effect = lambda *args, **_kwargs: (
                history_snapshots.append(list(args[1])) or "Salve!"
            )
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                max_new_tokens=12,
                temperature=0.7,
            )
            chat_class.assert_called_once_with(model_path, tokenizer_path)
            chat.reply.assert_called_once()
            self.assertEqual(chat.reply.call_args.args[0], "Ciao!")
            self.assertEqual(
                {key: value for key, value in chat.reply.call_args.kwargs.items()},
                {"max_new_tokens": 12, "temperature": 0.7},
            )
        self.assertEqual(history_snapshots, [[]])
        self.assertIn("BitGenesis: Salve!", output)

    def test_passes_previous_turns_as_context_for_next_reply(self) -> None:
        messages = iter(["Ciao", "Come ti chiami?", "esci"])
        output: list[str] = []
        history_snapshots: list[list[tuple[str, str]]] = []

        with (
            patch.object(Path, "is_file", return_value=True),
            patch("bitgenesis.__main__.ConversationChat") as chat_class,
        ):
            chat = chat_class.return_value

            def reply_with_history(
                _prompt: str,
                history: list[tuple[str, str]],
                **_kwargs: object,
            ) -> str:
                history_snapshots.append(list(history))
                return ("Salve!", "Sono BitGenesis.")[len(history_snapshots) - 1]

            chat.reply.side_effect = reply_with_history
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
            )

        chat_class.assert_called_once_with(Path("model.pt"), Path("tokenizer.json"))
        self.assertEqual(chat.reply.call_count, 2)
        self.assertEqual(history_snapshots, [[], [("Ciao", "Salve!")]])
        self.assertIn("BitGenesis: Sono BitGenesis.", output)

    def test_reports_missing_checkpoint_files_without_attempting_generation(
        self,
    ) -> None:
        messages = iter(["Ciao", "esci"])
        output: list[str] = []

        with patch("bitgenesis.__main__.ConversationChat") as chat_class:
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("missing/model.pt"),
                tokenizer_path=Path("missing/tokenizer.json"),
            )

        chat_class.assert_not_called()
        self.assertIn(
            "BitGenesis: Modello non disponibile. Addestrami prima con "
            "'python -m bitgenesis.train --data-format conversations'.",
            output,
        )

    def test_empty_messages_are_ignored_and_exit_commands_are_case_insensitive(
        self,
    ) -> None:
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
            load_config.return_value.models_dir = Path("local/models")
            main()

        load_config.assert_called_once_with()
        run_chat_mock.assert_called_once_with(
            model_path=Path("local/models/bitgenesis.pt"),
            tokenizer_path=Path("local/models/tokenizer.json"),
        )


if __name__ == "__main__":
    unittest.main()
