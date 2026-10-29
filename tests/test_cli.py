import unittest
from pathlib import Path
from unittest.mock import patch

from bitgenesis.__main__ import main, run_chat
from bitgenesis.generate import ContextLengthError


class RunChatTests(unittest.TestCase):
    def test_welcome_banner_reports_missing_model(self) -> None:
        output: list[str] = []

        run_chat(
            input_fn=lambda _prompt: "esci",
            output_fn=output.append,
            model_path=Path("missing/model.pt"),
            tokenizer_path=Path("missing/tokenizer.json"),
        )

        self.assertIn("BITGENESIS", "\n".join(output))
        self.assertIn(
            f"  |{'File modello: non trovati'.center(52)}|",
            output,
        )
        self.assertIn(
            f"  |{'Risposte sperimentali: possono essere imprecise.'.center(52)}|",
            output,
        )
        self.assertIn(
            f"  |{'/compact riassume i turni precedenti.'.center(52)}|",
            output,
        )
        self.assertTrue(
            any(
                "/impegno auto|minimo|medio|alto cambia il livello." in line
                for line in output
            )
        )
        self.assertEqual(output[-1], "  BitGenesis > A presto!")

    def test_reports_missing_model_and_exits_in_italian(self) -> None:
        messages = iter(["Ciao BitGenesis", " esci "])
        output: list[str] = []

        run_chat(input_fn=lambda _prompt: next(messages), output_fn=output.append)

        self.assertIn("File modello: non trovati", "\n".join(output))
        self.assertIn(
            "  BitGenesis > Modello non disponibile. Addestrami prima con "
            "'python -m bitgenesis.train --data-format conversations'.",
            output,
        )
        self.assertEqual(output[-1], "  BitGenesis > A presto!")

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
            self.assertIn("File modello: presenti", "\n".join(output))
            self.assertIn(
                f"  |{'In locale ti chiamerò Utente.'.center(52)}|",
                output,
            )
            chat_class.assert_called_once_with(model_path, tokenizer_path)
            chat.reply.assert_called_once()
            self.assertEqual(chat.reply.call_args.args[0], "Ciao!")
            self.assertEqual(
                {key: value for key, value in chat.reply.call_args.kwargs.items()},
                {"max_new_tokens": 3, "temperature": 0.7},
            )
        self.assertEqual(
            history_snapshots,
            [[
                (
                    "Regola per questa chat locale: chiama l'utente Utente.",
                    "In questa chat locale chiamerò l'utente Utente.",
                )
            ]],
        )
        self.assertIn("  BitGenesis > Salve!", output)

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
        local_identity = (
            "Regola per questa chat locale: chiama l'utente Utente.",
            "In questa chat locale chiamerò l'utente Utente.",
        )
        self.assertEqual(
            history_snapshots,
            [[local_identity], [("Ciao", "Salve!"), local_identity]],
        )
        self.assertIn("  BitGenesis > Sono BitGenesis.", output)

    def test_auto_effort_scales_generation_for_request_complexity(self) -> None:
        messages = iter(["Ciao", "Come stai?", "Spiegami la differenza", "esci"])
        output: list[str] = []

        with (
            patch.object(Path, "is_file", return_value=True),
            patch("bitgenesis.__main__.ConversationChat") as chat_class,
        ):
            chat = chat_class.return_value
            chat.reply.return_value = "Risposta."
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
                max_new_tokens=80,
            )

        self.assertEqual(
            [
                call.kwargs["max_new_tokens"]
                for call in chat.reply.call_args_list
            ],
            [20, 80, 160],
        )
        self.assertTrue(any("minimo (automatico)" in line for line in output))
        self.assertTrue(any("medio (automatico)" in line for line in output))
        self.assertTrue(any("alto (automatico)" in line for line in output))

    def test_effort_command_changes_and_reports_chat_mode(self) -> None:
        messages = iter(["/impegno alto", "Ciao", "/impegno", "esci"])
        output: list[str] = []

        with (
            patch.object(Path, "is_file", return_value=True),
            patch("bitgenesis.__main__.ConversationChat") as chat_class,
        ):
            chat_class.return_value.reply.return_value = "Salve."
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
            )

        self.assertEqual(
            chat_class.return_value.reply.call_args.kwargs["max_new_tokens"],
            160,
        )
        self.assertIn("  BitGenesis > Impegno impostato: alto.", output)
        self.assertIn(
            "  BitGenesis > Impegno attuale: alto. "
            "Usa /impegno auto|minimo|medio|alto per cambiarlo.",
            output,
        )

    def test_local_user_name_is_fixed_in_context(self) -> None:
        messages = iter(["Ciao", "Come mi chiamo?", "esci"])
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
                return "Ciao, Utente!"

            chat.reply.side_effect = reply_with_history
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
            )

        local_identity = (
            "Regola per questa chat locale: chiama l'utente Utente.",
            "In questa chat locale chiamerò l'utente Utente.",
        )
        self.assertEqual(
            history_snapshots[0],
            [local_identity],
        )
        self.assertEqual(
            history_snapshots[1],
            [
                ("Ciao", "Ciao, Utente!"),
                local_identity,
            ],
        )

    def test_compact_previews_summary_and_applies_it_after_confirmation(self) -> None:
        messages = iter(["Ciao", "Come stai?", "/compact", "s", "E poi?", "esci"])
        output: list[str] = []
        history_snapshots: list[list[tuple[str, str]]] = []
        compact_snapshots: list[list[tuple[str, str]]] = []

        with (
            patch.object(Path, "is_file", return_value=True),
            patch("bitgenesis.__main__.ConversationChat") as chat_class,
        ):
            chat = chat_class.return_value
            def compact_history(
                history: list[tuple[str, str]],
            ) -> tuple[str, list[tuple[str, str]], int]:
                compact_snapshots.append(list(history))
                summary = "L'utente ha salutato."
                return (
                    summary,
                    [
                        ("Riassunto della conversazione precedente", summary),
                        history[-1],
                    ],
                    1,
                )

            chat.compact.side_effect = compact_history
            chat.reply.side_effect = lambda _prompt, history, **_kwargs: (
                history_snapshots.append(list(history))
                or ("Salve", "Sto bene", "Continuiamo")[len(history_snapshots) - 1]
            )

            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
            )

        self.assertEqual(
            compact_snapshots,
            [[("Ciao", "Salve"), ("Come stai?", "Sto bene")]],
        )
        self.assertEqual(
            history_snapshots[-1],
            [
                ("Riassunto della conversazione precedente", "L'utente ha salutato."),
                ("Come stai?", "Sto bene"),
                (
                    "Regola per questa chat locale: chiama l'utente Utente.",
                    "In questa chat locale chiamerò l'utente Utente.",
                ),
            ],
        )
        self.assertTrue(
            any("1 scambio riassunto, 1 recenti mantenuti" in line for line in output)
        )

    def test_compact_cancel_keeps_original_history(self) -> None:
        messages = iter(["Ciao", "Come stai?", "/compact", "n", "E poi?", "esci"])
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
                return ("Salve", "Sto bene", "Continuiamo")[
                    len(history_snapshots) - 1
                ]

            chat.reply.side_effect = reply_with_history
            chat.compact.return_value = (
                "Riepilogo proposto",
                [("Riassunto", "Riepilogo proposto"), ("Come stai?", "Sto bene")],
                1,
            )
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
            )

        self.assertEqual(
            history_snapshots[-1],
            [
                ("Ciao", "Salve"),
                ("Come stai?", "Sto bene"),
                (
                    "Regola per questa chat locale: chiama l'utente Utente.",
                    "In questa chat locale chiamerò l'utente Utente.",
                ),
            ],
        )
        self.assertIn(
            "  BitGenesis > Compattazione annullata; cronologia invariata.",
            output,
        )

    def test_reports_overlong_message_without_ending_the_chat(self) -> None:
        messages = iter(["messaggio troppo lungo", "esci"])
        output: list[str] = []

        with (
            patch.object(Path, "is_file", return_value=True),
            patch("bitgenesis.__main__.ConversationChat") as chat_class,
        ):
            chat = chat_class.return_value
            chat.reply.side_effect = ContextLengthError("Messaggio troppo lungo.")
            run_chat(
                input_fn=lambda _prompt: next(messages),
                output_fn=output.append,
                model_path=Path("model.pt"),
                tokenizer_path=Path("tokenizer.json"),
            )

        self.assertIn("  BitGenesis > Messaggio troppo lungo.", output)
        self.assertEqual(output[-1], "  BitGenesis > A presto!")

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
            "  BitGenesis > Modello non disponibile. Addestrami prima con "
            "'python -m bitgenesis.train --data-format conversations'.",
            output,
        )

    def test_empty_messages_are_ignored_and_exit_commands_are_case_insensitive(
        self,
    ) -> None:
        messages = iter(["  ", "QUIT"])
        output: list[str] = []

        run_chat(input_fn=lambda _prompt: next(messages), output_fn=output.append)

        self.assertEqual(output[-1], "  BitGenesis > A presto!")

    def test_end_of_input_exits_cleanly(self) -> None:
        output: list[str] = []

        def end_input(_prompt: str) -> str:
            raise EOFError

        run_chat(input_fn=end_input, output_fn=output.append)

        self.assertEqual(output[-1], "  BitGenesis > A presto!")

    def test_main_loads_configuration_before_starting_chat(self) -> None:
        with (
            patch("bitgenesis.__main__.load_config") as load_config,
            patch("bitgenesis.__main__.run_chat") as run_chat_mock,
        ):
            load_config.return_value.models_dir = Path("local/models")
            load_config.return_value.generation.max_new_tokens = 12
            load_config.return_value.generation.temperature = 0.6
            main([])

        load_config.assert_called_once_with()
        run_chat_mock.assert_called_once_with(
            model_path=Path("local/models/bitgenesis.pt"),
            tokenizer_path=Path("local/models/tokenizer.json"),
            max_new_tokens=12,
            temperature=0.6,
            effort="auto",
        )

    def test_command_line_overrides_generation_settings(self) -> None:
        with (
            patch("bitgenesis.__main__.load_config") as load_config,
            patch("bitgenesis.__main__.run_chat") as run_chat_mock,
        ):
            load_config.return_value.models_dir = Path("local/models")
            load_config.return_value.generation.max_new_tokens = 12
            load_config.return_value.generation.temperature = 0.6
            main(
                [
                    "--max-new-tokens",
                    "24",
                    "--temperature",
                    "0.8",
                    "--effort",
                    "alto",
                ]
            )

        self.assertEqual(
            run_chat_mock.call_args.kwargs,
            {
                "model_path": Path("local/models/bitgenesis.pt"),
                "tokenizer_path": Path("local/models/tokenizer.json"),
                "max_new_tokens": 24,
                "temperature": 0.8,
                "effort": "alto",
            },
        )


if __name__ == "__main__":
    unittest.main()
