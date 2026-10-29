import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from bitgenesis.corpus import Conversation
from bitgenesis.generate import (
    CompactionError,
    ContextLengthError,
    ConversationChat,
    fit_history_to_context,
    generate_from_model,
    generate_text,
)
from bitgenesis.model import BitGenesisLM, ModelConfig
from bitgenesis.tokenizer import BPETokenizer
from bitgenesis.training import (
    encode_conversations,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    save_training_report,
    train_model,
)


def make_model(vocab_size: int = 256, context_length: int = 8) -> BitGenesisLM:
    return BitGenesisLM(
        ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0,
        )
    )


class ModelConfigTests(unittest.TestCase):
    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(vocab_size=256, embedding_dim=10, num_heads=4)
        with self.assertRaisesRegex(ValueError, "context_length"):
            ModelConfig(vocab_size=256, context_length=0)


class BitGenesisLMTests(unittest.TestCase):
    def test_chat_context_keeps_newest_complete_turns(self) -> None:
        tokenizer = BPETokenizer.train(["Ciao", "Risposta"], vocab_size=256)
        history = [
            ("Primo turno", "Prima risposta"),
            ("Secondo turno", "Seconda risposta"),
            ("Ultimo turno", "Ultima risposta"),
        ]
        context_length = len(tokenizer.encode_chat_prompt("x", history[-1:]))

        retained = fit_history_to_context(
            tokenizer,
            "x",
            history,
            context_length,
        )

        self.assertEqual(retained, history[-1:])

    def test_rejects_chat_message_that_exceeds_context(self) -> None:
        tokenizer = BPETokenizer.train(["Ciao"], vocab_size=256)

        with self.assertRaisesRegex(ContextLengthError, "troppo lungo"):
            fit_history_to_context(
                tokenizer,
                "messaggio troppo lungo",
                [],
                context_length=8,
            )

    def test_compaction_summarizes_old_turns_and_keeps_recent_ones(self) -> None:
        tokenizer = BPETokenizer.train(
            ["Riassumi in italiano fatti preferenze questioni aperte", "Ciao"],
            vocab_size=300,
        )
        model = make_model(tokenizer.vocab_size, context_length=256)
        chat = object.__new__(ConversationChat)
        chat.tokenizer = tokenizer
        chat.model = model
        chat.reply = Mock(return_value="L'utente preferisce risposte brevi.")
        history = [
            ("Ciao", "Salve"),
            ("Preferisco risposte brevi", "Va bene"),
            ("Qual è il piano?", "Continuiamo con il progetto"),
        ]

        summary, compacted_history, kept_turns = chat.compact(history)

        self.assertEqual(summary, "L'utente preferisce risposte brevi.")
        self.assertEqual(
            compacted_history[0],
            ("Riassunto della conversazione precedente", summary),
        )
        self.assertEqual(compacted_history[-1], history[-1])
        self.assertGreaterEqual(kept_turns, 1)
        self.assertEqual(chat.reply.call_count, 1)

    def test_compaction_requires_more_than_one_turn(self) -> None:
        tokenizer = BPETokenizer.train(["Ciao"], vocab_size=256)
        chat = object.__new__(ConversationChat)
        chat.tokenizer = tokenizer
        chat.model = make_model(tokenizer.vocab_size)

        with self.assertRaisesRegex(CompactionError, "almeno due"):
            chat.compact([("Ciao", "Salve")])

    def test_forward_computes_next_token_loss(self) -> None:
        model = make_model()
        token_ids = torch.randint(0, 256, (2, 8))

        logits, loss = model(token_ids, token_ids)

        self.assertEqual(logits.shape, (2, 8, 256))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_ignores_unsupervised_targets(self) -> None:
        model = make_model()
        token_ids = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        targets[:, :4] = -100

        logits, loss = model(token_ids, targets)

        expected_loss = torch.nn.functional.cross_entropy(
            logits[:, 4:, :].reshape(-1, 256),
            targets[:, 4:].reshape(-1),
        )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_generation_respects_context_and_keeps_valid_token_ids(self) -> None:
        model = make_model()
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        generated = model.generate(prompt, max_new_tokens=3)

        self.assertEqual(generated.shape, (1, 12))
        self.assertTrue(torch.all((generated >= 0) & (generated < 256)))

    def test_rejects_invalid_shapes_and_generation_parameters(self) -> None:
        model = make_model()
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.ones(8, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "temperature"):
            model.generate(torch.ones((1, 2), dtype=torch.long), 1, temperature=0)

    def test_generation_stops_after_end_token(self) -> None:
        model = make_model()
        end_token_id = 255

        def predict_end(token_ids: torch.Tensor, targets: torch.Tensor | None = None):
            logits = torch.zeros((token_ids.shape[0], token_ids.shape[1], 256))
            logits[:, :, end_token_id] = 100
            return logits, None

        with patch.object(model, "forward", side_effect=predict_end):
            generated = model.generate(
                torch.tensor([[1, 2]]),
                max_new_tokens=10,
                stop_token_id=end_token_id,
            )

        self.assertEqual(generated.tolist(), [[1, 2, end_token_id]])

    def test_conversation_generation_returns_only_assistant_reply(self) -> None:
        tokenizer = BPETokenizer.train(["Ciao"], vocab_size=256)
        model = make_model(tokenizer.vocab_size)
        end_token_id = tokenizer.special_token_id("<|end|>")

        def predict_end(token_ids: torch.Tensor, targets: torch.Tensor | None = None):
            logits = torch.zeros(
                (token_ids.shape[0], token_ids.shape[1], tokenizer.vocab_size)
            )
            logits[:, :, end_token_id] = 100
            return logits, None

        with patch.object(model, "forward", side_effect=predict_end):
            reply = generate_from_model(
                model,
                tokenizer,
                "Ciao",
                max_new_tokens=10,
            )

        self.assertEqual(reply, "")


class TrainingTests(unittest.TestCase):
    def test_saves_training_configuration_and_loss_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "training_metrics.json"
            save_training_report(
                path,
                configuration={"seed": 42, "data_format": "conversations"},
                history=[{"step": 1.0, "train": 1.5, "validation": 2.0}],
            )

            report = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(report["format_version"], 1)
        self.assertEqual(
            report["configuration"],
            {"seed": 42, "data_format": "conversations"},
        )
        self.assertEqual(
            report["history"],
            [{"step": 1.0, "train": 1.5, "validation": 2.0}],
        )

    def test_rejects_non_finite_training_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training_metrics.json"
            with self.assertRaises(ValueError):
                save_training_report(
                    path,
                    configuration={},
                    history=[{"step": 1.0, "train": float("nan")}],
                )
            self.assertFalse(path.exists())

    def test_batches_pair_each_input_with_next_token_target(self) -> None:
        tokens = torch.arange(20)

        inputs, targets = get_batch(tokens, batch_size=4, context_length=5)

        self.assertEqual(inputs.shape, (4, 5))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

    def test_conversation_batches_mask_user_targets(self) -> None:
        tokens = torch.arange(20)
        target_mask = torch.zeros(20, dtype=torch.bool)
        target_mask[[5, 6, 12, 13]] = True

        inputs, targets = get_batch(
            tokens,
            batch_size=8,
            context_length=4,
            generator=torch.Generator().manual_seed(7),
            target_mask=target_mask,
        )

        for row in targets:
            valid_offsets = torch.nonzero(row != -100, as_tuple=False).flatten()
            self.assertGreater(valid_offsets.numel(), 0)
            self.assertTrue(torch.all(target_mask[row[valid_offsets]]))

    def test_rejects_conversation_split_without_supervised_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "No training windows"):
            get_batch(
                torch.arange(20),
                batch_size=1,
                context_length=4,
                target_mask=torch.zeros(20, dtype=torch.bool),
            )

    def test_trains_saves_loads_and_generates(self) -> None:
        tokenizer = BPETokenizer.train(
            ["ciao bitgenesis", "salve bitgenesis"], vocab_size=256
        )
        train_conversation = Conversation(
            turns=(("ciao bitgenesis", "salve bitgenesis"),)
        )
        validation_conversation = Conversation(turns=(("come stai", "sto bene"),))
        train_tokens, train_target_mask = encode_conversations(
            tokenizer, [train_conversation] * 4
        )
        validation_tokens, validation_target_mask = encode_conversations(
            tokenizer, [validation_conversation] * 4
        )
        model = make_model(tokenizer.vocab_size)

        history = train_model(
            model,
            train_tokens,
            validation_tokens,
            steps=2,
            batch_size=2,
            eval_interval=1,
            eval_batches=1,
            seed=7,
            train_target_mask=train_target_mask,
            validation_target_mask=validation_target_mask,
        )
        model.training_format = "conversations"

        self.assertEqual(len(history), 2)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(item["train"])) for item in history)
        )
        self.assertTrue(
            all(torch.isfinite(torch.tensor(item["validation"])) for item in history)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "bitgenesis.pt"
            tokenizer_path = root / "tokenizer.json"
            save_checkpoint(model, model_path, tokenizer)
            tokenizer.save(tokenizer_path)
            loaded_tokenizer = BPETokenizer.load(tokenizer_path)
            loaded_model = load_checkpoint(model_path, tokenizer=loaded_tokenizer)
            generated = generate_text(
                "ciao",
                str(model_path),
                str(tokenizer_path),
                max_new_tokens=2,
            )
            different_tokenizer = BPETokenizer.train(
                ["zzzy zzzy"],
                vocab_size=270,
            )
            self.assertNotEqual(different_tokenizer.fingerprint, tokenizer.fingerprint)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint(model_path, tokenizer=different_tokenizer)

        self.assertEqual(loaded_model.config, model.config)
        self.assertEqual(loaded_model.training_format, "conversations")
        self.assertEqual(loaded_tokenizer.vocab_size, tokenizer.vocab_size)
        self.assertIsInstance(generated, str)
        self.assertNotIn("<|", generated)

    def test_rejects_too_short_token_streams(self) -> None:
        with self.assertRaisesRegex(ValueError, "more tokens"):
            get_batch(torch.arange(8), batch_size=1, context_length=8)


if __name__ == "__main__":
    unittest.main()
