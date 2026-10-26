import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from bitgenesis.corpus import Conversation
from bitgenesis.generate import generate_from_model, generate_text
from bitgenesis.model import BitGenesisLM, ModelConfig
from bitgenesis.tokenizer import BPETokenizer
from bitgenesis.training import (
    encode_conversations,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    train_model,
)


def make_model(vocab_size: int = 256) -> BitGenesisLM:
    return BitGenesisLM(
        ModelConfig(
            vocab_size=vocab_size,
            context_length=8,
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
    def test_forward_computes_next_token_loss(self) -> None:
        model = make_model()
        token_ids = torch.randint(0, 256, (2, 8))

        logits, loss = model(token_ids, token_ids)

        self.assertEqual(logits.shape, (2, 8, 256))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

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
    def test_batches_pair_each_input_with_next_token_target(self) -> None:
        tokens = torch.arange(20)

        inputs, targets = get_batch(tokens, batch_size=4, context_length=5)

        self.assertEqual(inputs.shape, (4, 5))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

    def test_trains_saves_loads_and_generates(self) -> None:
        tokenizer = BPETokenizer.train(
            ["ciao bitgenesis", "salve bitgenesis"], vocab_size=256
        )
        train_conversation = Conversation(
            turns=(("ciao bitgenesis", "salve bitgenesis"),)
        )
        validation_conversation = Conversation(turns=(("come stai", "sto bene"),))
        train_tokens = encode_conversations(tokenizer, [train_conversation] * 4)
        validation_tokens = encode_conversations(
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
                "ciao bitgenesis",
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
