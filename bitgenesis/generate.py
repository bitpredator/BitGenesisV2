"""Generate text from a saved local BitGenesis checkpoint."""

import argparse
from collections.abc import Sequence
import math
import sys

import torch

from bitgenesis.config import load_config
from bitgenesis.model import BitGenesisLM
from bitgenesis.tokenizer import BPETokenizer
from bitgenesis.training import load_checkpoint


def generate_from_model(
    model: BitGenesisLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
) -> str:
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match the model checkpoint"
        )
    if not prompt:
        raise ValueError("Prompt must not be empty")

    prompt_ids = torch.tensor([tokenizer.encode_chat_prompt(prompt)], dtype=torch.long)
    prompt_length = prompt_ids.shape[1]
    end_token_id = tokenizer.special_token_id("<|end|>")
    first_special_token_id = tokenizer.special_token_id("<|user|>")

    def allowed_tokens(sequence: torch.Tensor) -> list[int]:
        generated_count = sequence.shape[1] - prompt_length
        valid_ids = tokenizer.valid_next_token_ids(
            sequence[0].tolist(),
            max_remaining_tokens=max_new_tokens - generated_count - 1,
        )
        return [
            token_id
            for token_id in valid_ids
            if token_id < first_special_token_id or token_id == end_token_id
        ]

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens,
        temperature,
        allowed_token_ids=allowed_tokens,
        stop_token_id=end_token_id,
    )
    return tokenizer.decode_chat_reply(generated_ids[0, prompt_length:].tolist())


def generate_text_from_model(
    model: BitGenesisLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
) -> str:
    """Continue a plain-text prompt without conversation role markers."""
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match the model checkpoint"
        )
    if not prompt:
        raise ValueError("Prompt must not be empty")

    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    def allowed_tokens(sequence: torch.Tensor) -> list[int]:
        return [
            token_id
            for token_id in tokenizer.valid_next_token_ids(
                sequence[0].tolist(),
                max_remaining_tokens=max_new_tokens
                - (sequence.shape[1] - prompt_ids.shape[1])
                - 1,
            )
            if token_id < tokenizer.text_vocab_size
        ]

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens,
        temperature,
        allowed_token_ids=allowed_tokens,
    )
    return tokenizer.decode(generated_ids[0].tolist())


def generate_text(
    prompt: str,
    model_path: str,
    tokenizer_path: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
) -> str:
    tokenizer = BPETokenizer.load(tokenizer_path)
    model = load_checkpoint(model_path, device="cpu", tokenizer=tokenizer)
    generate = (
        generate_from_model
        if model.training_format == "conversations"
        else generate_text_from_model
    )
    return generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Generate text with a local BitGenesis model."
    )
    parser.add_argument("--model", default=str(config.models_dir / "bitgenesis.pt"))
    parser.add_argument(
        "--tokenizer", default=str(config.models_dir / "tokenizer.json")
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must not be negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than zero")
    print(
        generate_text(
            args.prompt,
            args.model,
            args.tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
