"""Utilities for training the language model on local token sequences."""

from collections.abc import Sequence
import math
from pathlib import Path

import torch
from torch import Tensor

from bitgenesis.model import BitGenesisLM, ModelConfig, model_config_to_dict
from bitgenesis.tokenizer import BPETokenizer


def encode_documents(tokenizer: BPETokenizer, documents: Sequence[str]) -> Tensor:
    """Encode documents with newline separators into one contiguous token stream."""
    text = "\n".join(document for document in documents if document)
    if not text:
        raise ValueError("At least one non-empty document is required")
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch(
    tokens: Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample input windows and their one-token-shifted next-token targets."""
    if tokens.ndim != 1:
        raise ValueError("tokens must be a one-dimensional tensor")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    if tokens.numel() <= context_length:
        raise ValueError("token sequence must contain more tokens than context_length")

    starts = torch.randint(
        tokens.numel() - context_length,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(context_length)
    input_ids = tokens[starts[:, None] + offsets]
    target_ids = tokens[starts[:, None] + offsets + 1]
    return input_ids.to(device), target_ids.to(device)


def estimate_loss(
    model: BitGenesisLM,
    train_tokens: Tensor,
    validation_tokens: Tensor,
    *,
    batch_size: int,
    eval_batches: int,
    device: torch.device | str,
    generator: torch.Generator,
) -> dict[str, float]:
    if isinstance(eval_batches, bool) or not isinstance(eval_batches, int) or eval_batches < 1:
        raise ValueError("eval_batches must be a positive integer")

    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        with torch.no_grad():
            for name, tokens in (
                ("train", train_tokens),
                ("validation", validation_tokens),
            ):
                losses = []
                for _ in range(eval_batches):
                    inputs, targets = get_batch(
                        tokens,
                        batch_size,
                        model.config.context_length,
                        device,
                        generator,
                    )
                    _, loss = model(inputs, targets)
                    if loss is None:
                        raise RuntimeError("Model did not return a loss for labeled data")
                    losses.append(loss)
                results[name] = torch.stack(losses).mean().item()
    finally:
        model.train(was_training)
    return results


def train_model(
    model: BitGenesisLM,
    train_tokens: Tensor,
    validation_tokens: Tensor,
    *,
    steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    eval_interval: int = 100,
    eval_batches: int = 10,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[dict[str, float]]:
    """Train from random initialization and return periodic loss measurements."""
    for name, value in (("steps", steps), ("batch_size", batch_size), ("eval_interval", eval_interval), ("eval_batches", eval_batches)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite number greater than zero")

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        model.train()
        inputs, targets = get_batch(
            train_tokens,
            batch_size,
            model.config.context_length,
            device,
            generator,
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return a loss for labeled data")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Training loss became non-finite at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            measured_losses = estimate_loss(
                model,
                train_tokens,
                validation_tokens,
                batch_size=batch_size,
                eval_batches=eval_batches,
                device=device,
                generator=generator,
            )
            record = {"step": float(step), **measured_losses}
            history.append(record)
    return history


def save_checkpoint(
    model: BitGenesisLM,
    path: Path,
    tokenizer: BPETokenizer,
) -> None:
    """Save model configuration and weights as a local PyTorch checkpoint."""
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the model")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(
            {
                "format_version": 1,
                "model_config": model_config_to_dict(model.config),
                "state_dict": model.state_dict(),
                "tokenizer_fingerprint": tokenizer.fingerprint,
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    tokenizer: BPETokenizer | None = None,
) -> BitGenesisLM:
    """Load model weights from a trusted local checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported or invalid model checkpoint")
    config_data = payload.get("model_config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_data, dict) or not isinstance(state_dict, dict):
        raise ValueError("Model checkpoint is missing its configuration or weights")
    if tokenizer is not None:
        fingerprint = payload.get("tokenizer_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != tokenizer.fingerprint:
            raise ValueError("Tokenizer does not match the model checkpoint")

    try:
        config = ModelConfig(**config_data)
    except (TypeError, ValueError) as error:
        raise ValueError("Model checkpoint has an invalid model configuration") from error
    model = BitGenesisLM(config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)
