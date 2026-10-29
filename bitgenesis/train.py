"""Train a small BitGenesis language model using local text files."""

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
import sys

import torch

from bitgenesis.config import load_config
from bitgenesis.corpus import load_conversations, load_corpus, split_corpus
from bitgenesis.generate import generate_from_model, generate_text_from_model
from bitgenesis.model import BitGenesisLM, ModelConfig, model_config_to_dict
from bitgenesis.tokenizer import BPETokenizer
from bitgenesis.training import (
    encode_conversations,
    encode_documents,
    save_checkpoint,
    save_training_report,
    train_model,
)


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Train BitGenesis locally from UTF-8 text files."
    )
    parser.add_argument("--data-dir", default=str(config.datasets_dir))
    parser.add_argument("--output-dir", default=str(config.models_dir))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Save the training configuration and loss history to this JSON file.",
    )
    parser.add_argument(
        "--data-format",
        choices=("text", "conversations"),
        default="text",
        help="Read UTF-8 .txt documents or structured .jsonl conversations.",
    )
    parser.add_argument("--vocab-size", type=int, default=config.model.vocab_size)
    parser.add_argument(
        "--context-length",
        type=int,
        default=config.model.context_length,
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=config.model.embedding_dim,
    )
    parser.add_argument("--heads", type=int, default=config.model.heads)
    parser.add_argument("--layers", type=int, default=config.model.layers)
    parser.add_argument("--dropout", type=float, default=config.model.dropout)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Ciao")
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=config.generation.max_new_tokens,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.generation.temperature,
    )
    args = parser.parse_args(argv)

    if args.generate_tokens < 0:
        parser.error("--generate-tokens must not be negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than zero")
    if not 0 < args.validation_fraction < 1:
        parser.error(
            "--validation-fraction must be greater than 0 and less than 1 for training."
        )

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "bitgenesis.pt"
    tokenizer_path = output_dir / "tokenizer.json"
    metrics_path = args.metrics_output or output_dir / "training_metrics.json"
    if metrics_path.resolve() in {
        checkpoint_path.resolve(),
        tokenizer_path.resolve(),
    }:
        parser.error("--metrics-output must not overwrite the model or tokenizer.")
    data_suffix = ".txt" if args.data_format == "text" else ".jsonl"
    if metrics_path.resolve() in {
        path.resolve()
        for path in Path(args.data_dir).rglob(f"*{data_suffix}")
        if path.is_file()
    }:
        parser.error("--metrics-output must not overwrite training data.")

    torch.manual_seed(args.seed)
    if args.data_format == "text":
        documents = load_corpus(args.data_dir)
        split = split_corpus(
            documents,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        tokenizer = BPETokenizer.train(split.train, vocab_size=args.vocab_size)
        train_tokens = encode_documents(tokenizer, split.train)
        validation_tokens = encode_documents(tokenizer, split.validation)
        train_target_mask = None
        validation_target_mask = None
    else:
        conversations = load_conversations(args.data_dir)
        split = split_corpus(
            conversations,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        training_text = [
            message
            for conversation in split.train
            for turn in conversation.turns
            for message in turn
        ]
        tokenizer = BPETokenizer.train(training_text, vocab_size=args.vocab_size)
        train_tokens, train_target_mask = encode_conversations(tokenizer, split.train)
        validation_tokens, validation_target_mask = encode_conversations(
            tokenizer, split.validation
        )

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    if train_tokens.numel() <= model_config.context_length:
        parser.error(
            "Training documents must produce more tokens than --context-length."
        )
    if validation_tokens.numel() <= model_config.context_length:
        parser.error(
            "Validation documents must produce more tokens than --context-length."
        )
    if train_target_mask is not None and not train_target_mask[1:].any():
        parser.error("Training conversations contain no supervised assistant tokens.")
    if (
        validation_target_mask is not None
        and not validation_target_mask[1:].any()
    ):
        parser.error(
            "Validation conversations contain no supervised assistant tokens."
        )

    model = BitGenesisLM(model_config)
    if args.data_format == "conversations":
        model.training_format = "conversations"
    else:
        model.training_format = "text"
    history = train_model(
        model,
        train_tokens,
        validation_tokens,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device="cpu",
        train_target_mask=train_target_mask,
        validation_target_mask=validation_target_mask,
    )

    save_checkpoint(model, checkpoint_path, tokenizer)
    tokenizer.save(tokenizer_path)
    save_training_report(
        metrics_path,
        configuration={
            "data_format": args.data_format,
            "data_dir": str(Path(args.data_dir)),
            "training_examples": len(split.train),
            "validation_examples": len(split.validation),
            "training_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
            "model": model_config_to_dict(model_config),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "eval_interval": args.eval_interval,
            "eval_batches": args.eval_batches,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "generate_tokens": args.generate_tokens,
            "temperature": args.temperature,
        },
        history=history,
    )

    print(f"Modello salvato in: {checkpoint_path}")
    print(f"Tokenizer salvato in: {tokenizer_path}")
    print(f"Metriche di training salvate in: {metrics_path}")
    for result in history:
        print(
            f"Passo {int(result['step'])}: "
            f"perdita train={result['train']:.4f}, "
            f"validazione={result['validation']:.4f}"
        )
    if args.prompt:
        if args.data_format == "conversations":
            generated_text = generate_from_model(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.generate_tokens,
                temperature=args.temperature,
            )
            print(f"\nRisposta generata:\n{generated_text}")
        else:
            generated_text = generate_text_from_model(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.generate_tokens,
                temperature=args.temperature,
            )
            print(f"\nTesto generato:\n{generated_text}")


if __name__ == "__main__":
    main()
