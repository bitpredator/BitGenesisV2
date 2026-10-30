"""Train a small BitGenesis language model using local text files."""

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
import sys

import torch

from bitgenesis.config import load_config
from bitgenesis.corpus import (
    CorpusRecord,
    Conversation,
    find_duplicate_groups,
    load_conversation_records,
    load_corpus_records,
    split_corpus,
)
from bitgenesis.generate import generate_from_model, generate_text_from_model
from bitgenesis.model import BitGenesisLM, ModelConfig, model_config_to_dict
from bitgenesis.tokenizer import BPETokenizer
from bitgenesis.training import (
    TrainingCheckpointState,
    encode_conversations,
    encode_documents,
    fingerprint_training_data,
    load_training_checkpoint,
    save_checkpoint,
    save_training_report,
    train_model,
)


def _report_duplicate_groups(
    records: Sequence[CorpusRecord[str] | CorpusRecord[Conversation]],
) -> None:
    duplicate_groups = find_duplicate_groups(records)
    if not duplicate_groups:
        return

    duplicate_copies = sum(len(group) - 1 for group in duplicate_groups)
    group_label = "gruppo" if len(duplicate_groups) == 1 else "gruppi"
    print(
        f"Attenzione: trovati {len(duplicate_groups)} {group_label} "
        "di duplicati esatti "
        f"({duplicate_copies} copie aggiuntive). Le copie restano nel corpus e "
        "saranno assegnate allo stesso split:",
        file=sys.stderr,
    )
    for group in duplicate_groups:
        locations = ", ".join(
            f"{record.path}:{record.line_number}" for record in group
        )
        print(f"  - {locations}", file=sys.stderr)


def _resume_configuration(
    args: argparse.Namespace,
    model_config: ModelConfig,
) -> dict[str, object]:
    return {
        "data_format": args.data_format,
        "model": model_config_to_dict(model_config),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }


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
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint in --output-dir.",
    )
    resume_group.add_argument(
        "--resume-from",
        type=Path,
        metavar="CHECKPOINT",
        help="Resume from a specific local checkpoint.",
    )
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
    if args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be a positive integer.")

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "bitgenesis.pt"
    training_checkpoint_path = output_dir / "bitgenesis_training.pt"
    if args.resume_from is not None:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = (
            training_checkpoint_path
            if training_checkpoint_path.is_file()
            else checkpoint_path
        )
    else:
        resume_path = None
    tokenizer_path = output_dir / "tokenizer.json"
    metrics_path = args.metrics_output or output_dir / "training_metrics.json"
    protected_paths = {
        checkpoint_path.resolve(),
        training_checkpoint_path.resolve(),
        tokenizer_path.resolve(),
    }
    if resume_path is not None:
        protected_paths.add(resume_path.resolve())
    if metrics_path.resolve() in protected_paths:
        parser.error(
            "--metrics-output must not overwrite a model, tokenizer, or "
            "resume checkpoint."
        )
    data_suffix = ".txt" if args.data_format == "text" else ".jsonl"
    if metrics_path.resolve() in {
        path.resolve()
        for path in Path(args.data_dir).rglob(f"*{data_suffix}")
        if path.is_file()
    }:
        parser.error("--metrics-output must not overwrite training data.")

    torch.manual_seed(args.seed)
    if args.data_format == "text":
        records = load_corpus_records(args.data_dir)
        _report_duplicate_groups(records)
        split = split_corpus(
            records,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            group_key=lambda record: record.content,
        )
        train_documents = [record.content for record in split.train]
        validation_documents = [record.content for record in split.validation]
        tokenizer = BPETokenizer.train(train_documents, vocab_size=args.vocab_size)
        train_tokens = encode_documents(tokenizer, train_documents)
        validation_tokens = encode_documents(tokenizer, validation_documents)
        train_target_mask = None
        validation_target_mask = None
    else:
        records = load_conversation_records(args.data_dir)
        _report_duplicate_groups(records)
        split = split_corpus(
            records,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            group_key=lambda record: record.content,
        )
        train_conversations = [record.content for record in split.train]
        validation_conversations = [record.content for record in split.validation]
        training_text = [
            message
            for conversation in train_conversations
            for turn in conversation.turns
            for message in turn
        ]
        tokenizer = BPETokenizer.train(training_text, vocab_size=args.vocab_size)
        train_tokens, train_target_mask = encode_conversations(
            tokenizer, train_conversations
        )
        validation_tokens, validation_target_mask = encode_conversations(
            tokenizer, validation_conversations
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

    data_fingerprint = fingerprint_training_data(
        train_tokens,
        validation_tokens,
        train_target_mask,
        validation_target_mask,
    )
    resume_config = _resume_configuration(args, model_config)
    training_state: TrainingCheckpointState | None = None
    if resume_path is None:
        model = BitGenesisLM(model_config)
        model.training_format = args.data_format
    else:
        if not resume_path.is_file():
            parser.error(f"Resume checkpoint not found: {resume_path}")
        model, training_state = load_training_checkpoint(
            resume_path,
            device="cpu",
            tokenizer=tokenizer,
        )
        if model.config != model_config:
            parser.error("Resume checkpoint model settings do not match this run.")
        if training_state is not None:
            if training_state.configuration != resume_config:
                parser.error(
                    "Resume checkpoint training settings do not match this run."
                )
            if training_state.data_fingerprint != data_fingerprint:
                parser.error(
                    "Training data does not match the resume checkpoint."
                )
            if args.steps <= training_state.step:
                parser.error(
                    f"--steps must be greater than the checkpoint step "
                    f"({training_state.step})."
                )
        else:
            torch.manual_seed(args.seed)

    def save_progress(
        step: int,
        checkpoint_model: BitGenesisLM,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
        history: Sequence[dict[str, float]],
    ) -> None:
        save_checkpoint(
            checkpoint_model,
            training_checkpoint_path,
            tokenizer,
            TrainingCheckpointState(
                step=step,
                optimizer_state=optimizer.state_dict(),
                generator_state=generator.get_state(),
                rng_state=torch.get_rng_state(),
                configuration=resume_config,
                data_fingerprint=data_fingerprint,
                history=list(history),
            ),
        )

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
        start_step=0 if training_state is None else training_state.step,
        optimizer_state=(
            None if training_state is None else training_state.optimizer_state
        ),
        generator_state=(
            None if training_state is None else training_state.generator_state
        ),
        rng_state=None if training_state is None else training_state.rng_state,
        initial_history=(
            [] if training_state is None else training_state.history
        ),
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_callback=save_progress,
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
            "checkpoint_interval": args.checkpoint_interval,
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
