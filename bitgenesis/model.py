"""A small decoder-only Transformer language model."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "embedding_dim",
            "num_heads",
            "num_layers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be greater than or equal to 0 and less than 1"
            )


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )
        causal_mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        sequence_length = values.shape[1]
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self.causal_mask[:sequence_length, :sequence_length],
            need_weights=False,
        )
        values = values + attended
        return values + self.feed_forward(self.feed_forward_norm(values))


class BitGenesisLM(nn.Module):
    """Predict the next token from a bounded context of token IDs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.training_format: Literal["text", "conversations"] = "text"
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_dim
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")
        if token_ids.shape[1] < 1:
            raise ValueError("token sequences must not be empty")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("token sequence exceeds the model context length")

        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        values = self.embedding_dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        for block in self.blocks:
            values = block(values)
        logits = self.language_head(self.final_norm(values))

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        allowed_token_ids: Callable[[Tensor], Sequence[int]] | None = None,
        stop_token_id: int | None = None,
    ) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError(
                "generation expects token_ids with shape (1, sequence_length)"
            )
        if token_ids.shape[1] < 1:
            raise ValueError("generation requires a non-empty prompt")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite number greater than zero")
        if (
            stop_token_id is not None
            and not 0 <= stop_token_id < self.config.vocab_size
        ):
            raise ValueError("stop_token_id is outside the model vocabulary")

        was_training = self.training
        self.eval()
        generated = token_ids
        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.context_length :]
                logits, _ = self(context)
                next_token_logits = logits[:, -1, :] / temperature
                if allowed_token_ids is not None:
                    allowed = list(allowed_token_ids(generated))
                    if not allowed:
                        raise ValueError(
                            "No allowed tokens are available for generation"
                        )
                    if any(
                        token_id < 0 or token_id >= self.config.vocab_size
                        for token_id in allowed
                    ):
                        raise ValueError(
                            "Allowed token ID is outside the model vocabulary"
                        )
                    allowed_mask = torch.zeros(
                        self.config.vocab_size,
                        dtype=torch.bool,
                        device=next_token_logits.device,
                    )
                    allowed_mask[allowed] = True
                    next_token_logits = next_token_logits.masked_fill(
                        ~allowed_mask,
                        float("-inf"),
                    )
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                if stop_token_id is not None and next_token.item() == stop_token_id:
                    break
        finally:
            self.train(was_training)
        return generated


def model_config_to_dict(config: ModelConfig) -> dict[str, int | float]:
    return asdict(config)
