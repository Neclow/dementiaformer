"""Delphi: a GPT-2 that reads ages instead of positions."""

# Port of model.py of github.com/gerstung-lab/Delphi at commit fb72166, itself
# based on nanoGPT. Copyright (c) 2024 Gerstung Lab, MIT License (see LICENSE)

import math

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from src.models._delphi import AgeEncoding, attention_mask, init_weights


@dataclass
class DelphiConfig:
    """Settings of Delphi, by default Delphi-2M as Delphi trains it."""

    vocab_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.0
    token_dropout: float = 0.0
    bias: bool = False
    mask_ties: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention under the attention mask."""

    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, width = x.shape
        # Queries, keys and values: (batch, head, token, head width)
        q, k, v = (
            part.view(batch, length, self.n_head, -1).transpose(1, 2)
            for part in self.c_attn(x).split(width, dim=2)
        )
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = self.attn_dropout(att) @ v
        y = y.transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """GPT-2's MLP: four times wider, with a GELU."""

    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """A transformer block: attention, then the MLP, each after a LayerNorm."""

    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))


class Delphi(nn.Module):
    """Delphi: one rate per token for the next event, from tokens and ages."""

    def __init__(self, config: DelphiConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wae = AgeEncoding(config.n_embd)
        self.token_drop = nn.Dropout(config.token_dropout)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # The output layer shares its weights with the token embeddings
        self.wte.weight = self.lm_head.weight
        init_weights(self, config.n_layer)

    def forward(
        self, tokens: Tensor, ages: Tensor, target_ages: Tensor | None = None
    ) -> Tensor:
        """Log rates of the next event: (batch, token, vocabulary)."""
        x = self.drop(self.token_drop(self.wte(tokens)) + self.wae(ages))
        mask = attention_mask(
            tokens, ages, target_ages if self.config.mask_ties else None
        )
        for block in self.h:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))
