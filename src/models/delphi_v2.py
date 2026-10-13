"""Delphi v2: Delphi's head on a modern backbone."""

import math

from dataclasses import dataclass

import torch

from torch import Tensor, nn
from torch.nn import functional as F

from src.models._delphi import AgeEncoding, attention_mask, init_weights


@dataclass
class DelphiV2Config:
    """Settings of Delphi v2, by default the size of Delphi-2M."""

    vocab_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.0
    token_dropout: float = 0.0
    mask_ties: bool = True


def rotate(x: Tensor, years: Tensor) -> Tensor:
    """Rotary encoding on age: turn pairs of features by angles proportional to age."""
    half = x.size(-1) // 2
    # Frequencies from 1 to about 1/10,000 per year, as in Delphi's age encoding
    frequencies = torch.exp(
        torch.arange(half, device=x.device) * (-math.log(10000.0) / half)
    )
    # Angles: (batch, 1, token, half head width)
    angles = years[:, None, :, None] * frequencies
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        [
            first * angles.cos() - second * angles.sin(),
            first * angles.sin() + second * angles.cos(),
        ],
        dim=-1,
    )


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention under the attention mask, with RoPE on age."""

    def __init__(self, config: DelphiV2Config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout

    def forward(self, x: Tensor, years: Tensor, mask: Tensor) -> Tensor:
        batch, length, width = x.shape
        # Queries, keys and values: (batch, head, token, head width)
        q, k, v = (
            part.view(batch, length, self.n_head, -1).transpose(1, 2)
            for part in self.c_attn(x).split(width, dim=2)
        )
        # Rotating queries and keys by their ages makes attention depend on
        # the time between two tokens
        y = F.scaled_dot_product_attention(
            rotate(q, years),
            rotate(k, years),
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    """Gated MLP, 8/3 times wider so that it has as many parameters as GPT-2's."""

    def __init__(self, config: DelphiV2Config):
        super().__init__()
        hidden = round(8 * config.n_embd / 3)
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        return self.dropout(self.c_proj(F.silu(gate) * value))


class Block(nn.Module):
    """A transformer block: attention, then the MLP, each after an RMSNorm."""

    def __init__(self, config: DelphiV2Config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, years: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x), years, mask)
        return x + self.mlp(self.ln_2(x))


class Backbone(nn.Module):
    """Token and age embeddings, then the blocks: one hidden state per token."""

    def __init__(self, config: DelphiV2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Absolute age, as in Delphi: RoPE alone only sees differences in age
        self.wae = AgeEncoding(config.n_embd)
        self.token_drop = nn.Dropout(config.token_dropout)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)

    def forward(
        self, tokens: Tensor, ages: Tensor, target_ages: Tensor | None = None
    ) -> Tensor:
        """Hidden states: (batch, token, width)."""
        x = self.drop(self.token_drop(self.wte(tokens)) + self.wae(ages))
        mask = attention_mask(
            tokens, ages, target_ages if self.config.mask_ties else None
        )
        years = ages / 365.25
        for block in self.h:
            x = block(x, years, mask)
        return self.ln_f(x)


class DelphiV2(nn.Module):
    """Delphi v2: one rate per token for the next event, as in Delphi."""

    def __init__(self, config: DelphiV2Config):
        super().__init__()
        self.config = config
        self.backbone = Backbone(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # The output layer shares its weights with the token embeddings
        self.backbone.wte.weight = self.lm_head.weight
        init_weights(self, config.n_layer)

    def forward(
        self, tokens: Tensor, ages: Tensor, target_ages: Tensor | None = None
    ) -> Tensor:
        """Log rates of the next event: (batch, token, vocabulary)."""
        return self.lm_head(self.backbone(tokens, ages, target_ages))
