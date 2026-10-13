"""Parts of Delphi shared by the three models: age encoding, attention mask, weights."""

# From model.py of github.com/gerstung-lab/Delphi at commit fb72166, itself
# based on nanoGPT. Copyright (c) 2024 Gerstung Lab, MIT License (see LICENSE)

import math

import torch

from torch import Tensor, nn


def attention_mask(
    tokens: Tensor, ages: Tensor, target_ages: Tensor | None = None
) -> Tensor:
    """Which tokens each token attends to: (batch, 1, query, key) booleans."""
    size = tokens.size(1)
    causal = torch.ones(size, size, dtype=torch.bool, device=tokens.device).tril()
    diagonal = torch.eye(size, dtype=torch.bool, device=tokens.device)
    real = tokens > 0
    # Real tokens attend to themselves and to earlier real tokens
    mask = real[:, None, None, :] & real[:, None, :, None] & causal
    if target_ages is not None:
        # Same-day masking: a token does not attend to the tokens dated as its
        # target, because the order of same-day events is arbitrary
        mask &= ages[:, None, None, :] != target_ages[:, None, :, None]
        # A token left with nothing to attend to attends to itself
        mask |= ~mask.any(-1, keepdim=True) & diagonal
    # Padding attends to itself, so that the softmax is defined
    return mask | (~real[:, None, None, :] & diagonal)


def init_weights(model: nn.Module, n_layer: int) -> None:
    """GPT-2's initial weights: normal with sd 0.02, smaller for the residual outputs."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, parameter in model.named_parameters():
        if name.endswith("c_proj.weight"):
            nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))


class AgeEncoding(nn.Module):
    """Sines and cosines of the age in years, then a linear layer."""

    div_term: Tensor

    def __init__(self, n_embd: int):
        super().__init__()
        # Frequencies from 1 to about 1/10,000 per year, as in the position
        # encoding of the original Transformer
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        self.register_buffer("div_term", div_term)
        self.linear = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, ages: Tensor) -> Tensor:
        angles = ages[..., None] / 365.25 * self.div_term
        # Sines in the even features, cosines in the odd ones
        waves = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.linear(waves)
