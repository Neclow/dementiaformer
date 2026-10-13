"""Losses: Delphi's (the next event and the time to it) and MOTOR's (piecewise exponential)."""

import torch

from torch import Tensor
from torch.nn import functional as F

from src.models._delphi import attention_mask

DAYS_PER_YEAR = 365.25
# The "no event" token, in Delphi's vocabulary and in ours
NO_EVENT = 1


def delphi_loss(
    logits: Tensor,
    tokens: Tensor,
    ages: Tensor,
    targets: Tensor,
    target_ages: Tensor,
    ignore_tokens: list[int],
    t_min: float = 0.1,
    mask_ties: bool = True,
    validation: bool = False,
) -> dict[str, Tensor]:
    """Delphi's loss: cross-entropy of the next token, exponential likelihood of the time to it."""
    # Port of the loss in the forward of Delphi's model.py (commit fb72166)
    if validation:
        # Validation also leaves out "no event", and left-out tokens get no rate
        ignore_tokens = ignore_tokens + [NO_EVENT]
        left_out = torch.tensor(ignore_tokens, device=logits.device)
        logits = logits.index_fill(-1, left_out, float("-inf"))
    kept = ~torch.isin(targets, torch.tensor(ignore_tokens, device=targets.device))
    loss_ce = F.cross_entropy(logits[kept], targets[kept])
    # Log of the total rate of events, with the waiting time (in days) bounded
    # below by t_min
    lse = torch.logsumexp(logits, dim=-1)
    lse = -torch.log(torch.exp(-lse) + t_min)
    dt = torch.clamp(target_ages - ages, min=1.0)
    if mask_ties:
        # Time from the last token not dated as the target: the last one the
        # token attends to under same-day masking
        mask = attention_mask(tokens, ages, target_ages)
        positions = torch.arange(tokens.size(1), device=tokens.device)
        dt = dt.gather(-1, (mask * positions).max(-1).indices.squeeze(1))
    ldt = -torch.log(dt + t_min)
    # Exponential log-likelihood of waiting dt
    loss_dt = -(lse - torch.exp(lse - ldt))
    return {"loss_ce": loss_ce, "loss_dt": loss_dt[kept].mean()}


def time_in_bins(years: Tensor, bin_boundaries: Tensor) -> Tensor:
    """Years spent in each time bin before the given times: (..., bin)."""
    starts = torch.cat([bin_boundaries.new_zeros(1), bin_boundaries])
    ends = torch.cat([bin_boundaries, bin_boundaries.new_full((1,), float("inf"))])
    return torch.minimum((years[..., None] - starts).clamp(min=0), ends - starts)


def motor_loss(
    log_hazards: Tensor,
    tokens: Tensor,
    ages: Tensor,
    first_ages: Tensor,
    end_ages: Tensor,
    bin_boundaries: tuple[float, ...],
    not_tasks: list[int],
) -> Tensor:
    """MOTOR's loss: piecewise-exponential likelihood of the time to every task, after each token."""
    # Log hazards per year: (batch, token, vocabulary, bin)
    log_hazards = log_hazards.transpose(2, 3)
    vocabulary = torch.arange(log_hazards.size(2), device=tokens.device)
    is_task = ~torch.isin(vocabulary, torch.tensor(not_tasks, device=tokens.device))
    real = tokens > 0
    # Per token and task: (batch, token, vocabulary)
    first = first_ages[:, None, :]
    age = ages[:, :, None]
    # A task is left out after a token if it happened on or before that day:
    # same-day events count from the last earlier day, as in Delphi
    kept = real[..., None] & is_task & (first > age)
    happens = kept & first.isfinite()
    # Years to the task, or to the end of the record if the task never happens
    years = torch.where(happens, first - age, end_ages[:, None, None] - age)
    years = years / DAYS_PER_YEAR
    boundaries = torch.tensor(bin_boundaries, dtype=years.dtype, device=years.device)
    cumulative_hazard = (time_in_bins(years, boundaries) * log_hazards.exp()).sum(-1)
    # The log hazard of the bin in which the task happens
    event_bin = torch.bucketize(years, boundaries, right=True)
    log_hazard = log_hazards.gather(-1, event_bin[..., None]).squeeze(-1)
    nll = torch.where(kept, cumulative_hazard - happens * log_hazard, 0.0)
    # Summed over tasks, averaged over tokens
    return nll.sum(-1)[real].mean()
