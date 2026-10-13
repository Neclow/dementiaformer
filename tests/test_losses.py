"""Tests of the losses against hand computations."""

import math

import torch

from src.training.losses import DAYS_PER_YEAR, delphi_loss, motor_loss, time_in_bins

# Delphi's t_min, in days
T_MIN = 0.1


def exponential_nll(total_rate: float, days: float) -> float:
    """Delphi's time loss for one token, by hand."""
    rate = 1 / (1 / total_rate + T_MIN)
    return -math.log(rate) + rate * (days + T_MIN)


def test_delphi_loss_by_hand():
    # One token at age 0; its target, token 3, comes at day 10
    logits = torch.log(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    loss = delphi_loss(
        logits,
        tokens=torch.tensor([[2]]),
        ages=torch.tensor([[0.0]]),
        targets=torch.tensor([[3]]),
        target_ages=torch.tensor([[10.0]]),
        ignore_tokens=[0],
    )
    assert math.isclose(loss["loss_ce"].item(), -math.log(4 / 10), rel_tol=1e-6)
    assert math.isclose(loss["loss_dt"].item(), exponential_nll(10, 10), rel_tol=1e-6)


def same_day_time_loss(mask_ties: bool) -> float:
    """Delphi's time loss for tokens at days 0 and 10, the second one's target also at day 10."""
    loss = delphi_loss(
        torch.zeros(1, 2, 4),
        tokens=torch.tensor([[2, 3]]),
        ages=torch.tensor([[0.0, 10.0]]),
        targets=torch.tensor([[3, 2]]),
        target_ages=torch.tensor([[10.0, 10.0]]),
        ignore_tokens=[0],
        mask_ties=mask_ties,
    )
    return loss["loss_dt"].item()


def test_delphi_loss_counts_same_day_targets_from_the_last_earlier_day():
    with_ties, without_ties = same_day_time_loss(True), same_day_time_loss(False)
    # With same-day masking, both targets are 10 days after the token at day 0
    assert math.isclose(with_ties, exponential_nll(4, 10), rel_tol=1e-6)
    # Without, the second target is 0 days away, counted as 1 day
    expected = (exponential_nll(4, 10) + exponential_nll(4, 1)) / 2
    assert math.isclose(without_ties, expected, rel_tol=1e-6)


def test_delphi_loss_ignores_left_out_targets():
    logits = torch.log(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    one = delphi_loss(
        logits,
        tokens=torch.tensor([[2]]),
        ages=torch.tensor([[0.0]]),
        targets=torch.tensor([[3]]),
        target_ages=torch.tensor([[10.0]]),
        ignore_tokens=[0, 1],
    )
    # A second token whose target, "no event", is left out
    two = delphi_loss(
        logits.repeat(1, 2, 1),
        tokens=torch.tensor([[2, 3]]),
        ages=torch.tensor([[0.0, 5.0]]),
        targets=torch.tensor([[3, 1]]),
        target_ages=torch.tensor([[10.0, 20.0]]),
        ignore_tokens=[0, 1],
    )
    for name in ["loss_ce", "loss_dt"]:
        assert math.isclose(one[name].item(), two[name].item(), rel_tol=1e-6)


def test_delphi_validation_loss_leaves_out_no_event():
    logits = torch.log(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
    loss = delphi_loss(
        logits,
        tokens=torch.tensor([[2]]),
        ages=torch.tensor([[0.0]]),
        targets=torch.tensor([[3]]),
        target_ages=torch.tensor([[10.0]]),
        ignore_tokens=[0],
        validation=True,
    )
    # Only tokens 2 and 3 keep a rate
    assert math.isclose(loss["loss_ce"].item(), -math.log(4 / 7), rel_tol=1e-6)
    assert math.isclose(loss["loss_dt"].item(), exponential_nll(7, 10), rel_tol=1e-6)


def test_time_in_bins_adds_up():
    years = torch.tensor([0.5, 1.5, 3.0, 30.0])
    bins = time_in_bins(years, torch.tensor([1.0, 2.0, 5.0, 10.0, 15.0, 25.0]))
    torch.testing.assert_close(bins.sum(-1), years)
    torch.testing.assert_close(bins[2], torch.tensor([1.0, 1.0, 1.0, 0, 0, 0, 0]))


def motor_case(padding: bool, noise: float) -> torch.Tensor:
    """MOTOR's loss for one person: token A at age 0, B at 3 years, end at 5 years."""
    # Vocabulary: padding, no event, A, B, C. Bins: 0-1, 1-2 and 2+ years
    log_hazards = noise * torch.randn(1, 1, 3, 5)
    log_hazards[0, 0, :, 3] = torch.log(torch.tensor([0.1, 0.2, 0.3]))
    log_hazards[0, 0, :, 4] = torch.log(torch.tensor([0.4, 0.5, 0.6]))
    tokens, ages = torch.tensor([[2]]), torch.tensor([[0.0]])
    if padding:
        log_hazards = torch.cat([torch.randn(1, 1, 3, 5), log_hazards], dim=1)
        tokens, ages = torch.tensor([[0, 2]]), torch.tensor([[-10000.0, 0.0]])
    inf = float("inf")
    return motor_loss(
        log_hazards,
        tokens=tokens,
        ages=ages,
        first_ages=torch.tensor([[inf, inf, 0.0, 3 * DAYS_PER_YEAR, inf]]),
        end_ages=torch.tensor([5 * DAYS_PER_YEAR]),
        bin_boundaries=(1, 2),
        not_tasks=[0, 1],
    )


def test_motor_loss_by_hand():
    # B happens at 3 years: a year in each bin, then the hazard of bin 3.
    # C never happens: censored at 5 years, so 1, 1 and 3 years in the bins
    expected = (0.1 + 0.2 + 0.3) - math.log(0.3) + (0.4 + 0.5 + 3 * 0.6)
    assert math.isclose(motor_case(False, 0.0).item(), expected, rel_tol=1e-6)


def test_motor_loss_leaves_out_seen_tasks_non_tasks_and_padding():
    # Random log hazards for A (already seen), padding and no event, plus a
    # padding token, change nothing
    torch.manual_seed(0)
    torch.testing.assert_close(motor_case(True, 1.0), motor_case(False, 0.0))
