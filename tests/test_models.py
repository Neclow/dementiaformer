"""Tests of the three models: output shapes, causality and same-day masking."""

import pytest
import torch

from src.models.delphi import Delphi, DelphiConfig
from src.models.delphi_v2 import DelphiV2, DelphiV2Config
from src.models.delphi_v2_motorhead import DelphiV2Motorhead, DelphiV2MotorheadConfig

VOCAB_SIZE = 20
YEAR = 365.25
# Padding ages, as in Delphi's batches
PADDING_AGE = -10000.0


MODELS = ["delphi", "delphi_v2", "delphi_v2_motorhead"]


def small_model(name: str) -> torch.nn.Module:
    """A small model of each kind, in evaluation mode."""
    torch.manual_seed(0)
    if name == "delphi":
        model = Delphi(DelphiConfig(VOCAB_SIZE, n_layer=2, n_head=2, n_embd=16))
    elif name == "delphi_v2":
        model = DelphiV2(DelphiV2Config(VOCAB_SIZE, n_layer=2, n_head=2, n_embd=16))
    else:
        model = DelphiV2Motorhead(
            DelphiV2MotorheadConfig(VOCAB_SIZE, n_layer=2, n_head=2, n_embd=16)
        )
    return model.eval()


def one_person() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens, ages and target ages of one person: padding, then events at 10, 20, 20, 20 and 30."""
    tokens = torch.tensor([[0, 3, 4, 5, 6, 7]])
    years = torch.tensor([[0, 10, 20, 20, 20, 30]]) * YEAR
    ages = torch.where(tokens > 0, years, PADDING_AGE)
    # Each token's target is the next token, the last one's at 40
    target_ages = torch.cat([ages[:, 1:], torch.tensor([[40 * YEAR]])], dim=1)
    target_ages[:, 0] = PADDING_AGE
    return tokens, ages, target_ages


@pytest.mark.parametrize("name", MODELS)
def test_output_shape(name):
    tokens, ages, _ = one_person()
    output = small_model(name)(tokens, ages)
    if name == "delphi_v2_motorhead":
        # Six bin boundaries: seven bins
        assert output.shape == (1, 6, 7, VOCAB_SIZE)
    else:
        assert output.shape == (1, 6, VOCAB_SIZE)


@pytest.mark.parametrize("name", MODELS)
def test_later_tokens_do_not_change_earlier_outputs(name):
    model = small_model(name)
    tokens, ages, _ = one_person()
    changed = tokens.clone()
    changed[0, -1] = 9
    with torch.no_grad():
        before, after = model(tokens, ages), model(changed, ages)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


@pytest.mark.parametrize("name", MODELS)
def test_same_day_tokens_are_hidden(name):
    # Token 5 (at 20) has its target on the same day: it must not see token 4 (at 20)
    model = small_model(name)
    tokens, ages, target_ages = one_person()
    changed = tokens.clone()
    changed[0, 2] = 9
    with torch.no_grad():
        masked = model(tokens, ages, target_ages), model(changed, ages, target_ages)
        unmasked = model(tokens, ages), model(changed, ages)
    torch.testing.assert_close(masked[0][:, 3], masked[1][:, 3])
    # Without target ages, there is no same-day masking
    assert not torch.allclose(unmasked[0][:, 3], unmasked[1][:, 3])


def test_delphi_v2_has_as_many_parameters_as_delphi():
    delphi = Delphi(DelphiConfig(vocab_size=1270))
    delphi_v2 = DelphiV2(DelphiV2Config(vocab_size=1270))
    count = [
        sum(p.numel() for p in model.parameters()) for model in [delphi, delphi_v2]
    ]
    assert abs(count[1] - count[0]) / count[0] < 0.01
