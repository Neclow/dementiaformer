"""Tests of the batches: windows, "no event" tokens, targets and whole records."""

import numpy as np
import polars as pl
import torch

from src.training.batches import (
    DELPHI_INPUT_ONLY,
    PAD,
    get_batch,
    left_out_tokens,
    read_timelines,
)
from src.training.losses import NO_EVENT
from tests.tiny_data import VOCABULARY, YEAR, write_variant


def test_read_timelines(tmp_path):
    timelines = read_timelines(write_variant(tmp_path / "variant"), "train")
    # P1 then P2, in PNR order; P3 is in validation
    assert timelines.p2i.tolist() == [[0, 6], [6, 3]]
    assert timelines.rows[:, 0].tolist() == [0] * 6 + [1] * 3
    assert timelines.end_ages.tolist() == [round(70 * YEAR), round(35 * YEAR)]
    assert timelines.vocab_size == len(VOCABULARY)


def test_targets_are_the_next_tokens(tmp_path):
    timelines = read_timelines(write_variant(tmp_path / "variant"), "train")
    batch = get_batch(timelines, np.array([0, 1, 0]))
    # Wherever the input is a real token, its target is the next input
    real = batch.tokens[:, :-1] != PAD
    assert torch.equal(batch.targets[:, :-1][real], batch.tokens[:, 1:][real])
    assert torch.equal(batch.target_ages[:, :-1][real], batch.ages[:, 1:][real])
    # Padding's target is padding
    assert (batch.targets[batch.tokens == PAD] == PAD).all()


def test_no_event_tokens(tmp_path):
    timelines = read_timelines(write_variant(tmp_path / "variant"), "train")
    for training in [True, False]:
        batch = get_batch(timelines, np.array([0, 1]), training=training)
        # The last token of each person is the last target
        tokens = torch.cat([batch.tokens, batch.targets[:, -1:]], dim=1)
        ages = torch.cat([batch.ages, batch.target_ages[:, -1:]], dim=1)
        no_event = tokens == NO_EVENT
        assert no_event.any()
        # None after the person's last real token
        last_real = torch.where(tokens > NO_EVENT, ages, -1.0).max(1).values
        assert (torch.where(no_event, ages, -1.0).max(1).values <= last_real).all()
    # In evaluation, every 5 years from day 1: for P2 (dead at 35), 7 of them
    batch = get_batch(timelines, np.array([1]), training=False)
    tokens = torch.cat([batch.tokens, batch.targets[:, -1:]], dim=1)
    assert int((tokens == NO_EVENT).sum()) == 7


def test_window_of_block_size_plus_one(tmp_path):
    timelines = read_timelines(write_variant(tmp_path / "variant"), "train")
    batch = get_batch(timelines, np.array([0]), block_size=3, training=False)
    assert batch.tokens.shape[1] <= 3
    # Only P1's first 4 tokens can be in the window: Female, A, B, C
    tokens = torch.cat([batch.tokens, batch.targets[:, -1:]], dim=1)
    assert set(tokens[tokens > NO_EVENT].tolist()) <= {3, 5, 6, 7}


def test_first_ages_cover_the_whole_record(tmp_path):
    timelines = read_timelines(write_variant(tmp_path / "variant"), "train")
    batch = get_batch(timelines, np.array([0, 1]), block_size=3)
    inf = float("inf")
    # P1's E, at 60, is outside a window of 4 tokens but in first_ages
    days = [round(years * YEAR) for years in [0, 20, 30, 40, 50, 60]]
    assert batch.first_ages[0].tolist() == [inf, inf, inf, *days[:1], inf, *days[1:]]
    assert batch.first_ages[1].tolist() == [
        *[inf, inf, round(35 * YEAR), inf, 0.0, inf],
        *[round(10 * YEAR), inf, inf, inf],
    ]
    assert batch.end_ages.tolist() == [round(70 * YEAR), round(35 * YEAR)]


def test_same_people_same_batch(tmp_path):
    timelines = read_timelines(write_variant(tmp_path / "variant"), "train")
    first, second = get_batch(timelines, np.array([0, 1])), get_batch(
        timelines, np.array([0, 1])
    )
    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(first.ages, second.ages)


def test_left_out_tokens():
    vocabulary = pl.DataFrame({"TOKEN": range(len(VOCABULARY)), "NAME": VOCABULARY})
    assert left_out_tokens(vocabulary, delphi=False) == [0, 3, 4]
    assert left_out_tokens(vocabulary, delphi=True) == DELPHI_INPUT_ONLY
