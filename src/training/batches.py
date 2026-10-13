"""Batches of timelines, built as Delphi's get_batch builds them."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import polars as pl
import torch

from torch import Tensor
from torch.utils.data import IterableDataset

from src.training.losses import NO_EVENT

# Padding: token 0, at Delphi's padding age
PAD = 0
PADDING_AGE = -10000.0
# Delphi's input-only tokens (ignore_tokens in its config/train_delphi.py):
# padding, sex and lifestyle
DELPHI_INPUT_ONLY = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Ours: padding and sex
OUR_INPUT_ONLY = ["PAD", "Female", "Male"]


@dataclass
class Timelines:
    """The tokens of one split's people, laid out as in Delphi's data."""

    # (rows, 3): person (0 to people - 1, in PNR order), age in days, token. A
    # person's rows are together, by age
    rows: np.ndarray
    # (people, 2): each person's first row and number of rows (Delphi's p2i)
    p2i: np.ndarray
    # (people,): age in days at the end of each person's record
    end_ages: np.ndarray
    vocab_size: int


@dataclass
class Batch:
    """Inputs and targets of some people, and for MOTOR's loss their whole records."""

    # (batch, length): tokens, PAD for padding
    tokens: Tensor
    # (batch, length): ages in days, PADDING_AGE for padding
    ages: Tensor
    # (batch, length): the next token and its age
    targets: Tensor
    target_ages: Tensor
    # (batch, vocabulary): first age of each token in the whole record, inf if
    # never
    first_ages: Tensor
    # (batch,): age at the end of the record
    end_ages: Tensor

    def to(self, device: str | torch.device) -> "Batch":
        """The batch on a device: cpu or cuda."""
        return Batch(*(getattr(self, field.name).to(device) for field in fields(self)))


def read_timelines(folder: Path, split: str) -> Timelines:
    """A variant's tokens (events.parquet, vocab.csv) for the people of one split."""
    # One row per token: PNR, AGE_DAYS, TOKEN, by person, age and token
    tokens = pl.read_parquet(folder / "events.parquet")
    # One row per person of the split with tokens: PERSON (0 to people - 1, in
    # PNR order, as in Delphi's .bin files), PNR, OBSERVED_UNTIL_AGE_DAYS
    persons = (
        pl.read_parquet(folder.parent / "persons.parquet")
        .filter(pl.col("SPLIT") == split)
        .join(tokens.select("PNR"), on="PNR", how="semi")
        .sort("PNR")
        .with_row_index("PERSON")
        .select("PERSON", "PNR", "OBSERVED_UNTIL_AGE_DAYS")
    )
    # One row per token of a person of the split: PERSON, AGE_DAYS, TOKEN
    rows = (
        tokens.join(persons.select("PNR", "PERSON"), on="PNR")
        .sort("PERSON", "AGE_DAYS", "TOKEN")
        .select("PERSON", "AGE_DAYS", "TOKEN")
        .to_numpy()
        .astype(np.uint32)
    )
    _, starts, lengths = np.unique(rows[:, 0], return_index=True, return_counts=True)
    return Timelines(
        rows=rows,
        p2i=np.stack([starts, lengths], axis=1),
        end_ages=persons["OBSERVED_UNTIL_AGE_DAYS"].to_numpy(),
        vocab_size=pl.read_csv(folder / "vocab.csv").height,
    )


def left_out_tokens(vocabulary: pl.DataFrame, delphi: bool) -> list[int]:
    """The input-only tokens, left out of the losses: padding, sex, and Delphi's lifestyle."""
    if delphi:
        return DELPHI_INPUT_ONLY
    return vocabulary.filter(pl.col("NAME").is_in(OUR_INPUT_ONLY))["TOKEN"].to_list()


def get_batch(
    timelines: Timelines,
    people: np.ndarray,
    block_size: int = 96,
    training: bool = True,
    no_event_years: float = 5,
) -> Batch:
    """Delphi's get_batch, with the options Delphi trains with, and the whole records."""
    # Port of get_batch in utils.py of Delphi (commit fb72166), with
    # select="left" and cut_batch=True; padding="random" in training and
    # "regular" in evaluation
    people = np.asarray(people)
    batch = len(people)
    # Random, but the same for the same people, as in Delphi
    generator = torch.Generator().manual_seed(int(people.sum()))
    starts, lengths = timelines.p2i[people, 0], timelines.p2i[people, 1]

    # The first block_size + 1 tokens of each person, then padding
    offsets = np.arange(block_size + 1)
    in_window = torch.from_numpy(offsets < lengths[:, None])
    rows = timelines.rows[
        np.minimum(starts[:, None] + offsets, len(timelines.rows) - 1)
    ]
    tokens = torch.from_numpy(rows[..., 2].astype(np.int64)).masked_fill(
        ~in_window, PAD
    )
    ages = torch.from_numpy(rows[..., 1].astype(np.float32))
    ages = ages.masked_fill(~in_window, PADDING_AGE)

    # "No event" tokens: in training, at random days within 100 years, as many
    # as one every no_event_years; in evaluation, every no_event_years from day 1
    if training:
        no_event_ages = torch.randint(
            1, 36525, (batch, int(100 / no_event_years)), generator=generator
        )
    else:
        no_event_ages = (
            torch.arange(0, 36525, 365.25 * no_event_years) * torch.ones(batch, 1) + 1
        )
    last_age = ages.max(1, keepdim=True).values
    tokens = torch.hstack([tokens, torch.full(no_event_ages.shape, NO_EVENT)])
    ages = torch.hstack([ages, no_event_ages])
    # None after the person's last token
    tokens = tokens.masked_fill(ages > last_age, PAD)
    ages = ages.masked_fill(ages > last_age, PADDING_AGE)

    # By age, so padding comes first
    order = torch.argsort(ages, 1)
    tokens, ages = tokens.gather(1, order), ages.gather(1, order)
    # Without the columns that are padding for every person, and at most
    # block_size + 1 columns, the last ones
    first_column = int((tokens == PAD).sum(1).min())
    first_column = max(first_column, tokens.size(1) - block_size - 1)
    tokens, ages = tokens[:, first_column:], ages[:, first_column:]

    # Targets: the next token. A padding token's target is padding too
    inputs, input_ages = tokens[:, :-1], ages[:, :-1]
    targets = tokens[:, 1:].masked_fill(inputs == PAD, PAD)
    target_ages = ages[:, 1:].masked_fill(inputs == PAD, PADDING_AGE)

    # Every token of each person's whole record, for MOTOR's loss
    owner = np.repeat(np.arange(batch), lengths)
    record = np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
    record = record + np.arange(lengths.sum())
    first_ages = torch.full((batch, timelines.vocab_size), float("inf"))
    first_ages[owner, timelines.rows[record, 2].astype(np.int64)] = torch.from_numpy(
        timelines.rows[record, 1].astype(np.float32)
    )
    return Batch(
        tokens=inputs,
        ages=input_ages,
        targets=targets,
        target_ages=target_ages,
        first_ages=first_ages,
        end_ages=torch.from_numpy(timelines.end_ages[people].astype(np.float32)),
    )


class RandomBatches(IterableDataset):
    """Endless batches of random people of one split, as Delphi draws them at each step."""

    def __init__(
        self,
        timelines: Timelines,
        batch_size: int,
        block_size: int,
        training: bool,
        no_event_years: float,
        seed: int,
    ):
        self.timelines = timelines
        self.batch_size = batch_size
        self.block_size = block_size
        self.training = training
        self.no_event_years = no_event_years
        self.seed = seed

    def __iter__(self):
        # A new generator at each pass, so that every pass draws the same
        # people: the same evaluation batches each time, and the same batches
        # at each epoch when overfitting
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            people = torch.randint(
                len(self.timelines.p2i), (self.batch_size,), generator=generator
            )
            yield get_batch(
                self.timelines,
                people.numpy(),
                self.block_size,
                self.training,
                self.no_event_years,
            )
