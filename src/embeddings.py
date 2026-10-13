"""Embeddings at the index age: each person's hidden state after their history before it."""

import numpy as np
import polars as pl
import torch

from torch import Tensor

from src.training.batches import PAD, PADDING_AGE
from src.training.losses import DAYS_PER_YEAR, NO_EVENT
from src.training.shared import Model


def index_sequences(
    events: pl.DataFrame, persons: pl.DataFrame, block_size: int, no_event_years: float
) -> tuple[list[str], Tensor, Tensor]:
    """PNRs, tokens and ages: each person's last block_size tokens up to the index age."""
    # As in the evaluation batches (get_batch): the person's events, "no event"
    # every no_event_years from day 1, then "no event" at the index age, whose
    # hidden state is the embedding. Only what happens before the index age
    # One row per person, in PNR order: ROW, PNR, INDEX_AGE_DAYS
    persons = persons.select("PNR", "INDEX_AGE_DAYS").sort("PNR").with_row_index("ROW")
    regular_ages = np.arange(0, 36525, DAYS_PER_YEAR * no_event_years) + 1
    # One row per token before the index age: ROW, INDEX_AGE_DAYS, AGE_DAYS, TOKEN
    before = pl.concat(
        [
            events.join(persons, on="PNR").select(
                "ROW", "INDEX_AGE_DAYS", pl.col("AGE_DAYS").cast(pl.Float64), "TOKEN"
            ),
            persons.join(pl.DataFrame({"AGE_DAYS": regular_ages}), how="cross").select(
                "ROW", "INDEX_AGE_DAYS", "AGE_DAYS", TOKEN=pl.lit(NO_EVENT)
            ),
        ],
        how="vertical_relaxed",
    ).filter(pl.col("AGE_DAYS") < pl.col("INDEX_AGE_DAYS"))
    # One row per person: "no event" at the index age
    at_index = persons.select(
        "ROW",
        "INDEX_AGE_DAYS",
        AGE_DAYS=pl.col("INDEX_AGE_DAYS").cast(pl.Float64),
        TOKEN=pl.lit(NO_EVENT),
    )
    # The last block_size tokens of each person, right-aligned: padding on the
    # left, as in Delphi's batches
    rows = (
        pl.concat([before, at_index], how="vertical_relaxed")
        .sort("ROW", "AGE_DAYS", "TOKEN")
        .with_columns(FROM_END=pl.int_range(pl.len()).reverse().over("ROW"))
        .filter(pl.col("FROM_END") < block_size)
    )
    row = torch.from_numpy(rows["ROW"].to_numpy().astype(np.int64))
    column = (
        block_size - 1 - torch.from_numpy(rows["FROM_END"].to_numpy().astype(np.int64))
    )
    tokens = torch.full((persons.height, block_size), PAD)
    ages = torch.full((persons.height, block_size), PADDING_AGE)
    tokens[row, column] = torch.from_numpy(rows["TOKEN"].to_numpy().astype(np.int64))
    ages[row, column] = torch.from_numpy(rows["AGE_DAYS"].to_numpy().astype(np.float32))
    return persons["PNR"].to_list(), tokens, ages


def embed(
    model: Model, tokens: Tensor, ages: Tensor, batch_size: int, device: str
) -> np.ndarray:
    """The hidden state at each person's last token, the index age: (people, width)."""
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            hidden = model.hidden(
                tokens[start : start + batch_size].to(device),
                ages[start : start + batch_size].to(device),
            )
            parts.append(hidden[:, -1].float().cpu())
    return torch.cat(parts).numpy()
