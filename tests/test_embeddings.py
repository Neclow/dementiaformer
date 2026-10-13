"""Tests of the embeddings at the index age: what the model reads, and nothing after it."""

import polars as pl
import torch

from src.embeddings import embed, index_sequences
from src.models.delphi import Delphi, DelphiConfig
from src.training.batches import PAD
from src.training.losses import NO_EVENT

YEAR = 365.25
INDEX_AGE_DAYS = round(70 * YEAR)
PERSONS = pl.DataFrame({"PNR": ["B", "A"], "INDEX_AGE_DAYS": [INDEX_AGE_DAYS] * 2})


def events(later: bool = True, earlier_token: int = 5) -> pl.DataFrame:
    """A at 10 and 69 (and 71 if later), B at 0: PNR, AGE_DAYS, TOKEN."""
    rows = [("A", 10, earlier_token), ("A", 69, 6), ("B", 0, 3)]
    if later:
        rows.append(("A", 71, 7))
    return pl.DataFrame(
        [(pnr, round(years * YEAR), token) for pnr, years, token in rows],
        schema=["PNR", "AGE_DAYS", "TOKEN"],
        orient="row",
    )


def test_index_sequences_stop_at_the_index_age():
    pnrs, tokens, ages = index_sequences(events(), PERSONS, 20, 5)
    assert pnrs == ["A", "B"]
    # The last token is "no event" at the index age, and nothing comes after it
    assert (tokens[:, -1] == NO_EVENT).all()
    assert (ages[:, -1] == INDEX_AGE_DAYS).all()
    assert 7 not in tokens[0].tolist()
    # A's events, among "no event" every 5 years from day 1, padding first
    real = tokens[0][tokens[0] > NO_EVENT].tolist()
    assert real == [5, 6]
    assert tokens[0, 0] == PAD
    assert ((tokens[0] == NO_EVENT) & (ages[0] < INDEX_AGE_DAYS)).sum() == 14


def test_index_sequences_keep_the_last_tokens():
    _, tokens, ages = index_sequences(events(), PERSONS, 3, 5)
    # A: "no event" at 65 years and 1 day, the event at 69, "no event" at 70
    assert tokens[0].tolist() == [NO_EVENT, 6, NO_EVENT]
    assert ages[0, 1] == round(69 * YEAR)


def test_embeddings_see_the_history_before_the_index_age_only():
    torch.manual_seed(0)
    model = Delphi(DelphiConfig(10, n_layer=1, n_head=2, n_embd=16))

    def embeddings(later: bool, earlier_token: int):
        """The embeddings of A and B."""
        _, tokens, ages = index_sequences(events(later, earlier_token), PERSONS, 20, 5)
        return embed(model, tokens, ages, batch_size=1, device="cpu")

    base = embeddings(later=True, earlier_token=5)
    assert base.shape == (2, 16)
    # An event after the index age changes nothing; an earlier one does
    assert (embeddings(later=False, earlier_token=5) == base).all()
    assert (embeddings(later=True, earlier_token=8)[0] != base[0]).any()
