"""A tiny variant of tokens for the tests: three people, eleven tokens."""

import polars as pl

YEAR = 365.25
VOCABULARY = ["PAD", "NO_EVENT", "DEATH", "Female", "Male", "A", "B", "C", "D", "E"]
# Tokens of three people (P3 in validation): PNR, age in years, token
EVENTS = [
    ("P1", 0, "Female"),
    ("P1", 20, "A"),
    ("P1", 30, "B"),
    ("P1", 40, "C"),
    ("P1", 50, "D"),
    ("P1", 60, "E"),
    ("P2", 0, "Male"),
    ("P2", 10, "B"),
    ("P2", 35, "DEATH"),
    ("P3", 0, "Female"),
    ("P3", 5, "A"),
]


def write_variant(folder):
    """A variant's files, and persons.parquet next to it."""
    folder.mkdir()
    token = {name: number for number, name in enumerate(VOCABULARY)}
    pl.DataFrame(
        [(pnr, round(years * YEAR), token[name]) for pnr, years, name in EVENTS],
        schema=["PNR", "AGE_DAYS", "TOKEN"],
        orient="row",
    ).write_parquet(folder / "events.parquet")
    pl.DataFrame({"TOKEN": range(len(VOCABULARY)), "NAME": VOCABULARY}).write_csv(
        folder / "vocab.csv"
    )
    pl.DataFrame(
        {
            "PNR": ["P1", "P2", "P3"],
            "SPLIT": ["train", "train", "validation"],
            "OBSERVED_UNTIL_AGE_DAYS": [
                round(70 * YEAR),
                round(35 * YEAR),
                round(10 * YEAR),
            ],
        }
    ).write_parquet(folder.parent / "persons.parquet")
    return folder
