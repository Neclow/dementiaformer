"""Build the timelines the models read: every person's events with their ages."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import polars as pl
import yaml

from src._config import DEFAULT_CONFIG_DIR, VARIANTS
from src.timelines import make_persons, make_timelines
from src.tokens import (
    cut_at_index,
    make_vocabulary,
    read_delphi_labels,
    select_codes,
    select_events,
    tokenise,
    write,
)


def parse_args() -> Namespace:
    """The command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="settings file in configs/, without .yaml: synthetic or server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(DEFAULT_CONFIG_DIR / f"{args.config}.yaml") as file:
        config = yaml.safe_load(file)
    root = Path(config["root"])
    paths = {name: root / path for name, path in config["paths"].items()}

    print("Reading the data...", end=" ")
    # One row per person (script 2): PNR, SEX, BIRTH_DATE, DEATH_DATE,
    # EMIGRATION_DATE, DEMENTIA_INCIDENT_DATE, EVENT_TYPE, ...
    per_person_raw = pl.read_parquet(paths["persons"])
    # One row per diagnosis at a hospital contact (script 2): PNR, INDATE, DIAG
    per_diagnosis_raw = pl.read_parquet(
        paths["contacts"], columns=["PNR", "INDATE", "DIAG"]
    )
    # One row per border crossing (script 2): PNR, HAEND_DATO, INDUD_KODE (U out,
    # I in)
    per_crossing_raw = pl.read_parquet(paths["migration"])
    # One row per person (split.R): PNR, in_training, fold
    split = pl.read_parquet(paths["split"])
    print("Done")

    print("Making the timelines...", end=" ")
    # One row per person: PNR, SEX, BIRTH_DATE, DEATH_DATE, EMIGRATION_DATE,
    # DEMENTIA_INCIDENT_DATE, SPLIT, INDEX_DATE, WINDOW_START, RECORD_END_DATE,
    # INDEX_AGE_DAYS, OBSERVED_UNTIL_AGE_DAYS
    persons = make_persons(per_person_raw, split, config["index_age"])
    # One row per event: PNR, AGE_DAYS, KIND, EVENT
    timelines = make_timelines(per_diagnosis_raw, per_crossing_raw, persons)
    print("Done")

    # Every person has a split, and no test person has an event from the index
    # date on
    assert persons.height == per_person_raw.height
    assert (
        timelines.join(persons.filter(pl.col("SPLIT") == "test"), on="PNR")
        .filter(pl.col("AGE_DAYS") >= pl.col("INDEX_AGE_DAYS"))
        .is_empty()
    )

    paths["tokens"].mkdir(parents=True, exist_ok=True)
    persons.select(
        "PNR", "SPLIT", "INDEX_AGE_DAYS", "OBSERVED_UNTIL_AGE_DAYS"
    ).write_parquet(paths["tokens"] / "persons.parquet")
    timelines.write_parquet(paths["tokens"] / "timelines.parquet")

    print(f"{persons.height} people, {timelines.height} events -> {paths['tokens']}")
    print(persons["SPLIT"].value_counts().sort("SPLIT"))
    print(timelines["KIND"].value_counts().sort("KIND"))

    # Every variant: the five steps, with the choices of its row in VARIANTS
    # One row per line of Delphi's labels.csv: NAME, TOKEN
    labels = read_delphi_labels(paths["delphi_labels"])
    for variant, choices in VARIANTS.items():
        # One row per event of the variant: PNR, AGE_DAYS, KIND, EVENT
        events = select_events(timelines, choices["events"])
        events = select_codes(events, choices["codes"], labels["NAME"].to_list())
        if choices["cut_at_index"]:
            events = cut_at_index(events, persons)
        # One row per token of the variant's vocabulary: TOKEN, NAME
        vocabulary = make_vocabulary(events, persons, choices["vocabulary"], labels)
        # One row per token of a person: PNR, AGE_DAYS, TOKEN
        tokens = tokenise(events, vocabulary)
        folder = paths["tokens"] / variant
        write(tokens, vocabulary, persons, folder, choices["vocabulary"])
        print(
            f"{variant}: {tokens.height} events, vocabulary of {vocabulary.height} -> {folder}"
        )


if __name__ == "__main__":
    main()
