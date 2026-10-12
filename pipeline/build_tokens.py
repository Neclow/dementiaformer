"""Build the models' input: one folder of tokens per variant of a config file.

Reads the client's preprocessed registry files and the split of
5 predictions.R (the paths are in the config), and writes, for each variant,
into <output>/<variant>/:

    events.parquet   PNR, AGE_DAYS, TOKEN: every person's timeline
    persons.parquet  PNR, SPLIT, INDEX_AGE_DAYS, OBSERVED_UNTIL_AGE_DAYS
    vocab.csv        TOKEN, NAME, KIND, CHAPTER, N_TRAIN_PEOPLE
    settings.json    the settings the folder was built with

and prints totals for checking. events.parquet and persons.parquet hold one row
per event or person: they are microdata and stay on the server.

Run from the repository root:

    pixi run build_tokens_all                                          # synthetic data
    python -m pipeline.build_tokens --config configs/data_server.toml  # on the server
    python -m pipeline.build_tokens --variant whole_lives              # one variant only

The decisions behind the tokens are in docs/tokens.md.
"""

import argparse
import json
import tomllib

from dataclasses import asdict
from pathlib import Path

import polars as pl

from src import tokens
from src._config import DEFAULT_CONFIG_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "data_synthetic.toml",
        help="settings file (default: %(default)s)",
    )
    parser.add_argument(
        "--variant", help="build this variant only (default: every variant)"
    )
    args = parser.parse_args()

    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    paths = {name: Path(path) for name, path in config["paths"].items()}
    settings = tokens.TokenSettings(**config["settings"])
    variants = [
        tokens.Variant(name=name, **table)
        for name, table in config["variants"].items()
        if args.variant in (None, name)
    ]
    if not variants:
        parser.error(f"no variant {args.variant!r} in {args.config}")

    # The inputs, the people and their whole timelines are the same for every
    # variant, so they are built once
    inputs = tokens.load_inputs(paths)
    persons = tokens.build_persons(inputs.persons, inputs.split, settings)
    timelines = tokens.build_timelines(
        inputs.contacts, inputs.migration, persons, settings
    )

    for variant in variants:
        variant_timelines, variant_persons = tokens.select_variant(
            timelines, persons, variant
        )
        vocabulary = tokens.build_vocabulary(
            variant_timelines, variant_persons, settings, variant
        )
        events = tokens.to_tokens(variant_timelines, vocabulary, settings)

        # What the split relies on: no test person has an event from the index
        # date on, so the model can never see a test person's outcome
        leaks = events.join(
            variant_persons.filter(pl.col("SPLIT") == "test"), on="PNR"
        ).filter(pl.col("AGE_DAYS") >= pl.col("INDEX_AGE_DAYS"))
        assert leaks.is_empty(), f"{leaks.height} test events after the index date"

        folder = paths["output"] / variant.name
        folder.mkdir(parents=True, exist_ok=True)
        events.write_parquet(folder / "events.parquet")
        variant_persons.select(
            "PNR", "SPLIT", "INDEX_AGE_DAYS", "OBSERVED_UNTIL_AGE_DAYS"
        ).write_parquet(folder / "persons.parquet")
        vocabulary.write_csv(folder / "vocab.csv")
        built_with = {
            "config": str(args.config),
            "settings": asdict(settings),
            "variant": asdict(variant),
        }
        (folder / "settings.json").write_text(
            json.dumps(built_with, indent=2, default=str), encoding="utf-8"
        )

        print(
            f"\n=== {variant.name}: {variant_persons.height:,} people, "
            f"{events.height:,} events, {vocabulary.height} tokens -> {folder}"
        )
        summary = tokens.summarise(
            events, variant_timelines, variant_persons, vocabulary
        )
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
            for title, table in summary.items():
                print(f"\n{title}\n{table}")


if __name__ == "__main__":
    main()
