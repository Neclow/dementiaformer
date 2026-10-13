"""Embed every person at the index age with a trained model, in the format of the user's script 4."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from src._config import DEFAULT_N_THREADS, VARIANTS
from src.embeddings import embed, index_sequences
from src.settings import read_data_settings, read_model_settings
from src.training.shared import MODELS, TrainingSettings, build_model

# People per forward pass
BATCH_SIZE = 512
# Columns script 5 needs in every embedding file, as checked by script 4
REQUIRED_COLUMNS = [
    "PNR",
    "EVENT_TYPE",
    "FOLLOW_UP_END_AGE",
    "BIRTH_COHORT",
    "DEMENTIA_BEFORE_INDEX",
    "SEX",
    "EDUCATION",
]


def parse_args() -> Namespace:
    """The command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="settings file in configs/data/, without .yaml: synthetic or server",
    )
    parser.add_argument("model", choices=list(MODELS), help="the trained model")
    parser.add_argument(
        "variant", choices=list(VARIANTS), help="the tokens it was trained on"
    )
    parser.add_argument(
        "--lightning",
        action="store_true",
        help="the run of train_lightning.py (best.ckpt), not of train_torch.py "
        "(best.pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="embed with an untrained model, from configs/models/, and write "
        "nothing (no run needed)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_N_THREADS,
        help="CPU threads (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    data_settings, paths = read_data_settings(args.config)
    # The run's folder, as the trainers name it; the file's tag says which
    tag = f"{args.model}_lightning" if args.lightning else args.model
    suffix = "_lightning" if args.lightning else ""
    run = paths["runs"] / f"{args.model}_{args.variant}{suffix}"
    folder = paths["tokens"] / args.variant

    # The model, with the settings and the best weights of its run; in a dry
    # run, untrained, with the settings of configs/models/
    if args.dry_run:
        settings = read_model_settings(args.model)
    else:
        with open(run / "settings.yaml", encoding="utf-8") as file:
            settings = yaml.safe_load(file)
    training = TrainingSettings(**settings["training"])
    model = build_model(
        args.model, settings["model"], pl.read_csv(folder / "vocab.csv").height
    )
    if args.lightning and not args.dry_run:
        checkpoint = torch.load(
            run / "best.ckpt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(
            {
                name.removeprefix("model."): value
                for name, value in checkpoint["state_dict"].items()
            }
        )
    elif not args.dry_run:
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu")["model"])
    model.to(args.device)

    # One row per person, as in script 4: the person columns of the strings
    # file (the same in every string variant)
    people = (
        pl.read_parquet(paths["strings"])
        .drop("STRING_VARIANT", "STRING")
        .unique("PNR")
        .sort("PNR")
    )
    # One row per person: PNR, INDEX_AGE_DAYS
    persons = pl.read_parquet(paths["tokens"] / "persons.parquet").join(
        people.select("PNR"), on="PNR", how="semi"
    )
    source = "an untrained model (dry run)" if args.dry_run else run
    print(f"Embedding {persons.height} people with {source}...", end=" ")
    pnrs, tokens, ages = index_sequences(
        pl.read_parquet(folder / "events.parquet"),
        persons,
        training.block_size,
        training.no_event_years,
    )
    # float32, as in script 4
    embeddings = embed(model, tokens, ages, BATCH_SIZE, args.device).astype(np.float32)
    print("Done")

    # One row per person: the person columns, then embedding_0 to
    # embedding_<width - 1>, as script 5 reads them (starts_with("embedding_"))
    columns = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    arm = people.join(
        pl.DataFrame({"PNR": pnrs}).hstack(
            pl.DataFrame(embeddings, schema=columns, orient="row")
        ),
        on="PNR",
    )
    # The checks of script 4: every person, every column, finite embeddings
    assert arm.height == people.height, (arm.height, people.height)
    assert not set(REQUIRED_COLUMNS) - set(arm.columns)
    assert np.isfinite(arm.select(columns).to_numpy()).all()
    if args.dry_run:
        print(f"Dry run: {arm.height} people, {len(columns)} columns, nothing written")
        return
    path = Path(data_settings["root"]) / (
        f"age{data_settings['index_age']}_dataset_embeddings_{tag}_{args.variant}.parquet"
    )
    arm.write_parquet(path)
    print(f"{arm.height} people, {len(columns)} embedding columns -> {path}")


if __name__ == "__main__":
    main()
