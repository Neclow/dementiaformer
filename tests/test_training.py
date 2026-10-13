"""Tests of training: learning rate, optimiser, and short runs of both trainers."""

import math

from pathlib import Path

import polars as pl
import pytest
import torch

from src.models.delphi import Delphi, DelphiConfig
from src.training.lightning_trainer import train as train_lightning
from src.training.shared import TrainingSettings, learning_rate, make_optimizer
from src.training.torch_trainer import train as train_torch
from tests.tiny_data import write_variant


def small_settings(max_steps: int) -> dict:
    """Settings of a small Delphi, trained fast on the tiny variant."""
    return {
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 16, "mask_ties": True},
        "loss": {"t_min": 0.1},
        "training": {
            "batch_size": 4,
            "block_size": 8,
            "learning_rate": 1e-2,
            "min_learning_rate": 1e-3,
            "warmup_steps": 1,
            "max_steps": max_steps,
            "beta1": 0.9,
            "beta2": 0.99,
            "weight_decay": 0.2,
            "grad_clip": 1.0,
            "log_interval": 1,
            "eval_interval": 3,
            "eval_batches": 1,
            "no_event_years": 5,
            "seed": 0,
        },
    }


def test_learning_rate():
    settings = TrainingSettings(**small_settings(100)["training"])
    settings.warmup_steps = 10
    rates = [learning_rate(step, settings) for step in range(201)]
    # Warm-up from 0, then a cosine from the peak down to the minimum
    assert rates[0] == 0 and math.isclose(rates[5], 0.5e-2)
    assert math.isclose(rates[10], 1e-2) and math.isclose(rates[100], 1e-3)
    assert all(a > b for a, b in zip(rates[10:100], rates[11:101]))
    assert rates[200] == 1e-3


def test_optimizer_decays_the_weights_of_linear_layers_only():
    model = Delphi(DelphiConfig(20, n_layer=1, n_head=2, n_embd=16))
    settings = TrainingSettings(**small_settings(10)["training"])
    decayed, other = make_optimizer(model, settings).param_groups
    assert decayed["weight_decay"] == 0.2 and other["weight_decay"] == 0
    # Every parameter once; the token embeddings, shared with the output layer,
    # and the LayerNorms are not decayed
    ids = [id(p) for p in decayed["params"] + other["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    parameters = dict(model.named_parameters())
    assert id(parameters["wte.weight"]) in {id(p) for p in other["params"]}
    assert id(parameters["h.0.ln_1.weight"]) in {id(p) for p in other["params"]}
    assert id(parameters["h.0.attn.c_attn.weight"]) in {
        id(p) for p in decayed["params"]
    }


# Each trainer, and the extension of its checkpoints
TRAINERS = {"torch": (train_torch, "pt"), "lightning": (train_lightning, "ckpt")}


def saved_step(run: Path, trainer: str) -> int:
    """The step a run would resume at, from its last checkpoint."""
    if trainer == "torch":
        return torch.load(run / "last.pt")["step"]
    return torch.load(run / "last.ckpt", weights_only=False)["global_step"]


@pytest.mark.parametrize("trainer", TRAINERS)
def test_overfitting_one_batch(tmp_path, trainer):
    folder = write_variant(tmp_path / "variant")
    run = tmp_path / "run"
    run.mkdir()
    settings = small_settings(40)
    settings["training"]["eval_interval"] = 100
    train, _ = TRAINERS[trainer]
    train("delphi", settings, folder, False, run, overfit_batches=1)
    losses = pl.read_csv(run / "metrics.csv")["batch/loss"].drop_nulls()
    assert losses[-1] < losses[0] / 2


@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("name", ["delphi", "delphi_v2_motorhead"])
def test_run_files_and_resume(tmp_path, trainer, name):
    folder = write_variant(tmp_path / "variant")
    run = tmp_path / "run"
    run.mkdir()
    settings = small_settings(6)
    if name == "delphi_v2_motorhead":
        settings["model"]["bin_boundaries"] = [1, 5]
    train, extension = TRAINERS[trainer]
    train(name, settings, folder, False, run)
    for file in ["train.log", "metrics.csv", "curves.png"]:
        assert (run / file).exists()
    for file in ["best", "last"]:
        assert (run / f"{file}.{extension}").exists()
    assert saved_step(run, trainer) == 6

    # Resumed up to step 9: every training step once in metrics.csv, and
    # evaluations after steps 2, 5 and 8
    settings["training"]["max_steps"] = 9
    train(name, settings, folder, False, run, resume=True)
    assert saved_step(run, trainer) == 9
    metrics = pl.read_csv(run / "metrics.csv")
    assert metrics.filter(pl.col("batch/loss").is_not_null())["step"].to_list() == list(
        range(9)
    )
    evaluations = metrics.filter(pl.col("validation/loss").is_not_null())
    assert evaluations["step"].to_list() == [2, 5, 8]
    assert evaluations["validation/loss"].is_finite().all()


def test_both_trainers_give_the_same_losses(tmp_path):
    folder = write_variant(tmp_path / "variant")
    metrics = {}
    for trainer, (train, _) in TRAINERS.items():
        run = tmp_path / trainer
        run.mkdir()
        train("delphi", small_settings(6), folder, False, run)
        metrics[trainer] = pl.read_csv(run / "metrics.csv")
    for column in ["batch/loss", "train/loss", "validation/loss"]:
        torch_rows, lightning_rows = (
            metrics[trainer].filter(pl.col(column).is_not_null()).select("step", column)
            for trainer in TRAINERS
        )
        assert torch_rows["step"].to_list() == lightning_rows["step"].to_list()
        # The evaluations are averaged in float32 by Lightning
        torch.testing.assert_close(
            torch_rows[column].to_torch(),
            lightning_rows[column].to_torch(),
            rtol=1e-5,
            atol=0,
        )


@pytest.mark.parametrize("trainer", TRAINERS)
def test_dry_run_writes_nothing(tmp_path, caplog, trainer):
    folder = write_variant(tmp_path / "variant")
    train, _ = TRAINERS[trainer]
    with caplog.at_level("INFO", logger="dementiaformer"):
        train("delphi", small_settings(100), folder, False, None, dry_run=True)
    # One training step, then one evaluation after it (after the first line,
    # on the model)
    messages = [
        record.message for record in caplog.records if record.name == "dementiaformer"
    ][1:]
    assert messages[0].startswith("step 0: batch loss")
    assert messages[1].startswith("step 0: training people")
    assert len(messages) == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "persons.parquet",
        "variant",
    ]
