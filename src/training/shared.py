"""The parts of training shared by both trainers: model, optimiser, learning rate, loss."""

import math
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from torch import Tensor, nn

from src.models.delphi import Delphi, DelphiConfig
from src.models.delphi_v2 import DelphiV2, DelphiV2Config
from src.models.delphi_v2_motorhead import DelphiV2Motorhead, DelphiV2MotorheadConfig
from src.settings import read_data_settings, read_model_settings
from src.training.batches import Batch, RandomBatches, read_timelines
from src.training.losses import NO_EVENT, delphi_loss, motor_loss

# Any of the three models
Model = Delphi | DelphiV2 | DelphiV2Motorhead
# Each model's class and settings class, by the name of its settings file
MODELS: dict[str, tuple[type[Model], type]] = {
    "delphi": (Delphi, DelphiConfig),
    "delphi_v2": (DelphiV2, DelphiV2Config),
    "delphi_v2_motorhead": (DelphiV2Motorhead, DelphiV2MotorheadConfig),
}


@dataclass
class TrainingSettings:
    """The training section of a model's settings file."""

    batch_size: int
    block_size: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    max_steps: int
    beta1: float
    beta2: float
    weight_decay: float
    grad_clip: float
    log_interval: int
    eval_interval: int
    eval_batches: int
    no_event_years: float
    seed: int


def batch_streams(folder: Path, settings: TrainingSettings) -> dict[str, RandomBatches]:
    """A run's batches: "batches" to train on, and "train" and "validation" to evaluate on."""
    timelines = {
        split: read_timelines(folder, split) for split in ["train", "validation"]
    }

    def stream(split: str, training: bool, seed_offset: int) -> RandomBatches:
        """Random batches of a split's people, each stream with its own seed."""
        return RandomBatches(
            timelines[split],
            settings.batch_size,
            settings.block_size,
            training,
            settings.no_event_years,
            settings.seed + seed_offset,
        )

    return {
        "batches": stream("train", True, 0),
        "train": stream("train", False, 1),
        "validation": stream("validation", False, 2),
    }


def build_model(name: str, model_settings: dict[str, Any], vocab_size: int) -> Model:
    """A new model, from the model section of its settings file."""
    model_class, config_class = MODELS[name]
    return model_class(config_class(vocab_size=vocab_size, **model_settings))


def make_optimizer(model: nn.Module, settings: TrainingSettings) -> torch.optim.AdamW:
    """AdamW, with weight decay on the weights of linear layers only, as in Delphi."""
    # Not the token embeddings, even where the output layer shares them
    embeddings = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    }
    decayed = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Linear) and id(module.weight) not in embeddings
    }
    parameters = list(model.parameters())
    groups = [
        {
            "params": [p for p in parameters if id(p) in decayed],
            "weight_decay": settings.weight_decay,
        },
        {
            "params": [p for p in parameters if id(p) not in decayed],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        groups, lr=settings.learning_rate, betas=(settings.beta1, settings.beta2)
    )


def learning_rate(step: int, settings: TrainingSettings) -> float:
    """Delphi's learning rate: a linear warm-up, then a cosine down to the minimum."""
    if step < settings.warmup_steps:
        return settings.learning_rate * step / settings.warmup_steps
    if step > settings.max_steps:
        return settings.min_learning_rate
    progress = (step - settings.warmup_steps) / (
        settings.max_steps - settings.warmup_steps
    )
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return settings.min_learning_rate + weight * (
        settings.learning_rate - settings.min_learning_rate
    )


def loss_terms(
    name: str,
    model: nn.Module,
    batch: Batch,
    left_out: list[int],
    settings: dict[str, Any],
    validation: bool,
) -> dict[str, Tensor]:
    """The model's loss terms on a batch, and their sum as "loss"."""
    output = model(batch.tokens, batch.ages, batch.target_ages)
    if name == "delphi_v2_motorhead":
        # MOTOR's loss also leaves out "no event": it is not an event to predict
        terms = {
            "loss_motor": motor_loss(
                output,
                batch.tokens,
                batch.ages,
                batch.first_ages,
                batch.end_ages,
                tuple(settings["model"]["bin_boundaries"]),
                left_out + [NO_EVENT],
            )
        }
    else:
        terms = delphi_loss(
            output,
            batch.tokens,
            batch.ages,
            batch.targets,
            batch.target_ages,
            left_out,
            t_min=settings["loss"]["t_min"],
            mask_ties=settings["model"]["mask_ties"],
            validation=validation,
        )
    return {**terms, "loss": sum(terms.values(), torch.tensor(0.0))}


def start_run(
    config: str, model: str, variant: str, dry_run: bool, resume: bool, suffix: str = ""
) -> tuple[dict[str, Any], Path, Path | None]:
    """A run's model settings, tokens folder, and folder (None in a dry run), created if new."""
    _, paths = read_data_settings(config)
    settings = read_model_settings(model)
    run = None if dry_run else paths["runs"] / f"{model}_{variant}{suffix}"
    # A new run never overwrites another
    if run is not None and not resume:
        if run.exists():
            sys.exit(f"{run} exists: add --resume to continue it, or delete it")
        run.mkdir(parents=True)
        with open(run / "settings.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(
                {"command": " ".join(sys.argv), **settings}, file, sort_keys=False
            )
    return settings, paths["tokens"] / variant, run
