"""Training with Lightning: the model, loss, optimiser, learning rate and batches of train_torch.py."""

import csv
import logging
import time
import warnings

from pathlib import Path
from typing import Any

import polars as pl
import torch

from lightning.pytorch import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.utils.data import DataLoader

from src.training.batches import Batch, left_out_tokens
from src.training.monitor import make_logger, plot_curves
from src.training.shared import (
    TrainingSettings,
    batch_streams,
    build_model,
    learning_rate,
    loss_terms,
    make_optimizer,
)

# The data loaders read the batches in the main process (num_workers=0): worker
# processes are slow to start on Windows, and the batches are cheap to build
warnings.filterwarnings("ignore", ".*does not have many workers.*")
# The run's folder also holds settings.yaml and train.log
warnings.filterwarnings("ignore", ".*exists and is not empty.*")

# The people of each evaluation loader, in the order of the loaders
EVALUATION_SPLITS = ["train", "validation"]


class DelphiModule(LightningModule):
    """One of the three models, with its loss, optimiser and learning rate."""

    def __init__(
        self,
        name: str,
        settings: dict[str, Any],
        vocab_size: int,
        left_out: list[int],
    ):
        super().__init__()
        self.name = name
        self.settings = settings
        self.training_settings = TrainingSettings(**settings["training"])
        self.left_out = left_out
        self.model = build_model(name, settings["model"], vocab_size)

    def transfer_batch_to_device(
        self, batch: Batch, device: torch.device, dataloader_idx: int
    ) -> Batch:
        return batch.to(device)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        terms = loss_terms(
            self.name, self.model, batch, self.left_out, self.settings, False
        )
        for term, value in terms.items():
            self.log(f"batch/{term}", value, batch_size=batch.tokens.size(0))
        self.log(
            "learning_rate", learning_rate(self.global_step, self.training_settings)
        )
        return terms["loss"]

    def validation_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        terms = loss_terms(
            self.name, self.model, batch, self.left_out, self.settings, True
        )
        # Averaged over the evaluation batches of each loader
        split = EVALUATION_SPLITS[dataloader_idx]
        for term, value in terms.items():
            self.log(
                f"{split}/{term}",
                value,
                batch_size=batch.tokens.size(0),
                add_dataloader_idx=False,
            )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = make_optimizer(self.model, self.training_settings)
        # Delphi's learning rate at each step, as a share of the peak
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: learning_rate(step, self.training_settings)
            / self.training_settings.learning_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": schedule, "interval": "step"},
        }


class KeepingWriter(ExperimentWriter):
    """Lightning's writer of metrics.csv, but keeping an earlier metrics.csv."""

    def _check_log_dir_exists(self) -> None:
        # Lightning deletes an earlier metrics.csv here. Instead, its columns
        # are kept, and --resume appends to it
        if Path(self.metrics_file_path).exists():
            with open(self.metrics_file_path, newline="", encoding="utf-8") as file:
                self.metrics_keys = next(csv.reader(file))


class RunLogger(CSVLogger):
    """Lightning's CSVLogger, writing metrics.csv into the run's folder at each evaluation."""

    def __init__(self, run: Path):
        # Written only at evaluations (by Monitor), with the checkpoint, so that
        # metrics.csv always matches last.ckpt
        super().__init__(run, name="", version="", flush_logs_every_n_steps=10**12)
        self._experiment = KeepingWriter(log_dir=self.log_dir)


class Monitor(Callback):
    """The lines of train.log and curves.png, as train_torch.py writes them."""

    def __init__(
        self,
        log: logging.Logger,
        run: Path | None,
        settings: TrainingSettings,
        dry_run: bool,
    ):
        self.run_log = log
        self.run = run
        self.settings = settings
        self.dry_run = dry_run
        self.started = 0.0

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.started = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        seconds = time.perf_counter() - self.started
        pl_module.log("seconds_per_step", seconds)
        # The step just done: global_step already counts it
        step = trainer.global_step - 1
        # Every log_interval steps, as Lightning logs to metrics.csv
        if self.dry_run or (step + 1) % self.settings.log_interval == 0:
            rate = learning_rate(step, self.settings)
            self.run_log.info(
                f"step {step}: batch loss {outputs['loss'].item():.4f}, "
                f"learning rate {rate:.2e}, {seconds:.3f} s"
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = trainer.callback_metrics
        # Labelled with the last training step before it, as in metrics.csv
        self.run_log.info(
            f"step {trainer.global_step - 1}: training people "
            f"{metrics['train/loss'].item():.4f}, validation people "
            f"{metrics['validation/loss'].item():.4f}"
        )
        if self.run is not None and trainer.logger is not None:
            # Write the logged rows first: Lightning's CSVLogger keeps them a
            # while before writing them
            trainer.logger.save()
            plot_curves(self.run / "metrics.csv", self.run / "curves.png")


def train(
    name: str,
    settings: dict[str, Any],
    folder: Path,
    delphi_vocabulary: bool,
    run: Path | None,
    device: str = "cpu",
    resume: bool = False,
    dry_run: bool = False,
    overfit_batches: int | None = None,
) -> None:
    """Train a model on the tokens in folder, writing into run (nothing if None)."""
    training = TrainingSettings(**settings["training"])
    seed_everything(training.seed)
    log = make_logger(None if run is None else run / "train.log")
    streams = batch_streams(folder, training)
    people = streams["batches"].timelines
    vocabulary = pl.read_csv(folder / "vocab.csv")
    left_out = left_out_tokens(vocabulary, delphi_vocabulary)
    module = DelphiModule(name, settings, people.vocab_size, left_out)
    log.info(
        f"{name} on {folder.name}: "
        f"{sum(p.numel() for p in module.parameters()):,} parameters, "
        f"{len(people.p2i)} training and "
        f"{len(streams['validation'].timelines.p2i)} validation people, on {device}"
    )

    trainer = Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_steps=training.max_steps,
        # Random people at each step, without epochs, as in Delphi
        max_epochs=-1,
        # Evaluation every eval_interval steps, on eval_batches batches
        val_check_interval=training.eval_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=training.eval_batches,
        num_sanity_val_steps=0,
        gradient_clip_val=training.grad_clip,
        log_every_n_steps=training.log_interval,
        # metrics.csv, best.ckpt and last.ckpt in the run's folder
        logger=False if run is None else RunLogger(run),
        enable_checkpointing=run is not None,
        callbacks=[
            Monitor(log, run, training, dry_run),
            *(
                []
                if run is None
                else [
                    ModelCheckpoint(
                        dirpath=run,
                        filename="best",
                        monitor="validation/loss",
                        save_last=True,
                    )
                ]
            ),
        ],
        fast_dev_run=dry_run,
        overfit_batches=overfit_batches or 0.0,
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(streams["batches"], batch_size=None),
        val_dataloaders=[
            DataLoader(streams[split], batch_size=None) for split in EVALUATION_SPLITS
        ],
        ckpt_path=run / "last.ckpt" if resume and run is not None else None,
    )
