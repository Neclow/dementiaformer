"""Training with plain PyTorch: the loop of Delphi's train.py."""

import time

from pathlib import Path
from typing import Any

import polars as pl
import torch

from torch import nn

from src.training.batches import RandomBatches, left_out_tokens
from src.training.monitor import append_metrics, make_logger, plot_curves
from src.training.shared import (
    TrainingSettings,
    batch_streams,
    build_model,
    learning_rate,
    loss_terms,
    make_optimizer,
)

# A dry run, as Lightning's fast_dev_run: one training batch, then one
# evaluation batch per split
DRY_RUN_EVAL_BATCHES = 1


def evaluate(
    name: str,
    model: nn.Module,
    streams: dict[str, RandomBatches],
    left_out: list[int],
    settings: dict[str, Any],
    batches: int,
    device: str,
) -> dict[str, float]:
    """Mean loss terms on batches of training and of validation people, as Delphi's estimate_loss."""
    model.eval()
    metrics = {}
    with torch.no_grad():
        for split in ["train", "validation"]:
            # The same people at each evaluation
            stream = iter(streams[split])
            for _ in range(batches):
                batch = next(stream).to(device)
                terms = loss_terms(
                    name, model, batch, left_out, settings, validation=True
                )
                for term, value in terms.items():
                    key = f"{split}/{term}"
                    metrics[key] = metrics.get(key, 0.0) + value.item() / batches
    model.train()
    return metrics


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
    torch.manual_seed(training.seed)
    log = make_logger(None if run is None else run / "train.log")
    streams = batch_streams(folder, training)
    people = streams["batches"].timelines
    vocabulary = pl.read_csv(folder / "vocab.csv")
    left_out = left_out_tokens(vocabulary, delphi_vocabulary)
    model = build_model(name, settings["model"], people.vocab_size).to(device)
    optimizer = make_optimizer(model, training)

    # The step to start at, and the best validation loss so far
    start, best = 0, float("inf")
    if resume and run is not None:
        checkpoint = torch.load(run / "last.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["step"], checkpoint["best_validation_loss"]
        log.info(f"Resuming {run} at step {start}")
    last_step = start + 1 if dry_run else training.max_steps
    batches = iter(streams["batches"])
    # Rows of metrics.csv not yet written: written at each evaluation, with the
    # checkpoint, so that the file always matches last.pt
    rows = []
    log.info(
        f"{name} on {folder.name}: {sum(p.numel() for p in model.parameters()):,} "
        f"parameters, {len(people.p2i)} training and "
        f"{len(streams['validation'].timelines.p2i)} validation people, on {device}"
    )

    for step in range(start, last_step + 1):
        rate = learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = rate

        # Evaluation every eval_interval steps and at the last step, not at
        # the first (as in Delphi)
        if step > start and (step % training.eval_interval == 0 or step == last_step):
            metrics = evaluate(
                name,
                model,
                streams,
                left_out,
                settings,
                # In a dry run, one batch; when overfitting, overfit_batches
                # batches, as Lightning's overfit_batches
                (
                    DRY_RUN_EVAL_BATCHES
                    if dry_run
                    else overfit_batches or training.eval_batches
                ),
                device,
            )
            # Labelled with the last training step before it, as Lightning's
            # CSVLogger labels it
            log.info(
                f"step {step - 1}: training people {metrics['train/loss']:.4f}, "
                f"validation people {metrics['validation/loss']:.4f}"
            )
            if run is not None:
                append_metrics(
                    run / "metrics.csv", [*rows, {"step": step - 1, **metrics}]
                )
                rows = []
                plot_curves(run / "metrics.csv", run / "curves.png")
                best = min(best, metrics["validation/loss"])
                # The step to resume at: this step's training is still to do
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_validation_loss": best,
                    "settings": settings,
                }
                if metrics["validation/loss"] == best:
                    torch.save(checkpoint, run / "best.pt")
                torch.save(checkpoint, run / "last.pt")
        if step == last_step:
            break

        # One training step
        started = time.perf_counter()
        # With overfit_batches, the same batches again every overfit_batches
        # steps, as Lightning's overfit_batches
        if overfit_batches and (step - start) % overfit_batches == 0:
            batches = iter(streams["batches"])
        batch = next(batches).to(device)
        terms = loss_terms(name, model, batch, left_out, settings, validation=False)
        terms["loss"].backward()
        if training.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seconds = time.perf_counter() - started

        # Every log_interval steps, at the end of the interval, as Lightning
        if dry_run or (step + 1) % training.log_interval == 0:
            log.info(
                f"step {step}: batch loss {terms['loss'].item():.4f}, "
                f"learning rate {rate:.2e}, {seconds:.3f} s"
            )
            rows.append(
                {
                    "step": step,
                    "learning_rate": rate,
                    "seconds_per_step": seconds,
                    **{f"batch/{term}": v.item() for term, v in terms.items()},
                }
            )
