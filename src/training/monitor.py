"""Monitoring a run offline: train.log, metrics.csv and curves.png."""

import csv
import logging

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

# The losses drawn in curves.png: the column of metrics.csv and its label
CURVES = {
    "batch/loss": "training batches",
    "train/loss": "training people (evaluation)",
    "validation/loss": "validation people (evaluation)",
}


def make_logger(log_file: Path | None) -> logging.Logger:
    """A logger that prints, with the time, and also writes to log_file if given."""
    logger = logging.getLogger("dementiaformer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def append_metrics(path: Path, new_rows: list[dict[str, float]]) -> None:
    """Add rows to metrics.csv, laid out as Lightning's CSVLogger lays it out."""
    # Columns: every name logged so far, sorted. Rows without a value leave
    # the cell empty
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    rows.extend(new_rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(set().union(*rows)))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(metrics_path: Path, image_path: Path) -> None:
    """Draw the losses of metrics.csv against the step, into curves.png."""
    metrics = pl.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in CURVES.items():
        if column in metrics.columns:
            points = metrics.select("step", column).drop_nulls()
            # The batch losses are noisy: drawn lighter. The evaluations get
            # markers, so that a single one shows
            batches = column == "batch/loss"
            ax.plot(
                points["step"],
                points[column],
                label=label,
                alpha=0.4 if batches else 1.0,
                marker="" if batches else "o",
            )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.savefig(image_path, dpi=100)
    # Closed, as pyplot keeps every figure open otherwise
    plt.close(fig)
