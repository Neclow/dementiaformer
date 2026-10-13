"""Plot the speed benchmarks and the times they imply, a figure per machine."""

import json

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src._config import DEFAULT_IMG_DIR
from src.settings import read_model_settings

# Results of pixi run benchmark: a folder per machine, a file per run
RESULTS_DIR = Path("benchmarks/results")
# The devices, in the order of the x axis
DEVICES = {
    "cpu-1": "1 thread",
    "cpu-4": "4 threads",
    "cpu-16": "16 threads",
    "cuda": "GPU",
}
MODELS = {
    "delphi": "Delphi",
    "delphi_v2": "Delphi v2",
    "delphi_v2_motorhead": "Delphi v2, MOTOR head",
}
# People in the inference estimate
PEOPLE = 1_000_000


def read_results(path: Path) -> pl.DataFrame:
    """One row per benchmark of a results file: measure, model, device, seconds, rounds."""
    results = json.loads(path.read_text(encoding="utf-8"))
    # The median time of the rounds
    return pl.DataFrame(
        [
            {
                "measure": benchmark["group"],
                "model": benchmark["params"]["name"],
                "device": benchmark["params"]["device"],
                "seconds": benchmark["stats"]["median"],
                "rounds": benchmark["stats"]["rounds"],
            }
            for benchmark in results["benchmarks"]
        ]
    )


def estimate_hours(results: pl.DataFrame) -> pl.DataFrame:
    """Hours of a training pass over PEOPLE people and of their inference, per model and device."""
    # One row per model: its training settings
    settings = pl.DataFrame(
        [{"model": model, **read_model_settings(model)["training"]} for model in MODELS]
    )
    # One row per model and device: seconds per training step and per
    # inference batch, and the model's training settings
    times = (
        results.filter(measure="training step")
        .select("model", "device", step="seconds")
        .join(
            results.filter(measure="inference").select(
                "model", "device", inference="seconds"
            ),
            on=["model", "device"],
        )
        .join(settings, on="model")
    )
    # Training: a pass over PEOPLE people, each step on a full batch, plus the
    # evaluations of those steps (eval_batches batches of each of the two
    # splits, every eval_interval steps)
    steps = PEOPLE / pl.col("batch_size")
    evaluations = steps / pl.col("eval_interval")
    training = steps * pl.col("step") + evaluations * 2 * pl.col(
        "eval_batches"
    ) * pl.col("inference")
    inference = PEOPLE / pl.col("batch_size") * pl.col("inference")
    return pl.concat(
        [
            times.select(
                "model", "device", measure=pl.lit("training"), hours=training / 3600
            ),
            times.select(
                "model", "device", measure=pl.lit("inference"), hours=inference / 3600
            ),
        ]
    )


def draw(ax: Axes, values: pl.DataFrame, column: str, devices: list[str]) -> None:
    """A line per model: the column against the device, on a log scale in plain numbers."""
    colors = dict(zip(MODELS, sns.color_palette("colorblind", len(MODELS))))
    for model in MODELS:
        points = values.filter(model=model).sort(
            pl.col("device").replace_strict(devices, range(len(devices)))
        )
        ax.plot(
            [devices.index(device) for device in points["device"]],
            points[column],
            color=colors[model],
            marker="o",
            markersize=5,
        )
    ax.set_xticks(range(len(devices)), [DEVICES[device] for device in devices])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", alpha=0.3)


def plot_machine(results: pl.DataFrame, title: str, path: Path) -> None:
    """The training step and inference as measured, then the hours they imply."""
    sns.set_theme(style="ticks", context="notebook")
    devices = [device for device in DEVICES if device in results["device"].to_list()]
    hours = estimate_hours(results)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    # The two measures share their y axis: both are seconds per batch
    axes[0, 1].sharey(axes[0, 0])
    for ax, measure, heading in [
        (axes[0, 0], "training step", "Training step"),
        (axes[0, 1], "inference", "Inference"),
    ]:
        values = results.filter(measure=measure)
        draw(ax, values, "seconds", devices)
        rounds = values["rounds"].max()
        ax.set_title(
            f"{heading}, median of {rounds} runs", loc="left", fontweight="bold"
        )
        ax.set_ylabel("Seconds per batch of 128 people")
    for ax, measure, heading in [
        (axes[1, 0], "training", f"Training, a pass over {PEOPLE:,} people, estimated"),
        (axes[1, 1], "inference", f"Inference for {PEOPLE:,} people, estimated"),
    ]:
        draw(ax, hours.filter(measure=measure), "hours", devices)
        ax.set_title(heading, loc="left", fontweight="bold")
        ax.set_ylabel("Hours")
    sns.despine(fig)

    # One legend for every panel: a colour per model
    colors = sns.color_palette("colorblind", len(MODELS))
    handles = [
        Line2D([], [], color=color, marker="o", label=label)
        for color, label in zip(colors, MODELS.values())
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    DEFAULT_IMG_DIR.mkdir(exist_ok=True)
    # The latest run of each machine
    for machine in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        latest = sorted(machine.glob("*.json"))[-1]
        results = json.loads(latest.read_text(encoding="utf-8"))
        cpu = results["machine_info"]["cpu"]["brand_raw"]
        title = f"Speed on {cpu} ({results['datetime'][:10]})"
        path = DEFAULT_IMG_DIR / f"benchmarks_{machine.name}.png"
        plot_machine(read_results(latest), title, path)
        print(f"{latest} -> {path}")


if __name__ == "__main__":
    main()
