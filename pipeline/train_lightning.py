"""Train a model on a variant's tokens with Lightning, as train_torch.py does with plain PyTorch."""

from argparse import ArgumentParser, Namespace

import torch

from src._config import DEFAULT_N_THREADS, VARIANTS
from src.training.lightning_trainer import train
from src.training.shared import MODELS, start_run


def parse_args() -> Namespace:
    """The command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="settings file in configs/data/, without .yaml: synthetic or server",
    )
    parser.add_argument(
        "model",
        choices=list(MODELS),
        help="the model, and its settings file in configs/models/",
    )
    parser.add_argument(
        "variant",
        choices=list(VARIANTS),
        help="the tokens: a folder of build_tokens.py",
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
    parser.add_argument(
        "--resume", action="store_true", help="continue the run from its last.ckpt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="one training batch, then one evaluation batch per split, writing "
        "nothing (Lightning's fast_dev_run)",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        metavar="N",
        help="train on the same N batches: the loss should drop sharply "
        "(Lightning's overfit_batches)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    settings, folder, run = start_run(
        args.config,
        args.model,
        args.variant,
        args.dry_run,
        args.resume,
        suffix="_lightning",
    )
    train(
        args.model,
        settings,
        folder,
        VARIANTS[args.variant]["vocabulary"] == "delphi",
        run,
        device=args.device,
        resume=args.resume,
        dry_run=args.dry_run,
        overfit_batches=args.overfit_batches,
    )


if __name__ == "__main__":
    main()
