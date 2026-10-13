"""Speed of the three models: a training step and inference, on each device."""

import pytest
import torch

from torch import nn

from src._config import DEFAULT_N_THREADS
from src.settings import read_model_settings
from src.training.batches import DELPHI_INPUT_ONLY, Batch
from src.training.shared import (
    TrainingSettings,
    build_model,
    loss_terms,
    make_optimizer,
)

MODELS = ["delphi", "delphi_v2", "delphi_v2_motorhead"]
# Where each benchmark runs: on 1, 4 or 16 CPU threads, or on a GPU
DEVICES = ["cpu-1", "cpu-4", "cpu-16", "cuda"]
# Vocabulary of the random batches: Delphi's size, close to what the server's
# should be
VOCAB_SIZE = 1270


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    """The device of a benchmark, with PyTorch's CPU threads set."""
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        torch.set_num_threads(DEFAULT_N_THREADS)
        return "cuda"
    torch.set_num_threads(int(request.param.split("-")[1]))
    return "cpu"


def random_batch(settings: TrainingSettings, device: str) -> Batch:
    """A full batch of random tokens at ordered ages: speed depends on the shapes only."""
    generator = torch.Generator().manual_seed(0)
    shape = (settings.batch_size, settings.block_size + 1)
    # No left-out tokens, so that every token counts in the losses
    tokens = torch.randint(
        max(DELPHI_INPUT_ONLY) + 1, VOCAB_SIZE, shape, generator=generator
    )
    ages = torch.randint(0, 80 * 365, shape, generator=generator).float()
    ages = ages.sort(dim=1).values
    first_ages = torch.full((settings.batch_size, VOCAB_SIZE), float("inf"))
    first_ages.scatter_(1, tokens, ages)
    return Batch(
        tokens[:, :-1],
        ages[:, :-1],
        tokens[:, 1:],
        ages[:, 1:],
        first_ages,
        ages[:, -1] + 365,
    ).to(device)


def synchronize(device: str) -> None:
    """Wait for the GPU to finish its work, so that it counts in the time."""
    if device == "cuda":
        torch.cuda.synchronize()


@pytest.mark.benchmark(group="training step")
@pytest.mark.parametrize("name", MODELS)
def test_training_step(benchmark, name: str, device: str) -> None:
    settings = read_model_settings(name)
    training = TrainingSettings(**settings["training"])
    torch.manual_seed(training.seed)
    model = build_model(name, settings["model"], VOCAB_SIZE).to(device)
    optimizer = make_optimizer(model, training)
    batch = random_batch(training, device)

    def step() -> None:
        """One training step, as in the trainers."""
        terms = loss_terms(
            name, model, batch, DELPHI_INPUT_ONLY, settings, validation=False
        )
        terms["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)

    benchmark.pedantic(step, rounds=20, warmup_rounds=3)
    benchmark.extra_info["people"] = training.batch_size


@pytest.mark.benchmark(group="inference")
@pytest.mark.parametrize("name", MODELS)
def test_inference(benchmark, name: str, device: str) -> None:
    settings = read_model_settings(name)
    training = TrainingSettings(**settings["training"])
    torch.manual_seed(training.seed)
    model = build_model(name, settings["model"], VOCAB_SIZE).to(device).eval()
    batch = random_batch(training, device)

    def forward() -> None:
        """One forward pass without gradients, as for the embeddings."""
        with torch.no_grad():
            model(batch.tokens, batch.ages)
        synchronize(device)

    benchmark.pedantic(forward, rounds=20, warmup_rounds=3)
    benchmark.extra_info["people"] = training.batch_size
