# Benchmarks

How fast the three models train and run, on 1, 4 and 16 CPU threads and on a
GPU, so that the time of a full run can be planned.

## Running them

```bash
# Here, with pixi (about 14 minutes, 9 of them on 1 thread)
pixi run benchmark
# On the server, in the conda environment
python -m pytest benchmarks --benchmark-storage=benchmarks/results --benchmark-autosave
# Some of them only, e.g. the training steps of Delphi
pixi run benchmark -k "training_step and delphi-"
# The plots, one per machine, into img/
pixi run plot_benchmarks
```

Each run saves its results in `benchmarks/results/<machine>/` (a JSON file of
pytest-benchmark), when it ends. They hold timings only, no data, so they can
leave the server, to be plotted with ours. The GPU benchmarks are skipped on a
machine without a GPU. Run one benchmark at a time: two runs at once compete for
the CPUs.

## What is measured

`benchmarks/test_steps.py`, for each model, on each device:

| Measure       | What is timed                                               | On                                                          | Rounds             |
| ------------- | ----------------------------------------------------------- | ----------------------------------------------------------- | ------------------ |
| training step | forward pass, loss, backward pass, gradient clipping, AdamW | a random batch: 128 people × 96 tokens, vocabulary of 1,270 | 20, after 3 unused |
| inference     | a forward pass without gradients, as for the embeddings     | the same batch                                              | 20, after 3 unused |

- A training step is one update of the weights on one batch: the forward pass
  (with same-day masking), the loss (Delphi's or MOTOR's), the backward pass,
  gradient clipping and the AdamW update. On a GPU, the time waits for the GPU
  to finish. Not in it: building the batch and moving it to the device (done
  once, before), the learning rate, logging, evaluations and checkpoints.
- The batches are full and random: speed depends on the shapes only, and full
  batches (96 tokens per person) are the slowest training can see. The
  vocabulary has Delphi's size, which matters for the output layers, the MOTOR
  head above all (its output has a value per token and time bin).
- The plots show the median of the rounds.

## The estimates

The plots also turn the medians into hours, with each model's settings file in
`configs/models/`:

- A full training run, at worst: `max_steps` training steps on full batches,
  plus the evaluations (`eval_batches` inference batches of each of the two
  splits, every `eval_interval` steps). Real batches are shorter, so real steps
  are faster. Not counted: building the batches (about 1 ms each here),
  checkpoints, and Lightning's own work (about 4% more on CPU).
- Inference for a million people: a million divided by 128, times the seconds
  per inference batch.

Here (Threadripper PRO 5995WX, 2026-10-13), a full run of 150,000 steps would
take, at worst, 33 to 65 hours on 16 threads depending on the model, and 3 to 4
hours on the GPU. The server's CPU differs: its own benchmark gives its own
estimates.

### On 24 threads

Not measured: the server is shared, so 24 threads is about the most a run can
take. Estimated from the 16-thread results, between a linear speed-up (the best
case) and the trend from 4 to 16 threads (4 times the threads made a step 2.2 to
2.7 times faster, so the gain shrinks with more threads):

| Model                 | Training step, 16 threads | 24 threads, estimated | Training run, 24 threads |
| --------------------- | ------------------------- | --------------------- | ------------------------ |
| Delphi                | 0.89 s                    | 0.59–0.71 s           | 26–31 h                  |
| Delphi v2             | 0.75 s                    | 0.50–0.56 s           | 22–24 h                  |
| Delphi v2, MOTOR head | 1.51 s                    | 1.00–1.18 s           | 43–51 h                  |

At worst (full batches), here; the training run includes the evaluations.

## Measured once: the dry run and the overfit task

The whole trainers were also timed once here (2026-10-13), on the synthetic
`whole_lives` tokens, before these two tasks were left out of the benchmark:
they add little to the training step, which matters most. In seconds, across the
three models (the dry run: median of 3 runs; the overfit task: 1 run):

| Task                               | Trainer   | 1 thread | 4 threads | 16 threads | GPU     |
| ---------------------------------- | --------- | -------- | --------- | ---------- | ------- |
| dry run (`--dry-run`)              | PyTorch   | 1.6–1.8  | 0.7–0.9   | 0.3–0.4    | 0.1     |
| dry run (`--dry-run`)              | Lightning | 1.6–1.9  | 0.7–1.0   | 0.3–0.5    | 0.2     |
| overfit (100 steps, an evaluation) | PyTorch   | 88–98    | 36–54     | 14–23      | 2.2–3.0 |
| overfit (100 steps, an evaluation) | Lightning | 89–104   | 36–53     | 14–23      | 2.5–3.5 |

- Starting a run (reading the tokens, building the model) takes a fraction of a
  second on these data.
- Lightning adds about 4% per step on CPU, and 10 to 15% on the GPU, where steps
  are short.
- The synthetic people have about 32 tokens, so these steps are shorter than the
  full batches above.
