# Models

Three models in `src/models/`. Each changes one thing from the one before, so
that their differences can be read directly.

| Model                    | Backbone         | Head                                       |
| ------------------------ | ---------------- | ------------------------------------------ |
| `delphi.py`              | Delphi's (GPT-2) | Delphi's: one rate per token               |
| `delphi_v2.py`           | modern           | Delphi's                                   |
| `delphi_v2_motorhead.py` | modern (v2's)    | MOTOR's: one hazard per token and time bin |

All three read a person's tokens with their ages in days, and give an output for
each token: what happens next, and when. They return only these outputs: the
losses are in `src/training/losses.py`.

With the default settings (the size of Delphi-2M) and Delphi's vocabulary of
1,270 tokens:

| Model                    | Parameters |
| ------------------------ | ---------- |
| `delphi.py`              | 2,243,400  |
| `delphi_v2.py`           | 2,243,400  |
| `delphi_v2_motorhead.py` | 2,344,200  |

## Delphi (`delphi.py`)

A port of `model.py` of [Delphi](https://github.com/gerstung-lab/Delphi) at
commit fb72166 (MIT License, see `LICENSE`).

- Same maths as Delphi:
  - The age in years, encoded by sines and cosines then a linear layer, takes
    the place of positions.
  - Same-day masking: a token does not see the tokens dated as its target,
    because the order of same-day events is arbitrary.
  - The token embeddings are shared with the output layer.
- The settings default to Delphi-2M as Delphi trains it
  (`config/train_delphi.py` and `train.py`): 12 layers, 12 heads, width 120, no
  dropout, no bias terms, same-day masking on.
- Left out: dead code, `generate` (only a later risk arm needs it) and the
  optimiser (it goes with the training, step 5).
- The loss moves out of the model, so that the two heads can be swapped and each
  loss tested on its own.
- Checked: with the same weights and the same batch (Delphi's `get_batch` on the
  `delphi` tokens, ages rounded to years to create 2,071 same-day targets), the
  outputs equal Delphi's to within 4e-6.

## Delphi v2 (`delphi_v2.py`)

Delphi's head on a modern backbone.

- RMSNorm in place of LayerNorm, and no bias terms.
- A SwiGLU MLP, 8/3 times wider than the width (320), so that the model has as
  many parameters as Delphi.
- PyTorch's fused attention, with Delphi's masks.
- Rotary encoding (RoPE) on age: queries and keys are turned by angles
  proportional to the age in years, so that attention depends on the time
  between two tokens. The frequencies span 1 to about 1/10,000 per year, as in
  Delphi's age encoding.
- Delphi's encoding of the age itself stays at the input: dementia risk depends
  on age, and RoPE alone only sees differences in age.

## Delphi v2 with a MOTOR head (`delphi_v2_motorhead.py`)

Delphi v2's backbone with the head of MOTOR (Steinberg et al., ICLR 2024).

- Output: after each token, the log hazard of every token in each time bin.
  Unlike Delphi's single rate until the next event, the hazard can change over
  time.
- Time bins: boundaries at 1, 2, 5, 10, 15 and 25 years, so 7 bins. From age 70,
  15 and 25 years are the horizons 85 and 95 of `5 predictions.R`.
- A small head, as in MOTOR: the hidden state becomes one vector per bin, which
  is multiplied with the token embeddings. This adds 7 × 120² parameters.
- Memory: the output has a value per person, token, bin and vocabulary token:
  about 450 MB in float32 for a batch of 128 people, 96 tokens and 1,300 tokens
  in the vocabulary. If that is too much, the batch gets smaller (step 7
  measures it).
- Its loss is `motor_loss` (see Losses).

## Losses (`src/training/losses.py`)

### Delphi's loss (`delphi_loss`), for `delphi.py` and `delphi_v2.py`

The loss of Delphi's `forward`, moved out of the model.

- Two terms, returned apart as in Delphi (training uses their sum):
  - the cross-entropy of the next token;
  - the exponential log-likelihood of the time to it, at the total rate of all
    tokens. `t_min` (0.1 day) bounds the expected waiting time from below.
- Same-day targets: the time is counted from the last token dated before the
  target, as in Delphi.
- Tokens left out of the loss: a list of token numbers. Validation also leaves
  out "no event", as in Delphi.
- Checked: for the same outputs and batches as for `delphi.py` (with real ages,
  and with ages rounded to years), both terms equal Delphi's to 6 digits, in
  training and in validation.

### MOTOR's loss (`motor_loss`), for `delphi_v2_motorhead.py`

A piecewise-exponential likelihood: after each token, the time to every task.

- Tasks: every token but padding, "no event" and sex, and in Delphi's vocabulary
  its lifestyle tokens (BMI, smoking, alcohol), which never occur in the
  registers. After each token, each task is:
  - an event, at its first age after that token's day;
  - censored at the end of the record (`OBSERVED_UNTIL_AGE_DAYS`), if it never
    happens;
  - left out, if it happened on or before that token's day. Same-day events so
    count from the last earlier day, as in Delphi's loss.
- The future comes from the person's whole record, not only from the tokens in
  the batch: events after the batch's window still count. The batch gives each
  person's first age of every token and the end of their record.
- Hazards are per year. The time spent in each bin is computed for every task
  and token (a tensor the size of the model's output). A cheaper computation for
  censored tasks, which share one time, comes only if memory runs short.
- Summed over tasks, averaged over tokens.

## Batches (`src/training/batches.py`)

A port of Delphi's `get_batch` (`utils.py`), with the options Delphi trains
with.

- `read_timelines` reads a variant's `events.parquet` and `vocab.csv`, for the
  people of one split. People are numbered in PNR order, as in Delphi's `.bin`
  files.
- Each person's window: their first 97 tokens (the context of 96, plus one for
  the last target).
- "No event" tokens: in training, 20 at random days within 100 years; in
  evaluation, every 5 years from day 1. None after the person's last token in
  the window.
- Padding on the left, and each batch trimmed to its longest person. Targets are
  the next tokens.
- For MOTOR's loss, each batch also has each person's first age of every token
  in their whole record, and the end of their record
  (`OBSERVED_UNTIL_AGE_DAYS`). They are built per batch: for every person at
  once, they would take gigabytes on the server.
- `left_out_tokens`: the tokens left out of the losses. With Delphi's
  vocabulary, its own list (padding, sex, lifestyle); with ours, `PAD`, `Female`
  and `Male`. MOTOR's loss also leaves out `NO_EVENT`.
- Dropped from Delphi's `get_batch`: the options Delphi does not train with, and
  the lifestyle augmentation (no lifestyle data).
- Random draws come from a generator seeded with the people's row numbers, as in
  Delphi: the same people give the same batch.
- Checked: on the `delphi` tokens, 800 batches equal those of Delphi's
  `get_batch` on `train.bin` and `val.bin`, in training and evaluation, except
  for 5 of 102,400 people. These are people at the end of Delphi's file, where
  its window starts before the person: only the order of same-day tokens
  differs, which is arbitrary.
