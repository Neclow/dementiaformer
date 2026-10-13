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
losses will be in `src/training/losses.py` (step 4b).

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
- The loss will censor each person at the end of their record
  (`OBSERVED_UNTIL_AGE_DAYS` in `tokens/persons.parquet`).
