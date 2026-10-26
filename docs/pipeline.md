# Pipeline

One section per script in `pipeline/`, in the order they run. They start from
the outputs of the user's scripts 2 and 3. Each takes the name of a settings
file in `configs/data/`, always given: `synthetic` or `server` (e.g.
`pixi run split synthetic`).

## Running an arm

An arm is a model (`delphi`, `delphi_v2` or `delphi_v2_motorhead`) trained on
one variant of the tokens (`delphi`, `delphi_min_people`, `delphi_all_codes`,
`whole_lives` or `cut_at_index`). Which arms to run is the user's choice. On the
server, in the conda environment:

1. Once, for every arm: the split, then the tokens.

   ```bash
   Rscript pipeline/split.R server
   python -m pipeline.build_tokens server
   ```

2. For each arm, first the dry runs of its training and its embeddings (a minute
   at most each, nothing written), then the training. It runs for hours or days:
   `docs/benchmarks.md` estimates how long. Its folder in `runs/` has
   `train.log` and `curves.png` to follow it; if the session stops, `--resume`
   continues from the last evaluation.

   ```bash
   python -m pipeline.train_torch server delphi whole_lives --dry-run
   python -m pipeline.extract_embeddings server delphi whole_lives --dry-run
   python -m pipeline.train_torch server delphi whole_lives
   ```

3. For each trained arm, the embeddings at the index age.

   ```bash
   python -m pipeline.extract_embeddings server delphi whole_lives
   ```

4. Once every chosen arm has its embeddings: the Cox models and their scores,
   for every arm at once, next to the baselines L1 and L2.

   ```bash
   Rscript pipeline/predict.R server
   ```

## `split.R`

Example run:

```bash
Rscript pipeline/split.R synthetic
```

Draws the train/test split and the cross-validation folds of `5 predictions.R`
(step 5a), before the tokens are built, so that the transformer never trains on
test people.

- Same lines and seed (709816) as `5 predictions.R`, so both write the same
  split, and `5 predictions.R` can stay as it is.
- Input: the tabular file of the user's script 3 (`PNR`, `BIRTH_COHORT`,
  `EVENT_TYPE`), as in `5 predictions.R`.
- 70% of people in training, drawn within each outcome (`EVENT_TYPE`) and birth
  year, after sorting by `PNR`.
- Training people get a fold from 1 to 5, and test people none.
- Paths come from the settings file named on the command line (`synthetic` or
  `server`).
- Once the tokens are built, the split must not change.

Writes (path in the config):

- `age70_split_and_folds.parquet`: `PNR`, `in_training`, `fold`

## `build_tokens.py`

Example run:

```bash
python -m pipeline.build_tokens synthetic
```

Builds every person's timeline (their events with their ages in days), then one
tokenisation per variant: the variant picks its events from the timelines and
gives each one a token number.

- Splits from `split.R`: test outside the training set, validation in fold 5 of
  the training set, train otherwise. The models train on train only.
- Events:
  - diagnoses: the first three characters of the code, at its first contact. The
    codes of the dementia definition of the user's script 2 (F00, F01, F02.0,
    F03.9, G30, G31.0B, G31.8B, G31.8E, G31.9), read on the full code, are
    marked `dementia_code`: Delphi keeps them, our variants drop them.
  - `DEMENTIA` at `DEMENTIA_INCIDENT_DATE`, the outcome of `5 predictions.R`,
    and `DEATH`.
  - `ABROAD_START` and `ABROAD_END` of each period of at least 180 days abroad
    before the index age, as in the user's script 3.
  - sex at age 0, as in Delphi.
- Window: from 1969-01-01 (start of the Psychiatric Central Research Register,
  as in the user's script 3) or birth if later, to death, emigration or
  2022-12-31.
- Test people's timelines stop before the index date, so that their outcome
  never reaches the models.

Variants (one folder each, all built in one run). From one to the next, one
thing changes:

| Variant             | Events                                                                           | Vocabulary                                              | Change from the previous one |
| ------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------- |
| `delphi`            | Delphi's: diagnoses with dementia codes, sex, death; codes of Delphi's list only | Delphi's `labels.csv`: every code its own token         | –                            |
| `delphi_min_people` | Delphi's; codes of Delphi's list only                                            | ours: codes in at least 5 training people, else chapter | the rare-code threshold      |
| `delphi_all_codes`  | Delphi's, codes of every chapter                                                 | ours                                                    | chapter coverage             |
| `whole_lives`       | ours: diagnoses without dementia codes, `DEMENTIA`, `DEATH`, periods abroad, sex | ours                                                    | the events                   |
| `cut_at_index`      | ours, but everyone's timeline stops before the index date                        | ours                                                    | what happens after 70        |

- Our vocabulary: `PAD` (0), `NO_EVENT` (1), `DEATH`, `DEMENTIA`,
  `ABROAD_START`, `ABROAD_END`, `UNKNOWN_CODE`, `Female`, `Male`, one
  `RARE_<chapter>` per ICD-10 chapter, then every three-character code that at
  least 5 training people have. A rarer code takes its chapter's token, or
  `UNKNOWN_CODE` outside every chapter.
- Delphi's vocabulary: `labels.csv` of github.com/gerstung-lab/Delphi at commit
  fb72166 (`data/ukb_real_data/labels.csv`), in `data/labels.csv` of this
  repository (`DELPHI_LABELS` in `src/_config.py`). Its 1,255 codes are all in
  ICD-10 chapters A to Q, so codes of chapters R to Z are dropped.
- Every variant writes ours: `events.parquet` and `vocab.csv`, where 0 is
  padding and 1 no event. With Delphi's vocabulary, a token's number is its line
  in `labels.csv` minus 1 (Female 2, Death 1269), as in Delphi's batches.
- `delphi` also writes Delphi's own files, for Delphi's code: there, tokens are
  one less (Female 1, Death 1268), as in Delphi's data.

Example: one training person, born on 1 June 1930 (index date 1 June 2000, at
70), and what each variant keeps of their timeline. D59 is in fewer than 5
training people, F00 and the other codes in at least 5. A test person's timeline
stops before 70 in every variant, as in `cut_at_index`.

| Age  | Event in the timeline          | `delphi`             | `delphi_min_people`  | `delphi_all_codes`   | `whole_lives`        | `cut_at_index`      |
| ---- | ------------------------------ | -------------------- | -------------------- | -------------------- | -------------------- | ------------------- |
| 0    | sex: female                    | `Female`             | `Female`             | `Female`             | `Female`             | `Female`            |
| 45.0 | leaves Denmark                 | –                    | –                    | –                    | `ABROAD_START`       | `ABROAD_START`      |
| 50.0 | comes back                     | –                    | –                    | –                    | `ABROAD_END`         | `ABROAD_END`        |
| 55.2 | I10.9, hypertension            | `I10`                | `I10`                | `I10`                | `I10`                | `I10`               |
| 61.0 | R51, headache                  | – (not in its list)  | – (not in its list)  | `R51`                | `R51`                | `R51`               |
| 66.3 | D59.1                          | `D59`                | `RARE_BLOOD_IMMUNE`  | `RARE_BLOOD_IMMUNE`  | `RARE_BLOOD_IMMUNE`  | `RARE_BLOOD_IMMUNE` |
| 72.5 | I10.9 again                    | – (first occurrence) | – (first occurrence) | – (first occurrence) | – (first occurrence) | –                   |
| 74.1 | dementia, the date of script 2 | –                    | –                    | –                    | `DEMENTIA`           | – (after 70)        |
| 74.3 | F00.1, a dementia code         | `F00`                | `F00`                | `F00`                | – (dementia code)    | –                   |
| 78.9 | death                          | `Death`              | `DEATH`              | `DEATH`              | `DEATH`              | –                   |

Writes, into the `tokens` folder of the config:

```
tokens/
  persons.parquet      PNR, SPLIT, INDEX_AGE_DAYS, OBSERVED_UNTIL_AGE_DAYS
  timelines.parquet    PNR, AGE_DAYS, KIND, EVENT  (shared by every variant)
  delphi_min_people/   events.parquet (PNR, AGE_DAYS, TOKEN), vocab.csv
  delphi_all_codes/    events.parquet, vocab.csv
  whole_lives/         events.parquet, vocab.csv
  cut_at_index/        events.parquet, vocab.csv
  delphi/              events.parquet, vocab.csv, and Delphi's format:
                       train.bin, val.bin, test.bin (uint32 rows of patient id,
                       age in days, token), patients.parquet (PATIENT_ID, PNR)
```

## `train_torch.py`

Example runs:

```bash
# Check that everything runs: one training batch and one evaluation batch per
# split, nothing written (as Lightning's fast_dev_run)
python -m pipeline.train_torch synthetic delphi whole_lives --dry-run
# Train, on 16 CPU threads (DEFAULT_N_THREADS in src/_config.py)
python -m pipeline.train_torch server delphi whole_lives
# Continue a run stopped early, from its last checkpoint, on 8 threads
python -m pipeline.train_torch server delphi whole_lives --resume --threads 8
```

Trains a model (`delphi`, `delphi_v2` or `delphi_v2_motorhead`) on a variant's
tokens with plain PyTorch, as Delphi's `train.py` trains Delphi. The model's
settings file in `configs/models/` (`delphi.yaml` and so on) holds its size and
its training settings.

- Training settings: Delphi's, for Delphi-2M (`config/train_delphi.py`): batches
  of 128 people with up to 96 tokens each; AdamW with a learning rate warmed up
  over 1,000 steps, then lowered along a cosine to a tenth at step 150,000;
  weight decay 0.2 on the weights of linear layers only; gradients clipped at 1.
- Each step draws random training people, with replacement, as Delphi does:
  there are no epochs.
- Every 250 steps and at the last step, the loss on 25 batches of training
  people and of validation people. Evaluation batches get "no event" every 5
  years, and Delphi's loss leaves out "no event" there: the evaluation losses
  are not comparable with the batch losses, only with each other.
- Unlike Delphi, each evaluation uses the same people (random, drawn once), so
  that the curves are less noisy.
- In `metrics.csv` and `train.log`, an evaluation is labelled with the last
  training step before it (step 249 for the evaluation after 250 steps), as
  Lightning labels it.
- `best.pt` is the model with the lowest validation loss so far. `last.pt` is
  rewritten at each evaluation, to continue with `--resume`.
- `--overfit-batches N` trains on the same N batches at every step: the loss
  should drop sharply, which shows that the model can learn.
- `--device cuda` trains on a GPU, if there is one. `--threads N` sets the
  number of CPU threads (default 16, `DEFAULT_N_THREADS` in `src/_config.py`).
- A new run never overwrites an earlier one: delete its folder first, or add
  `--resume`.

Writes, into the `runs` folder of the config, a folder per model and variant
(`runs/delphi_whole_lives/`):

```
settings.yaml   the settings file and the command line of the run
train.log       what the run printed, with times
metrics.csv     a row per log (every 25 steps) and per evaluation: step,
                learning_rate, seconds_per_step, batch/loss, train/loss,
                validation/loss, and each loss term (as Lightning's CSVLogger)
curves.png      the losses against the step, redrawn at each evaluation
best.pt         the model with the lowest validation loss, its optimiser, step
                and settings
last.pt         the same at the last evaluation
```

## `train_lightning.py`

Example runs:

```bash
python -m pipeline.train_lightning synthetic delphi whole_lives --dry-run
python -m pipeline.train_lightning server delphi whole_lives
python -m pipeline.train_lightning server delphi whole_lives --resume
```

The same training as `train_torch.py`, with Lightning: the same settings, model,
loss, optimiser, learning rate and batches, and the same options. Lightning may
make it easier to change the training (more devices, mixed precision,
callbacks).

- The options are Lightning's: `--dry-run` is `fast_dev_run`,
  `--overfit-batches` is `overfit_batches`, and `--resume` continues from
  `last.ckpt`.
- Checked: for the same settings, both trainers give the same batch losses, and
  the same evaluation losses to within float rounding.
- The evaluation runs every 250 steps only (not at a last step that is not a
  multiple of 250).
- `metrics.csv` is Lightning's `CSVLogger`'s, with an `epoch` column more
  (always 0: there are no epochs). Unlike Lightning's, `--resume` appends to it
  instead of deleting it.

Writes, into the `runs` folder of the config, a folder per model and variant
(`runs/delphi_whole_lives_lightning/`): `settings.yaml`, `train.log`,
`metrics.csv` and `curves.png` as `train_torch.py`, and Lightning's checkpoints
`best.ckpt` and `last.ckpt`.

## `extract_embeddings.py`

Example runs:

```bash
# The best model of a run of train_torch.py
python -m pipeline.extract_embeddings server delphi whole_lives
# The best model of a run of train_lightning.py
python -m pipeline.extract_embeddings server delphi whole_lives --lightning
# Check that it runs, before any training: an untrained model, nothing written
python -m pipeline.extract_embeddings server delphi whole_lives --dry-run
```

Embeds every person at the index age with a trained model, and writes the file
of an arm in the format of the user's script 4, which `predict.R` reads.

- What the model reads, for each person: their tokens before the index age, with
  "no event" every 5 years from day 1, then "no event" at the index age itself,
  as in the evaluation batches of training. Nothing from the index age on
  reaches the embedding.
- The embedding: the model's hidden state at that last token (the index age),
  all 120 numbers, as they are (no reduction).
- The model: `best.pt` (or `best.ckpt` with `--lightning`) of the run's folder,
  with the settings of its `settings.yaml`.
- `--dry-run` embeds everyone with an untrained model, from the settings in
  `configs/models/`, runs the checks, and writes nothing: no file that
  `predict.R` could take for an arm.
- Everyone gets an embedding (training, validation and test people), as the Cox
  models of `predict.R` are fitted on the training people and scored on the test
  people.

Writes, into the data folder of the config (`root`):
`age70_dataset_embeddings_<model>_<variant>.parquet` (`<model>_lightning` with
`--lightning`), one row per person: the person columns of the strings file of
script 3 (`PNR`, `SEX`, `BIRTH_COHORT`, `EDUCATION`, `EVENT_TYPE`,
`FOLLOW_UP_END_AGE`, `DEMENTIA_BEFORE_INDEX`, ...), then `embedding_0` to
`embedding_119` (float32), as script 4 writes them.

## `predict.R`

Example run:

```bash
Rscript pipeline/predict.R server
```

Fits and scores the transformer arms as `5 predictions.R` fits and scores its
arms, with its code: two penalised cause-specific Cox models (dementia, and
death without dementia as the competing event), and the risk of dementia by age
85 (primary) and 95 (secondary) on the test people of `split.R`.

- The arms: every file of `extract_embeddings.py` in the data folder, and the
  baselines L1 (sex, birth year) and L2 (+ education) of `5 predictions.R`.
- The predictors: sex, birth year and education, unpenalised, and the 120
  embedding columns, penalised (ridge, elastic net or lasso, the best of the
  three by cross-validation on the folds of `split.R`).
- The scores: Brier score, IPA and AUC by each horizon, among the cohorts that
  can be followed to it; the paired difference of each transformer arm with L2
  (as P2 of `5 predictions.R`); the calibration by tenth of predicted risk,
  suppressed for hjemtagelse.
- Left out from `5 predictions.R`: the memory guard, and the Jina and income
  arms. Each arm is still fitted in its own R process, two at a time.
- One change to its scoring code: among people without dementia, the observed
  risk is 0 (prodlim cannot estimate it otherwise). It only happens in small
  groups, as on the synthetic data.

Writes:

```
<root>/transformer_arm_results/   age70_<arm>.rds and .log, an arm each (not for
                                  hjemtagelse: the predictions of every person)
<root>/../Results/                age70_transformer_arm_performance.csv,
                                  _arm_contrasts.csv, _tuning_settings.csv,
                                  _calibration_by_risk_tenth.csv
```
