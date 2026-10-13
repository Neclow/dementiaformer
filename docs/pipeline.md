# Pipeline

One section per script in `pipeline/`, in the order they run.

## `make_synthetic_data.py`

Example run, after `scripts/post_install.sh`:

```bash
python -m pipeline.make_synthetic_data --config configs/synthetic.yaml
```

Turns the user's synthetic data (`Data/`, made to predict from age 65) into the
layout the pipeline reads, at the index age of the config. It stands in for the
user's preprocessing (script 2). The data only serves to check that the code
runs: its diagnoses and outcomes are random.

- `BDAY<age>` replaces the age-65 columns. 29 February becomes 28 February, as
  R's `%m+%` does.
- A death after the end of the registers (2022-12-31) is removed.
- Contacts after death or emigration are dropped.
- Dementia is the date of the first contact with a dementia code (the codes of
  `make_synthetic_data.R`). The raw dementia dates disagreed between files and
  with the codes.
- The population is everyone alive and without dementia at the index age.
- Every raw border crossing is a round trip between ages 40 and 63, so 3% of
  people (as in `make_synthetic_data.R`) get a departure on a random day after
  the index age.
- Follow-up ends at the earliest of dementia, death, emigration and 2022-12-31,
  and `EVENT_TYPE` says which came first.
- Person columns are kept in the person file only. The letter flags (`A_DIAG` …
  `U_DIAG`) are dropped.
- `CHAPTER_<name>` flags are added to the person file, as script 2 writes them:
  1 if a contact before the index age has a code of that ICD-10 chapter.
  `3 strings.R` compares them with its own flags (check 11f).
- The English ICD-10 titles that `3 strings.R` reads come from UK Biobank's
  data-coding 19 (WHO ICD-10), downloaded by `scripts/post_install.sh`.

Writes, into `p1_transformers/synthetic/` (paths in the config):

- `age<age>_df_born1923_1943.parquet`, `_short.parquet`, `_migration.parquet`
- `icd10_english_descriptions.csv`

## `strings.R`

Example run:

```bash
Rscript pipeline/strings.R configs/synthetic.yaml
```

The user's `3 strings.R`: the sentences of the text arms and the columns of the
tabular arms, one row per person, from the history before the index age.

- A copy of `3 strings.R` in which only the paths and the index age change: they
  come from the settings file given on the command line (`configs/server.yaml`
  without one, e.g. from RStudio). Everything else is the user's code.

Writes (paths in the config):

- `age70_df_born1923_1943_strings.parquet`: one row per person and string
  variant
- `age70_df_born1923_1943_tabular.parquet`: one row per person

## `split.R`

Example run:

```bash
Rscript pipeline/split.R configs/synthetic.yaml
```

Draws the train/test split and the cross-validation folds of `5 predictions.R`
(step 5a), before the tokens are built, so that the transformer never trains on
test people.

- Same lines and seed (709816) as `5 predictions.R`, so both write the same
  split, and `5 predictions.R` can stay as it is.
- Input: the tabular file of `strings.R` (`PNR`, `BIRTH_COHORT`, `EVENT_TYPE`),
  as in `5 predictions.R`.
- 70% of people in training, drawn within each outcome (`EVENT_TYPE`) and birth
  year, after sorting by `PNR`.
- Training people get a fold from 1 to 5, and test people none.
- Paths come from the settings file given on the command line
  (`configs/server.yaml` without one, e.g. from RStudio).
- Once the tokens are built, the split must not change.

Writes (path in the config):

- `age70_split_and_folds.parquet`: `PNR`, `in_training`, `fold`
