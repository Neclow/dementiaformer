# Pipeline

One section per script in `pipeline/`, in the order they run.

## `make_synthetic_data.py`

Turns the user's synthetic data (`data/raw/`, made to predict from age 65) into
the layout the pipeline reads, at any index age (`--index-age`, default 70). The
data only serves to check that the code runs: its diagnoses and outcomes are
random.

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
