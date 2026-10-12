# Data

Placeholder README for the data

## Synthetic data at age 70 (`synthetic_age70/`)

The synthetic files in `raw/` (and `raw/synthetic.zip`) follow an older layout
that predicts from age 65. `tmp/make_synthetic_age70.py` converts them to the
layout of the client's current scripts, which predict from age 70. It is a
throwaway helper: the data only serves to check that the code runs.

Run from the repository root:

```
pixi run python tmp/make_synthetic_age70.py
```

Files written (named like the server's):

- `age70_df_born1923_1943.parquet`: one row per contact, with the contact
  columns only (`PNR, INDATE, OUTDATE, DIAG, DIAG_TYPE, PATIENT_TYPE`).
- `age70_df_born1923_1943_short.parquet`: one row per person
  (`PNR, SEX, BIRTH_DATE, BIRTH_COHORT, EDU_CAT3, DEATH_DATE, BDAY70, EMIGRATION_DATE, DEMENTIA_INCIDENT_DATE, DEMENTIA, FOLLOW_UP_END_DATE, EVENT_TYPE`).
- `age70_df_born1923_1943_migration.parquet`: one row per border crossing
  (`PNR, HAEND_DATO, INDUD_KODE`).
- `age70_split_and_folds.parquet`: the split file of `5 predictions.R`
  (`PNR, in_training, fold`).

Decisions:

- **Age 70:** `BDAY70` is the birth date + 70 years, with 29 February moved to
  28 February as in R's `%m+%`. It replaces the age-65 columns.
- **No impossible dates:**
  - A death date after the register end (2022-12-31, e.g. 2035 in the raw data)
    is removed: the person is alive at the end.
  - Contacts after death or emigration are dropped.
- **Dementia:** `DEMENTIA_INCIDENT_DATE` is the date of the first contact with a
  dementia code (the codes of `make_synthetic_data.R`), and `DEMENTIA` is 1 when
  there is one.
  - The raw dementia dates are not used. They were floats of days since
    1970-01-01, and the diagnosis and person files disagreed on them (327 vs 170
    people). They matched the dementia codes for only 29 people, and some fell
    after death.
  - Read from the contacts, dementia agrees with the codes and never comes after
    death or emigration.
- **Population:** as in the client's preprocessing, people alive on their 70th
  birthday and without dementia before it (1,486 of 1,652).
- **Person columns in the person file only:**
  - The diagnosis file no longer repeats them, so the two files cannot disagree.
  - The letter flags (`A_DIAG` … `U_DIAG`) are dropped. They summarised contacts
    that were removed, and nothing reads them.
- **Emigration after 70:** every raw move is a round trip between ages 40 and
  63, so no one would leave after 70. To exercise that code path, 3% of people
  (the share of emigrants after the index age in `make_synthetic_data.R`) get a
  departure on a random day between their 70th birthday and their death or the
  register end. `EMIGRATION_DATE` is that departure.
- **Follow-up:** `FOLLOW_UP_END_DATE` is the earliest of dementia, death,
  emigration and 2022-12-31. `EVENT_TYPE` is `dementia`, `death` or `censored`,
  whichever came first.
- **Split:** 70% `in_training`, with folds 1–5 among them. It is a plain random
  draw (seed 709816), not stratified like the client's.
- **Checks:** the script asserts that none of the impossibilities above remain.
