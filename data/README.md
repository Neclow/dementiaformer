# Data

- `raw/`: the user's synthetic data, in an older layout that predicts from age
  65 (people born 1923–1943). Diagnoses and outcomes are drawn at random. Its
  files are listed in `raw/README.md`.
- `synthetic_age70/`: the raw data in the layout the pipeline reads, at index
  age 70, made by `pixi run make_synthetic_data` (decisions:
  `docs/pipeline.md`).
  - `age70_df_born1923_1943.parquet`: one row per hospital contact (`PNR`,
    `INDATE`, `OUTDATE`, `DIAG`, `DIAG_TYPE`, `PATIENT_TYPE`).
  - `age70_df_born1923_1943_short.parquet`: one row per person: sex, birth date
    and cohort, education, `BDAY70`, the dates of death, emigration and
    dementia, `FOLLOW_UP_END_DATE` and `EVENT_TYPE` (dementia, death or
    censored).
  - `age70_df_born1923_1943_migration.parquet`: one row per border crossing
    (`PNR`, `HAEND_DATO`, `INDUD_KODE`: U out, I in).
