# Data

- `raw/`: the user's synthetic data, in an older layout that predicts from age
  65 (people born 1923–1943). Diagnoses and outcomes are drawn at random.
  - `df_born1923_1943.parquet`: one row per hospital diagnosis, with the
    person's columns repeated on each row.
  - `df_born1923_1943_string_diag.parquet`: one row per person, with their
    diagnoses written as sentences.
  - `synthetic.zip`: the user's synthetic data folder: the R script that made it
    (`make_synthetic_data.R`), the person file
    (`df_born1923_1943_short.parquet`), the migration file
    (`df_born1923_1943_migration.parquet`), and files the pipeline does not use
    (raw registers, income, education, dementia prescriptions).
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
