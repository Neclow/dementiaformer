# Data

- `raw/`: the user's synthetic data.
- `synthetic_age70/`: the processed raw data.
  - `age70_df_born1923_1943.parquet`: one row per diagnosis at a hospital
    contact (`PNR`, `INDATE`, `OUTDATE`, `DIAG`, `DIAG_TYPE`, `PATIENT_TYPE`).
  - `age70_df_born1923_1943_short.parquet`: one row per person: sex, birth date
    and cohort, education, `BDAY70`, the dates of death, emigration and
    dementia, `FOLLOW_UP_END_DATE` and `EVENT_TYPE` (dementia, death or
    censored).
  - `age70_df_born1923_1943_migration.parquet`: one row per border crossing
    (`PNR`, `HAEND_DATO`, `INDUD_KODE`: U out, I in).
