# Data

Laid out as on the user's server by `scripts/post_install.sh`.

- `raw/`: the user's synthetic data (`synthetic.zip`), and the ICD-10 coding it
  downloads.
- `rawdata/`: the user's raw registers, as delivered.
- `workdata/709816/Personal folders/Mathilde/Projects/p1_transformers/`:
  - `Data/`: the user's synthetic data, as delivered (made to predict from age
    65).
  - `synthetic/`: the files our scripts make from it.
    - `age70_df_born1923_1943.parquet`: one row per diagnosis at a hospital
      contact (`PNR`, `INDATE`, `OUTDATE`, `DIAG`, `DIAG_TYPE`, `PATIENT_TYPE`).
    - `age70_df_born1923_1943_short.parquet`: one row per person: sex, birth
      date and cohort, education, `BDAY70`, the dates of death, emigration and
      dementia, `FOLLOW_UP_END_DATE`, `EVENT_TYPE` (dementia, death or censored)
      and the `CHAPTER_<name>` flags.
    - `age70_df_born1923_1943_migration.parquet`: one row per border crossing
      (`PNR`, `HAEND_DATO`, `INDUD_KODE`: U out, I in).
    - `icd10_english_descriptions.csv`: one row per ICD-10 code (`code`,
      `description`).
    - `age70_df_born1923_1943_strings.parquet`: one row per person and string
      variant (`strings.R`).
    - `age70_df_born1923_1943_tabular.parquet`: one row per person, with the
      columns of the tabular arms (`strings.R`).
    - `age70_split_and_folds.parquet`: one row per person (`PNR`, `in_training`,
      `fold`; `split.R`).
