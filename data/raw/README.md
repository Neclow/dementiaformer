# Raw data

Not in git. Put the user's synthetic data here (`DEFAULT_RAW_DIR` in
`src/_config.py`):

- `df_born1923_1943.parquet`: one row per hospital diagnosis, with the person's
  columns repeated on each row (layout that predicts from age 65).
- `df_born1923_1943_string_diag.parquet`: one row per person, with their
  diagnoses written as sentences.
- `synthetic.zip`: the user's synthetic data folder: the R script that made it
  (`make_synthetic_data.R`), the person file (`df_born1923_1943_short.parquet`),
  the migration file (`df_born1923_1943_migration.parquet`), and files the
  pipeline does not use (raw registers, income, education, dementia
  prescriptions).
