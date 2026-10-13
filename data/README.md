# Data

Not in git, apart from this README. Laid out as on the user's server (`E:/`), so
that `configs/data/synthetic.yaml` and `configs/data/server.yaml` differ only in
their `root`.

- `raw/`: an inbox for new data from the user. Files are moved from here to
  where they live on the server.
- `rawdata/709816/`: the user's raw registers. Empty here: the synthetic data
  has none.
- `workdata/709816/Personal folders/Mathilde/Projects/p1_transformers/synthetic/`:
  `Data/` on the server, renamed so that nothing here can be mistaken for the
  user's real data. The user's synthetic data (people born 1913–1947, index age
  70):
  - inputs of the user's script 2: `Diagnosis/data_all_diag.parquet`,
    `Prescriptions/lmdb_dementia.parquet`, `SEP/education_format.parquet`,
    `SEP/income.parquet`
  - outputs of the user's scripts 2 and 3:
    - `age70_df_born1913_1947.parquet`: one row per diagnosis at a hospital
      contact.
    - `age70_df_born1913_1947_short.parquet`: one row per person.
    - `age70_df_born1913_1947_migration.parquet`: one row per border crossing.
    - `age70_df_born1913_1947_strings.parquet`: one row per person and string
      variant.
    - `age70_df_born1913_1947_tabular.parquet`: one row per person, with the
      columns of the tabular arms.
  - what our scripts make from it:
    - `age70_split_and_folds.parquet`: one row per person (`PNR`, `in_training`,
      `fold`; `split.R`).
    - `tokens/`: the timelines, and one folder per tokenisation
      (`build_tokens.py`; layout in `docs/pipeline.md`).
