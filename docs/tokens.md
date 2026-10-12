# Tokens: from registry records to model input

The tokeniser turns each person's records into a timeline of events: a list of
(age in days, token) pairs. The models of this project read these timelines. The
code is `src/tokens.py` (the steps) and `pipeline/build_tokens.py` (the script
that runs them).

## How to run

From the repository root, on the server:

```
python -m pipeline.build_tokens --config configs/data_server.toml
```

This builds every variant of the config file. Add `--variant whole_lives` to
build only one. Here, on the synthetic data, `pixi run build_tokens_all` does
the same with `configs/data_synthetic.toml`.

**Inputs**, set under `[paths]` in the config:

- the outputs of the preprocessing (script 2):
  - `contacts`: one row per hospital contact (`PNR`, `INDATE`, `DIAG`)
  - `persons`: one row per person (`PNR`, `BIRTH_DATE`, `DEATH_DATE`,
    `EMIGRATION_DATE`, `DEMENTIA_INCIDENT_DATE`)
  - `migration`: one row per border crossing (`PNR`, `HAEND_DATO`, `INDUD_KODE`)
- `split`: the split file of `5 predictions.R` (`PNR`, `in_training`, `fold`).
  It must exist first: run `5 predictions.R` up to the line that writes
  `age70_split_and_folds.parquet`.

**Outputs**, one folder per variant under `output`:

| File              | Content                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `events.parquet`  | `PNR`, `AGE_DAYS`, `TOKEN`: every person's timeline                                                                           |
| `persons.parquet` | `PNR`, `SPLIT` (train, validation or test), `INDEX_AGE_DAYS`, `OBSERVED_UNTIL_AGE_DAYS` (the age at which the timeline stops) |
| `vocab.csv`       | `TOKEN`, `NAME`, `KIND`, `CHAPTER`, and `N_TRAIN_PEOPLE` (the number of training people with each code)                       |
| `settings.json`   | the settings the folder was built with                                                                                        |

`events.parquet` and `persons.parquet` are microdata and must stay on the
server.

The script also prints totals for each variant:

- events per person, by split, and the share of timelines longer than Delphi's
  context of 96 events
- how many codes, and what share of diagnoses, fall below 5, 10, 25, 50 and 100
  training people
- the share of diagnoses, and of people, with codes outside Delphi's chapters
  (A–Q)
- the number of training people with each outcome token

As with the outputs of the R scripts, check that no printed count is below 5
before taking any of them off the server.

## Variants

Each variant of the config file is one folder of tokens:

| Variant               | `dementia`  | `cut_at_index` | Training timelines                                    |
| --------------------- | ----------- | -------------- | ----------------------------------------------------- |
| `whole_lives`         | `"single"`  | `false`        | the whole record, with one `DEMENTIA` token           |
| `whole_lives_subtype` | `"subtype"` | `false`        | the whole record, with one token per dementia subtype |
| `cut_at_index`        | —           | `true`         | stopped before the index age, like everyone else's    |

The first two let the model learn from what happens after the index age,
dementia included, as Delphi does. `cut_at_index` keeps everything after the
index age away from the model, as the text arms of `3 strings.R` do. In every
variant, validation and test people's events are never used for training, and
test people's timelines stop before the index age.

## Decisions

Where possible, the tokeniser follows Delphi (Shmatko et al., _Nature_ 2025,
<https://github.com/gerstung-lab/Delphi>):

- diagnoses are three-character ICD-10 codes
- each code is kept at its first occurrence (setting `first_occurrence_only`)
- ages are in days

Where we differ, and why:

1. **A vocabulary from the data, covering every ICD-10 chapter.**
   - Delphi uses a fixed list of 1,255 codes from UK Biobank, all in chapters
     A–Q.
   - That list has no symptoms (R), injuries (S–T), external causes (V–Y),
     contacts with health services (Z) or special codes (U).
   - These chapters are common in Danish hospital data, and they include
     dementia risk factors such as head injury (S06). See `paper.md`.
2. **A code needs at least 5 training people to get its own token** (setting
   `min_people_per_code`).
   - Rarer codes share one token per ICD-10 chapter (`RARE_<chapter>`), as do
     codes that only validation or test people have. Rare codes outside every
     chapter share `UNKNOWN_CODE`.
   - 5 is the project's minimum cell count (`minimum_cell_count` in
     `5 predictions.R`), so no token rests on fewer than 5 people. This matters
     if the model or its vocabulary ever leaves the server.
   - With about a million people, codes this rare are a tiny share of all
     events. The printed totals show this on the real data.
3. **No sex, education, birth-cohort or lifestyle tokens.**
   - Delphi has tokens for sex, BMI, smoking and alcohol.
   - The registers have no lifestyle data.
   - Sex, birth cohort and education are left out to match the text arms of
     `3 strings.R`: they enter every model of `5 predictions.R` as covariates.
4. **Periods abroad**, as in `L3_diag_text_dated`.
   - `ABROAD_START` and `ABROAD_END` mark each period of at least 180 days
     abroad before the index age (setting `minimum_spell_days`). The periods are
     built exactly as in section 5 of `3 strings.R`.
   - They tell the model that years without diagnoses are years without Danish
     records.
5. **The split of `5 predictions.R`.**
   - The model trains on training people only. The fold `validation_fold` of the
     training set is held out to validate it.
   - Test people's timelines stop before the index age, so their outcome can
     never reach the model. The script stops with an error if a test person has
     an event from the index date on.
6. **Parquet files.** Delphi writes a raw binary file. Parquet can be opened in
   R, and the data fits in memory.
7. **Token numbers are stored as the model uses them**: 0 is padding and 1 is
   "no event". Delphi stores them shifted by one and corrects this when it
   builds batches.
8. **Dementia is placed at `DEMENTIA_INCIDENT_DATE`**, so the model's dementia
   is the outcome of `5 predictions.R`. That date may come from a prescription
   rather than a diagnosis. Delphi keeps the dementia codes (F00–F03, G30) as
   ordinary tokens. We drop them from the diagnoses, and the setting `dementia`
   chooses the form:
   - `"single"`: one `DEMENTIA` token.
   - `"subtype"`: one token per subtype at the same date, read from the person's
     first dementia code (setting `dementia_subtypes`):
     - Alzheimer's (`DEMENTIA_ALZHEIMER`): F00, G30
     - vascular (`DEMENTIA_VASCULAR`): F01
     - other (`DEMENTIA_OTHER`): F02, G31.0, G31.8, G31.9
     - unspecified (`DEMENTIA_UNSPECIFIED`): F03, or no dementia code at all
   - These codes come from the synthetic data and must be checked against the
     definition of dementia in script 2.
   - Either way, each person with dementia gets exactly one dementia token.
   - No one has dementia at the index age, so the setting does not change a
     timeline before it. It only changes what the model learns to predict after
     it.
