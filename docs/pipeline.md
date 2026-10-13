# Pipeline

One section per script in `pipeline/`, in the order they run. They start from
the outputs of the user's scripts 2 and 3. Each takes the name of a settings
file in `configs/`, always given: `synthetic` or `server` (e.g.
`pixi run split synthetic`).

## `split.R`

Example run:

```bash
Rscript pipeline/split.R synthetic
```

Draws the train/test split and the cross-validation folds of `5 predictions.R`
(step 5a), before the tokens are built, so that the transformer never trains on
test people.

- Same lines and seed (709816) as `5 predictions.R`, so both write the same
  split, and `5 predictions.R` can stay as it is.
- Input: the tabular file of the user's script 3 (`PNR`, `BIRTH_COHORT`,
  `EVENT_TYPE`), as in `5 predictions.R`.
- 70% of people in training, drawn within each outcome (`EVENT_TYPE`) and birth
  year, after sorting by `PNR`.
- Training people get a fold from 1 to 5, and test people none.
- Paths come from the settings file named on the command line (`synthetic` or
  `server`).
- Once the tokens are built, the split must not change.

Writes (path in the config):

- `age70_split_and_folds.parquet`: `PNR`, `in_training`, `fold`

## `build_tokens.py`

Example run:

```bash
python -m pipeline.build_tokens synthetic
```

Builds every person's timeline (their events with their ages in days), then one
tokenisation per variant: the variant picks its events from the timelines and
gives each one a token number.

- Splits from `split.R`: test outside the training set, validation in fold 5 of
  the training set, train otherwise. The models train on train only.
- Events:
  - diagnoses: the first three characters of the code, at its first contact. The
    codes of the dementia definition of the user's script 2 (F00, F01, F02.0,
    F03.9, G30, G31.0B, G31.8B, G31.8E, G31.9), read on the full code, are
    marked `dementia_code`: Delphi keeps them, our variants drop them.
  - `DEMENTIA` at `DEMENTIA_INCIDENT_DATE`, the outcome of `5 predictions.R`,
    and `DEATH`.
  - `ABROAD_START` and `ABROAD_END` of each period of at least 180 days abroad
    before the index age, as in the user's script 3.
  - sex at age 0, as in Delphi.
- Window: from 1969-01-01 (start of the Psychiatric Central Research Register,
  as in the user's script 3) or birth if later, to death, emigration or
  2022-12-31.
- Test people's timelines stop before the index date, so that their outcome
  never reaches the models.

Variants (one folder each, all built in one run). From one to the next, one
thing changes:

| Variant             | Events                                                                           | Vocabulary                                              | Change from the previous one |
| ------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------- |
| `delphi`            | Delphi's: diagnoses with dementia codes, sex, death; codes of Delphi's list only | Delphi's `labels.csv`: every code its own token         | –                            |
| `delphi_min_people` | Delphi's; codes of Delphi's list only                                            | ours: codes in at least 5 training people, else chapter | the rare-code threshold      |
| `delphi_all_codes`  | Delphi's, codes of every chapter                                                 | ours                                                    | chapter coverage             |
| `whole_lives`       | ours: diagnoses without dementia codes, `DEMENTIA`, `DEATH`, periods abroad, sex | ours                                                    | the events                   |
| `cut_at_index`      | ours, but everyone's timeline stops before the index date                        | ours                                                    | what happens after 70        |

- Our vocabulary: `PAD` (0), `NO_EVENT` (1), `DEATH`, `DEMENTIA`,
  `ABROAD_START`, `ABROAD_END`, `UNKNOWN_CODE`, `Female`, `Male`, one
  `RARE_<chapter>` per ICD-10 chapter, then every three-character code that at
  least 5 training people have. A rarer code takes its chapter's token, or
  `UNKNOWN_CODE` outside every chapter.
- Delphi's vocabulary: `labels.csv` of github.com/gerstung-lab/Delphi at commit
  fb72166 (`data/ukb_real_data/labels.csv`), in `p1_transformers/Delphi/`
  (`delphi_labels` in the config). On the server, copy it there. Its 1,255 codes
  are all in ICD-10 chapters A to Q, so codes of chapters R to Z are dropped.
- `delphi` writes Delphi's own files: a token's number is its line in
  `labels.csv` minus 2 (Female 1, Death 1268), as in Delphi's data. The other
  variants write ours.

Example: one training person, born on 1 June 1930 (index date 1 June 2000, at
70), and what each variant keeps of their timeline. D59 is in fewer than 5
training people, F00 and the other codes in at least 5. A test person's timeline
stops before 70 in every variant, as in `cut_at_index`.

| Age  | Event in the timeline          | `delphi`             | `delphi_min_people`  | `delphi_all_codes`   | `whole_lives`        | `cut_at_index`      |
| ---- | ------------------------------ | -------------------- | -------------------- | -------------------- | -------------------- | ------------------- |
| 0    | sex: female                    | `Female`             | `Female`             | `Female`             | `Female`             | `Female`            |
| 45.0 | leaves Denmark                 | –                    | –                    | –                    | `ABROAD_START`       | `ABROAD_START`      |
| 50.0 | comes back                     | –                    | –                    | –                    | `ABROAD_END`         | `ABROAD_END`        |
| 55.2 | I10.9, hypertension            | `I10`                | `I10`                | `I10`                | `I10`                | `I10`               |
| 61.0 | R51, headache                  | – (not in its list)  | – (not in its list)  | `R51`                | `R51`                | `R51`               |
| 66.3 | D59.1                          | `D59`                | `RARE_BLOOD_IMMUNE`  | `RARE_BLOOD_IMMUNE`  | `RARE_BLOOD_IMMUNE`  | `RARE_BLOOD_IMMUNE` |
| 72.5 | I10.9 again                    | – (first occurrence) | – (first occurrence) | – (first occurrence) | – (first occurrence) | –                   |
| 74.1 | dementia, the date of script 2 | –                    | –                    | –                    | `DEMENTIA`           | – (after 70)        |
| 74.3 | F00.1, a dementia code         | `F00`                | `F00`                | `F00`                | – (dementia code)    | –                   |
| 78.9 | death                          | `Death`              | `DEATH`              | `DEATH`              | `DEATH`              | –                   |

Writes, into the `tokens` folder of the config:

```
tokens/
  persons.parquet      PNR, SPLIT, INDEX_AGE_DAYS, OBSERVED_UNTIL_AGE_DAYS
  timelines.parquet    PNR, AGE_DAYS, KIND, EVENT  (shared by every variant)
  delphi_min_people/   events.parquet (PNR, AGE_DAYS, TOKEN), vocab.csv
  delphi_all_codes/    events.parquet, vocab.csv
  whole_lives/         events.parquet, vocab.csv
  cut_at_index/        events.parquet, vocab.csv
  delphi/              train.bin, val.bin, test.bin (Delphi's format: uint32
                       rows of patient id, age in days, token), patients.parquet
                       (PATIENT_ID, PNR)
```
