# Dementiaformer: transformers for dementia prediction

Mockup of a transformer for dementia prediction from register-based data. Only
synthetic toy data was used for testing.

## Usage (conda)

Run every command from the root of this repository.

1. **Install** the packages into a conda environment that already has python,
   numpy, torch and transformers, and activate it:

   ```bash
   conda env update -n <environment> -f environment.yml
   conda activate <environment>
   ```

2. **Settings**: in `configs/data/server.yaml`, set `root` to the folder of the
   outputs of scripts 2 and 3 (with a trailing slash), and check their file
   names under `paths`. The pipeline writes its own files there too. Every
   command below takes the name of this file: `server`.

3. **Split**, once: draws the train/test split and the cross-validation folds of
   `5 predictions.R`, with its seed, before any tokens are built.

   ```bash
   Rscript pipeline/split.R server
   ```

4. **Tokens**, once: builds every person's timeline from the outputs of scripts
   2 and 3, then the tokens of every variant.

   ```bash
   python -m pipeline.build_tokens server
   ```

5. **Training**, for each arm you choose. An arm is one model trained on one
   variant of the tokens, in any combination. The settings of each model are in
   `configs/models/`.

   | Model                 | Backbone         | Head                                       |
   | --------------------- | ---------------- | ------------------------------------------ |
   | `delphi`              | Delphi's (GPT-2) | Delphi's: one rate per token               |
   | `delphi_v2`           | modern           | Delphi's                                   |
   | `delphi_v2_motorhead` | modern           | MOTOR's: one hazard per token and time bin |

   | Variant             | Events                                                                       | Vocabulary                                      |
   | ------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------- |
   | `delphi`            | Delphi's: diagnoses with dementia codes, sex, death; codes of Delphi's list  | Delphi's: every code its own token              |
   | `delphi_min_people` | Delphi's                                                                     | ours: codes of 5+ training people, else chapter |
   | `delphi_all_codes`  | Delphi's, codes of every ICD-10 chapter                                      | ours                                            |
   | `whole_lives`       | ours: diagnoses without dementia codes, dementia, death, periods abroad, sex | ours                                            |
   | `cut_at_index`      | ours, every timeline stopped before the index age                            | ours                                            |

   A dry run first (a minute at most, nothing written), then the training, which
   takes hours or days. Its folder in `runs/` has `train.log` and `curves.png`
   to follow it. Options: `--resume` continues a stopped run, `--threads N` sets
   the number of CPU threads, `--device cuda` trains on a GPU.

   ```bash
   python -m pipeline.train_torch server delphi whole_lives --dry-run
   python -m pipeline.train_torch server delphi whole_lives
   ```

6. **Embeddings**, for each trained arm: every person's hidden state at the
   index age, in the format of script 4.

   ```bash
   python -m pipeline.extract_embeddings server delphi whole_lives
   ```

7. **Prediction**, once every chosen arm has its embeddings: the Cox models of
   `5 predictions.R` on the embeddings of every arm, next to the baselines L1
   and L2, and their scores (Brier score, IPA, AUC at 85 and 95).

   ```bash
   Rscript pipeline/predict.R server
   ```

Details of each script, its options and what it writes: `docs/pipeline.md`.

## Development (pixi)

With [pixi](https://pixi.sh), on the synthetic data (`pixi run` installs
everything on first use):

```bash
pixi run split synthetic
pixi run build_tokens synthetic
pixi run train_torch synthetic delphi whole_lives --dry-run
pixi run train_torch synthetic delphi whole_lives
pixi run extract_embeddings synthetic delphi whole_lives
pixi run predict synthetic
```

Tests: `pixi run test`. Benchmarks: `pixi run benchmark` (`docs/benchmarks.md`).
