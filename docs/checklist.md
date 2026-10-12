# Checklist for the server

Some settings of this project depend on scripts we cannot see, or on how they
are run on the server. Please check each point below before running the pipeline
on the real data, and tell us if anything differs.

## Before building the tokens

- [ ] **The dementia codes.** `configs/data_server.toml` lists, under
      `[settings.dementia_subtypes]`, the codes the tokeniser treats as
      dementia. They come from the synthetic data (`make_synthetic_data.R`):
      F00–F03, G30, G31.0, G31.8, G31.9 and their sub-codes. Please check them
      against the definition of dementia in script 2, the one behind
      `DEMENTIA_INCIDENT_DATE`, and correct the list there. Why it matters:
  - The tokeniser places dementia as one token at `DEMENTIA_INCIDENT_DATE`, and
    removes the dementia codes from the ordinary diagnoses so that dementia is
    not counted twice.
  - A code in our list but not in your definition would be removed from people's
    diagnoses, also before age 70, while `3 strings.R` keeps it.
  - A code in your definition but not in our list would stay as an ordinary
    diagnosis after age 70, so the model would see dementia twice, under two
    tokens.
  - If your definition also uses prescriptions (anti-dementia drugs), people
    whose date comes from a prescription only get the subtype "unspecified". The
    subtypes are only used by the variant `whole_lives_subtype`.
- [ ] **The outputs of script 2.** The tokeniser reads the columns listed in
      `docs/tokens.md`. Please confirm that:
  - the contacts file holds contacts after age 70 too, up to the end of the
    registers: the model learns from them (`3 strings.R` filters them out, which
    suggests they are there);
  - `EMIGRATION_DATE` is the date of emigration after age 70, where follow-up is
    censored;
  - the registers end on 2022-12-31 (`register_end_date` in the config).
- [ ] **The split.** Run `5 predictions.R` up to the line that writes
      `age70_split_and_folds.parquet`, so that the split exists before the
      tokens are built.

## Once the tokens are built

- [ ] **Keep the split.** Do not redraw the split (re-running script 3 with
      other inputs changes it). The tokens, and every model trained on them, are
      only valid with the split they were built with; `settings.json` keeps its
      fingerprint. If the split has to change, rebuild the tokens and retrain.
- [ ] **Check the printed totals** of `build_tokens` for surprises, e.g. a large
      share of `UNKNOWN_CODE` or of rare codes, or no dementia tokens. As with
      any output, check that no count is below 5 before taking it off the
      server.
