# Points for the paper

## Delphi's vocabulary misses ICD-10 chapters common in Danish hospital data

- Delphi-2M's vocabulary (`labels.csv` in gerstung-lab/Delphi) has 1,255
  three-character ICD-10 codes, all in chapters A–Q (I–XVII). It has no code
  from:
  - R: symptoms and abnormal findings
  - S–T: injuries and poisoning
  - V–Y: external causes
  - Z: factors influencing health status and contact with health services
  - U: special purposes
- This follows from its source: UK Biobank's "first occurrence" fields.
- In Danish hospital data these chapters are common, and some matter for
  dementia:
  - S06 (intracranial injury, i.e. traumatic brain injury) is a Lancet
    Commission risk factor, and one of the risk-factor flags of `3 strings.R`.
  - S72 (hip fracture) is a frequent event in old age.
- In the synthetic data, 7 of the 34 non-dementia codes are missing from
  Delphi's list: R51, R55, S72, Z09, Z50, V89, U07.
- Our vocabulary keeps every chapter (see `tokens.md`). Applying Delphi-2M
  itself to Danish data would mean dropping these codes.
  - To check: how the Delphi paper handled this if it validated on Danish
    registry data (two co-authors, Brunak and Mortensen, work with Danish
    registries).
- **To quantify on the server:** the share of contacts, and of people, with at
  least one code outside A–Q. `build_tokens` prints it. (Synthetic data: 21% of
  diagnoses, 56% of training people, but its codes are drawn uniformly.)
