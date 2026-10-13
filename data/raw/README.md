# Raw data

Not in git. Holds:

- `synthetic.zip`: the user's synthetic data, put here by hand.
  `scripts/post_install.sh` unpacks it into `data/rawdata/` and
  `data/workdata/`, as on the user's server.
- `icd10_coding19.tsv`: WHO ICD-10 codes and titles (UK Biobank data-coding 19),
  downloaded by `scripts/post_install.sh`.
