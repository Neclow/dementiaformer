#!/usr/bin/env bash
# Set up the data here: the user's synthetic data laid out as on their server, and the ICD-10 descriptions
set -euo pipefail
cd "$(dirname "$0")/.."

# Unpack the synthetic data into data/rawdata and data/workdata, as on the server's E:/, without the macOS files
unpacked=$(mktemp -d)
unzip -q data/raw/synthetic.zip -d "$unpacked" -x "__MACOSX/*" "*.DS_Store"
cp -r "$unpacked/synthetic/data/." data/
rm -rf "$unpacked"

# WHO ICD-10 codes and descriptions, from UK Biobank's data-coding 19
curl -sS -X POST -d id=19 https://biobank.ndph.ox.ac.uk/ukb/codown.cgi -o data/raw/icd10_coding19.tsv
