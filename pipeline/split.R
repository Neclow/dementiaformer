# Train/test split and cross-validation folds (step 5a), as in 5 predictions.R

library(dplyr)
library(arrow)

# ==== Configuration ===========================================================

# Share of people in training (the rest is the test set), as in 5 predictions.R
training_share <- 0.7
# Folds among training people, as in 5 predictions.R
cross_validation_folds <- 5
# Random seed, as in 5 predictions.R
SEED <- 709816

# Paths from the settings file given on the command line, configs/server.yaml
# by default (e.g. from RStudio)
arguments <- commandArgs(trailingOnly = TRUE)
config <- yaml::read_yaml(
  if (length(arguments) >= 1) arguments[1] else "configs/server.yaml"
)
# Input: the tabular file of strings.R, as in 5 predictions.R
input <- paste0(config$root, config$paths$tabular)
output <- paste0(config$root, config$paths$split)

# ==== The split and the cross-validation folds ================================
# As in 5 predictions.R, from set.seed() to write_parquet()

set.seed(SEED)
split_and_folds <- read_parquet(
  input,
  # Columns selected: PNR, birth cohort, event type
  col_select = c(PNR, BIRTH_COHORT, EVENT_TYPE)
) |>
  # Sort by PNR, so that identical inputs give identical splits
  arrange(PNR) |>
  # Create columns:
  #   - in_training: TRUE for training_share of the people, drawn within each
  #     event type and birth cohort
  mutate(
    in_training = seq_len(n()) %in% sample(n(), round(training_share * n())),
    .by = c(EVENT_TYPE, BIRTH_COHORT)
  ) |>
  # Create columns:
  #   - fold: 1 to cross_validation_folds, spread evenly over training people
  #     in random order; NA for test people
  mutate(
    fold = replace(
      rep(NA_integer_, n()),
      in_training,
      sample(rep_len(seq_len(cross_validation_folds), sum(in_training)))
    )
  ) |>
  # Columns selected: PNR, in_training, fold
  select(PNR, in_training, fold)

write_parquet(split_and_folds, output)

# ==== Check ===================================================================
# Count and print the people per set and fold

split_and_folds |>
  count(in_training, fold, name = "people") |>
  print()
