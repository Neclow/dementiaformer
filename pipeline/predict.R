# Fits the transformer arms as 5 predictions.R fits its arms: two penalised
# cause-specific Cox models (dementia, and death without dementia as the
# competing event), on the embeddings of extract_embeddings.py, and scores each
# arm's risk of dementia by age 85 (primary) and 95 (secondary) on the test
# people of split.R, next to the baselines L1 and L2. The code that fits and
# scores an arm is that of 5 predictions.R, unchanged.

library(tidyverse)
library(arrow)
library(prodlim)
library(riskRegression)

# ==== Configuration ===========================================================

# As in 5 predictions.R
horizon_ages <- c(primary = 85, secondary = 95)
# The registers end on 2022-12-31
last_register_year <- 2022
# glmnet alpha: ridge, elastic net, lasso
elastic_net_mixing <- c(0, 0.5, 1)
# CPUs per arm for its folds
cross_validation_workers <- 5
# Arms fitted side by side: 2 x 5 = 10 CPUs
arms_at_once <- 2
# hjemtagelse: no exported count from 1 to 4
minimum_cell_count <- 5
# TRUE resumes after a crash from the arms already saved
reuse_saved_arms <- FALSE
# Every cohort
first_birth_cohort <- 1913
# The person columns enter every arm unpenalised, so the penalty only selects
# history
unpenalised_columns <- c("SEX", "BIRTH_COHORT_CATEGORY", "EDUCATION")

# Paths from the settings file named on the command line: synthetic or server
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1) {
  stop("Name the settings file: Rscript pipeline/predict.R synthetic (or server)")
}
config <- yaml::read_yaml(
  file.path("configs", "data", paste0(arguments[1], ".yaml"))
)
index_age <- config$index_age
# Time in the models is years since the index age
horizon_years <- horizon_ages - index_age
arm_results_folder <- paste0(config$root, config$paths$arm_results)
results_folder <- paste0(config$root, config$paths$results)
dir.create(arm_results_folder, showWarnings = FALSE, recursive = TRUE)
dir.create(results_folder, showWarnings = FALSE, recursive = TRUE)

# ==== The arms ================================================================

# The baselines, L1 and L2 of 5 predictions.R: sex and birth year, then education
tabular_arms <- tribble(
  ~arm,         ~predictors,
  "tabular_L1", quote(c(SEX, BIRTH_COHORT_CATEGORY)),
  "tabular_L2", quote(c(SEX, BIRTH_COHORT_CATEGORY, EDUCATION))) |>
  mutate(fit_id = arm, embedding_model = NA_character_,
         file = paste0(config$root, config$paths$tabular))

# The transformer arms: every file of extract_embeddings.py in the data folder,
# age70_dataset_embeddings_<model>_<variant>.parquet, the model with
# _lightning for a run of train_lightning.py
embeddings_pattern <- str_c(
  "^age", index_age,
  "_dataset_embeddings_((delphi_v2_motorhead|delphi_v2|delphi)(_lightning)?)_(.+)[.]parquet$"
)
transformer_arms <- tibble(file = list.files(config$root, pattern = embeddings_pattern, full.names = TRUE)) |>
  mutate(parts           = str_match(basename(file), embeddings_pattern),
         embedding_model = parts[, 2],
         arm             = parts[, 5],
         fit_id          = str_c(embedding_model, "_", arm),
         predictors      = list(quote(c(SEX, BIRTH_COHORT_CATEGORY, EDUCATION, starts_with("embedding_"))))) |>
  select(-parts)
if (nrow(transformer_arms) == 0) {
  stop("No embeddings in ", config$root, ": run extract_embeddings.py first")
}

# All arms, each with the file its result is saved to
model_arms <- bind_rows(tabular_arms, transformer_arms) |>
  mutate(first_birth_cohort = first_birth_cohort,
         saved_result = file.path(arm_results_folder, str_c("age", index_age, "_", fit_id, ".rds")))

# ==== The split and the cross-validation folds ================================

# The split and folds of split.R, the same as 5 predictions.R draws
split_and_folds <- read_parquet(paste0(config$root, config$paths$split))

# ==== Fitting one arm =========================================================
# As in 5 predictions.R

# Runs in a fresh R process, which sees only its arguments
fit_arm <- function(fit_id, file, predictors, first_birth_cohort, saved_result, split_and_folds, settings) {
  library(tidyverse)
  library(arrow)
  library(prodlim)
  library(riskRegression)

  # The arm's cohorts with their split, folds and outcome: event_status is 1 for
  # dementia, 2 for death without dementia and 0 for censored, the coding Hist() reads
  cohort <- read_parquet(file) |>
    filter(DEMENTIA_BEFORE_INDEX == 0) |>
    filter(BIRTH_COHORT >= first_birth_cohort) |>
    inner_join(split_and_folds, by = "PNR") |>
    arrange(PNR) |>
    mutate(follow_up_years = FOLLOW_UP_END_AGE - settings$index_age,
           event_status = match(EVENT_TYPE, c("dementia", "death"), nomatch = 0L),
           BIRTH_COHORT_CATEGORY = factor(BIRTH_COHORT),
           across(c(SEX, EDUCATION), as.factor))
  predictor_names <- names(tidyselect::eval_select(predictors, cohort))
  cohort <- select(cohort, PNR, BIRTH_COHORT, in_training, fold, follow_up_years, event_status, all_of(predictor_names))
  training <- filter(cohort, in_training)
  penalised_columns <- setdiff(predictor_names, settings$unpenalised_columns)

  # L1 and L2 have nothing to penalise: standard cause-specific Cox, Breslow ties like glmnet
  if (length(penalised_columns) == 0) {
    model <- CSC(reformulate(predictor_names, response = quote(Hist(follow_up_years, event_status))),
                 data = training, cause = 1, ties = "breslow")
    tuning <- tibble(cause = c("dementia", "death"), alpha = NA_real_, lambda = 0, cv_deviance = NA_real_,
                     penalised_columns_kept = 0L)
    coefficients <- map(model$models, coef)
  } else {
    # One glmnet fit per mixing value, the penalty of each cause tuned on the
    # shared folds in parallel; keep the mixing value with the lowest summed
    # deviance (the cross-validated likelihood of the competing-risks model)
    doParallel::registerDoParallel(parallel::makePSOCKcluster(settings$cross_validation_workers))
    model_formula <- reformulate(c(str_c("unpenalized(", intersect(predictor_names, settings$unpenalised_columns), ")"),
                                   penalised_columns),
                                 response = quote(Hist(follow_up_years, event_status)))
    candidates <- map(settings$elastic_net_mixing, \(alpha)
      CSC(model_formula, data = training, cause = 1, fitter = "glmnet", alpha = alpha, foldid = training$fold, parallel = TRUE))
    cv_deviance <- map(candidates, \(candidate) map_dbl(candidate$models, \(cause_model) min(cause_model$fit$cvm)))
    best <- which.min(map_dbl(cv_deviance, sum))
    model <- candidates[[best]]
    rm(candidates)
    tuning <- tibble(cause = c("dementia", "death"), alpha = settings$elastic_net_mixing[best],
                     lambda = map_dbl(model$models, "selected.lambda"), cv_deviance = cv_deviance[[best]],
                     penalised_columns_kept = map_int(model$models, \(cause_model)
                       sum(cause_model$selected.beta[cause_model$penalty.factor > 0, 1] != 0)))
    coefficients <- map(model$models, \(cause_model) cause_model$selected.beta[, 1])
  }

  # The same scores, without intervals, on a random sample of the training
  # people as large as the test set, among the cohorts each horizon is scored in:
  # a train score much better than the test score shows the arm overfits. A
  # sample, because predicting is the slow part of an arm. A horizon no cohort of
  # the arm reaches (95 for the income arms) is skipped
  set.seed(709816)
  training_sample <- slice_sample(training, n = sum(!cohort$in_training))
  training_risk <- matrix(predictRisk(model, newdata = training_sample, times = settings$horizon_years, cause = 1),
                          ncol = 2, dimnames = list(NULL, names(settings$horizon_years)))
  reached_horizons <- keep(settings$horizon_ages, \(horizon_age) any(training_sample$BIRTH_COHORT + horizon_age <= settings$last_register_year))
  training_scores <- imap_dfr(reached_horizons, \(horizon_age, horizon) {
    scored <- training_sample$BIRTH_COHORT + horizon_age <= settings$last_register_year
    scores <- Score(list(arm = training_risk[scored, horizon, drop = FALSE]),
                    formula = Hist(follow_up_years, event_status) ~ BIRTH_COHORT, data = training_sample[scored, ],
                    cause = 1, times = settings$horizon_years[[horizon]], metrics = c("auc", "brier"),
                    summary = "ipa", cens.model = "cox", se.fit = FALSE)
    brier <- filter(as_tibble(scores$Brier$score), model == "arm")
    tibble(fit_id, horizon, train_brier = brier$Brier, train_ipa = brier$IPA, train_auc = scores$AUC$score$AUC)
  })

  # Save the held-out people's risk of dementia by each horizon, the scores on
  # the training people, the tuning and the coefficients, but not the model,
  # which holds copies of the training data
  held_out <- filter(cohort, !in_training)
  saveRDS(list(tuning = mutate(tuning, fit_id = fit_id, penalised_columns = length(penalised_columns), .before = 1),
               training_scores = training_scores,
               coefficients = imap_dfr(coefficients, \(log_hazard_ratio, cause)
                 tibble(fit_id, cause = c("Cause 1" = "dementia", "Cause 2" = "death")[[cause]],
                        term = names(log_hazard_ratio), log_hazard_ratio = unname(log_hazard_ratio))),
               predictions = held_out |>
                 select(PNR, BIRTH_COHORT, follow_up_years, event_status) |>
                 mutate(risk = matrix(predictRisk(model, newdata = held_out, times = settings$horizon_years, cause = 1),
                                      ncol = 2, dimnames = list(NULL, names(settings$horizon_years))))),
          saved_result)
}

# ==== Fitting every arm =======================================================

# Each arm not yet saved in a background R process, as in 5 predictions.R (an
# arm's process frees its memory when it ends), at most arms_at_once at a time;
# each logs next to its result
arm_settings <- lst(index_age, horizon_ages, horizon_years, last_register_year, elastic_net_mixing,
                    unpenalised_columns, cross_validation_workers)
arms_to_fit <- filter(model_arms, !(reuse_saved_arms & file.exists(saved_result)))
arm_processes <- list()
for (arm in seq_len(nrow(arms_to_fit))) {
  while (sum(map_lgl(arm_processes, \(process) process$is_alive())) >= arms_at_once) Sys.sleep(5)
  message(Sys.time(), "  fitting ", arms_to_fit$fit_id[arm])
  arm_processes[[arm]] <- callr::r_bg(fit_arm,
                                      args = list(arms_to_fit$fit_id[arm], arms_to_fit$file[arm], arms_to_fit$predictors[[arm]],
                                                  arms_to_fit$first_birth_cohort[arm], arms_to_fit$saved_result[arm],
                                                  split_and_folds, arm_settings),
                                      stdout = str_replace(arms_to_fit$saved_result[arm], "rds$", "log"),
                                      stderr = "2>&1", supervise = TRUE)
}

# Wait for the last arms, and stop with the error of any arm that failed
walk(arm_processes, \(process) {
  process$wait()
  process$get_result()
})

# Read the saved results back, the tuning each arm and cause ended with, and the scores on the training people
fitted_arms <- mutate(model_arms, result = map(saved_result, readRDS))
tuning_settings <- map_dfr(fitted_arms$result, "tuning")
training_scores <- map_dfr(fitted_arms$result, "training_scores")

# ==== The contrasts ===========================================================

# Each transformer arm against L2, as P2 of 5 predictions.R: what does the
# diagnosis history add, read by the transformer?
arm_contrasts <- transformer_arms |>
  transmute(contrast = "P2_transformer", arm_a = fit_id, arm_b = "tabular_L2",
            question = "What does the diagnosis history add? (transformer embedding)",
            embedding_model)

# Every arm holds the whole test set, so all are scored on it together
scoring_populations <- tibble(population = "all_test_people", fit_ids = list(fitted_arms$fit_id))

# ==== Scores ==================================================================
# As in 5 predictions.R

# Aalen-Johansen risk of dementia by a time, with death as a competing event
aalen_johansen_risk <- function(follow_up_years, event_status, horizon_time) {
  # Without dementia among these people, the risk is 0 (prodlim cannot estimate
  # a cause that never occurs; small groups only, e.g. on synthetic data). The
  # only change to the scoring code of 5 predictions.R
  if (!any(event_status == 1)) return(0)
  predict(prodlim(Hist(follow_up_years, event_status) ~ 1, data = tibble(follow_up_years, event_status)),
          times = horizon_time, cause = 1)
}

# Scores the arms of one population at each horizon, among the cohorts that can
# be followed to it before the registers end (born up to 1937 for 85, up to
# 1927 for 95): the Brier score, IPA (scaled Brier score against the
# Aalen-Johansen null model) and AUC of every arm, the paired contrasts between
# its arms, and observed (Aalen-Johansen) against mean predicted risk by tenth
# of predicted risk. Censoring weights come from a Cox model on birth year; the
# risks go in as one-column matrices, as this version of Score() needs;
# conservative = TRUE saves memory at a negligible cost
score_population <- function(population, fit_ids) {
  # The population's people, from its first arm, and every arm's risks for them in that order
  results <- fitted_arms$result[match(fit_ids, fitted_arms$fit_id)]
  people <- select(results[[1]]$predictions, -risk)
  risks <- set_names(map(results, \(result) result$predictions$risk[match(people$PNR, result$predictions$PNR), , drop = FALSE]),
                     fit_ids)
  contrasts <- filter(arm_contrasts, arm_a %in% fit_ids, arm_b %in% fit_ids)

  # Only the horizons some of the people can be followed to: not 95 for the
  # income arms, whose risk by 95 is missing, as no one they were fitted on reached it
  reached_horizons <- keep(horizon_ages, \(horizon_age) any(people$BIRTH_COHORT + horizon_age <= last_register_year))
  imap(reached_horizons, \(horizon_age, horizon) {
    scored <- people$BIRTH_COHORT + horizon_age <= last_register_year
    stopifnot(!anyNA(unlist(map(risks, \(risk) risk[scored, horizon]))))
    scored_people <- people[scored, ]
    horizon_time <- horizon_years[[horizon]]
    scores <- Score(map(risks, \(risk) risk[scored, horizon, drop = FALSE]),
                    formula = Hist(follow_up_years, event_status) ~ BIRTH_COHORT, data = scored_people,
                    cause = 1, times = horizon_time, metrics = c("auc", "brier"), summary = "ipa",
                    contrasts = unique(map2(match(contrasts$arm_b, fit_ids), match(contrasts$arm_a, fit_ids), c)),
                    cens.model = "cox", conservative = TRUE)

    # One row per arm, with the people scored and the calibration in the large
    performance <- as_tibble(scores$Brier$score) |>
      transmute(fit_id = as.character(model), brier = Brier, brier_conf_low = lower, brier_conf_high = upper, ipa = IPA) |>
      left_join(as_tibble(scores$AUC$score) |>
                  transmute(fit_id = as.character(model), auc = AUC, auc_conf_low = lower, auc_conf_high = upper),
                by = "fit_id") |>
      mutate(population, horizon, horizon_age, n_scored = nrow(scored_people),
             n_dementia_by_horizon = sum(scored_people$event_status == 1 & scored_people$follow_up_years <= horizon_time),
             mean_predicted_risk = map_dbl(fit_id, \(id) if (id %in% fit_ids) mean(risks[[id]][scored, horizon]) else NA),
             observed_risk = aalen_johansen_risk(scored_people$follow_up_years, scored_people$event_status, horizon_time))

    # One row per contrast: the paired Brier and AUC differences, arm_a minus
    # arm_b. A lower Brier score and a higher AUC are better, so the two point in
    # opposite directions
    paired <- contrasts |>
      inner_join(as_tibble(scores$Brier$contrasts) |>
                   transmute(arm_a = as.character(model), arm_b = as.character(reference),
                             brier_difference = delta.Brier, brier_conf_low = lower, brier_conf_high = upper),
                 by = c("arm_a", "arm_b")) |>
      inner_join(as_tibble(scores$AUC$contrasts) |>
                   transmute(arm_a = as.character(model), arm_b = as.character(reference),
                             auc_difference = delta.AUC, auc_conf_low = lower, auc_conf_high = upper),
                 by = c("arm_a", "arm_b")) |>
      mutate(population, horizon, horizon_age)

    # One row per arm and tenth of predicted risk
    calibration <- map_dfr(fit_ids, \(id) mutate(scored_people, fit_id = id, risk = risks[[id]][scored, horizon])) |>
      mutate(risk_tenth = ntile(risk, 10), .by = fit_id) |>
      summarise(people = n(),
                dementia_by_horizon = sum(event_status == 1 & follow_up_years <= horizon_time),
                mean_predicted_risk = mean(risk),
                observed_risk = aalen_johansen_risk(follow_up_years, event_status, horizon_time),
                .by = c(fit_id, risk_tenth)) |>
      mutate(population, horizon, horizon_age, .before = 1)

    lst(performance, paired, calibration)
  })
}

# Every population at every horizon
population_scores <- list_flatten(pmap(scoring_populations, score_population))

# One row per population, arm and horizon, with the scores on the training
# people as train_brier, train_ipa and train_auc
arm_performance <- map_dfr(population_scores, "performance") |>
  left_join(training_scores, by = c("fit_id", "horizon")) |>
  left_join(select(model_arms, fit_id, arm, embedding_model), by = "fit_id")

# One row per population, contrast and horizon
contrast_results <- map_dfr(population_scores, "paired")

# ==== Calibration by tenth of predicted risk ==================================
# As in 5 predictions.R

# A count is too small to export when it, or what is left of its total, lies
# between 1 and minimum_cell_count - 1; a zero reveals no one
is_small_count <- function(count, total) {
  coalesce(between(count, 1, minimum_cell_count - 1) | between(total - count, 1, minimum_cell_count - 1), FALSE)
}

# A single blank among cells with a published total can be recovered, so the
# smallest other non-zero cell is blanked with it
add_secondary_suppression <- function(suppressed, count) {
  candidates <- if_else(!suppressed & count > 0, count, Inf)
  if (sum(suppressed) != 1 || all(is.infinite(candidates))) return(suppressed)
  suppressed | candidates == min(candidates)
}

# The calibration by tenth of every population, suppressed for hjemtagelse
calibration_by_risk_tenth <- map_dfr(population_scores, "calibration") |>
  mutate(suppressed = is_small_count(dementia_by_horizon, people)) |>
  mutate(suppressed = add_secondary_suppression(suppressed, dementia_by_horizon), .by = c(population, fit_id, horizon)) |>
  mutate(across(c(dementia_by_horizon, mean_predicted_risk, observed_risk), \(value) if_else(suppressed, NA, value))) |>
  select(-suppressed)

# ==== Save ====================================================================

# Aggregated and suppressed, ready for hjemtagelse; the arm results are not.
# Named transformer_, next to the tables of 5 predictions.R
results_file <- function(table) file.path(results_folder, str_c("age", index_age, "_transformer_", table, ".csv"))
write_csv(arm_performance, results_file("arm_performance"))
write_csv(contrast_results, results_file("arm_contrasts"))
write_csv(tuning_settings, results_file("tuning_settings"))
write_csv(calibration_by_risk_tenth, results_file("calibration_by_risk_tenth"))

# Record the package versions in the output
sessionInfo()
