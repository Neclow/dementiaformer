bootstrap_auc <- function(risks, people, horizon_time) {
  map_dfr(seq_len(bootstrap_replicates), function(replicate) {
    resampled <- sample(nrow(people), replace = TRUE)
    scores <- Score(map(risks, function(risk) risk[resampled, , drop = FALSE]),
                    formula = Hist(follow_up_years, event_status) ~ BIRTH_COHORT, data = people[resampled, ],
                    cause = 1, times = horizon_time, metrics = "auc", cens.model = "cox", se.fit = FALSE)
    as_tibble(scores$AUC$score) |>
      transmute(replicate, model = as.character(model), auc = AUC)
  })
}