# 3 strings.R ------------------------------------------------------------------
#
# Builds the inputs for the arms of the analysis plan ("Analysis plan:
# information and granularity levels", Meeting notes):
#   the text arms     -> age70_df_born1913_1947_strings.parquet (one row per person
#                        and string variant)
#   the tabular arms  -> age70_df_born1913_1947_tabular.parquet (one row per person)
#
# Rows of the plan add information, columns add granularity. The string
# variants, one per text cell of the design grid:
#
#   L3_chapters                L3 c   one clause per ICD-10 chapter, in words, random order
#   L3_diag_text_random        L3 d   one clause per diagnosis, its full English
#                                     ICD-10 description, random order
#   L3_diag_text_ordered       L3 e   the same clauses in date order
#   L3_diag_text_dated         L3 f   the same clauses with the age at diagnosis,
#                                     plus the periods spent outside Denmark
#   L4_diag_text_dated_income  L4 f   L3_diag_text_dated plus wage drops before
#                                     age 60, with the age, as events in the
#                                     same sequence. Supplementary analysis only,
#                                     in the cohorts born 1930 or later (see 6b)
#
# Every diagnosis is written as its description, never as its code, and every
# time in a sentence is an age with one decimal, never a calendar year.
#
# The sentences hold diagnoses (and, in f, periods abroad and income drops)
# only. Sex, birth cohort and education are never written into them: they are
# columns, carried on every row of both files, and script 5 adds them to every
# model as categorical predictors. L1 and L2 have no history and no text.
#
# The tabular file holds the columns of the tabular cells: sex, birth cohort,
# education, the risk-factor flags (a), the ICD-10 chapter flags (b) and the
# timing of each chapter (b timed: before 65 only, 65 to 70 only, or both), plus
# three descriptive columns for Table 1 (scripts 3B and 5) that no model uses.
#
# The script reads the long and short files of script 2, the migration file,
# the income file and the ICD-10 descriptions, and writes two parquet files. It
# changes nothing on disk that it did not create.

library(tidyverse)
library(arrow)

# ==== 1. Configuration ========================================================
# Paths and the index age come from the settings file given on the command
# line, configs/server.yaml by default (e.g. from RStudio).
arguments <- commandArgs(trailingOnly = TRUE)
config <- yaml::read_yaml(
  if (length(arguments) >= 1) arguments[1] else "configs/server.yaml"
)
path <- function(name) paste0(config$root, config$paths[[name]])

index_age <- config$index_age # age at which prediction starts
earliest_register_date <- as.Date("1969-01-01") # start of the Psychiatric Central Research Register
minimum_spell_days <- 180 # a period abroad shorter than this gets no clause
income_drop_threshold_percent <- 10 # a smaller year-on-year fall gets no clause
income_drop_before_age <- 60 # from 60 on, a drop is mostly retirement (efterløn)
income_first_birth_cohort <- 1930 # earlier cohorts have under 9 years of wages before 60
recent_contact_from_age <- 65 # b timed: a contact from 65 to 70 is recent, an earlier one remote

# English ICD-10 titles for the diagnosis clauses of d, e, f and L4 f, kept next to the scripts: one
# row per international or Danish ICD-10 code (code, description), taken from
# the ICD10 and ICD10_dk rows of the group's code list metadata_diag_meds_proc_v2.tsv
icd10_descriptions_file <- path("icd10_descriptions")

# Fixes the permutations in section 8, so a re-run reproduces the same strings.
set.seed(709816)

# ==== 2. Code lists ===========================================================

# Risk-factor flags (step a): the risk factors of the Lancet Commission that can
# be identified from hospital diagnoses, each as the three-character ICD-10 codes
# that set it. Physical inactivity, social isolation and air pollution have no
# registry diagnosis, and less education is L2 of the plan, so they are not here.
#   Livingston G, Huntley J, Liu KY, et al. Dementia prevention, intervention,
#   and care: 2024 report of the Lancet standing Commission. Lancet 2024;
#   404(10452):572-628. doi:10.1016/S0140-6736(24)01296-0
risk_factor_codes <- tribble(
  ~risk_factor             , ~codes                               ,
  "HYPERTENSION"           , c("I10", "I11", "I12", "I13", "I15") ,
  "DIABETES"               , c("E10", "E11", "E12", "E13", "E14") ,
  "DEPRESSION"             , c("F32", "F33")                      ,
  "TRAUMATIC_BRAIN_INJURY" , "S06"                                ,
  "HEARING_LOSS"           , c("H90", "H91")                      ,
  "ALCOHOL"                , c("F10", "K70")                      ,
  "HIGH_CHOLESTEROL"       , "E78"                                ,
  "OBESITY"                , "E66"                                ,
  "SMOKING"                , "F17"                                ,
  "VISION_LOSS"            , "H54"
) %>%
  unnest(codes) %>%
  rename(CODE_3 = codes)

# The 22 ICD-10 chapters (steps b and c). A letter is not a chapter: D is split
# between neoplasms and blood, H between eye and ear, and V to Y are one
# chapter. A code belongs to the chapter whose range holds its first three
# characters; flag_name names the column CHAPTER_<flag_name> in the tabular file
# and chapter_name is the wording of the chapter clause in L3_chapters.
icd10_chapters <- tribble(
  ~chapter , ~first_code , ~last_code , ~flag_name         , ~chapter_name                                                        ,
  "I"      , "A00"       , "B99"      , "INFECTIOUS"       , "certain infectious and parasitic diseases"                          ,
  "II"     , "C00"       , "D48"      , "NEOPLASMS"        , "neoplasms"                                                          ,
  "III"    , "D50"       , "D89"      , "BLOOD_IMMUNE"     , "diseases of the blood and immune system"                            ,
  "IV"     , "E00"       , "E90"      , "ENDOCRINE"        , "endocrine, nutritional and metabolic diseases"                      ,
  "V"      , "F00"       , "F99"      , "MENTAL"           , "mental and behavioural disorders"                                   ,
  "VI"     , "G00"       , "G99"      , "NERVOUS"          , "diseases of the nervous system"                                     ,
  "VII"    , "H00"       , "H59"      , "EYE"              , "diseases of the eye"                                                ,
  "VIII"   , "H60"       , "H95"      , "EAR"              , "diseases of the ear"                                                ,
  "IX"     , "I00"       , "I99"      , "CIRCULATORY"      , "diseases of the circulatory system"                                 ,
  "X"      , "J00"       , "J99"      , "RESPIRATORY"      , "diseases of the respiratory system"                                 ,
  "XI"     , "K00"       , "K93"      , "DIGESTIVE"        , "diseases of the digestive system"                                   ,
  "XII"    , "L00"       , "L99"      , "SKIN"             , "diseases of the skin"                                               ,
  "XIII"   , "M00"       , "M99"      , "MUSCULOSKELETAL"  , "diseases of the musculoskeletal system and connective tissue"       ,
  "XIV"    , "N00"       , "N99"      , "GENITOURINARY"    , "diseases of the genitourinary system"                               ,
  "XV"     , "O00"       , "O99"      , "PREGNANCY"        , "pregnancy, childbirth and the puerperium"                           ,
  "XVI"    , "P00"       , "P96"      , "PERINATAL"        , "conditions originating in the perinatal period"                     ,
  "XVII"   , "Q00"       , "Q99"      , "CONGENITAL"       , "congenital malformations and chromosomal abnormalities"             ,
  "XVIII"  , "R00"       , "R99"      , "SYMPTOMS"         , "symptoms, signs and abnormal findings"                              ,
  "XIX"    , "S00"       , "T98"      , "INJURY"           , "injury and poisoning"                                               ,
  "XX"     , "V01"       , "Y98"      , "EXTERNAL_CAUSES"  , "external causes of morbidity"                                       ,
  "XXI"    , "Z00"       , "Z99"      , "HEALTH_SERVICES"  , "factors influencing health status and contact with health services" ,
  "XXII"   , "U00"       , "U99"      , "SPECIAL_PURPOSES" , "codes for special purposes"
) %>%
  mutate(COLUMN = str_c("CHAPTER_", flag_name))

# ==== 3. Load =================================================================
# contacts  one row per person and hospital contact, the output of script 2
# person_level  one row per person, the short file of script 2
# migration one row per border crossing from VNDS, HAEND_DATO the date and
#           INDUD_KODE "U" out or "I" in
# income    one row per person and year, WAGE the yearly wage income (LOENMV_13)
# icd10_descriptions  one row per ICD-10 code: code without a dot, as the
#           registers write it, and description, its English title

contacts <- read_parquet(path("contacts"))
person_level <- read_parquet(path("persons"))
migration <- read_parquet(path("migration"))
income <- read_parquet(path("income"))
icd10_descriptions <- read_csv(
  icd10_descriptions_file,
  col_types = cols(.default = "c")
)

# ==== 4. Cohort spine: one row per person =====================================
# Taken from script 2's short file, which holds the fixed person characteristics
# and the outcome once per person, so the contacts need not be grouped.
#
# EDUCATION is EDU_CAT3 (short, medium, long) as it stood before the 70th
# birthday, with a missing record kept as its own level "unknown" rather than
# dropped. DEMENTIA_BEFORE_INDEX is 0 for everyone, because script 2 already
# excludes prevalent cases; it is kept as the contract with script 5.
# The follow-up columns from script 2 are carried for script 5 and Table 1 and
# are not predictors in any arm: FOLLOW_UP_END_DATE and EVENT_TYPE ("dementia",
# "death" or "censored"), plus EMIGRATION_DATE, where follow-up is censored.
# FOLLOW_UP_END_AGE is the age at that date, the time scale of the survival
# models, counted in days since the 70th birthday (365.25 to a year): every
# event then falls on one of at most 366 times a year, which keeps the survival
# fits and predictions fast.

cohort <- person_level %>%
  transmute(
    PNR,
    SEX,
    BIRTH_DATE,
    BIRTH_COHORT,
    EDUCATION = coalesce(EDU_CAT3, "unknown"),
    DEATH_DATE,
    FOLLOW_UP_END_DATE,
    EVENT_TYPE,
    EMIGRATION_DATE,
    DEMENTIA,
    DEMENTIA_INCIDENT_DATE
  ) %>%
  mutate(
    INDEX_DATE = BIRTH_DATE %m+% years(index_age),
    FOLLOW_UP_END_AGE = index_age +
      as.numeric(FOLLOW_UP_END_DATE - INDEX_DATE) / 365.25,
    WINDOW_START = pmax(BIRTH_DATE, earliest_register_date),
    DEMENTIA_BEFORE_INDEX = as.integer(
      !is.na(DEMENTIA_INCIDENT_DATE) &
        DEMENTIA_INCIDENT_DATE < INDEX_DATE
    )
  )

# The birth date and the two window dates on their own, so later steps can join
# them onto events without dragging the whole cohort table along. The birth date
# turns every event date into the age that section 7 writes into the sentences.
observation_window <- cohort %>%
  select(PNR, BIRTH_DATE, WINDOW_START, INDEX_DATE)

# ==== 5. Periods spent outside Denmark ========================================
# Filter first, bind last: migration stays in its own table until section 7, so
# no filter on a diagnosis column can silently drop its rows.
#
# 5a. An ordered event list per person. A code that repeats the previous one is
# dropped, so a departure is never paired with another departure, and lead()
# lets each departure see the return that follows it.

migration_events <- migration %>%
  semi_join(cohort, by = "PNR") %>%
  mutate(EVENT_DATE = as.Date(HAEND_DATO)) %>%
  filter(!is.na(EVENT_DATE), INDUD_KODE %in% c("U", "I")) %>%
  arrange(PNR, EVENT_DATE) %>%
  group_by(PNR) %>%
  filter(is.na(lag(INDUD_KODE)) | INDUD_KODE != lag(INDUD_KODE)) %>%
  mutate(
    NEXT_CODE = lead(INDUD_KODE),
    NEXT_DATE = lead(EVENT_DATE),
    IS_FIRST = row_number() == 1
  ) %>%
  ungroup()

# 5b. Spells: a departure opens one, closed by the next return or left open; a
# first event that is a return means the person was abroad before the registers
# start. Every spell is clipped to the observation window, and a spell still open
# at age 69 is cut there, because the population was defined as resident in
# Denmark from 69 to 70.

spells <- bind_rows(
  migration_events %>%
    filter(INDUD_KODE == "U") %>%
    transmute(
      PNR,
      FROM_DATE = EVENT_DATE,
      TO_DATE = if_else(NEXT_CODE == "I", NEXT_DATE, as.Date(NA))
    ),
  migration_events %>%
    filter(IS_FIRST, INDUD_KODE == "I") %>%
    transmute(PNR, FROM_DATE = as.Date(NA), TO_DATE = EVENT_DATE)
) %>%
  inner_join(observation_window, by = "PNR") %>%
  mutate(
    AGE69_DATE = INDEX_DATE %m-% years(1),
    START_DATE = pmax(coalesce(FROM_DATE, WINDOW_START), WINDOW_START),
    END_DATE = pmin(coalesce(TO_DATE, INDEX_DATE), INDEX_DATE),
    END_DATE = if_else(
      is.na(TO_DATE) & END_DATE > AGE69_DATE,
      AGE69_DATE,
      END_DATE
    ),
    SPELL_DAYS = as.numeric(END_DATE - START_DATE)
  ) %>%
  filter(SPELL_DAYS > 0)

# ==== 6. Hospital contacts inside the observation window ======================
# One row per person, contact date and code, kept only on or after WINDOW_START
# and strictly before the 70th birthday, so nothing from the index date itself
# leaks in. distinct() on these three columns only: with patient and contact
# type gone, two rows that differ only in those would otherwise give the same
# clause twice.
#
#   CODE_3   the first three characters, which the risk-factor flags (a) and the
#            chapter ranges (b, c) match on. The codes arrive as plain ICD-10:
#            the preprocessing scripts remove the Danish "D" prefix and map
#            ICD-8 to ICD-10, so no prefix is removed here (it would destroy
#            genuine D50-D89 codes)
#   COLUMN,  the chapter flag and chapter name from the range join on
#   chapter_name  icd10_chapters; NA for a code outside every range, which
#            keeps its clauses in d, e and f and is counted in 11c. The ranges
#            do not overlap, so the join adds no rows
#   DESCRIPTION  the English title of the code, or of its nearest parent when
#            the code itself is not in the list (a Danish extension such as
#            G318B gets the title of G318); NA when no parent is either, which
#            keeps the code in d, e and f and is counted in 11c

# The chapter and the description of every distinct code, looked up once
# rather than on every contact
code_lookup <- contacts %>%
  distinct(DIAG) %>%
  filter(!is.na(DIAG)) %>%
  mutate(CODE_3 = str_sub(DIAG, 1, 3)) %>%
  left_join(
    icd10_chapters %>% select(first_code, last_code, COLUMN, chapter_name),
    by = join_by(between(CODE_3, first_code, last_code)),
    relationship = "many-to-one"
  ) %>%
  mutate(
    DESCRIPTION = reduce(
      map(7:3, \(code_length) {
        icd10_descriptions$description[match(
          str_sub(DIAG, 1, code_length),
          icd10_descriptions$code
        )]
      }),
      coalesce
    )
  ) %>%
  select(DIAG, CODE_3, COLUMN, chapter_name, DESCRIPTION)

hospital_contacts <- contacts %>%
  distinct(PNR, INDATE, DIAG) %>%
  filter(!is.na(INDATE), !is.na(DIAG)) %>%
  inner_join(observation_window, by = "PNR") %>%
  filter(INDATE >= WINDOW_START, INDATE < INDEX_DATE) %>%
  left_join(code_lookup, by = "DIAG")

# ==== 6b. Wage drops before age 60 ============================================
# A drop is a year in which the wage fell by at least
# income_drop_threshold_percent from the year before. Only the years before the
# year of the 60th birthday count: from 60 on, efterløn and then folkepension
# make most drops ordinary retirement rather than a sign of ill health. Only
# people born in income_first_birth_cohort or later get drops: the income
# register starts in 1980, so they have at least 9 years in which a drop can be
# seen, while those born before 1922 have none. Script 5 fits L4 on these
# cohorts only, as a supplementary analysis.
#
#   the join    puts the wage of the year before next to each year, matched on
#               the year, so a year with no record before it gets no drop and
#               two years that are not adjacent are never compared
#   the filter  a fall needs a positive wage the year before and a recorded wage
#               this year. A fall to zero is a 100% drop; a rise from zero is not
#               a fall, so there is never a division by zero
#   EVENT_DATE  the income year is dated to its last day, which places it after
#               every diagnosis of the same year in the chronological strings,
#               and the age on that day is the age the sentence gives the drop

income_drops <- income %>%
  inner_join(observation_window, by = "PNR") %>%
  filter(year(BIRTH_DATE) >= income_first_birth_cohort) %>%
  filter(Year < year(BIRTH_DATE) + income_drop_before_age) %>%
  inner_join(
    income %>% transmute(PNR, Year = Year + 1, PREVIOUS_WAGE = WAGE),
    by = c("PNR", "Year")
  ) %>%
  filter(PREVIOUS_WAGE > 0, !is.na(WAGE)) %>%
  mutate(DROP_PERCENT = round(100 * (PREVIOUS_WAGE - WAGE) / PREVIOUS_WAGE)) %>%
  filter(DROP_PERCENT >= income_drop_threshold_percent) %>%
  transmute(PNR, BIRTH_DATE, DROP_PERCENT, EVENT_DATE = make_date(Year, 12, 31))

# ==== 7. Clauses ==============================================================
# One row per event, with every wording a variant may need. SOURCE says what the
# event is, so section 8 can choose the events of each variant. Diagnoses carry
# two wordings of the same event, both with the description in place of the
# code (the code itself when it has none), so the diagnosis variants differ only
# in order and age, never in which events they describe:
#
#   CLAUSE        no reference to time   L3_diag_text_random, L3_diag_text_ordered
#   CLAUSE_DATED  the age in front       L3_diag_text_dated, L4_diag_text_dated_income
#
# The chapter clauses (c) are one per distinct chapter per person, with the
# chapter named in words, so they carry exactly the information of the chapter
# flags (b). The periods abroad and the income drops are dated events only,
# since they only appear in dated variants; their one wording serves both.
#
# age_one_decimal() gives the age on a date as text with one decimal, cut rather
# than rounded, so a contact the day before the 70th birthday reads 69.9 and
# never 70.0, which would look like part of the outcome period.
age_one_decimal <- function(birth_date, date) {
  sprintf(
    "%.1f",
    floor(10 * time_length(interval(birth_date, date), "years")) / 10
  )
}

clauses <- bind_rows(
  hospital_contacts %>%
    mutate(DIAGNOSIS_TEXT = coalesce(DESCRIPTION, DIAG)) %>%
    transmute(
      PNR,
      SOURCE = "diagnosis",
      EVENT_DATE = INDATE,
      CLAUSE = str_c("The person had a diagnosis of ", DIAGNOSIS_TEXT, "."),
      CLAUSE_DATED = str_c(
        "At age ",
        age_one_decimal(BIRTH_DATE, INDATE),
        " the person had a diagnosis of ",
        DIAGNOSIS_TEXT,
        "."
      )
    ),
  hospital_contacts %>%
    filter(!is.na(chapter_name)) %>%
    group_by(PNR, chapter_name) %>%
    summarise(EVENT_DATE = min(INDATE), .groups = "drop") %>%
    transmute(
      PNR,
      SOURCE = "chapter",
      EVENT_DATE,
      CLAUSE = str_c("The person had a diagnosis of ", chapter_name, ".")
    ),
  spells %>%
    filter(SPELL_DAYS >= minimum_spell_days) %>%
    transmute(
      PNR,
      SOURCE = "abroad",
      EVENT_DATE = START_DATE,
      CLAUSE_DATED = str_c(
        "From age ",
        age_one_decimal(BIRTH_DATE, START_DATE),
        " to age ",
        age_one_decimal(BIRTH_DATE, END_DATE),
        " the person lived outside Denmark, so no Danish",
        " health records exist for that time."
      )
    ),
  income_drops %>%
    transmute(
      PNR,
      SOURCE = "income",
      EVENT_DATE,
      CLAUSE_DATED = str_c(
        "At age ",
        age_one_decimal(BIRTH_DATE, EVENT_DATE),
        " the person's wage fell by ",
        DROP_PERCENT,
        "%."
      )
    )
)

# ==== 8. Bodies: the clauses of one person joined into one text ===============
# One row per history variant: which events it holds, which wording it uses,
# and whether the clauses are shuffled or kept in date order.
#
# A shuffle is one independent permutation per person, drawn by sorting each
# person's clauses on a random number, so the score of a random arm is already an
# average over as many orderings as there are people. A draw that happens to come
# out in date order is kept: re-drawing until the order differs would make every
# two-clause history come out reversed, a systematic anti-chronological signal
# that S4 would then detect. The clauses, sorted by person, are joined into one
# text per person by stri_join_list() in one call, not one call per person.
#
# HAS_DIAGNOSIS_CLAUSE marks whether the body holds at least one diagnosis or
# chapter clause, so section 9 can say "no hospital diagnoses" wherever it
# holds none: a person with no contact, one whose only events are a period
# abroad or an income drop, and, in L3_chapters, one whose codes fall in no
# chapter. Without it, that last group would get an empty string.

history_variants <- tribble(
  ~STRING_VARIANT             , ~sources                           , ~wording       , ~shuffle ,
  "L3_chapters"               , "chapter"                          , "CLAUSE"       , TRUE     ,
  "L3_diag_text_random"       , "diagnosis"                        , "CLAUSE"       , TRUE     ,
  "L3_diag_text_ordered"      , "diagnosis"                        , "CLAUSE"       , FALSE    ,
  "L3_diag_text_dated"        , c("diagnosis", "abroad")           , "CLAUSE_DATED" , FALSE    ,
  "L4_diag_text_dated_income" , c("diagnosis", "abroad", "income") , "CLAUSE_DATED" , FALSE
)

bodies <- history_variants %>%
  pmap(\(STRING_VARIANT, sources, wording, shuffle) {
    variant_clauses <- clauses %>%
      filter(SOURCE %in% sources) %>%
      arrange(PNR, if (shuffle) runif(n()) else EVENT_DATE)
    tibble(
      PNR = unique(variant_clauses$PNR),
      BODY = stringi::stri_join_list(
        split(
          variant_clauses[[wording]],
          factor(variant_clauses$PNR, levels = PNR)
        ),
        sep = " "
      ),
      HAS_DIAGNOSIS_CLAUSE = PNR %in%
        variant_clauses$PNR[
          variant_clauses$SOURCE %in% c("diagnosis", "chapter")
        ],
      STRING_VARIANT = STRING_VARIANT
    )
  }) %>%
  bind_rows()

# ==== 9. The strings, one row per person and variant ==========================
# 9a. The share of the window the person was resident in Denmark. A column only:
# it is not a predictor in any arm and is not written into any sentence.
days_abroad <- spells %>%
  group_by(PNR) %>%
  summarise(DAYS_ABROAD = sum(SPELL_DAYS), .groups = "drop")

person_covariates <- cohort %>%
  left_join(days_abroad, by = "PNR") %>%
  mutate(
    FRACTION_OBSERVED_IN_DENMARK = 1 -
      coalesce(DAYS_ABROAD, 0) /
        as.numeric(INDEX_DATE - WINDOW_START)
  )

# 9b. Every person once per variant: a no-diagnoses sentence where the body
# holds no diagnosis clause, then the body. No opening sentence: sex, birth
# cohort and education are columns, not text.
no_diagnoses_sentence <- str_c(
  "The person had no hospital diagnoses before age ",
  index_age,
  "."
)

strings_long <- person_covariates %>%
  expand_grid(STRING_VARIANT = history_variants$STRING_VARIANT) %>%
  left_join(bodies, by = c("PNR", "STRING_VARIANT")) %>%
  mutate(
    STRING = case_when(
      coalesce(HAS_DIAGNOSIS_CLAUSE, FALSE) ~ BODY,
      is.na(BODY) ~ no_diagnoses_sentence,
      .default = str_c(no_diagnoses_sentence, " ", BODY)
    )
  ) %>%
  select(
    PNR,
    STRING_VARIANT,
    STRING,
    SEX,
    BIRTH_COHORT,
    EDUCATION,
    BIRTH_DATE,
    INDEX_DATE,
    DEMENTIA,
    DEMENTIA_INCIDENT_DATE,
    DEMENTIA_BEFORE_INDEX,
    DEATH_DATE,
    FOLLOW_UP_END_DATE,
    FOLLOW_UP_END_AGE,
    EVENT_TYPE,
    EMIGRATION_DATE,
    FRACTION_OBSERVED_IN_DENMARK
  )

# ==== 10. Tabular inputs ======================================================
# One row per person, from the same windowed contacts as the strings, so the
# tabular and text arms see exactly the same events.
#   RISK_<factor>      1 if any contact carries one of the factor's codes (step a)
#   CHAPTER_<chapter>  1 if any contact carries a code in that chapter (step b);
#                      the prefixes script 5 selects on
#   TIMING_<chapter>_REMOTE_ONLY, TIMING_<chapter>_RECENT_ONLY  (b timed) 1 if
#                      all the person's contacts in the chapter fall before the
#                      65th birthday, or all from 65 to 70. Both are 0 without a
#                      contact in the chapter and for contacts in both periods,
#                      so with the chapter flag they give the four patterns none,
#                      remote only, recent only and both without collinearity,
#                      and an arm whose timing terms are all shrunk to zero is
#                      the chapter arm b. Not in Table 1
#   DISTINCT_DIAGNOSIS_CODES, ANY_INCOME_DROP, ANY_PERIOD_ABROAD  for Table 1
#                      only: the number of different codes in the window (0 for
#                      no contact), whether the person has an income drop in L4,
#                      and whether they have a period abroad long enough to get a
#                      clause in f (minimum_spell_days). None starts with RISK_,
#                      CHAPTER_ or TIMING_, so no arm picks them up

risk_factor_flags <- hospital_contacts %>%
  inner_join(
    risk_factor_codes,
    by = "CODE_3",
    relationship = "many-to-many"
  ) %>%
  distinct(PNR, COLUMN = str_c("RISK_", risk_factor)) %>%
  mutate(FLAG = 1L)

chapter_flags <- hospital_contacts %>%
  filter(!is.na(COLUMN)) %>%
  distinct(PNR, COLUMN) %>%
  mutate(FLAG = 1L)

# One row per person, chapter and period with a contact; a chapter left with
# one row has contacts in one period only
chapter_timing_flags <- hospital_contacts %>%
  filter(!is.na(COLUMN)) %>%
  mutate(RECENT = INDATE >= BIRTH_DATE %m+% years(recent_contact_from_age)) %>%
  distinct(PNR, COLUMN, RECENT) %>%
  filter(n() == 1, .by = c(PNR, COLUMN)) %>%
  mutate(
    COLUMN = str_c(
      str_replace(COLUMN, "^CHAPTER_", "TIMING_"),
      if_else(RECENT, "_RECENT_ONLY", "_REMOTE_ONLY")
    )
  ) %>%
  transmute(PNR, COLUMN, FLAG = 1L)

tabular <- person_covariates %>%
  select(
    PNR,
    SEX,
    BIRTH_COHORT,
    EDUCATION,
    BIRTH_DATE,
    INDEX_DATE,
    DEMENTIA,
    DEMENTIA_INCIDENT_DATE,
    DEMENTIA_BEFORE_INDEX,
    DEATH_DATE,
    FOLLOW_UP_END_DATE,
    FOLLOW_UP_END_AGE,
    EVENT_TYPE,
    EMIGRATION_DATE,
    FRACTION_OBSERVED_IN_DENMARK
  ) %>%
  left_join(
    bind_rows(risk_factor_flags, chapter_flags, chapter_timing_flags) %>%
      pivot_wider(names_from = COLUMN, values_from = FLAG),
    by = "PNR"
  ) %>%
  left_join(
    hospital_contacts %>%
      distinct(PNR, DIAG) %>%
      count(PNR, name = "DISTINCT_DIAGNOSIS_CODES"),
    by = "PNR"
  ) %>%
  mutate(
    DISTINCT_DIAGNOSIS_CODES = coalesce(DISTINCT_DIAGNOSIS_CODES, 0L),
    ANY_INCOME_DROP = as.integer(PNR %in% income_drops$PNR),
    ANY_PERIOD_ABROAD = as.integer(
      PNR %in% spells$PNR[spells$SPELL_DAYS >= minimum_spell_days]
    )
  )

# A flag that is set for no one still needs its column, or the toy run and the
# server run would fit different model formulas. The NAs left by the join are
# people with no such contact, so they become zeros.
flag_columns <- c(
  str_c("RISK_", unique(risk_factor_codes$risk_factor)),
  icd10_chapters$COLUMN,
  str_c("TIMING_", icd10_chapters$flag_name, "_REMOTE_ONLY"),
  str_c("TIMING_", icd10_chapters$flag_name, "_RECENT_ONLY")
)
tabular[setdiff(flag_columns, names(tabular))] <- 0L
tabular <- tabular %>%
  mutate(across(all_of(flag_columns), ~ replace_na(.x, 0L)))

# ==== 11. Checks ==============================================================
# Aggregates only, printed for checking on the server and not written to disk.
# The sentences themselves are microdata and must not leave the server; check
# that no printed count is below five before taking one home.

# 11a. Shape of each variant. Expect the same number of people in all of them,
# and the dated variants a little longer than L3_diag_text_ordered. The last
# column counts strings that mention sex, birth, education, patient type or
# contact type, none of which belongs in a sentence any more: it must be 0 in
# L3_chapters, while the diagnosis variants may hold such a word in a disease
# name (female, newborn), the same count in each of them.
strings_long %>%
  group_by(STRING_VARIANT) %>%
  summarise(
    people = n(),
    median_characters = median(nchar(STRING)),
    p99_characters = quantile(nchar(STRING), 0.99, names = FALSE),
    with_period_abroad = sum(str_detect(STRING, "outside Denmark")),
    with_income_drop = sum(str_detect(STRING, "wage fell")),
    without_diagnoses = sum(str_detect(STRING, "no hospital diagnoses")),
    with_forbidden_word = sum(str_detect(
      STRING,
      regex(
        "male|born|education|inpatient|outpatient|emergency|hospital contact|main diagnosis|secondary diagnosis",
        ignore_case = TRUE
      )
    )),
    .groups = "drop"
  ) %>%
  print(width = Inf)

# 11b. The checks the granularity contrasts depend on, person by person:
#   random_is_permutation  L3_diag_text_random must hold exactly the clauses of
#                          L3_diag_text_ordered, or S4 compares content as well as
#                          order. The sorted clauses must be identical
#   chapters_match_flags   L3_chapters must hold one clause per chapter flag, or
#                          S2 compares information as well as representation.
#                          The regex only counts the 22 chapter names, so a
#                          clause with any other wording would show up here
chapter_clause_pattern <- str_c(
  "had a diagnosis of (",
  str_c(str_escape(icd10_chapters$chapter_name), collapse = "|"),
  ")\\."
)

strings_long %>%
  filter(
    STRING_VARIANT %in%
      c("L3_diag_text_ordered", "L3_diag_text_random", "L3_chapters")
  ) %>%
  select(PNR, STRING_VARIANT, STRING) %>%
  pivot_wider(names_from = STRING_VARIANT, values_from = STRING) %>%
  left_join(
    tabular %>%
      transmute(PNR, chapter_flags = rowSums(pick(starts_with("CHAPTER_")))),
    by = "PNR"
  ) %>%
  summarise(
    random_is_permutation = mean(map2_lgl(
      str_split(L3_diag_text_ordered, "(?<=\\.) "),
      str_split(L3_diag_text_random, "(?<=\\.) "),
      \(ordered, random) identical(sort(ordered), sort(random))
    )),
    chapters_match_flags = mean(
      str_count(L3_chapters, chapter_clause_pattern) == chapter_flags
    )
  ) %>%
  print()

# 11c. The diagnoses that fall in none of the 22 chapters, by calendar period.
# They stay in d, e and f but are absent from b and c. Should be near 0 in both
# periods: a high share before 1994 means the ICD-8 to ICD-10 mapping of the
# preprocessing left codes unmapped, a high share from 1994 on means codes that
# are not plain ICD-10. The same for the diagnoses with no description at any
# level, which keep their code in d, e and f.
hospital_contacts %>%
  group_by(
    period = if_else(INDATE < as.Date("1994-01-01"), "before 1994", "1994 on")
  ) %>%
  summarise(
    diagnoses = n(),
    without_chapter = sum(is.na(COLUMN)),
    share_without_chapter = mean(is.na(COLUMN)),
    without_description = sum(is.na(DESCRIPTION)),
    share_without_description = mean(is.na(DESCRIPTION)),
    .groups = "drop"
  ) %>%
  print()

# 11d. The wage drops before 60, in the cohorts born 1930 or later. The share
# of those people with a drop is the denominator Table 1 does not show.
income_drops %>%
  summarise(
    people_with_drop = n_distinct(PNR),
    share_of_people = n_distinct(PNR) /
      sum(year(cohort$BIRTH_DATE) >= income_first_birth_cohort),
    drops = n(),
    share_fall_to_zero = mean(DROP_PERCENT == 100),
    median_age_at_drop = median(time_length(
      interval(BIRTH_DATE, EVENT_DATE),
      "years"
    )),
    median_drop_percent = median(DROP_PERCENT)
  ) %>%
  print()

# 11e. The tabular file: education levels, and the prevalence of every flag.
# One chapter near 1 and the rest near 0 would mean the codes carry a prefix
# after all; every chapter near 0, that they are not ICD-10 (see 11c).
tabular %>% count(EDUCATION, name = "people") %>% print()

tabular %>%
  summarise(people = n(), across(all_of(flag_columns), mean)) %>%
  pivot_longer(-people, names_to = "flag", values_to = "prevalence") %>%
  print(n = Inf)

# 11f. The chapter flags here against the chapter flags script 2 wrote into its
# short file. Same codes and same ranges, but script 2 has no lower bound on the
# contact date, so a contact before 1969 sets its chapter there and not here; a
# non-zero count most likely means pre-1969 records exist rather than that
# either definition is broken.
script2_chapter_flags <- person_level %>%
  select(PNR, all_of(icd10_chapters$COLUMN)) %>%
  pivot_longer(-PNR, names_to = "flag", values_to = "script2")

script2_chapter_flags %>%
  left_join(
    chapter_flags %>% select(PNR, flag = COLUMN, script3 = FLAG),
    by = c("PNR", "flag")
  ) %>%
  mutate(script3 = coalesce(script3, 0L)) %>%
  summarise(
    disagreements = sum(script3 != script2),
    people_affected = n_distinct(PNR[script3 != script2]),
    set_in_script2_only = sum(script2 == 1 & script3 == 0),
    set_in_script3_only = sum(script3 == 1 & script2 == 0)
  ) %>%
  print()

# ==== 12. Save ================================================================

write_parquet(strings_long, path("strings"))
write_parquet(tabular, path("tabular"))

sessionInfo()
