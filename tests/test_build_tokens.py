"""Tests of pipeline/build_tokens.py on small hand-made tables."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from pipeline.build_tokens import (
    cut_at_index,
    make_persons,
    make_timelines,
    make_vocabulary,
    read_delphi_labels,
    select_codes,
    select_events,
    tokenise,
    write,
)
from src._config import ICD10_CHAPTERS, SPECIAL_TOKENS

# Everyone's birth date unless a test says otherwise: index date 2000-06-01 at
# 70, a year before it 1999-06-01
BIRTH = date(1930, 6, 1)


def person(
    pnr: str,
    split: str = "train",
    birth: date = BIRTH,
    death: date | None = None,
    emigration: date | None = None,
    dementia: date | None = None,
) -> dict[str, Any]:
    """One row of the person file, with its split: train, validation or test."""
    return {
        "PNR": pnr,
        "SEX": "Female",
        "BIRTH_DATE": birth,
        "DEATH_DATE": death,
        "EMIGRATION_DATE": emigration,
        "DEMENTIA_INCIDENT_DATE": dementia,
        "SPLIT": split,
    }


def run(
    people: list[dict[str, Any]],
    contacts: list[tuple[str, date, str]] | None = None,
    crossings: list[tuple[str, date, str]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """make_persons and make_timelines at index age 70 on these rows."""
    folds = {"train": 1, "validation": 5, "test": None}
    per_person_raw = pl.DataFrame(
        people,
        schema={
            "PNR": pl.String,
            "SEX": pl.String,
            "BIRTH_DATE": pl.Date,
            "DEATH_DATE": pl.Date,
            "EMIGRATION_DATE": pl.Date,
            "DEMENTIA_INCIDENT_DATE": pl.Date,
            "SPLIT": pl.String,
        },
    ).drop("SPLIT")
    split = pl.DataFrame(
        {
            "PNR": [p["PNR"] for p in people],
            "in_training": [p["SPLIT"] != "test" for p in people],
            "fold": [folds[p["SPLIT"]] for p in people],
        }
    )
    per_diagnosis_raw = pl.DataFrame(
        contacts or [],
        schema={"PNR": pl.String, "INDATE": pl.Date, "DIAG": pl.String},
        orient="row",
    )
    per_crossing_raw = pl.DataFrame(
        crossings or [],
        schema={"PNR": pl.String, "HAEND_DATO": pl.Date, "INDUD_KODE": pl.String},
        orient="row",
    )
    persons = make_persons(per_person_raw, split, 70)
    return persons, make_timelines(per_diagnosis_raw, per_crossing_raw, persons)


def events(timelines: pl.DataFrame, pnr: str, birth: date = BIRTH) -> set:
    """One person's events as a set of (event, date), without sex."""
    rows = timelines.filter(pl.col("PNR") == pnr, pl.col("KIND") != "sex")
    return {
        (row["EVENT"], birth + timedelta(days=row["AGE_DAYS"]))
        for row in rows.iter_rows(named=True)
    }


def events_of(events: pl.DataFrame, pnr: str, birth: date = BIRTH) -> set:
    """One person's events as a set of (event, date)."""
    rows = events.filter(pl.col("PNR") == pnr).iter_rows(named=True)
    return {(row["EVENT"], birth + timedelta(days=row["AGE_DAYS"])) for row in rows}


def test_index_date_of_a_29_february_birthday_is_28_february():
    persons, _ = run([person("P1", birth=date(1924, 2, 29))])
    assert persons["INDEX_DATE"][0] == date(1994, 2, 28)


def test_split():
    persons, _ = run(
        [person("P1", "train"), person("P2", "validation"), person("P3", "test")]
    )
    assert persons["SPLIT"].to_list() == ["train", "validation", "test"]


def test_diagnoses_inside_the_record_at_their_first_contact():
    people = [
        person("P1", death=date(2010, 1, 1)),
        person("P2", emigration=date(2005, 1, 1), death=date(2008, 1, 1)),
    ]
    contacts = [
        # Before the registers start
        ("P1", date(1965, 1, 1), "I10"),
        ("P1", date(1980, 1, 1), "E11"),
        # Same three-character code later
        ("P1", date(1990, 1, 1), "E119"),
        # On the day of death
        ("P1", date(2010, 1, 1), "J18"),
        # After death
        ("P1", date(2011, 1, 1), "K21"),
        ("P2", date(1980, 1, 1), "E11"),
        # After emigration
        ("P2", date(2006, 1, 1), "J18"),
    ]
    _, timelines = run(people, contacts)
    assert events(timelines, "P1") == {
        ("E11", date(1980, 1, 1)),
        ("J18", date(2010, 1, 1)),
        ("DEATH", date(2010, 1, 1)),
    }
    # A death after emigration is outside the record too
    assert events(timelines, "P2") == {("E11", date(1980, 1, 1))}


def test_dementia_at_its_date_and_dementia_codes_kept():
    people = [person("P1", dementia=date(2004, 1, 1))]
    _, timelines = run(people, [("P1", date(2005, 1, 1), "F001")])
    assert events(timelines, "P1") == {
        ("DEMENTIA", date(2004, 1, 1)),
        ("F00", date(2005, 1, 1)),
    }


def test_sex_at_age_0():
    _, timelines = run([person("P1")])
    assert timelines.filter(pl.col("KIND") == "sex").rows() == [
        ("P1", 0, "sex", "Female")
    ]


def test_test_people_stop_before_the_index_date():
    people = [
        person(pnr, split, death=date(2010, 1, 1), dementia=date(2008, 1, 1))
        for pnr, split in [("P1", "train"), ("P2", "test")]
    ]
    contacts = [
        (pnr, day, code)
        for pnr in ["P1", "P2"]
        # The second contact is on the index date
        for day, code in [(date(1990, 1, 1), "I10"), (date(2000, 6, 1), "E11")]
    ]
    persons, timelines = run(people, contacts)
    assert events(timelines, "P1") == {
        ("I10", date(1990, 1, 1)),
        ("E11", date(2000, 6, 1)),
        ("DEMENTIA", date(2008, 1, 1)),
        ("DEATH", date(2010, 1, 1)),
    }
    assert events(timelines, "P2") == {("I10", date(1990, 1, 1))}
    p1, p2 = persons.iter_rows(named=True)
    assert p1["OBSERVED_UNTIL_AGE_DAYS"] == (date(2010, 1, 1) - BIRTH).days
    assert p2["OBSERVED_UNTIL_AGE_DAYS"] == p2["INDEX_AGE_DAYS"]


@pytest.mark.parametrize(
    ("crossings", "period"),
    [
        # A departure and a return
        (
            [("U", date(1980, 1, 1)), ("I", date(1985, 1, 1))],
            (date(1980, 1, 1), date(1985, 1, 1)),
        ),
        # A repeated departure is dropped
        (
            [("U", date(1980, 1, 1)), ("U", date(1981, 1, 1)), ("I", date(1985, 1, 1))],
            (date(1980, 1, 1), date(1985, 1, 1)),
        ),
        # A first crossing that is a return: abroad since the registers started
        ([("I", date(1975, 1, 1))], (date(1969, 1, 1), date(1975, 1, 1))),
        # Still abroad: cut a year before the index date
        ([("U", date(1995, 1, 1))], (date(1995, 1, 1), date(1999, 6, 1))),
        # Clipped to the start of the registers
        (
            [("U", date(1960, 1, 1)), ("I", date(1975, 1, 1))],
            (date(1969, 1, 1), date(1975, 1, 1)),
        ),
        # Shorter than 180 days
        ([("U", date(1980, 1, 1)), ("I", date(1980, 3, 1))], None),
        # Before the registers start
        ([("U", date(1960, 1, 1)), ("I", date(1965, 1, 1))], None),
    ],
)
def test_periods_abroad(
    crossings: list[tuple[str, date]], period: tuple[date, date] | None
):
    _, timelines = run([person("P1")], crossings=[("P1", d, c) for c, d in crossings])
    expected = set()
    if period is not None:
        expected = {("ABROAD_START", period[0]), ("ABROAD_END", period[1])}
    assert events(timelines, "P1") == expected


def make_events(rows: list[tuple[str, int, str, str]]) -> pl.DataFrame:
    """Events as in a timeline: (PNR, AGE_DAYS, KIND, EVENT)."""
    return pl.DataFrame(
        rows,
        schema={
            "PNR": pl.String,
            "AGE_DAYS": pl.Int32,
            "KIND": pl.String,
            "EVENT": pl.String,
        },
        orient="row",
    )


def test_vocabulary_has_the_codes_of_at_least_5_training_people():
    splits = ["train"] * 5 + ["validation", "test"]
    persons = pl.DataFrame({"PNR": [f"P{i}" for i in range(7)], "SPLIT": splits})
    # I10 in 5 training people; E11 in 4 training people, 1 validation and 1 test
    rows = [(f"P{i}", 100, "diagnosis", "I10") for i in range(5)]
    rows += [(f"P{i}", 100, "diagnosis", "E11") for i in range(1, 7)]
    vocabulary = make_vocabulary(make_events(rows), persons, "ours", None)
    names = vocabulary["NAME"].to_list()
    assert names[0] == "PAD"
    assert names[1] == "NO_EVENT"
    assert names[len(SPECIAL_TOKENS)] == "RARE_INFECTIOUS"
    assert names[len(SPECIAL_TOKENS) + len(ICD10_CHAPTERS) :] == ["I10"]
    assert vocabulary["TOKEN"].to_list() == list(range(len(names)))


def test_tokenise_codes_without_their_own_token():
    # No code has its own token: every code goes to its chapter
    persons = pl.DataFrame({"PNR": ["P1"], "SPLIT": ["train"]})
    events = make_events(
        [
            # Neoplasms, not blood
            ("P1", 100, "diagnosis", "D48"),
            # Blood, twice: one token at its first age
            ("P1", 200, "diagnosis", "D50"),
            ("P1", 300, "diagnosis", "D55"),
            # Eye, then ear
            ("P1", 400, "diagnosis", "H59"),
            ("P1", 400, "diagnosis", "H60"),
            # Outside every chapter
            ("P1", 500, "diagnosis", "D90"),
            ("P1", 600, "death", "DEATH"),
        ]
    )
    vocabulary = make_vocabulary(events, persons, "ours", None)
    tokens = tokenise(events, vocabulary).join(vocabulary, on="TOKEN")
    assert set(tokens.select("NAME", "AGE_DAYS").iter_rows()) == {
        ("RARE_NEOPLASMS", 100),
        ("RARE_BLOOD_IMMUNE", 200),
        ("RARE_EYE", 400),
        ("RARE_EAR", 400),
        ("UNKNOWN_CODE", 500),
        ("DEATH", 600),
    }


def test_dementia_codes_are_read_on_the_full_code():
    day = date(1980, 1, 1)
    # F02.0 and G31.0B are dementia codes, F02.1 and G31.2 are not
    contacts = [("P1", day, code) for code in ["F020", "F021", "G310B", "G312"]]
    _, timelines = run([person("P1")], contacts)
    kinds = timelines.filter(pl.col("KIND") != "sex").select("KIND", "EVENT").rows()
    assert sorted(kinds) == [
        ("dementia_code", "F02"),
        ("dementia_code", "G31"),
        ("diagnosis", "F02"),
        ("diagnosis", "G31"),
    ]


@pytest.mark.parametrize("cut", [False, True])
def test_our_events_whole_lives_or_cut_at_index(cut: bool):
    people = [person("P1", death=date(2010, 1, 1), dementia=date(2008, 1, 1))]
    contacts = [
        ("P1", date(1990, 1, 1), "I10"),
        ("P1", date(2005, 1, 1), "E11"),
        ("P1", date(2008, 1, 1), "F001"),
    ]
    persons, timelines = run(people, contacts)
    events = select_events(timelines, "ours")
    if cut:
        events = cut_at_index(events, persons)
    # Sex at age 0 is kept; the dementia code F00 never is
    expected = {("Female", BIRTH), ("I10", date(1990, 1, 1))}
    if not cut:
        expected |= {
            ("E11", date(2005, 1, 1)),
            ("DEMENTIA", date(2008, 1, 1)),
            ("DEATH", date(2010, 1, 1)),
        }
    assert events_of(events, "P1") == expected


# A short labels.csv, laid out as Delphi's
DELPHI_LABELS = [
    "Padding",
    "No event",
    "Female",
    "Male",
    "F00 (dementia in alzheimer's disease)",
    "I10 (essential (primary) hypertension)",
    "Death",
]


def test_read_delphi_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("\n".join(DELPHI_LABELS) + "\n")
    labels = dict(read_delphi_labels(path).iter_rows())
    assert (labels["Female"], labels["Male"], labels["F00"], labels["DEATH"]) == (
        1,
        2,
        3,
        5,
    )


def test_delphi_tokens_and_files(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("\n".join(DELPHI_LABELS) + "\n")
    labels = read_delphi_labels(path)
    people = [person("P1", dementia=date(2008, 1, 1), death=date(2010, 1, 1))]
    contacts = [
        ("P1", date(1990, 1, 1), "I10"),
        # Same code later: first occurrence only
        ("P1", date(1995, 1, 1), "I109"),
        # Not in Delphi's list: dropped
        ("P1", date(1991, 1, 1), "R51"),
        # A dementia code: kept, as in Delphi
        ("P1", date(2009, 1, 1), "F001"),
        # On the day of death: death still comes last
        ("P1", date(2010, 1, 1), "I10"),
    ]
    crossings = [("P1", date(1980, 1, 1), "U"), ("P1", date(1985, 1, 1), "I")]
    persons, timelines = run(people, contacts, crossings)
    delphi_codes = labels["NAME"].to_list()
    events = select_codes(select_events(timelines, "delphi"), "delphi", delphi_codes)
    tokens = tokenise(events, labels)
    names = {token: name for name, token in labels.iter_rows()}
    # No DEMENTIA token and no periods abroad
    assert [(names[t], a) for _, a, t in tokens.iter_rows()] == [
        ("Female", 0),
        ("I10", (date(1990, 1, 1) - BIRTH).days),
        ("F00", (date(2009, 1, 1) - BIRTH).days),
        ("DEATH", (date(2010, 1, 1) - BIRTH).days),
    ]

    write(tokens, labels, persons, tmp_path, "delphi")
    rows = np.fromfile(tmp_path / "train.bin", dtype=np.uint32).reshape(-1, 3)
    assert rows[:, 0].tolist() == [0, 0, 0, 0]
    assert rows[:, 2].tolist() == [1, 4, 3, 5]
    assert np.fromfile(tmp_path / "test.bin", dtype=np.uint32).size == 0


def test_delphi_all_codes_keeps_every_chapter():
    people = [person("P1", dementia=date(2008, 1, 1), death=date(2010, 1, 1))]
    contacts = [
        ("P1", date(1990, 1, 1), "I10"),
        # Not in Delphi's list, but kept here
        ("P1", date(1991, 1, 1), "R51"),
        # A dementia code, kept as a diagnosis
        ("P1", date(2009, 1, 1), "F001"),
    ]
    crossings = [("P1", date(1980, 1, 1), "U"), ("P1", date(1985, 1, 1), "I")]
    persons, timelines = run(people, contacts, crossings)
    events = select_codes(select_events(timelines, "delphi"), "ours", [])
    vocabulary = make_vocabulary(events, persons, "ours", None)
    tokens = tokenise(events, vocabulary).join(vocabulary, on="TOKEN").sort("AGE_DAYS")
    # One person: every code is rare and takes its chapter's token. No DEMENTIA
    # token and no periods abroad
    assert tokens["NAME"].to_list() == [
        "Female",
        "RARE_CIRCULATORY",
        "RARE_SYMPTOMS",
        "RARE_MENTAL",
        "DEATH",
    ]


def test_delphi_min_people_keeps_delphi_codes_and_groups_the_rare_ones(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("\n".join(DELPHI_LABELS) + "\n")
    labels = read_delphi_labels(path)
    people = [person("P1", death=date(2010, 1, 1))]
    contacts = [
        # On Delphi's list, in one training person: its chapter's token
        ("P1", date(1990, 1, 1), "I10"),
        # Not on Delphi's list: dropped
        ("P1", date(1991, 1, 1), "R51"),
    ]
    persons, timelines = run(people, contacts)
    events = select_codes(
        select_events(timelines, "delphi"), "delphi", labels["NAME"].to_list()
    )
    vocabulary = make_vocabulary(events, persons, "ours", None)
    tokens = tokenise(events, vocabulary).join(vocabulary, on="TOKEN").sort("AGE_DAYS")
    assert tokens["NAME"].to_list() == ["Female", "RARE_CIRCULATORY", "DEATH"]
