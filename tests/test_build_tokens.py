"""Tests of pipeline/build_tokens.py on small hand-made tables."""

from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from pipeline.build_tokens import make_persons, make_timelines

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
