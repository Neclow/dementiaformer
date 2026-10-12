"""Tests of src/tokens.py on small hand-made tables.

Run from the repository root: pixi run test
"""

from dataclasses import replace
from datetime import date
from typing import Any

import polars as pl
import pytest

from src import tokens

SETTINGS = tokens.TokenSettings(
    index_age=70,
    earliest_register_date=date(1969, 1, 1),
    register_end_date=date(2022, 12, 31),
    minimum_spell_days=180,
    min_people_per_code=1,
    validation_fold=5,
    first_occurrence_only=True,
    dementia_subtypes={
        "ALZHEIMER": ["F00", "G30"],
        "VASCULAR": ["F01"],
        "OTHER": ["F02", "G310", "G318", "G319"],
        "UNSPECIFIED": ["F03"],
    },
)
SINGLE = tokens.Variant("single")
SUBTYPE = tokens.Variant("subtype", dementia="subtype")
CUT = tokens.Variant("cut", cut_at_index=True)

# Everyone is born on this day unless a test says otherwise: index date
# 2000-06-01, a year before it 1999-06-01
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
        "SPLIT": split,
        "BIRTH_DATE": birth,
        "DEATH_DATE": death,
        "EMIGRATION_DATE": emigration,
        "DEMENTIA_INCIDENT_DATE": dementia,
    }


def inputs(
    people: list[dict[str, Any]],
    contacts: list[tuple[str, date, str]] | None = None,
    migration: list[tuple[str, date, str]] | None = None,
    settings: tokens.TokenSettings = SETTINGS,
) -> tokens.Inputs:
    """The four inputs as load_inputs returns them. Training people are in fold 1."""
    folds = {"train": 1, "validation": settings.validation_fold, "test": None}
    columns = tokens.INPUT_COLUMNS
    return tokens.Inputs(
        contacts=pl.DataFrame(contacts or [], schema=columns["contacts"], orient="row"),
        persons=pl.DataFrame(people, schema=columns["persons"]),
        migration=pl.DataFrame(
            migration or [], schema=columns["migration"], orient="row"
        ),
        split=pl.DataFrame(
            {
                "PNR": [p["PNR"] for p in people],
                "in_training": [p["SPLIT"] != "test" for p in people],
                "fold": [folds[p["SPLIT"]] for p in people],
            },
            schema=columns["split"],
        ),
    )


def run(
    data: tokens.Inputs,
    variant: tokens.Variant = SINGLE,
    settings: tokens.TokenSettings = SETTINGS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Every step, as pipeline/build_tokens.py runs them.

    Returns the events with the NAME of their token, the people and the
    vocabulary.
    """
    persons = tokens.build_persons(data.persons, data.split, settings)
    timelines = tokens.build_timelines(data.contacts, data.migration, persons, settings)
    timelines, persons = tokens.select_variant(timelines, persons, variant)
    vocabulary = tokens.build_vocabulary(timelines, persons, settings, variant)
    events = tokens.to_tokens(timelines, vocabulary, settings)
    named = events.join(vocabulary.select("TOKEN", "NAME"), on="TOKEN")
    return named, persons, vocabulary


def timeline(events: pl.DataFrame, pnr: str, birth: date = BIRTH) -> set:
    """One person's events as a set of (token name, date)."""
    rows = events.filter(pl.col("PNR") == pnr).iter_rows(named=True)
    return {
        (row["NAME"], date.fromordinal(birth.toordinal() + row["AGE_DAYS"]))
        for row in rows
    }


def test_index_date_of_a_29_february_birthday_is_28_february():
    data = inputs([person("P1", birth=date(1924, 2, 29))])
    persons = tokens.build_persons(data.persons, data.split, SETTINGS)
    row = persons.row(0, named=True)
    assert row["INDEX_DATE"] == date(1994, 2, 28)
    assert row["INDEX_AGE_DAYS"] == (date(1994, 2, 28) - date(1924, 2, 29)).days


def test_split_labels():
    data = inputs(
        [person("P1", "train"), person("P2", "validation"), person("P3", "test")]
    )
    persons = tokens.build_persons(data.persons, data.split, SETTINGS)
    assert persons["SPLIT"].to_list() == ["train", "validation", "test"]


def test_people_missing_from_the_split_file_are_an_error():
    data = inputs([person("P1")])
    with pytest.raises(ValueError, match="missing from the split file"):
        tokens.build_persons(data.persons, data.split.clear(), SETTINGS)


def test_diagnoses_inside_the_record_only():
    data = inputs(
        [
            person("P1", death=date(2010, 1, 1)),
            person("P2", emigration=date(2005, 1, 1), death=date(2008, 1, 1)),
        ],
        contacts=[
            ("P1", date(1965, 1, 1), "I10"),  # before the registers start
            ("P1", date(1980, 1, 1), "E11"),
            ("P1", date(2010, 1, 1), "J18"),  # on the day of death
            ("P1", date(2011, 1, 1), "K21"),  # after death
            ("P2", date(1980, 1, 1), "E11"),
            ("P2", date(2006, 1, 1), "J18"),  # after emigration
        ],
    )
    events, _, _ = run(data)
    assert timeline(events, "P1") == {
        ("E11", date(1980, 1, 1)),
        ("J18", date(2010, 1, 1)),
        ("DEATH", date(2010, 1, 1)),
    }
    # A death after emigration is outside the record too
    assert timeline(events, "P2") == {("E11", date(1980, 1, 1))}


@pytest.mark.parametrize("first_occurrence_only", [True, False])
def test_first_occurrence(first_occurrence_only: bool):
    data = inputs(
        [person("P1")],
        contacts=[
            ("P1", date(1980, 1, 1), "I10"),
            ("P1", date(1990, 1, 1), "I109"),  # same three-character code
            ("P1", date(1990, 1, 1), "I10"),  # same code on the same day: one event
        ],
    )
    settings = replace(SETTINGS, first_occurrence_only=first_occurrence_only)
    events, _, _ = run(data, settings=settings)
    expected = {("I10", date(1980, 1, 1))}
    if not first_occurrence_only:
        expected.add(("I10", date(1990, 1, 1)))
    assert timeline(events, "P1") == expected


def test_dementia_codes_make_way_for_one_dementia_token():
    data = inputs(
        # The client's date (e.g. a prescription) comes before the first code
        [person("P1", dementia=date(2004, 1, 1))],
        contacts=[
            ("P1", date(2005, 1, 1), "F001"),
            ("P1", date(2006, 1, 1), "G309"),
            ("P1", date(2006, 1, 1), "G312"),  # not a dementia code
        ],
    )
    events, _, _ = run(data)
    assert timeline(events, "P1") == {
        ("DEMENTIA", date(2004, 1, 1)),
        ("G31", date(2006, 1, 1)),
    }


def test_dementia_subtype_comes_from_the_first_dementia_code():
    data = inputs(
        [
            person("P1", dementia=date(2005, 1, 1)),
            person("P2", dementia=date(2005, 1, 1)),
            person("P3", dementia=date(2005, 1, 1)),
            person("P4", dementia=date(2005, 1, 1)),
        ],
        contacts=[
            ("P1", date(2005, 1, 1), "F01"),  # first code wins
            ("P1", date(2006, 1, 1), "F00"),
            ("P2", date(2005, 1, 1), "G318B"),
            # P3 has no dementia code: UNSPECIFIED
            ("P4", date(2005, 1, 1), "F03"),  # same day: alphabetical order
            ("P4", date(2005, 1, 1), "F00"),
        ],
    )
    events, _, vocabulary = run(data, variant=SUBTYPE)
    subtypes = {pnr: timeline(events, pnr) for pnr in ["P1", "P2", "P3", "P4"]}
    assert subtypes == {
        "P1": {("DEMENTIA_VASCULAR", date(2005, 1, 1))},
        "P2": {("DEMENTIA_OTHER", date(2005, 1, 1))},
        "P3": {("DEMENTIA_UNSPECIFIED", date(2005, 1, 1))},
        "P4": {("DEMENTIA_ALZHEIMER", date(2005, 1, 1))},
    }
    assert vocabulary.filter(pl.col("KIND") == "dementia")["NAME"].to_list() == [
        "DEMENTIA_ALZHEIMER",
        "DEMENTIA_VASCULAR",
        "DEMENTIA_OTHER",
        "DEMENTIA_UNSPECIFIED",
    ]


def test_rare_codes_share_their_chapter_token():
    day = date(1980, 1, 1)
    data = inputs(
        [person("P1"), person("P2")],
        contacts=[
            ("P1", day, "I10"),
            ("P2", day, "I10"),  # two people: its own token
            ("P1", day, "D48"),  # neoplasms, not blood
            ("P1", day, "D50"),  # blood, not neoplasms
            ("P1", date(1985, 1, 1), "D55"),  # blood again: same token, kept once
            ("P1", day, "H59"),  # eye
            ("P1", day, "H60"),  # ear
            ("P1", day, "D90"),  # outside every chapter
        ],
    )
    events, _, _ = run(data, settings=replace(SETTINGS, min_people_per_code=2))
    assert timeline(events, "P1") == {
        ("I10", day),
        ("RARE_NEOPLASMS", day),
        ("RARE_BLOOD_IMMUNE", day),
        ("RARE_EYE", day),
        ("RARE_EAR", day),
        ("UNKNOWN_CODE", day),
    }


def test_vocabulary_counts_training_people_only():
    day = date(1980, 1, 1)
    data = inputs(
        [person("P1", "train"), person("P2", "validation"), person("P3", "test")],
        contacts=[("P1", day, "I10"), ("P2", day, "I10"), ("P3", day, "I10")],
    )
    # Three people have I10, but only one of them trains the model
    events, _, vocabulary = run(data, settings=replace(SETTINGS, min_people_per_code=2))
    assert "I10" not in vocabulary["NAME"].to_list()
    assert set(events["NAME"]) == {"RARE_CIRCULATORY"}


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
        # Shorter than 180 days, or before the registers: no events
        ([("U", date(1980, 1, 1)), ("I", date(1980, 3, 1))], None),
        ([("U", date(1960, 1, 1)), ("I", date(1965, 1, 1))], None),
    ],
)
def test_periods_abroad(
    crossings: list[tuple[str, date]], period: tuple[date, date] | None
):
    data = inputs(
        [person("P1")],
        migration=[("P1", day, code) for code, day in crossings],
    )
    events, _, _ = run(data)
    expected = set()
    if period is not None:
        expected = {("ABROAD_START", period[0]), ("ABROAD_END", period[1])}
    assert timeline(events, "P1") == expected


def test_test_people_stop_before_the_index_date():
    people = [
        person(pnr, split, death=date(2010, 1, 1), dementia=date(2008, 1, 1))
        for pnr, split in [("P1", "train"), ("P2", "test")]
    ]
    contacts = [
        (pnr, day, code)
        for pnr in ["P1", "P2"]
        for day, code in [
            (date(1990, 1, 1), "I10"),
            (date(2000, 6, 1), "E11"),
        ]  # on the index date
    ]
    events, persons, _ = run(inputs(people, contacts))
    assert timeline(events, "P1") == {
        ("I10", date(1990, 1, 1)),
        ("E11", date(2000, 6, 1)),
        ("DEMENTIA", date(2008, 1, 1)),
        ("DEATH", date(2010, 1, 1)),
    }
    assert timeline(events, "P2") == {("I10", date(1990, 1, 1))}
    observed = persons.select(
        "PNR", "INDEX_AGE_DAYS", "RECORD_END_AGE_DAYS", "OBSERVED_UNTIL_AGE_DAYS"
    )
    p1, p2 = observed.sort("PNR").iter_rows(named=True)
    assert p1["OBSERVED_UNTIL_AGE_DAYS"] == p1["RECORD_END_AGE_DAYS"]
    assert p2["OBSERVED_UNTIL_AGE_DAYS"] == p2["INDEX_AGE_DAYS"]


def test_cut_at_index_stops_everyone_before_the_index_date():
    data = inputs(
        [person("P1", death=date(2010, 1, 1), dementia=date(2008, 1, 1))],
        contacts=[("P1", date(1990, 1, 1), "I10"), ("P1", date(2005, 1, 1), "E11")],
    )
    events, persons, vocabulary = run(data, variant=CUT)
    assert timeline(events, "P1") == {("I10", date(1990, 1, 1))}
    assert persons["OBSERVED_UNTIL_AGE_DAYS"][0] == persons["INDEX_AGE_DAYS"][0]
    # A code seen only after the index date gets no token
    assert "E11" not in vocabulary["NAME"].to_list()


def test_token_numbers():
    data = inputs([person("P1")], contacts=[("P1", date(1980, 1, 1), "I10")])
    _, _, vocabulary = run(data)
    names = vocabulary["NAME"].to_list()
    assert names[tokens.PAD] == "PAD"
    assert names[tokens.NO_EVENT] == "NO_EVENT"
    assert vocabulary["TOKEN"].to_list() == list(range(vocabulary.height))
    assert len(set(names)) == len(names)


def test_variant_rejects_an_unknown_dementia_setting():
    with pytest.raises(ValueError, match="dementia must be"):
        tokens.Variant("typo", dementia="subtypes")  # ty: ignore[invalid-argument-type]
