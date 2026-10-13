"""ICD-10 chapter of a three-character code."""

import polars as pl

from src._config import ICD10_CHAPTERS


def icd10_chapter(code: pl.Expr) -> pl.Expr:
    """The ICD10_CHAPTERS name of a three-character code, null outside every chapter."""
    return pl.coalesce(
        pl.when(code.is_between(pl.lit(first), pl.lit(last))).then(pl.lit(name))
        for first, last, name in ICD10_CHAPTERS
    )
