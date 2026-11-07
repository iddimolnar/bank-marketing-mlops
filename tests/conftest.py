"""Shared pytest fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from bank_marketing.data import load_data, prepare_xy, split
from bank_marketing.model import build_default_pipeline

FIXTURE_PATH = Path("data/fixtures/tiny_bank.csv")

VALID_RECORD: dict[str, Any] = {
    "age": 30,
    "job": "unemployed",
    "marital": "married",
    "education": "primary",
    "default": "no",
    "balance": 1787,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 19,
    "month": "oct",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


@dataclass
class InMemoryLogStore:
    """Fake :class:`RequestLogStore` that keeps records in memory for tests."""

    records: list[dict[str, object]] = field(default_factory=list)

    def append(self, record: object) -> None:
        self.records.append(dict(record))  # type: ignore[arg-type]  # test fake accepts any mapping

    def read_all(self) -> list[dict[str, object]]:
        return list(self.records)


@pytest.fixture
def fixture_path() -> Path:
    """Path to the small, class-balanced CSV fixture committed to the repo."""
    return FIXTURE_PATH


@pytest.fixture
def raw_df(fixture_path: Path) -> pd.DataFrame:
    """The raw fixture DataFrame."""
    return load_data(fixture_path)


@pytest.fixture
def xy(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and binary target derived from the fixture."""
    return prepare_xy(raw_df)


@pytest.fixture
def split_data(
    xy: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A deterministic stratified train/test split of the fixture."""
    features, target = xy
    return split(features, target, test_size=0.3, random_state=42)


@pytest.fixture
def fitted_pipeline(xy: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    """A pipeline fitted on the whole fixture (fast, deterministic)."""
    features, target = xy
    pipeline = build_default_pipeline(random_state=42)
    pipeline.fit(features, target)
    return pipeline


@pytest.fixture
def valid_record() -> dict[str, Any]:
    """A valid raw client record for the serving boundary."""
    return dict(VALID_RECORD)


@pytest.fixture
def in_memory_store() -> InMemoryLogStore:
    """A fresh in-memory request log store."""
    return InMemoryLogStore()
