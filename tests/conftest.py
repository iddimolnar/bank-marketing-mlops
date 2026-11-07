"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bank_marketing.data import load_data, prepare_xy, split

FIXTURE_PATH = Path("data/fixtures/tiny_bank.csv")


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
