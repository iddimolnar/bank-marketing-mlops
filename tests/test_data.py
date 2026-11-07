"""Tests for :mod:`bank_marketing.data`."""

from __future__ import annotations

import pandas as pd
import pytest

from bank_marketing.data import (
    CATEGORICAL_COLUMNS,
    DURATION_COLUMN,
    NUMERIC_BASE,
    feature_columns,
    prepare_xy,
    split,
)


def test_load_data_shape(raw_df: pd.DataFrame) -> None:
    assert raw_df.shape == (90, 17)
    assert "y" in raw_df.columns
    assert DURATION_COLUMN in raw_df.columns


def test_feature_columns_drops_duration_by_default() -> None:
    numeric, categorical = feature_columns()
    assert DURATION_COLUMN not in numeric
    assert numeric == list(NUMERIC_BASE)
    assert categorical == list(CATEGORICAL_COLUMNS)


def test_feature_columns_keeps_duration_when_requested() -> None:
    numeric, _ = feature_columns(drop_duration=False)
    assert DURATION_COLUMN in numeric


def test_prepare_xy_drops_duration_and_target(raw_df: pd.DataFrame) -> None:
    features, target = prepare_xy(raw_df)
    assert DURATION_COLUMN not in features.columns
    assert "y" not in features.columns
    assert set(target.unique()) <= {0, 1}
    assert target.name == "y"


def test_prepare_xy_keeps_duration_when_requested(raw_df: pd.DataFrame) -> None:
    features, _ = prepare_xy(raw_df, drop_duration=False)
    assert DURATION_COLUMN in features.columns


def test_prepare_xy_rejects_unknown_target_values(raw_df: pd.DataFrame) -> None:
    corrupted = raw_df.copy()
    corrupted.loc[0, "y"] = "maybe"
    with pytest.raises(ValueError, match="outside"):
        prepare_xy(corrupted)


def test_split_is_stratified(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = xy
    x_train, x_test, y_train, y_test = split(features, target, test_size=0.3, random_state=42)
    assert len(x_train) + len(x_test) == len(features)
    assert len(y_train) + len(y_test) == len(target)
    # Both classes present in both splits (stratification preserves balance).
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
