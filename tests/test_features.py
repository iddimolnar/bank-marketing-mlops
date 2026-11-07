"""Tests for :mod:`bank_marketing.features`."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer

from bank_marketing.data import feature_columns
from bank_marketing.features import build_preprocessor


def test_build_preprocessor_returns_column_transformer() -> None:
    numeric, categorical = feature_columns()
    pre = build_preprocessor(numeric, categorical)
    assert isinstance(pre, ColumnTransformer)


def test_preprocessor_expands_categoricals(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, _ = xy
    numeric, categorical = feature_columns()
    pre = build_preprocessor(numeric, categorical)
    transformed = pre.fit_transform(features)
    # One-hot expansion makes the output wider than the raw column count.
    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(numeric) + len(categorical)


def test_preprocessor_ignores_unknown_categories(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, _ = xy
    numeric, categorical = feature_columns()
    pre = build_preprocessor(numeric, categorical)
    pre.fit(features)
    unseen = features.iloc[[0]].copy()
    unseen.loc[unseen.index[0], "job"] = "astronaut"  # never seen in training
    transformed = pre.transform(unseen)
    assert transformed.shape[0] == 1  # does not raise
