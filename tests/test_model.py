"""Tests for :mod:`bank_marketing.model`."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from bank_marketing.data import feature_columns
from bank_marketing.model import build_default_pipeline, build_pipeline


def test_build_pipeline_structure() -> None:
    numeric, categorical = feature_columns()
    pipeline = build_pipeline(numeric, categorical, random_state=0)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["pre", "clf"]


def test_build_default_pipeline_fits_and_predicts(
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    x_train, x_test, y_train, _ = split_data
    pipeline = build_default_pipeline(random_state=42)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    assert len(predictions) == len(x_test)
    assert set(predictions).issubset({0, 1})
    proba = pipeline.predict_proba(x_test)
    assert proba.shape == (len(x_test), 2)
