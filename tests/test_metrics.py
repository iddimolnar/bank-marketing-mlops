"""Tests for :mod:`bank_marketing.metrics`."""

from __future__ import annotations

import pandas as pd

from bank_marketing.metrics import METRIC_NAMES, evaluate
from bank_marketing.model import build_default_pipeline


def test_evaluate_returns_all_metrics_in_unit_range(
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    x_train, x_test, y_train, y_test = split_data
    pipeline = build_default_pipeline(random_state=42)
    pipeline.fit(x_train, y_train)

    metrics = evaluate(pipeline, x_test, y_test)

    assert set(metrics) == set(METRIC_NAMES)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name}={value} out of range"
