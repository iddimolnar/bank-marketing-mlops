"""Classification metrics used for evaluation and the promotion gate."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Metric keys produced by :func:`evaluate`.
METRIC_NAMES: tuple[str, ...] = ("roc_auc", "f1", "accuracy", "precision", "recall")


class SupportsPredict(Protocol):
    """Structural type for a fitted binary classifier (e.g. a sklearn Pipeline)."""

    def predict(self, x: pd.DataFrame) -> np.ndarray: ...  # pragma: no cover - typing only

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...  # pragma: no cover - typing only


def evaluate(estimator: SupportsPredict, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Evaluate a fitted classifier and return a metric-name -> value mapping."""
    probabilities = estimator.predict_proba(features)[:, 1]
    predictions = estimator.predict(features)

    # sklearn is largely untyped; pyright infers ``zero_division`` as ``str`` from its
    # default. Cast to a permissive callable at this boundary to keep the module typed.
    f1 = cast("Callable[..., float]", f1_score)
    precision = cast("Callable[..., float]", precision_score)
    recall = cast("Callable[..., float]", recall_score)

    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "f1": float(f1(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision(target, predictions, zero_division=0)),
        "recall": float(recall(target, predictions, zero_division=0)),
    }
